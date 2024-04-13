using JACC
using Test


# @testset "TestBackend" begin
#     @test JACC.JACCPreferences.backend == "threads"
# end

@testset "VectorAddLambda" begin
    function f(x, a)
        @inbounds a[x] += 5.0
    end

    dims = (10)
    a = round.(rand(Float32, dims) * 100)
    a_expected = a .+ 5.0

    ## Using multiple dispatch the JACCArray version is first called then it unwraps the array and passes it along to the correct backend version.
    JACC.parallel_for(10, f, JACC.JACCArgsList{typeof(a)}(a))
    @test a ≈ a_expected rtol = 1e-5
end

@testset "AXPY" begin
    function seq_axpy(N, alpha, x, y)
        Threads.@threads for i in 1:N
            @inbounds x[i] += alpha * y[i]
        end
    end

    function axpy(i, alpha, x, y)
        if i <= length(x)
            @inbounds x[i] += alpha * y[i]
        end
    end

    N = 10
    # Generate random vectors x and y of length N for the interval [0, 100]
    x = round.(rand(Float32, N) * 100)
    y = round.(rand(Float32, N) * 100)
    alpha = 2.5

    x_host_JACC = JACC.Array(x)
    y_host_JACC = JACC.Array(y)
    JACC.parallel_for(N, axpy, JACC.JACCArgsList{typeof(x_host_JACC)}(alpha, x_host_JACC, y_host_JACC))

    x_expected = x
    seq_axpy(N, alpha, x_expected, y)

    @test x_host_JACC ≈ x_expected rtol = 1e-1
end

@testset "CG" begin

    function matvecmul(i, a1, a2, a3, x, y, SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i+1]
        elseif i == SIZE
            y[i] = a3[i] * x[i-1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i-1] + a1[i] * +x[i] + a1[i] * +x[i+1]
        end
    end

    function dot(i, x, y)
        @inbounds return x[i] * y[i]
    end

    function axpy(i, alpha, x, y)
        @inbounds x[i] += alpha[1, 1] * y[i]
    end

    SIZE = 10
    a0 = ones(SIZE)
    a1 = ones(SIZE)
    a2 = ones(SIZE)
    r = ones(SIZE)
    p = ones(SIZE)
    s = zeros(SIZE)
    x = zeros(SIZE)
    r_old = zeros(SIZE)
    r_aux = zeros(SIZE)
    a1 = a1 * 4
    r = r * 0.5
    p = p * 0.5
    global cond = one(Float64)

    while cond[1, 1] >= 1e-14

        r_old = copy(r)

        JACC.parallel_for(SIZE, matvecmul, JACC.JACCArgsList{typeof(a0)}(a0, a1, a2, p, s, SIZE))

        alpha0 = JACC.parallel_reduce(SIZE, dot, JACC.JACCArgsList{typeof(r)}(r, r))
        alpha1 = JACC.parallel_reduce(SIZE, dot, JACC.JACCArgsList{typeof(p)}(p, s))

        alpha = alpha0 / alpha1
        negative_alpha = alpha * (-1.0)

        JACC.parallel_for(SIZE, axpy, JACC.JACCArgsList{typeof(r)}(negative_alpha, r, s))
        JACC.parallel_for(SIZE, axpy, JACC.JACCArgsList{typeof(x)}(alpha, x, p))

        beta0 = JACC.parallel_reduce(SIZE, dot, JACC.JACCArgsList{typeof(r)}(r, r))
        beta1 = JACC.parallel_reduce(SIZE, dot, JACC.JACCArgsList{typeof(r)}(r_old, r_old))
        beta = beta0 / beta1

        r_aux = copy(r)

        JACC.parallel_for(SIZE, axpy, JACC.JACCArgsList{typeof(r_aux)}(beta, r_aux, p))
        ccond = JACC.parallel_reduce(SIZE, dot, JACC.JACCArgsList{typeof(r)}(r, r))
        global cond = ccond
        p = copy(r_aux)

        println(cond)

    end
    @test cond[1, 1] <= 1e-14
end

@testset "LBM" begin

    function lbm_kernel(x, y, f, f1, f2, t, w, cx, cy, SIZE)
        u = 0.0
        v = 0.0
        p = 0.0
        x_stream = 0
        y_stream = 0

        if x > 1 && x < SIZE && y > 1 && y < SIZE
            for k in 1:9
                @inbounds x_stream = x - cx[k]
                @inbounds y_stream = y - cy[k]
                ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                iind = (k - 1) * SIZE * SIZE + x_stream * SIZE + y_stream
                @inbounds f[trunc(Int, ind)] = f1[trunc(Int, iind)]
            end
            for k in 1:9
                ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                @inbounds p = p[1, 1] + f[ind]
                @inbounds u = u[1, 1] + f[ind] * cx[k]
                @inbounds v = v[1, 1] + f[ind] * cy[k]
            end
            u = u / p
            v = v / p
            for k in 1:9
                @inbounds cu = cx[k] * u + cy[k] * v
                @inbounds feq = w[k] * p * (1.0 + 3.0 * cu + cu * cu - 1.5 * ((u * u) + (v * v)))
                ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                @inbounds f2[trunc(Int, ind)] = f[trunc(Int, ind)] * (1.0 - 1.0 / t) + feq * 1 / t
            end
        end
    end

    function lbm_threads(f, f1, f2, t, w, cx, cy, SIZE)

        Threads.@sync Threads.@threads for x in 1:SIZE
            for y in 1:SIZE
                u = 0.0
                v = 0.0
                p = 0.0
                x_stream = 0
                y_stream = 0

                if x > 1 && x < SIZE && y > 1 && y < SIZE
                    for k in 1:9
                        @inbounds x_stream = x - cx[k]
                        @inbounds y_stream = y - cy[k]
                        ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                        iind = (k - 1) * SIZE * SIZE + x_stream * SIZE + y_stream
                        @inbounds f[trunc(Int, ind)] = f1[trunc(Int, iind)]
                    end
                    for k in 1:9
                        ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                        @inbounds p = p[1, 1] + f[ind]
                        @inbounds u = u[1, 1] + f[ind] * cx[k]
                        @inbounds v = v[1, 1] + f[ind] * cy[k]
                    end
                    u = u / p
                    v = v / p
                    for k in 1:9
                        @inbounds cu = cx[k] * u + cy[k] * v
                        @inbounds feq = w[k] * p * (1.0 + 3.0 * cu + cu * cu - 1.5 * ((u * u) + (v * v)))
                        ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                        @inbounds f2[trunc(Int, ind)] = f[trunc(Int, ind)] * (1.0 - 1.0 / t) + feq * 1 / t
                    end
                end
            end
        end
    end

    SIZE = 10
    f = ones(SIZE * SIZE * 9) .* 2.0
    f1 = ones(SIZE * SIZE * 9) .* 3.0
    f2 = ones(SIZE * SIZE * 9) .* 4.0
    cx = Vector{Float64}([0,1,-1,0,0,1,-1,-1,1])
    cy = Vector{Float64}([0,0,0,1,-1,1,1,-1,-1])
    w = ones(9)
    t = 1.0

    df = JACC.Array(f)
    df1 = JACC.Array(f1)
    df2 = JACC.Array(f2)
    dcx = JACC.Array(cx)
    dcy = JACC.Array(cy)
    dw = JACC.Array(w)

    JACC.parallel_for((SIZE, SIZE), lbm_kernel, JACC.JACCArgsList{typeof(df)}(df, df1, df2, t, dw, dcx, dcy, SIZE))

    lbm_threads(f, f1, f2, t, w, cx, cy, SIZE)

    @test f2 ≈ df2 rtol = 1e-1

end
