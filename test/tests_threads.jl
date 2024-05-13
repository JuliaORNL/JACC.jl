import JACC
using Test

@testset "TestBackend" begin
    @test JACC.JACCPreferences.backend == "threads"
end

@testset "VectorAddLambda" begin
    function f(x, a)
        @inbounds a[x] += 5.0
    end

    dims = (10)
    a = round.(rand(Float32, dims) * 100)
    a_expected = a .+ 5.0

    JACC.parallel_for(10, f, a)

    @test a≈a_expected rtol=1e-5
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
    JACC.parallel_for(N, axpy, alpha, x_host_JACC, y_host_JACC)

    x_expected = x
    seq_axpy(N, alpha, x_expected, y)

    @test x_host_JACC≈x_expected rtol=1e-1
end

@testset "AtomicCounter" begin
    function axpy_counter!(i, alpha, x, y, counter)
        @inbounds x[i] += alpha * y[i]
        JACC.@atomic counter[1] += 1
    end

    N = Int32(10)
    # Generate random vectors x and y of length N for the interval [0, 100]
    alpha = 2.5
    counter = zeros(Int32, 1)

    x_device = JACC.Array(round.(rand(Float32, N) * 100))
    y_device = JACC.Array(round.(rand(Float32, N) * 100))
    counter = JACC.Array{Int32}([0])
    JACC.parallel_for(N, axpy_counter!, alpha, x_device, y_device, counter)

    @test counter[1] == N
end

@testset "zeros" begin
    N = 10
    x = JACC.zeros(Float32, N)
    @test typeof(x) == Vector{Float32}
    @test eltype(x) == Float32
    @test zeros(N)≈x rtol=1e-5

    function add_one(i, x)
        @inbounds x[i] += 1
    end

    JACC.parallel_for(N, add_one, x)
    @test ones(N)≈x rtol=1e-5
end

@testset "ones" begin
    N = 10
    x = JACC.ones(Float64, N)
    @test typeof(x) == Vector{Float64}
    @test eltype(x) == Float64
    @test ones(N)≈x rtol=1e-5

    function minus_one(i, x)
        @inbounds x[i] -= 1
    end

    JACC.parallel_for(N, minus_one, x)
    @test zeros(N)≈x rtol=1e-5
end

@testset "CG" begin
    function matvecmul(i, a1, a2, a3, x, y, SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
        elseif i == SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i - 1] + a1[i] * +x[i] + a1[i] * +x[i + 1]
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

        JACC.parallel_for(SIZE, matvecmul, a0, a1, a2, p, s, SIZE)

        alpha0 = JACC.parallel_reduce(SIZE, dot, r, r)
        alpha1 = JACC.parallel_reduce(SIZE, dot, p, s)

        alpha = alpha0 / alpha1
        negative_alpha = alpha * (-1.0)

        JACC.parallel_for(SIZE, axpy, negative_alpha, r, s)
        JACC.parallel_for(SIZE, axpy, alpha, x, p)

        beta0 = JACC.parallel_reduce(SIZE, dot, r, r)
        beta1 = JACC.parallel_reduce(SIZE, dot, r_old, r_old)
        beta = beta0 / beta1

        r_aux = copy(r)

        JACC.parallel_for(SIZE, axpy, beta, r_aux, p)
        ccond = JACC.parallel_reduce(SIZE, dot, r, r)
        global cond = ccond
        p = copy(r_aux)
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
                @inbounds feq = w[k] * p *
                                (1.0 + 3.0 * cu + cu * cu -
                                 1.5 * ((u * u) + (v * v)))
                ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                @inbounds f2[trunc(Int, ind)] = f[trunc(Int, ind)] *
                                                (1.0 - 1.0 / t) + feq * 1 / t
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
                        iind = (k - 1) * SIZE * SIZE + x_stream * SIZE +
                               y_stream
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
                        @inbounds feq = w[k] * p *
                                        (1.0 + 3.0 * cu + cu * cu -
                                         1.5 * ((u * u) + (v * v)))
                        ind = (k - 1) * SIZE * SIZE + x * SIZE + y
                        @inbounds f2[trunc(Int, ind)] = f[trunc(Int, ind)] *
                                                        (1.0 - 1.0 / t) +
                                                        feq * 1 / t
                    end
                end
            end
        end
    end

    SIZE = 10
    f = ones(SIZE * SIZE * 9) .* 2.0
    f1 = ones(SIZE * SIZE * 9) .* 3.0
    f2 = ones(SIZE * SIZE * 9) .* 4.0
    cx = zeros(9)
    cy = zeros(9)
    cx[1] = 0
    cy[1] = 0
    cx[2] = 1
    cy[2] = 0
    cx[3] = -1
    cy[3] = 0
    cx[4] = 0
    cy[4] = 1
    cx[5] = 0
    cy[5] = -1
    cx[6] = 1
    cy[6] = 1
    cx[7] = -1
    cy[7] = 1
    cx[8] = -1
    cy[8] = -1
    cx[9] = 1
    cy[9] = -1
    w = ones(9)
    t = 1.0

    df = JACC.Array(f)
    df1 = JACC.Array(f1)
    df2 = JACC.Array(f2)
    dcx = JACC.Array(cx)
    dcy = JACC.Array(cy)
    dw = JACC.Array(w)

    JACC.parallel_for(
        (SIZE, SIZE), lbm_kernel, df, df1, df2, t, dw, dcx, dcy, SIZE)

    lbm_threads(f, f1, f2, t, w, cx, cy, SIZE)

    @test f2≈df2 rtol=1e-1
end

@testset "JACC.BLAS" begin
    elt = Float64

    x = ones(1_000)
    y = ones(1_000)
    jx = JACC.ones(elt, 1_000)
    jy = JACC.ones(elt, 1_000)
    alpha = 2.0

    function seq_axpy(N, alpha, x, y)
        for i in 1:N
            @inbounds x[i] += alpha * y[i]
        end
    end
    
    function seq_dot(N, x, y)
        r = 0.0
        for i in 1:N
            @inbounds r += x[i] * y[i]
        end
        return r
    end
 
    seq_axpy(1_000, alpha, x, y)
    ref_result = seq_dot(1_000, x, y)

    JACC.BLAS.axpy(1_000, alpha, jx, jy)
    jresult = JACC.BLAS.dot(1_000, jx, jy)
    result = jresult[1]     
    
    @test result≈ref_result rtol=1e-8

end
