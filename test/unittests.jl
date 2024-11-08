import JACC
using Test

@testset "VectorAddLambda" begin
    function f(i, a)
        @inbounds a[i] += 5.0
    end

    N = 10
    dims = (N)
    a = round.(rand(Float32, dims) * 100)

    a_device = JACC.Array(a)
    JACC.parallel_for(N, f, a_device)

    a_expected = a .+ 5.0
    @test Array(a_device)≈a_expected rtol=1e-5
end

@testset "AXPY" begin
    function axpy(i, alpha, x, y)
        @inbounds x[i] += alpha * y[i]
    end

    function seq_axpy(N, alpha, x, y)
        @inbounds for i in 1:N
            x[i] += alpha * y[i]
        end
    end

    N = 10
    # Generate random vectors x and y of length N for the interval [0, 100]
    x = round.(rand(Float32, N) * 100)
    y = round.(rand(Float32, N) * 100)
    alpha = 2.5

    x_device = JACC.Array(x)
    y_device = JACC.Array(y)
    JACC.parallel_for(N, axpy, alpha, x_device, y_device)

    x_expected = x
    seq_axpy(N, alpha, x_expected, y)

    @test Array(x_device)≈x_expected rtol=1e-1
end

@testset "zeros" begin
    N = 10
    x = JACC.zeros(Float64, N)
    @test eltype(x) == Float64
    @test zeros(N)≈Array(x) rtol=1e-5

    function add_one(i, x)
        @inbounds x[i] += 1
    end

    JACC.parallel_for(N, add_one, x)
    @test ones(N) ≈ Core.Array(x) rtol=1e-5
end

@testset "ones" begin
    N = 10
    x = JACC.ones(Float64, N)
    @test eltype(x) == Float64
    @test ones(N)≈Array(x) rtol=1e-5

    function minus_one(i, x)
        @inbounds x[i] -= 1
    end

    JACC.parallel_for(N, minus_one, x)
    @test zeros(N) ≈ Array(x) rtol=1e-5
end

@testset "AtomicCounter" begin
    function axpy_counter!(i, alpha, x, y, counter)
        @inbounds x[i] += alpha * y[i]
        JACC.@atomic counter[1] += 1
    end

    N = Int32(10)
    # Generate random vectors x and y of length N for the interval [0, 100]
    alpha = 2.5

    x = JACC.Array(round.(rand(Float32, N) * 100))
    y = JACC.Array(round.(rand(Float32, N) * 100))
    counter = JACC.Array{Int32}([0])
    JACC.parallel_for(N, axpy_counter!, alpha, x, y, counter)

    @test Array(counter)[1] == N
end

@testset "reduce" begin
    SIZE = 1000
    ah = randn(SIZE)
    ad = JACC.Array(ah)
    mxd = JACC.parallel_reduce(SIZE, max, (i,a)->a[i], ad; init = -Inf)
    @test mxd == maximum(ah)
end

@testset "shared" begin
    N = 100
    alpha = 2.5
    x = JACC.ones(Float64, N)
    x_shared = JACC.ones(Float64, N)
    y = JACC.ones(Float64, N)

    function scal(i, x, y, alpha)
        @inbounds x[i] = y[i] * alpha
    end
    
    function scal_shared(i, x, y, alpha)
        y_shared = JACC.shared(y) 
        @inbounds x[i] = y_shared[i] * alpha
    end

    JACC.parallel_for(N, scal, x, y, alpha)
    JACC.parallel_for(N, scal_shared, x_shared, y, alpha)
    @test x≈x_shared rtol=1e-8
end

@testset "JACC.BLAS" begin
    x = ones(1_000)
    y = ones(1_000)
    jx = JACC.ones(1_000)
    jy = JACC.ones(1_000)
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

    @test jresult≈ref_result rtol=1e-8
end

@testset "Add-2D" begin
    function add!(i, j, A, B, C)
        @inbounds C[i, j] = A[i, j] + B[i, j]
    end

    M = 10
    N = 10
    A = JACC.ones(Float32, M, N)
    B = JACC.ones(Float32, M, N)
    C = JACC.zeros(Float32, M, N)

    JACC.parallel_for((M, N), add!, A, B, C)

    C_expected = Float32(2.0) .* ones(Float32, M, N)
    @test Array(C)≈C_expected rtol=1e-5
end

@testset "Add-3D" begin
    function add!(i, j, k, A, B, C)
        @inbounds C[i, j, k] = A[i, j, k] + B[i, j, k]
    end

    L = 10
    M = 10
    N = 10
    A = JACC.ones(Float32, L, M, N)
    B = JACC.ones(Float32, L, M, N)
    C = JACC.zeros(Float32, L, M, N)

    JACC.parallel_for((L, M, N), add!, A, B, C)

    C_expected = Float32(2.0) .* ones(Float32, L, M, N)
    @test Array(C)≈C_expected rtol=1e-5
 
    function seq_scal(N, alpha, x)
        for i in 1:N
            @inbounds x[i] = alpha * x[i]
        end
    end
  
    function seq_asum(N, x)
        r = 0.0
        for i in 1:N
            @inbounds r += abs(x[i])
        end
        return r
    end

    function seq_nrm2(N, x)
        sum_sq = 0.0
        for i in 1:N
            @inbounds sum_sq += x[i]*x[i]
        end
        r = sqrt(sum_sq)
        return r
    end

    function seq_swap(N, x, y1)
        for i in 1:N
            @inbounds t = x[i]
            @inbounds x[i] = y1[i]
            @inbounds y1[i] = t
        end       
    end

    # Comparing JACC.BLAS with regular seuential functions
    # seq_axpy(1_000, alpha, x, y)
    # ref_result = seq_dot(1_000, x, y)  
    # JACC.BLAS.axpy(1_000, alpha, jx, jy)
    # jresult = JACC.BLAS.dot(1_000, jx, jy)
    # result = Array(jresult)        
    # @test result[1]≈ref_result rtol=1e-8


    # seq_scal(1_000, alpha, x)
    # JACC.BLAS.scal(1_000, alpha, jx)
    # @test x≈Array(jx) atol=1e-8 

    # seq_axpy(1_000, alpha, x, y)
    # JACC.BLAS.axpy(1_000, alpha, jx, jy)
    # @test x≈Array(jx) atol=1e-8

    # r1 = seq_dot(1_000, x, y) 
    # r2 = JACC.BLAS.dot(1_000, jx, jy)
    # @test r1≈Array(r2)[1] atol=1e-8 
    
    # r1 = seq_asum(1_000, x)
    # r2 = JACC.BLAS.asum(1_000, jx)
    # r1 = seq_nrm2(1_000, x)
    # r2 = JACC.BLAS.nrm2(1_000, jx)
    # @test r1≈Array(r2)[1] atol=1e-8

    #seq_swap(1_000, x, y1)  
    #JACC.BLAS.swap(1_000, jx, jy1)
    #@test x == Array(jx)
    #@test y1 == Array(jy1)
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
	a0 = JACC.ones(Float64, SIZE)
	a1 = JACC.ones(Float64, SIZE)
	a2 = JACC.ones(Float64, SIZE)
	r = JACC.ones(Float64, SIZE)
	p = JACC.ones(Float64, SIZE)
	s = JACC.zeros(Float64, SIZE)
	x = JACC.zeros(Float64, SIZE)
	r_old = JACC.zeros(Float64, SIZE)
	r_aux = JACC.zeros(Float64, SIZE)
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

    @test f2≈Array(df2) rtol=1e-1
end
