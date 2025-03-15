
@testset "VectorAddLambda" begin
    function f(i, a)
        @inbounds a[i] += 5.0
    end

    alpha = 2.5

    N = 10
    dims = (N)
    a = round.(rand(Float32, dims) * 100)
    a_expected = a .+ 5.0

    a_device = JACC.array(a)
    JACC.parallel_for(N, f, a_device)

    @test Base.Array(a_device)≈a_expected rtol=1e-5
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

    alpha = 2.5

    N = 10
    # Generate random vectors x and y of length N for the interval [0, 100]
    x = round.(rand(Float32, N) * 100)
    y = round.(rand(Float32, N) * 100)
    alpha = 2.5

    x_device = JACC.array(x)
    y_device = JACC.array(y)
    JACC.parallel_for(N, axpy, alpha, x_device, y_device)

    x_expected = x
    seq_axpy(N, alpha, x_expected, y)

    @test Base.Array(x_device)≈x_expected rtol=1e-1
end

@testset "zeros" begin
    N = 10
    x = JACC.zeros(N)
    @test eltype(x) == FloatType
    @test zeros(N)≈Base.Array(x) rtol=1e-5

    function add_one(i, x)
        @inbounds x[i] += 1
    end

    JACC.parallel_for(N, add_one, x)
    @test ones(N)≈Base.Array(x) rtol=1e-5
end

@testset "ones" begin
    N = 10
    x = JACC.ones(N)
    @test eltype(x) == FloatType
    @test ones(N)≈Base.Array(x) rtol=1e-5

    function minus_one(i, x)
        @inbounds x[i] -= 1
    end

    JACC.parallel_for(N, minus_one, x)
    @test zeros(N)≈Base.Array(x) rtol=1e-5
end

@testset "AtomicCounter" begin
    function axpy_counter!(i, alpha, x, y, counter)
        @inbounds x[i] += alpha * y[i]
        JACC.@atomic counter[1] += 1
    end

    N = Int32(10)
    # Generate random vectors x and y of length N for the interval [0, 100]
    alpha = 2.5

    x = JACC.array(round.(rand(Float32, N) * 100))
    y = JACC.array(round.(rand(Float32, N) * 100))
    counter = JACC.array(Int32[0])
    JACC.parallel_for(N, axpy_counter!, alpha, x, y, counter)

    @test Base.Array(counter)[1] == N
end

@testset "reduce" begin
    a = JACC.array([1 for i in 1:10])
    @test JACC.parallel_reduce(a) == 10
    @test JACC.parallel_reduce(min, a) == 1
    a2 = JACC.ones(Int, (2, 2))
    @test JACC.parallel_reduce(min, a2) == 1

    SIZE = 1000
    ah = randn(FloatType, SIZE)
    ad = JACC.array(ah)
    mxd = JACC.parallel_reduce(SIZE, max, (i, a) -> a[i], ad; init = -Inf)
    @test mxd == maximum(ah)
    mxd = JACC.parallel_reduce(max, ad)
    @test mxd == maximum(ah)
    mnd = JACC.parallel_reduce(SIZE, min, (i, a) -> a[i], ad; init = Inf)
    @test mnd == minimum(ah)
    mnd = JACC.parallel_reduce(min, ad)
    @test mnd == minimum(ah)

    ah2 = randn(FloatType, (SIZE, SIZE))
    ad2 = JACC.array(ah2)
    mxd = JACC.parallel_reduce(
        (SIZE, SIZE), max, (i, j, a) -> a[i, j], ad2; init = -Inf)
    @test mxd == maximum(ah2)
    mxd = JACC.parallel_reduce(max, ad2)
    @test mxd == maximum(ah2)
    mnd = JACC.parallel_reduce(
        (SIZE, SIZE), min, (i, j, a) -> a[i, j], ad2; init = Inf)
    @test mnd == minimum(ah2)
    mnd = JACC.parallel_reduce(min, ad2)
    @test mnd == minimum(ah2)
end

@testset "LaunchSpec" begin
    # 1D
    N = 100
    dims = (N)
    a = round.(rand(Float32, dims) * 100)
    a_expected = a .+ 5.0
    a_device = JACC.array(a)
    JACC.parallel_for(JACC.launch_spec(; threads = 1000), N,
        (i, a) -> begin
            @inbounds a[i] += 5.0
        end, a_device)
    @test Base.Array(a_device)≈a_expected rtol=1e-5

    # 2D
    A = JACC.ones(Float32, N, N)
    B = JACC.ones(Float32, N, N)
    C = JACC.zeros(Float32, N, N)
    JACC.parallel_for(JACC.launch_spec(; threads = (16, 16)),
        (N, N), (i, j, A, B, C) -> begin
            @inbounds C[i, j] = A[i, j] + B[i, j]
        end,
        A, B, C)
    C_expected = Float32(2.0) .* ones(Float32, N, N)
    @test Base.Array(C)≈C_expected rtol=1e-5

    # 3D
    A = JACC.ones(Float32, N, N, N)
    B = JACC.ones(Float32, N, N, N)
    C = JACC.zeros(Float32, N, N, N)
    JACC.parallel_for(JACC.launch_spec(; threads = (4, 4, 4)),
        (N, N, N), (i, j, k, A, B, C) -> begin
            @inbounds C[i, j, k] = A[i, j, k] + B[i, j, k]
        end,
        A, B, C)
    C_expected = Float32(2.0) .* ones(Float32, N, N, N)
    @test Base.Array(C)≈C_expected rtol=1e-5

    # reduce
    a = JACC.ones(N)
    res = JACC.parallel_reduce(JACC.launch_spec(), a)
    @test Base.Array(res)[] == N
    res = JACC.parallel_reduce(JACC.launch_spec(), N, (i, a) -> a[i], a)
    @test Base.Array(res)[] == N
    res = JACC.parallel_reduce(JACC.launch_spec(), min, a)
    @test Base.Array(res)[] == 1
    res = JACC.parallel_reduce(
        JACC.launch_spec(), N, max, (i, a) -> a[i], a; init = -Inf)
    @test Base.Array(res)[] == 1
    a2 = JACC.ones(N, N)
    res = JACC.parallel_reduce(JACC.launch_spec(), a2)
    @test Base.Array(res)[] == N * N
    res = JACC.parallel_reduce(
        JACC.launch_spec(), (N, N), (i, j, a) -> a[i, j], a2)
    @test Base.Array(res)[] == N * N
    res = JACC.parallel_reduce(JACC.launch_spec(), min, a2)
    @test Base.Array(res)[] == 1
    res = JACC.parallel_reduce(
        JACC.launch_spec(), (N, N), max, (i, j, a) -> a[i, j], a2; init = -Inf)
    @test Base.Array(res)[] == 1
end

@testset "shared" begin
    N = 100
    alpha = 2.5
    x = JACC.ones(N)
    x_shared = JACC.ones(N)
    y = JACC.ones(N)

    function scal(i, x, y, alpha)
        @inbounds x[i] = y[i] * alpha
    end

    function scal_shared(i, x, y, alpha)
        y_shared = JACC.shared(y)
        @inbounds x[i] = y_shared[i] * alpha
    end

    JACC.parallel_for(N, scal, x, y, alpha)
    JACC.parallel_for(N, scal_shared, x_shared, y, alpha)
    @test Base.Array(x)≈Base.Array(x_shared) rtol=1e-8
end

@testset "JACC.BLAS" begin
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

    x = ones(1_000)
    y = ones(1_000)
    jx = JACC.ones(1_000)
    jy = JACC.ones(1_000)
    alpha = 2.0

    seq_axpy(1_000, alpha, x, y)
    ref_result = seq_dot(1_000, x, y)

    JACC.BLAS.axpy(1_000, alpha, jx, jy)
    jresult = JACC.BLAS.dot(1_000, jx, jy)

    @test jresult≈ref_result rtol=1e-8

    x = ones(1_000)
    y = ones(1_000)
    y1 = y * 2
    jx = JACC.ones(1_000)
    jy = JACC.ones(1_000)
    jy1 = jy * 2
    alpha = 2.0

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
            @inbounds sum_sq += x[i] * x[i]
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

    ref_result = seq_axpy(1_000, alpha, x, y)
    ref_result = seq_dot(1_000, x, y)
    JACC.BLAS.axpy(1_000, alpha, jx, jy)
    jresult = JACC.BLAS.dot(1_000, jx, jy)
    @test jresult≈ref_result rtol=1e-8

    seq_scal(1_000, alpha, x)
    JACC.BLAS.scal(1_000, alpha, jx)
    @test x≈Base.Array(jx) rtol=1e-8

    seq_axpy(1_000, alpha, x, y)
    JACC.BLAS.axpy(1_000, alpha, jx, jy)
    @test x≈Base.Array(jx) atol=1e-8

    r1 = seq_dot(1_000, x, y)
    r2 = JACC.BLAS.dot(1_000, jx, jy)
    @test r1≈r2 atol=1e-8

    r1 = seq_asum(1_000, x)
    r2 = JACC.BLAS.asum(1_000, jx)
    @test r1≈r2 atol=1e-8
    r1 = seq_nrm2(1_000, x)
    r2 = JACC.BLAS.nrm2(1_000, jx)
    @test r1≈r2 atol=1e-8

    seq_swap(1_000, x, y1)
    JACC.BLAS.swap(1_000, jx, jy1)
    @test x == Base.Array(jx)
    @test y1 == Base.Array(jy1)
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
    @test Base.Array(C)≈C_expected rtol=1e-5
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
    @test Base.Array(C)≈C_expected rtol=1e-5
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
    a0 = JACC.ones(SIZE)
    a1 = JACC.ones(SIZE)
    a2 = JACC.ones(SIZE)
    r = JACC.ones(SIZE)
    p = JACC.ones(SIZE)
    s = JACC.zeros(SIZE)
    x = JACC.zeros(SIZE)
    r_old = JACC.zeros(SIZE)
    r_aux = JACC.zeros(SIZE)
    a1 = a1 * 4
    r = r * 0.5
    p = p * 0.5
    cond = 1.0

    while cond[1, 1] >= 1e-14
        r_old = copy(r)

        JACC.parallel_for(SIZE, matvecmul, a0, a1, a2, p, s, SIZE)

        alpha0 = JACC.parallel_reduce(SIZE, dot, r, r)
        alpha1 = JACC.parallel_reduce(SIZE, dot, p, s)

        alpha = alpha0 / alpha1
        negative_alpha = alpha * -1.0

        JACC.parallel_for(SIZE, axpy, negative_alpha, r, s)
        JACC.parallel_for(SIZE, axpy, alpha, x, p)

        beta0 = JACC.parallel_reduce(SIZE, dot, r, r)
        beta1 = JACC.parallel_reduce(SIZE, dot, r_old, r_old)
        beta = beta0 / beta1

        r_aux = copy(r)

        JACC.parallel_for(SIZE, axpy, beta, r_aux, p)
        ccond = JACC.parallel_reduce(SIZE, dot, r, r)
        cond = ccond

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
                @inbounds f[floor(Int, ind)] = f1[floor(Int, iind)]
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
                @inbounds f2[floor(Int, ind)] = f[floor(Int, ind)] *
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
    cx = zeros(Int, 9)
    cy = zeros(Int, 9)
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

    df = JACC.array(f)
    df1 = JACC.array(f1)
    df2 = JACC.array(f2)
    dcx = JACC.array(cx)
    dcy = JACC.array(cy)
    dw = JACC.array(w)

    JACC.parallel_for(
        (SIZE, SIZE), lbm_kernel, df, df1, df2, t, dw, dcx, dcy, SIZE)

    lbm_threads(f, f1, f2, t, w, cx, cy, SIZE)

    @test f2≈Base.Array(df2) rtol=1e-1
end
