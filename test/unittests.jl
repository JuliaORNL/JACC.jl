
function seq_axpy(N, alpha, x, y)
    for i in 1:N
        @inbounds x[i] += alpha * y[i]
    end
end

function seq_axpy(M, N, alpha, x, y)
    for j in 1:N
        for i in 1:M
            @inbounds x[i, j] += alpha * y[i, j]
        end
    end
end

function seq_dot(N, x, y)
    r = 0.0
    for i in 1:N
        @inbounds r += x[i] * y[i]
    end
    return r
end

function seq_dot(M, N, x, y)
    r = 0.0
    for j in 1:N
        for i in 1:M
            @inbounds r += x[i, j] * y[i, j]
        end
    end
    return r
end

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

@testset "fill" begin
    N = 10
    x = JACC.fill(10.0, N)
    @test fill(10.0, N)≈Base.Array(x) rtol=1e-5
    fill!(x, 22.2)
    @test fill(22.2, N)≈Base.Array(x) rtol=1e-5
end

# using Cthulhu
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

    # TODO: clean this up
    # counter = JACC.zeros((1,1,1))
    # try
    #     JACC.parallel_for(N,
    #         (i, counter) -> begin
    #             JACC.@atomic counter[1,1,1] += 1.0
    #         end,
    #         counter)
    # catch err
    #     code_warntype(err; interactive = true)
    # end
    # @test Base.Array(counter)[1,1,1] == N
end

@testset "reduce" begin
    a = JACC.array([1 for i in 1:10])
    @test JACC.parallel_reduce(a) == 10
    @test JACC.parallel_reduce(min, a) == 1
    reducer = JACC.reducer(; dims = JACC.array_size(a), op = +)
    reducer.spec = JACC.launch_spec(; sync = true)
    reducer(a)
    @test JACC.get_result(reducer) == 10
    a2 = JACC.ones(Int, (2, 2))
    @test JACC.parallel_reduce(min, a2) == 1
    reducer = JACC.reducer(; dims = JACC.array_size(a2), op = min)
    reducer.spec = JACC.launch_spec(; sync = true)
    reducer(a2)
    @test JACC.get_result(reducer) == 1
    reducer(a2) do i, j, a
        a[i, j]
    end
    @test JACC.get_result(reducer) == 1

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

    function seq_dot(M, N, x, y)
        r = 0.0
        for i in 1:M
            for j in 1:N
                @inbounds r += x[i, j] * y[i, j]
            end
        end
        return r
    end
    function dot_2d(i, j, x, y)
        return x[i, j] * y[i, j]
    end
    SIZE = 10
    x = round.(rand(Float64, SIZE, SIZE) * 100)
    y = round.(rand(Float64, SIZE, SIZE) * 100)
    alpha = 2.5
    dx = JACC.array(x)
    dy = JACC.array(y)
    # JACC.Multi.parallel_for((SIZE, SIZE), axpy_2d, alpha, dx, dy)
    # x_expected = x
    # seq_axpy(SIZE, SIZE, alpha, x_expected, y)
    # @test convert(Base.Array, dx)≈x_expected rtol=1e-1
    res = JACC.parallel_reduce((SIZE, SIZE), dot_2d, dx, dy)
    # @test res≈seq_dot(SIZE, SIZE, x_expected, y) rtol=1e-1
    @test res≈seq_dot(SIZE, SIZE, x, y) rtol=1e-1
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
    JACC.synchronize()
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
    JACC.synchronize()
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
    JACC.synchronize()
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

    function test_sync()
        ix = JACC.zeros(Int, N)
        JACC.parallel_for(JACC.launch_spec(threads = N), N, ix) do i, x
            shared_mem = JACC.shared(x)
            shared_mem[i] = i
            JACC.sync_workgroup()
            if i > 50
                shared_mem[i] = shared_mem[i - 50]
            end
            x[i] = shared_mem[i]
        end
        ix_h = Base.Array(ix)
        for i = 1:50
            @test ix_h[i] == i
            @test ix_h[i+50] == i
        end
    end
    test_sync()
    test_sync()
end

@testset "JACC.BLAS" begin
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

@testset "do" begin
    L = 10
    M = 10
    N = 10

    # 1D
    a = round.(rand(Float32, N) * 100)
    a_expected = a .+ 5.0

    a_device = JACC.array(a)
    JACC.parallel_for(N, a_device) do i, a
        @inbounds a[i] += 5.0
    end
    @test Base.Array(a_device)≈a_expected rtol=1e-5

    a_device = JACC.array(a)
    res = JACC.parallel_reduce(N, a_device) do i, a
        a[i] * a[i]
    end
    @test res≈seq_dot(N, a, a) rtol=1e-1

    res = JACC.parallel_reduce(N, min, a_device; init = Inf) do i, a
        a[i]
    end
    @test res≈minimum(a)

    # 2D
    A2 = JACC.ones(Float32, M, N)
    B2 = JACC.ones(Float32, M, N)
    C2 = JACC.zeros(Float32, M, N)
    JACC.parallel_for((M, N), A2, B2, C2) do i, j, A, B, C
        @inbounds C[i, j] = A[i, j] + B[i, j]
    end
    C2_expected = Float32(2.0) .* ones(Float32, M, N)
    @test Base.Array(C2)≈C2_expected rtol=1e-5

    res = JACC.parallel_reduce((M, N), A2, B2) do i, j, a, b
        a[i, j] * b[i, j]
    end
    @test res≈seq_dot(M, N, Base.Array(A2), Base.Array(B2)) rtol=1e-1

    res = JACC.parallel_reduce((M, N), min, A2; init = Inf) do i, j, a
        a[i]
    end
    @test res≈1

    # 3D
    A3 = JACC.ones(Float32, L, M, N)
    B3 = JACC.ones(Float32, L, M, N)
    C3 = JACC.zeros(Float32, L, M, N)

    JACC.parallel_for((L, M, N), A3, B3, C3) do i, j, k, A, B, C
        @inbounds C[i, j, k] = A[i, j, k] + B[i, j, k]
    end

    C3_expected = Float32(2.0) .* ones(Float32, L, M, N)
    @test Base.Array(C3)≈C3_expected rtol=1e-5

    # 1D
    N = 100
    a = round.(rand(Float32, N) * 100)
    a_expected = a .+ 5.0
    a_device = JACC.array(a)
    JACC.parallel_for(JACC.launch_spec(; threads = 1000), N, a_device) do i, a
        @inbounds a[i] += 5.0
    end
    @test Base.Array(a_device)≈a_expected rtol=1e-5

    # 2D
    A = JACC.ones(Float32, N, N)
    B = JACC.ones(Float32, N, N)
    C = JACC.zeros(Float32, N, N)
    JACC.parallel_for(JACC.launch_spec(; threads = (16, 16)),
        (N, N), A, B, C) do i, j, A, B, C
        @inbounds C[i, j] = A[i, j] + B[i, j]
    end
    C_expected = Float32(2.0) .* ones(Float32, N, N)
    @test Base.Array(C)≈C_expected rtol=1e-5

    # 3D
    A = JACC.ones(Float32, N, N, N)
    B = JACC.ones(Float32, N, N, N)
    C = JACC.zeros(Float32, N, N, N)
    JACC.parallel_for(JACC.launch_spec(; threads = (4, 4, 4)),
        (N, N, N), A, B, C) do i, j, k, A, B, C
        @inbounds C[i, j, k] = A[i, j, k] + B[i, j, k]
    end
    C_expected = Float32(2.0) .* ones(Float32, N, N, N)
    @test Base.Array(C)≈C_expected rtol=1e-5
end

@testset "CG" begin
    function matvecmul(i, a1, a2, a3, x, y, SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
        elseif i == SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
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

if JACC.backend != "oneapi"
@testset "Multi" begin
    # Unidimensional arrays
    function axpy(i, alpha, x, y)
        x[i] += alpha * y[i]
    end
    function dot(i, x, y)
        return x[i] * y[i]
    end
    SIZE = 10
    x = round.(rand(Float64, SIZE) * 100)
    y = round.(rand(Float64, SIZE) * 100)
    alpha = 2.5
    dx = JACC.Multi.array(x)
    dy = JACC.Multi.array(y)
    JACC.Multi.parallel_for(SIZE, alpha, dx, dy) do i, alpha, x, y
        x[i] += alpha * y[i]
    end
    x_expected = x
    seq_axpy(SIZE, alpha, x_expected, y)
    @test convert(Base.Array, dx)≈x_expected rtol=1e-1
    res = JACC.Multi.parallel_reduce(SIZE, dot, dx, dy)
    @test res≈seq_dot(SIZE, x_expected, y) rtol=1e-1

    # Multidimensional arrays
    function dot_2d(i, j, x, y)
        return x[i, j] * y[i, j]
    end
    SIZE = 10
    x = round.(rand(Float64, SIZE, SIZE) * 100)
    y = round.(rand(Float64, SIZE, SIZE) * 100)
    alpha = 2.5
    dx = JACC.Multi.array(x)
    dy = JACC.Multi.array(y)
    JACC.Multi.parallel_for((SIZE, SIZE), alpha, dx, dy) do i, j, alpha, x, y
        x[i, j] += alpha * y[i, j]
    end
    x_expected = x
    seq_axpy(SIZE, SIZE, alpha, x_expected, y)
    @test convert(Base.Array, dx)≈x_expected rtol=1e-1
    res = JACC.Multi.parallel_reduce((SIZE, SIZE), dot_2d, dx, dy)
    @test res≈seq_dot(SIZE, SIZE, x_expected, y) rtol=1e-1

    # HPCG example
    function matvecmul(i, a1, a2, a3, x, y, SIZE, ndev)
        ind = JACC.Multi.ghost_shift(i, a1)
        dev_id = JACC.Multi.device_id(a1)
        if dev_id == 1 && i == 1
            y[ind] = a2[ind] * x[ind] + a1[ind] * x[ind + 1]
        elseif dev_id == ndev && i == SIZE
            y[ind] = a3[ind] * x[ind - 1] + a2[ind] * x[ind]
        else
            y[ind] = a3[ind] * x[ind - 1] + a2[ind] * x[ind] +
                     a1[ind] * x[ind + 1]
        end
    end

    SIZE = 10
    # Initialization of inputs
    a1 = ones(SIZE)
    a2 = ones(SIZE)
    a3 = ones(SIZE)
    r = ones(SIZE)
    p = ones(SIZE)
    s = zeros(SIZE)
    x = zeros(SIZE)
    r_old = zeros(SIZE)
    r_aux = zeros(SIZE)
    a2 = a2 * 4
    r = r * 0.5
    p = p * 0.5
    cond = 1.0
    ndev = JACC.Multi.ndev()
    gja1 = JACC.Multi.array(a1; ghost_dims = 1)
    gja2 = JACC.Multi.array(a2; ghost_dims = 1)
    gja3 = JACC.Multi.array(a3; ghost_dims = 1)
    jr = JACC.Multi.array(r)
    jp = JACC.Multi.array(p)
    gjp = JACC.Multi.array(p; ghost_dims = 1)
    js = JACC.Multi.array(s)
    gjs = JACC.Multi.array(s; ghost_dims = 1)
    jx = JACC.Multi.array(x)
    jr_old = JACC.Multi.array(r_old)
    jr_aux = JACC.Multi.array(r_aux)
    ssize = JACC.Multi.part_length(jp)
    # HPCG Algorithm
    while cond >= 1e-14
        JACC.Multi.copy!(jr_old, jr)
        JACC.Multi.parallel_for(
            SIZE, matvecmul, gja1, gja2, gja3, gjp, gjs, ssize, ndev)
        JACC.Multi.sync_ghost_elems!(gjs)
        JACC.Multi.copy!(js, gjs) #js = gjs
        alpha0 = JACC.Multi.parallel_reduce(SIZE, dot, jr, jr)
        alpha1 = JACC.Multi.parallel_reduce(SIZE, dot, jp, js)
        alpha = alpha0 / alpha1
        m_alpha = alpha * (-1.0)
        JACC.Multi.parallel_for(SIZE, axpy, m_alpha, jr, js)
        JACC.Multi.parallel_for(SIZE, axpy, alpha, jx, jp)
        beta0 = JACC.Multi.parallel_reduce(SIZE, dot, jr, jr)
        beta1 = JACC.Multi.parallel_reduce(SIZE, dot, jr_old, jr_old)
        beta = beta0 / beta1
        JACC.Multi.copy!(jr_aux, jr)
        JACC.Multi.parallel_for(SIZE, axpy, beta, jr_aux, jp)
        ccond = JACC.Multi.parallel_reduce(SIZE, dot, jr, jr)
        cond = ccond
        JACC.Multi.copy!(jp, jr_aux)
        JACC.Multi.copy!(gjp, jp) #gjp = jp
        JACC.Multi.sync_ghost_elems!(gjp)
    end
    @test cond <= 1e-14
end
end
