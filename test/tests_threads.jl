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
    JACC.parallel_for(N, axpy, alpha, x_host_JACC, y_host_JACC)

    x_expected = x
    seq_axpy(N, alpha, x_expected, y)

    @test x_host_JACC ≈ x_expected rtol = 1e-1
end

@testset "CG" begin

    function matvecmul(i,a1,a2,a3,x,y,SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i+1]
        elseif i == SIZE
            y[i] = a3[i] * x[i-1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i-1] + a1[i] * + x[i] + a1[i] * + x[i+1]
        end
    end
 
    function dot(i,x,y)
        @inbounds return x[i] * y[i]
    end

    function axpy(i, alpha, x, y)
        @inbounds x[i] += alpha[1,1] * y[i]
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

    while cond[1,1] >= 1e-14

        r_old = copy(r)

        JACC.parallel_for(SIZE,matvecmul,a0,a1,a2,p,s,SIZE)

        alpha0 = JACC.parallel_reduce(SIZE,dot,r,r)
        alpha1 = JACC.parallel_reduce(SIZE,dot,p,s)

        alpha = alpha0 / alpha1
        negative_alpha = alpha * (-1.0)

        JACC.parallel_for(SIZE,axpy,negative_alpha,r,s)
        JACC.parallel_for(SIZE,axpy,alpha,x,p)

        beta0 = JACC.parallel_reduce(SIZE,dot,r,r)
        beta1 = JACC.parallel_reduce(SIZE,dot,r_old,r_old)
        beta = beta0 / beta1

        r_aux = copy(r)

        JACC.parallel_for(SIZE,axpy,beta,r_aux,p)
        ccond = JACC.parallel_reduce(SIZE,dot,r,r)
        global cond = ccond
        p = copy(r_aux)

        println(cond)

    end
    @test cond <= 1e-14
end
