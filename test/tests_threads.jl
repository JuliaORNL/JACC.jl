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