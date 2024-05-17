using oneAPI
import JACC
using Test

@testset "TestBackend" begin
    @test JACC.JACCPreferences.backend == "oneapi"
end

@testset "VectorAddLambda" begin
    function f(i, a)
        @inbounds a[i] += 5.0
    end

    N = 10
    dims = (N)
    a = round.(rand(Float32, dims) * 100)

    a_device = JACC.array(a)
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
    alpha::Float32 = 2.5

    x_device = JACC.array(x)
    y_device = JACC.array(y)
    JACC.parallel_for(N, axpy, alpha, x_device, y_device)

    x_expected = x
    seq_axpy(N, alpha, x_expected, y)

    @test Array(x_device)≈x_expected rtol=1e-1
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
    
    SIZE = Int32(1_000)
    x = ones(Float32, SIZE)
    y = ones(Float32, SIZE)
    jx = JACC.ones(Float32, SIZE)
    jy = JACC.ones(Float32, SIZE)
    alpha = Float32(2.0)
    
    seq_axpy(SIZE, alpha, x, y)
    ref_result = seq_dot(SIZE, x, y)
    
    JACC.BLAS.axpy(SIZE, alpha, jx, jy)
    jresult = JACC.BLAS.dot(SIZE, jx, jy)
    result = Array(jresult)     
    
    @test result[1]≈ref_result rtol=1e-8

end

