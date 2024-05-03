import AMDGPU
import JACC
using Test

@testset "TestBackend" begin
    @test JACC.JACCPreferences.backend == "amdgpu"
end

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

@testset "zeros" begin
    N = 10
    x = JACC.zeros(Float64, N)
    @test typeof(x) == AMDGPU.ROCArray{Float64, 1, AMDGPU.Runtime.Mem.HIPBuffer}
    @test eltype(x) == Float64
    @test zeros(N)≈Array(x) rtol=1e-5

    function add_one(i, x)
        @inbounds x[i] += 1
    end

    JACC.parallel_for(N, add_one, x)
    @test ones(N)≈Array(x) rtol=1e-5
end

@testset "ones" begin
    N = 10
    x = JACC.ones(Float64, N)
    @test typeof(x) == AMDGPU.ROCArray{Float64, 1, AMDGPU.Runtime.Mem.HIPBuffer}
    @test eltype(x) == Float64
    @test ones(N)≈Array(x) rtol=1e-5

    function minus_one(i, x)
        @inbounds x[i] -= 1
    end

    JACC.parallel_for(N, minus_one, x)
    @test zeros(N)≈Array(x) rtol=1e-5
end
