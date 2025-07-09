import CUDA

@testset "TestBackend" begin
    @test JACC.backend == "cuda"
end

@testset "zeros_type" begin
    using CUDA
    N = 10
    x = JACC.zeros(Float64, N)
    @test typeof(x) == CUDA.CuArray{Float64, 1, CUDA.DeviceMemory}
    @test eltype(x) == Float64
    y = JACC.zeros(Int32, N)
    @test typeof(y) == CUDA.CuArray{Int32, 1, CUDA.DeviceMemory}
    @test eltype(y) == Int32
end

@testset "ones_type" begin
    using CUDA
    N = 10
    x = JACC.ones(Float64, N)
    @test typeof(x) == CUDA.CuArray{Float64, 1, CUDA.DeviceMemory}
    @test eltype(x) == Float64
    y = JACC.ones(Int32, N)
    @test typeof(y) == CUDA.CuArray{Int32, 1, CUDA.DeviceMemory}
    @test eltype(y) == Int32
end

@testset "fill_type" begin
    N = 10
    x = JACC.fill(10.0, N)
    @test typeof(x) == CUDA.CuArray{Float64, 1, CUDA.DeviceMemory}
    y = JACC.fill(10, (N,))
    @test typeof(y) == CUDA.CuArray{Int, 1, CUDA.DeviceMemory}
    x2 = JACC.fill(10.0, N, N)
    @test typeof(x2) == CUDA.CuArray{Float64, 2, CUDA.DeviceMemory}
    y2 = JACC.fill(10, (N, N))
    @test typeof(y2) == CUDA.CuArray{Int, 2, CUDA.DeviceMemory}
    x3 = JACC.fill(10.0, N, N, N)
    @test typeof(x3) == CUDA.CuArray{Float64, 3, CUDA.DeviceMemory}
    y3 = JACC.fill(10, (N, N, N))
    @test typeof(y3) == CUDA.CuArray{Int, 3, CUDA.DeviceMemory}
end
