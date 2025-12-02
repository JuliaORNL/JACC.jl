import oneAPI

@testset "TestBackend" begin
    @test JACC.backend == "oneapi"
end

@testset "zeros_type" begin
    using oneAPI, oneAPI.oneL0
    N = 10
    x = JACC.zeros(N)
    @test typeof(x) == oneVector{FloatType, oneL0.DeviceBuffer}
    @test eltype(x) == FloatType
    y = JACC.zeros(Int32, N)
    @test typeof(y) == oneVector{Int32, oneL0.DeviceBuffer}
    @test eltype(y) == Int32
end

@testset "ones_type" begin
    using oneAPI, oneAPI.oneL0
    N = 10
    x = JACC.ones(N)
    @test typeof(x) == oneVector{FloatType, oneL0.DeviceBuffer}
    @test eltype(x) == FloatType
    y = JACC.ones(Int32, N)
    @test typeof(y) == oneVector{Int32, oneL0.DeviceBuffer}
    @test eltype(y) == Int32
end

@testset "fill_type" begin
    N = 10
    x = JACC.fill(10.0, N)
    @test typeof(x) == oneVector{Float64, oneL0.DeviceBuffer}
    y = JACC.fill(10, (N,))
    @test typeof(y) == oneVector{Int, oneL0.DeviceBuffer}
    x2 = JACC.fill(10.0, N, N)
    @test typeof(x2) == oneMatrix{Float64, oneL0.DeviceBuffer}
    y2 = JACC.fill(10, (N, N))
    @test typeof(y2) == oneMatrix{Int, oneL0.DeviceBuffer}
    x3 = JACC.fill(10.0, N, N, N)
    @test typeof(x3) == oneArray{Float64, 3, oneL0.DeviceBuffer}
    y3 = JACC.fill(10, (N, N, N))
    @test typeof(y3) == oneArray{Int, 3, oneL0.DeviceBuffer}
end

@testset "stream" begin
    using oneAPI, oneAPI.oneL0
    sd1 = JACC.default_stream()
    @test typeof(sd1) == ZeCommandQueue
    sd2 = JACC.default_stream()
    @test sd2 === sd1
    s1 = JACC.create_stream()
    @test typeof(s1) == ZeCommandQueue
    @test s1 != sd1
    s2 = JACC.create_stream()
    @test s2 != s1
end
