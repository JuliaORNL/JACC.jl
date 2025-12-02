import AMDGPU

@testset "TestBackend" begin
    @test JACC.backend == "amdgpu"
end

@testset "zeros_type" begin
    N = 10
    x = JACC.zeros(Float64, N)
    @test typeof(x) == AMDGPU.ROCArray{Float64, 1, AMDGPU.Runtime.Mem.HIPBuffer}
    @test eltype(x) == Float64
    x = JACC.zeros(Int32, N)
    @test typeof(x) == AMDGPU.ROCArray{Int32, 1, AMDGPU.Runtime.Mem.HIPBuffer}
    @test eltype(x) == Int32
end

@testset "ones_type" begin
    N = 10
    x = JACC.ones(Float64, N)
    @test typeof(x) == AMDGPU.ROCArray{Float64, 1, AMDGPU.Runtime.Mem.HIPBuffer}
    @test eltype(x) == Float64
    x = JACC.ones(Int32, N)
    @test typeof(x) == AMDGPU.ROCArray{Int32, 1, AMDGPU.Runtime.Mem.HIPBuffer}
    @test eltype(x) == Int32
end

@testset "fill_type" begin
    N = 10
    x = JACC.fill(10.0, N)
    @test typeof(x) == AMDGPU.ROCArray{Float64, 1, AMDGPU.Runtime.Mem.HIPBuffer}
    y = JACC.fill(10, (N,))
    @test typeof(y) == AMDGPU.ROCArray{Int, 1, AMDGPU.Runtime.Mem.HIPBuffer}
    x2 = JACC.fill(10.0, N, N)
    @test typeof(x2) ==
          AMDGPU.ROCArray{Float64, 2, AMDGPU.Runtime.Mem.HIPBuffer}
    y2 = JACC.fill(10, (N, N))
    @test typeof(y2) == AMDGPU.ROCArray{Int, 2, AMDGPU.Runtime.Mem.HIPBuffer}
    x3 = JACC.fill(10.0, N, N, N)
    @test typeof(x3) ==
          AMDGPU.ROCArray{Float64, 3, AMDGPU.Runtime.Mem.HIPBuffer}
    y3 = JACC.fill(10, (N, N, N))
    @test typeof(y3) == AMDGPU.ROCArray{Int, 3, AMDGPU.Runtime.Mem.HIPBuffer}
end

@testset "stream" begin
    using AMDGPU
    sd1 = JACC.default_stream()
    @test typeof(sd1) == HIPStream
    sd2 = JACC.default_stream()
    @test sd2 === sd1
    s1 = JACC.create_stream()
    @test typeof(s1) == HIPStream
    @test s1 != sd1
    s2 = JACC.create_stream()
    @test s2 != s1
end
