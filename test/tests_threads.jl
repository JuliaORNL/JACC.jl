
@testset "TestBackend" begin
    @test JACC.backend == "threads"
end

@testset "zeros_type" begin
    N = 10
    x = JACC.zeros(Float32, N)
    @test typeof(x) == Vector{Float32}
    @test eltype(x) == Float32
    y = JACC.zeros(Int32, N)
    @test typeof(y) == Vector{Int32}
    @test eltype(y) == Int32
end

@testset "ones_type" begin
    N = 10
    x = JACC.ones(Float64, N)
    @test typeof(x) == Vector{Float64}
    @test eltype(x) == Float64
    y = JACC.ones(Int32, N)
    @test typeof(y) == Vector{Int32}
    @test eltype(y) == Int32
end

@testset "fill_type" begin
    N = 10
    x = JACC.fill(10.0, N)
    @test typeof(x) == Vector{Float64}
    y = JACC.fill(10, (N,))
    @test typeof(y) == Vector{Int}
    x2 = JACC.fill(10.0, N, N)
    @test typeof(x2) == Matrix{Float64}
    y2 = JACC.fill(10, (N, N))
    @test typeof(y2) == Matrix{Int}
    x3 = JACC.fill(10.0, N, N, N)
    @test typeof(x3) == Array{Float64, 3}
    y3 = JACC.fill(10, (N, N, N))
    @test typeof(y3) == Array{Int, 3}
end
