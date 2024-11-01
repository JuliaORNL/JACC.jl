import JACC
using Test

@testset "TestBackend" begin
    @test JACC.JACCPreferences.backend == "threads"
end

@testset "zeros_type" begin
    N = 10
    x = JACC.zeros(Float32, N)
    @test typeof(x) == Vector{Float32}
    @test eltype(x) == Float32
end

@testset "ones_type" begin
    N = 10
    x = JACC.ones(Float64, N)
    @test typeof(x) == Vector{Float64}
    @test eltype(x) == Float64
end
