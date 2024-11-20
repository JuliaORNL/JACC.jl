
@testitem "TestBackend" tags=[:threads] begin
    @test JACC.JACCPreferences.backend == "threads"
end

@testitem "zeros_type" tags=[:threads] begin
    N = 10
    x = JACC.zeros(Float32, N)
    @test typeof(x) == Vector{Float32}
    @test eltype(x) == Float32
end

@testitem "ones_type" tags=[:threads] begin
    N = 10
    x = JACC.ones(Float64, N)
    @test typeof(x) == Vector{Float64}
    @test eltype(x) == Float64
end
