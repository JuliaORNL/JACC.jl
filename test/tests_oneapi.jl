
@testset "TestBackend" begin
    @test JACC.JACCPreferences.backend == "oneapi"
end

@testset "zeros_type" begin
    using oneAPI, oneAPI.oneL0
    N = 10
    x = JACC.zeros(N)
    @test typeof(x) == oneVector{FloatType,oneL0.DeviceBuffer}
    @test eltype(x) == FloatType
end

@testset "ones_type" begin
    using oneAPI, oneAPI.oneL0
    N = 10
    x = JACC.ones(N)
    @test typeof(x) == oneVector{FloatType,oneL0.DeviceBuffer}
    @test eltype(x) == FloatType
end
