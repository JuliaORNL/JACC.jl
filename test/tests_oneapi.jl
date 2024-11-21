
@testitem "TestBackend" setup=[JACCTestItem] tags=[:oneapi] begin
    @test JACC.JACCPreferences.backend == "oneapi"
end

@testitem "zeros_type" setup=[JACCTestItem] tags=[:oneapi] begin
    using oneAPI, oneAPI.oneL0
    N = 10
    x = JACC.zeros(N)
    @test typeof(x) == oneVector{FloatType,oneL0.DeviceBuffer}
    @test eltype(x) == FloatType
end

@testitem "ones_type" setup=[JACCTestItem] tags=[:oneapi] begin
    using oneAPI, oneAPI.oneL0
    N = 10
    x = JACC.ones(N)
    @test typeof(x) == oneVector{FloatType,oneL0.DeviceBuffer}
    @test eltype(x) == FloatType
end
