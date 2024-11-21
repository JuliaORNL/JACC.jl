
@testitem "TestBackend" tags=[:oneapi] begin
    @test JACC.JACCPreferences.backend == "oneapi"
end
