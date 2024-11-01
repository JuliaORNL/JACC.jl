using oneAPI
import JACC
using Test

@testset "TestBackend" begin
    @test JACC.JACCPreferences.backend == "oneapi"
end
