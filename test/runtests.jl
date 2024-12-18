import JACC
JACC.@init_backend

using ReTest
include("JACCTests.jl")

if isempty(ARGS)
    retest(JACCTests)
else
    retest(JACCTests, ARGS)
end
