import JACC
JACC._check_install_backend()
JACC.@init_backend

using ReTest
include("JACCTests.jl")

if isempty(ARGS)
    retest(JACCTests)
else
    retest(JACCTests, ARGS)
end
