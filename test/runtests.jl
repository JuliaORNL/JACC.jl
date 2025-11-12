import JACC
JACC._check_install_backend()
JACC.@init_backend

using ReTest
include("common.jl")
include("JACCBench.jl")
include("JACCTests.jl")

if JACCBench.matches(ARGS)
    popfirst!(ARGS)
    filter = JACCBench.getconf().filter
    if isempty(filter)
        retest(JACCBench; spin = false, stats = true)
    else
        retest(JACCBench, filter; spin = false)
    end
else
    if isempty(ARGS)
        retest(JACCTests)
    else
        retest(JACCTests, ARGS)
    end
end
