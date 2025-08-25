module JACCBench

function matches(args)
    if isempty(args)
        return false
    end
    return match(r"bench", args[1]) !== nothing
end

import JACC
using ReTest
using ..JACCTestCommon
include("benchmarks/comps.jl")
include("benchmarks/config.jl")

if JACC.backend == "cuda"
    include("benchmarks/backend/cuda.jl")
elseif JACC.backend == "amdgpu"
    include("benchmarks/backend/amdgpu.jl")
elseif JACC.backend == "oneapi"
    include("benchmarks/backend/oneapi.jl")
elseif JACC.backend == "threads"
    include("benchmarks/backend/threads.jl")
end

const FloatType = JACC.default_float()
using ChangePrecision
@changeprecision FloatType begin
    include("benchmarks/benchmarks.jl")
end # @changeprecision

end
