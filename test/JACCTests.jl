module JACCTests

import JACC
using ReTest
using ..JACCTestCommon

if JACC.backend == "cuda"
    include("backend/cuda.jl")
elseif JACC.backend == "amdgpu"
    include("backend/amdgpu.jl")
elseif JACC.backend == "oneapi"
    include("backend/oneapi.jl")
elseif JACC.backend == "threads"
    include("backend/threads.jl")
end

const FloatType = JACC.default_float()
using ChangePrecision
@changeprecision FloatType begin
    include("unittests.jl")
end # @changeprecision

end
