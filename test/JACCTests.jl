module JACCTests

import JACC
using ReTest

const backend = JACC.JACCPreferences.backend

@static if backend == "cuda"
    using CUDA
    include("tests_cuda.jl")
elseif backend == "amdgpu"
    using AMDGPU
    include("tests_amdgpu.jl")
elseif backend == "oneapi"
    using oneAPI
    include("tests_oneapi.jl")
elseif backend == "threads"
    include("tests_threads.jl")
end

const FloatType = JACC.default_float()
using ChangePrecision
@changeprecision FloatType begin
    include("unittests.jl")
end # @changeprecision

end
