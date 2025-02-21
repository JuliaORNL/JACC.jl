module JACCTests

import JACC
using ReTest

@static if JACC.backend == "cuda"
    using CUDA
    include("tests_cuda.jl")
elseif JACC.backend == "amdgpu"
    using AMDGPU
    include("tests_amdgpu.jl")
elseif JACC.backend == "oneapi"
    using oneAPI
    include("tests_oneapi.jl")
elseif JACC.backend == "threads"
    include("tests_threads.jl")
end

const FloatType = JACC.default_float()
using ChangePrecision
@changeprecision FloatType begin
    include("unittests.jl")
end # @changeprecision

end
