import JACC

using Pkg

const backend = JACC.JACCPreferences.backend

@static if backend == "cuda"
    # Pkg.add(; name = "CUDA", version = "v5.1.1")
    Pkg.add("CUDA")
    @info "CUDA backend loaded"
    include("tests_cuda.jl")

elseif backend == "amdgpu"
    Pkg.add(; name = "AMDGPU", version = "v0.8.6")
    # Pkg.add("AMDGPU")
    @info "AMDGPU backend loaded"
    include("tests_amdgpu.jl")

elseif backend == "oneapi"
    Pkg.add("oneAPI")
    @info "OneAPI backend loaded"
    include("tests_oneapi.jl")

elseif backend == "threads"
    @info "Threads backend loaded"
    include("tests_threads.jl")
end

const FloatType = JACC.default_float()
using ChangePrecision
@changeprecision FloatType begin
include("unittests.jl")
end # @changeprecision
