import JACC

using Pkg

const backend = JACC.JACCPreferences.backend

@static if backend == "cuda"
    Pkg.add(; name = "CUDA", version = "v5.1.1")
    # Pkg.add("CUDA")
    @info "CUDA backend loaded"
    using CUDA

elseif backend == "amdgpu"
    Pkg.add(; name = "AMDGPU", version = "v0.8.6")
    # Pkg.add("AMDGPU")
    @info "AMDGPU backend loaded"
    using AMDGPU

elseif backend == "oneapi"
    Pkg.add("oneAPI")
    @info "oneAPI backend loaded"
    using oneAPI

elseif backend == "threads"
    @info "Threads backend loaded"
end

using ReTest
include("JACCTests.jl")

if isempty(ARGS)
    retest(JACCTests)
else
    retest(JACCTests, ARGS)
end
