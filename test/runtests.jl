import JACC

using Pkg

const backend = JACC.JACCPreferences.backend

@static if backend == "cuda"
    # Pkg.add(; name = "CUDA", version = "v5.1.1")
    Pkg.add("CUDA")
    println("CUDA backend loaded")
    include("tests_cuda.jl")

elseif backend == "amdgpu"
    Pkg.add(; name = "AMDGPU", version = "v0.8.6")
    # Pkg.add("AMDGPU")
    println("AMDGPU backend loaded")
    include("tests_amdgpu.jl")

elseif backend == "oneapi"
    Pkg.add("oneAPI")
    println("OneAPI backend loaded")
    include("tests_oneapi.jl")

elseif backend == "threads"
    println("Threads backend loaded")
    include("tests_threads.jl")
end

include("unittests.jl")
