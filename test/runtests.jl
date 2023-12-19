import JACC

const backend = JACC.JACCPreferences.backend

@static if backend == "cuda"
    println("CUDA backend loaded")
    include("tests_cuda.jl")
    if "perf" in ARGS
        println("Running performance tests")
        include("tests_cuda_perf.jl")
    end
elseif backend == "amdgpu"
    @show "AMDGPU backend loaded"
    include("tests_amdgpu.jl")
elseif backend == "oneapi"
    @show "OneAPI backend loaded"
    include("tests_oneapi.jl")
elseif backend == "threads"
    @show "Threads backend loaded"
    include("tests_threads.jl")
    if "perf" in ARGS
        println("Running performance tests")
        include("tests_threads_perf.jl")
    end
end
