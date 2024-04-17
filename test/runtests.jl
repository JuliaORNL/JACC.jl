using JACC: JACC

using Pkg

const backend = JACC.JACCPreferences.backend

@static if backend == "cuda"
	Pkg.add(name = "CUDA", version = "v5.1.1")
	@show "CUDA backend loaded"
	include("tests_cuda.jl")

elseif backend == "amdgpu"
	Pkg.add(name = "AMDGPU", version = "v0.8.6")
	@show "AMDGPU backend loaded"
	include("tests_amdgpu.jl")

elseif backend == "oneapi"
	Pkg.add("oneAPI")
	@show "OneAPI backend loaded"
	include("tests_oneapi.jl")

elseif backend == "threads"
	@show "Threads backend loaded"
	include("tests_threads.jl")

end
