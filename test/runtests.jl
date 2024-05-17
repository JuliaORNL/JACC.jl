import JACC

using Pkg

const backend = JACC.JACCPreferences.backend
using Test: @test

@static if backend == "cuda"
    Pkg.add(; name = "CUDA", version = "v5.1.1")
    @show "CUDA backend loaded"
    using CUDA: CuArray
    @test JACC.arraytype() <: CuArray
    include("tests_cuda.jl")

elseif backend == "amdgpu"
    Pkg.add(; name = "AMDGPU")
    @show "AMDGPU backend loaded"
    using AMDGPU: ROCArray
    @test JACC.arraytype() <: ROCArray
    include("tests_amdgpu.jl")

elseif backend == "oneapi"
    Pkg.add("oneAPI")
    @show "OneAPI backend loaded"
    using OneAPI: oneArray
    @test JACC.arraytype() <: oneArray
    include("tests_oneapi.jl")

elseif backend == "threads"
    @show "Threads backend loaded"
    @test JACC.arraytype() <: Array
end

include("tests_threads.jl")
