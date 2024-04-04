using JACC
using CUDA
using AMDGPU
using oneAPI
using Test

@testset "JACC Tests" begin
if CUDA.functional()
    @testset "CUDA" begin
        println("CUDA backend")
        include("tests_cuda.jl")
    end
end

if AMDGPU.functional()
    @testset "AMDGPU" begin
        println("AMDGPU backend")
        include("tests_amdgpu.jl")
    end
end

if oneAPI.functional()
    @testset "oneAPI" begin
        println("OneAPI backend")
        include("tests_oneapi.jl")
    end
end

@testset "ThreadsBackend" begin
    println("Threads backend")
    include("tests_threads.jl")
end
end
