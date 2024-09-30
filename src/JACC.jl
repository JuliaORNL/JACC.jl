__precompile__(false)
module JACC

import Atomix: @atomic
# module to set back end preferences 
include("JACCPreferences.jl")
include("helper.jl")
# overloaded array functions
include("array.jl")

include("JACCBLAS.jl")
using .BLAS

include("JACCMULTI.jl")
using .multi

include("JACCEXPERIMENTAL.jl")
using .experimental

export Array, @atomic
export parallel_for

global Array

function parallel_for(::Val{:threads}, N::I, f::F, x...) where {I <: Integer, F <: Function}
    @maybe_threaded for i in 1:N
        f(i, x...)
    end
end

function parallel_for(
        ::Val{:threads}, (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    @maybe_threaded for j in 1:N
        for i in 1:M
            f(i, j, x...)
        end
    end
end

function parallel_for(
        ::Val{:threads}, (L, M, N)::Tuple{I, I, I}, f::F, x...) where {
        I <: Integer, F <: Function}
    # only threaded at the first level (no collapse equivalent)
    @maybe_threaded for k in 1:N
        for j in 1:M
            for i in 1:L
                f(i, j, k, x...)
            end
        end
    end
end

function parallel_reduce(::Val{:threads}, N::I, f::F, x...) where {I <: Integer, F <: Function}
    tmp = zeros(Threads.nthreads())
    ret = zeros(1)
    @maybe_threaded for i in 1:N
        tmp[Threads.threadid()] = tmp[Threads.threadid()] .+ f(i, x...)
    end
    for i in 1:Threads.nthreads()
        ret = ret .+ tmp[i]
    end
    return ret
end

function parallel_reduce(
        ::Val{:threads}, (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    tmp = zeros(Threads.nthreads())
    ret = zeros(1)
    @maybe_threaded for j in 1:N
        for i in 1:M
            tmp[Threads.threadid()] = tmp[Threads.threadid()] .+ f(i, j, x...)
        end
    end
    for i in 1:Threads.nthreads()
        ret = ret .+ tmp[i]
    end
    return ret
end

function shared(x::Base.Array{T,N}) where {T,N}
  return x
end

array_type() = array_type(JACCPreferences._backend_dispatchable)
array_type(::Val{:threads}) = Base.Array{T, N} where {T, N}
function array_type(::Val{backend}) where backend
    if backend == :cuda
        mod = "CUDA"
    elseif backend == :amdgpu
        mod = "AMDGPU"
    elseif backend == :oneapi
        mod = "oneAPI"
    else
        mod = nothing
    end
    @error("""
           Unavailable backend: $backend\n
           $(mod !== nothing ? "Please do `using $(mod)` before `using JACC`" : "Backend must be one of: \"cuda\", \"amdgpu\", \"oneapi\"")
           """)
    throw(UnavailableBackendException())
end
struct UnavailableBackendException <: Exception end

function init()
    # FIXME: This is racey, and depends on the correct extension already being loaded
    # A better solution may become available if module property access becomes dispatchable
    try
        JACC.eval(:(const JACC.Array = $array_type()))
        return true
    catch err
        err isa UnavailableBackendException || rethrow(err)
        return false
    end
end

end # module JACC
