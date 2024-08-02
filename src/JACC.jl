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

include("JACCEXPERIMENTAL.jl")
using .experimental

export Array, @atomic
export parallel_for

global Array

function parallel_for(N::I, f::F, x...) where {I <: Integer, F <: Function}
    @maybe_threaded for i in 1:N
        f(i, x...)
    end
end

function parallel_for(
        (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    @maybe_threaded for j in 1:N
        for i in 1:M
            f(i, j, x...)
        end
    end
end

function parallel_for(
        (L, M, N)::Tuple{I, I, I}, f::F, x...) where {
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

function parallel_reduce(N::I, f::F, x...) where {I <: Integer, F <: Function}
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
        (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
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

function __init__()
    const JACC.Array = Base.Array{T, N} where {T, N}
end

end # module JACC
