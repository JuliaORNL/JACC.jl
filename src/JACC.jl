module JACC

global BackendTag

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

@inline function parallel_for(N::I, f::F, x...) where {I <: Integer, F <: Function}
    _parallel_for_impl(BackendTag(), N, f, x...)
end

@inline function parallel_for((M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    _parallel_for_impl(BackendTag(), (M, N), f, x...)
end

@inline function parallel_for((L, M, N)::Tuple{I, I, I}, f::F, x...) where {
        I <: Integer, F <: Function}
    _parallel_for_impl(BackendTag(), (L,M,N), f, x...)
end

@inline function parallel_reduce(N::I, f::F, x...) where {I <: Integer, F <: Function}
    _parallel_reduce_impl(BackendTag(), N, f, x...)
end

@inline function parallel_reduce((M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    _parallel_reduce_impl(BackendTag(), (M, N), f, x...)
end

function shared(x::Base.Array{T,N}) where {T,N}
  return x
end

include("JACCThreads.jl")

end # module JACC
