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

struct Array{T, N} end
(::Type{Array{T, N}})(args...; kwargs...) where {T, N} =
    array_type(){T, N}(args...; kwargs...)
(::Type{Array{T}})(args...; kwargs...) where {T} =
    array_type(){T}(args...; kwargs...)
(::Type{Array})(args...; kwargs...) =
    array_type()(args...; kwargs...)

end # module JACC
