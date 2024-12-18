
module JACC

import Atomix: @atomic

# module to set backend preferences 
include("JACCPreferences.jl")

default_backend() = get_backend(JACCPreferences._backend_dispatchable)

struct ThreadsBackend end

include("helper.jl")
# overloaded array functions
include("array.jl")

include("JACCBLAS.jl")
using .BLAS

include("JACCMULTI.jl")
using .Multi

include("JACCASYNC.jl")
using .Async

include("JACCEXPERIMENTAL.jl")
using .Experimental

get_backend(::Val{:threads}) = ThreadsBackend()

export Array, @atomic
export parallel_for
export parallel_reduce

global Array

function parallel_for(
        ::ThreadsBackend, N::I, f::F, x...) where {I <: Integer, F <: Function}
    @maybe_threaded for i in 1:N
        f(i, x...)
    end
end

function parallel_for(
        ::ThreadsBackend, (M, N)::Tuple{I, I}, f::F, x...) where {
        I <: Integer, F <: Function}
    @maybe_threaded for j in 1:N
        for i in 1:M
            f(i, j, x...)
        end
    end
end

function parallel_for(
        ::ThreadsBackend, (L, M, N)::Tuple{I, I, I}, f::F,
        x...) where {
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

function parallel_reduce(
        ::ThreadsBackend, N::Integer, op, f::Function, x...; init)
    ret = init
    tmp = fill(init, Threads.nthreads())
    @maybe_threaded for i in 1:N
        tmp[Threads.threadid()] = op.(tmp[Threads.threadid()], f(i, x...))
    end
    for i in 1:Threads.nthreads()
        ret = op.(ret, tmp[i])
    end
    return ret
end

function parallel_reduce(
        ::ThreadsBackend, (M, N)::Tuple{Integer, Integer}, op, f::Function, x...; init)
    ret = init
    tmp = fill(init, Threads.nthreads())
    @maybe_threaded for j in 1:N
        for i in 1:M
            tmp[Threads.threadid()] = op.(
                tmp[Threads.threadid()], f(i, j, x...))
        end
    end
    for i in 1:Threads.nthreads()
        ret = op.(ret, tmp[i])
    end
    return ret
end

array_type(::ThreadsBackend) = Base.Array{T, N} where {T, N}

default_float(::Any) = Float64

function shared(x::Base.Array{T, N}) where {T, N}
    return x
end

struct Array{T, N} end
function (::Type{Array{T, N}})(args...; kwargs...) where {T, N}
    array_type(){T, N}(args...; kwargs...)
end
function (::Type{Array{T}})(args...; kwargs...) where {T}
    array_type(){T}(args...; kwargs...)
end
(::Type{Array})(args...; kwargs...) = array_type()(args...; kwargs...)

array_type() = array_type(default_backend())

default_float() = default_float(default_backend())

function parallel_for(N::I, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_for(default_backend(), N, f, x...)
end

function parallel_for(
        (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_for(default_backend(), (M, N), f, x...)
end

function parallel_for((L, M, N)::Tuple{I, I, I}, f::F,
        x...) where {I <: Integer, F <: Function}
    return parallel_for(default_backend(), (L, M, N), f, x...)
end

function parallel_reduce(
        N::I, op, f::F, x...; init) where {I <: Integer, F <: Function}
    return parallel_reduce(default_backend(), N, op, f, x...; init = init)
end

function parallel_reduce(N::Integer, f::Function, x...)
    return parallel_reduce(N, +, f, x...; init = zero(default_float()))
end

function parallel_reduce((M, N)::Tuple{I, I}, op, f::F, x...;
        init) where {I <: Integer, F <: Function}
    return parallel_reduce(default_backend(), (M, N), op, f, x...; init = init)
end

function parallel_reduce((M, N)::Tuple{Integer, Integer}, f::Function, x...)
    return parallel_reduce((M, N), +, f, x...; init = zero(default_float()))
end

end # module JACC
