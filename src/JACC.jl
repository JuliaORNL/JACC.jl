
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

default_init(::Type{T}, ::typeof(+)) where {T} = zero(T)
default_init(::Type{T}, ::typeof(*)) where {T} = one(T)
default_init(::Type{T}, ::typeof(max)) where{T} = typemin(T)
default_init(::Type{T}, ::typeof(min)) where{T} = typemax(T)
default_init(op::Function) = default_init(default_float(), op)

function parallel_reduce(
        N::I, op::Function, f::Function, x...; init) where {I <: Integer}
    return parallel_reduce(default_backend(), N, op, f, x...; init = init)
end

function parallel_reduce((M, N)::Tuple{I, I}, op::Function, f::Function, x...;
        init) where {I <: Integer}
    return parallel_reduce(default_backend(), (M, N), op, f, x...; init = init)
end

function parallel_reduce(N::Integer, f::Function, x...)
    return parallel_reduce(N, +, f, x...; init = default_init(+))
end

function parallel_reduce((M, N)::Tuple{Integer, Integer}, f::Function, x...)
    return parallel_reduce((M, N), +, f, x...; init = default_init(+))
end

array_size(a::AbstractArray) = size(a)
array_size(a::AbstractVector) = length(a)

elem_access(a::AbstractArray) = (i,j,k,a) -> a[i,j,k]
elem_access(a::AbstractMatrix) = (i,j,a) -> a[i,j]
elem_access(a::AbstractVector) = (i,a) -> a[i]

function parallel_reduce(op::Function, a::AbstractArray; init = default_init(eltype(a), op))
    return parallel_reduce(array_size(a), op, elem_access(a), a; init = init)
end

parallel_reduce(a::AbstractArray; kw...) = parallel_reduce(+, a; kw...)

end # module JACC
