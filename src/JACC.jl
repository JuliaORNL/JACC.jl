
module JACC

import Base: Callable
import Atomix: @atomic

# module to set backend preferences 
include("preferences.jl")

function get_backend end

default_backend() = get_backend(_backend_dispatchable)

include("array.jl")

include("blas.jl")
using .BLAS

include("multi.jl")
using .Multi

include("async.jl")
using .Async

include("experimental/experimental.jl")
using .Experimental

export array_type, array
export default_float
export @atomic
export parallel_for, parallel_reduce
export shared
export LaunchSpec
export synchronize

function default_stream end

@kwdef mutable struct LaunchSpec{Backend}
    stream = default_stream(Backend)
    threads = 0
    blocks = 0
    shmem_size::Int = 0
    sync::Bool = false
end

launch_spec(; kw...) = LaunchSpec{typeof(default_backend())}(; kw...)

default_float(::Any) = Float64

function shared(x::Base.Array{T, N}) where {T, N}
    return x
end

array_type() = array_type(default_backend())

array(x::Base.Array) = array(default_backend(), x)

default_float() = default_float(default_backend())

synchronize(; kw...) = synchronize(default_backend(); kw...)

function parallel_for(N::Integer, f::Callable, x...)
    return parallel_for(default_backend(), N, f, x...)
end

function parallel_for((M, N)::NTuple{2, Integer}, f::Callable, x...)
    return parallel_for(default_backend(), (M, N), f, x...)
end

function parallel_for((L, M, N)::NTuple{3, Integer}, f::Callable, x...)
    return parallel_for(default_backend(), (L, M, N), f, x...)
end

default_init(::Type{T}, ::typeof(+)) where {T} = zero(T)
default_init(::Type{T}, ::typeof(*)) where {T} = one(T)
default_init(::Type{T}, ::typeof(max)) where {T} = typemin(T)
default_init(::Type{T}, ::typeof(min)) where {T} = typemax(T)
default_init(op::Callable) = default_init(default_float(), op)

function parallel_reduce(N::Integer, op::Callable, f::Callable, x...; init)
    return parallel_reduce(default_backend(), N, op, f, x...; init = init)
end

function parallel_reduce(
        (M, N)::NTuple{2, Integer}, op::Callable, f::Callable, x...;
        init)
    return parallel_reduce(default_backend(), (M, N), op, f, x...; init = init)
end

function parallel_reduce(N::Integer, f::Callable, x...)
    return parallel_reduce(N, +, f, x...; init = default_init(+))
end

function parallel_reduce((M, N)::NTuple{2, Integer}, f::Callable, x...)
    return parallel_reduce((M, N), +, f, x...; init = default_init(+))
end

array_size(a::AbstractArray) = size(a)
array_size(a::AbstractVector) = length(a)

_elem_access(a::AbstractArray) = (i, j, k, a) -> a[i, j, k]
_elem_access(a::AbstractMatrix) = (i, j, a) -> a[i, j]
_elem_access(a::AbstractVector) = (i, a) -> a[i]

function parallel_reduce(
        op::Callable, a::AbstractArray; init = default_init(eltype(a), op))
    return parallel_reduce(array_size(a), op, _elem_access(a), a; init = init)
end

parallel_reduce(a::AbstractArray; kw...) = parallel_reduce(+, a; kw...)

include("threads/threads.jl")

end # module JACC
