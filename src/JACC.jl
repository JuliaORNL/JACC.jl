
module JACC

import Base: Callable
import Atomix: @atomic

# module to set backend preferences 
include("preferences.jl")

function get_backend end

default_backend() = get_backend(_backend_dispatchable)

include("array.jl")
include("blas.jl")
include("multi.jl")
include("async.jl")
include("experimental/experimental.jl")

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

shared(x::AbstractArray) = shared(default_backend(), x)

sync_workgroup() = sync_workgroup(default_backend())

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

abstract type ReduceWorkspace end

reduce_workspace() = reduce_workspace(default_backend(), default_float()())

reduce_workspace(init::T) where {T} = reduce_workspace(default_backend(), init)

const Dims = Union{Integer, NTuple{2, Integer}, NTuple{3, Integer}}

@kwdef mutable struct ParallelReduce{Backend, T}
    dims::Dims = 0
    op::Callable = () -> nothing
    init::T = default_init(op)
    workspace::ReduceWorkspace = reduce_workspace(Backend(), init)
    spec::LaunchSpec{Backend} = LaunchSpec{Backend}()
end

function reducer(; dims, op, init = default_init(op))
    ParallelReduce{typeof(default_backend()), typeof(init)}(;
        dims = dims, op = op, init = init)
end

function reducer(dims::Dims, op::Callable; init = default_init(op))
    reducer(; dims = dims, op = op, init = init)
end

function _parallel_reduce! end

@inline function (reducer::ParallelReduce)(f::Callable, x...)
    _parallel_reduce!(reducer, reducer.dims, f, x...)
end

@inline function (reducer::ParallelReduce)(a::AbstractArray)
    reducer(_elem_access(a), a)
end

function set_init!(reducer::ParallelReduce, init)
    reducer.init = init
end

get_result(reducer::ParallelReduce) = get_result(reducer.workspace)

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

function parallel_reduce(spec::LaunchSpec, N::Integer, f::Callable, x...)
    return parallel_reduce(spec, N, +, f, x...; init = default_init(+))
end

function parallel_reduce((M, N)::NTuple{2, Integer}, f::Callable, x...)
    return parallel_reduce((M, N), +, f, x...; init = default_init(+))
end

function parallel_reduce(
        spec::LaunchSpec, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    return parallel_reduce(spec, (M, N), +, f, x...; init = default_init(+))
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
function parallel_reduce(
        spec::LaunchSpec, op::Callable, a::AbstractArray; init = default_init(
            eltype(a), op))
    return parallel_reduce(
        spec, array_size(a), op, _elem_access(a), a; init = init)
end

parallel_reduce(a::AbstractArray; kw...) = parallel_reduce(+, a; kw...)
function parallel_reduce(spec::LaunchSpec, a::AbstractArray; kw...)
    parallel_reduce(spec, +, a; kw...)
end

include("threads/threads.jl")

end # module JACC
