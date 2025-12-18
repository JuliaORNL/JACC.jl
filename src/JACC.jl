
module JACC

import Atomix: @atomic

# module to set backend preferences
include("preferences.jl")

get_backend(backend::Symbol) = get_backend(Val(backend))

@inline default_backend() = get_backend(_backend_dispatchable)

const IDims = Union{Integer, NTuple{2, Integer}, NTuple{3, Integer}}
const AllDims = Union{Integer, NTuple{N, Integer}} where {N}

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

ilog2(n::T) where {T <: Integer} = sizeof(T) * 8 - 1 - leading_zeros(n)

default_stream() = default_stream(default_backend())

create_stream() = create_stream(default_backend())

@kwdef mutable struct LaunchSpec{Backend}
    stream = default_stream(Backend())
    threads = 0
    blocks = 0
    shmem_size::Int = -1
    sync::Bool = true
end

launch_spec(; kw...) = LaunchSpec{typeof(default_backend())}(; kw...)

default_float(::Any) = Float64

shared(x::AbstractArray) = shared(default_backend(), x)

sync_workgroup() = sync_workgroup(default_backend())

default_float() = default_float(default_backend())

synchronize(; kw...) = synchronize(default_backend(); kw...)

@inline function parallel_for(dims::AllDims, f, x...)
    parallel_for(f, dims, x...)
end

@inline function parallel_for(f, dims::NTuple{N, Integer}, x...) where {N}
    ids = CartesianIndices(dims)
    @inline function _parallel_for_kernel_1d_nd(i, ids, f, x...)::Nothing
        f(Tuple(@inbounds ids[i])..., x...)
        return nothing
    end
    parallel_for(_parallel_for_kernel_1d_nd, prod(dims), ids, f, x...)
end

@inline function parallel_for(f, dims::IDims, x...)
    parallel_for(f, default_backend(), dims, x...)
end

@inline function parallel_for(spec::LaunchSpec, dims::AllDims, f, x...)
    parallel_for(f, spec, dims, x...)
end

@inline function parallel_for(spec::LaunchSpec, dims::IDims, f, x...)
    parallel_for(f, spec, dims, x...)
end

@inline function parallel_for(; dims::AllDims, f, args::Tuple, kw...)
    parallel_for(f, launch_spec(; kw...), dims, args...)
end

default_init(::Type{T}, ::typeof(+)) where {T} = zero(T)
default_init(::Type{T}, ::typeof(*)) where {T} = one(T)
default_init(::Type{T}, ::typeof(max)) where {T} = typemin(T)
default_init(::Type{T}, ::typeof(min)) where {T} = typemax(T)
default_init(op::Function) = default_init(default_float(), op)

abstract type ReduceWorkspace end

abstract type WkProp end
struct Managed <: WkProp end
struct Unmanaged <: WkProp end

reduce_workspace() = reduce_workspace(default_backend(), default_float()())

reduce_workspace(init::T) where {T} = reduce_workspace(default_backend(), init)

@kwdef mutable struct ParallelReduce{Backend, T, Op, Dim}
    dims::Dim = zeros(Int, Dim)
    op::Op = () -> nothing
    init::T = default_init(T, op)
    stream = default_stream(Backend())
    sync::Bool = true
    workspace::ReduceWorkspace = reduce_workspace(Backend(), init)
end

@inline function ParallelReduce{Backend, T}(;
        dims, op, kw...) where {Backend, T}
    ParallelReduce{Backend, T, typeof(op), typeof(dims)}(;
        dims = dims, op = op, kw...)
end

@inline function reducer(; type = nothing, dims, op = +, init = nothing)
    _init = _resolve_init_type(op, type, init)
    ParallelReduce{
        typeof(default_backend()), typeof(_init), typeof(op), typeof(dims)}(;
        dims = dims, op = op, init = _init)
end

@inline function reducer(::Type{T}, dims::AllDims, op = +;
        init = default_init(T, op)) where {T}
    reducer(; type = T, dims = dims, op = op, init = init)
end

struct ReduceKernel1DND{T} end

@inline function (::ReduceKernel1DND{T})(i, ids, f, x...)::T where {T}
    return f(Tuple(@inbounds ids[i])..., x...)::T
end

function _parallel_reduce!(
        reducer::ParallelReduce, dims::NTuple{N, Integer}, f, x...) where {N}
    ids = CartesianIndices(dims)
    _parallel_reduce!(reducer, prod(dims),
        ReduceKernel1DND{typeof(reducer.init)}(), ids, f, x...)
end

@inline function (reducer::ParallelReduce)(f, x...)
    _parallel_reduce!(reducer, reducer.dims, f, x...)
end

@inline function (reducer::ParallelReduce)(a::AbstractArray)
    reducer(_elem_access(a), a)
end

function set_init!(reducer::ParallelReduce{B, T}, init) where {B, T}
    reducer.init = convert(T, init)
end

@inline function get_result(reducer::ParallelReduce{B, T}) where {B, T}
    get_result(reducer.workspace)::T
end

@inline _resolve_init_type(op, type, init) = convert(type, init)
@inline _resolve_init_type(op, type, init::Nothing) = default_init(type, op)
@inline _resolve_init_type(op, type::Nothing, init) = init
@inline _resolve_init_type(op, type::Nothing, init::Nothing) = default_init(op)

@inline function parallel_reduce(f, dims::AllDims, x...;
        type = nothing, op = +, init = nothing)
    _init = _resolve_init_type(op, type, init)
    return parallel_reduce(
        f, default_backend(), dims, x...; op = op, init = _init)
end

@inline function parallel_reduce(dims::AllDims, f, x...; kw...)
    return parallel_reduce(f, dims, x...; kw...)
end

@inline function JACC.parallel_reduce(f, spec::LaunchSpec{TBackend},
        dims::AllDims, x...; type = nothing, op = +,
        init = nothing) where {TBackend}
    _init = _resolve_init_type(op, type, init)
    reducer = ParallelReduce{TBackend, typeof(_init), typeof(op), typeof(dims)}(;
        dims = dims,
        op = op,
        init = _init,
        stream = spec.stream,
        sync = spec.sync,
        workspace = JACC.reduce_workspace(TBackend(), _init)
    )
    reducer(f, x...)
    return reducer.workspace.ret
end

@inline function parallel_reduce(
        spec::LaunchSpec, dims::AllDims, f, x...; kw...)
    return parallel_reduce(f, spec, dims, x...; kw...)
end

@inline function parallel_reduce(; dims::AllDims, f, args::Tuple,
        type = nothing, op = +, init = nothing, kw...)
    return parallel_reduce(f, launch_spec(; kw...), dims, args...; type = type,
        op = op, init = init)
end

array_size(a::AbstractArray) = size(a)
array_size(a::AbstractVector) = length(a)

_elem_access(a::AbstractArray) = (args...) -> args[end][args[1:(end - 1)]...]
_elem_access(a::AbstractArray{T, 3}) where {T} = (i, j, k, a) -> a[i, j, k]
_elem_access(a::AbstractMatrix) = (i, j, a) -> a[i, j]
_elem_access(a::AbstractVector) = (i, a) -> a[i]

@inline function parallel_reduce(
        op, a::AbstractArray; init = default_init(eltype(a), op))
    return parallel_reduce(
        _elem_access(a), array_size(a), a; op = op, init = init)
end

@inline parallel_reduce(a::AbstractArray; kw...) = parallel_reduce(+, a)

@inline function parallel_reduce(spec::LaunchSpec, op, a::AbstractArray;
        init = default_init(eltype(a), op))
    return parallel_reduce(
        _elem_access(a), spec, array_size(a), a; op = op, init = init)
end

@inline function parallel_reduce(spec::LaunchSpec, a::AbstractArray)
    return parallel_reduce(spec, +, a)
end

include("threads/threads.jl")

end # module JACC
