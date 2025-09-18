module ThreadsImpl

import Base: Callable
import JACC
import JACC: LaunchSpec
import OhMyThreads
import Base.Iterators: partition

struct ThreadsBackend end

@inline JACC.get_backend(::Val{:threads}) = ThreadsBackend()

function _maybe_threaded(ex)
    quote
        if Threads.nthreads() == 1
            $ex
        else
            OhMyThreads.@tasks $ex
        end
    end
end

macro maybe_threaded(ex)
    esc(_maybe_threaded(ex))
end

include("array.jl")
include("multi.jl")
include("async.jl")
include("experimental/experimental.jl")

JACC.synchronize(::ThreadsBackend) = nothing

JACC.default_stream(::Type{ThreadsBackend}) = nothing

function JACC.parallel_for(::ThreadsBackend, N::Integer, f::Callable, x...)
    @maybe_threaded for i in 1:N
        f(i, x...)
    end
end

function JACC.parallel_for(
        ::LaunchSpec{ThreadsBackend}, N::Integer, f::Callable, x...)
    JACC.parallel_for(ThreadsBackend(), N, f, x...)
end

function JACC.parallel_for(
        ::ThreadsBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    @maybe_threaded for ij in CartesianIndices((M, N))
        f(ij[1], ij[2], x...)
    end
end

function JACC.parallel_for(
        ::LaunchSpec{ThreadsBackend}, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    JACC.parallel_for(ThreadsBackend(), (M, N), f, x...)
end

function JACC.parallel_for(
        ::ThreadsBackend, (L, M, N)::NTuple{3, Integer}, f::Callable, x...)
    @maybe_threaded for ijk in CartesianIndices((L, M, N))
        f(ijk[1], ijk[2], ijk[3], x...)
    end
end

function JACC.parallel_for(
        ::LaunchSpec{ThreadsBackend}, (L, M, N)::NTuple{3, Integer}, f::Callable, x...)
    JACC.parallel_for(ThreadsBackend(), (L, M, N), f, x...)
end

mutable struct ThreadsReduceWorkspace{T} <: JACC.ReduceWorkspace
    tmp::Vector{T}
    ret::Vector{T}
end

function JACC.reduce_workspace(::ThreadsBackend, init::T) where {T}
    if Threads.nthreads() == 1
        ThreadsReduceWorkspace{T}([], [init])
    else
        # ThreadsReduceWorkspace{T}(fill(init, Threads.nthreads()), [init])
        ThreadsReduceWorkspace{T}(Vector{T}(undef, Threads.nthreads()), [init])
    end
end

JACC.get_result(wk::ThreadsReduceWorkspace) = wk.ret[]

@inline function _serial_reduce!(reducer::JACC.ParallelReduce{ThreadsBackend},
        N::Integer, f::Callable, x...)
    wk = reducer.workspace
    op = reducer.op
    tmp = reducer.init
    for i in 1:N
        tmp = op(tmp, f(i, x...))
    end
    wk.ret[] = tmp
    return nothing
end

@inline function _chunk_reduce!(reducer::JACC.ParallelReduce{ThreadsBackend},
        N::Integer, f::Callable, x...)
    wk = reducer.workspace
    op = reducer.op
    chunks = OhMyThreads.chunks(1:N, n = Threads.nthreads())
    nchunks = length(chunks)
    Threads.@threads :static for n in 1:nchunks
        @inbounds begin
            tp = reducer.init
            for i in chunks[n]
                tp = op(tp, f(i, x...))
            end
            wk.tmp[n] = tp
        end
    end
    wk.ret[] = reduce(op, wk.tmp[1:nchunks])
end

@inline function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{ThreadsBackend},
        N::Integer, f::Callable, x...)
    if Threads.nthreads() == 1
        _serial_reduce!(reducer, N, f, x...)
    else
        _chunk_reduce!(reducer, N, f, x...)
    end
end

function JACC.parallel_reduce(
        ::ThreadsBackend, N::Integer, op, f::Callable, x...; init)
    reducer = JACC.ParallelReduce{ThreadsBackend, typeof(init)}(;
        dims = N, op = op, init = init)
    reducer(f, x...)
    return JACC.get_result(reducer)
end

function JACC.parallel_reduce(
        ::LaunchSpec{ThreadsBackend}, N::Integer, op, f::Callable, x...; init)
    reducer = JACC.ParallelReduce{ThreadsBackend, typeof(init)}(;
        dims = N, op = op, init = init)
    reducer(f, x...)
    return reducer.workspace.ret
end

@inline function _serial_reduce!(reducer::JACC.ParallelReduce{ThreadsBackend},
        (M, N)::NTuple{2, Integer}, f::Callable, x...)
    wk = reducer.workspace
    op = reducer.op
    tmp = reducer.init
    for j in 1:N
        for i in 1:M
            tmp = op(tmp, f(i, j, x...))
        end
    end
    wk.ret[] = tmp
    return nothing
end

@inline function _chunk_reduce!(reducer::JACC.ParallelReduce{ThreadsBackend},
        (M, N)::NTuple{2, Integer}, f::Callable, x...)
    wk = reducer.workspace
    op = reducer.op
    ids = CartesianIndices((1:M, 1:N))
    chunks = OhMyThreads.chunks(ids, n = Threads.nthreads())
    nchunks = length(chunks)
    Threads.@threads :static for n in 1:nchunks
        @inbounds begin
            tp = reducer.init
            for ij in chunks[n]
                tp = op(tp, f(ij[1], ij[2], x...))
            end
            wk.tmp[n] = tp
        end
    end
    wk.ret[] = reduce(op, wk.tmp[1:nchunks])
end

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{ThreadsBackend},
        (M, N)::NTuple{2, Integer}, f::Callable, x...)
    if Threads.nthreads() == 1
        _serial_reduce!(reducer, (M, N), f, x...)
    else
        _chunk_reduce!(reducer, (M, N), f, x...)
    end
end

function JACC.parallel_reduce(
        ::ThreadsBackend, (M, N)::NTuple{2, Integer}, op, f::Callable, x...; init)
    reducer = JACC.ParallelReduce{ThreadsBackend, typeof(init)}(;
        dims = (M, N), op = op, init = init)
    reducer(f, x...)
    return JACC.get_result(reducer)
end

function JACC.parallel_reduce(
        ::LaunchSpec{ThreadsBackend}, (M, N)::NTuple{2, Integer}, op, f::Callable, x...; init)
    reducer = JACC.ParallelReduce{ThreadsBackend, typeof(init)}(;
        dims = (M, N), op = op, init = init)
    reducer(f, x...)
    return reducer.workspace.ret
end

# TODO: Might need to implement locally
# _barrier::Union{Nothing, OhMyThreads.Tools.SimpleBarrier} = nothing
const _BARRIER = Ref(OhMyThreads.Tools.SimpleBarrier(0))

# JACC.sync_workgroup(::ThreadsBackend) = wait(_barrier)
JACC.sync_workgroup(::ThreadsBackend) = wait(_BARRIER[])

JACC.array_type(::ThreadsBackend) = Base.Array

JACC.array(::ThreadsBackend, x::Base.Array) = x

JACC.shared(::ThreadsBackend, x::AbstractArray) = x

function __init__()
    # global _barrier = OhMyThreads.Tools.SimpleBarrier(Threads.nthreads())
    _BARRIER[] = OhMyThreads.Tools.SimpleBarrier(Threads.nthreads())
end

end
