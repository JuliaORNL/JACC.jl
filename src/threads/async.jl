module Async

import Base: Callable
using JACC
using JACC.ThreadsImpl: ThreadsBackend

function JACC.Async.zeros(::ThreadsBackend, T, id, dims...)
    JACC.zeros(ThreadsBackend(), T, dims...)
end

function JACC.Async.ones(::ThreadsBackend, T, id, dims...)
    JACC.ones(ThreadsBackend(), T, dims...)
end

function JACC.Async.fill(::ThreadsBackend, id, value, dims...)
    JACC.fill(ThreadsBackend(), value, dims...)
end

function JACC.Async.synchronize(::ThreadsBackend, id = 0)
end

function JACC.Async.array(::ThreadsBackend, id::Integer, x::AbstractArray)
    return JACC.array(x)
end

function JACC.Async.parallel_for(
        ::ThreadsBackend, id::Integer, dims::JACC.IDims, f::Callable, x...)
    JACC.parallel_for(f, ThreadsBackend(), dims, x...)
end

function JACC.Async.parallel_reduce(::ThreadsBackend, id::Integer,
        dims::JACC.IDims, op::Callable, f::Callable, x...; init)
    ret = JACC.parallel_reduce(
        f, ThreadsBackend(), dims, x...; op = op, init = init)
    return [ret]
end

end
