module Async

import Base: Callable
using JACC
using JACC.ThreadsImpl: ThreadsBackend

function JACC.Async.array(::ThreadsBackend, queue_id::Integer,
        x::Base.Array{T, N}) where {T, N}
    return JACC.array(x)
end

function JACC.Async.copy(
        ::ThreadsBackend, queue_id_dest::Integer, x::Base.Array{T, N},
        queue_id_orig::Integer, y::Base.Array{T, N}) where {T, N}
    copyto!(x, y)
end

function JACC.Async.parallel_for(
        ::ThreadsBackend, queue_id::Integer, N::Integer, f::Callable,
        x...)
    JACC.parallel_for(N, f, x...)
end

function JACC.Async.parallel_for(
        ::ThreadsBackend, queue_id::Integer, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    JACC.parallel_for((M, N), f, x...)
end

function JACC.Async.parallel_reduce(
        ::ThreadsBackend, queue_id::Integer, N::Integer, f::Callable, x...)
    return JACC.parallel_reduce(N, f, x...)
end

function JACC.Async.parallel_reduce(
        ::ThreadsBackend, queue_id::Integer, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    return JACC.parallel_reduce((M, N), f, x...)
end

function JACC.Async.synchronize(::ThreadsBackend)
end

end
