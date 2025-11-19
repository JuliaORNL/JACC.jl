module Multi

import Base.Callable
import JACC
import JACC.ThreadsImpl: ThreadsBackend

function JACC.Multi.ndev(::ThreadsBackend)
    return 1
end

function JACC.Multi.device_id(::ThreadsBackend, x)
    return 1
end

function JACC.Multi.part_length(::ThreadsBackend, x)
    return size(x)[end]
end

function JACC.Multi.array(::ThreadsBackend, x::Base.Array; ghost_dims)
    return x
end

JACC.Multi.multi_array_type(::ThreadsBackend) = Base.Array

function JACC.Multi.ghost_shift(::ThreadsBackend, idx, arr)
    return idx
end

function JACC.Multi.sync_ghost_elems!(::ThreadsBackend, arr)
end

function JACC.Multi.copy!(::ThreadsBackend, x, y)
    copy!(x, y)
end

function JACC.Multi.parallel_for(
        ::ThreadsBackend, N::Integer, f::Callable, x...)
    return JACC.parallel_for(f, ThreadsBackend(), N, x...)
end

function JACC.Multi.parallel_for(
        ::ThreadsBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    return JACC.parallel_for(f, ThreadsBackend(), (M, N), x...)
end

function JACC.Multi.parallel_reduce(
        ::ThreadsBackend, N::Integer, f::Callable, x...)
    return JACC.parallel_reduce(f, ThreadsBackend(), N, x...; op = +,
        init = 0.0)
end

function JACC.Multi.parallel_reduce(
        ::ThreadsBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    return JACC.parallel_reduce(f, ThreadsBackend(), (M, N), x...; op = +,
        init = 0.0)
end

end
