module Multi

import Base.Callable
import JACC
import JACC.ThreadsImpl: ThreadsBackend

function JACC.Multi.ndev(::ThreadsBackend)
end

function JACC.Multi.device_id(::ThreadsBackend, x)
    return 0
end

function JACC.Multi.array(::ThreadsBackend, x::Base.Array; ghost_dims)
    return x
end

JACC.Multi.multi_array_type(::ThreadsBackend) = Base.Array

function JACC.Multi.ghost_shift(::ThreadsBackend, idx, arr)
    return idx
end

function JACC.Multi.copy(::ThreadsBackend, x::Vector{Any}, y::Vector{Any})
end

function JACC.Multi.parallel_for(
        ::ThreadsBackend, N::Integer, f::Callable, x...)
    return JACC.parallel_for(ThreadsBackend(), N, (p...) -> f(1, p...), x...)
end

function JACC.Multi.parallel_for(
        ::ThreadsBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    return JACC.parallel_for(ThreadsBackend(), (M,N), (p...) -> f(1, p...), x...)
end

function JACC.Multi.parallel_reduce(
        ::ThreadsBackend, N::Integer, f::Callable, x...)
    return JACC.parallel_reduce(ThreadsBackend(), N, +, (p...) -> f(1, p...), x...; init = 0.0)
end

function JACC.Multi.parallel_reduce(
        ::ThreadsBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    return JACC.parallel_reduce(ThreadsBackend(), (M,N), +, (p...) -> f(1, p...), x...; init = 0.0)
end

end
