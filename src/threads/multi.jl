module Multi

import Base.Callable
using JACC
using JACC.ThreadsImpl: ThreadsBackend

function JACC.Multi.ndev(::ThreadsBackend)
end

function JACC.Multi.array(::ThreadsBackend, x::Base.Array{T, N}) where {T, N}
    return x
end

function JACC.Multi.gArray(::ThreadsBackend, x::Base.Array{T, N}) where {T, N}
end

function JACC.Multi.gid(
        ::ThreadsBackend, dev_id::Integer, i::Integer, ndev::Integer)
end

function JACC.Multi.gswap(::ThreadsBackend, x::Vector{Any})
end

function JACC.Multi.gcopytoarray(
        ::ThreadsBackend, x::Vector{Any}, y::Vector{Any})
end

function JACC.Multi.copytogarray(
        ::ThreadsBackend, x::Vector{Any}, y::Vector{Any})
end

function JACC.Multi.copy(::ThreadsBackend, x::Vector{Any}, y::Vector{Any})
end

function JACC.Multi.parallel_for(
        ::ThreadsBackend, N::Integer, f::Callable, x...)
end

function JACC.Multi.parallel_for(
        ::ThreadsBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
end

function JACC.Multi.parallel_reduce(
        ::ThreadsBackend, N::Integer, f::Callable, x...)
end

function JACC.Multi.parallel_reduce(
        ::ThreadsBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
end

end
