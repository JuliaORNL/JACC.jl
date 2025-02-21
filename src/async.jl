module Async

import Base: Callable
using JACC

function Array(queue_id::Integer, x::Base.Array{T, N}) where {T, N}
    return Array(JACC.default_backend(), queue_id, x)
end

function copy(
        queue_id_dest::Integer, x::Base.Array{T, N}, queue_id_orig::Integer,
        y::Base.Array{T, N}) where {T, N}
    return copy(JACC.default_backend(), queue_id_dest, x, queue_id_orig, y)
end

function parallel_for(queue_id::Integer, N::Integer, f::Callable, x...)
    return parallel_for(JACC.default_backend(), queue_id, N, f, x...)
end

function parallel_for(
        queue_id::Integer, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    return parallel_for(JACC.default_backend(), queue_id, (M, N), f, x...)
end

function parallel_reduce(queue_id::Integer, N::Integer, f::Callable, x...)
    return parallel_reduce(JACC.default_backend(), queue_id, N, f, x...)
end

function parallel_reduce(
        queue_id::Integer, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    return parallel_reduce(JACC.default_backend(), queue_id, (M, N), f, x...)
end

function synchronize()
    return synchronize(JACC.default_backend())
end

end # module Async
