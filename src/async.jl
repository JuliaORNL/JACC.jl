module Async

import Base: Callable
using JACC

function ndev()
    return ndev(JACC.default_backend())
end

function array(id::Integer, x::Base.Array{T, N}) where {T, N}
    return array(JACC.default_backend(), id, x)
end

function copy(
        #id_dest::Integer, x::Base.Array{T, N}, id_orig::Integer,
        #y::Base.Array{T, N}) where {T, N}
        x...)
    #return copy(JACC.default_backend(), id_dest, x, id_orig, y)
    return copy(JACC.default_backend(), x...)
end

function parallel_for(id::Integer, N::Integer, f::Callable, x...)
    return parallel_for(JACC.default_backend(), id, N, f, x...)
end

function parallel_for(
        id::Integer, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    return parallel_for(JACC.default_backend(), id, (M, N), f, x...)
end

function parallel_reduce(id::Integer, N::Integer, f::Callable, x...)
    return parallel_reduce(JACC.default_backend(), id, N, f, x...)
end

function parallel_reduce(
        id::Integer, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    return parallel_reduce(JACC.default_backend(), id, (M, N), f, x...)
end

function synchronize()
    return synchronize(JACC.default_backend())
end

function synchronize(id::Integer)
    return synchronize(JACC.default_backend(), id)
end

end # module Async
