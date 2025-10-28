module Async

import Base: Callable
using JACC

function zeros(::Type{T}, id, dims...) where {T}
    return zeros(JACC.default_backend(), T, id, dims...)
end

function ones(::Type{T}, id, dims...) where {T}
    return ones(JACC.default_backend(), T, id, dims...)
end

zeros(id::Integer, dims::Integer...) = zeros(default_float(), id, dims...)
ones(id::Integer, dims::Integer...) = ones(default_float(), id, dims...)
fill(id, value, dims...) = fill(JACC.default_backend(), id, value, dims...)

function ndev()
    return ndev(JACC.default_backend())
end

function synchronize()
    return synchronize(JACC.default_backend())
end

function synchronize(id::Integer)
    return synchronize(JACC.default_backend(), id)
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

function parallel_for(id::Integer, dims::JACC.IDims, f::Callable, x...)
    return parallel_for(JACC.default_backend(), id, dims, f, x...)
end

function parallel_for(f::Callable, id::Integer, dims::JACC.IDims, x...)
    return parallel_for(id, dims, f, x...)
end

function parallel_reduce(
        id::Integer, dims::JACC.IDims, op::Callable, f::Callable, x...; init)
    return parallel_reduce(
        JACC.default_backend(), id, dims, op, f, x...; init = init)
end

function parallel_reduce(id::Integer, dims::JACC.IDims, f::Callable, x...)
    return parallel_reduce(id, dims, +, f, x...; init = JACC.default_init(+))
end

function parallel_reduce(
        f::Callable, id::Integer, dims::JACC.IDims, op::Callable, x...; init)
    return parallel_reduce(id, dims, op, f, x...; init = init)
end

function parallel_reduce(f::Callable, id::Integer, dims::JACC.IDims, x...)
    return parallel_reduce(id, dims, f, x...)
end

function parallel_reduce(
        op::Callable, a::AbstractArray; init = JACC.default_init(eltype(a), op))
    return parallel_reduce(
        JACC.array_size(a), op, JACC._elem_access(a), a; init = init)
end

parallel_reduce(a::AbstractArray; kw...) = parallel_reduce(+, a; kw...)

end # module Async
