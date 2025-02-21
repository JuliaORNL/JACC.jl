module Multi

import Base: Callable
using JACC

function ndev()
    return ndev(JACC.default_backend())
end

function array(x::Base.Array{T, N}) where {T, N}
    return array(JACC.default_backend(), x)
end

function gArray(x::Base.Array{T, N}) where {T, N}
    return gArray(JACC.default_backend(), x)
end

function gid(dev_id::Integer, i::Integer, ndev::Integer)
    return gid(JACC.default_backend(), dev_id, i, ndev)
end

function gswap(x::Vector{Any})
    return gswap(JACC.default_backend(), x)
end

function gcopytoarray(x::Vector{Any}, y::Vector{Any})
    return gcopytoarray(JACC.default_backend(), x, y)
end

function copytogarray(x::Vector{Any}, y::Vector{Any})
    return copytogarray(JACC.default_backend(), x, y)
end

function copy(x::Vector{Any}, y::Vector{Any})
    return copy(JACC.default_backend(), x, y)
end

function parallel_for(N::Integer, f::Callable, x...)
    return parallel_for(JACC.default_backend(), N, f, x...)
end

function parallel_for((M, N)::NTuple{2, Integer}, f::Callable, x...)
    return parallel_for(JACC.default_backend(), (M, N), f, x...)
end

function parallel_reduce(N::Integer, f::Callable, x...)
    return parallel_reduce(JACC.default_backend(), N, f, x...)
end

function parallel_reduce((M, N)::NTuple{2, Integer}, f::Callable, x...)
    return parallel_reduce(JACC.default_backend(), (M, N), f, x...)
end

end # module Multi
