module Multi

using JACC
import JACC: ThreadsBackend

function ndev(::ThreadsBackend)
end

function Array(::ThreadsBackend, x::Base.Array{T, N}) where {T, N}
    return x
end

function gArray(::ThreadsBackend, x::Base.Array{T, N}) where {T, N}
end

function gid(::ThreadsBackend, dev_id::I, i::I, ndev::I) where {I <: Integer}
end

function gswap(::ThreadsBackend, x::Vector{Any})
end

function gcopytoarray(::ThreadsBackend, x::Vector{Any}, y::Vector{Any})
end

function copytogarray(::ThreadsBackend, x::Vector{Any}, y::Vector{Any})
end

function copy(::ThreadsBackend, x::Vector{Any}, y::Vector{Any})
end

function parallel_for(
        ::ThreadsBackend, N::I, f::F, x...) where {I <: Integer, F <: Function}
end

function parallel_for(::ThreadsBackend, (M, N)::Tuple{I, I}, f::F,
        x...) where {I <: Integer, F <: Function}
end

function parallel_reduce(
        ::ThreadsBackend, N::I, f::F, x...) where {I <: Integer, F <: Function}
end

function parallel_reduce(::ThreadsBackend, (M, N)::Tuple{I, I}, f::F,
        x...) where {I <: Integer, F <: Function}
end

function ndev()
    return ndev(JACC.default_backend())
end

function Array(x::Base.Array{T, N}) where {T, N}
    return Array(JACC.default_backend(), x)
end

function gArray(x::Base.Array{T, N}) where {T, N}
    return gArray(JACC.default_backend(), x)
end

function gid(dev_id::I, i::I, ndev::I) where {I <: Integer}
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

function parallel_for(N::I, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_for(JACC.default_backend(), N, f, x...)
end

function parallel_for(
        (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_for(JACC.default_backend(), (M, N), f, x...)
end

function parallel_reduce(N::I, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_reduce(JACC.default_backend(), N, f, x...)
end

function parallel_reduce(
        (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_reduce(JACC.default_backend(), (M, N), f, x...)
end

end # module Multi
