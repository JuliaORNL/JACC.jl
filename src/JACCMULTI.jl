module multi

using JACC
import JACC: ThreadsBackend

function Array(::ThreadsBackend, x::Base.Array{T,N}) where {T,N}
  return x
end

function copy(::ThreadsBackend, x::Vector{Any}, y::Vector{Any})
end

function parallel_for(::ThreadsBackend, N::I, f::F, x...) where {I <: Integer, F <: Function}
end

function parallel_for(::ThreadsBackend, (M, N)::Tuple{I,I}, f::F, x...) where {I <: Integer, F <: Function}
end

function parallel_reduce(::ThreadsBackend, N::I, f::F, x...) where {I <: Integer, F <: Function}
end

function parallel_reduce(::ThreadsBackend, (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
end

function Array(x::Base.Array{T,N}) where {T,N}
    return Array(default_backend(), x)
end

function copy(x::Vector{Any}, y::Vector{Any})
    return copy(default_backend(), x, y)
end

function parallel_for(N::I, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_for(default_backend(), N, f, x...)
end

function parallel_for((M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_for(default_backend(), (M, N), f, x...)
end

function parallel_reduce(N::I, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_reduce(default_backend(), N, f, x...)
end

function parallel_reduce((M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_reduce(default_backend(), (M, N), f, x...)
end

end # module multi
