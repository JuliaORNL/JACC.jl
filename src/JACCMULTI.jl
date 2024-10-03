module multi

using JACC

function Array(x::Base.Array{T,N}) where {T,N}
    return Array(JACCPreferences._backend_dispatchable, x)
end
function Array(::Val{:threads}, x::Base.Array{T,N}) where {T,N}
  return x
end

function copy(x::Vector{Any}, y::Vector{Any})
    return copy(JACCPreferences._backend_dispatchable, x, y)
end
function copy(::Val{:threads}, x::Vector{Any}, y::Vector{Any})
end

function parallel_for(N::I, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_for(JACCPreferences._backend_dispatchable, N, f, x...)
end
function parallel_for(::Val{:threads}, N::I, f::F, x...) where {I <: Integer, F <: Function}
end

function parallel_for((M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_for(JACCPreferences._backend_dispatchable, (M, N), f, x...)
end
function parallel_for(::Val{:threads}, (M, N)::Tuple{I,I}, f::F, x...) where {I <: Integer, F <: Function}
end

function parallel_reduce(N::I, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_reduce(JACCPreferences._backend_dispatchable, N, f, x...)
end
function parallel_reduce(::Val{:threads}, N::I, f::F, x...) where {I <: Integer, F <: Function}
end

function parallel_reduce((M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_reduce(JACCPreferences._backend_dispatchable, (M, N), f, x...)
end
function parallel_reduce(::Val{:threads}, (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
end

end # module multi
