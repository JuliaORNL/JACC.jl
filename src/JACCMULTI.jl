module multi

using JACC

function ndev()
end

function Array(x::Base.Array{T,N}) where {T,N}
  return x
end

function copy(x::Vector{Any}, y::Vector{Any})
end

function parallel_for(N::I, f::F, x...) where {I <: Integer, F <: Function}
end

function parallel_for((M, N)::Tuple{I,I}, f::F, x...) where {I <: Integer, F <: Function}
end

function parallel_reduce(N::I, f::F, x...) where {I <: Integer, F <: Function}
end

function parallel_reduce((M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
end

end # module multi
