module multi

using JACC

function ndev()
end

function Array(x::Base.Array{T,N}) where {T,N}
  return x
end

function gArray(x::Base.Array{T,N}) where {T,N}
end

function gid(dev_id::I, i::I, ndev::I) where{I <: Integer}
end

function gswap(x::Vector{Any})
end

function gcopytoarray(x::Vector{Any}, y::Vector{Any})
end

function copytogarray(x::Vector{Any}, y::Vector{Any})
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
