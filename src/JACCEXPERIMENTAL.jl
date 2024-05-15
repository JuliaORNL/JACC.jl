module experimental

using JACC

function shared(x::Base.Array{T,N}) where {T,N}
  return x
end

end # module experimental
