function default_eltype()
	return Float64
end

function zeros(T::Type, dims...)
	return fill!(similar(JACC.Array{T}, dims...), zero(T))
end

function ones(T::Type, dims...)
	return fill!(similar(JACC.Array{T}, dims...), one(T))
end

function zeros(dims...)
	return zeros(default_eltype(), dims...)
end

function ones(dims...)
	return ones(default_eltype(), dims...)
end
