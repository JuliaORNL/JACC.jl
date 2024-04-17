

function JACC.zeros(T, dims...)
	return CUDA.zeros(T, dims...)
end

function JACC.ones(T, dims...)
	return CUDA.ones(T, dims...)
end

function JACC.zeros(dims...)
	return CUDA.zeros(dims...)
end

function JACC.ones(dims...)
	return CUDA.ones(dims...)
end
