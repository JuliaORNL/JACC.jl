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

function parallel_for(::JACCArrayType{<:Array}, N::Integer, f::Function, x...)
	@maybe_threaded for i in 1:N
		f(i, x...)
	end
end

function parallel_for(::JACCArrayType{<:Array}, (M, N)::Tuple{Integer, Integer}, f::Function, x...)
	@maybe_threaded for j in 1:N
		for i in 1:M
			f(i, j, x...)
		end
	end
end

function parallel_reduce(::JACCArrayType{<:Array}, N::Integer, f::Function, x...)
	tmp = zeros(Threads.nthreads())
	ret = zeros(1)
	@maybe_threaded for i in 1:N
		tmp[Threads.threadid()] = tmp[Threads.threadid()] .+ f(i, x...)
	end
	for i in 1:Threads.nthreads()
		ret = ret .+ tmp[i]
	end
	return ret
end

function parallel_reduce(::JACCArrayType{<:Array}, (M, N)::Tuple{Integer, Integer}, f::Function, x...)
	tmp = zeros(Threads.nthreads())
	ret = zeros(1)
	@maybe_threaded for j in 1:N
		for i in 1:M
			tmp[Threads.threadid()] = tmp[Threads.threadid()] .+ f(i, j, x...)
		end
	end
	for i in 1:Threads.nthreads()
		ret = ret .+ tmp[i]
	end
	return ret
end