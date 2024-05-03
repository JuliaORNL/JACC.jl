
function zeros(T, dims...)
		return fill!(similar(arraytype(){T}, dims...), zero(T))
end

function ones(T, dims...)
	return fill!(similar(arraytype(){T}, dims...), one(T))
end

array(T::AbstractArray) = arraytype()(T)

function parallel_for(::JACCArrayType{<:Array}, N::Integer, f::Function, x...)
	@maybe_threaded for i in 1:N
			f(i, x...)
	end
end

function parallel_for(::JACCArrayType{<:Array},
			(M, N)::Tuple{Integer, Integer}, f::Function, x...)
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

function parallel_reduce(::JACCArrayType{<:Array},
			(M, N)::Tuple{Integer, Integer}, f::Function, x...)
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
