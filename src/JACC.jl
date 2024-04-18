# __precompile__(false)
module JACC

# module to set back end preferences 
include("JACCPreferences.jl")
include("helper.jl")

export Array
export parallel_for

global Array
global Tag

struct ThreadsTag end

function parallel_for(::ThreadsTag, N::I, f::F, x...) where {I <: Integer, F <: Function}
	@maybe_threaded for i in 1:N
		f(i, x...)
	end
end

@inline function parallel_for(N::I, f::F, x...) where {I <: Integer, F <: Function}
    parallel_for(Tag(), N, f, x...)
end

function parallel_for(::ThreadsTag, (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
	@maybe_threaded for j in 1:N
		for i in 1:M
			f(i, j, x...)
		end
	end
end

@inline function parallel_for((M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    parallel_for(Tag(), (M, N), f, x...)
end

function parallel_reduce(::ThreadsTag, N::I, f::F, x...) where {I <: Integer, F <: Function}
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

@inline function parallel_reduce(N::I, f::F, x...) where {I <: Integer, F <: Function}
    parallel_reduce(Tag(), N, f, x...)
end

function parallel_reduce(::ThreadsTag, (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
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

@inline function parallel_reduce((M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    parallel_reduce(Tag(), (M, N), f, x...)
end

function __init__()
	const JACC.Array = Base.Array{T, N} where {T, N}
    const JACC.Tag = ThreadsTag
end


end # module JACC
