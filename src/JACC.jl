module JACC

# module to set back end preferences 
include("JACCPreferences.jl")

export Array
export parallel_for

global Array

function parallel_for(N::I, f::F, x...) where {I<:Integer,F<:Function}
  Threads.@threads :static for i in 1:N
    f(i, x...)
  end
end

function parallel_for((M, N)::Tuple{I,I}, f::F, x...) where {I<:Integer,F<:Function}
  Threads.@threads :static for j in 1:N
    for i in 1:M
      f(i, j, x...)
    end
  end
end

function parallel_reduce(N::I, f::F, x...) where {I<:Integer,F<:Function}
  tmp = zeros(Threads.nthreads())
  ret = zeros(1)
  Threads.@threads :static for i in 1:N
    tmp[Threads.threadid()] = tmp[Threads.threadid()] .+ f(i, x...)
  end
  for i in 1:Threads.nthreads()
    ret = ret .+ tmp[i]
  end
  return ret
end

function parallel_reduce((M, N)::Tuple{I,I}, f::F, x...) where {I<:Integer,F<:Function}
  tmp = zeros(Threads.nthreads())
  ret = zeros(1)
  Threads.@threads :static for j in 1:N
    for i in 1:M
      tmp[Threads.threadid()] = tmp[Threads.threadid()] .+ f(i, j, x...)
    end
  end
  for i in 1:Threads.nthreads()
    ret = ret .+ tmp[i]
  end
  return ret
end

function __init__()
  @info("Using JACC backend: $(JACCPreferences.backend)")

  if JACCPreferences.backend == "threads"
    const JACC.Array = Base.Array{T,N} where {T,N}
  end
end


end # module JACC
