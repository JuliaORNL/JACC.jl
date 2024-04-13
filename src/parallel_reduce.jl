function parallel_reduce(::Union{<:Integer, <:Tuple}, ::Function, a::JACCArgsList{<:AbstractArray})
  error("parallel_reduce not defined for the arraytype of $(arraytype(a))")
end

function parallel_reduce(N::Integer, f::Function, x::JACCArgsList{<:Array})
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

function parallel_reduce((M, N)::Tuple{Integer,Integer}, f::Function,  x::JACCArgsList{<:Array})
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