module JACC

# module to set back end preferences
include("JACCPreferences.jl")
include("helper.jl")

export parallel_for, parallel_reduce, ThreadsBackend, print_default_backend

struct ThreadsBackend end

export default_backend

global default_backend = ThreadsBackend()

# default backend API
function parallel_for(N::I, f::F, x...) where {I<:Integer,F<:Function}
  parallel_for(default_backend, N, f, x...)
end

function parallel_for((M, N)::Tuple{I,I}, f::F, x...) where {I<:Integer,F<:Function}
  parallel_for(default_backend, N, f, x...)
end

function parallel_reduce(N::I, f::F, x...) where {I<:Integer,F<:Function}
  parallel_reduce(default_backend, N, f, x...)
end

function parallel_reduce((M, N)::Tuple{I,I}, f::F, x...) where {I<:Integer,F<:Function}
  parallel_reduce(default_backend, (M, N), f, x...)
end

function parallel_for(::ThreadsBackend, N::I, f::F, x...) where {I<:Integer,F<:Function}
  @maybe_threaded for i in 1:N
    f(i, x...)
  end
end

function parallel_for(::ThreadsBackend, (M, N)::Tuple{I,I}, f::F, x...) where {I<:Integer,F<:Function}
  @maybe_threaded for j in 1:N
    for i in 1:M
      f(i, j, x...)
    end
  end
end

function parallel_reduce(::ThreadsBackend, N::I, f::F, x...) where {I<:Integer,F<:Function}
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

function parallel_reduce(::ThreadsBackend, (M, N)::Tuple{I,I}, f::F, x...) where {I<:Integer,F<:Function}
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

function __init__()
  if JACCPreferences.backend == "threads"
      const JACC.default_backend = ThreadsBackend()
      @info "Set default backend to $(JACC.default_backend)"
  end
end

function print_default_backend()
  println("Default backend is $default_backend")
end


end # module JACC
