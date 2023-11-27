
module JACCONEAPI

using JACC, oneAPI

function JACC.parallel_for(N::I, f::F, x...) where {I<:Integer,F<:Function}
  #maxPossibleItems = oneAPI.oneL0.compute_properties(device().maxTotalGroupSize)
  maxPossibleItems = 256
  items = min(N, maxPossibleItems)
  groups = ceil(Int, N / items)
  oneAPI.@sync @oneapi items = items groups = groups _parallel_for_oneapi(f, x...)
end

function JACC.parallel_for((M, N)::Tuple{I,I}, f::F, x...) where {I<:Integer,F<:Function}
  maxPossibleItems = 16  
  Mitems = min(M, maxPossibleItems)
  Nitems = min(N, maxPossibleItems)
  Mgroups = ceil(Int, M / Mitems)
  Ngroups = ceil(Int, N / Nitems)
  oneAPI.@sync @oneapi items = (Mitems, Nitems) groups = (Mgroups, Ngroups) _parallel_for_openapi_MN(f, x...)
end

function JACC.parallel_reduce(N::I, f::F, x...) where {I<:Integer,F<:Function}
  numItems = 256
  items = min(N, numItems)
  ret = oneAPI.zeros(1)
  oneAPI.@sync @oneapi items = items groups = 1 _parallel_reduce_oneapi(N, ret, f, x...)
  return ret[1]
end


function JACC.parallel_reduce((M, N)::Tuple{I,I}, f::F, x...) where {I<:Integer,F<:Function}
  numItems = 16
  Mitems = min(M, numItems)
  Nitems = min(N, numItems)
  ret = oneAPI.zeros(1)
  oneAPI.@sync @oneapi items = (Mitems, Nitems) groups = 1 _parallel_reduce_oneapi_MN((M, N), ret, f, x...)
  return ret[1]
end

function _parallel_for_oneapi(f, x...)
  i = get_global_id()
  f(i, x...)
  return nothing
end

function _parallel_for_oneapi_MN(f, x...)
  i = get_global_id(0)
  j = get_global_id(1)
  f(i, j, x...)
  return nothing
end

function _parallel_reduce_cuda(N, ret, f, x...)
  shared_mem = oneLocalArray(Float64, 256)
  i = get_global_id()
  ii = i
  tmp::Float64 = 0.0
  if N > 256
    while ii <= N
      tmp += f(ii, x...)
      ii += 256
    end
  else
    tmp = f(i, x...)
  end
  shared_mem[i] = tmp
  barrier() 
  if (i <= 128)
    shared_mem[i] += shared_mem[i+128]
  end
  barrier() 
  if (i <= 64)
    shared_mem[i] += shared_mem[i+64]
  end
  barrier() 
  if (i <= 32)
    shared_mem[i] += shared_mem[i+32]
  end
  barrier() 
  if (i <= 16)
    shared_mem[i] += shared_mem[i+16]
  end
  barrier() 
  if (i <= 8)
    shared_mem[i] += shared_mem[i+8]
  end
  barrier() 
  if (i <= 4)
    shared_mem[i] += shared_mem[i+4]
  end
  barrier() 
  if (i <= 2)
    shared_mem[i] += shared_mem[i+2]
  end
  barrier() 
  if (i == 1)
    shared_mem[i] += shared_mem[i+1]
    ret[1] = shared_mem[i]
  end
  return nothing
end


function _parallel_reduce_cuda_MN((M, N), ret, f, x...)
  shared_mem = oneLocalArray(Float64, 16, 16)

  i = get_global_id(0)
  j = get_global_id(1)
  ii = i
  jj = j

  tmp::Float64 = 0.0

  if M > 16 && N > 16
    while ii <= M
      jj = get_global_id(1)
      while jj <= N
        tmp += @inbounds f(ii, jj, x...)
        jj += 16
      end
      ii += 16
    end
  elseif M > 16
    while ii <= N
      tmp += @inbounds f(ii, jj, x...)
      ii += 16
    end
  elseif N > 16
    while jj <= N
      tmp += @inbounds f(ii, jj, x...)
      jj += 16
    end
  else
    tmp = f(i, j, x...)
  end
  shared_mem[i, j] = tmp
  sync_threads()
  if (i <= 8 && j <= 8)
    shared_mem[i, j] += shared_mem[i+8, j+8]
    shared_mem[i, j] += shared_mem[i, j+8]
    shared_mem[i, j] += shared_mem[i+8, j]
  end
  sync_threads()
  if (i <= 4 && j <= 4)
    shared_mem[i, j] += shared_mem[i+4, j+4]
    shared_mem[i, j] += shared_mem[i, j+4]
    shared_mem[i, j] += shared_mem[i+4, j]
  end
  sync_threads()
  if (i <= 2 && j <= 2)
    shared_mem[i, j] += shared_mem[i+2, j+2]
    shared_mem[i, j] += shared_mem[i, j+2]
    shared_mem[i, j] += shared_mem[i+2, j]
  end
  sync_threads()
  if (i == 1 && j == 1)
    shared_mem[i, j] += shared_mem[i+1, j+1]
    shared_mem[i, j] += shared_mem[i, j+1]
    shared_mem[i, j] += shared_mem[i+1, j]
    ret[1] += shared_mem[i, j]
  end
  return nothing
end


function __init__()
  const JACC.Array = oneAPI.oneArray{T,N} where {T,N}
end

end # module JACCONEAPI
