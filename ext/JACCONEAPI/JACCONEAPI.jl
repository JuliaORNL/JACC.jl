
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
  oneAPI.@sync @oneapi items = (Mitems, Nitems) groups = (Mgroups, Ngroups) _parallel_for_oneapi_MN(f, x...)
end

function JACC.parallel_reduce(N::I, f::F, x...) where {I<:Integer,F<:Function}
  numItems = 256
  items = min(N, numItems)
  groups = ceil(Int, N/items)
  ret = oneAPI.zeros(Float32, groups)
  rret = oneAPI.zeros(Float32, 1)
  oneAPI.@sync @oneapi items = items groups = groups _parallel_reduce_oneapi(N, ret, f, x...)
  oneAPI.@sync @oneapi items = items groups = 1 reduce_kernel_oneapi(N, ret, rret)
  return rret
end

function JACC.parallel_reduce((M, N)::Tuple{I,I}, f::F, x...) where {I<:Integer,F<:Function}
  numItems = 16
  Mitems = min(M, numItems)
  Nitems = min(N, numItems)
  ret = oneAPI.zeros(Float32, 1)
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

function _parallel_reduce_oneapi(N, ret, f, x...)
  #shared_mem = oneLocalArray(Float32, 256)
  shared_mem = oneLocalArray(Float64, 256)
  i = get_global_id(0)
  ti = get_local_id(0)
  #tmp::Float32 = 0.0
  tmp::Float64 = 0.0
  shared_mem[ti] = 0.0
  if i <= N
    tmp = @inbounds f(i, x...)
    shared_mem[ti] = tmp
    barrier() 
  end
  if (ti <= 128)
    shared_mem[ti] += shared_mem[ti+128]
  end
  barrier() 
  if (ti <= 64)
    shared_mem[ti] += shared_mem[ti+64]
  end
  barrier() 
  if (ti <= 32)
    shared_mem[ti] += shared_mem[ti+32]
  end
  barrier() 
  if (ti <= 16)
    shared_mem[ti] += shared_mem[ti+16]
  end
  barrier() 
  if (ti <= 8)
    shared_mem[ti] += shared_mem[ti+8]
  end
  barrier() 
  if (ti <= 4)
    shared_mem[ti] += shared_mem[ti+4]
  end
  barrier() 
  if (ti <= 2)
    shared_mem[ti] += shared_mem[ti+2]
  end
  barrier() 
  if (ti == 1)
    shared_mem[ti] += shared_mem[ti+1]
    ret[get_group_id(0)] = shared_mem[ti]
  end
  barrier() 
  return nothing
end

function reduce_kernel_oneapi(N, red, ret)
  #shared_mem = oneLocalArray(Float32, 256)
  shared_mem = oneLocalArray(Float64, 256)
  i = get_global_id()
  ii = i
  #tmp::Float32 = 0.0
  tmp::Float64 = 0.0
  if N > 256
    while ii <= N
      tmp += @inbounds red[ii]
      ii += 256
    end
  else
    tmp = @inbounds red[i]
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
    ret[1] = shared_mem[1]
  end
  return nothing
end

function _parallel_reduce_oneapi_MN((M, N), ret, f, x...)
  shared_mem = oneLocalArray(Float32, 16 * 16)

  i = get_global_id(0)
  j = get_global_id(1)
  ii = i
  jj = j

  tmp::Float32 = 0.0

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
  else M <= 16 && N <= 16
    if i <= M && j <= N
      tmp = f(i, j, x...)
    end
  end
  shared_mem[(i-1)*16+j] = tmp
  barrier() 
  if (i <= 8 && j <= 8 && i+8 <= M && j+8 <= N)
    shared_mem[((i-1)*16)+j] = shared_mem[((i-1)*16)+j] + shared_mem[((i+7)*16)+(j+8)]
    shared_mem[((i-1)*16)+j] = shared_mem[((i-1)*16)+j] + shared_mem[((i-1)*16)+(j+8)]
    shared_mem[((i-1)*16)+j] = shared_mem[((i-1)*16)+j] + shared_mem[((i+7)*16)+j]
  end
  barrier() 
  if (i <= 4 && j <= 4 && i+4 <= M && j+4 <= N)
    shared_mem[((i-1)*16)+j] = shared_mem[((i-1)*16)+j] + shared_mem[((i+3)*16)+(j+4)]
    shared_mem[((i-1)*16)+j] = shared_mem[((i-1)*16)+j] + shared_mem[((i-1)*16)+(j+4)]
    shared_mem[((i-1)*16)+j] = shared_mem[((i-1)*16)+j] + shared_mem[((i+3)*16)+j]
  end
  barrier() 
  if (i <= 2 && j <= 2 && i+2 <= M && j+2 <= N)
    shared_mem[((i-1)*16)+j] = shared_mem[((i-1)*16)+j] + shared_mem[((i+1)*16)+(j+2)]
    shared_mem[((i-1)*16)+j] = shared_mem[((i-1)*16)+j] + shared_mem[((i-1)*16)+(j+2)]
    shared_mem[((i-1)*16)+j] = shared_mem[((i-1)*16)+j] + shared_mem[((i+1)*16)+j]
  end
  barrier() 
  if (i == 1 && j == 1 && i+1 <= M && j+1 <= N)
    shared_mem[((i-1)*16)+j] = shared_mem[((i-1)*16)+j] + shared_mem[i*16+(j+1)]
    shared_mem[((i-1)*16)+j] = shared_mem[((i-1)*16)+j] + shared_mem[((i-1)*16)+(j+1)]
    shared_mem[((i-1)*16)+j] = shared_mem[((i-1)*16)+j] + shared_mem[i*16+j]
    ret[1] = shared_mem[((i-1)*16)+j]
  end
  return nothing
end


function __init__()
  const JACC.Array = oneAPI.oneArray{T,N} where {T,N}
end

end # module JACCONEAPI
