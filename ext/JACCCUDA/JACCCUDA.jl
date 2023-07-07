
module JACCCUDA

using JACC, CUDA

function JACC.parallel_for(N::I, f::F, x...) where {I<:Integer,F<:Function}
  maxPossibleThreads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
  threads = min(N, maxPossibleThreads)
  blocks = ceil(Int, N / threads)
  CUDA.@sync @cuda threads = threads blocks = blocks _parallel_for_cuda(f, x...)
end

function JACC.parallel_for((M, N)::Tuple{I,I}, f::F, x...) where {I<:Integer,F<:Function}
  numThreads = 16
  Mthreads = min(M, numThreads)
  Nthreads = min(N, numThreads)
  Mblocks = ceil(Int, M / Mthreads)
  Nblocks = ceil(Int, N / Nthreads)
  CUDA.@sync @cuda threads = (Mthreads, Nthreads) blocks = (Mblocks, Nblocks) _parallel_for_cuda_MN(f, x...)
end

function JACC.parallel_reduce(N::I, f::F, x...) where {I<:Integer,F<:Function}
  numThreads = 512
  threads = min(N, numThreads)
  ret = CUDA.zeros(1)
  CUDA.@sync @cuda threads = threads blocks = 1 shmem = 512 * sizeof(Float64) parallel_reduce_cuda(N, ret, f, x...)
  return ret[1]
end


function JACC.parallel_reduce((M, N)::Tuple{I,I}, f::F, x...) where {I<:Integer,F<:Function}
  numThreads = 16
  Mthreads = min(M, numThreads)
  Nthreads = min(N, numThreads)
  ret = CUDA.zeros(1)
  CUDA.@sync @cuda threads = (Mthreads, Nthreads) blocks = 1 shmem = 16 * 16 * sizeof(Float64) parallel_reduce_cuda_MN((M, N), ret, f, x...)
  return ret[1]
end

function _parallel_for_cuda(f, x...)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  f(i, x...)
  return nothing
end

function _parallel_for_cuda_MN(f, x...)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  f(i, j, x...)
  return nothing
end

function _parallel_reduce_cuda(N, ret, f, x...)
  shared_mem = @cuDynamicSharedMem(Float64, 512)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  ii = i
  tmp::Float64 = 0.0
  if N > 512
    while ii < N
      tmp += f(ii, x...)
      ii += 512
    end
  else
    tmp = f(i, x...)
  end
  shared_mem[i] = tmp
  sync_threads()
  if (i <= 256)
    shared_mem[i] += shared_mem[i+256]
  end
  sync_threads()
  if (i <= 128)
    shared_mem[i] += shared_mem[i+128]
  end
  sync_threads()
  if (i <= 64)
    shared_mem[i] += shared_mem[i+64]
  end
  sync_threads()
  if (i <= 32)
    shared_mem[i] += shared_mem[i+32]
  end
  sync_threads()
  if (i <= 16)
    shared_mem[i] += shared_mem[i+16]
  end
  sync_threads()
  if (i <= 8)
    shared_mem[i] += shared_mem[i+8]
  end
  sync_threads()
  if (i <= 4)
    shared_mem[i] += shared_mem[i+4]
  end
  sync_threads()
  if (i <= 2)
    shared_mem[i] += shared_mem[i+2]
  end
  sync_threads()
  if (i == 1)
    shared_mem[i] += shared_mem[i+1]
    ret[1] = shared_mem[i]
  end
  return nothing
end


function parallel_reduce_cuda_MN((M, N), ret, f, x...)
  shared_mem = @cuDynamicSharedMem(Float64, 16, 16)

  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  ii = i
  jj = j

  tmp::Float64 = 0.0

  if M > 16 && N > 16
    while ii <= M
      jj = (blockIdx().y - 1) * blockDim().y + threadIdx().y
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
  const JACC.Array = CUDA.CuArray{T,N} where {T,N}
end

end # module JACCCUDA
