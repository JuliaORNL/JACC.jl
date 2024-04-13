using JACC: JACC, JACCArgsList
using CUDA

function JACC.parallel_for(N::Integer, f::Function, x::JACCArgsList{<:CuArray})
  maxPossibleThreads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
  threads = min(N, maxPossibleThreads)
  blocks = ceil(Int, N / threads)
  CUDA.@sync @cuda threads = threads blocks = blocks _parallel_for_cuda(f, x...)
end

function JACC.parallel_for((M, N)::Tuple{Integer,Integer}, f::Function, x::JACCArgsList{<:CuArray})
  numThreads = 16
  Mthreads = min(M, numThreads)
  Nthreads = min(N, numThreads)
  Mblocks = ceil(Int, M / Mthreads)
  Nblocks = ceil(Int, N / Nthreads)
  CUDA.@sync @cuda threads = (Mthreads, Nthreads) blocks = (Mblocks, Nblocks) _parallel_for_cuda_MN(f, x...)
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