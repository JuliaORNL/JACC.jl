
module JACCCUDA

using JACC, CUDA

function JACC.parallel_for(N::I, f::F, x...) where {I<:Integer,F<:Function}
  maxPossibleThreads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
  threads = min(N, maxPossibleThreads)
  blocks = ceil(Int, N / threads)
  CUDA.@sync @cuda threads = threads blocks = blocks _parallel_for_cuda(f, x...)
end

function _parallel_for_cuda(f, x...)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  f(i, x...)
  return nothing
end

function __init__()
  const JACC.Array = CUDA.CuArray{T,N} where {T,N}
end

end # module JACCCUDA
