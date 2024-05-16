module experimental

using JACC, CUDA

function JACC.experimental.shared(x::CuDeviceArray)
  T = eltype(x)
  size = length(x)
  shmem = @cuDynamicSharedMem(T, size)
  num_threads = blockDim().x * blockDim().y
  if (size <= num_threads)
    if blockDim().y == 1
      ind = threadIdx().x
        @inbounds shmem[ind] = x[ind]
    else
      i_local = threadIdx().x
      j_local = threadIdx().y
      ind = (i_local - 1) * blockDim().x + j_local
      if ndims(x) == 1
          @inbounds shmem[ind] = x[ind]
      elseif ndims(x) == 2
          @inbounds shmem[ind] = x[i_local,j_local]
      end
    end
  else
    if blockDim().y == 1
      ind = threadIdx().x
      for i in blockDim().x:blockDim().x:size
        @inbounds shmem[ind] = x[ind]
        ind += blockDim().x
      end
    else
      i_local = threadIdx().x
      j_local = threadIdx().y
      ind = (i_local - 1) * blockDim().x + j_local
      if ndims(x) == 1
        for i in num_threads:num_threads:size
          @inbounds shmem[ind] = x[ind]
          ind += num_threads
        end
      elseif ndims(x) == 2
        for i in num_threads:num_threads:size
          @inbounds shmem[ind] = x[i_local,j_local]
          ind += num_threads
        end
      end  
    end
  end
  sync_threads()
  return shmem
end

end # module experimental
