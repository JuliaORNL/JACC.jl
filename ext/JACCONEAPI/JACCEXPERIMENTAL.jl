module experimental

using JACC, oneAPI

function JACC.experimental.shared(x::oneDeviceArray)
  T = eltype(x)
  size = length(x)
  shmem = oneLocalArray(T, size)
  num_threads = get_local_size(0) * get_local_size(1)
  if (size <= num_threads)
    if get_local_size(1) == 1
      ind = get_global_id(0)
      @inbounds shmem[ind] = x[ind]
    else
      i_local = get_local_id(0)
      j_local = get_local_id(1)
      ind = i_local - 1 * get_local_size(0) + j_local
      if ndims(x) == 1
        @inbounds shmem[ind] = x[ind]
      elseif ndims(x) == 2
        @inbounds shmem[ind] = x[i_local,j_local]
      end
    end
  else
    if get_local_size(1) == 1
      ind = get_local_id(0)
      for i in get_local_size(0):get_local_size(0):size
        @inbounds shmem[ind] = x[ind]
        ind += get_local_size(0)
      end
    else
      i_local = get_local_id(0)
      j_local = get_local_id(1)
      ind = (i_local - 1) * get_local_size(0) + j_local
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
  barrier()
  return shmem
end

end # module experimental
