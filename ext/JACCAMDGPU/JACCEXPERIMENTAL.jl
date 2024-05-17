module experimental

using JACC, AMDGPU

function JACC.experimental.shared(x::ROCDeviceArray)
  T = eltype(x)
  size = length(x)
  shmem = @ROCDynamicLocalArray(T, size)
  num_threads = workgroupDim().x * workgroupDim().y
  if (size <= num_threads)
    if workgroupDim().y == 1
      ind = workitemIdx().x
      @inbounds shmem[ind] = x[ind]
    else
      i_local = workitemIdx().x
      j_local = workitemIdx().y
      ind = (i_local - 1) * workgroupDim().x + j_local
      if ndims(x) == 1
        @inbounds shmem[ind] = x[ind]
      elseif ndims(x) == 2
        @inbounds shmem[ind] = x[i_local,j_local]
      end
    end
  else
    if workgroupDim().y == 1
      ind = workgroupIdx().x
     for i in workgroupDim().x:workgroupDim().x:size
       @inbounds shmem[ind] = x[ind]
       ind += workgroupDim().x
      end
    else
      i_local = workgroupIdx().x
      j_local = workgroupIdx().y
      ind = (i_local - 1) * workgroupDim().x + j_local
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
  AMDGPU.sync_workgroup()
  return shmem
end

end # module experimental
