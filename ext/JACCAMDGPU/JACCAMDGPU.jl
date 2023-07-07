module JACCAMDGPU

using JACC, AMDGPU

function JACC.parallel_for(N::I, f::F, x...) where {I<:Integer,F<:Function}
  numThreads = 512
  threads = min(N, numThreads)
  blocks = ceil(Int, N / threads)
  @roc groupsize=threads gridsize=threads*blocks _parallel_for_amdgpu(f, x...)
  AMDGPU.synchronize()
end

function JACC.parallel_for((M, N), f::F, x...) where {F<:Function}
  numThreads = 16 
  Mthreads = min(M, numThreads)
  Nthreads = min(N, numThreads)  
  Mblocks = ceil(Int, M/Mthreads)
  Nblocks = ceil(Int, N/Nthreads)
  @roc groupsize=(Mthreads, Nthreads) gridsize=(Mblocks*Mthreads, Nblocks*Nthreads) _parallel_for_amdgpu_MN(f, x...)
  AMDGPU.synchronize()
end

function JACC.parallel_reduce(N::I, f::F, x...) where {I<:Integer,F<:Function}
  numThreads = 1024
  threads = min(N, numThreads)
  ret = AMDGPU.zeros(1)
  @roc groupsize=threads gridsize=threads localmem=1024*sizeof(Float64) _parallel_reduce_amdgpu(N, ret, f, x...)
  AMDGPU.synchronize()
  return ret[1]
end

function JACC.parallel_reduce((M, N), f::F, x...) where {F<:Function}
  numThreads = 16
  Mthreads = min(M, numThreads)
  Nthreads = min(N, numThreads)
  ret = AMDGPU.zeros(1)
  @roc groupsize=(Mthreads, Nthreads) gridsize=(Mthreads, Nthreads) localmem=16*16*sizeof(Float64) _parallel_reduce_amdgpu_MN((M, N), ret, f, x...)
  AMDGPU.synchronize()
  return ret[1]
end

function _parallel_for_amdgpu(f, x...)
  i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
  f(i, x...)
  return nothing
end

function _parallel_for_amdgpu_MN(f, x...)
  i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
  j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
  f(i, j, x...)
  return nothing
end

function _parallel_reduce_amdgpu(N, ret, f, x...)
    shared_mem = @ROCDynamicLocalArray(Float64, 1024)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ii = i
    tmp::Float64 = 0.0
    if N > 1024
      while ii <= N
        tmp += @inbounds f(ii, x...)
        ii += 1024
      end
    else
      tmp = @inbounds f(i,x...)
    end
    shared_mem[i] = tmp
    AMDGPU.sync_workgroup()
    if (i <= 512)
      shared_mem[i] += shared_mem[i+512]
    end
    AMDGPU.sync_workgroup()
    if (i <= 256)
      shared_mem[i] += shared_mem[i+256]
    end
    AMDGPU.sync_workgroup()
    if (i <= 128)
      shared_mem[i] += shared_mem[i+128]
    end
    AMDGPU.sync_workgroup()
    if (i <= 64)
      shared_mem[i] += shared_mem[i+64]
    end
    AMDGPU.sync_workgroup()
    if (i <= 32)
      shared_mem[i] += shared_mem[i+32]
    end
    AMDGPU.sync_workgroup()
    if (i <= 16)
      shared_mem[i] += shared_mem[i+16]
    end
    AMDGPU.sync_workgroup()
    if (i <= 8)
      shared_mem[i] += shared_mem[i+8]
    end
    AMDGPU.sync_workgroup()
    if (i <= 4)
      shared_mem[i] += shared_mem[i+4]
    end
    AMDGPU.sync_workgroup()
    if (i <= 2)
      shared_mem[i] += shared_mem[i+2]
    end
    AMDGPU.sync_workgroup()
    if (i == 1)
      shared_mem[i] += shared_mem[i+1]
      ret[1] = shared_mem[i] 
    end
    return nothing
end

function _parallel_reduce_amdgpu_MN((M, N), ret, f, x...)
    shared_mem = @ROCDynamicLocalArray(Float64, (16, 16))
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x   
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y    
    ii = i
    jj = j
    tmp::Float64 = 0.0
    
    if M > 16 && N > 16
      while ii <= M
        jj = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
        while jj <= N
          tmp += f(ii, jj, x...)
          jj += 16
        end
        ii += 16 
      end
    elseif M > 16
      while ii <= N
        tmp += f(ii, jj, x...)
        ii += 16
      end
    elseif N > 16
      while jj <= N
        tmp += f(ii, jj, x...)
        jj += 16
      end
    else
      tmp = f(i,j,x...)
    end
    shared_mem[i,j] = tmp
    AMDGPU.sync_workgroup()
    if (i <= 8 && j <= 8)
      shared_mem[i,j] += shared_mem[i+8,j+8]
      shared_mem[i,j] += shared_mem[i  ,j+8]
      shared_mem[i,j] += shared_mem[i+8,j  ]
    end
    AMDGPU.sync_workgroup()
    if (i <= 4 && j <= 4)
      shared_mem[i,j] += shared_mem[i+4,j+4]
      shared_mem[i,j] += shared_mem[i  ,j+4]
      shared_mem[i,j] += shared_mem[i+4,j  ]
    end
    AMDGPU.sync_workgroup()
    if (i <= 2 && j <= 2)
      shared_mem[i,j] += shared_mem[i+2,j+2]
      shared_mem[i,j] += shared_mem[i  ,j+2]
      shared_mem[i,j] += shared_mem[i+2,j  ]
    end
    AMDGPU.sync_workgroup()
    if (i == 1 && j == 1)
      shared_mem[i,j] += shared_mem[i+1,j+1]
      shared_mem[i,j] += shared_mem[i  ,j+1]
      shared_mem[i,j] += shared_mem[i+1,j  ]
      ret[1] += shared_mem[i,j]
    end
    
    return nothing
end

function __init__()
    const JACC.Array = AMDGPU.ROCArray{T,N} where {T,N}
end

end # module JACCAMDGPU
