module multi

using JACC, AMDGPU

function JACC.multi.ndev(::AMDGPUBackend)
  return length(AMDGPU.devices())
end

function get_portable_rocarray(x::Base.Array{T, N}) where {T, N}
    dims = size(x)
    bytesize = sizeof(T) * prod(dims)
    buf = AMDGPU.Mem.HostBuffer(bytesize, AMDGPU.HIP.hipHostAllocPortable)
    ROCArray{T, N}(AMDGPU.GPUArrays.DataRef(AMDGPU.pool_free, AMDGPU.Managed(buf)), dims)
end

function JACC.multi.Array(::AMDGPUBackend, x::Base.Array{T,N}) where {T,N}

  ret = Vector{Any}(undef, 2)  
  ndev = length(AMDGPU.devices())

  if ndims(x) == 1

    AMDGPU.device!(AMDGPU.device(1))
    s_array = length(x)
    s_arrays = ceil(Int, s_array/ndev)
    #println(s_arrays)
    array_ret = Vector{Any}(undef, ndev)  
    pointer_ret = Vector{AMDGPU.Device.ROCDeviceVector{T,AMDGPU.Device.AS.Global}}(undef, ndev)  

    for i in 1:ndev
      AMDGPU.device!(AMDGPU.device(i))
      array_ret[i] = ROCArray(x[((i-1)*s_arrays)+1:i*s_arrays])
      pointer_ret[i] = AMDGPU.rocconvert(array_ret[i])
    end
    
    AMDGPU.device!(AMDGPU.device(1))
    #amdgpu_pointer_ret = ROCArray(pointer_ret)
    amdgpu_pointer_ret = get_portable_rocarray(pointer_ret)
    copyto!(amdgpu_pointer_ret, pointer_ret)
    ret[1] = amdgpu_pointer_ret
    ret[2] = array_ret

  elseif ndims(x) == 2

    AMDGPU.device!(AMDGPU.device(1))
    #s_row_array = size(x,1)
    s_col_array = size(x,2)
    s_col_arrays = ceil(Int, s_col_array/ndev)
    array_ret = Vector{Any}(undef, ndev)  
    pointer_ret = Vector{AMDGPU.Device.ROCDeviceMatrix{T,1}}(undef, ndev)  

    for i in 1:ndev
      AMDGPU.device!(AMDGPU.device(i))
      array_ret[i] = ROCArray(x[:,((i-1)*s_col_arrays)+1:i*s_col_arrays])
      pointer_ret[i] = AMDGPU.rocconvert(array_ret[i])
    end
  
    AMDGPU.device!(AMDGPU.device(1))
    #amdgpu_pointer_ret = ROCArray(pointer_ret)
    amdgpu_pointer_ret = get_portable_rocarray(pointer_ret)
    copyto!(amdgpu_pointer_ret, pointer_ret)
    ret[1] = amdgpu_pointer_ret
    ret[2] = array_ret

  end

  return ret

end

function JACC.multi.copy(::AMDGPUBackend, x::Vector{Any}, y::Vector{Any})
   
   AMDGPU.device!(AMDGPU.device(1))
   ndev = length(AMDGPU.devices())

   for i in 1:ndev
       AMDGPU.device!(AMDGPU.device(i))
       size = length(x[2][i])
       numThreads = 512
       threads = min(size, numThreads)
       blocks = ceil(Int, size / threads)
       @roc groupsize=threads gridsize=blocks _multi_copy(i, x[1], y[1])
       #AMDGPU.synchronize()
   end

   for i in 1:ndev
      AMDGPU.device!(AMDGPU.device(i))
      AMDGPU.synchronize()
   end

   AMDGPU.device!(AMDGPU.device(1))

end

function JACC.multi.parallel_for(::AMDGPUBackend, N::I, f::F, x...) where {I <: Integer, F <: Function}

  ndev = length(AMDGPU.devices())
  N_multi = ceil(Int, N/ndev)
  numThreads = 256
  threads = min(N_multi, numThreads)
  blocks = ceil(Int, N_multi / threads)
  
  for i in 1:ndev
    AMDGPU.device!(AMDGPU.device(i))
    dev_id = i
    @roc groupsize=threads gridsize=blocks _multi_parallel_for_amdgpu(N_multi, dev_id, f, x...)
  end

  for i in 1:ndev
    AMDGPU.device!(AMDGPU.device(i))
    AMDGPU.synchronize()
  end
  
  AMDGPU.device!(AMDGPU.device(1))

end

function JACC.multi.parallel_for(::AMDGPUBackend, (M, N)::Tuple{I,I}, f::F, x...) where {I <: Integer, F <: Function}

  ndev = length(AMDGPU.devices())
  N_multi = ceil(Int, N/ndev)
  numThreads = 16
  Mthreads = min(M, numThreads)
  Nthreads = min(N_multi, numThreads)
  Mblocks = ceil(Int, M / Mthreads)
  Nblocks = ceil(Int, N_multi / Nthreads)

  for i in 1:ndev
    AMDGPU.device!(AMDGPU.device(i))
    dev_id = i
    @roc groupsize=(Mthreads, Nthreads) gridsize=(Mblocks, Nblocks) _multi_parallel_for_amdgpu_MN(M, N_multi, dev_id, f, x...)
  end

  for i in 1:ndev
    AMDGPU.device!(AMDGPU.device(i))
    AMDGPU.synchronize()
  end
  
  AMDGPU.device!(AMDGPU.device(1))

end

function JACC.multi.parallel_reduce(::AMDGPUBackend, N::I, f::F, x...) where {I <: Integer, F <: Function}

    AMDGPU.device!(AMDGPU.device(1))
    ndev = length(AMDGPU.devices())
    ret = Vector{Any}(undef, ndev)
    rret = Vector{Any}(undef, ndev)
    N_multi = ceil(Int, N/ndev)
    numThreads = 512
    threads = min(N_multi, numThreads)
    blocks = ceil(Int, N_multi / threads)
    final_rret = AMDGPU.zeros(Float64, 1)

    for i in 1:ndev
      AMDGPU.device!(AMDGPU.device(i))
      ret[i] = AMDGPU.zeros(Float64, blocks)
      rret[i] = AMDGPU.zeros(Float64, 1)
    end

    for i in 1:ndev
      AMDGPU.device!(AMDGPU.device(i))
      dev_id = i
      @roc groupsize=threads gridsize=blocks _multi_parallel_reduce_amdgpu(
        N_multi, dev_id, ret[i], f, x...)
    end
    for i in 1:ndev
      AMDGPU.device!(AMDGPU.device(i))
      dev_id = i
      @roc groupsize=threads gridsize=1 _multi_reduce_kernel_amdgpu(
        blocks, ret[i], rret[i])
    end

    for i in 1:ndev
      AMDGPU.device!(AMDGPU.device(i))
      AMDGPU.synchronize()
    end

    tmp_rret = Vector{Any}(undef, ndev)
    tmp_final_rret = 0.0

    for i in 1:ndev
      tmp_rret[i] = zeros(Float64, 1)
      AMDGPU.device!(AMDGPU.device(i))
      tmp_rret[i] = Base.Array(rret[i])
      #println(tmp_rret[i][1])
    end

    AMDGPU.device!(AMDGPU.device(1))
    for i in 1:ndev
      tmp_final_rret += tmp_rret[i][1]
    end
    final_rret = tmp_final_rret

    return final_rret
end

function JACC.multi.parallel_reduce(::AMDGPUBackend, 
        (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}

  ndev = length(devices())
  ret = Vector{Any}(undef, ndev)
  rret = Vector{Any}(undef, ndev)
  N_multi = ceil(Int, N/ndev)
  numThreads = 16
  Mthreads = min(M, numThreads)
  Nthreads = min(N_multi, numThreads)
  Mblocks = ceil(Int, M / Mthreads)
  Nblocks = ceil(Int, N_multi / Nthreads)
  final_rret = AMDGPU.zeros(Float64, 1)

  for i in 1:ndev
    AMDGPU.device!(AMDGPU.device(i))
    ret[i] = AMDGPU.zeros(Float64, (Mblocks, Nblocks))
    rret[i] = AMDGPU.zeros(Float64, 1)
  end

  for i in 1:ndev
	  AMDGPU.device!(AMDGPU.device(i))
    dev_id = i

    @roc groupsize=(Mthreads, Nthreads) gridsize=(Mblocks, Nblocks) _multi_parallel_reduce_amdgpu_MN(
        (M, N_multi), dev_id, ret[i], f, x...)

    @roc groupsize=(Mthreads, Nthreads) gridsize=(1, 1) _multi_reduce_kernel_amdgpu_MN(
        (Mblocks, Nblocks), ret[i], rret[i])
  end

  for i in 1:ndev
    AMDGPU.device!(AMDGPU.device(i))
    AMDGPU.synchronize()
  end
  
  tmp = zeros(ndev)

  for i in 1:ndev
    AMDGPU.device!(AMDGPU.device(i))
    tmp[i] = Base.Array(rret[i])
  end

  AMDGPU.device!(AMDGPU.device(1))
  for i in 1:ndev
    final_rret += tmp[i]
  end

  return final_rret

end

function _multi_parallel_for_amdgpu(N, dev_id, f, x...)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if i <= N
        f(dev_id, i, x...)
    end
    return nothing
end

function _multi_parallel_for_amdgpu_MN(M, N, dev_id, f, x...)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    if (i <= M) && (j <= N)
        f(dev_id, i, j, x...)
    end
    return nothing
end

function _multi_parallel_reduce_amdgpu(N, dev_id, ret, f, x...)
    shared_mem = @ROCStaticLocalArray(Float64, 512)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ti = workitemIdx().x
    tmp::Float64 = 0.0
    shared_mem[ti] = 0.0

    if i <= N
        tmp = @inbounds f(dev_id, i, x...)
        shared_mem[workitemIdx().x] = tmp
    end
    AMDGPU.sync_workgroup()
    if (ti <= 256)
        shared_mem[ti] += shared_mem[ti + 256]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 128)
        shared_mem[ti] += shared_mem[ti + 128]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 64)
        shared_mem[ti] += shared_mem[ti + 64]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 32)
        shared_mem[ti] += shared_mem[ti + 32]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 16)
        shared_mem[ti] += shared_mem[ti + 16]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 8)
        shared_mem[ti] += shared_mem[ti + 8]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 4)
        shared_mem[ti] += shared_mem[ti + 4]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 2)
        shared_mem[ti] += shared_mem[ti + 2]
    end
    AMDGPU.sync_workgroup()
    if (ti == 1)
        shared_mem[ti] += shared_mem[ti + 1]
	ret[workgroupIdx().x] = shared_mem[ti]
    end
    return nothing
end

function _multi_reduce_kernel_amdgpu(N, red, ret)
    shared_mem = @ROCStaticLocalArray(Float64, 512)
    i = workitemIdx().x
    ii = i
    tmp::Float64 = 0.0
    if N > 512
        while ii <= N
            tmp += @inbounds red[ii]
            ii += 512
        end
    elseif (i <= N)
          tmp = @inbounds red[i]
    end
    shared_mem[workitemIdx().x] = tmp
    AMDGPU.sync_workgroup()
    if (i <= 256)
        shared_mem[i] += shared_mem[i + 256]
    end
    AMDGPU.sync_workgroup()
    if (i <= 128)
        shared_mem[i] += shared_mem[i + 128]
    end
    AMDGPU.sync_workgroup()
    if (i <= 64)
        shared_mem[i] += shared_mem[i + 64]
    end
    AMDGPU.sync_workgroup()
    if (i <= 32)
        shared_mem[i] += shared_mem[i + 32]
    end
    AMDGPU.sync_workgroup()
    if (i <= 16)
        shared_mem[i] += shared_mem[i + 16]
    end
    AMDGPU.sync_workgroup()
    if (i <= 8)
        shared_mem[i] += shared_mem[i + 8]
    end
    AMDGPU.sync_workgroup()
    if (i <= 4)
        shared_mem[i] += shared_mem[i + 4]
    end
    AMDGPU.sync_workgroup()
    if (i <= 2)
        shared_mem[i] += shared_mem[i + 2]
    end
    AMDGPU.sync_workgroup()
    if (i == 1)
        shared_mem[i] += shared_mem[i + 1]
        ret[1] = shared_mem[1]
    end
    AMDGPU.sync_workgroup()
    return nothing
end

function _multi_parallel_reduce_amdgpu_MN((M, N), dev_id, ret, f, x...)
    shared_mem = @ROCStaticLocalArray(Float64, 16*16)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    ti = workitemIdx().x
    tj = workitemIdx().y
    bi = workgroupIdx().x
    bj = workgroupIdx().y

    tmp::Float64 = 0.0
    shared_mem[((ti - 1) * 16) + tj] = tmp

    if (i <= M && j <= N)
        tmp = @inbounds f(dev_id, i, j, x...)
        shared_mem[(ti - 1) * 16 + tj] = tmp
    end
    AMDGPU.sync_workgroup()
    if (ti <= 8 && tj <= 8 && ti + 8 <= M && tj + 8 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + tj]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 4 && tj <= 4 && ti + 4 <= M && tj + 4 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + tj]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 2 && tj <= 2 && ti + 2 <= M && tj + 2 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + tj]
    end
    AMDGPU.sync_workgroup()
    if (ti == 1 && tj == 1 && ti + 1 <= M && tj + 1 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + tj]
        ret[bi, bj] = shared_mem[((ti - 1) * 16) + tj]
    end
    return nothing
end

function _multi_reduce_kernel_amdgpu_MN((M, N), red, ret)
    shared_mem = @ROCStaticLocalArray(Float64, 16*16)
    i = workitemIdx().x
    j = workitemIdx().y
    ii = i
    jj = j

    tmp::Float64 = 0.0
    shared_mem[(i - 1) * 16 + j] = tmp

    if M > 16 && N > 16
        while ii <= M
            jj = workitemIdx().y
            while jj <= N
                tmp = tmp + @inbounds red[ii, jj]
                jj += 16
            end
            ii += 16
        end
    elseif M > 16
        while ii <= N
            tmp = tmp + @inbounds red[ii, jj]
            ii += 16
        end
    elseif N > 16
        while jj <= N
            tmp = tmp + @inbounds red[ii, jj]
            jj += 16
        end
    elseif M <= 16 && N <= 16
        if i <= M && j <= N
            tmp = tmp + @inbounds red[i, j]
        end
    end
    shared_mem[(i - 1) * 16 + j] = tmp
    red[i, j] = shared_mem[(i - 1) * 16 + j]
    AMDGPU.sync_workgroup()
    if (i <= 8 && j <= 8)
        if (i + 8 <= M && j + 8 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 7) * 16) + (j + 8)]
        end
        if (i <= M && j + 8 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 8)]
        end
        if (i + 8 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 7) * 16) + j]
        end
    end
    AMDGPU.sync_workgroup()
    if (i <= 4 && j <= 4)
        if (i + 4 <= M && j + 4 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 3) * 16) + (j + 4)]
        end
        if (i <= M && j + 4 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 4)]
        end
        if (i + 4 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 3) * 16) + j]
        end
    end
    AMDGPU.sync_workgroup()
    if (i <= 2 && j <= 2)
        if (i + 2 <= M && j + 2 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 1) * 16) + (j + 2)]
        end
        if (i <= M && j + 2 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 2)]
        end
        if (i + 2 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i + 1) * 16) + j]
        end
    end
    AMDGPU.sync_workgroup()
    if (i == 1 && j == 1)
        if (i + 1 <= M && j + 1 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[i * 16 + (j + 1)]
        end
        if (i <= M && j + 1 <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[((i - 1) * 16) + (j + 1)]
        end
        if (i + 1 <= M && j <= N)
            shared_mem[((i - 1) * 16) + j] += shared_mem[i * 16 + j]
        end
        ret[1] = shared_mem[((i - 1) * 16) + j]
    end
    return nothing
end

end # module multi
