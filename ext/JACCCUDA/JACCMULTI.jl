module multi

using JACC, CUDA

function JACC.multi.ndev(::CUDABackend)
  return length(devices())
end

function JACC.multi.Array(::CUDABackend, x::Base.Array{T,N}) where {T,N}

  ndev = length(devices())
  ret = Vector{Any}(undef, 2)  

  if ndims(x) == 1

    device!(0)
    s_array = length(x)
    s_arrays = ceil(Int, s_array/ndev)
    array_ret = Vector{Any}(undef, ndev)  
    pointer_ret = Vector{CuDeviceVector{T,CUDA.AS.Global}}(undef, ndev)  

    for i in 1:ndev
      device!(i-1)
      array_ret[i] = CuArray(x[((i-1)*s_arrays)+1:i*s_arrays])
      pointer_ret[i] = cudaconvert(array_ret[i])
    end
  
    device!(0)
    cuda_pointer_ret = CuArray(pointer_ret)
    ret[1] = cuda_pointer_ret
    ret[2] = array_ret

  elseif ndims(x) == 2

    device!(0)
    s_col_array = size(x,2)
    s_col_arrays = ceil(Int, s_col_array/ndev)
    array_ret = Vector{Any}(undef, ndev)  
    pointer_ret = Vector{CuDeviceMatrix{T,CUDA.AS.Global}}(undef, ndev)  

    for i in 1:ndev
      device!(i-1)
      array_ret[i] = CuArray(x[:,((i-1)*s_col_arrays)+1:i*s_col_arrays])
      pointer_ret[i] = cudaconvert(array_ret[i])
    end
  
    device!(0)
  
    cuda_pointer_ret = CuArray(pointer_ret)
    ret[1] = cuda_pointer_ret
    ret[2] = array_ret

  end

  return ret

end

function JACC.multi.gArray(::CUDABackend, x::Base.Array{T,N}) where {T,N}

  ndev = length(devices())
  ret = Vector{Any}(undef, 2)  

  if ndims(x) == 1

    device!(0)
    s_array = length(x)
    s_arrays = ceil(Int, s_array/ndev)
    array_ret = Vector{Any}(undef, ndev)  
    pointer_ret = Vector{CuDeviceVector{T,CUDA.AS.Global}}(undef, ndev)  

    for i in 1:ndev
      device!(i-1)
      if i == 1
          array_ret[i] = CuArray(x[((i-1)*s_arrays)+1:(i*s_arrays)+1])
      elseif i == ndev
          array_ret[i] = CuArray(x[(((i-1)*s_arrays)+1)-1:i*s_arrays])
      else
          array_ret[i] = CuArray(x[(((i-1)*s_arrays)+1)-1:(i*s_arrays)+1])
      end
      pointer_ret[i] = cudaconvert(array_ret[i])
    end
  
    device!(0)
    cuda_pointer_ret = CuArray(pointer_ret)
    ret[1] = cuda_pointer_ret
    ret[2] = array_ret

  elseif ndims(x) == 2

    device!(0)
    s_col_array = size(x,2)
    s_col_arrays = ceil(Int, s_col_array/ndev)
    array_ret = Vector{Any}(undef, ndev)  
    pointer_ret = Vector{CuDeviceMatrix{T,CUDA.AS.Global}}(undef, ndev)  

    for i in 1:ndev
      device!(i-1)
      array_ret[i] = CuArray(x[:,((i-1)*s_col_arrays)+1:i*s_col_arrays])
      pointer_ret[i] = cudaconvert(array_ret[i])
    end
  
    device!(0)
  
    cuda_pointer_ret = CuArray(pointer_ret)
    ret[1] = cuda_pointer_ret
    ret[2] = array_ret

  end

  return ret

end

function JACC.multi.copy(::CUDABackend, x::Vector{Any}, y::Vector{Any})
   
  device!(0) 
  ndev = length(devices())
   
  if ndims(x[2][1]) == 1
  
    for i in 1:ndev
      device!(i-1)
      size = length(x[2][i])
      numThreads = 512
      threads = min(size, numThreads)
      blocks = ceil(Int, size / threads)
      @cuda threads=threads blocks=blocks _multi_copy(i, x[1], y[1])
    end
    
    for i in 1:ndev
      device!(i-1)
      synchronize()
    end 

  elseif ndims(x[2][1]) == 2

    for i in 1:ndev
      device!(i-1)
      ssize = size(x[2][i])
      numThreads = 16
      Mthreads = min(ssize[1], numThreads)
      Mblocks = ceil(Int, ssize[1] / Mthreads)
      Nthreads = min(ssize[2], numThreads)
      Nblocks = ceil(Int, ssize[2] / Nthreads)
      @cuda threads=(Mthreads,Nthreads) blocks=(Mblocks,Nblocks) _multi_copy_2d(i, x[1], y[1])
    end
     
    for i in 1:ndev
      device!(i-1)
      synchronize()
    end
   
   end

   device!(0) 
  
end

function JACC.multi.gid(::CUDABackend, dev_id::I, i::I, ndev::I) where{I <: Integer}
    ind = 0
    if dev_id == 1
        ind = i
    elseif dev_id == ndev
        ind = i + 1
    else
        ind = i + 1
    end    
    return ind
end

function JACC.multi.gid(::CUDABackend, dev_id::I, (i,j)::Tuple{I,I}, ndev::I) where{I <: Integer}
    
    ind = (0, 0)
    
    if dev_id == 1
      ind = (i, j)
    elseif dev_id == ndev
      ind = (i, j+1)
    else
      ind = (i, j+1)
    end
    
    return ind

end

function JACC.multi.gswap(::CUDABackend, x::Vector{Any})
   
   device!(0) 
   ndev = length(devices())
   
   if ndims(x[2][1]) == 1
   
     #Left to right swapping
     for i in 1:ndev-1
       device!(i-1)
       tmp = Base.Array(x[2][i])
       size = length(tmp)
       ghost_lr = tmp[size-1]
       device!(i)
       @cuda threads=32 blocks=1 _multi_swap_ghost_lr(i+1, x[1], ndev, size, ghost_lr)
     end
   
     #Right to left swapping
     for i in 2:ndev
       device!(i-1)
       tmp = Base.Array(x[2][i])
       size = length(tmp)
       ghost_rl = tmp[2]
       device!(i-2)
       @cuda threads=32 blocks=1 _multi_swap_ghost_rl(i-1, x[1], ndev, size, ghost_rl)
     end
   
     for i in 1:ndev
       device!(i-1)
       synchronize()
     end 

   elseif ndims(x[2][1]) == 2
  
     #Left to right swapping
     for i in 1:ndev-1
       device!(i-1)
       dim = size(x[2][i])
       tmp = Base.Array(x[2][i][:,dim[2]-1])
       device!(i)
       ghost_lr = CuArray(tmp)
       numThreads = 512
       threads = min(dim[1], numThreads)
       blocks = ceil(Int, dim[1] / threads)
       @cuda threads=threads blocks=blocks _multi_swap_2d_ghost_lr(i+1, x[1], ndev, dim[1], ghost_lr)
     end

     #Right to left swapping
     for i in 2:ndev
       device!(i-1)
       tmp = Base.Array(x[2][i][:,2])
       device!(i-2)
       dim = size(x[2][i-1])
       ghost_rl = CuArray(tmp)
       numThreads = 512
       threads = min(dim[1], numThreads)
       blocks = ceil(Int, dim[1] / threads)
       @cuda threads=threads blocks=blocks _multi_swap_2d_ghost_rl(i-1, x[1], ndev, dim[1], dim[2], ghost_rl)
     end

     for i in 1:ndev
        device!(i-1)
        synchronize()
     end

   end 
     
   device!(0)

end

function JACC.multi.gcopytoarray(::CUDABackend, x::Vector{Any}, y::Vector{Any})
   #x is the array and y is the ghost array
   device!(0) 
   ndev = length(devices())
   
   for i in 1:ndev
       device!(i-1)
       size = length(y[2][i])
       numThreads = 512
       threads = min(size, numThreads)
       blocks = ceil(Int, size / threads)
       @cuda threads=threads blocks=blocks _multi_copy_ghosttoarray(i, x[1], y[1], size, ndev)
       #synchronize() 
   end
   
   for i in 1:ndev
      device!(i-1)
      synchronize()
   end 
   
   device!(0) 
  
end

function JACC.multi.copytogarray(::CUDABackend, x::Vector{Any}, y::Vector{Any})
   #x is the ghost array and y is the array
   device!(0) 
   ndev = length(devices())
   
   for i in 1:ndev
       device!(i-1)
       size = length(x[2][i])
       numThreads = 512
       threads = min(size, numThreads)
       blocks = ceil(Int, size / threads)
       @cuda threads=threads blocks=blocks _multi_copy_arraytoghost(i, x[1], y[1], size, ndev)
       #synchronize() 
   end
   
   for i in 1:ndev
      device!(i-1)
      synchronize()
   end 
   
   device!(0) 
  
end

function JACC.multi.parallel_for(::CUDABackend, N::I, f::F, x...) where {I <: Integer, F <: Function}

  device!(0)
  ndev = length(devices())
  N_multi = ceil(Int, N/ndev)
  numThreads = 256
  threads = min(N_multi, numThreads)
  blocks = ceil(Int, N_multi / threads)

  for i in 1:ndev
    device!(i-1)
    dev_id = i
    @cuda threads=threads blocks=blocks _multi_parallel_for_cuda(N_multi, dev_id, f, x...)
  end

  for i in 1:ndev
    device!(i-1)
    synchronize()
  end
  
  device!(0)

end

function JACC.multi.parallel_reduce(::CUDABackend, N::I, f::F, x...) where {I <: Integer, F <: Function}
    
    device!(0)
    ndev = length(devices())
    ret = Vector{Any}(undef, ndev)  
    rret = Vector{Any}(undef, ndev)  
    N_multi = ceil(Int, N/ndev)
    numThreads = 512
    threads = min(N_multi, numThreads)
    blocks = ceil(Int, N_multi / threads)
    final_rret = CUDA.zeros(Float64, 1)
    
    for i in 1:ndev
      device!(i-1)
      ret[i] = CUDA.zeros(Float64, blocks)
      rret[i] = CUDA.zeros(Float64, 1)
    end

    for i in 1:ndev
      device!(i-1)
      dev_id = i
      @cuda threads=threads blocks=blocks shmem=512 * sizeof(Float64) _multi_parallel_reduce_cuda(
        N_multi, dev_id, ret[i], f, x...)
      @cuda threads=threads blocks=1 shmem=512 * sizeof(Float64) _multi_reduce_kernel_cuda(
        blocks, ret[i], rret[i])
    end

    for i in 1:ndev
      device!(i-1)
      synchronize()
    end
    
    for i in 1:ndev
      final_rret += rret[i]
    end
  
    device!(0)
    
    return final_rret
end

function JACC.multi.parallel_for(::CUDABackend, 
         (M, N)::Tuple{I,I}, f::F, x...) where {I <: Integer, F <: Function}

  ndev = length(devices())
  N_multi = ceil(Int, N/ndev)
  numThreads = 16
  Mthreads = min(M, numThreads)
  Nthreads = min(N_multi, numThreads)
  Mblocks = ceil(Int, M / Mthreads)
  Nblocks = ceil(Int, N_multi / Nthreads)

  for i in 1:ndev
    device!(i-1)
    dev_id = i
    @cuda threads=(Mthreads, Nthreads) blocks=(Mblocks, Nblocks) _multi_parallel_for_cuda_MN(M, N_multi, dev_id, f, x...)
  end

  for i in 1:ndev
    device!(i-1)
    synchronize()
  end
  
  device!(0)

end

function JACC.multi.parallel_reduce(::CUDABackend, 
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
  final_rret = CUDA.zeros(Float64, 1)
  
  for i in 1:ndev
    device!(i-1)
    ret[i] = CUDA.zeros(Float64, (Mblocks, Nblocks))
    rret[i] = CUDA.zeros(Float64, 1)
  end
    
  for i in 1:ndev
    device!(i-1)
    dev_id = i

    @cuda threads=(Mthreads, Nthreads) blocks=(Mblocks, Nblocks) shmem=16 * 16 * sizeof(Float64) _multi_parallel_reduce_cuda_MN(
        (M, N_multi), dev_id, ret[i], f, x...)
    
    @cuda threads=(Mthreads, Nthreads) blocks=(1, 1) shmem=16 * 16 *sizeof(Float64) _multi_reduce_kernel_cuda_MN(
        (Mblocks, Nblocks), ret[i], rret[i])
  end
    
  for i in 1:ndev
    device!(i-1)
    synchronize()
  end
    
  for i in 1:ndev
    final_rret += rret[i]
  end
  
  device!(0)
    
  return final_rret

end

function _multi_copy(dev_id, x, y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds x[dev_id][i] = y[dev_id][i]
    return nothing
end

function _multi_copy_2d(dev_id, x, y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds x[dev_id][i,j] = y[dev_id][i,j]
    return nothing
end

function _multi_copy_ghosttoarray(dev_id, x, y, size, ndev)
    #x is the array and y is the ghost array
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if dev_id == 1 && i < size
        @inbounds x[dev_id][i] = y[dev_id][i]
    elseif dev_id == ndev && i > 1
        @inbounds x[dev_id][i-1] = y[dev_id][i]
    elseif i > 1 && i < size
        @inbounds x[dev_id][i-1] = y[dev_id][i]
    end
    return nothing
end

function _multi_copy_arraytoghost(dev_id, x, y, size, ndev)
    #x is the ghost array and y is the array
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if dev_id == 1 && i < size
        @inbounds x[dev_id][i] = y[dev_id][i]
    elseif dev_id == ndev && i < size
        @inbounds x[dev_id][i+1] = y[dev_id][i]
    elseif i > 1 && i < size
        @inbounds x[dev_id][i] = y[dev_id][i]
    end
    return nothing
end

function _multi_swap_ghost_lr(dev_id, x, ndev, size, ghost)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i == 1
        x[dev_id][i] = ghost
    end
    return nothing
end

function _multi_swap_2d_ghost_lr(dev_id, x, ndev, size, ghost)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i < size + 1
      x[dev_id][i, 1] = ghost[i]
    end
    return nothing
end

function _multi_swap_ghost_rl(dev_id, x, ndev, size, ghost)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i == 1
        x[dev_id][size] = ghost
    end
    return nothing
end

function _multi_swap_2d_ghost_rl(dev_id, x, ndev, size, col, ghost)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i < size + 1
      x[dev_id][i, col] = ghost[i]
    end
    return nothing
end

function _multi_parallel_for_cuda(N, dev_id, f, x...)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        f(dev_id, i, x...)
    end
    return nothing
end

function _multi_parallel_reduce_cuda(N, dev_id, ret, f, x...)
    shared_mem = @cuDynamicSharedMem(Float64, 512)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ti = threadIdx().x
    tmp::Float64 = 0.0
    shared_mem[ti] = 0.0

    if i <= N
        tmp = @inbounds f(dev_id, i, x...)
        shared_mem[threadIdx().x] = tmp
    end
    sync_threads()
    if (ti <= 256)
        shared_mem[ti] += shared_mem[ti + 256]
    end
    sync_threads()
    if (ti <= 128)
        shared_mem[ti] += shared_mem[ti + 128]
    end
    sync_threads()
    if (ti <= 64)
        shared_mem[ti] += shared_mem[ti + 64]
    end
    sync_threads()
    if (ti <= 32)
        shared_mem[ti] += shared_mem[ti + 32]
    end
    sync_threads()
    if (ti <= 16)
        shared_mem[ti] += shared_mem[ti + 16]
    end
    sync_threads()
    if (ti <= 8)
        shared_mem[ti] += shared_mem[ti + 8]
    end
    sync_threads()
    if (ti <= 4)
        shared_mem[ti] += shared_mem[ti + 4]
    end
    sync_threads()
    if (ti <= 2)
        shared_mem[ti] += shared_mem[ti + 2]
    end
    sync_threads()
    if (ti == 1)
        shared_mem[ti] += shared_mem[ti + 1]
        ret[blockIdx().x] = shared_mem[ti]
    end
    return nothing
end

function _multi_reduce_kernel_cuda(N, red, ret)
    shared_mem = @cuDynamicSharedMem(Float64, 512)
    i = threadIdx().x
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
    shared_mem[threadIdx().x] = tmp
    sync_threads()
    if (i <= 256)
        shared_mem[i] += shared_mem[i + 256]
    end
    sync_threads()
    if (i <= 128)
        shared_mem[i] += shared_mem[i + 128]
    end
    sync_threads()
    if (i <= 64)
        shared_mem[i] += shared_mem[i + 64]
    end
    sync_threads()
    if (i <= 32)
        shared_mem[i] += shared_mem[i + 32]
    end
    sync_threads()
    if (i <= 16)
        shared_mem[i] += shared_mem[i + 16]
    end
    sync_threads()
    if (i <= 8)
        shared_mem[i] += shared_mem[i + 8]
    end
    sync_threads()
    if (i <= 4)
        shared_mem[i] += shared_mem[i + 4]
    end
    sync_threads()
    if (i <= 2)
        shared_mem[i] += shared_mem[i + 2]
    end
    sync_threads()
    if (i == 1)
        shared_mem[i] += shared_mem[i + 1]
        ret[1] = shared_mem[1]
    end
    return nothing
end

function _multi_parallel_for_cuda_MN(M, N, dev_id, f, x...)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (i <= M) && (j <= N)
        f(dev_id, i, j, x...)
    end
    return nothing
end

function _multi_parallel_reduce_cuda_MN((M, N), dev_id, ret, f, x...)
    shared_mem = @cuDynamicSharedMem(Float64, 16*16)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ti = threadIdx().x
    tj = threadIdx().y
    bi = blockIdx().x
    bj = blockIdx().y

    tmp::Float64 = 0.0
    shared_mem[((ti - 1) * 16) + tj] = tmp

    if (i <= M && j <= N)
        tmp = @inbounds f(dev_id, i, j, x...)
        shared_mem[(ti - 1) * 16 + tj] = tmp
    end
    sync_threads()
    if (ti <= 8 && tj <= 8 && ti + 8 <= M && tj + 8 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + tj]
    end
    sync_threads()
    if (ti <= 4 && tj <= 4 && ti + 4 <= M && tj + 4 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + tj]
    end
    sync_threads()
    if (ti <= 2 && tj <= 2 && ti + 2 <= M && tj + 2 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + tj]
    end
    sync_threads()
    if (ti == 1 && tj == 1 && ti + 1 <= M && tj + 1 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + tj]
        ret[bi, bj] = shared_mem[((ti - 1) * 16) + tj]
    end
    return nothing
end

function _multi_reduce_kernel_cuda_MN((M, N), red, ret)
    shared_mem = @cuDynamicSharedMem(Float64, 16*16)
    i = threadIdx().x
    j = threadIdx().y
    ii = i
    jj = j

    tmp::Float64 = 0.0
    shared_mem[(i - 1) * 16 + j] = tmp

    if M > 16 && N > 16
        while ii <= M
            jj = threadIdx().y
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
    sync_threads()
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
    sync_threads()
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
    sync_threads()
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
    sync_threads()
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
