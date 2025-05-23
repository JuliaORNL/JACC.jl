module Async

import Base: Callable
using JACC, CUDA

function JACC.Async.ndev(::CUDABackend)
    return length(CUDA.devices())
end

function JACC.Async.synchronize(::CUDABackend)
    ndev = length(devices())
    for i in 1:ndev
        CUDA.device!(i - 1)
        CUDA.synchronize()
    end
    #CUDA.device!(0)
end

function JACC.Async.synchronize(::CUDABackend, id::Integer)
    ndev = length(CUDA.devices())
    local_id = (id - 1) % ndev
    CUDA.device!(local_id)
    CUDA.synchronize()
    #CUDA.device!(0)
end

function JACC.Async.array(::CUDABackend, id::Integer, x::Base.Array{T, N}) where {T, N}
    ndev = length(CUDA.devices())
    local_id = (id - 1) % ndev
    #if (local_id <= 0) 
    CUDA.device!(local_id)
    return JACC.array(x)
    #CUDA.device!(0)
end

function JACC.Async.copy(::CUDABackend, id_dest::Integer, x::CUDA.CuArray{T, N}, id_orig::Integer, y::CUDA.CuArray{T, N}) where {T, N}
    ndev = length(CUDA.devices())
    local_id_orig = (id_orig - 1) % ndev
    local_id_dest = (id_dest - 1) % ndev
    
    CUDA.device!(local_id_orig)
    hy = Base.Array(y)    
    
    CUDA.device!(local_id_dest)
    x = CUDA.CuArray(hy)
    
    #CUDA.device!(0)
end

function JACC.Async.parallel_for(::CUDABackend, id::Integer, N::Integer, f::Callable, x...)
    ndev = length(CUDA.devices())
    local_id = (id - 1) % ndev
    CUDA.device!(local_id)
    
    numThreads = 256
    threads = min(N, numThreads)
    blocks = ceil(Int, N / threads)

    @cuda threads=threads blocks=blocks _async_parallel_for_cuda(N, f, x...)
    
    #CUDA.device!(0)
end

function JACC.Async.parallel_reduce(
        ::CUDABackend, id::Integer, N::Integer, f::Callable, x...)
    ndev = length(CUDA.devices())
    local_id = (id - 1) % ndev
    CUDA.device!(local_id)
    
    numThreads = 512
    threads = min(N, numThreads)
    blocks = ceil(Int, N / threads)
    
    ret = CUDA.zeros(Float64, blocks)
    rret = CUDA.zeros(Float64, 1)
        
    @cuda threads=threads blocks=blocks shmem=512 * sizeof(Float64) _async_parallel_reduce_cuda(
        N, ret, f, x...)
    @cuda threads=threads blocks=1 shmem=512 * sizeof(Float64) _async_reduce_kernel_cuda(
        blocks, ret, rret)

    #CUDA.device!(0)

    return rret
end

function JACC.Async.parallel_for(::CUDABackend,
        id::Integer, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    ndev = length(CUDA.devices())
    local_id = (id - 1) % ndev
    CUDA.device!(local_id)
    
    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N, numThreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nblocks = ceil(Int, N / Nthreads)

    @cuda threads=(Mthreads, Nthreads) blocks=(Mblocks, Nblocks) _async_parallel_for_cuda_MN(
            M, N, f, x...)

    #CUDA.device!(0)
end

function JACC.Async.parallel_reduce(::CUDABackend,
        id::Integer, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    ndev = length(CUDA.devices())
    local_id = (id - 1) % ndev
    CUDA.device!(local_id)

    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N, numThreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nblocks = ceil(Int, N / Nthreads)
    
    ret = CUDA.zeros(Float64, (Mblocks, Nblocks))
    rret = CUDA.zeros(Float64, 1)

    @cuda threads=(Mthreads, Nthreads) blocks=(Mblocks, Nblocks) shmem=16 * 16 *
                                                                      sizeof(Float64) _async_parallel_reduce_cuda_MN(
            (M, N), dev, ret[i], f, x...)
    @cuda threads=(Mthreads, Nthreads) blocks=(1, 1) shmem=16 * 16 *
                                                          sizeof(Float64) _async_reduce_kernel_cuda_MN(
            (Mblocks, Nblocks), ret, rret)
    
    #CUDA.device!(0)

    return rret
end

function _async_parallel_for_cuda(N, f, x...)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        f(i, x...)
    end
    return nothing
end

function _async_parallel_reduce_cuda(N, ret, f, x...)
    shared_mem = @cuDynamicSharedMem(Float64, 512)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ti = threadIdx().x
    tmp::Float64 = 0.0
    shared_mem[ti] = 0.0

    if i <= N
        tmp = @inbounds f(i, x...)
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

function _async_reduce_kernel_cuda(N, red, ret)
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

function _async_parallel_for_cuda_MN(M, N, f, x...)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (i <= M) && (j <= N)
        f(i, j, x...)
    end
    return nothing
end

function _async_parallel_reduce_cuda_MN((M, N), ret, f, x...)
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
        tmp = @inbounds f(i, j, x...)
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

function _async_reduce_kernel_cuda_MN((M, N), red, ret)
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

end # module Async
