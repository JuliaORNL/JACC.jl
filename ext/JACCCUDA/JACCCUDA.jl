module JACCCUDA

using JACC, CUDA

# overloaded array functions
include("array.jl")

# overloaded experimental functions
include("JACCEXPERIMENTAL.jl")
using .experimental

function JACC.parallel_for(N::I, f::F, x...) where {I <: Integer, F <: Function}
    parallel_args = (N, f, x...)
    parallel_kargs = cudaconvert.(parallel_args)
    parallel_tt = Tuple{Core.Typeof.(parallel_kargs)...}
    parallel_kernel = cufunction(_parallel_for_cuda, parallel_tt)
    maxThreads = CUDA.maxthreads(parallel_kernel)
    threads = min(N, maxThreads)
    blocks = ceil(Int, N / threads)
    parallel_kernel(parallel_kargs...; threads = threads, blocks = blocks)
end

abstract type BlockIndexer2D end

struct BlockIndexerBasic <: BlockIndexer2D end

function (blkIter::BlockIndexerBasic)(blockIdx, blockDim, threadIdx)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    return (i, j)
end

struct BlockIndexerSwapped <: BlockIndexer2D end

function (blkIter::BlockIndexerSwapped)(blockIdx, blockDim, threadIdx)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    return (i, j)
end

function JACC.parallel_for(
        (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    #To use JACC.shared, it is recommended to use a high number of threads per block to maximize the
    # potential benefit from using shared memory.
    #numThreads = 32

    dev = CUDA.device()
    maxBlocks = (
        x = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
        y = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),
    )
    indexer = BlockIndexerBasic()
    m, n = (M, N)
    if M < N && maxBlocks.x > maxBlocks.y
        indexer = BlockIndexerSwapped()
        m, n = (N, M)
    end

    parallel_args = (indexer, (M, N), f, x...)
    parallel_kargs = cudaconvert.(parallel_args)
    parallel_tt = Tuple{Core.Typeof.(parallel_kargs)...}
    parallel_kernel = cufunction(_parallel_for_cuda_MN, parallel_tt)
    maxThreads = CUDA.maxthreads(parallel_kernel)
    blockAttrs = (
        max_x = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
        max_y = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
        total = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
    x_thr = min(
        blockAttrs.max_x,
        nextpow(2, m / blockAttrs.total + 1),
        blockAttrs.total,
        maxThreads
    )
    y_thr = min(
        blockAttrs.max_y,
        ceil(Int, blockAttrs.total / x_thr),
        ceil(Int, maxThreads / x_thr),
    )
    threads = (x_thr, y_thr)
    blocks = (
        ceil(Int, (m - 1) / x_thr + 1),
        ceil(Int, (n - 1) / y_thr + 1),
    )

    parallel_kernel(parallel_kargs...; threads = threads, blocks = blocks)

    # To use JACC.shared, we need to define shmem size using the dynamic shared memory API. The size should be the biggest size of shared memory available for the GPU
    #CUDA.@sync @cuda threads=(Mthreads, Nthreads) blocks=(Mblocks, Nblocks) shmem = 4 * numThreads * numThreads * sizeof(Float64) _parallel_for_cuda_MN(
    #    f, x...)
end

function JACC.parallel_reduce(
        N::I, f::F, x...) where {I <: Integer, F <: Function}
    numThreads = 512
    threads = min(N, numThreads)
    blocks = ceil(Int, N / threads)
    ret = CUDA.zeros(Float64, blocks)
    rret = CUDA.zeros(Float64, 1)
    CUDA.@sync @cuda threads=threads blocks=blocks shmem=512 * sizeof(Float64) _parallel_reduce_cuda(
        N, ret, f, x...)
    CUDA.@sync @cuda threads=threads blocks=1 shmem=512 * sizeof(Float64) reduce_kernel_cuda(
        blocks, ret, rret)
    return rret
end

function JACC.parallel_reduce(
        (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N, numThreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nblocks = ceil(Int, N / Nthreads)
    ret = CUDA.zeros(Float64, (Mblocks, Nblocks))
    rret = CUDA.zeros(Float64, 1)
    CUDA.@sync @cuda threads=(Mthreads, Nthreads) blocks=(Mblocks, Nblocks) shmem=16 *
                                                                                  16 *
                                                                                  sizeof(Float64) _parallel_reduce_cuda_MN(
        (M, N), ret, f, x...)
    CUDA.@sync @cuda threads=(Mthreads, Nthreads) blocks=(1, 1) shmem=16 * 16 *
                                                                      sizeof(Float64) reduce_kernel_cuda_MN(
        (Mblocks, Nblocks), ret, rret)
    return rret
end

function _parallel_for_cuda(N, f, x...)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > N && return
    f(i, x...)
    return nothing
end

function _parallel_for_cuda_MN(indexer::T, (M, N), f, x...) where {T<:BlockIndexer2D}
    i, j = indexer(blockIdx, blockDim, threadIdx)
    i > M && return
    j > N && return
    f(i, j, x...)
    return nothing
end

function _parallel_reduce_cuda(N, ret, f, x...)
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

function reduce_kernel_cuda(N, red, ret)
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

function _parallel_reduce_cuda_MN((M, N), ret, f, x...)
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

function reduce_kernel_cuda_MN((M, N), red, ret)
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

function __init__()
    const JACC.Array = CUDA.CuArray{T, N} where {T, N}
end

end # module JACCCUDA
