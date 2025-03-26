module CUDAExt

import Base: Callable
using JACC, CUDA

# overloaded array functions
include("array.jl")

include("multi.jl")
using .Multi

# overloaded experimental functions
include("experimental/experimental.jl")
using .Experimental

JACC.get_backend(::Val{:cuda}) = CUDABackend()

default_stream() = CUDA.stream()

JACC.default_stream(::Type{CUDABackend}) = default_stream()

function JACC.synchronize(::CUDABackend; stream = default_stream())
    CUDA.synchronize(stream)
end

@inline kernel_args(args...) = cudaconvert.((args))

@inline function kernel_maxthreads(kernel_function, kargs)
    p_tt = Tuple{Core.Typeof.(kargs)...}
    p_kernel = cufunction(kernel_function, p_tt)
    maxThreads = CUDA.maxthreads(p_kernel)
    return (p_kernel, CUDA.maxthreads(p_kernel))
end

function JACC.parallel_for(::CUDABackend, N::Integer, f::Callable, x...)
    kargs = kernel_args(N, f, x...)
    kernel, maxThreads = kernel_maxthreads(_parallel_for_cuda, kargs)
    threads = min(N, maxThreads)
    blocks = cld(N, threads)
    shmem_size = 2 * threads * sizeof(Float64)
    CUDA.@sync kernel(
        kargs...; threads = threads, blocks = blocks, shmem = shmem_size)
end

function JACC.parallel_for(
        spec::LaunchSpec{CUDABackend}, N::Integer, f::Callable, x...)
    kargs = kernel_args(N, f, x...)
    kernel, maxThreads = kernel_maxthreads(_parallel_for_cuda, kargs)
    if spec.threads == 0
        spec.threads = min(N, maxThreads)
    end
    if spec.blocks == 0
        spec.blocks = cld(N, spec.threads)
    end
    if spec.shmem_size == 0
        spec.shmem_size = 2 * spec.threads * sizeof(Float64)
    end
    kernel(kargs...; threads = spec.threads, blocks = spec.blocks,
        shmem = spec.shmem_size, stream = spec.stream)
    if spec.sync
        CUDA.synchronize(spec.stream)
    end
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
        ::CUDABackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    #To use JACC.shared, it is recommended to use a high number of threads per block to maximize the
    # potential benefit from using shared memory.

    dev = CUDA.device()
    maxBlocks = (
        x = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
        y = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    )
    indexer = BlockIndexerBasic()
    m, n = (M, N)
    if M < N && maxBlocks.x > maxBlocks.y
        indexer = BlockIndexerSwapped()
        m, n = (N, M)
    end

    kargs = kernel_args(indexer, (M, N), f, x...)
    kernel, maxThreads = kernel_maxthreads(_parallel_for_cuda_MN, kargs)
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
        cld(blockAttrs.total, x_thr),
        cld(maxThreads, x_thr)
    )
    threads = (x_thr, y_thr)
    blocks = (cld(m, x_thr), cld(n, y_thr))

    shmem_size = 2 * x_thr * y_thr * sizeof(Float64)

    CUDA.@sync kernel(
        kargs...; threads = threads, blocks = blocks, shmem = shmem_size)
end

function JACC.parallel_for(
        spec::LaunchSpec{CUDABackend}, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    dev = CUDA.device()
    indexer = BlockIndexerBasic()
    m, n = (M, N)

    kargs = kernel_args(indexer, (M, N), f, x...)
    kernel, maxThreads = kernel_maxthreads(_parallel_for_cuda_MN, kargs)

    if spec.threads == 0
        maxBlocks = (
            x = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
            y = attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
        )
        if M < N && maxBlocks.x > maxBlocks.y
            indexer = BlockIndexerSwapped()
            m, n = (N, M)
        end
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
            cld(blockAttrs.total, x_thr),
            cld(maxThreads, x_thr)
        )
        spec.threads = (x_thr, y_thr)
    end

    if spec.blocks == 0
        spec.blocks = (cld(m, spec.threads[1]), cld(n, spec.threads[2]))
    end

    if spec.shmem_size == 0
        spec.shmem_size = 2 * spec.threads[1] * spec.threads[2] *
                          sizeof(Float64)
    end

    kernel(kargs...; threads = spec.threads, blocks = spec.blocks,
        shmem = spec.shmem_size, stream = spec.stream)
    if spec.sync
        CUDA.synchronize(spec.stream)
    end
end

function JACC.parallel_for(
        ::CUDABackend, (L, M, N)::NTuple{3, Integer}, f::Callable, x...)
    #To use JACC.shared, it is recommended to use a high number of threads per block to maximize the
    # potential benefit from using shared memory.
    numThreads = 32
    Lthreads = min(L, numThreads)
    Mthreads = min(M, numThreads)
    Nthreads = 1
    Lblocks = cld(L, Lthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N, Nthreads)
    shmem_size = 2 * Lthreads * Mthreads * Nthreads * sizeof(Float64)
    CUDA.@sync @cuda threads=(Lthreads, Mthreads, Nthreads) blocks=(
        Lblocks, Mblocks, Nblocks) shmem=shmem_size _parallel_for_cuda_LMN(
        (L, M, N), f, x...)
end

function JACC.parallel_for(
        spec::LaunchSpec{CUDABackend}, (L, M, N)::NTuple{3, Integer}, f::Callable,
        x...)
    if spec.threads == 0
        numThreads = 32
        Lthreads = min(L, numThreads)
        Mthreads = min(M, numThreads)
        Nthreads = 1
        spec.threads = (Lthreads, Mthreads, Nthreads)
    end
    if spec.blocks == 0
        Lblocks = cld(L, spec.threads[1])
        Mblocks = cld(M, spec.threads[2])
        Nblocks = cld(N, spec.threads[3])
        spec.blocks = (Lblocks, Mblocks, Nblocks)
    end
    if spec.shmem_size == 0
        spec.shmem_size = 2 * spec.threads[1] * spec.threads[2] *
                          spec.threads[3] * sizeof(Float64)
    end
    @cuda threads=spec.threads blocks=spec.blocks shmem=spec.shmem_size stream=spec.stream _parallel_for_cuda_LMN(
        (L, M, N), f, x...)
    if spec.sync
        CUDA.synchronize(spec.stream)
    end
end

function JACC.parallel_reduce(
        ::CUDABackend, N::Integer, op, f::Callable, x...; init)
    ret_inst = CUDA.CuArray{typeof(init)}(undef, 0)

    kargs_1 = kernel_args(N, op, ret_inst, f, x...)
    kernel_1, maxThreads_1 = kernel_maxthreads(_parallel_reduce_cuda, kargs_1)

    rret = CUDA.CuArray([init])
    kargs_2 = kernel_args(1, op, ret_inst, rret)
    kernel_2, maxThreads_2 = kernel_maxthreads(reduce_kernel_cuda, kargs_2)

    threads = min(maxThreads_1, maxThreads_2, 512)
    blocks = cld(N, threads)

    shmem_size = threads * sizeof(init)

    ret = fill!(CUDA.CuArray{typeof(init)}(undef, blocks), init)
    kargs = kernel_args(N, op, ret, f, x...)
    kernel_1(kargs...; threads = threads, blocks = blocks, shmem = shmem_size)

    kargs = kernel_args(blocks, op, ret, rret)
    kernel_2(kargs...; threads = threads, blocks = 1, shmem = shmem_size)

    return Base.Array(rret)[]
end

function JACC.parallel_reduce(
        spec::LaunchSpec{CUDABackend}, N::Integer, op, f::Callable, x...; init)
    ret_inst = CUDA.CuArray{typeof(init)}(undef, 0)

    kargs_1 = kernel_args(N, op, ret_inst, f, x...)
    kernel_1, maxThreads_1 = kernel_maxthreads(_parallel_reduce_cuda, kargs_1)

    rret = CUDA.CuArray([init])
    kargs_2 = kernel_args(1, op, ret_inst, rret)
    kernel_2, maxThreads_2 = kernel_maxthreads(reduce_kernel_cuda, kargs_2)

    if spec.threads != 0
        @warn "JACC.parallel_reduce: Ignoring threads spec: $(spec.threads)"
    end
    spec.threads = min(maxThreads_1, maxThreads_2, 512)
    if spec.blocks != 0
        @warn "JACC.parallel_reduce: Ignoring blocks spec: $(spec.blocks)"
    end
    spec.blocks = cld(N, spec.threads)

    if spec.shmem_size != 0
        @warn "JACC.parallel_reduce: Ignoring shmem_size spec: $(spec.shmem_size)"
    end
    spec.shmem_size = spec.threads * sizeof(init)

    ret = fill!(CUDA.CuArray{typeof(init)}(undef, spec.blocks), init)
    kargs = kernel_args(N, op, ret, f, x...)
    kernel_1(kargs...; threads = spec.threads, blocks = spec.blocks,
        shmem = spec.shmem_size, stream = spec.stream)

    kargs = kernel_args(spec.blocks, op, ret, rret)
    kernel_2(kargs...; threads = spec.threads, blocks = 1,
        shmem = spec.shmem_size, stream = spec.stream)

    if spec.sync
        CUDA.synchronize(spec.stream)
    end

    return rret
end

function JACC.parallel_reduce(
        ::CUDABackend, (M, N)::Tuple{Integer, Integer}, op, f::Callable, x...; init)
    numThreads = 16
    Mthreads = numThreads
    Nthreads = numThreads
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N, Nthreads)
    ret = fill!(CUDA.CuArray{typeof(init)}(undef, (Mblocks, Nblocks)), init)
    rret = CUDA.CuArray([init])
    shmem_size = 16 * 16 * sizeof(init)
    CUDA.@sync @cuda threads=(Mthreads, Nthreads) blocks=(Mblocks, Nblocks) shmem=shmem_size _parallel_reduce_cuda_MN(
        (M, N), op, ret, f, x...)
    CUDA.@sync @cuda threads=(Mthreads, Nthreads) blocks=(1, 1) shmem=shmem_size reduce_kernel_cuda_MN(
        (Mblocks, Nblocks), op, ret, rret)
    return Base.Array(rret)[]
end

function JACC.parallel_reduce(
        spec::LaunchSpec{CUDABackend}, (M, N)::Tuple{Integer, Integer}, op, f::Callable, x...; init)
    if spec.threads != 0
        @warn "JACC.parallel_reduce: Ignoring threads spec: $(spec.threads)"
    end
    numThreads = 16
    Mthreads = numThreads
    Nthreads = numThreads
    spec.threads = (Mthreads, Nthreads)
    if spec.blocks != 0
        @warn "JACC.parallel_reduce: Ignoring blocks spec: $(spec.blocks)"
    end
    Mblocks = cld(M, spec.threads[1])
    Nblocks = cld(N, spec.threads[2])
    spec.blocks = (Mblocks, Nblocks)
    if spec.shmem_size != 0
        @warn "JACC.parallel_reduce: Ignoring shmem_size spec: $(spec.shmem_size)"
    end
    spec.shmem_size = 16 * 16 * sizeof(init)

    ret = fill!(CUDA.CuArray{typeof(init)}(undef, spec.blocks), init)
    rret = CUDA.CuArray([init])
    @cuda threads=spec.threads blocks=spec.blocks shmem=spec.shmem_size stream=spec.stream _parallel_reduce_cuda_MN(
        (M, N), op, ret, f, x...)
    @cuda threads=spec.threads blocks=(1, 1) shmem=spec.shmem_size stream=spec.stream reduce_kernel_cuda_MN(
        spec.blocks, op, ret, rret)

    if spec.sync
        CUDA.synchronize(spec.stream)
    end

    return rret
end

function _parallel_for_cuda(N, f, x...)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > N && return nothing
    f(i, x...)
    return nothing
end

function _parallel_for_cuda_MN(indexer::BlockIndexer2D, (M, N), f, x...)
    i, j = indexer(blockIdx, blockDim, threadIdx)
    i > M && return nothing
    j > N && return nothing
    f(i, j, x...)
    return nothing
end

function _parallel_for_cuda_LMN((L, M, N), f, x...)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    i > L && return nothing
    j > M && return nothing
    k > N && return nothing
    f(i, j, k, x...)
    return nothing
end

function _parallel_reduce_cuda(N, op, ret, f, x...)
    shmem_length = blockDim().x
    shared_mem = CuDynamicSharedArray(eltype(ret), shmem_length)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ti = threadIdx().x
    shared_mem[ti] = ret[blockIdx().x]

    if i <= N
        tmp = @inbounds f(i, x...)
        shared_mem[ti] = tmp
    end
    sync_threads()

    max_pwr = floor(Int, log2(shmem_length)) - 1
    for p in (max_pwr:-1:1)
        tn = 2^p
        if (ti <= tn)
            shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + tn])
        end
        sync_threads()
    end

    if ti == 1
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 1])
        ret[blockIdx().x] = shared_mem[ti]
    end

    return nothing
end

function reduce_kernel_cuda(N, op, red, ret)
    shmem_length = blockDim().x
    shared_mem = CuDynamicSharedArray(eltype(ret), shmem_length)
    i = threadIdx().x
    ii = i
    tmp = ret[1]
    if N > shmem_length
        for ii in i:shmem_length:N
            tmp = op(tmp, @inbounds red[ii])
        end
    elseif (i <= N)
        tmp = @inbounds red[i]
    end
    shared_mem[i] = tmp
    sync_threads()

    max_pwr = floor(Int, log2(shmem_length)) - 1
    for p in (max_pwr:-1:1)
        tn = 2^p
        if i <= tn
            shared_mem[i] = op(shared_mem[i], shared_mem[i + tn])
        end
        sync_threads()
    end

    if i == 1
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 1])
        ret[1] = shared_mem[1]
    end

    return nothing
end

function _parallel_reduce_cuda_MN((M, N), op, ret, f, x...)
    shared_mem = CuDynamicSharedArray(eltype(ret), 16 * 16)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ti = threadIdx().x
    tj = threadIdx().y
    bi = blockIdx().x
    bj = blockIdx().y

    sid = ((ti - 1) * 16) + tj
    shared_mem[sid] = ret[bi, bj]

    if (i <= M && j <= N)
        tmp = @inbounds f(i, j, x...)
        shared_mem[sid] = tmp
    end
    sync_threads()
    if (ti <= 8 && tj <= 8 && ti + 8 <= M && tj + 8 <= N)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 7) * 16) + (tj + 8)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 8)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 7) * 16) + tj])
    end
    sync_threads()
    if (ti <= 4 && tj <= 4 && ti + 4 <= M && tj + 4 <= N)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 3) * 16) + (tj + 4)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 4)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 3) * 16) + tj])
    end
    sync_threads()
    if (ti <= 2 && tj <= 2 && ti + 2 <= M && tj + 2 <= N)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 1) * 16) + (tj + 2)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 2)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 1) * 16) + tj])
    end
    sync_threads()
    if (ti == 1 && tj == 1 && ti + 1 <= M && tj + 1 <= N)
        shared_mem[sid] = op(shared_mem[sid], shared_mem[ti * 16 + (tj + 1)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 1)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[ti * 16 + tj])
        ret[bi, bj] = shared_mem[sid]
    end
    return nothing
end

function reduce_kernel_cuda_MN((M, N), op, red, ret)
    shared_mem = CuDynamicSharedArray(eltype(ret), 16 * 16)
    i = threadIdx().x
    j = threadIdx().y

    tmp = ret[1]
    sid = ((i - 1) * 16) + j
    shared_mem[sid] = tmp

    if M > 16 && N > 16
        for ii in i:16:M
            for jj in j:16:N
                tmp = op(tmp, @inbounds red[ii, jj])
            end
        end
    elseif M > 16
        for ii in i:16:M
            tmp = op(tmp, @inbounds red[ii, j])
        end
    elseif N > 16
        for jj in j:16:N
            tmp = op(tmp, @inbounds red[i, jj])
        end
    elseif M <= 16 && N <= 16
        if i <= M && j <= N
            tmp = op(tmp, @inbounds red[i, j])
        end
    end
    shared_mem[sid] = tmp
    sync_threads()
    if (i <= 8 && j <= 8)
        if (i + 8 <= M && j + 8 <= N)
            shared_mem[sid] = op(
                shared_mem[sid], shared_mem[((i + 7) * 16) + (j + 8)])
        end
        if (i <= M && j + 8 <= N)
            shared_mem[sid] = op(
                shared_mem[sid], shared_mem[((i - 1) * 16) + (j + 8)])
        end
        if (i + 8 <= M && j <= N)
            shared_mem[sid] = op(
                shared_mem[sid], shared_mem[((i + 7) * 16) + j])
        end
    end
    sync_threads()
    if (i <= 4 && j <= 4)
        if (i + 4 <= M && j + 4 <= N)
            shared_mem[sid] = op(
                shared_mem[sid], shared_mem[((i + 3) * 16) + (j + 4)])
        end
        if (i <= M && j + 4 <= N)
            shared_mem[sid] = op(
                shared_mem[sid], shared_mem[((i - 1) * 16) + (j + 4)])
        end
        if (i + 4 <= M && j <= N)
            shared_mem[sid] = op(
                shared_mem[sid], shared_mem[((i + 3) * 16) + j])
        end
    end
    sync_threads()
    if (i <= 2 && j <= 2)
        if (i + 2 <= M && j + 2 <= N)
            shared_mem[sid] = op(
                shared_mem[sid], shared_mem[((i + 1) * 16) + (j + 2)])
        end
        if (i <= M && j + 2 <= N)
            shared_mem[sid] = op(
                shared_mem[sid], shared_mem[((i - 1) * 16) + (j + 2)])
        end
        if (i + 2 <= M && j <= N)
            shared_mem[sid] = op(
                shared_mem[sid], shared_mem[((i + 1) * 16) + j])
        end
    end
    sync_threads()
    if (i == 1 && j == 1)
        if (i + 1 <= M && j + 1 <= N)
            shared_mem[sid] = op(shared_mem[sid], shared_mem[i * 16 + (j + 1)])
        end
        if (i <= M && j + 1 <= N)
            shared_mem[sid] = op(
                shared_mem[sid], shared_mem[((i - 1) * 16) + (j + 1)])
        end
        if (i + 1 <= M && j <= N)
            shared_mem[sid] = op(shared_mem[sid], shared_mem[i * 16 + j])
        end
        ret[1] = shared_mem[sid]
    end
    return nothing
end

function JACC.shared(x::CuDeviceArray{T, N}) where {T, N}
    size = length(x)
    shmem = CuDynamicSharedArray(T, size)
    num_threads = blockDim().x * blockDim().y
    if (size <= num_threads)
        if blockDim().y == 1
            ind = threadIdx().x
            #if (ind <= size)
            @inbounds shmem[ind] = x[ind]
            #end
        else
            i_local = threadIdx().x
            j_local = threadIdx().y
            ind = (i_local - 1) * blockDim().x + j_local
            if ndims(x) == 1
                #if (ind <= size)
                @inbounds shmem[ind] = x[ind]
                #end
            elseif ndims(x) == 2
                #if (ind <= size)
                @inbounds shmem[ind] = x[i_local, j_local]
                #end
            end
        end
    else
        if blockDim().y == 1
            ind = threadIdx().x
            for i in (blockDim().x):(blockDim().x):size
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
                    @inbounds shmem[ind] = x[i_local, j_local]
                    ind += num_threads
                end
            end
        end
    end
    sync_threads()
    return shmem
end

JACC.array_type(::CUDABackend) = CUDA.CuArray

JACC.array(::CUDABackend, x::Base.Array) = CUDA.CuArray(x)

end # module CUDAExt
