module AMDGPUExt

import Base: Callable
using JACC, AMDGPU
using AMDGPU: HIP

const AMDGPUBackend = ROCBackend

include("array.jl")
include("multi.jl")
include("experimental/experimental.jl")

JACC.get_backend(::Val{:amdgpu}) = AMDGPUBackend()

default_stream() = AMDGPU.stream()

JACC.default_stream(::Type{AMDGPUBackend}) = default_stream()

function JACC.synchronize(::AMDGPUBackend; stream = default_stream())
    AMDGPU.synchronize(stream)
end

@inline function max_shmem_size()
    return HIP.properties(AMDGPU.device()).sharedMemPerBlock
end

@inline kernel_args(args...) = rocconvert.((args))

function JACC.parallel_for(::AMDGPUBackend, N::Integer, f::Callable, x...)
    kernel = @roc launch=false _parallel_for_amdgpu(N, f, x...)
    config = AMDGPU.launch_configuration(kernel)
    threads = min(N, config.groupsize)
    blocks = cld(N, threads)
    shmem_size = max_shmem_size()
    kernel(
        N, f, x...; groupsize = threads, gridsize = blocks, shmem = shmem_size)
    AMDGPU.synchronize()
end

function JACC.parallel_for(
        spec::LaunchSpec{AMDGPUBackend}, N::Integer, f::Callable, x...)
    kernel = @roc launch=false _parallel_for_amdgpu(N, f, x...)
    if spec.threads == 0
        config = AMDGPU.launch_configuration(kernel)
        spec.threads = min(N, config.groupsize)
    end
    if spec.blocks == 0
        spec.blocks = cld(N, spec.threads)
    end
    if spec.shmem_size < 0
        spec.shmem_size = max_shmem_size()
    end
    kernel(
        N, f, x...; groupsize = spec.threads, gridsize = spec.blocks,
        shmem = spec.shmem_size, stream = spec.stream)
    if spec.sync
        AMDGPU.synchronize(spec.stream)
    end
end

abstract type BlockIndexer2D end

struct BlockIndexerBasic <: BlockIndexer2D end

function (blkIter::BlockIndexerBasic)(blockIdx, blockDim, threadIdx)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    return (i, j)
end

struct BlockIndexerSwapped <: BlockIndexer2D end

function (blkIter::BlockIndexerSwapped)(blockIdx, blockDim, threadIdx)
    j = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    i = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    return (i, j)
end

function JACC.parallel_for(
        ::AMDGPUBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    dev = AMDGPU.device()
    props = AMDGPU.HIP.properties(dev)
    maxBlocks = (x = props.maxGridSize[1], y = props.maxGridSize[2])
    indexer = BlockIndexerBasic()
    m, n = (M, N)
    if M < N && maxBlocks.x > maxBlocks.y
        indexer = BlockIndexerSwapped()
        m, n = (N, M)
    end

    kernel = @roc launch=false _parallel_for_amdgpu_MN(indexer, (M, N), f, x...)
    config = AMDGPU.launch_configuration(kernel)
    maxThreads = config.groupsize
    blockAttrs = (
        max_x = props.maxThreadsDim[1],
        max_y = props.maxThreadsDim[2],
        total = props.maxThreadsPerBlock
    )
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
    x_thr = fld(maxThreads, y_thr)
    threads = (x_thr, y_thr)
    blocks = (cld(m, x_thr), cld(n, y_thr))

    shmem_size = max_shmem_size()
    kernel(indexer, (M, N), f, x...; groupsize = threads,
        gridsize = blocks, shmem = shmem_size)
    AMDGPU.synchronize()
end

function JACC.parallel_for(
        spec::LaunchSpec{AMDGPUBackend}, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    dev = AMDGPU.device()
    props = AMDGPU.HIP.properties(dev)
    indexer = BlockIndexerBasic()
    m, n = (M, N)

    kernel = @roc launch=false _parallel_for_amdgpu_MN(indexer, (M, N), f, x...)
    config = AMDGPU.launch_configuration(kernel)

    if spec.threads == 0
        maxBlocks = (x = props.maxGridSize[1], y = props.maxGridSize[2])
        if M < N && maxBlocks.x > maxBlocks.y
            indexer = BlockIndexerSwapped()
            m, n = (N, M)
        end
        maxThreads = config.groupsize
        blockAttrs = (
            max_x = props.maxThreadsDim[1],
            max_y = props.maxThreadsDim[2],
            total = props.maxThreadsPerBlock
        )
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
        x_thr = fld(maxThreads, y_thr)
        spec.threads = (x_thr, y_thr)
    end

    if spec.blocks == 0
        spec.blocks = (cld(m, spec.threads[1]), cld(n, spec.threads[2]))
    end

    if spec.shmem_size < 0
        spec.shmem_size = max_shmem_size()
    end

    kernel(indexer, (M, N), f, x...; groupsize = spec.threads,
        gridsize = spec.blocks, shmem = spec.shmem_size, stream = spec.stream)
    if spec.sync
        AMDGPU.synchronize(spec.stream)
    end
end

function JACC.parallel_for(
        ::AMDGPUBackend, (L, M, N)::NTuple{3, Integer}, f::Callable, x...)
    numThreads = 32
    Lthreads = min(L, numThreads)
    Mthreads = min(M, numThreads)
    Nthreads = 1
    Lblocks = cld(L, Lthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N, Nthreads)
    shmem_size = max_shmem_size()
    @roc groupsize=(Lthreads, Mthreads, Nthreads) gridsize=(
        Lblocks, Mblocks, Nblocks) shmem=shmem_size _parallel_for_amdgpu_LMN(
        (L, M, N), f, x...)
    AMDGPU.synchronize()
end

function JACC.parallel_for(
        spec::LaunchSpec{AMDGPUBackend}, (L, M, N)::NTuple{3, Integer}, f::Callable,
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
    if spec.shmem_size < 0
        spec.shmem_size = max_shmem_size()
    end
    @roc groupsize=spec.threads gridsize=spec.blocks shmem=spec.shmem_size stream=spec.stream _parallel_for_amdgpu_LMN(
        (L, M, N), f, x...)
    if spec.sync
        AMDGPU.synchronize(spec.stream)
    end
end

mutable struct AMDGPUReduceWorkspace{T, TP <: JACC.WkProp} <:
               JACC.ReduceWorkspace
    tmp::AMDGPU.ROCArray{T}
    ret::AMDGPU.ROCArray{T}
end

function JACC.reduce_workspace(::AMDGPUBackend, init::T) where {T}
    AMDGPUReduceWorkspace{T, JACC.Managed}(
        AMDGPU.ROCArray{T}(undef, 0), AMDGPU.ROCArray([init]))
end

function JACC.reduce_workspace(::AMDGPUBackend, tmp::AMDGPU.ROCArray{T},
        init::AMDGPU.ROCArray{T}) where {T}
    AMDGPUReduceWorkspace{T, JACC.Unmanaged}(tmp, init)
end

@inline function _init!(wk::AMDGPUReduceWorkspace{T, JACC.Managed}, spec, init) where {T}
    if length(wk.tmp) != spec.blocks
        wk.tmp = AMDGPU.ROCArray{typeof(init)}(undef, spec.blocks)
    end
    fill!(wk.tmp, init)
    fill!(wk.ret, init)
    return nothing
end

@inline function _init!(wk::AMDGPUReduceWorkspace{T, JACC.Unmanaged}, spec, init) where {T}
    nothing
end

JACC.get_result(wk::AMDGPUReduceWorkspace) = Base.Array(wk.ret)[]

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{AMDGPUBackend},
        N::Integer, f::Callable, x...)
    wk = reducer.workspace
    op = reducer.op
    init = reducer.init

    kernel1 = @roc launch=false _parallel_reduce_amdgpu(
        N, op, wk.ret, f, x...)
    config1 = AMDGPU.launch_configuration(kernel1)
    threads1 = config1.groupsize

    kernel2 = @roc launch=false reduce_kernel_amdgpu(1, op, wk.ret, wk.ret)
    config2 = AMDGPU.launch_configuration(kernel2)
    threads2 = config2.groupsize

    spec = reducer.spec
    spec.threads = min(threads1, threads2, 512)
    spec.blocks = cld(N, spec.threads)
    spec.shmem_size = spec.threads * sizeof(init)

    _init!(wk, spec, init)

    kargs1 = kernel_args(N, op, wk.tmp, f, x...)
    kernel1(kargs1...; groupsize = spec.threads, gridsize = spec.blocks,
        shmem = spec.shmem_size, stream = spec.stream)

    kargs2 = kernel_args(spec.blocks, op, wk.tmp, wk.ret)
    kernel2(kargs2...; groupsize = spec.threads, gridsize = 1,
        shmem = spec.shmem_size, stream = spec.stream)

    if spec.sync
        AMDGPU.synchronize(spec.stream)
    end

    return nothing
end

function JACC.parallel_reduce(
        ::AMDGPUBackend, N::Integer, op, f::Callable, x...; init)
    ret_inst = AMDGPU.ROCArray{typeof(init)}(undef, 0)
    kernel1 = @roc launch=false _parallel_reduce_amdgpu(
        N, op, ret_inst, f, x...)
    config1 = AMDGPU.launch_configuration(kernel1)
    threads1 = config1.groupsize

    rret = AMDGPU.ROCArray([init])
    kernel2 = @roc launch=false reduce_kernel_amdgpu(1, op, ret_inst, rret)
    config2 = AMDGPU.launch_configuration(kernel2)
    threads2 = config2.groupsize

    threads = min(threads1, threads2, 512)
    blocks = cld(N, threads)

    shmem_size = threads * sizeof(init)

    ret = fill!(AMDGPU.ROCArray{typeof(init)}(undef, blocks), init)

    kargs1 = kernel_args(N, op, ret, f, x...)
    kernel1(kargs1...; groupsize = threads, gridsize = blocks,
        shmem = shmem_size)

    kargs2 = kernel_args(blocks, op, ret, rret)
    kernel2(kargs2...; groupsize = threads, gridsize = 1, shmem = shmem_size)
    AMDGPU.synchronize()

    return Base.Array(rret)[]
end

function JACC.parallel_reduce(
        spec::LaunchSpec{AMDGPUBackend}, N::Integer, op, f::Callable, x...; init)
    ret_inst = AMDGPU.ROCArray{typeof(init)}(undef, 0)
    kernel1 = @roc launch=false _parallel_reduce_amdgpu(
        N, op, ret_inst, f, x...)
    config1 = AMDGPU.launch_configuration(kernel1)
    threads1 = config1.groupsize

    rret = AMDGPU.ROCArray([init])
    kernel2 = @roc launch=false reduce_kernel_amdgpu(1, op, ret_inst, rret)
    config2 = AMDGPU.launch_configuration(kernel2)
    threads2 = config2.groupsize

    spec.threads = min(threads1, threads2, 512)
    spec.blocks = cld(N, spec.threads)
    spec.shmem_size = spec.threads * sizeof(init)

    ret = fill!(AMDGPU.ROCArray{typeof(init)}(undef, spec.blocks), init)

    kargs1 = kernel_args(N, op, ret, f, x...)
    kernel1(kargs1...; groupsize = spec.threads, gridsize = spec.blocks,
        shmem = spec.shmem_size, stream = spec.stream)

    kargs2 = kernel_args(spec.blocks, op, ret, rret)
    kernel2(kargs2...; groupsize = spec.threads, gridsize = 1,
        shmem = spec.shmem_size, stream = spec.stream)

    if spec.sync
        AMDGPU.synchronize(spec.stream)
    end

    return rret
end

function JACC._parallel_reduce!(
        reducer::JACC.ParallelReduce{AMDGPUBackend}, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    init = reducer.init
    op = reducer.op
    spec = reducer.spec
    numThreads = 16
    Mthreads = numThreads
    Nthreads = numThreads
    spec.threads = (Mthreads, Nthreads)
    Mblocks = cld(M, spec.threads[1])
    Nblocks = cld(N, spec.threads[2])
    spec.blocks = (Mblocks, Nblocks)
    spec.shmem_size = 16 * 16 * sizeof(init)

    wk = reducer.workspace
    _init!(wk, spec, init)

    kargs1 = kernel_args((M, N), op, wk.tmp, f, x...)
    kernel1 = @roc launch=false _parallel_reduce_amdgpu_MN(kargs1...)
    kernel1(kargs1...; groupsize = spec.threads, gridsize = spec.blocks,
        shmem = spec.shmem_size, stream = spec.stream)

    kargs2 = kernel_args(spec.blocks, op, wk.tmp, wk.ret)
    kernel2 = @roc launch=false reduce_kernel_amdgpu_MN(kargs2...)
    kernel2(kargs2...; groupsize = spec.threads, gridsize = (1, 1),
        shmem = spec.shmem_size, stream = spec.stream)

    if spec.sync
        AMDGPU.synchronize(spec.stream)
    end

    return nothing
end

function JACC.parallel_reduce(
        ::AMDGPUBackend, (M, N)::NTuple{2, Integer}, op, f::Callable, x...; init)
    numThreads = 16
    Mthreads = numThreads
    Nthreads = numThreads
    threads = (Mthreads, Nthreads)
    Mblocks = cld(M, Mthreads)
    Nblocks = cld(N, Nthreads)
    blocks = (Mblocks, Nblocks)
    shmem_size = 16 * 16 * sizeof(init)
    ret = fill!(AMDGPU.ROCArray{typeof(init)}(undef, blocks), init)
    rret = AMDGPU.ROCArray([init])

    kargs1 = kernel_args((M, N), op, ret, f, x...)
    kernel1 = @roc launch=false _parallel_reduce_amdgpu_MN(kargs1...)
    kernel1(kargs1...; groupsize = threads, gridsize = blocks, shmem = shmem_size)
    kargs2 = kernel_args(blocks, op, ret, rret)
    kernel2 = @roc launch=false reduce_kernel_amdgpu_MN(kargs2...)
    kernel2(kargs2...; groupsize = threads, gridsize = (1, 1), shmem = shmem_size)
    AMDGPU.synchronize()
    return Base.Array(rret)[]
end

function JACC.parallel_reduce(
        spec::LaunchSpec{AMDGPUBackend}, (M, N)::NTuple{2, Integer}, op, f::Callable, x...; init)
    numThreads = 16
    Mthreads = numThreads
    Nthreads = numThreads
    spec.threads = (Mthreads, Nthreads)
    Mblocks = cld(M, spec.threads[1])
    Nblocks = cld(N, spec.threads[2])
    spec.blocks = (Mblocks, Nblocks)
    spec.shmem_size = 16 * 16 * sizeof(init)

    ret = fill!(AMDGPU.ROCArray{typeof(init)}(undef, spec.blocks), init)
    rret = AMDGPU.ROCArray([init])

    kargs1 = kernel_args((M, N), op, ret, f, x...)
    kernel1 = @roc launch=false _parallel_reduce_amdgpu_MN(kargs1...)
    kernel1(kargs1...; groupsize = spec.threads, gridsize = spec.blocks,
        shmem = spec.shmem_size, stream = spec.stream)

    kargs2 = kernel_args(spec.blocks, op, ret, rret)
    kernel2 = @roc launch=false reduce_kernel_amdgpu_MN(kargs2...)
    kernel2(kargs2...; groupsize = spec.threads, gridsize = (1, 1),
        shmem = spec.shmem_size, stream = spec.stream)

    if spec.sync
        AMDGPU.synchronize(spec.stream)
    end

    return rret
end

function _parallel_for_amdgpu(N, f, x...)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    i > N && return nothing
    f(i, x...)
    return nothing
end

function _parallel_for_amdgpu_MN(indexer::BlockIndexer2D, (M, N), f, x...)
    i, j = indexer(workgroupIdx, workgroupDim, workitemIdx)
    i > M && return nothing
    j > N && return nothing
    f(i, j, x...)
    return nothing
end

function _parallel_for_amdgpu_LMN((L, M, N), f, x...)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    k = (workgroupIdx().z - 1) * workgroupDim().z + workitemIdx().z
    i > L && return nothing
    j > M && return nothing
    k > N && return nothing
    f(i, j, k, x...)
    return nothing
end

function _parallel_reduce_amdgpu(N, op, ret, f, x...)
    shmem_length = workgroupDim().x
    shared_mem = @ROCDynamicLocalArray(eltype(ret), shmem_length)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ti = workitemIdx().x
    shared_mem[ti] = ret[workgroupIdx().x]

    if i <= N
        tmp = @inbounds f(i, x...)
        shared_mem[ti] = tmp
    end
    AMDGPU.sync_workgroup()

    max_pwr = floor(Int, log2(shmem_length)) - 1
    for p in (max_pwr:-1:1)
        tn = 2^p
        if ti <= tn
            shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + tn])
        end
        AMDGPU.sync_workgroup()
    end

    if ti == 1
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 1])
        ret[workgroupIdx().x] = shared_mem[ti]
    end

    return nothing
end

function reduce_kernel_amdgpu(N, op, red, ret)
    shmem_length = workgroupDim().x
    shared_mem = @ROCDynamicLocalArray(eltype(ret), shmem_length)
    i = workitemIdx().x
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
    AMDGPU.sync_workgroup()

    max_pwr = floor(Int, log2(shmem_length)) - 1
    for p in (max_pwr:-1:1)
        tn = 2^p
        if i <= tn
            shared_mem[i] = op(shared_mem[i], shared_mem[i + tn])
        end
        AMDGPU.sync_workgroup()
    end

    if i == 1
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 1])
        ret[1] = shared_mem[1]
    end

    return nothing
end

function _parallel_reduce_amdgpu_MN((M, N), op, ret, f, x...)
    shared_mem = @ROCDynamicLocalArray(eltype(ret), 256, false)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    ti = workitemIdx().x
    tj = workitemIdx().y
    bi = workgroupIdx().x
    bj = workgroupIdx().y

    sid = ((ti - 1) * 16) + tj
    shared_mem[sid] = ret[bi, bj]

    if (i <= M && j <= N)
        tmp = @inbounds f(i, j, x...)
        shared_mem[sid] = tmp
    end
    AMDGPU.sync_workgroup()
    if (ti <= 8 && tj <= 8)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 7) * 16) + (tj + 8)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 8)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 7) * 16) + tj])
    end
    AMDGPU.sync_workgroup()
    if (ti <= 4 && tj <= 4)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 3) * 16) + (tj + 4)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 4)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 3) * 16) + tj])
    end
    AMDGPU.sync_workgroup()
    if (ti <= 2 && tj <= 2)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 1) * 16) + (tj + 2)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 2)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 1) * 16) + tj])
    end
    AMDGPU.sync_workgroup()
    if (ti == 1 && tj == 1)
        shared_mem[sid] = op(shared_mem[sid], shared_mem[ti * 16 + (tj + 1)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 1)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[ti * 16 + tj])
        ret[bi, bj] = shared_mem[sid]
    end
    return nothing
end

function reduce_kernel_amdgpu_MN((M, N), op, red, ret)
    shared_mem = @ROCDynamicLocalArray(eltype(ret), 256)
    i = workitemIdx().x
    j = workitemIdx().y

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
    AMDGPU.sync_workgroup()
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
    AMDGPU.sync_workgroup()
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
    AMDGPU.sync_workgroup()
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
    AMDGPU.sync_workgroup()
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

function JACC.shared(::AMDGPUBackend, x::AbstractArray)
    size = length(x)
    shmem = @ROCDynamicLocalArray(eltype(x), size)
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
                @inbounds shmem[ind] = x[i_local, j_local]
            end
        end
    else
        if workgroupDim().y == 1
            ind = workgroupIdx().x
            for i in (workgroupDim().x):(workgroupDim().x):size
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
                    @inbounds shmem[ind] = x[i_local, j_local]
                    ind += num_threads
                end
            end
        end
    end
    AMDGPU.sync_workgroup()
    return shmem
end

JACC.sync_workgroup(::AMDGPUBackend) = AMDGPU.sync_workgroup()

JACC.array_type(::AMDGPUBackend) = AMDGPU.ROCArray

JACC.array(::AMDGPUBackend, x::AbstractArray) = AMDGPU.ROCArray(x)

end # module AMDGPUExt
