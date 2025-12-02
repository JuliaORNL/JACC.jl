module oneAPIExt

using JACC, oneAPI, oneAPI.oneL0

# overloaded array functions
include("array.jl")
include("async.jl")
include("experimental/experimental.jl")

JACC.get_backend(::Val{:oneapi}) = oneAPIBackend()

default_stream() = oneAPI.global_queue(oneAPI.context(), oneAPI.device())

JACC.default_stream(::Type{oneAPIBackend}) = default_stream()

function JACC.synchronize(::oneAPIBackend; stream = default_stream())
    oneAPI.synchronize(stream)
end

@inline kernel_args(args...) = kernel_convert.((args))

function JACC.parallel_for(f, ::oneAPIBackend, N::Integer, x...)
    kernel = @oneapi launch=false _parallel_for_oneapi(N, f, x...)
    config_items = div(oneAPI.launch_configuration(kernel), 2)
    items = min(N, config_items, 256)
    groups = cld(N, items)
    kernel(N, f, x...; items = items, groups = groups)
    oneAPI.synchronize()
end

function JACC.parallel_for(
        f, spec::LaunchSpec{oneAPIBackend}, N::Integer, x...)
    kernel = @oneapi launch=false _parallel_for_oneapi(N, f, x...)
    if spec.threads == 0
        maxItems = oneAPI.launch_configuration(kernel)
        spec.threads = min(N, maxItems)
    end
    if spec.blocks == 0
        spec.blocks = cld(N, spec.threads)
    end
    kernel(N, f, x...; items = spec.threads,
        groups = spec.blocks, queue = spec.stream)
    if spec.sync
        oneAPI.synchronize(spec.stream)
    end
end

abstract type BlockIndexer2D end

struct BlockIndexerBasic <: BlockIndexer2D end

function (blkIter::BlockIndexerBasic)()
    i = get_global_id(1)
    j = get_global_id(2)
    return (i, j)
end

struct BlockIndexerSwapped <: BlockIndexer2D end

function (blkIter::BlockIndexerSwapped)()
    j = get_global_id(1)
    i = get_global_id(2)
    return (i, j)
end

function _parallel_for(indexer::TI, f, (m, n), (M, N), x...) where {TI}
    kernel = @oneapi launch=false _parallel_for_oneapi_MN(indexer, (M, N), f, x...)
    maxThreads = div(oneAPI.launch_configuration(kernel), 2)
    maxThreadsX = sqrt(maxThreads)
    y_thr = floor(Int, (n / m) * maxThreadsX)
    x_thr = fld(maxThreads, y_thr)
    items = (x_thr, y_thr)
    groups = (cld(m, items[1]), cld(n, items[2]))

    kernel(indexer, (M, N), f, x...; items = items, groups = groups)
    oneAPI.synchronize()
end

function JACC.parallel_for(
        f, ::oneAPIBackend, (M, N)::NTuple{2, Integer}, x...)
    dev = oneAPI.device()
    props = oneAPI.compute_properties(dev)
    maxBlocks = (x = props.maxGroupCountX, y = props.maxGroupCountY)
    if M < N && maxBlocks.x > maxBlocks.y
        _parallel_for(BlockIndexerSwapped(), f, (N, M), (M, N), x...)
    else
        _parallel_for(BlockIndexerBasic(), f, (M, N), (M, N), x...)
    end
end

function _parallel_for(indexer::TI, f, spec::LaunchSpec{oneAPIBackend}, (m, n),
        (M, N), x...) where {TI}
    kernel = @oneapi launch=false _parallel_for_oneapi_MN(indexer, (M, N), f, x...)

    if spec.threads == 0
        maxThreads = oneAPI.launch_configuration(kernel)
        maxThreadsX = sqrt(maxThreads)
        y_thr = floor(Int, (n / m) * maxThreadsX)
        x_thr = fld(maxThreads, y_thr)
        spec.threads = (x_thr, y_thr)
    end

    if spec.blocks == 0
        spec.blocks = (cld(M, spec.threads[1]), cld(N, spec.threads[2]))
    end
    kernel(
        indexer, (M, N), f, x...; items = spec.threads, groups = spec.blocks,
        queue = spec.stream)
    if spec.sync
        oneAPI.synchronize(spec.stream)
    end
end

function JACC.parallel_for(
        f, spec::LaunchSpec{oneAPIBackend}, (M, N)::NTuple{2, Integer}, x...)
    dev = oneAPI.device()
    props = oneAPI.compute_properties(dev)
    maxBlocks = (x = props.maxGroupCountX, y = props.maxGroupCountY)
    if M < N && maxBlocks.x > maxBlocks.y
        _parallel_for(BlockIndexerSwapped(), f, spec, (N, M), (M, N), x...)
    else
        _parallel_for(BlockIndexerBasic(), f, spec, (M, N), (M, N), x...)
    end
end

function JACC.parallel_for(
        f, ::oneAPIBackend, (L, M, N)::NTuple{3, Integer}, x...)
    maxItems = 8
    Litems = min(L, maxItems)
    Mitems = min(M, maxItems)
    Nitems = 1
    Lgroups = cld(L, Litems)
    Mgroups = cld(M, Mitems)
    Ngroups = cld(N, Nitems)
    oneAPI.@sync @oneapi items=(Litems, Mitems, Nitems) groups=(
        Lgroups, Mgroups, Ngroups) _parallel_for_oneapi_LMN((L, M, N),
        f, x...)
end

function JACC.parallel_for(
        f, spec::LaunchSpec{oneAPIBackend}, (L, M, N)::NTuple{3, Integer}, x...)
    if spec.threads == 0
        maxItems = 8
        Litems = min(L, maxItems)
        Mitems = min(M, maxItems)
        Nitems = 1
        spec.threads = (Litems, Mitems, Nitems)
    end
    if spec.blocks == 0
        Lgroups = cld(L, spec.threads[1])
        Mgroups = cld(M, spec.threads[2])
        Ngroups = cld(N, spec.threads[3])
        spec.blocks = (Lgroups, Mgroups, Ngroups)
    end
    @oneapi items=spec.threads groups=spec.blocks queue=spec.stream _parallel_for_oneapi_LMN(
        (L, M, N),
        f, x...)
    if spec.sync
        oneAPI.synchronize(spec.stream)
    end
end

mutable struct oneAPIReduceWorkspace{T, TP <: JACC.WkProp} <:
               JACC.ReduceWorkspace
    tmp::oneAPI.oneArray{T}
    ret::oneAPI.oneArray{T}
end

function JACC.reduce_workspace(::oneAPIBackend, init::T) where {T}
    oneAPIReduceWorkspace{T, JACC.Managed}(
        oneAPI.oneArray{T}(undef, 0), oneAPI.oneArray([init]))
end

function JACC.reduce_workspace(::oneAPIBackend, tmp::oneAPI.oneArray{T},
        init::oneAPI.oneArray{T}) where {T}
    oneAPIReduceWorkspace{T, JACC.Unmanaged}(tmp, init)
end

@inline function _init!(wk::oneAPIReduceWorkspace{T, JACC.Managed}, blocks, init) where {T}
    if length(wk.tmp) != prod(blocks)
        wk.tmp = oneAPI.oneArray{typeof(init)}(undef, blocks)
    end
    fill!(wk.tmp, init)
    fill!(wk.ret, init)
    return nothing
end

@inline function _init!(wk::oneAPIReduceWorkspace{T, JACC.Unmanaged}, blocks, init) where {T}
    nothing
end

JACC.get_result(wk::oneAPIReduceWorkspace) = Base.Array(wk.ret)[]

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{oneAPIBackend},
        N::Integer, f, x...)
    wk = reducer.workspace
    op = reducer.op
    init = reducer.init

    kernel1 = @oneapi launch=false _parallel_reduce_oneapi(
        Val(256), N, op, wk.ret, f, x...)
    threads1 = oneAPI.launch_configuration(kernel1)

    kernel2 = @oneapi launch=false reduce_kernel_oneapi(
        Val(256), 1, op, wk.ret, wk.ret)
    threads2 = oneAPI.launch_configuration(kernel2)

    threads = min(threads1, threads2, 256)
    blocks = cld(N, threads)
    shmem_size = threads * sizeof(init)

    _init!(wk, blocks, init)

    kernel1(Val(threads), N, op, wk.tmp, f, x...; items = threads,
        groups = blocks, queue = reducer.stream)
    kernel2(Val(threads), blocks, op, wk.tmp, wk.ret; items = threads,
        groups = 1, queue = reducer.stream)

    if reducer.sync
        oneAPI.synchronize(reducer.stream)
    end

    return nothing
end

function JACC.parallel_reduce(f, ::oneAPIBackend, N::Integer, x...; op, init)
    ret_inst = oneAPI.oneArray{typeof(init)}(undef, 0)
    kernel1 = @oneapi launch=false _parallel_reduce_oneapi(
        Val(256), N, op, ret_inst, f, x...)
    threads1 = oneAPI.launch_configuration(kernel1)

    rret = oneAPI.oneArray([init])
    kernel2 = @oneapi launch=false reduce_kernel_oneapi(
        Val(256), 1, op, ret_inst, rret)
    threads2 = oneAPI.launch_configuration(kernel2)

    items = 256
    groups = cld(N, items)

    ret = fill!(oneAPI.oneArray{typeof(init)}(undef, groups), init)

    @oneapi items=items groups=groups _parallel_reduce_oneapi(
        Val(items), N, op, ret, f, x...)

    @oneapi items=items groups=1 reduce_kernel_oneapi(
        Val(items), groups, op, ret, rret)

    oneAPI.synchronize()

    return Base.Array(rret)[]
end

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{oneAPIBackend},
        (M, N)::NTuple{2, Integer}, f, x...)
    init = reducer.init
    numItems = 16
    Mitems = numItems
    Nitems = numItems
    threads = (Mitems, Nitems)
    Mgroups = cld(M, threads[1])
    Ngroups = cld(N, threads[2])
    blocks = (Mgroups, Ngroups)

    wk = reducer.workspace
    _init!(wk, blocks, init)

    @oneapi items=threads groups=blocks queue=reducer.stream _parallel_reduce_oneapi_MN(
        (M, N), reducer.op, wk.tmp, f, x...)

    @oneapi items=threads groups=(1, 1) queue=reducer.stream reduce_kernel_oneapi_MN(
        blocks, reducer.op, wk.tmp, wk.ret)

    if reducer.sync
        oneAPI.synchronize(reducer.stream)
    end

    return nothing
end

function JACC.parallel_reduce(f, ::oneAPIBackend, (M, N)::NTuple{2, Integer},
        x...; op, init)
    numItems = 16
    Mitems = numItems
    Nitems = numItems
    items = (Mitems, Nitems)
    Mgroups = cld(M, Mitems)
    Ngroups = cld(N, Nitems)
    groups = (Mgroups, Ngroups)
    ret = fill!(oneAPI.oneArray{typeof(init)}(undef, (Mgroups, Ngroups)), init)
    rret = oneAPI.oneArray([init])
    @oneapi items=items groups=groups _parallel_reduce_oneapi_MN(
        (M, N), op, ret, f, x...)
    @oneapi items=items groups=(1, 1) reduce_kernel_oneapi_MN(
        groups, op, ret, rret)
    oneAPI.synchronize()
    return Base.Array(rret)[]
end

@inline function JACC.parallel_reduce(f, ::oneAPIBackend,
        dims::NTuple{N, Integer}, x...; op, init) where {N}
    ids = CartesianIndices(dims)
    return JACC.parallel_reduce(JACC.ReduceKernel1DND{typeof(init)}(),
        prod(dims), ids, f, x...; op = op, init = init)
end

@inline function _parallel_for_oneapi(N, f, x...)
    i = get_global_id()
    i > N && return nothing
    @inline f(i, x...)
    return nothing
end

@inline function _parallel_for_oneapi_MN(indexer, (M, N), f, x...)
    i, j = indexer()
    i > M && return nothing
    j > N && return nothing
    @inline f(i, j, x...)
    return nothing
end

@inline function _parallel_for_oneapi_LMN((L, M, N), f, x...)
    i = get_global_id(1)
    j = get_global_id(2)
    k = get_global_id(3)
    i > L && return nothing
    j > M && return nothing
    k > N && return nothing
    @inline f(i, j, k, x...)
    return nothing
end

function _parallel_reduce_oneapi(
        ::Val{shmem_length}, N, op, ret, f, x...) where {shmem_length}
    shared_mem = oneLocalArray(eltype(ret), shmem_length)
    i = get_global_id()
    ti = get_local_id()
    @inbounds shared_mem[ti] = ret[get_group_id()]

    if i <= N
        tmp = @inline f(i, x...)
        @inbounds shared_mem[ti] = tmp
    end

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        barrier()
        tn = 2^p
        if ti <= tn
            @inbounds shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + tn])
        end
    end

    if (ti == 1)
        @inbounds ret[get_group_id()] = shared_mem[ti]
    end
    return nothing
end

function reduce_kernel_oneapi(
        ::Val{shmem_length}, N, op, red, ret) where {shmem_length}
    shared_mem = oneLocalArray(eltype(ret), shmem_length)
    i = get_global_id()
    ii = i
    @inbounds tmp = ret[1]
    for ii in i:shmem_length:N
        tmp = op(tmp, @inbounds red[ii])
    end
    @inbounds shared_mem[i] = tmp

    max_pwr = JACC.ilog2(shmem_length) - 1
    for p in (max_pwr:-1:0)
        barrier()
        tn = 2^p
        if i <= tn
            @inbounds shared_mem[i] = op(shared_mem[i], shared_mem[i + tn])
        end
    end

    if (i == 1)
        @inbounds ret[1] = shared_mem[1]
    end
    return nothing
end

function _parallel_reduce_oneapi_MN((M, N), op, ret, f, x...)
    shared_mem = oneLocalArray(eltype(ret), (16, 16))
    i = get_global_id(1)
    j = get_global_id(2)
    ti = get_local_id(1)
    tj = get_local_id(2)
    bi = get_group_id(1)
    bj = get_group_id(2)

    @inbounds shared_mem[ti, tj] = ret[bi, bj]

    if (i <= M && j <= N)
        tmp = @inline f(i, j, x...)
        @inbounds shared_mem[ti, tj] = tmp
    end

    for n in (8, 4, 2, 1)
        barrier()
        if (ti <= n && tj <= n)
            @inbounds shared_mem[ti, tj] = op(shared_mem[ti, tj], shared_mem[ti + n, tj + n])
            @inbounds shared_mem[ti, tj] = op(shared_mem[ti, tj], shared_mem[ti, tj + n])
            @inbounds shared_mem[ti, tj] = op(shared_mem[ti, tj], shared_mem[ti + n, tj])
        end
    end

    if (ti == 1 && tj == 1)
        @inbounds ret[bi, bj] = shared_mem[ti, tj]
    end
    return nothing
end

function reduce_kernel_oneapi_MN((M, N), op, red, ret)
    shared_mem = oneLocalArray(eltype(ret), (16, 16))
    i = get_local_id(1)
    j = get_local_id(2)

    @inbounds tmp = ret[1]
    for ci in CartesianIndices((i:16:M, j:16:N))
        tmp = op(tmp, @inbounds red[ci])
    end
    @inbounds shared_mem[i, j] = tmp

    for n in (8, 4, 2, 1)
        barrier()
        if i <= n && j <= n
            @inbounds shared_mem[i, j] = op(shared_mem[i, j], shared_mem[i + n, j + n])
            @inbounds shared_mem[i, j] = op(shared_mem[i, j], shared_mem[i, j + n])
            @inbounds shared_mem[i, j] = op(shared_mem[i, j], shared_mem[i + n, j])
        end
    end

    if (i == 1 && j == 1)
        @inbounds ret[1] = shared_mem[i, j]
    end
    return nothing
end

function JACC.shared(::oneAPIBackend, x::AbstractArray)
    size::Int32 = length(x)
    # This is wrong, we should use size not 512 ...
    shmem = oneLocalArray(eltype(x), 512)
    num_threads = get_local_size(1) * get_local_size(2)
    if (size <= num_threads)
        if get_local_size(2) == 1
            ind = get_global_id(1)
            @inbounds shmem[ind] = x[ind]
        else
            i_local = get_local_id(1)
            j_local = get_local_id(2)
            ind = i_local - 1 * get_local_size(1) + j_local
            if ndims(x) == 1
                @inbounds shmem[ind] = x[ind]
            elseif ndims(x) == 2
                @inbounds shmem[ind] = x[i_local, j_local]
            end
        end
    else
        if get_local_size(2) == 1
            ind = get_local_id(1)
            for i in get_local_size(1):get_local_size(1):size
                @inbounds shmem[ind] = x[ind]
                ind += get_local_size(1)
            end
        else
            i_local = get_local_id(1)
            j_local = get_local_id(2)
            ind = (i_local - 1) * get_local_size(1) + j_local
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
    barrier()
    return shmem
end

JACC.sync_workgroup(::oneAPIBackend) = oneAPI.barrier()

JACC.array_type(::oneAPIBackend) = oneAPI.oneArray

JACC.array(::oneAPIBackend, x::Array) = oneAPI.oneArray(x)

DefaultFloat = Union{Type, Nothing}

function _get_default_float()
    if oneL0.module_properties(device()).fp64flags &
       oneL0.ZE_DEVICE_MODULE_FLAG_FP64 == oneL0.ZE_DEVICE_MODULE_FLAG_FP64
        return Float64
    else
        @info """Float64 unsupported on the current device.
        Default float for JACC.jl changed to Float32.
        """
        return Float32
    end
end

function JACC.default_float(::oneAPIBackend)
    global DefaultFloat
    if isa(nothing, DefaultFloat)
        DefaultFloat = _get_default_float()
    end
    return DefaultFloat
end

include("multi.jl")

end # module oneAPIExt
