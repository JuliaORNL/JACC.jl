module oneAPIExt

import Base: Callable
using JACC, oneAPI, oneAPI.oneL0

# overloaded array functions
include("array.jl")
include("experimental/experimental.jl")

JACC.get_backend(::Val{:oneapi}) = oneAPIBackend()

default_stream() = oneAPI.global_queue(oneAPI.context(), oneAPI.device())

JACC.default_stream(::Type{oneAPIBackend}) = default_stream()

function JACC.synchronize(::oneAPIBackend; stream = default_stream())
    oneAPI.synchronize(stream)
end

function JACC.parallel_for(::oneAPIBackend, N::Integer, f::Callable, x...)
    kernel = @oneapi launch=false _parallel_for_oneapi(N, f, x...)
    config_items = oneAPI.launch_configuration(kernel)
    items = min(N, config_items)
    groups = cld(N, items)
    kernel(N, f, x...; items = items, groups = groups)
    oneAPI.synchronize()
end

function JACC.parallel_for(
        spec::LaunchSpec{oneAPIBackend}, N::Integer, f::Callable, x...)
    kernel = @oneapi launch=false _parallel_for_oneapi(N, f, x...)
    if spec.threads == 0
        maxItems = oneAPI.launch_configuration(kernel)
        spec.threads = min(N, maxItems)
    end
    if spec.blocks == 0
        spec.blocks = cld(N, spec.threads)
    end
    kernel(N, f, x...; items = spec.threads, groups = spec.blocks, queue = spec.stream)
    if spec.sync
        oneAPI.synchronize(spec.stream)
    end
end

function blockIndexerBasic()
    i = get_global_id(1)
    j = get_global_id(2)
    return (i, j)
end

function blockIndexerSwapped()
    j = get_global_id(1)
    i = get_global_id(2)
    return (i, j)
end

function JACC.parallel_for(
        ::oneAPIBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    dev = oneAPI.device()
    props = oneAPI.compute_properties(dev)
    maxBlocks = (x = props.maxGroupCountX, y = props.maxGroupCountY)
    indexer = blockIndexerBasic
    m, n = (M, N)
    if M < N && maxBlocks.x > maxBlocks.y
        indexer = blockIndexerSwapped
        m, n = (N, M)
    end

    kernel = @oneapi launch=false _parallel_for_oneapi_MN(indexer, (M, N), f, x...)
    maxThreads = oneAPI.launch_configuration(kernel)
    blockAttrs = (
        max_x = props.maxGroupSizeX,
        max_y = props.maxGroupSizeY,
        total = props.maxTotalGroupSize
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
    items = (x_thr, y_thr)
    groups = (cld(m, x_thr), cld(n, y_thr))

    kernel(indexer, (M, N), f, x...; items = items, groups = groups)
    oneAPI.synchronize()
end

function JACC.parallel_for(
        spec::LaunchSpec{oneAPIBackend}, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    dev = oneAPI.device()
    props = oneAPI.compute_properties(dev)
    maxBlocks = (x = props.maxGroupCountX, y = props.maxGroupCountY)
    indexer = blockIndexerBasic
    m, n = (M, N)

    kernel = @oneapi launch=false _parallel_for_oneapi_MN(indexer, (M, N), f, x...)
    maxThreads = oneAPI.launch_configuration(kernel)

    if spec.threads == 0
        if M < N && maxBlocks.x > maxBlocks.y
            indexer = blockIndexerSwapped
            m, n = (N, M)
        end
        blockAttrs = (
            max_x = props.maxGroupSizeX,
            max_y = props.maxGroupSizeY,
            total = props.maxTotalGroupSize
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
        spec.blocks = (cld(M, spec.threads[1]), cld(N, spec.threads[2]))
    end
    kernel(indexer, (M, N), f, x...; items = spec.threads, groups = spec.blocks,
        queue = spec.stream)
    if spec.sync
        oneAPI.synchronize(spec.stream)
    end
end

function JACC.parallel_for(
        ::oneAPIBackend, (L, M, N)::NTuple{3, Integer}, f::Callable, x...)
    maxItems = 16
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
        spec::LaunchSpec{oneAPIBackend}, (L, M, N)::NTuple{3, Integer}, f::Callable, x...)
    if spec.threads == 0
        maxItems = 16
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

mutable struct oneAPIReduceWorkspace{T} <: JACC.ReduceWorkspace
    tmp::oneAPI.oneArray{T}
    ret::oneAPI.oneArray{T}
end

function JACC.reduce_workspace(::oneAPIBackend, init::T) where {T}
    oneAPIReduceWorkspace{T}(
        oneAPI.oneArray{T}(undef, 0), oneAPI.oneArray([init]))
end

JACC.get_result(wk::oneAPIReduceWorkspace) = Base.Array(wk.ret)[]

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{oneAPIBackend},
        N::Integer, f::Callable, x...)
    wk = reducer.workspace
    op = reducer.op
    init = reducer.init

    kernel1 = @oneapi launch=false _parallel_reduce_oneapi(
        Val(256), N, op, wk.ret, f, x...)
    threads1 = oneAPI.launch_configuration(kernel1)

    kernel2 = @oneapi launch=false reduce_kernel_oneapi(
        Val(256), 1, op, wk.ret, wk.ret)
    threads2 = oneAPI.launch_configuration(kernel2)

    spec = reducer.spec
    spec.threads = min(threads1, threads2, 256)
    spec.blocks = cld(N, spec.threads)
    spec.shmem_size = spec.threads * sizeof(init)

    if length(wk.tmp) != spec.blocks
        wk.tmp = oneAPI.oneArray{typeof(init)}(undef, spec.blocks)
    end
    fill!(wk.tmp, init)
    fill!(wk.ret, init)

    kernel1(Val(spec.threads), N, op, wk.tmp, f, x...; items = spec.threads,
        groups = spec.blocks, queue = spec.stream)

    oneAPI.synchronize(spec.stream)

    kernel2(Val(spec.threads), spec.blocks, op, wk.tmp, wk.ret;
        items = spec.threads, groups = 1, queue = spec.stream)

    if spec.sync
        oneAPI.synchronize(spec.stream)
    end

    return nothing
end

function JACC.parallel_reduce(
        ::oneAPIBackend, N::Integer, op, f::Callable, x...; init)
    ret_inst = oneAPI.oneArray{typeof(init)}(undef, 0)
    kernel1 = @oneapi launch=false _parallel_reduce_oneapi(
        Val(256), N, op, ret_inst, f, x...)
    threads1 = oneAPI.launch_configuration(kernel1)

    rret = oneAPI.oneArray([init])
    kernel2 = @oneapi launch=false reduce_kernel_oneapi(
        Val(256), 1, op, ret_inst, rret)
    threads2 = oneAPI.launch_configuration(kernel2)

    items = min(threads1, threads2, 256)
    groups = cld(N, items)

    ret = fill!(oneAPI.oneArray{typeof(init)}(undef, groups), init)

    kernel1(Val(items), N, op, ret, f, x...; items = items, groups = groups)
    oneAPI.synchronize()
    kernel2(Val(items), groups, op, ret, rret; items = items, groups = 1)
    oneAPI.synchronize()

    return Base.Array(rret)[]
end

function JACC.parallel_reduce(
        spec::LaunchSpec{oneAPIBackend}, N::Integer, op, f::Callable, x...; init)
    reducer = JACC.ParallelReduce{oneAPIBackend, typeof(init)}(
        dims = N,
        op = op,
        init = init,
        workspace = JACC.reduce_workspace(oneAPIBackend(), init),
        spec = spec,
    )
    JACC._parallel_reduce!(reducer, N, f, x...)
    return reducer.workspace.ret
end

function JACC._parallel_reduce!(reducer::JACC.ParallelReduce{oneAPIBackend},
        (M, N)::NTuple{2, Integer}, f::Callable, x...)
    init = reducer.init
    spec = reducer.spec
    numItems = 16
    Mitems = numItems
    Nitems = numItems
    spec.threads = (Mitems, Nitems)
    Mgroups = cld(M, spec.threads[1])
    Ngroups = cld(N, spec.threads[2])
    spec.blocks = (Mgroups, Ngroups)

    wk = reducer.workspace
    if size(wk.tmp) != spec.blocks
        wk.tmp = oneAPI.oneArray{typeof(init)}(undef, (Mgroups, Ngroups))
    end
    fill!(wk.tmp, init)
    fill!(wk.ret, init)

    @oneapi items=spec.threads groups=spec.blocks queue=spec.stream _parallel_reduce_oneapi_MN(
        (M, N), reducer.op, wk.tmp, f, x...)
    oneAPI.synchronize(spec.stream)
    @oneapi items=spec.threads groups=(1, 1) queue=spec.stream reduce_kernel_oneapi_MN(
        spec.blocks, reducer.op, wk.tmp, wk.ret)

    if spec.sync
        oneAPI.synchronize(spec.stream)
    end

    return nothing
end

function JACC.parallel_reduce(
        ::oneAPIBackend, (M, N)::NTuple{2, Integer}, op, f::Callable, x...; init)
    numItems = 16
    Mitems = numItems
    Nitems = numItems
    Mgroups = cld(M, Mitems)
    Ngroups = cld(N, Nitems)
    ret = fill!(oneAPI.oneArray{typeof(init)}(undef, (Mgroups, Ngroups)), init)
    rret = oneAPI.oneArray([init])
    oneAPI.@sync @oneapi items=(Mitems, Nitems) groups=(Mgroups, Ngroups) _parallel_reduce_oneapi_MN(
        (M, N), op, ret, f, x...)
    oneAPI.@sync @oneapi items=(Mitems, Nitems) groups=(1, 1) reduce_kernel_oneapi_MN(
        (Mgroups, Ngroups), op, ret, rret)
    return Base.Array(rret)[]
end

function JACC.parallel_reduce(
        spec::LaunchSpec{oneAPIBackend}, (M, N)::NTuple{2, Integer}, op, f::Callable, x...; init)
    numItems = 16
    Mitems = numItems
    Nitems = numItems
    spec.threads = (Mitems, Nitems)
    Mgroups = cld(M, spec.threads[1])
    Ngroups = cld(N, spec.threads[2])
    spec.blocks = (Mgroups, Ngroups)
    ret = fill!(oneAPI.oneArray{typeof(init)}(undef, (Mgroups, Ngroups)), init)
    rret = oneAPI.oneArray([init])
    @oneapi items=(Mitems, Nitems) groups=(Mgroups, Ngroups) queue=spec.stream _parallel_reduce_oneapi_MN(
        (M, N), op, ret, f, x...)
    oneAPI.synchronize(spec.stream)
    @oneapi items=(Mitems, Nitems) groups=(1, 1) queue=spec.stream reduce_kernel_oneapi_MN(
        (spec.blocks[1], spec.blocks[2]), op, ret, rret)
    if spec.sync
        oneAPI.synchronize(spec.stream)
    end
    return rret
end

function _parallel_for_oneapi(N, f, x...)
    i = get_global_id()
    i > N && return nothing
    f(i, x...)
    return nothing
end

function _parallel_for_oneapi_MN(indexer, (M, N), f, x...)
    i, j = indexer()
    i > M && return nothing
    j > N && return nothing
    f(i, j, x...)
    return nothing
end

function _parallel_for_oneapi_LMN((L, M, N), f, x...)
    i = get_global_id(1)
    j = get_global_id(2)
    k = get_global_id(3)
    i > L && return nothing
    j > M && return nothing
    k > N && return nothing
    f(i, j, k, x...)
    return nothing
end

function _parallel_reduce_oneapi(
        ::Val{shmem_length}, N, op, ret, f, x...) where {shmem_length}
    shared_mem = oneLocalArray(eltype(ret), shmem_length)
    i = get_global_id()
    ti = get_local_id()
    shared_mem[ti] = ret[get_group_id()]

    if i <= N
        tmp = f(i, x...)
        @inbounds shared_mem[ti] = tmp
    end
    barrier()

    max_pwr = floor(Int, log2(shmem_length)) - 1
    for p in (max_pwr:-1:1)
        tn = 2^p
        if ti <= tn
            @inbounds shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + tn])
        end
        barrier()
    end

    if (ti == 1)
        @inbounds shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 1])
        @inbounds ret[get_group_id()] = shared_mem[ti]
    end
    barrier()
    return nothing
end

function reduce_kernel_oneapi(
        ::Val{shmem_length}, N, op, red, ret) where {shmem_length}
    shared_mem = oneLocalArray(eltype(ret), shmem_length)
    i = get_global_id()
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
    barrier()

    max_pwr = floor(Int, log2(shmem_length)) - 1
    for p in (max_pwr:-1:1)
        tn = 2^p
        if i <= tn
            shared_mem[i] = op(shared_mem[i], shared_mem[i + tn])
        end
        barrier()
    end

    if (i == 1)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 1])
        ret[1] = shared_mem[1]
    end
    return nothing
end

function _parallel_reduce_oneapi_MN((M, N), op, ret, f, x...)
    shared_mem = oneLocalArray(eltype(ret), 16 * 16)
    i = get_global_id(1)
    j = get_global_id(2)
    ti = get_local_id(1)
    tj = get_local_id(2)
    bi = get_group_id(1)
    bj = get_group_id(2)

    sid = ((ti - 1) * 16) + tj
    shared_mem[sid] = ret[bi, bj]

    if (i <= M && j <= N)
        tmp = @inbounds f(i, j, x...)
        shared_mem[sid] = tmp
    end
    barrier()
    if (ti <= 8 && tj <= 8)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 7) * 16) + (tj + 8)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 8)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 7) * 16) + tj])
    end
    barrier()
    if (ti <= 4 && tj <= 4)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 3) * 16) + (tj + 4)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 4)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 3) * 16) + tj])
    end
    barrier()
    if (ti <= 2 && tj <= 2)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 1) * 16) + (tj + 2)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 2)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 1) * 16) + tj])
    end
    barrier()
    if (ti == 1 && tj == 1)
        shared_mem[sid] = op(shared_mem[sid], shared_mem[ti * 16 + (tj + 1)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 1)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[ti * 16 + tj])
        ret[bi, bj] = shared_mem[sid]
    end
    return nothing
end

function reduce_kernel_oneapi_MN((M, N), op, red, ret)
    shared_mem = oneLocalArray(eltype(ret), 16 * 16)
    i = get_local_id(1)
    j = get_local_id(2)

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
    barrier()
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
    barrier()
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
    barrier()
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
    barrier()
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

JACC.array(::oneAPIBackend, x::Base.Array) = oneAPI.oneArray(x)

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

end # module oneAPIExt
