module JACCONEAPI

using JACC, oneAPI, oneAPI.oneL0

# overloaded array functions
include("array.jl")

# overloaded experimental functions
include("JACCEXPERIMENTAL.jl")
using .Experimental

JACC.get_backend(::Val{:oneapi}) = oneAPIBackend()

default_stream() = oneAPI.global_queue(oneAPI.context(), oneAPI.device())

JACC.default_stream(::Type{oneAPIBackend}) = default_stream()

function JACC.synchronize(::oneAPIBackend; stream = default_stream())
    oneAPI.synchronize(stream)
end

function JACC.parallel_for(::oneAPIBackend, N::Integer, f::Function, x...)
    maxPossibleItems = 256
    items = min(N, maxPossibleItems)
    groups = ceil(Int, N / items)
    oneAPI.@sync @oneapi items=items groups=groups _parallel_for_oneapi(
        N, f, x...)
end

function JACC.parallel_for(
        spec::LaunchSpec{oneAPIBackend}, N::Integer, f::Function, x...)
    if spec.threads == 0
        maxPossibleItems = 256
        spec.threads = min(N, maxPossibleItems)
    end
    if spec.blocks == 0
        spec.blocks = ceil(Int, N / spec.threads)
    end
    @oneapi items=spec.threads groups=spec.blocks queue=spec.stream _parallel_for_oneapi(
        N, f, x...)
    if spec.sync
        oneAPI.synchronize(spec.stream)
    end
end

function JACC.parallel_for(
        ::oneAPIBackend, (M, N)::NTuple{2, Integer}, f::Function, x...)
    maxPossibleItems = 16
    Mitems = min(M, maxPossibleItems)
    Nitems = min(N, maxPossibleItems)
    Mgroups = ceil(Int, M / Mitems)
    Ngroups = ceil(Int, N / Nitems)
    oneAPI.@sync @oneapi items=(Mitems, Nitems) groups=(Mgroups, Ngroups) _parallel_for_oneapi_MN(
        (M, N),
        f, x...)
end

function JACC.parallel_for(
        spec::LaunchSpec{oneAPIBackend}, (M, N)::NTuple{2, Integer}, f::Function, x...)
    if spec.threads == 0
        maxPossibleItems = 16
        Mitems = min(M, maxPossibleItems)
        Nitems = min(N, maxPossibleItems)
        spec.threads = (Mitems, Nitems)
    end
    if spec.blocks == 0
        Mgroups = ceil(Int, M / spec.threads[1])
        Ngroups = ceil(Int, N / spec.threads[2])
        spec.blocks = (Mgroups, Ngroups)
    end
    @oneapi items=spec.threads groups=spec.blocks queue=spec.stream _parallel_for_oneapi_MN(
        (M, N),
        f, x...)
    if spec.sync
        oneAPI.synchronize(spec.stream)
    end
end

function JACC.parallel_for(
        ::oneAPIBackend, (L, M, N)::NTuple{3, Integer}, f::Function, x...)
    maxPossibleItems = 16
    Litems = min(L, maxPossibleItems)
    Mitems = min(M, maxPossibleItems)
    Nitems = 1
    Lgroups = ceil(Int, L / Litems)
    Mgroups = ceil(Int, M / Mitems)
    Ngroups = ceil(Int, N / Nitems)
    oneAPI.@sync @oneapi items=(Litems, Mitems, Nitems) groups=(
        Lgroups, Mgroups, Ngroups) _parallel_for_oneapi_LMN((L, M, N),
        f, x...)
end

function JACC.parallel_for(
        spec::LaunchSpec{oneAPIBackend}, (L, M, N)::NTuple{3, Integer}, f::Function, x...)
    if spec.threads == 0
        maxPossibleItems = 16
        Litems = min(L, maxPossibleItems)
        Mitems = min(M, maxPossibleItems)
        Nitems = 1
        spec.threads = (Litems, Mitems, Nitems)
    end
    if spec.blocks == 0
        Lgroups = ceil(Int, L / spec.threads[1])
        Mgroups = ceil(Int, M / spec.threads[2])
        Ngroups = ceil(Int, N / spec.threads[3])
        spec.blocks = (Lgroups, Mgroups, Ngroups)
    end
    @show spec.threads
    @show spec.blocks
    @oneapi items=spec.threads groups=spec.blocks queue=spec.stream _parallel_for_oneapi_LMN(
        (L, M, N),
        f, x...)
    if spec.sync
        oneAPI.synchronize(spec.stream)
    end
end

function JACC.parallel_reduce(
        ::oneAPIBackend, N::Integer, op, f::Function, x...; init)
    numItems = 256
    items = numItems
    groups = ceil(Int, N / items)
    ret = fill!(oneAPI.oneArray{typeof(init)}(undef, groups), init)
    rret = oneAPI.oneArray([init])
    oneAPI.@sync @oneapi items=items groups=groups _parallel_reduce_oneapi(
        N, op, ret, f, x...)
    oneAPI.@sync @oneapi items=items groups=1 reduce_kernel_oneapi(
        groups, op, ret, rret)
    return Base.Array(rret)[]
end

function JACC.parallel_reduce(
        ::oneAPIBackend, (M, N)::Tuple{Integer, Integer}, op, f::Function, x...; init)
    numItems = 16
    Mitems = numItems
    Nitems = numItems
    Mgroups = ceil(Int, M / Mitems)
    Ngroups = ceil(Int, N / Nitems)
    ret = fill!(oneAPI.oneArray{typeof(init)}(undef, (Mgroups, Ngroups)), init)
    rret = oneAPI.oneArray([init])
    oneAPI.@sync @oneapi items=(Mitems, Nitems) groups=(Mgroups, Ngroups) _parallel_reduce_oneapi_MN(
        (M, N), op, ret, f, x...)
    oneAPI.@sync @oneapi items=(Mitems, Nitems) groups=(1, 1) reduce_kernel_oneapi_MN(
        (Mgroups, Ngroups), op, ret, rret)
    return Base.Array(rret)[]
end

function _parallel_for_oneapi(N, f, x...)
    i = get_global_id()
    i > N && return nothing
    f(i, x...)
    return nothing
end

function _parallel_for_oneapi_MN((M, N), f, x...)
    j = get_global_id(0)
    i = get_global_id(1)
    i > M && return nothing
    j > N && return nothing
    f(i, j, x...)
    return nothing
end

function _parallel_for_oneapi_LMN((L, M, N), f, x...)
    i = get_global_id(0)
    j = get_global_id(1)
    k = get_global_id(2)
    i > L && return nothing
    j > M && return nothing
    k > N && return nothing
    f(i, j, k, x...)
    return nothing
end

function _parallel_reduce_oneapi(N, op, ret, f, x...)
    shared_mem = oneLocalArray(eltype(ret), 256)
    i = get_global_id(0)
    ti = get_local_id(0)
    shared_mem[ti] = ret[get_group_id(0)]

    if i <= N
        tmp = @inbounds f(i, x...)
        shared_mem[ti] = tmp
    end
    barrier()
    if (ti <= 128)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 128])
    end
    barrier()
    if (ti <= 64)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 64])
    end
    barrier()
    if (ti <= 32)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 32])
    end
    barrier()
    if (ti <= 16)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 16])
    end
    barrier()
    if (ti <= 8)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 8])
    end
    barrier()
    if (ti <= 4)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 4])
    end
    barrier()
    if (ti <= 2)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 2])
    end
    barrier()
    if (ti == 1)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 1])
        ret[get_group_id(0)] = shared_mem[ti]
    end
    barrier()
    return nothing
end

function reduce_kernel_oneapi(N, op, red, ret)
    shared_mem = oneLocalArray(eltype(ret), 256)
    i = get_global_id(0)
    ii = i
    tmp = ret[1]
    if N > 256
        while ii <= N
            tmp = op(tmp, @inbounds red[ii])
            ii += 256
        end
    elseif (i <= N)
        tmp = @inbounds red[i]
    end
    shared_mem[i] = tmp
    barrier()
    if (i <= 128)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 128])
    end
    barrier()
    if (i <= 64)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 64])
    end
    barrier()
    if (i <= 32)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 32])
    end
    barrier()
    if (i <= 16)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 16])
    end
    barrier()
    if (i <= 8)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 8])
    end
    barrier()
    if (i <= 4)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 4])
    end
    barrier()
    if (i <= 2)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 2])
    end
    barrier()
    if (i == 1)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 1])
        ret[1] = shared_mem[1]
    end
    return nothing
end

function _parallel_reduce_oneapi_MN((M, N), op, ret, f, x...)
    shared_mem = oneLocalArray(eltype(ret), 16 * 16)
    i = get_global_id(0)
    j = get_global_id(1)
    ti = get_local_id(0)
    tj = get_local_id(1)
    bi = get_group_id(0)
    bj = get_group_id(1)

    sid = ((ti - 1) * 16) + tj
    shared_mem[sid] = ret[bi, bj]

    if (i <= M && j <= N)
        tmp = @inbounds f(i, j, x...)
        shared_mem[sid] = tmp
    end
    barrier()
    if (ti <= 8 && tj <= 8 && ti + 8 <= M && tj + 8 <= N)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 7) * 16) + (tj + 8)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 8)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 7) * 16) + tj])
    end
    barrier()
    if (ti <= 4 && tj <= 4 && ti + 4 <= M && tj + 4 <= N)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 3) * 16) + (tj + 4)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 4)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 3) * 16) + tj])
    end
    barrier()
    if (ti <= 2 && tj <= 2 && ti + 2 <= M && tj + 2 <= N)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 1) * 16) + (tj + 2)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 2)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 1) * 16) + tj])
    end
    barrier()
    if (ti == 1 && tj == 1 && ti + 1 <= M && tj + 1 <= N)
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
    i = get_local_id(0)
    j = get_local_id(1)
    ii = i
    jj = j

    tmp = ret[1]
    sid = ((i - 1) * 16) + j
    shared_mem[sid] = tmp

    if M > 16 && N > 16
        while ii <= M
            jj = get_local_id(1)
            while jj <= N
                tmp = op(tmp, @inbounds red[ii, jj])
                jj += 16
            end
            ii += 16
        end
    elseif M > 16
        while ii <= N
            tmp = op(tmp, @inbounds red[ii, jj])
            ii += 16
        end
    elseif N > 16
        while jj <= N
            tmp = op(tmp, @inbounds red[ii, jj])
            jj += 16
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

function JACC.shared(x::oneDeviceArray{T, N}) where {T, N}
    size::Int32 = length(x)
    # This is wrong, we should use size not 512 ...
    shmem = oneLocalArray(T, 512)
    num_threads = get_local_size(0) * get_local_size(1)
    if (size <= num_threads)
        if get_local_size(1) == 1
            ind = get_global_id(0)
            @inbounds shmem[ind] = x[ind]
        else
            i_local = get_local_id(0)
            j_local = get_local_id(1)
            ind = i_local - 1 * get_local_size(0) + j_local
            if ndims(x) == 1
                @inbounds shmem[ind] = x[ind]
            elseif ndims(x) == 2
                @inbounds shmem[ind] = x[i_local, j_local]
            end
        end
    else
        if get_local_size(1) == 1
            ind = get_local_id(0)
            for i in get_local_size(0):get_local_size(0):size
                @inbounds shmem[ind] = x[ind]
                ind += get_local_size(0)
            end
        else
            i_local = get_local_id(0)
            j_local = get_local_id(1)
            ind = (i_local - 1) * get_local_size(0) + j_local
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

end # module JACCONEAPI
