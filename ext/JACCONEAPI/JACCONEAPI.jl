
module JACCONEAPI

using JACC, oneAPI
using JACC: JACCArrayType
function JACC.parallel_for(::JACCArrayType{<:oneArray}, N::Integer, f::Function, x...)
    #maxPossibleItems = oneAPI.oneL0.compute_properties(device().maxTotalGroupSize)
    maxPossibleItems = 256
    items = min(N, maxPossibleItems)
    groups = ceil(Int, N / items)
    oneAPI.@sync @oneapi items=items groups=groups _parallel_for_oneapi(f, x...)
end

function JACC.parallel_for(::JACCArrayType{<:oneArray},
        (M, N)::Tuple{Integer, Integer}, f::Function, x...)
    maxPossibleItems = 16
    Mitems = min(M, maxPossibleItems)
    Nitems = min(N, maxPossibleItems)
    Mgroups = ceil(Int, M / Mitems)
    Ngroups = ceil(Int, N / Nitems)
    oneAPI.@sync @oneapi items=(Mitems, Nitems) groups=(Mgroups, Ngroups) _parallel_for_oneapi_MN(
        f, x...)
end

function JACC.parallel_reduce(::JACCArrayType{<:oneArray},
        N::Integer, f::Function, x...)
    numItems = 256
    items = min(N, numItems)
    groups = ceil(Int, N / items)
    ret = oneAPI.zeros(Float32, groups)
    rret = oneAPI.zeros(Float32, 1)
    oneAPI.@sync @oneapi items=items groups=groups _parallel_reduce_oneapi(
        N, ret, f, x...)
    oneAPI.@sync @oneapi items=items groups=1 reduce_kernel_oneapi(N, ret, rret)
    return rret
end

function JACC.parallel_reduce(::JACCArrayType{<:oneArray},
        (M, N)::Tuple{Integer, Integer}, f::Function, x...)
    numItems = 16
    Mitems = min(M, numItems)
    Nitems = min(N, numItems)
    Mgroups = ceil(Int, M / Mitems)
    Ngroups = ceil(Int, N / Nitems)
    ret = oneAPI.zeros(Float32, (Mgroups, Ngroups))
    rret = oneAPI.zeros(Float32, 1)
    oneAPI.@sync @oneapi items=(Mitems, Nitems) groups=(Mgroups, Ngroups) _parallel_reduce_oneapi_MN(
        (M, N), ret, f, x...)
    oneAPI.@sync @oneapi items=(Mitems, Nitems) groups=(1, 1) reduce_kernel_oneapi_MN(
        (Mgroups, Ngroups), ret, rret)
    return rret
end

function _parallel_for_oneapi(f, x...)
    i = get_global_id()
    f(i, x...)
    return nothing
end

function _parallel_for_oneapi_MN(f, x...)
    i = get_global_id(0)
    j = get_global_id(1)
    f(i, j, x...)
    return nothing
end

function _parallel_reduce_oneapi(N, ret, f, x...)
    #shared_mem = oneLocalArray(Float32, 256)
    shared_mem = oneLocalArray(Float64, 256)
    i = get_global_id(0)
    ti = get_local_id(0)
    #tmp::Float32 = 0.0
    tmp::Float64 = 0.0
    shared_mem[ti] = 0.0
    if i <= N
        tmp = @inbounds f(i, x...)
        shared_mem[ti] = tmp
    end
    barrier()
    if (ti <= 128)
        shared_mem[ti] += shared_mem[ti + 128]
    end
    barrier()
    if (ti <= 64)
        shared_mem[ti] += shared_mem[ti + 64]
    end
    barrier()
    if (ti <= 32)
        shared_mem[ti] += shared_mem[ti + 32]
    end
    barrier()
    if (ti <= 16)
        shared_mem[ti] += shared_mem[ti + 16]
    end
    barrier()
    if (ti <= 8)
        shared_mem[ti] += shared_mem[ti + 8]
    end
    barrier()
    if (ti <= 4)
        shared_mem[ti] += shared_mem[ti + 4]
    end
    barrier()
    if (ti <= 2)
        shared_mem[ti] += shared_mem[ti + 2]
    end
    barrier()
    if (ti == 1)
        shared_mem[ti] += shared_mem[ti + 1]
        ret[get_group_id(0)] = shared_mem[ti]
    end
    barrier()
    return nothing
end

function reduce_kernel_oneapi(N, red, ret)
    #shared_mem = oneLocalArray(Float32, 256)
    shared_mem = oneLocalArray(Float64, 256)
    i = get_global_id()
    ii = i
    #tmp::Float32 = 0.0
    tmp::Float64 = 0.0
    if N > 256
        while ii <= N
            tmp += @inbounds red[ii]
            ii += 256
        end
    elseif (i <= N)
        tmp = @inbounds red[i]
    end
    shared_mem[i] = tmp
    barrier()
    if (i <= 128)
        shared_mem[i] += shared_mem[i + 128]
    end
    barrier()
    if (i <= 64)
        shared_mem[i] += shared_mem[i + 64]
    end
    barrier()
    if (i <= 32)
        shared_mem[i] += shared_mem[i + 32]
    end
    barrier()
    if (i <= 16)
        shared_mem[i] += shared_mem[i + 16]
    end
    barrier()
    if (i <= 8)
        shared_mem[i] += shared_mem[i + 8]
    end
    barrier()
    if (i <= 4)
        shared_mem[i] += shared_mem[i + 4]
    end
    barrier()
    if (i <= 2)
        shared_mem[i] += shared_mem[i + 2]
    end
    barrier()
    if (i == 1)
        shared_mem[i] += shared_mem[i + 1]
        ret[1] = shared_mem[1]
    end
    return nothing
end

function _parallel_reduce_oneapi_MN((M, N), ret, f, x...)
    #shared_mem = oneLocalArray(Float32, 16 * 16)
    shared_mem = oneLocalArray(Float64, 16 * 16)
    i = get_global_id(0)
    j = get_global_id(1)
    ti = get_local_id(0)
    tj = get_local_id(1)
    bi = get_group_id(0)
    bj = get_group_id(1)

    #tmp::Float32 = 0.0
    tmp::Float64 = 0.0
    shared_mem[((ti - 1) * 16) + tj] = tmp

    if (i <= M && j <= N)
        tmp = @inbounds f(i, j, x...)
        shared_mem[(ti - 1) * 16 + tj] = tmp
    end
    barrier()
    if (ti <= 8 && tj <= 8 && ti + 8 <= M && tj + 8 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + tj]
    end
    barrier()
    if (ti <= 4 && tj <= 4 && ti + 4 <= M && tj + 4 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + tj]
    end
    barrier()
    if (ti <= 2 && tj <= 2 && ti + 2 <= M && tj + 2 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + tj]
    end
    barrier()
    if (ti == 1 && tj == 1 && ti + 1 <= M && tj + 1 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + tj]
        ret[bi, bj] = shared_mem[((ti - 1) * 16) + tj]
    end
    return nothing
end

function reduce_kernel_oneapi_MN((M, N), red, ret)
    #shared_mem = oneLocalArray(Float32, 16 * 16)
    shared_mem = oneLocalArray(Float64, 16 * 16)
    i = get_local_id(0)
    j = get_local_id(1)
    ii = i
    jj = j

    #tmp::Float32 = 0.0
    tmp::Float64 = 0.0
    shared_mem[(i - 1) * 16 + j] = tmp

    if M > 16 && N > 16
        while ii <= M
            jj = get_local_id(1)
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
    barrier()
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
    barrier()
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
    barrier()
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
    barrier()
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

JACC.arraytype(::Val{:oneapi}) = oneArray

function __init__()
end

end # module JACCONEAPI
