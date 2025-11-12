using oneAPI

function axpy_oneapi_kernel(SIZE::Integer, alpha, x, y)
    i = get_global_id()
    i > SIZE && return nothing
    axpy(i, alpha, x, y)
    return nothing
end

function axpy_oneapi(SIZE::Integer, alpha, x, y)
    maxPossibleItems = 256
    items = min(SIZE, maxPossibleItems)
    groups = cld(SIZE, items)
    oneAPI.@sync @oneapi items=items groups=groups axpy_oneapi_kernel(
        SIZE, alpha, x, y)
end

function axpy_oneapi_kernel((M, N)::NTuple{2, Integer}, alpha, x, y)
    i = get_global_id(1)
    j = get_global_id(2)
    i > M && return nothing
    j > N && return nothing
    axpy(i, j, alpha, x, y)
    return nothing
end

function axpy_oneapi((M, N)::NTuple{2, Integer}, alpha, x, y)
    maxPossibleItems = 16
    Mitems = min(M, maxPossibleItems)
    Nitems = min(N, maxPossibleItems)
    items = (Mitems, Nitems)
    Mgroups = cld(M, Mitems)
    Ngroups = cld(N, Nitems)
    groups = (Mgroups, Ngroups)
    oneAPI.@sync @oneapi items=items groups=groups axpy_oneapi_kernel(
        (M, N), alpha, x, y)
end

push!(JACCBench.axpy_comps, axpy_oneapi)

function dot_oneapi_kernel(SIZE::Integer, ret, x, y)
    shared_mem = oneLocalArray(Float32, 256)
    i = get_global_id()
    ti = get_local_id()
    tmp::Float32 = 0.0
    shared_mem[ti] = 0.0
    if i <= SIZE
        @inbounds tmp = x[i] * y[i]
        @inbounds shared_mem[ti] = tmp
    end
    barrier()
    if (ti <= 128)
        @inbounds shared_mem[ti] += shared_mem[ti + 128]
    end
    barrier()
    if (ti <= 64)
        @inbounds shared_mem[ti] += shared_mem[ti + 64]
    end
    barrier()
    if (ti <= 32)
        @inbounds shared_mem[ti] += shared_mem[ti + 32]
    end
    barrier()
    if (ti <= 16)
        @inbounds shared_mem[ti] += shared_mem[ti + 16]
    end
    barrier()
    if (ti <= 8)
        @inbounds shared_mem[ti] += shared_mem[ti + 8]
    end
    barrier()
    if (ti <= 4)
        @inbounds shared_mem[ti] += shared_mem[ti + 4]
    end
    barrier()
    if (ti <= 2)
        @inbounds shared_mem[ti] += shared_mem[ti + 2]
    end
    barrier()
    if (ti == 1)
        @inbounds shared_mem[ti] += shared_mem[ti + 1]
        ret[get_group_id()] = shared_mem[ti]
    end
    return nothing
end

function oneapi_reduce_kernel(SIZE::Integer, red, ret)
    @inbounds begin
        shared_mem = oneLocalArray(Float32, 256)
        i = get_global_id()
        ii = i
        tmp::Float32 = 0.0
        if SIZE > 256
            while ii <= SIZE
                tmp += red[ii]
                ii += 256
            end
        else
            tmp = red[i]
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
    end
    return nothing
end

function dot_oneapi(SIZE::Integer, x, y)
    numItems = 256
    items = min(SIZE, numItems)
    groups = cld(SIZE, items)
    ret = oneAPI.zeros(Float32, groups)
    rret = oneAPI.zeros(Float32, 1)
    @oneapi items=items groups=groups dot_oneapi_kernel(SIZE, ret, x, y)
    oneAPI.synchronize()
    @oneapi items=items groups=1 oneapi_reduce_kernel(SIZE, ret, rret)
    oneAPI.synchronize()
    return Base.Array(rret)[]
end

function dot_oneapi_kernel((M, N)::NTuple{2, Integer}, ret, x, y)
    shared_mem = oneLocalArray(Float32, 16 * 16)
    i = get_global_id(0)
    j = get_global_id(1)
    ti = get_local_id(1)
    tj = get_local_id(2)
    bi = get_group_id(1)
    bj = get_group_id(2)

    tmp::Float32 = 0.0
    shared_mem[((ti - 1) * 16) + tj] = tmp

    if (i <= M && j <= N)
        tmp = @inbounds x[i, j] * y[i, j]
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
    barrier()
    return nothing
end

function reduce_kernel((M, N)::NTuple{2, Integer}, red, ret)
    shared_mem = oneLocalArray(Float32, 16 * 16)
    i = get_local_id(1)
    j = get_local_id(2)
    ii = i
    jj = j

    tmp::Float32 = 0.0
    shared_mem[(i - 1) * 16 + j] = tmp

    if M > 16 && N > 16
        while ii <= M
            jj = get_local_id(2)
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

function dot_oneapi((M, N)::NTuple{2, Integer}, x, y)
    maxPossibleItems = 16
    Mitems = min(M, maxPossibleItems)
    Nitems = min(N, maxPossibleItems)
    items = (Mitems, Nitems)
    Mgroups = cld(M, Mitems)
    Ngroups = cld(N, Nitems)
    groups = (Mgroups, Ngroups)
    ret = oneAPI.zeros(Float32, groups)
    rret = oneAPI.zeros(Float32, 1)
    @oneapi items=items groups=groups dot_oneapi_kernel((M, N), ret, x, y)
    oneAPI.synchronize()
    @oneapi items=items groups=(1, 1) reduce_kernel(groups, ret, rret)
    oneAPI.synchronize()
    return rret
end

push!(JACCBench.dot_comps, dot_oneapi)
