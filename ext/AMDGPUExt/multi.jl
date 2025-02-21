module Multi

import Base: Callable
using JACC, AMDGPU
using JACCAMDGPU: AMDGPUBackend

function JACC.Multi.ndev(::AMDGPUBackend)
    return length(AMDGPU.devices())
end

function get_portable_rocarray(x::Base.Array{T, N}) where {T, N}
    dims = size(x)
    bytesize = sizeof(T) * prod(dims)
    buf = AMDGPU.Mem.HostBuffer(bytesize, AMDGPU.HIP.hipHostAllocPortable)
    ROCArray{T, N}(
        AMDGPU.GPUArrays.DataRef(AMDGPU.pool_free, AMDGPU.Managed(buf)), dims)
end

function JACC.Multi.array(::AMDGPUBackend, x::Base.Array{T, N}) where {T, N}
    ret = Vector{Any}(undef, 2)
    ndev = length(AMDGPU.devices())

    if ndims(x) == 1
        AMDGPU.device!(AMDGPU.device(1))
        s_array = length(x)
        s_arrays = ceil(Int, s_array / ndev)
        #println(s_arrays)
        array_ret = Vector{Any}(undef, ndev)
        pointer_ret = Vector{AMDGPU.Device.ROCDeviceVector{
            T, AMDGPU.Device.AS.Global}}(undef, ndev)

        for i in 1:ndev
            AMDGPU.device!(AMDGPU.device(i))
            array_ret[i] = ROCArray(x[(((i - 1) * s_arrays) + 1):(i * s_arrays)])
            pointer_ret[i] = AMDGPU.rocconvert(array_ret[i])
        end

        AMDGPU.device!(AMDGPU.device(1))
        #amdgpu_pointer_ret = ROCArray(pointer_ret)
        amdgpu_pointer_ret = get_portable_rocarray(pointer_ret)
        copyto!(amdgpu_pointer_ret, pointer_ret)
        ret[1] = amdgpu_pointer_ret
        ret[2] = array_ret

    elseif ndims(x) == 2
        AMDGPU.device!(AMDGPU.device(1))
        #s_row_array = size(x,1)
        s_col_array = size(x, 2)
        s_col_arrays = ceil(Int, s_col_array / ndev)
        array_ret = Vector{Any}(undef, ndev)
        pointer_ret = Vector{AMDGPU.Device.ROCDeviceMatrix{T, 1}}(undef, ndev)

        for i in 1:ndev
            AMDGPU.device!(AMDGPU.device(i))
            array_ret[i] = ROCArray(x[
                :, (((i - 1) * s_col_arrays) + 1):(i * s_col_arrays)])
            pointer_ret[i] = AMDGPU.rocconvert(array_ret[i])
        end

        AMDGPU.device!(AMDGPU.device(1))
        #amdgpu_pointer_ret = ROCArray(pointer_ret)
        amdgpu_pointer_ret = get_portable_rocarray(pointer_ret)
        copyto!(amdgpu_pointer_ret, pointer_ret)
        ret[1] = amdgpu_pointer_ret
        ret[2] = array_ret
    end

    return ret
end

function JACC.Multi.gArray(::AMDGPUBackend, x::Base.Array{T, N}) where {T, N}
    ndev = length(AMDGPU.devices())
    ret = Vector{Any}(undef, 2)

    if ndims(x) == 1
        AMDGPU.device!(AMDGPU.device(1))
        s_array = length(x)
        s_arrays = ceil(Int, s_array / ndev)
        array_ret = Vector{Any}(undef, ndev)
        pointer_ret = Vector{AMDGPU.Device.ROCDeviceVector{T, AMDGPU.AS.Global}}(
            undef, ndev)

        for i in 1:ndev
            AMDGPU.device!(AMDGPU.device(i))
            if i == 1
                array_ret[i] = ROCArray(x[(((i - 1) * s_arrays) + 1):((i * s_arrays) + 1)])
            elseif i == ndev
                array_ret[i] = ROCArray(x[((((i - 1) * s_arrays) + 1) - 1):(i * s_arrays)])
            else
                array_ret[i] = ROCArray(x[((((i - 1) * s_arrays) + 1) - 1):((i * s_arrays) + 1)])
            end
            pointer_ret[i] = AMDGPU.rocconvert(array_ret[i])
        end

        AMDGPU.device!(AMDGPU.device(1))
        #amdgpu_pointer_ret = ROCArray(pointer_ret)
        amdgpu_pointer_ret = get_portable_rocarray(pointer_ret)
        copyto!(amdgpu_pointer_ret, pointer_ret)
        ret[1] = amdgpu_pointer_ret
        ret[2] = array_ret

    elseif ndims(x) == 2
        AMDGPU.device!(AMDGPU.device(1))
        #s_row_array = size(x,1)
        s_col_array = size(x, 2)
        s_col_arrays = ceil(Int, s_col_array / ndev)
        #println(s_col_arrays)
        array_ret = Vector{Any}(undef, ndev)
        pointer_ret = Vector{AMDGPU.Device.ROCDeviceMatrix{T, 1}}(undef, ndev)

        i_col_arrays = floor(Int, s_col_array / ndev)
        for i in 1:ndev
            AMDGPU.device!(AMDGPU.device(i))
            if i == 1
                array_ret[i] = ROCArray(x[
                    :, (((i - 1) * s_col_arrays) + 1):(i * s_col_arrays + 1)])
            elseif i == ndev
                array_ret[i] = ROCArray(x[
                    :, ((((i - 1) * s_col_arrays) + 1) - 1):(i * s_col_arrays)])
            else
                array_ret[i] = ROCArray(x[:,
                    ((((i - 1) * s_col_arrays) + 1) - 1):(i * s_col_arrays + 1)])
            end
            pointer_ret[i] = AMDGPU.rocconvert(array_ret[i])
        end

        AMDGPU.device!(AMDGPU.device(1))
        #amdgpu_pointer_ret = ROCArray(pointer_ret)
        amdgpu_pointer_ret = get_portable_rocarray(pointer_ret)
        copyto!(amdgpu_pointer_ret, pointer_ret)
        ret[1] = amdgpu_pointer_ret
        ret[2] = array_ret
    end

    return ret
end

function JACC.Multi.copy(::AMDGPUBackend, x::Vector{Any}, y::Vector{Any})
    AMDGPU.device!(AMDGPU.device(1))
    ndev = length(AMDGPU.devices())

    if ndims(x[2][1]) == 1
        for i in 1:ndev
            AMDGPU.device!(AMDGPU.device(i))
            size = length(x[2][i])
            numThreads = 512
            threads = min(size, numThreads)
            blocks = ceil(Int, size / threads)
            @roc groupsize=threads gridsize=blocks _multi_copy(i, x[1], y[1])
        end

        for i in 1:ndev
            AMDGPU.device!(AMDGPU.device(i))
            AMDGPU.synchronize()
        end

    elseif ndims(x[2][1]) == 2
        for i in 1:ndev
            AMDGPU.device!(AMDGPU.device(i))
            ssize = size(x[2][i])
            numThreads = 16
            Mthreads = min(ssize[1], numThreads)
            Mblocks = ceil(Int, ssize[1] / Mthreads)
            Nthreads = min(ssize[2], numThreads)
            Nblocks = ceil(Int, ssize[2] / Nthreads)
            @roc groupsize=(Mthreads, Nthreads) gridsize=(Mblocks, Nblocks) _multi_copy_2d(
                i, x[1], y[1])
        end

        for i in 1:ndev
            AMDGPU.device!(AMDGPU.device(i))
            AMDGPU.synchronize()
        end
    end

    AMDGPU.device!(AMDGPU.device(1))
end

function JACC.Multi.gid(
        ::AMDGPUBackend, dev_id::Integer, i::Integer, ndev::Integer)
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

function JACC.Multi.gid(
        ::AMDGPUBackend, dev_id::Integer, (i, j)::NTuple{2, Integer},
        ndev::Integer)
    ind = (0, 0)

    if dev_id == 1
        ind = (i, j)
    elseif dev_id == ndev
        ind = (i, j + 1)
    else
        ind = (i, j + 1)
    end

    return ind
end

function JACC.Multi.gswap(::AMDGPUBackend, x::Vector{Any})
    AMDGPU.device!(AMDGPU.device(1))
    ndev = length(AMDGPU.devices())

    if ndims(x[2][1]) == 1

        #Left to right swapping
        for i in 1:(ndev - 1)
            AMDGPU.device!(AMDGPU.device(i))
            tmp = Base.Array(x[2][i])
            size = length(tmp)
            ghost_lr = tmp[size - 1]
            AMDGPU.device!(AMDGPU.device(i + 1))
            @roc groupsize=32 gridsize=1 _multi_swap_ghost_lr(
                i + 1, x[1], ndev, size, ghost_lr)
        end

        #Right to left swapping
        for i in 2:ndev
            AMDGPU.device!(AMDGPU.device(i))
            tmp = Base.Array(x[2][i])
            if (i - 1) == 1
                size = length(tmp) - 1
            else
                size = length(tmp)
            end
            ghost_rl = tmp[2]
            AMDGPU.device!(AMDGPU.device(i - 1))
            @roc groupsize=32 gridsize=1 _multi_swap_ghost_rl(
                i - 1, x[1], ndev, size, ghost_rl)
        end

        for i in 1:ndev
            AMDGPU.device!(AMDGPU.device(i))
            AMDGPU.synchronize()
        end

    elseif ndims(x[2][1]) == 2

        #Left to right swapping
        for i in 1:(ndev - 1)
            AMDGPU.device!(AMDGPU.device(i))
            dim = size(x[2][i])
            tmp = Base.Array(x[2][i][:, dim[2] - 1])
            AMDGPU.device!(AMDGPU.device(i + 1))
            ghost_lr = ROCArray(tmp)
            numThreads = 512
            threads = min(dim[1], numThreads)
            blocks = ceil(Int, dim[1] / threads)
            #x[2][i+1][:,1] = ghost_lr
            @roc groupsize=threads gridsize=blocks _multi_swap_2d_ghost_lr(
                i + 1, x[1], ndev, dim[1], ghost_lr)
            #AMDGPU.synchronize()
        end

        #Right to left swapping
        for i in 2:ndev
            AMDGPU.device!(AMDGPU.device(i))
            tmp = Base.Array(x[2][i][:, 2])
            AMDGPU.device!(AMDGPU.device(i - 1))
            dim = size(x[2][i - 1])
            ghost_rl = ROCArray(tmp)
            numThreads = 512
            threads = min(dim[1], numThreads)
            blocks = ceil(Int, dim[1] / threads)
            @roc groupsize=threads gridsize=blocks _multi_swap_2d_ghost_rl(
                i - 1, x[1], ndev, dim[1], dim[2], ghost_rl)
            #AMDGPU.synchronize()
        end

        for i in 1:ndev
            AMDGPU.device!(AMDGPU.device(i))
            AMDGPU.synchronize()
        end
    end

    AMDGPU.device!(AMDGPU.device(1))
end

function JACC.Multi.gcopytoarray(
        ::AMDGPUBackend, x::Vector{Any}, y::Vector{Any})

    #x is the array and y is the ghost array
    AMDGPU.device!(AMDGPU.device(1))
    ndev = length(AMDGPU.devices())

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        size = length(y[2][i])
        numThreads = 512
        threads = min(size, numThreads)
        blocks = ceil(Int, size / threads)
        @roc groupsize=threads gridsize=blocks _multi_copy_ghosttoarray(
            i, x[1], y[1], size, ndev)
    end

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        AMDGPU.synchronize()
    end

    AMDGPU.device!(AMDGPU.device(1))
end

function JACC.Multi.copytogarray(
        ::AMDGPUBackend, x::Vector{Any}, y::Vector{Any})

    #x is the ghost array and y is the array
    AMDGPU.device!(AMDGPU.device(1))
    ndev = length(AMDGPU.devices())

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        size = length(x[2][i])
        numThreads = 512
        threads = min(size, numThreads)
        blocks = ceil(Int, size / threads)
        @roc groupsize=threads gridsize=blocks _multi_copy_arraytoghost(
            i, x[1], y[1], size, ndev)
    end

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        AMDGPU.synchronize()
    end

    AMDGPU.device!(AMDGPU.device(1))
end

function JACC.Multi.parallel_for(::AMDGPUBackend, N::Integer, f::Callable, x...)
    ndev = length(AMDGPU.devices())
    N_multi = ceil(Int, N / ndev)
    numThreads = 256
    threads = min(N_multi, numThreads)
    blocks = ceil(Int, N_multi / threads)

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        dev_id = i
        @roc groupsize=threads gridsize=blocks _multi_parallel_for_amdgpu(
            N_multi, dev_id, f, x...)
    end

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        AMDGPU.synchronize()
    end

    AMDGPU.device!(AMDGPU.device(1))
end

function JACC.Multi.parallel_for(
        ::AMDGPUBackend, (M, N)::NTuple{2, Integer}, f::Callable, x...)
    ndev = length(AMDGPU.devices())
    N_multi = ceil(Int, N / ndev)
    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N_multi, numThreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nblocks = ceil(Int, N_multi / Nthreads)

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        dev_id = i
        @roc groupsize=(Mthreads, Nthreads) gridsize=(Mblocks, Nblocks) _multi_parallel_for_amdgpu_MN(
            M, N_multi, dev_id, f, x...)
    end

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        AMDGPU.synchronize()
    end

    AMDGPU.device!(AMDGPU.device(1))
end

function JACC.Multi.parallel_reduce(
        ::AMDGPUBackend, N::Integer, f::Callable, x...)
    AMDGPU.device!(AMDGPU.device(1))
    ndev = length(AMDGPU.devices())
    ret = Vector{Any}(undef, ndev)
    rret = Vector{Any}(undef, ndev)
    N_multi = ceil(Int, N / ndev)
    numThreads = 512
    threads = min(N_multi, numThreads)
    blocks = ceil(Int, N_multi / threads)
    final_rret = AMDGPU.zeros(Float64, 1)

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        ret[i] = AMDGPU.zeros(Float64, blocks)
        rret[i] = AMDGPU.zeros(Float64, 1)
    end

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        dev_id = i
        @roc groupsize=threads gridsize=blocks _multi_parallel_reduce_amdgpu(
            N_multi, dev_id, ret[i], f, x...)
    end
    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        dev_id = i
        @roc groupsize=threads gridsize=1 _multi_reduce_kernel_amdgpu(
            blocks, ret[i], rret[i])
    end

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        AMDGPU.synchronize()
    end

    tmp_rret = Vector{Any}(undef, ndev)
    tmp_final_rret = 0.0

    for i in 1:ndev
        tmp_rret[i] = zeros(Float64, 1)
        AMDGPU.device!(AMDGPU.device(i))
        tmp_rret[i] = Base.Array(rret[i])
        #println(tmp_rret[i][1])
    end

    AMDGPU.device!(AMDGPU.device(1))
    for i in 1:ndev
        tmp_final_rret += tmp_rret[i][1]
    end
    final_rret = tmp_final_rret

    return final_rret
end

function JACC.Multi.parallel_reduce(::AMDGPUBackend,
        (M, N)::NTuple{2, Integer}, f::Callable, x...)
    ndev = length(devices())
    ret = Vector{Any}(undef, ndev)
    rret = Vector{Any}(undef, ndev)
    N_multi = ceil(Int, N / ndev)
    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N_multi, numThreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nblocks = ceil(Int, N_multi / Nthreads)
    final_rret = AMDGPU.zeros(Float64, 1)

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        ret[i] = AMDGPU.zeros(Float64, (Mblocks, Nblocks))
        rret[i] = AMDGPU.zeros(Float64, 1)
    end

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        dev_id = i

        @roc groupsize=(Mthreads, Nthreads) gridsize=(Mblocks, Nblocks) _multi_parallel_reduce_amdgpu_MN(
            (M, N_multi), dev_id, ret[i], f, x...)

        @roc groupsize=(Mthreads, Nthreads) gridsize=(1, 1) _multi_reduce_kernel_amdgpu_MN(
            (Mblocks, Nblocks), ret[i], rret[i])
    end

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        AMDGPU.synchronize()
    end

    tmp = zeros(ndev)

    for i in 1:ndev
        AMDGPU.device!(AMDGPU.device(i))
        tmp[i] = Base.Array(rret[i])
    end

    AMDGPU.device!(AMDGPU.device(1))
    for i in 1:ndev
        final_rret += tmp[i]
    end

    return final_rret
end

function _multi_copy(dev_id, x, y)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    @inbounds x[dev_id][i] = y[dev_id][i]
    return nothing
end

function _multi_copy_2d(dev_id, x, y)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    @inbounds x[dev_id][i, j] = y[dev_id][i, j]
    return nothing
end

function _multi_copy_ghosttoarray(dev_id, x, y, size, ndev)
    #x is the array and y is the ghost array
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if dev_id == 1 && i < size
        @inbounds x[dev_id][i] = y[dev_id][i]
    elseif dev_id == ndev && i > 1
        @inbounds x[dev_id][i - 1] = y[dev_id][i]
    elseif i > 1 && i < size
        @inbounds x[dev_id][i - 1] = y[dev_id][i]
    end
    return nothing
end

function _multi_copy_arraytoghost(dev_id, x, y, size, ndev)
    #x is the ghost array and y is the array
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if dev_id == 1 && i < size
        @inbounds x[dev_id][i] = y[dev_id][i]
    elseif dev_id == ndev && i < size
        @inbounds x[dev_id][i + 1] = y[dev_id][i]
    elseif i > 1 && i < size
        @inbounds x[dev_id][i] = y[dev_id][i]
    end
    return nothing
end

function _multi_swap_ghost_lr(dev_id, x, ndev, size, ghost)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if i == 1
        x[dev_id][i] = ghost
    end
    return nothing
end

function _multi_swap_2d_ghost_lr(dev_id, x, ndev, size, ghost)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if i < size + 1
        x[dev_id][i, 1] = ghost[i]
    end
    return nothing
end

function _multi_swap_ghost_rl(dev_id, x, ndev, size, ghost)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if i == 1
        #x[dev_id][120] = ghost
        x[dev_id][size] = ghost
    end
    return nothing
end

function _multi_swap_2d_ghost_rl(dev_id, x, ndev, size, col, ghost)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if i < size + 1
        x[dev_id][i, col] = ghost[i]
    end
    return nothing
end

function _multi_parallel_for_amdgpu(N, dev_id, f, x...)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if i <= N
        f(dev_id, i, x...)
    end
    return nothing
end

function _multi_parallel_for_amdgpu_MN(M, N, dev_id, f, x...)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    if (i <= M) && (j <= N)
        f(dev_id, i, j, x...)
    end
    return nothing
end

function _multi_parallel_reduce_amdgpu(N, dev_id, ret, f, x...)
    shared_mem = @ROCStaticLocalArray(Float64, 512)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ti = workitemIdx().x
    tmp::Float64 = 0.0
    shared_mem[ti] = 0.0

    if i <= N
        tmp = @inbounds f(dev_id, i, x...)
        shared_mem[workitemIdx().x] = tmp
    end
    AMDGPU.sync_workgroup()
    if (ti <= 256)
        shared_mem[ti] += shared_mem[ti + 256]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 128)
        shared_mem[ti] += shared_mem[ti + 128]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 64)
        shared_mem[ti] += shared_mem[ti + 64]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 32)
        shared_mem[ti] += shared_mem[ti + 32]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 16)
        shared_mem[ti] += shared_mem[ti + 16]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 8)
        shared_mem[ti] += shared_mem[ti + 8]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 4)
        shared_mem[ti] += shared_mem[ti + 4]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 2)
        shared_mem[ti] += shared_mem[ti + 2]
    end
    AMDGPU.sync_workgroup()
    if (ti == 1)
        shared_mem[ti] += shared_mem[ti + 1]
        ret[workgroupIdx().x] = shared_mem[ti]
    end
    return nothing
end

function _multi_reduce_kernel_amdgpu(N, red, ret)
    shared_mem = @ROCStaticLocalArray(Float64, 512)
    i = workitemIdx().x
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
    shared_mem[workitemIdx().x] = tmp
    AMDGPU.sync_workgroup()
    if (i <= 256)
        shared_mem[i] += shared_mem[i + 256]
    end
    AMDGPU.sync_workgroup()
    if (i <= 128)
        shared_mem[i] += shared_mem[i + 128]
    end
    AMDGPU.sync_workgroup()
    if (i <= 64)
        shared_mem[i] += shared_mem[i + 64]
    end
    AMDGPU.sync_workgroup()
    if (i <= 32)
        shared_mem[i] += shared_mem[i + 32]
    end
    AMDGPU.sync_workgroup()
    if (i <= 16)
        shared_mem[i] += shared_mem[i + 16]
    end
    AMDGPU.sync_workgroup()
    if (i <= 8)
        shared_mem[i] += shared_mem[i + 8]
    end
    AMDGPU.sync_workgroup()
    if (i <= 4)
        shared_mem[i] += shared_mem[i + 4]
    end
    AMDGPU.sync_workgroup()
    if (i <= 2)
        shared_mem[i] += shared_mem[i + 2]
    end
    AMDGPU.sync_workgroup()
    if (i == 1)
        shared_mem[i] += shared_mem[i + 1]
        ret[1] = shared_mem[1]
    end
    AMDGPU.sync_workgroup()
    return nothing
end

function _multi_parallel_reduce_amdgpu_MN((M, N), dev_id, ret, f, x...)
    shared_mem = @ROCStaticLocalArray(Float64, 16*16)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    ti = workitemIdx().x
    tj = workitemIdx().y
    bi = workgroupIdx().x
    bj = workgroupIdx().y

    tmp::Float64 = 0.0
    shared_mem[((ti - 1) * 16) + tj] = tmp

    if (i <= M && j <= N)
        tmp = @inbounds f(dev_id, i, j, x...)
        shared_mem[(ti - 1) * 16 + tj] = tmp
    end
    AMDGPU.sync_workgroup()
    if (ti <= 8 && tj <= 8 && ti + 8 <= M && tj + 8 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 8)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 7) * 16) + tj]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 4 && tj <= 4 && ti + 4 <= M && tj + 4 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 4)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 3) * 16) + tj]
    end
    AMDGPU.sync_workgroup()
    if (ti <= 2 && tj <= 2 && ti + 2 <= M && tj + 2 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 2)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti + 1) * 16) + tj]
    end
    AMDGPU.sync_workgroup()
    if (ti == 1 && tj == 1 && ti + 1 <= M && tj + 1 <= N)
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[((ti - 1) * 16) + (tj + 1)]
        shared_mem[((ti - 1) * 16) + tj] += shared_mem[ti * 16 + tj]
        ret[bi, bj] = shared_mem[((ti - 1) * 16) + tj]
    end
    return nothing
end

function _multi_reduce_kernel_amdgpu_MN((M, N), red, ret)
    shared_mem = @ROCStaticLocalArray(Float64, 16*16)
    i = workitemIdx().x
    j = workitemIdx().y
    ii = i
    jj = j

    tmp::Float64 = 0.0
    shared_mem[(i - 1) * 16 + j] = tmp

    if M > 16 && N > 16
        while ii <= M
            jj = workitemIdx().y
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
    AMDGPU.sync_workgroup()
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
    AMDGPU.sync_workgroup()
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
    AMDGPU.sync_workgroup()
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
    AMDGPU.sync_workgroup()
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

end # module Multi
