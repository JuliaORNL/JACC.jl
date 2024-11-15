module JACCAMDGPU

using JACC, AMDGPU

const AMDGPUBackend = ROCBackend

# overloaded array functions
include("array.jl")

include("JACCMULTI.jl")
using .Multi

# overloaded experimental functions
include("JACCEXPERIMENTAL.jl")
using .Experimental

JACC.get_backend(::Val{:amdgpu}) = AMDGPUBackend()

function JACC.parallel_for(
        ::AMDGPUBackend, N::I, f::F, x...) where {I <: Integer, F <: Function}
    numThreads = 512
    threads = min(N, numThreads)
    blocks = ceil(Int, N / threads)
    # shmem_size = attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    # We must know how to get the max shared memory to be used in AMDGPU as it is done in CUDA
    shmem_size = 2 * threads * sizeof(Float64)
    @roc groupsize=threads gridsize=blocks shmem=shmem_size _parallel_for_amdgpu(
        N, f, x...)
    AMDGPU.synchronize()
end

function JACC.parallel_for(
        ::AMDGPUBackend, (M, N)::Tuple{I, I}, f::F, x...) where {
        I <: Integer, F <: Function}
    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N, numThreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nblocks = ceil(Int, N / Nthreads)
    # shmem_size = attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    # We must know how to get the max shared memory to be used in AMDGPU as it is done in CUDA
    shmem_size = 2 * Mthreads * Nthreads * sizeof(Float64)
    @roc groupsize=(Mthreads, Nthreads) gridsize=(Mblocks, Nblocks) shmem=shmem_size _parallel_for_amdgpu_MN(
        (M, N), f, x...)
    AMDGPU.synchronize()
end

function JACC.parallel_for(
        ::AMDGPUBackend, (L, M, N)::Tuple{I, I, I}, f::F,
        x...) where {
        I <: Integer, F <: Function}
    numThreads = 32
    Lthreads = min(L, numThreads)
    Mthreads = min(M, numThreads)
    Nthreads = 1
    Lblocks = ceil(Int, L / Lthreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nblocks = ceil(Int, N / Nthreads)
    # shmem_size = attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    # We must know how to get the max shared memory to be used in AMDGPU as it is done in CUDA
    shmem_size = 2 * Lthreads * Mthreads * Nthreads * sizeof(Float64)
    @roc groupsize=(Lthreads, Mthreads, Nthreads) gridsize=(
        Lblocks, Mblocks, Nblocks) shmem=shmem_size _parallel_for_amdgpu_LMN(
        (L, M, N), f, x...)
    AMDGPU.synchronize()
end

function JACC.parallel_reduce(
        ::AMDGPUBackend, N::Integer, op, f::Function, x...; init)
    numThreads = 512
    threads = min(N, numThreads)
    blocks = ceil(Int, N / threads)
    ret = fill!(AMDGPU.ROCArray{typeof(init)}(undef, blocks), init)
    rret = AMDGPU.ROCArray([init])
    @roc groupsize=threads gridsize=blocks _parallel_reduce_amdgpu(
        N, op, ret, f, x...)
    AMDGPU.synchronize()
    @roc groupsize=threads gridsize=1 reduce_kernel_amdgpu(
        blocks, op, ret, rret)
    AMDGPU.synchronize()
    return Core.Array(rret)[]
end

function JACC.parallel_reduce(
        ::AMDGPUBackend, (M, N)::Tuple{Integer, Integer}, op, f::Function, x...; init)
    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N, numThreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nblocks = ceil(Int, N / Nthreads)
    ret = fill!(AMDGPU.ROCArray{typeof(init)}(undef, (Mblocks, Nblocks)), init)
    rret = AMDGPU.ROCArray([init])
    @roc groupsize=(Mthreads, Nthreads) gridsize=(Mblocks, Nblocks) _parallel_reduce_amdgpu_MN(
        (M, N), op, ret, f, x...)
    AMDGPU.synchronize()
    @roc groupsize=(Mthreads, Nthreads) gridsize=(1, 1) reduce_kernel_amdgpu_MN(
        (Mblocks, Nblocks), op, ret, rret)
    AMDGPU.synchronize()
    return Core.Array(rret)[]
end

function _parallel_for_amdgpu(N, f, x...)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    i > N && return nothing
    f(i, x...)
    return nothing
end

function _parallel_for_amdgpu_MN((M, N), f, x...)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
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
    shared_mem = @ROCStaticLocalArray(eltype(ret), 512)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    ti = workitemIdx().x
    tmp::eltype(ret) = 0.0
    shared_mem[ti] = 0.0

    if i <= N
        tmp = @inbounds f(i, x...)
        shared_mem[ti] = tmp
    end
    AMDGPU.sync_workgroup()
    if (ti <= 256)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 256])
    end
    AMDGPU.sync_workgroup()
    if (ti <= 128)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 128])
    end
    AMDGPU.sync_workgroup()
    if (ti <= 64)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 64])
    end
    AMDGPU.sync_workgroup()
    if (ti <= 32)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 32])
    end
    AMDGPU.sync_workgroup()
    if (ti <= 16)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 16])
    end
    AMDGPU.sync_workgroup()
    if (ti <= 8)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 8])
    end
    AMDGPU.sync_workgroup()
    if (ti <= 4)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 4])
    end
    AMDGPU.sync_workgroup()
    if (ti <= 2)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 2])
    end
    AMDGPU.sync_workgroup()
    if (ti == 1)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 1])
        ret[workgroupIdx().x] = shared_mem[ti]
    end
    AMDGPU.sync_workgroup()
    return nothing
end

function reduce_kernel_amdgpu(N, op, red, ret)
    shared_mem = @ROCStaticLocalArray(eltype(ret), 512)
    i = workitemIdx().x
    ii = i
    tmp::eltype(ret) = 0.0
    if N > 512
        while ii <= N
            tmp = op(tmp, @inbounds red[ii])
            ii += 512
        end
    elseif (i <= N)
        tmp = @inbounds red[i]
    end
    shared_mem[i] = tmp
    AMDGPU.sync_workgroup()
    if (i <= 256)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 256])
    end
    AMDGPU.sync_workgroup()
    if (i <= 128)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 128])
    end
    AMDGPU.sync_workgroup()
    if (i <= 64)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 64])
    end
    AMDGPU.sync_workgroup()
    if (i <= 32)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 32])
    end
    AMDGPU.sync_workgroup()
    if (i <= 16)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 16])
    end
    AMDGPU.sync_workgroup()
    if (i <= 8)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 8])
    end
    AMDGPU.sync_workgroup()
    if (i <= 4)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 4])
    end
    AMDGPU.sync_workgroup()
    if (i <= 2)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 2])
    end
    AMDGPU.sync_workgroup()
    if (i == 1)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 1])
        ret[1] = shared_mem[1]
    end
    return nothing
end

function _parallel_reduce_amdgpu_MN((M, N), op, ret, f, x...)
    shared_mem = @ROCStaticLocalArray(eltype(ret), 256)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    ti = workitemIdx().x
    tj = workitemIdx().y
    bi = workgroupIdx().x
    bj = workgroupIdx().y

    tmp::eltype(ret) = 0.0
    sid = ((ti - 1) * 16) + tj
    shared_mem[sid] = tmp

    if (i <= M && j <= N)
        tmp = @inbounds f(i, j, x...)
        shared_mem[sid] = tmp
    end
    AMDGPU.sync_workgroup()
    if (ti <= 8 && tj <= 8 && ti + 8 <= M && tj + 8 <= N)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 7) * 16) + (tj + 8)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 8)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 7) * 16) + tj])
    end
    AMDGPU.sync_workgroup()
    if (ti <= 4 && tj <= 4 && ti + 4 <= M && tj + 4 <= N)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 3) * 16) + (tj + 4)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 4)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 3) * 16) + tj])
    end
    AMDGPU.sync_workgroup()
    if (ti <= 2 && tj <= 2 && ti + 2 <= M && tj + 2 <= N)
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti + 1) * 16) + (tj + 2)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 2)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[((ti + 1) * 16) + tj])
    end
    AMDGPU.sync_workgroup()
    if (ti == 1 && tj == 1 && ti + 1 <= M && tj + 1 <= N)
        shared_mem[sid] = op(shared_mem[sid], shared_mem[ti * 16 + (tj + 1)])
        shared_mem[sid] = op(
            shared_mem[sid], shared_mem[((ti - 1) * 16) + (tj + 1)])
        shared_mem[sid] = op(shared_mem[sid], shared_mem[ti * 16 + tj])
        ret[bi, bj] = shared_mem[sid]
    end
    return nothing
end

function reduce_kernel_amdgpu_MN((M, N), op, red, ret)
    shared_mem = @ROCStaticLocalArray(eltype(ret), 256)
    i = workitemIdx().x
    j = workitemIdx().y
    ii = i
    jj = j

    tmp::eltype(ret) = 0.0
    sid = ((i - 1) * 16) + j
    shared_mem[sid] = tmp

    if M > 16 && N > 16
        while ii <= M
            jj = workitemIdx().y
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

function JACC.shared(x::ROCDeviceArray{T, N}) where {T, N}
    size = length(x)
    shmem = @ROCDynamicLocalArray(T, size)
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

JACC.array_type(::AMDGPUBackend) = AMDGPU.ROCArray{T, N} where {T, N}

end # module JACCAMDGPU
