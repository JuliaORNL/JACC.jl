module JACCCUDA

using JACC, CUDA

# overloaded array functions
include("array.jl")

include("JACCMULTI.jl")
using .Multi

# overloaded experimental functions
include("JACCEXPERIMENTAL.jl")
using .Experimental

JACC.get_backend(::Val{:cuda}) = CUDABackend()

function JACC.parallel_for(
        ::CUDABackend, N::I, f::F, x...) where {I <: Integer, F <: Function}
    #parallel_args = (N, f, x...)
    #parallel_kargs = cudaconvert.(parallel_args)
    #parallel_tt = Tuple{Core.Typeof.(parallel_kargs)...}
    #parallel_kernel = cufunction(_parallel_for_cuda, parallel_tt)
    #maxPossibleThreads = CUDA.maxthreads(parallel_kernel)
    maxPossibleThreads = 512
    threads = min(N, maxPossibleThreads)
    blocks = ceil(Int, N / threads)
    shmem_size = attribute(
        device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    #parallel_kernel(parallel_kargs...; threads = threads, blocks = blocks)
    CUDA.@sync @cuda threads=threads blocks=blocks shmem=shmem_size _parallel_for_cuda(
        N, f, x...)
end

function JACC.parallel_for(
        ::CUDABackend, (M, N)::Tuple{I, I}, f::F, x...) where {
        I <: Integer, F <: Function}
    #To use JACC.shared, it is recommended to use a high number of threads per block to maximize the
    # potential benefit from using shared memory.
    #numThreads = 32
    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N, numThreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nblocks = ceil(Int, N / Nthreads)
    shmem_size = attribute(
        device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    CUDA.@sync @cuda threads=(Mthreads, Nthreads) blocks=(Mblocks, Nblocks) shmem=shmem_size _parallel_for_cuda_MN(
        (M, N), f, x...)
end

function JACC.parallel_for(
        ::CUDABackend, (L, M, N)::Tuple{I, I, I}, f::F,
        x...) where {
        I <: Integer, F <: Function}
    #To use JACC.shared, it is recommended to use a high number of threads per block to maximize the
    # potential benefit from using shared memory.
    numThreads = 32
    Lthreads = min(L, numThreads)
    Mthreads = min(M, numThreads)
    Nthreads = 1
    Lblocks = ceil(Int, L / Lthreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nblocks = ceil(Int, N / Nthreads)
    shmem_size = attribute(
        device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    CUDA.@sync @cuda threads=(Lthreads, Mthreads, Nthreads) blocks=(
        Lblocks, Mblocks, Nblocks) shmem=shmem_size _parallel_for_cuda_LMN(
        (L, M, N), f, x...)
end

function JACC.parallel_reduce(
        ::CUDABackend, N::Integer, op, f::Function, x...; init)
    numThreads = 512
    threads = min(N, numThreads)
    blocks = ceil(Int, N / threads)
    ret = fill!(CUDA.CuArray{typeof(init)}(undef, blocks), init)
    rret = CUDA.CuArray([init])
    CUDA.@sync @cuda threads=threads blocks=blocks shmem=512 * sizeof(typeof(init)) _parallel_reduce_cuda(
        N, op, ret, f, x...)
    CUDA.@sync @cuda threads=threads blocks=1 shmem=512 * sizeof(typeof(init)) reduce_kernel_cuda(
        blocks, op, ret, rret)
    return Core.Array(rret)[]
end

function JACC.parallel_reduce(
        ::CUDABackend, (M, N)::Tuple{Integer, Integer}, op, f::Function, x...; init)
    numThreads = 16
    Mthreads = min(M, numThreads)
    Nthreads = min(N, numThreads)
    Mblocks = ceil(Int, M / Mthreads)
    Nblocks = ceil(Int, N / Nthreads)
    ret = fill!(CUDA.CuArray{typeof(init)}(undef, (Mblocks, Nblocks)), init)
    rret = CUDA.CuArray([init])
    CUDA.@sync @cuda threads=(Mthreads, Nthreads) blocks=(Mblocks, Nblocks) shmem=16 *
                                                                                  16 *
                                                                                  sizeof(typeof(init)) _parallel_reduce_cuda_MN(
        (M, N), op, ret, f, x...)
    CUDA.@sync @cuda threads=(Mthreads, Nthreads) blocks=(1, 1) shmem=16 * 16 *
    sizeof(typeof(init)) reduce_kernel_cuda_MN(
        (Mblocks, Nblocks), op, ret, rret)
    return Core.Array(rret)[]
end

function _parallel_for_cuda(N, f, x...)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > N && return nothing
    f(i, x...)
    return nothing
end

function _parallel_for_cuda_MN((M, N), f, x...)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
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
    shared_mem = @cuDynamicSharedMem(eltype(ret), 512)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ti = threadIdx().x
    tmp::eltype(ret) = 0.0
    shared_mem[ti] = 0.0

    if i <= N
        tmp = @inbounds f(i, x...)
        shared_mem[threadIdx().x] = tmp
    end
    sync_threads()
    if (ti <= 256)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 256])
    end
    sync_threads()
    if (ti <= 128)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 128])
    end
    sync_threads()
    if (ti <= 64)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 64])
    end
    sync_threads()
    if (ti <= 32)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 32])
    end
    sync_threads()
    if (ti <= 16)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 16])
    end
    sync_threads()
    if (ti <= 8)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 8])
    end
    sync_threads()
    if (ti <= 4)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 4])
    end
    sync_threads()
    if (ti <= 2)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 2])
    end
    sync_threads()
    if (ti == 1)
        shared_mem[ti] = op(shared_mem[ti], shared_mem[ti + 1])
        ret[blockIdx().x] = shared_mem[ti]
    end
    return nothing
end

function reduce_kernel_cuda(N, op, red, ret)
    shared_mem = @cuDynamicSharedMem(eltype(ret), 512)
    i = threadIdx().x
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
    sync_threads()
    if (i <= 256)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 256])
    end
    sync_threads()
    if (i <= 128)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 128])
    end
    sync_threads()
    if (i <= 64)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 64])
    end
    sync_threads()
    if (i <= 32)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 32])
    end
    sync_threads()
    if (i <= 16)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 16])
    end
    sync_threads()
    if (i <= 8)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 8])
    end
    sync_threads()
    if (i <= 4)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 4])
    end
    sync_threads()
    if (i <= 2)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 2])
    end
    sync_threads()
    if (i == 1)
        shared_mem[i] = op(shared_mem[i], shared_mem[i + 1])
        ret[1] = shared_mem[1]
    end
    return nothing
end

function _parallel_reduce_cuda_MN((M, N), op, ret, f, x...)
    shared_mem = @cuDynamicSharedMem(eltype(ret), 16*16)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ti = threadIdx().x
    tj = threadIdx().y
    bi = blockIdx().x
    bj = blockIdx().y

    tmp::eltype(ret) = 0.0
    sid = ((ti - 1) * 16) + tj
    shared_mem[sid] = tmp

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
    shared_mem = @cuDynamicSharedMem(eltype(ret), 16*16)
    i = threadIdx().x
    j = threadIdx().y
    ii = i
    jj = j

    tmp::eltype(ret) = 0.0
    sid = ((i - 1) * 16) + j
    shared_mem[sid] = tmp

    if M > 16 && N > 16
        while ii <= M
            jj = threadIdx().y
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
    red[i, j] = shared_mem[sid]
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
    shmem = @cuDynamicSharedMem(T, size)
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

JACC.array_type(::CUDABackend) = CUDA.CuArray{T, N} where {T, N}

end # module JACCCUDA
