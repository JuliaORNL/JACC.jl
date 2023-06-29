using CUDA
import JACC
using Test

@testset "perf-AXPY" begin

    N = 300_000_000
    x = round.(rand(Float32, N) * 100)
    y = round.(rand(Float32, N) * 100)
    alpha = 2.5

    # CUDA.jl version
    function axpy_cuda_kernel(alpha, x, y)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if i <= length(x)
            @inbounds x[i] += alpha * y[i]
        end
        return nothing
    end

    function axpy_cuda(SIZE, alpha, x, y)
        maxPossibleThreads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
        threads = min(SIZE, maxPossibleThreads)
        blocks = ceil(Int, SIZE / threads)
        CUDA.@sync @cuda threads = threads blocks = blocks axpy_cuda_kernel(alpha, x, y)
    end


    x_device = CUDA.CuArray(x)
    y_device = CUDA.CuArray(y)

    for i in 1:11
        @time axpy_cuda(N, alpha, x_device, y_device)
    end

    # JACCCUDA version 
    function axpy(i, alpha, x, y)
        if i <= length(x)
            @inbounds x[i] += alpha * y[i]
        end
    end

    x_device_JACC = JACC.Array(x)
    y_device_JACC = JACC.Array(y)

    for i in 1:11
        @time JACC.parallel_for(N, axpy, alpha, x_device_JACC, y_device_JACC)
    end
end