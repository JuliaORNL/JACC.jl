using CUDA
import JACC
using Test


@testset "perf-AXPY" begin

    N = 300_000_000
    x = round.(rand(Float32, N) * 100)
    y = round.(rand(Float32, N) * 100)
    alpha = 2.5

    x_JACC = JACC.Array(x)
    y_JACC = JACC.Array(y)

    # Threads version
    function axpy_threads(N, alpha, x, y)
        Threads.@threads for i in 1:N
            @inbounds x[i] += alpha * y[i]
        end
        return nothing
    end

    for i in 1:11
        @time axpy_threads(N, alpha, x, y)
    end

    # JACC version 
    function axpy(i, alpha, x, y)
        if i <= length(x)
            @inbounds x[i] += alpha * y[i]
        end
    end

    for i in 1:11
        @time JACC.parallel_for(N, axpy, alpha, x_JACC, y_JACC)
    end
end