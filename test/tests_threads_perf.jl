import JACC
using Test


@testset "perf-AXPY" begin

    N = 300_000_000
    ntimes = 6
    x = round.(rand(Float32, N) * 100)
    y = round.(rand(Float32, N) * 100)
    alpha = 2.5

    x_JACC = Array(x)
    y_JACC = Array(y)

    # Threads version
    function axpy_threads(N, alpha, x, y)
        Threads.@threads for i in 1:N
            @inbounds x[i] += alpha * y[i]
        end
        return nothing
    end

    for i in 1:ntimes
        @time axpy_threads(N, alpha, x, y)
    end

    # JACC version
    function axpy(i, alpha, x, y)
        if i <= length(x)
            @inbounds x[i] += alpha * y[i]
        end
    end

    for i in 1:ntimes
        @time JACC.parallel_for(ThreadsBackend(), N, axpy, alpha, x_JACC, y_JACC)
    end
end
