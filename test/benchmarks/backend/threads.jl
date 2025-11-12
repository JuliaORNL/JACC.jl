
function axpy_threads(SIZE::Integer, alpha, x, y)
    Threads.@threads for i in 1:SIZE
        axpy(i, alpha, x, y)
    end
end

function axpy_threads((SIZE, SIZE)::NTuple{2, Integer}, alpha, x, y)
    Threads.@threads for j in 1:SIZE
        for i in 1:SIZE
            axpy(i, j, alpha, x, y)
        end
    end
end

push!(JACCBench.axpy_comps, axpy_threads)

function dot_threads(SIZE::Integer, x, y)
    tmp = zeros(Threads.nthreads())
    ret = zeros(1)
    Threads.@threads for i in 1:SIZE
        tmp[Threads.threadid()] = tmp[Threads.threadid()] .+ x[i] * y[i]
    end
    for i in 1:Threads.nthreads()
        ret = ret .+ tmp[i]
    end
    return ret
end

function dot_threads((M, N)::NTuple{2, Integer}, x, y)
    tmp = zeros(Threads.nthreads())
    ret = zeros(1)
    Threads.@threads for j in 1:N
        for i in 1:M
            tmp[Threads.threadid()] = tmp[Threads.threadid()] .+
                                      x[i, j] * y[i, j]
        end
    end
    for i in 1:Threads.nthreads()
        ret = ret .+ tmp[i]
    end
    return ret
end

push!(JACCBench.dot_comps, dot_threads)
