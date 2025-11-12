
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
    tmp = zeros(eltype(x), Threads.nthreads())
    nchunks = Threads.nthreads()
    chunks = collect(Base.Iterators.partition(1:SIZE, cld(SIZE, nchunks)))
    nchunks = length(chunks)
    Threads.@threads :static for n in 1:nchunks
        @inbounds begin
            tp = tmp[n]
            for i in chunks[n]
                tp += x[i] * y[i]
            end
            tmp[n] = tp
        end
    end
    return sum(tmp[1:nchunks])
end

function dot_threads((M, N)::NTuple{2, Integer}, x, y)
    tmp = zeros(eltype(x), Threads.nthreads())
    nchunks = Threads.nthreads()
    ids = CartesianIndices((1:M, 1:N))
    chunks = collect(Base.Iterators.partition(ids, cld(length(ids), nchunks)))
    nchunks = length(chunks)
    Threads.@threads :static for n in 1:nchunks
        @inbounds begin
            tp = tmp[n]
            for ij in chunks[n]
                tp += x[ij] * y[ij]
            end
            tmp[n] = tp
        end
    end
    return sum(tmp[1:nchunks])
end

push!(JACCBench.dot_comps, dot_threads)
