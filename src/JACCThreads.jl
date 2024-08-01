struct ThreadsBackendTag end

function _zeros_impl(::ThreadsBackendTag, T, dims...)
    return Base.zeros(T, dims...)
end

function _ones_impl(::ThreadsBackendTag, T, dims...)
    return Base.ones(T, dims...)
end

function _parallel_for_impl(::ThreadsBackendTag, N::I, f::F, x...) where {I <: Integer, F <: Function}
    @maybe_threaded for i in 1:N
        f(i, x...)
    end
end

function _parallel_for_impl(::ThreadsBackendTag, (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    @maybe_threaded for j in 1:N
        for i in 1:M
            f(i, j, x...)
        end
    end
end

function _parallel_for_impl(::ThreadsBackendTag,
        (L, M, N)::Tuple{I, I, I}, f::F, x...) where {
        I <: Integer, F <: Function}
    # only threaded at the first level (no collapse equivalent)
    @maybe_threaded for k in 1:N
        for j in 1:M
            for i in 1:L
                f(i, j, k, x...)
            end
        end
    end
end

function _parallel_reduce_impl(::ThreadsBackendTag, N::I, f::F, x...) where {I <: Integer, F <: Function}
    tmp = zeros(Threads.nthreads())
    ret = zeros(1)
    @maybe_threaded for i in 1:N
        tmp[Threads.threadid()] = tmp[Threads.threadid()] .+ f(i, x...)
    end
    for i in 1:Threads.nthreads()
        ret = ret .+ tmp[i]
    end
    return ret
end

function _parallel_reduce_impl(::ThreadsBackendTag, (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    tmp = zeros(Threads.nthreads())
    ret = zeros(1)
    @maybe_threaded for j in 1:N
        for i in 1:M
            tmp[Threads.threadid()] = tmp[Threads.threadid()] .+ f(i, j, x...)
        end
    end
    for i in 1:Threads.nthreads()
        ret = ret .+ tmp[i]
    end
    return ret
end

function __init__()
    const JACC.Array = Base.Array{T, N} where {T, N}
    const JACC.BackendTag = ThreadsBackendTag
end
