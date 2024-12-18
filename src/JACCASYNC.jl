module Async

using JACC
import JACC: ThreadsBackend

function Array(::ThreadsBackend, queue_id::I, x::Base.Array{T,N}) where {I <: Integer, T, N}
    return JACC.Array(x)
end

function copy(::ThreadsBackend, queue_id_dest::I, x::Base.Array{T,N}, queue_id_orig::I, y::Base.Array{T,N}) where {I <: Integer, T, N}
    copyto!(x, y)
end

function parallel_for(::ThreadsBackend, queue_id::I, N::I, f::F, x...) where {I <: Integer, F <: Function}
    JACC.parallel_for(N, f, x...)
end

function parallel_for(::ThreadsBackend, queue_id::I, (M, N)::Tuple{I,I}, f::F, x...) where {I <: Integer, F <: Function}
    JACC.parallel_for((M, N), f, x...)
end

function parallel_reduce(::ThreadsBackend, queue_id::I, N::I, f::F, x...) where {I <: Integer, F <: Function}
    return JACC.parallel_reduce(N, f, x...)
end

function parallel_reduce(::ThreadsBackend, queue_id::I, (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    return JACC.parallel_reduce((M, N), f, x...)
end

function synchronize(::ThreadsBackend)
end

function Array(queue_id::I, x::Base.Array{T,N}) where {I <: Integer, T, N}
    return Array(JACC.default_backend(), queue_id, x)
end

function copy(queue_id_dest::I, x::Base.Array{T,N}, queue_id_orig::I, y::Base.Array{T,N}) where {I <: Integer, T, N}
    return copy(JACC.default_backend(), queue_id_dest, x, queue_id_orig, y)
end

function parallel_for(queue_id::I, N::I, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_for(JACC.default_backend(), queue_id, N, f, x...)
end

function parallel_for(queue_id::I, (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_for(JACC.default_backend(), queue_id, (M, N), f, x...)
end

function parallel_reduce(queue_id::I, N::I, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_reduce(JACC.default_backend(), queue_id, N, f, x...)
end

function parallel_reduce(queue_id::I, (M, N)::Tuple{I, I}, f::F, x...) where {I <: Integer, F <: Function}
    return parallel_reduce(JACC.default_backend(), queue_id, (M, N), f, x...)
end

function synchronize()
    return synchronize(JACC.default_backend())
end

end # module Async
