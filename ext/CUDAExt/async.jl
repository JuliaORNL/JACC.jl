module Async

import Base: Callable
using JACC, CUDA

@inline ndevices() = length(devices())
@inline get_device_id(id) = (id - 1) % ndevices()
@inline set_relative_device!(id) = CUDA.device!(get_device_id(id))

JACC.Async.ndev(::CUDABackend) = ndevices()

function JACC.Async.zeros(::CUDABackend, T, id, dims...)
    set_relative_device!(id)
    ret = CUDA.zeros(T, dims...)
    CUDA.device!(0)
    return ret
end

function JACC.Async.ones(::CUDABackend, T, id, dims...)
    set_relative_device!(id)
    ret = CUDA.ones(T, dims...)
    CUDA.device!(0)
    return ret
end

function JACC.Async.fill(::CUDABackend, id, value, dims...)
    set_relative_device!(id)
    ret = CUDA.fill(value, dims...)
    CUDA.device!(0)
    return ret
end

function JACC.Async.synchronize(::CUDABackend)
    ndev = ndevices()
    for i in 1:ndev
        CUDA.device!(i - 1)
        CUDA.synchronize()
    end
    CUDA.device!(0)
end

function JACC.Async.synchronize(::CUDABackend, id::Integer)
    set_relative_device!(id)
    CUDA.synchronize()
    CUDA.device!(0)
end

function JACC.Async.array(::CUDABackend, id::Integer, x::AbstractArray)
    set_relative_device!(id)
    ret = JACC.array(CUDABackend(), x)
    CUDA.device!(0)
    return ret
end

function JACC.Async.parallel_for(
        ::CUDABackend, id::Integer, dims::JACC.IDims, f::Callable, x...)
    set_relative_device!(id)
    JACC.parallel_for(JACC.LaunchSpec{CUDABackend}(; sync = false), dims, f, x...)
    CUDA.device!(0)
end

function JACC.Async.parallel_reduce(
        ::CUDABackend, id::Integer, dims::JACC.IDims,
        op::Callable, f::Callable, x...; init)
    set_relative_device!(id)
    reducer = JACC.ParallelReduce{CUDABackend, typeof(init)}(; dims = dims,
        op = op, init = init, sync = false)
    reducer(f, x...)
    ret = reducer.workspace.ret
    CUDA.device!(0)
    return ret
end

end # module Async
