module Async

import Base: Callable
using JACC, oneAPI

@inline ndevices() = length(oneAPI.devices())
@inline get_device_id(id) = ((id - 1) % ndevices()) + 1
@inline set_relative_device!(id) = oneAPI.device!(get_device_id(id))

JACC.Async.ndev(::oneAPIBackend) = ndevices()

function JACC.Async.zeros(::oneAPIBackend, T, id, dims...)
    set_relative_device!(id)
    ret = oneAPI.zeros(T, dims...)
    oneAPI.device!(1)
    return ret
end

function JACC.Async.ones(::oneAPIBackend, T, id, dims...)
    set_relative_device!(id)
    ret = oneAPI.ones(T, dims...)
    oneAPI.device!(1)
    return ret
end

function JACC.Async.fill(::oneAPIBackend, id, value, dims...)
    set_relative_device!(id)
    ret = oneAPI.fill(value, dims...)
    oneAPI.device!(1)
    return ret
end

function JACC.Async.synchronize(::oneAPIBackend)
    ndev = ndevices()
    for i in 1:ndev
        oneAPI.device!(i)
        oneAPI.synchronize()
    end
    oneAPI.device!(1)
end

function JACC.Async.synchronize(::oneAPIBackend, id::Integer)
    set_relative_device!(id)
    oneAPI.synchronize()
    oneAPI.device!(1)
end

function JACC.Async.array(::oneAPIBackend, id::Integer, x::AbstractArray)
    set_relative_device!(id)
    ret = JACC.array(oneAPIBackend(), x)
    oneAPI.device!(1)
    return ret
end

function JACC.Async.parallel_for(
        ::oneAPIBackend, id::Integer, dims::JACC.IDims, f::Callable, x...)
    set_relative_device!(id)
    JACC.parallel_for(JACC.LaunchSpec{oneAPIBackend}(; sync = false), dims, f, x...)
    oneAPI.device!(1)
end

function JACC.Async.parallel_reduce(::oneAPIBackend, id::Integer,
        dims::JACC.IDims, op::Callable, f::Callable, x...; init)
    set_relative_device!(id)
    reducer = JACC.ParallelReduce{oneAPIBackend, typeof(init)}(; dims = dims,
        op = op, init = init, sync = false)
    reducer(f, x...)
    ret = reducer.workspace.ret
    oneAPI.device!(1)
    return ret
end

end # module Async
