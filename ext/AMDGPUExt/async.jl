module Async

import Base: Callable
using JACC, AMDGPU
using AMDGPUExt: AMDGPUBackend

@inline ndevices() = length(AMDGPU.devices())
@inline get_device_id(id) = ((id - 1) % ndevices()) + 1
@inline set_relative_device!(id) = AMDGPU.device_id!(get_device_id(id))

JACC.Async.ndev(::AMDGPUBackend) = ndevices()

function JACC.Async.zeros(::AMDGPUBackend, T, id, dims...)
    set_relative_device!(id)
    ret = AMDGPU.zeros(T, dims...)
    AMDGPU.device_id!(1)
    return ret
end

function JACC.Async.ones(::AMDGPUBackend, T, id, dims...)
    set_relative_device!(id)
    ret = AMDGPU.ones(T, dims...)
    AMDGPU.device_id!(1)
    return ret
end

function JACC.Async.fill(::AMDGPUBackend, id, value, dims...)
    set_relative_device!(id)
    ret = AMDGPU.fill(value, dims...)
    AMDGPU.device_id!(1)
    return ret
end

function JACC.Async.synchronize(::AMDGPUBackend)
    ndev = ndevices()
    for i in 1:ndev
        AMDGPU.device_id!(i)
        AMDGPU.synchronize()
    end
    AMDGPU.device_id!(1)
end

function JACC.Async.synchronize(::AMDGPUBackend, id::Integer)
    set_relative_device!(id)
    AMDGPU.synchronize()
    AMDGPU.device_id!(1)
end

function JACC.Async.array(::AMDGPUBackend, id::Integer, x::AbstractArray)
    set_relative_device!(id)
    ret = JACC.array(AMDGPUBackend(), x)
    AMDGPU.device_id!(1)
    return ret
end

function JACC.Async.parallel_for(
        ::AMDGPUBackend, id::Integer, dims::JACC.IDims, f::Callable, x...)
    set_relative_device!(id)
    JACC.parallel_for(JACC.LaunchSpec{AMDGPUBackend}(; sync = false), dims, f, x...)
    AMDGPU.device_id!(1)
end

function JACC.Async.parallel_reduce(::AMDGPUBackend, id::Integer,
        dims::JACC.IDims, op::Callable, f::Callable, x...; init)
    set_relative_device!(id)
    reducer = JACC.ParallelReduce{AMDGPUBackend, typeof(init)}(; dims = dims,
        op = op, init = init, sync = false)
    reducer(f, x...)
    ret = reducer.workspace.ret
    AMDGPU.device_id!(1)
    return ret
end

end # module Async
