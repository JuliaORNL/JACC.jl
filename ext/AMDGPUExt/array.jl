
function JACC.zeros(::AMDGPUBackend, T, dims...)
    return AMDGPU.zeros(T, dims...)
end

function JACC.ones(::AMDGPUBackend, T, dims...)
    return AMDGPU.ones(T, dims...)
end
