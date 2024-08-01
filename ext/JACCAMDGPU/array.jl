
function JACC._zeros_impl(::AMDGPUBackendTag, T, dims...)
    return AMDGPU.zeros(T, dims...)
end

function JACC._ones_impl(::AMDGPUBackendTag, T, dims...)
    return AMDGPU.ones(T, dims...)
end
