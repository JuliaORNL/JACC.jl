
function JACC.zeros(T, dims...)
    return AMDGPU.zeros(T, dims...)
end

function JACC.ones(T, dims...)
    return AMDGPU.ones(T, dims...)
end

function JACC.zeros(dims...)
    return AMDGPU.zeros(dims...)
end

function JACC.ones(dims...)
    return AMDGPU.ones(dims...)
end
