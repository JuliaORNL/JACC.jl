
function JACC.zeros(T, dims...)
    return AMDGPU.zeros(T, dims...)
end

function JACC.ones(T, dims...)
    return AMDGPU.ones(T, dims...)
end
