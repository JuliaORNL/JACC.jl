
function JACC.zeros(::Val{:amdgpu}, T, dims...)
    return AMDGPU.zeros(T, dims...)
end

function JACC.ones(::Val{:amdgpu}, T, dims...)
    return AMDGPU.ones(T, dims...)
end
