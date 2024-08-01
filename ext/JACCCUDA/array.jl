
function JACC._zeros_impl(::CUDABackendTag, T, dims...)
    return CUDA.zeros(T, dims...)
end

function JACC._ones_impl(::CUDABackendTag, T, dims...)
    return CUDA.ones(T, dims...)
end
