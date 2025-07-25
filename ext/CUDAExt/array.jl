
function JACC.zeros(::CUDABackend, T, dims...)
    return CUDA.zeros(T, dims...)
end

function JACC.ones(::CUDABackend, T, dims...)
    return CUDA.ones(T, dims...)
end

function JACC.fill(::CUDABackend, value, dims...)
    CUDA.fill(value, dims...)
end
