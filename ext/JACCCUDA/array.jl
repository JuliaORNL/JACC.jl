
function JACC.zeros(T, dims...)
    return CUDA.zeros(T, dims...)
end

function JACC.ones(T, dims...)
    return CUDA.ones(T, dims...)
end
