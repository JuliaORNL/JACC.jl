
function JACC.zeros(::Val{:cuda}, T, dims...)
    return CUDA.zeros(T, dims...)
end

function JACC.ones(::Val{:cuda}, T, dims...)
    return CUDA.ones(T, dims...)
end
