
function JACC.zeros(::Val{:oneapi}, T, dims...)
    return oneAPI.zeros(T, dims...)
end

function JACC.ones(::Val{:oneapi}, T, dims...)
    return oneAPI.ones(T, dims...)
end
