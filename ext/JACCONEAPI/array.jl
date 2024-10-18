
function JACC.zeros(::oneAPIBackend, T, dims...)
    return oneAPI.zeros(T, dims...)
end

function JACC.ones(::oneAPIBackend, T, dims...)
    return oneAPI.ones(T, dims...)
end
