
function JACC._zeros_impl(::oneAPIBackendTag, T, dims...)
    return oneAPI.zeros(T, dims...)
end

function JACC._ones_impl(::oneAPIBackendTag, T, dims...)
    return oneAPI.ones(T, dims...)
end
