
function JACC.zeros(::oneAPIBackend, ::Type{T}, dims...) where {T}
    return oneAPI.zeros(T, dims...)
end

function JACC.ones(::oneAPIBackend, ::Type{T}, dims...) where {T}
    return oneAPI.ones(T, dims...)
end

# TODO: move these to main implementation and use default_float_type()
function JACC.zeros(::oneAPIBackend, dims...)
    return oneAPI.zeros(Float32, dims...)
end
function JACC.ones(::oneAPIBackend, dims...)
    return oneAPI.ones(Float32, dims...)
end
