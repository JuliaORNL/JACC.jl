
function JACC.zeros(::oneAPIBackend, ::Type{T}, dims...) where {T}
    return oneAPI.zeros(T, dims...)
end

function JACC.ones(::oneAPIBackend, ::Type{T}, dims...) where {T}
    return oneAPI.ones(T, dims...)
end
