function JACC.zeros(::MetalBackend, ::Type{T}, dims...) where {T}
    return Metal.zeros(T, dims...)
end

function JACC.ones(::MetalBackend, ::Type{T}, dims...) where {T}
    return Metal.ones(T, dims...)
end

function JACC.fill(::MetalBackend, value, dims...)
    return Metal.fill(value, dims...)
end
