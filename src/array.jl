
function zeros(::ThreadsBackend, T, dims...)
    return Base.zeros(T, dims...)
end

function ones(::ThreadsBackend, T, dims...)
    return Base.ones(T, dims...)
end

zeros(::Type{T}, dims...) where {T} = zeros(default_backend(), T, dims...)
ones(::Type{T}, dims...) where {T} = ones(default_backend(), T, dims...)

zeros(dims...) = zeros(default_float(), dims...)
ones(dims...) = ones(default_float(), dims...)
