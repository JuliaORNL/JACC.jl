
zeros(::Type{T}, dims...) where {T} = zeros(default_backend(), T, dims...)
ones(::Type{T}, dims...) where {T} = ones(default_backend(), T, dims...)

zeros(dims...) = zeros(default_float(), dims...)
ones(dims...) = ones(default_float(), dims...)

fill(value, dims...) = fill(default_backend(), value, dims...)
