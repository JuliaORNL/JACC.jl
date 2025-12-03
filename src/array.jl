
array_type() = array_type(default_backend())

to_device(x::AbstractArray) = convert(array_type(), x)
to_host(x::AbstractArray) = convert(Base.Array, x)

array(x::AbstractArray) = to_device(x)

array(::Type{T}, dims) where {T} = array_type(){T,length(dims)}(undef, dims)
array(::Type{T}, dims...) where {T} = array(T, dims)
array(dims) = array(default_float(), dims)
array(dims...) = array(dims)
array(; type = default_float(), dims = 0) = array(type, dims)

zeros(::Type{T}, dims...) where {T} = zeros(default_backend(), T, dims...)
ones(::Type{T}, dims...) where {T} = ones(default_backend(), T, dims...)

zeros(dims...) = zeros(default_float(), dims...)
ones(dims...) = ones(default_float(), dims...)

fill(value, dims...) = fill(default_backend(), value, dims...)
