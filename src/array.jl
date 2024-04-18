
@inline function zeros(T, dims...)
    _zeros_impl(BackendTag(), T, dims...)
end

@inline function ones(T, dims...)
    _ones_impl(BackendTag(), T, dims...)
end
