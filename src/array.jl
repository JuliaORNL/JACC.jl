
function zeros(::ThreadsBackend, T, dims...)
    return Base.zeros(T, dims...)
end

function ones(::ThreadsBackend, T, dims...)
    return Base.ones(T, dims...)
end

zeros(T, dims...) = zeros(default_backend(), T, dims...)

ones(T, dims...) = ones(default_backend(), T, dims...)
