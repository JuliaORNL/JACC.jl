
function JACC.zeros(::ThreadsBackend, T, dims...)
    return Base.zeros(T, dims...)
end

function JACC.ones(::ThreadsBackend, T, dims...)
    return Base.ones(T, dims...)
end
