
zeros(T, dims...) = zeros(JACCPrefereces._backend_dispatchable, T, dims...)
function zeros(::Val{:threads}, T, dims...)
    return Base.zeros(T, dims...)
end

ones(T, dims...) = ones(JACCPrefereces._backend_dispatchable, T, dims...)
function ones(::Val{:threads}, T, dims...)
    return Base.ones(T, dims...)
end
