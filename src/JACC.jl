module JACC

# module to set back end preferences 
include("JACCPreferences.jl")

export Array
export parallel_for

global Array

function parallel_for(N::I, f::F, x...) where {I<:Integer,F<:Function}
    Threads.@threads for i in 1:N
        f(i, x...)
    end
end

function __init__()
    @info("Using JACC backend: $(JACCPreferences.backend)")

    if JACCPreferences.backend == "threads"
        const JACC.Array = Base.Array{T,N} where {T,N}
    end
end


end # module JACC
