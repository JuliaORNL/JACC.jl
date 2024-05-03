__precompile__(false)
module JACC

import Atomix: @atomic
# module to set back end preferences 
include("JACCPreferences.jl")
include("helper.jl")
# overloaded array functions
include("array.jl")

export @atomic
export parallel_for

function parallel_for(N, f::Function, x...)
	return parallel_for(JACC.JAT, N, f, x...)
end

function parallel_reduce(N, f::Function, x...)
	return parallel_reduce(JACC.JAT, N, f, x...)
end

function __init__()
    # const JACC.Array = Base.Array{T, N} where {T, N}
    const JACC.JAT = JACCArrayType{Array}()
end

end # module JACC
