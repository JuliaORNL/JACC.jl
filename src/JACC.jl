__precompile__(false)
module JACC

using Atomix: @atomic
# module to set back end preferences 
include("JACCPreferences.jl")
include("helper.jl")
# overloaded array functions
include("array.jl")


include("JACCBLAS.jl")
using .BLAS

export parallel_for

function parallel_for(N, f::Function, x...)
	return parallel_for(JACCPreferences.JACC_BACKEND_TYPE(), N, f, x...)
end

function parallel_reduce(N, f::Function, x...)
	return parallel_reduce(JACCPreferences.JACC_BACKEND_TYPE(), N, f, x...)
end

function __init__()
end

end # module JACC
