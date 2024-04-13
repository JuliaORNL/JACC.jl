module JACC

# module to set back end preferences 
include("JACCPreferences.jl")
include("helper.jl")
include("JACCArgsList.jl")
include("parallel_for.jl")
include("parallel_reduce.jl")
#export Array
export parallel_for

#global Array

using PackageExtensionCompat
function __init__()
  @require_extensions
  # @info("Using JACC backend: $(JACCPreferences.backend)")

  # if JACCPreferences.backend == "threads"
  #   const JACC.Array = Base.Array{T,N} where {T,N}
  # end
end


end # module JACC
