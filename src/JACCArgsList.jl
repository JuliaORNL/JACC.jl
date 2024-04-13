## This is a helper compiled type to keep track of the arraytype
## Such that it is possible to utilize multiple dispatch.
## holding the args type (array vs tuple) is a julia related issue
## as calling `tuple(([1,2,3], ))` attempts to convert the array to a tuple which
## is unfavorable. If this is fixed all calls can use tuple.
struct JACCArgsList{ArrayT<:AbstractArray}
  args
end

function JACCArgsList{ArrayT}(args...) where {ArrayT} 
  ar = tuple(args...)
  return JACCArgsList{ArrayT}(ar)
end

function JACCArgsList{ArrayT}(arg::AbstractArray) where {ArrayT<:AbstractArray} 
  return JACCArgsList{ArrayT}((arg,))
end

function JACCArgsList(arraytype::Type, args)
  return JACCArgsList{arraytype}(args)
end

arraytype(::JACCArgsList{T}) where {T} = T

Base.iterate(JA::JACCArgsList) = iterate(JA.args)
Base.iterate(JA::JACCArgsList, i::Core.Any) = iterate(JA.args, i)