
macro maybe_threaded(ex)
    if Threads.nthreads() == 1
        return esc(ex)
    else
        return esc(:(Threads.@threads :static $ex))
    end
end

struct JACCArrayType{T}
end

arraytype() = arraytype(Val(Symbol(JACCPreferences.backend)))
arraytype(::Val{:threads}) = Array
arraytype(::Val{T}) where T = error("The backend $(T) is either not recognized or the associated package is not loaded.")
arraytype(J::JACCArrayType) = arraytype(typeof(J))
arraytype(::Type{<:JACCArrayType{T}}) where {T} = T
