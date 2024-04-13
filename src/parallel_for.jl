function parallel_for(N::Union{<:Integer, <:Tuple}, f::Function, a::JACCArgsList{<:AbstractArray})
  error("parallel_for not defined for the arraytype of $(arraytype(a))")
end

function parallel_for(N::Integer, f::Function, x::JACCArgsList{<:Array})
  @maybe_threaded for i in 1:N
    f(i, x...)
  end
end

function parallel_for((M, N)::Tuple{Integer,Integer}, f::Function, x::JACCArgsList{<:Array})
  @maybe_threaded for j in 1:N
    for i in 1:M
      f(i, j, x...)
    end
  end
end