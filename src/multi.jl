module Multi

import Base: Callable
using JACC

function ndev()
    return ndev(JACC.default_backend())
end

function part_length(x)
    return part_length(JACC.default_backend(), x)
end

function device_id(x)
    return device_id(JACC.default_backend(), x)
end

function multi_array_type()
    return multi_array_type(JACC.default_backend())
end

function array(x::Base.Array; ghost_dims = 0)
    return array(JACC.default_backend(), x; ghost_dims=ghost_dims)
end

function ghost_shift(idx::Union{Integer,NTuple{2,Integer}}, arr)
    return ghost_shift(JACC.default_backend(), idx, arr)
end

function sync_ghost_elems!(arr)
    return sync_ghost_elems!(JACC.default_backend(), arr)
end

function copy!(dest, src)
    return copy!(JACC.default_backend(), dest, src)
end

function parallel_for(N::Integer, f::Callable, x...)
    return parallel_for(JACC.default_backend(), N, f, x...)
end

function parallel_for((M, N)::NTuple{2, Integer}, f::Callable, x...)
    return parallel_for(JACC.default_backend(), (M, N), f, x...)
end

function parallel_reduce(N::Integer, f::Callable, x...)
    return parallel_reduce(JACC.default_backend(), N, f, x...)
end

function parallel_reduce((M, N)::NTuple{2, Integer}, f::Callable, x...)
    return parallel_reduce(JACC.default_backend(), (M, N), f, x...)
end
end # module Multi
