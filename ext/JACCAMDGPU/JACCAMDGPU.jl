

module JACCAMDGPU

using JACC, AMDGPU

function JACC.parallel_for(N::I, f::F, x...) where {I<:Integer,F<:Function}
    # @TODO get the gridsize and groupsize
    AMDGPU.AMDGPU.@roc gridsize = N groupsize = N _parallel_for_amdgpu(f, d_a)
end

function _parallel_for_amdgpu(f, d_a)
    i = (AMDGPU.workgroupIdx().x - 1) * AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x
    f(i, d_a)
    return nothing
end

function __init__()
    const JACC.Array = AMDGPU.ROCArray{T,N} where {T,N}
end

end # module JACCAMDGPU
