module MetalExt

import Base: Callable
using JACC, Metal

# overloaded array functions
include("array.jl")

JACC.get_backend(::Val{:metal}) = MetalBackend()

default_stream() = Metal.global_queue(Metal.device())

JACC.default_stream(::Type{MetalBackend}) = default_stream()

function JACC.synchronize(::MetalBackend)
    Metal.synchronize()
end

function dummy() end

@inline function _kernel_maxthreads(N, f, x...)
    kernel = @metal launch=false dummy()
    max_threads = Int32(kernel.pipeline.maxTotalThreadsPerThreadgroup)
    return max_threads
end

function JACC.parallel_for(f, ::MetalBackend, N::Integer, x...)
    max_threads_per_group = _kernel_maxthreads(N, f, x)
    #max_threads_per_group = 1024
    threads = min(N, max_threads_per_group)
    blocks = cld(N, threads)
    @metal threads=threads groups=blocks _parallel_for_metal(N, f, x...)
    Metal.synchronize()
end

function JACC.parallel_reduce(
        ::MetalBackend, N::Integer, op, f::Callable, x...; init)
    items = 1024
    groups = cld(N, items)
    ret = fill!(Metal.MtlArray{typeof(init)}(undef, groups), init)
    rret = Metal.MtlArray([init])
    Metal.@sync @metal threads=items groups=groups _parallel_reduce_metal(
        Val(items), N, op, ret, f, x...)
    Metal.@sync @metal threads=items groups=1 reduce_kernel_metal(
        Val(items), groups, op, ret, rret)
    return Base.Array(rret)[]
end

function _parallel_for_metal(N, f, x...)
    i = Metal.thread_position_in_grid_1d()
    # if i > N
    #     return nothing
    #end
    f(i, x...)
    return nothing
end

function JACC.default_float(::MetalBackend)
    return Float32
end

JACC.array_type(::MetalBackend) = Metal.MtlArray

JACC.array(::MetalBackend, x::Base.Array) = Metal.MtlArray(x)

end # module MetalExt