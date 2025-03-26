import CUDA

@testset "TestBackend" begin
    @test JACC.backend == "cuda"
end

@testset "zeros_type" begin
    using CUDA
    N = 10
    x = JACC.zeros(Float64, N)
    @test typeof(x) == CUDA.CuArray{Float64, 1, CUDA.DeviceMemory}
    @test eltype(x) == Float64
end

@testset "ones_type" begin
    using CUDA
    N = 10
    x = JACC.ones(Float64, N)
    @test typeof(x) == CUDA.CuArray{Float64, 1, CUDA.DeviceMemory}
    @test eltype(x) == Float64
end

#@testset "JACC.Multi" begin

#    x = ones(1_000)
#    y = ones(1_000)
#    jx = JACC.Multi.Array(x)
#    jy = JACC.Multi.Array(y)
#    jxresult = ones(1_000)
#    alpha = 2.0

#    function seq_axpy(N, alpha, x, y)
#        for i in 1:N
#            @inbounds x[i] += alpha * y[i]
#        end
#    end

#    function multi_axpy(dev_id, i, alpha, x, y)
#	    @inbounds x[dev_id][i] += alpha * y[dev_id][i]
#    end

#    function seq_dot(N, x, y)
#        r = 0.0
#        for i in 1:N
#            @inbounds r += x[i] * y[i]
#        end
#        return r
#    end

#    function multi_dot(dev_id, i, x, y)
#        @inbounds x[dev_id][i] * y[dev_id][i]
#    end

#ref_result = seq_axpy(1_000, x, y)
#jresult = JACC.Multi.parallel_reduce(1_000, multi_dot, jx[1], jy[1])
#    seq_axpy(1_000, alpha, x, y)
#    JACC.Multi.parallel_for(1_000, multi_axpy, alpha, jx[1], jy[1])

#result = Base.Array(jresult)

#@test jresult[1]≈ref_result rtol=1e-8
#end
