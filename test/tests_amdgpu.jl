import AMDGPU
import JACC
using Test

@testset "TestBackend" begin
    @test JACC.JACCPreferences.backend == "amdgpu"
end

@testset "zeros_type" begin
    N = 10
    x = JACC.zeros(Float64, N)
    @test typeof(x) == AMDGPU.ROCArray{Float64, 1, AMDGPU.Runtime.Mem.HIPBuffer}
    @test eltype(x) == Float64
end

@testset "ones_type" begin
    N = 10
    x = JACC.ones(Float64, N)
    @test typeof(x) == AMDGPU.ROCArray{Float64, 1, AMDGPU.Runtime.Mem.HIPBuffer}
    @test eltype(x) == Float64
end

#@testset "JACC.BLAS" begin
#    function seq_axpy(N, alpha, x, y)
#        for i in 1:N
#            @inbounds x[i] += alpha * y[i]
#        end
#    end

#    function seq_dot(N, x, y)
#        r = 0.0
#        for i in 1:N
#            @inbounds r += x[i] * y[i]
#        end
#        return r
#    end

#    x = ones(1_000)
#    y = ones(1_000)
#    jx = JACC.ones(1_000)
#    jy = JACC.ones(1_000)
#    alpha = 2.0

#    seq_axpy(1_000, alpha, x, y)
#    ref_result = seq_dot(1_000, x, y)

#    JACC.BLAS.axpy(1_000, alpha, jx, jy)
#    jresult = JACC.BLAS.dot(1_000, jx, jy)
#    result = Array(jresult)     

#    @test result[1]≈ref_result rtol=1e-8
#    x = ones(1_000)
#    y = ones(1_000)
#    y1 = y*2
#    jx = JACC.ones(1_000)
#    jy = JACC.ones(1_000)
#    jy1 = jy*2
#    alpha = 2.0
# 
#   function seq_axpy(N, alpha, x, y)
#       for i in 1:N
#           @inbounds x[i] += alpha * y[i]
#       end
#   end
#    
#   function seq_dot(N, x, y)
#       r = 0.0
#       for i in 1:N
#           @inbounds r += x[i] * y[i]
#       end
#       return r
#   end
#   function seq_scal(N, alpha, x)
#        for i in 1:N
#            @inbounds x[i] = alpha * x[i]
#        end
#    end
# 
#    function seq_asum(N, x)
#        r = 0.0
#        for i in 1:N
#            @inbounds r += abs(x[i])
#        end
#        return r
#    end
# 
#    function seq_nrm2(N, x)
#        sum_sq = 0.0
#        for i in 1:N
#            @inbounds sum_sq += x[i]*x[i]
#        end
#        r = sqrt(sum_sq)
#        return r
#    end
# 
#    function seq_swap(N, x, y1)
#        for i in 1:N
#            @inbounds t = x[i]
#            @inbounds x[i] = y1[i]
#            @inbounds y1[i] = t
#        end       
#    end

# ref_result = seq_axpy(1_000, alpha, x, y)
# ref_result = seq_dot(1_000, x, y)   
# # jresult = JACC.BLAS.axpy(1_000, alpha, jx, jy)
# jresult = JACC.BLAS.dot(1_000, jx, jy)
# result = Array(jresult)       
# @test result[1]≈ref_result rtol=1e-8

# seq_scal(1_000, alpha, x)
# JACC.BLAS.scal(1_000, alpha, jx)
# @test x≈Array(jx) rtol=1e-8 

# seq_axpy(1_000, alpha, x, y)
# JACC.BLAS.axpy(1_000, alpha, jx, jy)
# @test x≈Array(jx) atol=1e-8

# r1 = seq_dot(1_000, x, y) 
# r2 = JACC.BLAS.dot(1_000, jx, jy)
# @test r1≈Array(r2)[1] atol=1e-8 

# r1 = seq_asum(1_000, x)
# r2 = JACC.BLAS.asum(1_000, jx)
# r1 = seq_nrm2(1_000, x)
# r2 = JACC.BLAS.nrm2(1_000, jx)
# @test r1≈Array(r2)[1] atol=1e-8

# seq_swap(1_000, x, y1)  
# JACC.BLAS.swap(1_000, jx, jy1)
# @test x == Array(jx)
# @test y1 == Array(jy1)

# end

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
