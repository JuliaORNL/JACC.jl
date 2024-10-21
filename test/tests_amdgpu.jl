import AMDGPU
import JACC
using Test

@testset "TestBackend" begin
    @test JACC.JACCPreferences.backend == "amdgpu"
end

@testset "VectorAddLambda" begin
    function f(i, a)
        @inbounds a[i] += 5.0
    end

    N = 10
    dims = (N)
    a = round.(rand(Float32, dims) * 100)

    a_device = JACC.Array(a)
    JACC.parallel_for(N, f, a_device)

    a_expected = a .+ 5.0
    @test Array(a_device)≈a_expected rtol=1e-5
end

@testset "AXPY" begin
    function axpy(i, alpha, x, y)
        @inbounds x[i] += alpha * y[i]
    end

    function seq_axpy(N, alpha, x, y)
        @inbounds for i in 1:N
            x[i] += alpha * y[i]
        end
    end

    N = 10
    # Generate random vectors x and y of length N for the interval [0, 100]
    x = round.(rand(Float32, N) * 100)
    y = round.(rand(Float32, N) * 100)
    alpha = 2.5

    x_device = JACC.Array(x)
    y_device = JACC.Array(y)
    JACC.parallel_for(N, axpy, alpha, x_device, y_device)

    x_expected = x
    seq_axpy(N, alpha, x_expected, y)

    @test Array(x_device)≈x_expected rtol=1e-1
end

@testset "AtomicCounter" begin
    function axpy_counter!(i, alpha, x, y, counter)
        @inbounds x[i] += alpha * y[i]
        JACC.@atomic counter[1] += 1
    end

    N = Int32(10)
    # Generate random vectors x and y of length N for the interval [0, 100]
    alpha = 2.5

    x = JACC.Array(round.(rand(Float32, N) * 100))
    y = JACC.Array(round.(rand(Float32, N) * 100))
    counter = JACC.Array{Int32}([0])
    JACC.parallel_for(N, axpy_counter!, alpha, x, y, counter)

    @test Array(counter)[1] == N
end

@testset "zeros" begin
    N = 10
    x = JACC.zeros(Float64, N)
    @test typeof(x) == AMDGPU.ROCArray{Float64, 1, AMDGPU.Runtime.Mem.HIPBuffer}
    @test eltype(x) == Float64
    @test zeros(N)≈Array(x) rtol=1e-5

    function add_one(i, x)
        @inbounds x[i] += 1
    end

    JACC.parallel_for(N, add_one, x)
    @test ones(N)≈Array(x) rtol=1e-5
end

@testset "ones" begin
    N = 10
    x = JACC.ones(Float64, N)
    @test typeof(x) == AMDGPU.ROCArray{Float64, 1, AMDGPU.Runtime.Mem.HIPBuffer}
    @test eltype(x) == Float64
    @test ones(N)≈Array(x) rtol=1e-5

    function minus_one(i, x)
        @inbounds x[i] -= 1
    end

    JACC.parallel_for(N, minus_one, x)
    @test zeros(N)≈Array(x) rtol=1e-5
end

@testset "shared" begin
    N = 100
    alpha = 2.5
    x = JACC.ones(Float64, N)
    x_shared = JACC.ones(Float64, N)
    y = JACC.ones(Float64, N)

    function scal(i, x, y, alpha)
        @inbounds x[i] = y[i] * alpha
    end
    
    function scal_shared(i, x, y, alpha)
        y_shared = JACC.shared(y) 
        @inbounds x[i] = y_shared[i] * alpha
    end

    JACC.parallel_for(N, scal, x, y, alpha)
    JACC.parallel_for(N, scal_shared, x_shared, y, alpha)
    @test x≈x_shared rtol=1e-8
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
    x = ones(1_000)
    y = ones(1_000)
    y1 = y*2
    jx = JACC.ones(1_000)
    jy = JACC.ones(1_000)
    jy1 = jy*2
    alpha = 2.0

   function seq_axpy(N, alpha, x, y)
       for i in 1:N
           @inbounds x[i] += alpha * y[i]
       end
   end
    
   function seq_dot(N, x, y)
       r = 0.0
       for i in 1:N
           @inbounds r += x[i] * y[i]
       end
       return r
   end
   function seq_scal(N, alpha, x)
        for i in 1:N
            @inbounds x[i] = alpha * x[i]
        end
    end

    function seq_asum(N, x)
        r = 0.0
        for i in 1:N
            @inbounds r += abs(x[i])
        end
        return r
    end

    function seq_nrm2(N, x)
        sum_sq = 0.0
        for i in 1:N
            @inbounds sum_sq += x[i]*x[i]
        end
        r = sqrt(sum_sq)
        return r
    end

    function seq_swap(N, x, y1)
        for i in 1:N
            @inbounds t = x[i]
            @inbounds x[i] = y1[i]
            @inbounds y1[i] = t
        end       
    end
  
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

end

@testset "Add-2D" begin
    function add!(i, j, A, B, C)
        @inbounds C[i, j] = A[i, j] + B[i, j]
    end

    M = 10
    N = 10
    A = JACC.Array(ones(Float32, M, N))
    B = JACC.Array(ones(Float32, M, N))
    C = JACC.Array(zeros(Float32, M, N))

    JACC.parallel_for((M, N), add!, A, B, C)

    C_expected = Float32(2.0) .* ones(Float32, M, N)
    @test Array(C)≈C_expected rtol=1e-5
end

@testset "Add-3D" begin
    function add!(i, j, k, A, B, C)
        @inbounds C[i, j, k] = A[i, j, k] + B[i, j, k]
    end

    L = 10
    M = 10
    N = 10
    A = JACC.Array(ones(Float32, L, M, N))
    B = JACC.Array(ones(Float32, L, M, N))
    C = JACC.Array(zeros(Float32, L, M, N))

    JACC.parallel_for((L, M, N), add!, A, B, C)

    C_expected = Float32(2.0) .* ones(Float32, L, M, N)
    @test Array(C)≈C_expected rtol=1e-5
end

#@testset "JACC.multi" begin

#    x = ones(1_000)
#    y = ones(1_000)
#    jx = JACC.multi.Array(x)
#    jy = JACC.multi.Array(y)
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
    #jresult = JACC.multi.parallel_reduce(1_000, multi_dot, jx[1], jy[1])
#    seq_axpy(1_000, alpha, x, y)
#    JACC.multi.parallel_for(1_000, multi_axpy, alpha, jx[1], jy[1])

    #result = Base.Array(jresult)

    #@test jresult[1]≈ref_result rtol=1e-8
#end
