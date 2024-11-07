using CUDA
import JACC
using Test

@testset "TestBackend" begin
    @test JACC.JACCPreferences.backend == "cuda"
end

@testset "zeros_type" begin
    N = 10
    x = JACC.zeros(Float64, N)
    @test typeof(x) == CUDA.CuArray{Float64, 1, CUDA.DeviceMemory}
    @test eltype(x) == Float64
end

@testset "ones_type" begin
    N = 10
    x = JACC.ones(Float64, N)
    @test typeof(x) == CUDA.CuArray{Float64, 1, CUDA.DeviceMemory}
    @test eltype(x) == Float64
end


# @testset "LBM" begin

# 	function lbm_kernel(x, y, f, f1, f2, t, w, cx, cy, SIZE)
# 		u = 0.0
# 		v = 0.0
# 		p = 0.0
# 		x_stream = 0
# 		y_stream = 0

# 		if x > 1 && x < SIZE && y > 1 && y < SIZE
# 			for k in 1:9
# 				@inbounds x_stream = x - cx[k]
# 				@inbounds y_stream = y - cy[k]
# 				ind = (k - 1) * SIZE * SIZE + x * SIZE + y
# 				iind = (k - 1) * SIZE * SIZE + x_stream * SIZE + y_stream
# 				@inbounds f[trunc(Int, ind)] = f1[trunc(Int, iind)]
# 			end
# 			for k in 1:9
# 				ind = (k - 1) * SIZE * SIZE + x * SIZE + y
# 				@inbounds p = p[1, 1] + f[ind]
# 				@inbounds u = u[1, 1] + f[ind] * cx[k]
# 				@inbounds v = v[1, 1] + f[ind] * cy[k]
# 			end
# 			u = u / p
# 			v = v / p
# 			for k in 1:9
# 				@inbounds cu = cx[k] * u + cy[k] * v
# 				@inbounds feq = w[k] * p * (1.0 + 3.0 * cu + cu * cu - 1.5 * ((u * u) + (v * v)))
# 				ind = (k - 1) * SIZE * SIZE + x * SIZE + y
# 				@inbounds f2[trunc(Int, ind)] = f[trunc(Int, ind)] * (1.0 - 1.0 / t) + feq * 1 / t
# 			end
# 		end
# 	end

# 	function lbm_threads(f, f1, f2, t, w, cx, cy, SIZE)

# 		Threads.@sync Threads.@threads for x in 1:SIZE
# 			for y in 1:SIZE
# 				u = 0.0
# 				v = 0.0
# 				p = 0.0
# 				x_stream = 0
# 				y_stream = 0

# 				if x > 1 && x < SIZE && y > 1 && y < SIZE
# 					for k in 1:9
# 						@inbounds x_stream = x - cx[k]
# 						@inbounds y_stream = y - cy[k]
# 						ind = (k - 1) * SIZE * SIZE + x * SIZE + y
# 						iind = (k - 1) * SIZE * SIZE + x_stream * SIZE + y_stream
# 						@inbounds f[trunc(Int, ind)] = f1[trunc(Int, iind)]
# 					end
# 					for k in 1:9
# 						ind = (k - 1) * SIZE * SIZE + x * SIZE + y
# 						@inbounds p = p[1, 1] + f[ind]
# 						@inbounds u = u[1, 1] + f[ind] * cx[k]
# 						@inbounds v = v[1, 1] + f[ind] * cy[k]
# 					end
# 					u = u / p
# 					v = v / p
# 					for k in 1:9
# 						@inbounds cu = cx[k] * u + cy[k] * v
# 						@inbounds feq = w[k] * p * (1.0 + 3.0 * cu + cu * cu - 1.5 * ((u * u) + (v * v)))
# 						ind = (k - 1) * SIZE * SIZE + x * SIZE + y
# 						@inbounds f2[trunc(Int, ind)] = f[trunc(Int, ind)] * (1.0 - 1.0 / t) + feq * 1 / t
# 					end
# 				end
# 			end
# 		end
# 	end

# 	SIZE = 10
# 	f = JACC.ones(Float64, SIZE * SIZE * 9) .* 2.0
# 	f1 = JACC.ones(Float64, SIZE * SIZE * 9) .* 3.0
# 	f2 = JACC.ones(Float64, SIZE * SIZE * 9) .* 4.0
# 	cx = JACC.zeros(Float64, 9)
# 	cy = JACC.zeros(Float64, 9)
# 	cx[1] = 0
# 	cy[1] = 0
# 	cx[2] = 1
# 	cy[2] = 0
# 	cx[3] = -1
# 	cy[3] = 0
# 	cx[4] = 0
# 	cy[4] = 1
# 	cx[5] = 0
# 	cy[5] = -1
# 	cx[6] = 1
# 	cy[6] = 1
# 	cx[7] = -1
# 	cy[7] = 1
# 	cx[8] = -1
# 	cy[8] = -1
# 	cx[9] = 1
# 	cy[9] = -1
# 	w = JACC.ones(Float64, 9)
# 	t = 1.0

# 	df = JACC.Array(f)
# 	df1 = JACC.Array(f1)
# 	df2 = JACC.Array(f2)
# 	dcx = JACC.Array(cx)
# 	dcy = JACC.Array(cy)
# 	dw = JACC.Array(w)

# 	JACC.parallel_for((SIZE, SIZE), lbm_kernel, df, df1, df2, t, dw, dcx, dcy, SIZE)

# 	lbm_threads(f, f1, f2, t, w, cx, cy, SIZE)

# 	@test f2 ≈ df2 rtol = 1e-1

# end


# @testset VectorAddLoop begin
#     N = 1024
#     A = JACC.Array{Float32}(1, N)
#     B = JACC.Array{Float32}(1, N)

#     @jacc
#     for i in 1:N
#         C[i] = A[i] + B[i]
#     end
# end

# @testset VectorReduce begin
#     N = 1024
#     A = JACC.Array{Float32}(1, N)
#     B = JACC.Array{Float32}(1, N)

#     @jacc reduction(C)
#     for i in 1:N
#         C += A[i] * B[i]
#     end
# end

# @testset VectorAddLoopKernel begin
#     N = 1024
#     A = JACC.Array{Float32}(1, N)
#     B = JACC.Array{Float32}(1, N)

#     function kernel(i, A, B)

#     end

#     @jacc
#     for i in 1:N
#         C[i] = A[i] + B[i]
#     end
# end


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
