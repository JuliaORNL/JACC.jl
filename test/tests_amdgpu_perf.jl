using AMDGPU
import JACC
using Test

@testset "perf-AXPY-1D" begin

 function axpy_amdgpu_kernel(alpha,x,y)
   i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
   @inbounds x[i] = x[i] + alpha * y[i]
   return nothing
 end

 function axpy_amdgpu(SIZE,alpha,x,y)
   maxPossibleThreads = 512
   threads = min(SIZE, maxPossibleThreads)
   blocks = ceil(Int, SIZE/threads)
   @roc groupsize=threads gridsize=threads*blocks axpy_amdgpu_kernel(alpha,x,y)
 end

 SIZE = 100_000_000
 x = ones(SIZE)
 y = ones(SIZE)
 alpha = 2.0
 dx = ROCArray(x)
 dy = ROCArray(y)
 axpy_amdgpu(10,alpha,dx,dy)
 for i in [10,100,1_000,10_000,100_000,1_000_000,10_000_000,100_000_000]
  @time begin
   axpy_amdgpu(i,alpha,dx,dy)
  end
 end

 function axpy(i, alpha, x, y)
   @inbounds x[i] += alpha * y[i]
 end

 x = ones(SIZE)
 y = ones(SIZE)
 alpha = 2.0
 jx = ROCArray(x)
 jy = ROCArray(y)

 JACC.parallel_for(ROCBackend(), 10, axpy, alpha, jx, jy)
 for i in [10,100,1_000,1_0000,100_000,1_000_000,10_000_000,100_000_000]
   @time begin
     JACC.parallel_for(ROCBackend(), i, axpy, alpha, jx, jy)
   end
 end

end
