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
 jx = JACC.Array(x)
 jy = JACC.Array(y)
 
 JACC.parallel_for(10, axpy, alpha, jx, jy)
 for i in [10,100,1_000,1_0000,100_000,1_000_000,10_000_000,100_000_000]
   @time begin
     JACC.parallel_for(i, axpy, alpha, jx, jy)
   end
 end

end

@testset "perf-DOT-1D" begin

 function dot_amdgpu_kernel(SIZE, ret, x, y)
   shared_mem = @ROCDynamicLocalArray(Float64, 512)
   i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
   ti = workitemIdx().x
   tmp::Float64 = 0.0
   shared_mem[ti] = 0.0

   if i <= SIZE
     tmp = @inbounds x[i] * y[i]
     shared_mem[ti] = tmp
   end
   AMDGPU.sync_workgroup()
   if (ti <= 256)
     shared_mem[ti] += shared_mem[ti+256]
   end
   AMDGPU.sync_workgroup()
   if (ti <= 128)
     shared_mem[ti] += shared_mem[ti+128]
   end
   AMDGPU.sync_workgroup()
   if (ti <= 64)
     shared_mem[ti] += shared_mem[ti+64]
   end
   AMDGPU.sync_workgroup()
   if (ti <= 32)
     shared_mem[ti] += shared_mem[ti+32]
   end
   AMDGPU.sync_workgroup()
   if (ti <= 16)
     shared_mem[ti] += shared_mem[ti+16]
   end
   AMDGPU.sync_workgroup()
   if (ti <= 8)
     shared_mem[ti] += shared_mem[ti+8]
   end
   AMDGPU.sync_workgroup()
   if (ti <= 4)
    shared_mem[ti] += shared_mem[ti+4]
   end
   AMDGPU.sync_workgroup()
   if (ti <= 2)
    shared_mem[ti] += shared_mem[ti+2]
   end
   AMDGPU.sync_workgroup()
   if (ti == 1)
     shared_mem[ti] += shared_mem[ti+1]
     ret[workgroupIdx().x] = shared_mem[ti]
   end
   AMDGPU.sync_workgroup()
   return nothing
 end

 function reduce_kernel(SIZE, red, ret)
   shared_mem = @ROCDynamicLocalArray(Float64, 512)
   i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
   ii = i
   tmp::Float64 = 0.0
   if SIZE > 512
     while ii <= SIZE
       tmp += @inbounds red[ii]
       ii += 512
     end
   else
     tmp = @inbounds red[i]
   end
   shared_mem[i] = tmp
   AMDGPU.sync_workgroup()
   if (i <= 256)
     shared_mem[i] += shared_mem[i+256]
   end
   AMDGPU.sync_workgroup()
   if (i <= 128)
     shared_mem[i] += shared_mem[i+128]
   end
   AMDGPU.sync_workgroup()
   if (i <= 64)
     shared_mem[i] += shared_mem[i+64]
   end
   AMDGPU.sync_workgroup()
   if (i <= 32)
     shared_mem[i] += shared_mem[i+32]
   end
   AMDGPU.sync_workgroup()
   if (i <= 16)
     shared_mem[i] += shared_mem[i+16]
   end
   AMDGPU.sync_workgroup()
   if (i <= 8)
     shared_mem[i] += shared_mem[i+8]
   end
   AMDGPU.sync_workgroup()
   if (i <= 4)
     shared_mem[i] += shared_mem[i+4]
   end
   AMDGPU.sync_workgroup()
   if (i <= 2)
     shared_mem[i] += shared_mem[i+2]
   end
   AMDGPU.sync_workgroup()
   if (i == 1)
     shared_mem[i] += shared_mem[i+1]
     ret[1] = shared_mem[1]
   end
   return nothing
 end

 function dot_amdgpu(SIZE, x, y)
   maxPossibleThreads = 512
   threads = min(SIZE, maxPossibleThreads)
   blocks = ceil(Int, SIZE/threads)
   ret = AMDGPU.zeros(Float64,blocks)
   rret = AMDGPU.zeros(Float64,1)
   @roc groupsize=threads gridsize=threads*blocks dot_amdgpu_kernel(SIZE, ret, x, y)
   @roc groupsize=threads gridsize=threads reduce_kernel(blocks, ret, rret)
   return rret
 end

 SIZE = 100_000_000
 x = ones(SIZE)
 y = ones(SIZE)
 dx = ROCArray(x)
 dy = ROCArray(y)
 res = dot_amdgpu(10,dx,dy)
 for i in [10,100,1_000,10_000,100_000,1_000_000,10_000_000,100_000_000]
   @time begin
     res = dot_amdgpu(i,dx,dy)
   end
  end

 funtion dot(i, x, y)
   return @inbounds x[i] * y[i]
 end

 x = ones(SIZE)
 y = ones(SIZE)
 jx = JACC.Array(x)
 jy = JACC.Array(y)
 res = JACC.parallel_reduce(10, dot, jx, jy)
 for i in [10,100,1_000,10_000,100_000,1_000_000,10_000_000,100_000_000]
   @time begin
     res = JACC.parallel_reduce(i, dot, jx, jy)
   end
 end

end

@testset "perf-AXPY-2D" begin

 function axpy_amdgpu_kernel(alpha,x,y)
   i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
   j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
   @inbounds x[i,j] = x[i,j] + alpha * y[i,j]
   return nothing
 end

 function axpy_amdgpu((M,N),alpha,x,y)
   maxPossibleThreads = 16 
   Mthreads = min(M, maxPossibleThreads)
   Mblocks = ceil(Int, M/Mthreads)
   Nthreads = min(N, maxPossibleThreads)
   Nblocks = ceil(Int, N/Nthreads)
   @roc groupsize=(Mthreads, Nthreads) gridsize=(Mthreads*Mblocks, Nthreads*Nblocks) axpy_amdgpu_kernel(alpha,x,y)
 end

 SIZE = 10_000
 x = ones(SIZE, SIZE)
 y = ones(SIZE, SIZE)
 alpha = 2.0
 dx = ROCArray(x)
 dy = ROCArray(y)
 axpy_amdgpu((1_000,1_000),alpha,dx,dy)
 for i in [1_000,2_000,3_000,4_000,5_000,6_000,7_000,8_000,9_000,10_000]
   @time begin
     axpy_amdgpu((i,i),alpha,dx,dy)
   end
 end

 function axpy(i, j, alpha, x, y)
   @inbounds x[i,j] = x[i,j] + alpha * y[i,j]
 end

 x = ones(SIZE, SIZE)
 y = ones(SIZE, SIZE)
 jx = JACC.Array(x)
 jy = JACC.Array(y)
 JACC.parallel_for((1_000,1_000), axpy, alpha, jx, jy)
 for i in [1_000,2_000,3_000,4_000,5_000,6_000,7_000,8_000,9_000,10_000]
   @time begin
     JACC.parallel_for((i,i), axpy, alpha, jx, jy)
   end
 end

end


@testset "perf-DOT-2D" begin

 function dot_amdgpu_kernel((M, N), ret, x, y)
   shared_mem = @ROCDynamicLocalArray(Float64, 16*16)
   i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
   j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
   ti = workitemIdx().x
   tj = workitemIdx().y
   bi = workgroupIdx().x
   bj = workgroupIdx().y

   tmp::Float64 = 0.0
   shared_mem[((ti-1)*16)+tj] = tmp

   if (i <= M && j <= N)
     tmp = @inbounds x[i,j] * y[i,j]
     shared_mem[(ti-1)*16+tj] = tmp
   end
   AMDGPU.sync_workgroup()
   if (ti <= 8 && tj <= 8 && ti+8 <= M && tj+8 <= N)
     shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+7)*16)+(tj+8)]
     shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+8)]
     shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+7)*16)+tj]
   end
   AMDGPU.sync_workgroup()
   if (ti <= 4 && tj <= 4 && ti+4 <= M && tj+4 <= N)
     shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+3)*16)+(tj+4)]
     shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+4)]
     shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+3)*16)+tj]
   end
   AMDGPU.sync_workgroup()
   if (ti <= 2 && tj <= 2 && ti+2 <= M && tj+2 <= N)
     shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+1)*16)+(tj+2)]
     shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+2)]
     shared_mem[((ti-1)*16)+tj] += shared_mem[((ti+1)*16)+tj]
   end
   AMDGPU.sync_workgroup()
   if (ti == 1 && tj == 1 && ti+1 <= M && tj+1 <= N)
     shared_mem[((ti-1)*16)+tj] += shared_mem[ti*16+(tj+1)]
     shared_mem[((ti-1)*16)+tj] += shared_mem[((ti-1)*16)+(tj+1)]
     shared_mem[((ti-1)*16)+tj] += shared_mem[ti*16+tj]
     ret[bi,bj] = shared_mem[((ti-1)*16)+tj]
   end
   return nothing
 end

 function reduce_kernel((M, N), red, ret)
   shared_mem = @ROCDynamicLocalArray(Float64, 16*16)
   i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
   j = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
   ii = i
   jj = j

   tmp::Float64 = 0.0
   shared_mem[(i-1)*16+j] = tmp
  
   if M > 16 && N > 16
     while ii <= M
       jj = workitemIdx().y
       while jj <= N
         tmp = tmp + @inbounds red[ii,jj]
         jj += 16
       end
       ii += 16
     end
   elseif M > 16
     while ii <= N
       tmp = tmp + @inbounds red[ii,jj]
       ii += 16
     end
   elseif N > 16
     while jj <= N
       tmp = tmp + @inbounds red[ii,jj]
       jj += 16
     end
   elseif M <= 16 && N <= 16
     if i <= M && j <= N
       tmp = tmp + @inbounds red[i,j]
     end
   end
   shared_mem[(i-1)*16+j] = tmp
   AMDGPU.sync_workgroup()
   if (i <= 8 && j <= 8)
     if (i+8 <= M && j+8 <= N)
       shared_mem[((i-1)*16)+j] += shared_mem[((i+7)*16)+(j+8)]
     end
     if (i <= M && j+8 <= N)
       shared_mem[((i-1)*16)+j] += shared_mem[((i-1)*16)+(j+8)]
     end
     if (i+8 <= M && j <= N)
       shared_mem[((i-1)*16)+j] += shared_mem[((i+7)*16)+j]
     end
   end
   AMDGPU.sync_workgroup()
   if (i <= 4 && j <= 4)
     if (i+4 <= M && j+4 <= N)
       shared_mem[((i-1)*16)+j] += shared_mem[((i+3)*16)+(j+4)]
     end
     if (i <= M && j+4 <= N)
       shared_mem[((i-1)*16)+j] += shared_mem[((i-1)*16)+(j+4)]
     end
     if (i+4 <= M && j <= N)
       shared_mem[((i-1)*16)+j] += shared_mem[((i+3)*16)+j]
     end
   end
   AMDGPU.sync_workgroup()
   if (i <= 2 && j <= 2)
     if (i+2 <= M && j+2 <= N)
       shared_mem[((i-1)*16)+j] += shared_mem[((i+1)*16)+(j+2)]
     end
     if (i <= M && j+2 <= N)
       shared_mem[((i-1)*16)+j] += shared_mem[((i-1)*16)+(j+2)]
     end
     if (i+2 <= M && j <= N)
       shared_mem[((i-1)*16)+j] += shared_mem[((i+1)*16)+j]
     end
   end
   AMDGPU.sync_workgroup()
   if (i == 1 && j == 1)
     if (i+1 <= M && j+1 <= N)
       shared_mem[((i-1)*16)+j] += shared_mem[i*16+(j+1)]
     end
     if (i <= M && j+1 <= N)
       shared_mem[((i-1)*16)+j] += shared_mem[((i-1)*16)+(j+1)]
     end
     if (i+1 <= M && j <= N)  
       shared_mem[((i-1)*16)+j] += shared_mem[i*16+j]
     end
     ret[1] = shared_mem[((i-1)*16)+j] 
   end
   return nothing
 end

 function dot_amdgpu((M,N), x, y)
   maxPossibleThreads = 16 
   Mthreads = min(M, maxPossibleThreads)
   Nthreads = min(N, maxPossibleThreads)
   Mblocks = ceil(Int, M/Mthreads)
   Nblocks = ceil(Int, N/Nthreads)
   ret = AMDGPU.zeros(Float64,(Mblocks, Nblocks))
   rret = AMDGPU.zeros(Float64,1)
   @roc groupsize=(Mthreads, Nthreads) gridsize=(Mblocks*Mthreads, Nblocks*Nthreads) localmem=16*16*sizeof(Float64) dot_amdgpu_kernel((M, N), ret, x, y)
   @roc groupsize=(Mthreads, Nthreads) gridsize=(Mthreads, Nthreads) localmem=16*16*sizeof(Float64) reduce_kernel((Mblocks, Nblocks), ret, rret)
   return rret
 end

 SIZE = 10_000
 x = ones(SIZE,SIZE)
 y = ones(SIZE,SIZE)
 dx = ROCArray(x)
 dy = ROCArray(y)
 res = dot_amdgpu((1_000,1_000), dx, dy)
 for i in [1_000,2_000,3_000,4_000,5_000,6_000,7_000,8_000,9_000,1_0000]
   @time begin
     res = dot_amdgpu((i,i),dx,dy)
   end
 end

 function dot(i, j, x, y)
   return @inbounds x[i,j] * y[i,j]
 end

 x = ones(SIZE,SIZE)
 y = ones(SIZE,SIZE)
 jx = JACC.Array(x)
 jy = JACC.Array(y)
 res = JACC.parallel_reduce((1_000,1_000), dot, jx, jy)
 for i in [1_000,2_000,3_000,4_000,5_000,6_000,7_000,8_000,9_000,1_0000]
   @time begin
     res = JACC.parallel_reduce((i,i), dot, jx, jy)
   end
 end

end
