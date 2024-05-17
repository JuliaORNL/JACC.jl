module BLAS

using JACC

function _axpy(i, alpha, x, y)
  @inbounds x[i] += alpha * y[i]
end

function _dot(i, x, y)
  return @inbounds x[i] * y[i]
end

function axpy(n::Integer, alpha, x, y)
  JACC.parallel_for(n, _axpy, alpha, x, y)
end

function dot(n::Integer, x, y)
  JACC.parallel_reduce(n, _dot, x, y)
end

end # module BLAS
