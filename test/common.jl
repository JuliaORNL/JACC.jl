module JACCTestCommon

export axpy, dot, seq_axpy, seq_dot

function axpy(i, alpha, x, y)
    @inbounds x[i] += alpha * y[i]
end

function axpy(i, j, alpha, x, y)
    @inbounds x[i, j] = x[i, j] + alpha * y[i, j]
end

function dot(i, x, y)
    @inbounds return x[i] * y[i]
end

function dot(i, j, x, y)
    @inbounds return x[i, j] * y[i, j]
end

function seq_axpy(N, alpha, x, y)
    for i in 1:N
        @inbounds x[i] += alpha * y[i]
    end
end

function seq_axpy(M, N, alpha, x, y)
    for j in 1:N
        for i in 1:M
            @inbounds x[i, j] += alpha * y[i, j]
        end
    end
end

function seq_dot(N, x, y)
    r = 0.0
    for i in 1:N
        @inbounds r += x[i] * y[i]
    end
    return r
end

function seq_dot(M, N, x, y)
    r = 0.0
    for j in 1:N
        for i in 1:M
            @inbounds r += x[i, j] * y[i, j]
        end
    end
    return r
end

end
