using JACC
using BenchmarkTools
using Statistics

function axpy_jacc(SIZE::Integer, alpha, x, y)
    JACC.parallel_for(SIZE, axpy, alpha, x, y)
end

function axpy_jacc((M, N)::NTuple{2, Integer}, alpha, x, y)
    JACC.parallel_for((M, N), axpy, alpha, x, y)
end
push!(JACCBench.axpy_comps, axpy_jacc)

function axpy_benchmark(name::AbstractString, SIZE)
    printstyled("\n", name, "\n", bold=true, underline=true)
    @show SIZE
    x = round.(rand(Float32, SIZE) * 100)
    y = round.(rand(Float32, SIZE) * 100)
    alpha = 2.5
    dx = JACC.array(x)
    dy = JACC.array(y)
    # io = IOContext(stdout, :histmin=>0.001, :histmax=>8, :logbins=>true)
    io = IOContext(stdout)
    for comp in JACCBench.axpy_comps
        printstyled("\nBenchmarking ", comp, "\n", bold=true)
        durs = zeros(Float64, 10)
        for n in 1:10
            durs[n] = @elapsed comp(SIZE, alpha, dx, dy)
        end
        println("First run: ", durs[1])
        println("Mean:      ", mean(durs))
        println("Mean (-1): ", mean(durs[2:10]))
        b = @benchmark $(comp)($SIZE, $alpha, $dx, $dy)
        show(io, MIME("text/plain"), b)
        println()
    end
end

@testset "AXPY 1D" begin
    SIZE = JACCBench.getconf().size_1d
    axpy_benchmark("AXPY 1D", SIZE)
end

@testset "AXPY 2D" begin
    SIZE = JACCBench.getconf().size_2d
    axpy_benchmark("AXPY 2D", SIZE)
end

function dot_jacc(SIZE::Integer, x, y)
    JACC.parallel_reduce(SIZE, dot, x, y)
end

function dot_jacc((M, N)::NTuple{2, Integer}, x, y)
    JACC.parallel_reduce((M, N), dot, x, y)
end

push!(JACCBench.dot_comps, dot_jacc)

function dot_benchmark(name::AbstractString, SIZE)
    printstyled("\n", name, "\n", bold=true, underline=true)
    @show SIZE
    x = JACC.ones(SIZE)
    y = JACC.ones(SIZE)
    # io = IOContext(stdout, :histmin=>0.001, :histmax=>8, :logbins=>true)
    io = IOContext(stdout)
    for comp in JACCBench.dot_comps
        printstyled("\nBenchmarking ", comp, "\n", bold=true)
        durs = zeros(Float64, 10)
        for n in 1:10
            durs[n] = @elapsed comp(SIZE, x, y)
        end
        println("First run: ", durs[1])
        println("Mean:      ", mean(durs))
        println("Mean (-1): ", mean(durs[2:10]))
        b = @benchmark $(comp)($SIZE, $x, $y)
        show(io, MIME("text/plain"), b)
        println()
    end
end

@testset "DOT 1D" begin
    SIZE = JACCBench.getconf().size_1d
    dot_benchmark("DOT 1D", SIZE)
end

@testset "DOT 2D" begin
    SIZE = JACCBench.getconf().size_2d
    dot_benchmark("DOT 2D", SIZE)
end
