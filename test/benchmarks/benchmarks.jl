using JACC
using BenchmarkTools
using Statistics
using Base: Callable

function print_header(name::AbstractString, SIZE)
    printstyled("\n", name, "\n"; bold = true, underline = true)
    @show SIZE
end

function run_benchmark_elapsed(f::Callable, x...)
    durs = zeros(Float64, 10)
    for n in 1:10
        durs[n] = @elapsed f(x...)
    end
    println("First run: ", durs[1])
    println("Mean:      ", mean(durs))
    println("Mean (-1): ", mean(durs[2:10]))
end

function run_benchmark_bench(f::Callable, x...)
    b = @benchmark $(f)($x...)
    io = IOContext(stdout)
    show(io, MIME("text/plain"), b)
    println()
end

function run_benchmark_set(set, x...)
    conf = JACCBench.getconf()
    if conf.warmup
        for bf in set
            bf(x...)
        end
    end
    for bf in set
        printstyled("\nBenchmarking ", bf, "\n"; bold = true)
        if conf.method === Method.time
            run_benchmark_elapsed(bf, x...)
        elseif conf.method === Method.bench
            run_benchmark_bench(bf, x...)
        end
    end
end

function axpy_jacc(SIZE::JACC.Dims, alpha, x, y)
    JACC.parallel_for(SIZE, axpy, alpha, x, y)
end

push!(JACCBench.axpy_comps, axpy_jacc)

function axpy_benchmark(name::AbstractString, SIZE)
    print_header(name, SIZE)
    x = round.(rand(Float32, SIZE) * 100)
    y = round.(rand(Float32, SIZE) * 100)
    alpha = 2.5
    dx = JACC.array(x)
    dy = JACC.array(y)
    run_benchmark_set(JACCBench.axpy_comps, SIZE, alpha, dx, dy)
end

@testset "AXPY_1D" begin
    SIZE = JACCBench.getconf().size_1d
    axpy_benchmark("AXPY 1D", SIZE)
end

@testset "AXPY_2D" begin
    SIZE = JACCBench.getconf().size_2d
    axpy_benchmark("AXPY 2D", SIZE)
end

function dot_jacc(SIZE::JACC.Dims, x, y)
    JACC.parallel_reduce(SIZE, dot, x, y)
end

push!(JACCBench.dot_comps, dot_jacc)

function dot_benchmark(name::AbstractString, SIZE)
    print_header(name, SIZE)
    x = JACC.ones(SIZE)
    y = JACC.ones(SIZE)
    run_benchmark_set(JACCBench.dot_comps, SIZE, x, y)
end

@testset "DOT_1D" begin
    SIZE = JACCBench.getconf().size_1d
    dot_benchmark("DOT 1D", SIZE)
end

@testset "DOT_2D" begin
    SIZE = JACCBench.getconf().size_2d
    dot_benchmark("DOT 2D", SIZE)
end
