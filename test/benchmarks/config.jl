using ArgParse

module Method
@enum T begin
    time
    bench
end
T(s::String) = eval(Symbol(s))
end

struct Config
    warmup::Bool
    method::Method.T
    size_1d::Int64
    size_2d::NTuple{2, Int64}
    filter::Vector{String}
end

function Config(args::Vector{String})
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! s begin
        "--warmup-first", "-w"
        help = "Run each benchmark once without timing"
        arg_type = Bool
        default = true
        "--method", "-m"
        help = "Timing method: bench|time"
        arg_type = String
        default = "bench"
        "--size-1d"
        help = "workload size for 1d cases"
        arg_type = Int64
        default = 1_000_000
        "--size-2d"
        help = "workload size for 2d cases"
        nargs = 2
        arg_type = Int64
        default = [1000, 1000]
        "filter"
        help = "arguments to be left for ReTest"
        nargs = '*'
        arg_type = String
        # default = String[]

    end
    pa = ArgParse.parse_args(args, s)
    return Config(pa["warmup-first"], Method.T(pa["method"]), pa["size-1d"],
        tuple(pa["size-2d"]...), pa["filter"])
end

const _conf = Ref{Union{Config, Nothing}}(nothing)

function getconf()
    global _conf
    if _conf[] === nothing
        _conf[] = Config(ARGS)
    end
    return _conf[]
end
