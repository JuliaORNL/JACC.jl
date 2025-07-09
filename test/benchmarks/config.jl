using ArgParse

struct Config
    size_1d::Int64
    size_2d::NTuple{2, Int64}
    filter::Vector{String}
end

function Config(args::Vector{String})
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! s begin
        "--size_1d"
            help = "workload size for 1d cases"
            arg_type = Int64
            default = 1_000_000
        "--size_2d"
            help = "workload size for 2d cases"
            nargs = 2
            arg_type = Int64
            default = [1_000, 1_000]
        "filter"
            # help = "arguments to be left for ReTest"
            nargs = '*'
            arg_type = String
            # default = String[]

    end
    pa = ArgParse.parse_args(args, s)
    return Config(pa["size_1d"], tuple(pa["size_2d"]...), pa["filter"])
end

_conf::Union{Config, Nothing} = nothing

function getconf()
    global _conf
    if _conf === nothing
        _conf = Config(ARGS)
    end
    return _conf
end
