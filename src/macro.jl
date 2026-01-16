function _construct_call(api_func, valid_keys, args...)
    call = args[end]

    kwargs = map(args[1:(end - 1)]) do kwarg
        if kwarg isa Symbol
            :($kwarg = $kwarg)
        elseif Meta.isexpr(kwarg, :(=))
            kwarg
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg'"))
        end
    end

    Meta.isexpr(call, :call) ||
        throw(ArgumentError("last argument should be a function call"))
    func = call.args[1]
    args = call.args[2:end]

    for kwarg in kwargs
        key::Symbol, val = kwarg.args

        if !(key in valid_keys)
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end

    return quote
        $api_func(; f = $func, args = ($(args...),), $(kwargs...))
    end
end

macro parallel_for(args...)
    esc(_construct_call(JACC.parallel_for,
        [:dims, :threads, :blocks, :stream, :sync], args...))
end

macro parallel_reduce(args...)
    esc(_construct_call(JACC.parallel_reduce,
        [:dims, :type, :op, :init, :stream, :sync], args...))
end
