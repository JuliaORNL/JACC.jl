
using Preferences
using Pkg

function _notify(backend::String)
    @info "Added $backend (be careful about committing Project.toml)"
end

function _check_install_backend(backend::String)
    if backend == "cuda"
        deps = Pkg.project().dependencies
        if !haskey(deps, "CUDA")
            Pkg.add("CUDA")
            _notify("CUDA")
        end

    elseif backend == "amdgpu"
        deps = Pkg.project().dependencies
        if !haskey(deps, "AMDGPU")
            Pkg.add("AMDGPU")
            _notify("AMDGPU")
        end

    elseif backend == "oneapi"
        deps = Pkg.project().dependencies
        if !haskey(deps, "oneAPI")
            Pkg.add("oneAPI")
            _notify("oneAPI")
        end
    end
end

const supported_backends = ("threads", "cuda", "amdgpu", "oneapi")

function set_backend(new_backend::String)
    new_backend_lc = lowercase(new_backend)
    if !(new_backend_lc in supported_backends)
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("backend"=>new_backend_lc)

    _check_install_backend(new_backend_lc)

    @info("New backend set; restart your Julia session for this change to take effect!")
end

const backend = @load_preference("backend", "threads")
const _backend_dispatchable = Val{Symbol(backend)}()

_check_install_backend() = _check_install_backend(backend)

function _init_backend()
    quote
        if JACC.backend == "cuda"
            import CUDA
            @info "CUDA backend loaded"

        elseif JACC.backend == "amdgpu"
            import AMDGPU
            @info "AMDGPU backend loaded"

        elseif JACC.backend == "oneapi"
            import oneAPI
            @info "oneAPI backend loaded"

        elseif JACC.backend == "threads"
            @info "Threads backend loaded"
        end
    end
end

macro init_backend()
    return esc(_init_backend())
end
