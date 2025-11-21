
using Preferences
using Pkg

function _notify_add(backend::AbstractString)
    @info "Added $backend (be careful about committing Project.toml)"
end

const proj = Pkg.Types.read_project(Pkg.Types.find_project_file())

function _check_install_backend(backend, backend_lc)
    # Check original placement
    place_dict = Preferences.Backend._PLACE[]
    if !haskey(place_dict, backend_lc)
        if haskey(proj.deps, backend)
            place_dict[backend_lc] = "deps"
        elseif haskey(proj.weakdeps, backend)
            place_dict[backend_lc] = "weakdeps"
        else
            place_dict[backend_lc] = "none"
        end
    end

    if !haskey(proj.deps, backend)
        Pkg.add(backend)
        _notify_add(backend)
    end
end

function _check_install_backend(backend::AbstractString)
    match = filter(b -> backend == lowercase(b), ["CUDA", "AMDGPU", "oneAPI"])
    if !isempty(match)
        _check_install_backend(match[], backend)
    end
end

_check_install_backend() = _check_install_backend(Preferences.Backend.default)

function _notify_rm(backend::AbstractString)
    @info "Removed $backend (be careful about committing Project.toml)"
end

function _check_uninstall_backend(backend, backend_lc)
    if haskey(proj.deps, backend)
        place_dict = Preferences.Backend._PLACE[]
        if haskey(place_dict, backend_lc)
            if place_dict[backend_lc] != "deps"
                Pkg.rm(backend)
                _notify_rm(backend)
                if place_dict[backend_lc] == "weakdeps"
                    Pkg.add(backend; target = :weakdeps)
                end
            end
            delete!(place_dict, backend_lc)
        end
    end
end

function _uninstall_backend(backend::AbstractString)
    match = filter(b -> backend == lowercase(b), ["CUDA", "AMDGPU", "oneAPI"])
    if !isempty(match)
        _check_uninstall_backend(match[], backend)
    end
end

_uninstall_backends() = _uninstall_backend.(Preferences.Backend._LIST[])

const supported_backends = ("threads", "cuda", "amdgpu", "oneapi")

baremodule Backend
const threads = :threads
const cuda = :cuda
const amdgpu = :amdgpu
const oneapi = :oneapi
end

module Preferences
module Backend

import Base: deepcopy, Dict
import Preferences: @load_preference
const default = @load_preference("default_backend", "threads")
const _DEFAULT = Ref(String(default))
const list = @load_preference("backends", ["threads"])
const _LIST = Ref(deepcopy(list))
const _PLACE = Ref(@load_preference("placement", Dict{String, String}()))

function backend_import(backend::String)
    backend == "cuda" && return quote
        import CUDA
        @info "CUDA backend loaded"
    end
    backend == "amdgpu" && return quote
        import AMDGPU
        @info "AMDGPU backend loaded"
    end
    backend == "oneapi" && return quote
        import oneAPI
        @info "oneAPI backend loaded"
    end
    backend == "threads" && return quote
        @info "Threads backend loaded with $(Threads.nthreads()) threads"
    end
end

const imports = Expr(:block, backend_import.(list)...)

end
end

const backend = Preferences.Backend.default
const _backend_dispatchable = Val{Symbol(backend)}()

function unset_backend()
    _uninstall_backends()
    Preferences.Backend._DEFAULT[] = ""
    empty!(Preferences.Backend._LIST[])
    empty!(Preferences.Backend._PLACE[])
    @delete_preferences!("default_backend")
    @delete_preferences!("backends")
    @delete_preferences!("placement")
    @info """
        Backend preferences deleted
        Restart your Julia session for this change to take effect!
        """
end

function set_default_backend(new_backend::AbstractString)
    new_backend_lc = lowercase(new_backend)
    if new_backend_lc == Preferences.Backend._DEFAULT[]
        return
    end

    if new_backend_lc ∉ supported_backends
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    if new_backend_lc ∉ Preferences.Backend._LIST[]
        add_backend(new_backend_lc)
    end
    Preferences.Backend._DEFAULT[] = new_backend_lc
    @set_preferences!("default_backend"=>Preferences.Backend._DEFAULT[])

    @info """
        New default backend set
        Restart your Julia session for this change to take effect!
        """
end

function set_default_backend(new_backend::Symbol)
    set_default_backend(String(new_backend))
end

set_backend(b::Union{Symbol, AbstractString}) = set_default_backend(b)

function add_backend(new_backend::AbstractString)
    new_backend_lc = lowercase(new_backend)
    backend_list = Preferences.Backend._LIST[]
    if new_backend_lc in backend_list
        return
    end

    if new_backend_lc ∉ supported_backends
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    Preferences.Backend._LIST[] = vcat(backend_list, [new_backend_lc])
    @set_preferences!("backends"=>Preferences.Backend._LIST[])

    _check_install_backend(new_backend_lc)
    @set_preferences!("placement"=>Preferences.Backend._PLACE[])

    @info """
        New backend added
        Restart your Julia session for this change to take effect!
        """
end

function add_backend(new_backend::Symbol)
    add_backend(String(new_backend))
end

function remove_backend(backend::AbstractString)
    backend_lc = lowercase(backend)
    backend_list = Preferences.Backend._LIST[]
    if backend_lc ∉ backend_list
        return
    end

    Preferences.Backend._LIST[] = filter(b -> b != backend_lc, backend_list)
    @set_preferences!("backends"=>Preferences.Backend._LIST[])
    if backend_lc == Preferences.Backend._DEFAULT[]
        Preferences.Backend._DEFAULT[] = ""
        @delete_preferences!("default_backend")
    end

    _uninstall_backend(backend_lc)
    @set_preferences!("placement"=>Preferences.Backend._PLACE[])

    @info """
        \"$(backend_lc)\" backend removed
        Restart your Julia session for this change to take effect!
        """
end

function remove_backend(backend::Symbol)
    remove_backend(String(backend))
end

macro init_backends()
    return JACC.Preferences.Backend.imports
end

function _init_backend()
    JACC.Preferences.Backend.backend_import(JACC.Preferences.Backend.default)
end

macro init_backend()
    return esc(_init_backend())
end
