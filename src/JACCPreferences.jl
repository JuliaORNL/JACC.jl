
module JACCPreferences

using Preferences

const supported_backends = ("threads", "cuda", "amdgpu", "oneapi")

# taken from https://github.com/JuliaPackaging/Preferences.jl
function set_backend(new_backend::String)
    new_backend_lc = lowercase(new_backend)
    if !(new_backend_lc in supported_backends)
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("backend"=>new_backend_lc)
    @info("New backend set; restart your Julia session for this change to take effect!")
end

const backend = @load_preference("backend", "threads")
const _backend_dispatchable = Val{Symbol(backend)}()

end # module JACCPreferences
