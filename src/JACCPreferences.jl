
module JACCPreferences

using Preferences

# taken from https://github.com/JuliaPackaging/Preferences.jl
function set_backend(new_backend::String)

    new_backend_lc = lowercase(new_backend)
    if !(new_backend_lc in ("threads", "cuda", "amdgpu"))
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("backend" => new_backend_lc)
    @info("New backend set; restart your Julia session for this change to take effect!")
end

const backend = @load_preference("backend", "threads")


end # module JACCPreferences