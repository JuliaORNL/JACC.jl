
function _init_backend()
    quote
        using Pkg
        deps = Pkg.project().dependencies
        if JACC.backend == "cuda"
            if !haskey(deps, "CUDA")
                Pkg.add("CUDA")
                @info "Added CUDA (be careful about committing Project.toml)"
            end
            using CUDA
            @info "CUDA backend loaded"

        elseif JACC.backend == "amdgpu"
            if !haskey(deps, "AMDGPU")
                Pkg.add("AMDGPU")
                @info "Added AMDGPU (be careful about committing Project.toml)"
            end
            using AMDGPU
            @info "AMDGPU backend loaded"

        elseif JACC.backend == "oneapi"
            if !haskey(deps, "oneAPI")
                Pkg.add("oneAPI")
                @info "Added oneAPI (be careful about committing Project.toml)"
            end
            using oneAPI
            @info "oneAPI backend loaded"

        elseif JACC.backend == "threads"
            @info "Threads backend loaded"
        end
    end
end

macro init_backend()
    return esc(_init_backend())
end
