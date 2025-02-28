function _maybe_threaded(ex)
    quote
        if Threads.nthreads() == 1
            $ex
        else
            Threads.@threads :static $ex
        end
    end
end

macro maybe_threaded(ex)
    esc(_maybe_threaded(ex))
end

function _init_backend()
    quote
        const backend = JACC.JACCPreferences.backend
        if backend == "cuda"
            using CUDA
            @info "CUDA backend loaded"

        elseif backend == "amdgpu"
            using AMDGPU
            @info "AMDGPU backend loaded"

        elseif backend == "oneapi"
            using oneAPI
            @info "oneAPI backend loaded"

        elseif backend == "threads"
            @info "Threads backend loaded"
        end
    end
end

macro init_backend()
    return esc(_init_backend())
end
