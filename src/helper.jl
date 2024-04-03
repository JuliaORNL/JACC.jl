
macro maybe_threaded(ex)
    if Threads.nthreads() == 1
        return esc(ex)
    else
        return esc(:(
            if threads
                Threads.@threads :static $ex
            else
                $ex
            end
        ))
    end
end
