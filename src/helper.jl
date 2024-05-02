
macro maybe_threaded(ex)
	if Threads.nthreads() == 1
		return esc(ex)
	else
		return esc(:(Threads.@threads :static $ex))
	end
end


struct JACCArrayType{T}
end
