# JACC.jl
CPU/GPU performance portable layer for Julia

JACC.jl follows a function as a argument approach in combination with the power of Julia's ecosystem for multiple dispatch, GPU back end access, and weak dependencies since Julia v1.9 . Similar to portable layers like Kokkos, users would pass a size and a function including its arguments to a `parallel_for` or `parallel_reduce` function.
The overall goal is to write a single source code that can be executed in multiple CPU and GPU parallel programming environments. Following the principle for OpenACC, it's meant to simplify programming for heterogeneous CPU and GPU systems.

JuliaCon 2023 presentation [video](https://live.juliacon.org/talk/AY8EUX).

1. Set a back end: "cuda", "amdgpu", or "threads" (default) with `JACC.JACCPreferences` generating a `LocalPreferences.toml` file

```
julia> import JACC.JACCPreferences
julia> JACCPreferences.set_backend("cuda")
```

2. Run a kernel example (see tests directory)

```
import JACC

function axpy(i, alpha, x, y)
  if i <= length(x)
    @inbounds x[i] += alpha * y[i]
  end
end

N = 10
# Generate random vectors x and y of length N for the interval [0, 100]
x = round.(rand(Float32, N) * 100)
y = round.(rand(Float32, N) * 100)
alpha = 2.5

x_d = JACC.Array(x)
y_d = JACC.Array(y)
JACC.parallel_for(N, axpy, alpha, x_d, y_d)
```

We currently have a limited number of configurations. 
We hope to study and incoorporate more relevant cases and dimensions shapes as needed.
