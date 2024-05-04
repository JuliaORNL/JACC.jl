# JACC.jl

[![CI-CPU](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-cpu.yaml/badge.svg)](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-cpu.yaml)
[![CI-GPU-NVIDIA](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-gpu-NVIDIA.yaml/badge.svg)](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-gpu-NVIDIA.yaml)
[![CI-GPU-AMD](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-gpu-AMD.yaml/badge.svg)](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-gpu-AMD.yaml)


CPU/GPU performance portable layer for Julia

JACC.jl follows a function as a argument approach in combination with the power of Julia's ecosystem for multiple dispatch, GPU access via [JuliaGPU back ends](https://juliagpu.org/), and [package extensions](https://julialang.org/blog/2023/04/julia-1.9-highlights/#package_extensions) since Julia v1.9 . Similar to portable layers like Kokkos, users would pass a size and a function including its arguments to a `parallel_for` or `parallel_reduce` function.
The overall goal is to write a single source code that can be executed on multiple vendor CPU and GPU parallel programming environments. JACC meant to simplify CPU/GPU kernel programming using a simple application programming interface (API).

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
For an app integration example see the [GrayScott.jl JACC branch](https://github.com/JuliaORNL/GrayScott.jl/tree/GrayScott-JACC) and the [Simulation.jl](https://github.com/JuliaORNL/GrayScott.jl/blob/GrayScott-JACC/src/simulation/Simulation.jl) for writing kernels with JACC.jl and selecting specific vendor back ends in Julia.


Funded by the US Department of Energy Advanced Scientific Computing Research (ASCR) projects:

- PESO and S4PST as part of the Next Generation of Scientific Software Technologies (NGSST)
- [Bluestone X-Stack](https://csmd.ornl.gov/Bluestone)

Past sponsors:
- The Exascale Computing Project (ECP) [PROTEAS-TUNE](https://www.ornl.gov/project/proteas-tune) 
