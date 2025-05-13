## API components

JACC API consist of three main components:

- Backend selection: e.g. `JACC.set_backend("CUDA")` and `JACC.@init_backend`
- Memory models: e.g. `JACC.array{T}()`, `JACC.zeros`, `JACC.ones`, `JACC.shared`, `JACC.Multi`
- Kernel launching: e.g. `JACC.parallel_for`, `JACC.parallel_reduce`

## Backend selection

- `JACC.set_backend`: allows selecting the runtime back end on **CPU**: `Threads` (default) and **GPU**: `CUDA`, `AMDGPU`, `oneAPI`. Uses Preferences.jl and stores the selected back end in a [LocalPreferences.jl](https://github.com/JuliaPackaging/Preferences.jl) file if JACC.jl is a project dependency. Use `JACC.set_backend` prior to running any code targeting a particular back end.

Example:
```julia
using JACC
JACC.set_backend("CUDA")
```

!!! danger 

    `set_backend` will polute your project's `Project.toml` adding the selected backend package. Beware of committing this change (e.g. during development). 

!!! warning

    This step might take a while the very first time downloading all back end dependencies. 

!!! tip "Tip for CUDA" 

    CUDA.jl uses its own prebuilt CUDA stack by default, please refer to CUDA.jl docs if wanting to [use a local CUDA installation](https://cuda.juliagpu.org/stable/installation/overview/#Using-a-local-CUDA) to set up `LocalPreferences.toml`. 

!!! tip "Tip for AMDGPU"

    AMDGPU.jl relies on standard `rocm` installation under `/opt/rocm`, for non-standard locations set the environment variable `ROCM_PATH`, see [docs](https://amdgpu.juliagpu.org/stable/install_tips).


- `JACC.@init_backend`: initializes the selected back end automatically from `LocalPreferences.toml` from `JACC.set_backend`. `JACC.@init_backend` should be used in your code before using any JACC.jl functionality. Recent improvements have made this process more seamless.

Example:
```julia
using JACC
JACC.@init_backend
```

!!! tip

    Always use `JACC.@init_backend` right after `import JACC` or `using JACC` for portable back end agnostic code.

## Memory models

- `JACC.array{T}()`: create a new array on the device with the specified type and size.
- `JACC.zeros`: create a new array on the device filled with zeros.
- `JACC.ones`: create a new array on the device filled with ones.
   
Advanced memory:
- `JACC.shared`: exploit fast-access cache memory on device. Use it inside kernel functions. Please see the paper [Valero-Lara IEEE HPEC 2024](https://ieeexplore.ieee.org/document/10938453) for more details.
- `JACC.Multi`: allows the programmability of multiple-GPU devices on a single node without the need of MPI. Please see the paper [Valero-Lara et al. IEEE eScience 2025]() 
- `JACC.@atomic`: create an atomic variable on the device for safe concurrent access. Wraps `@atomic` from the supported [Atomix.jl](https://github.com/JuliaConcurrent/Atomix.jl) package in the JuliaGPU ecosystem.

## Kernel launching

- `JACC.parallel_for`: launch a parallel for loop (each `i` index is independent).
- `JACC.parallel_reduce`: launch a parallel reduce operation.
