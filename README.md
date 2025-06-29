# JACC.jl

[![CI-CPU](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-cpu.yaml/badge.svg)](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-cpu.yaml)
[![CI-GPU-NVIDIA](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-gpu-NVIDIA.yaml/badge.svg)](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-gpu-NVIDIA.yaml)
[![CI-GPU-AMD](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-gpu-AMD.yaml/badge.svg)](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-gpu-AMD.yaml)

CPU/GPU performance portable layer for Julia

JACC.jl follows a function as a argument approach in combination with the power
of Julia's ecosystem for multiple dispatch, GPU access via [JuliaGPU back
ends](https://juliagpu.org/), and [package
extensions](https://julialang.org/blog/2023/04/julia-1.9-highlights/#package_extensions)
since Julia v1.9 . Similar to portable layers like Kokkos, users would pass a
size and a function including its arguments to a `parallel_for` or
`parallel_reduce` function. The overall goal is to write a single source code
that can be executed on multiple vendor CPU and GPU parallel programming
environments. JACC meant to simplify CPU/GPU kernel programming using a simple
application programming interface (API).

## Quick start

1. Set a back end: `"cuda"`, `"amdgpu"`, or `"threads"` (default). This will
    generate a `LocalPreferences.toml` file.

    ```julia
    julia> import JACC
    julia> JACC.set_backend("cuda")
    ```
    **NOTE:** This will also add the backend package (`CUDA` in this case)
    as a dependency to the current project.

2. Import JACC and load appropriate extension. `@init_backend` inserts an
    `import` statement so that you don't have to reference a specific backend in
    your code. (It must therefore be called at a top-level scope.)

    ```julia
    import JACC
    JACC.@init_backend
    ```

    **NOTE:** Without calling `@init_backend` if your backend is something other
    than `"threads"`, using most of JACC functions will result in an error
    related to, using CUDA for example, `get_backend(::Val(:cuda))`.

3. Run a kernel example (see tests directory)

    ```julia
    function axpy(i, alpha, x, y)
      @inbounds x[i] += alpha * y[i]
    end

    N = 10
    # Generate random vectors x and y of length N for the interval [0, 100]
    x = round.(rand(Float32, N) * 100)
    y = round.(rand(Float32, N) * 100)
    alpha = 2.5

    x_d = JACC.array(x)
    y_d = JACC.array(y)
    JACC.parallel_for(N, axpy, alpha, x_d, y_d)
    ```

    **NOTE:** `JACC.array` converts a `Base.Array` to the array type used by the
    current backend. If you need to use the array type in your code (for example
    as function parameter or in a struct, use `JACC.array_type()`:
    ```julia
    const JACCArray = JACC.array_type()
    function f(a::JACCArray{Float32, 1})
        # ...
    end
    ```

## Resources

- For an app integration example see the [GrayScott.jl](https://github.com/JuliaORNL/GrayScott.jl) and the
  [Simulation.jl code](https://github.com/JuliaORNL/GrayScott.jl/blob/main/src/simulation/Simulation.jl) and search for `JACC` keyword.
  for writing kernels with JACC.jl and selecting specific vendor back ends in Julia.
- JuliaCon 2023 presentation [video](https://live.juliacon.org/talk/AY8EUX).
- SC24 JACC [Best Research Poster Finalist](https://sc24.supercomputing.org/proceedings/poster/poster_pages/post113.html)
- SC24 WACCPD [presentation](https://sc24.conference-program.com/presentation/?id=ws_waccpd101&sess=sess760)
  and [paper](https://conferences.computer.org/sc-wpub/pdfs/SC-W2024-6oZmigAQfgJ1GhPL0yE3pS/555400b955/555400b955.pdf)
- MiniVATES.jl proxy application [repository](https://github.com/JuliaORNL/MiniVATES.jl)
  and SC24 XLOOP [best paper using JACC.jl](https://conferences.computer.org/sc-wpub/pdfs/SC-W2024-6oZmigAQfgJ1GhPL0yE3pS/555400c107/555400c107.pdf)
- [OLCF Tutorial 2025](https://www.olcf.ornl.gov/calendar/juliaforsci2025/)

## Citation

If you find JACC.jl useful please cite the following paper from [SC24-WAACPD](https://doi.org/10.1109/SCW63240.2024.00245), open version available [here](https://conferences.computer.org/sc-wpub/pdfs/SC-W2024-6oZmigAQfgJ1GhPL0yE3pS/555400b955/555400b955.pdf).

```
@INPROCEEDINGS{JACC,
  author={Valero-Lara, Pedro and Godoy, William F and Mankad, Het and Teranishi, Keita and Vetter, Jeffrey S and Blaschke, Johannes and Schanen, Michel},
  booktitle={Proceedings of the SC '24 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis},
  title={{JACC: Leveraging HPC Meta-Programming and Performance Portability with the Just-in-Time and LLVM-based Julia Language}},
  year={2024},
  volume={},
  number={},
  pages={},
  doi={10.1109/SCW63240.2024.00245}
}
```

## Sponsors:

Funded by the US Department of Energy Advanced Scientific Computing Research
(ASCR) projects:

- [S4PST](https://s4pst.org/) and [PESO](https://pesoproject.org/) as part of the Next Generation of Scientific Software Technologies (NGSST) ASCR Program. 
- NGSST sponsors the Consortium for the Advancement of Scientific Software, [CASS](https://cass.community/)
- ASCR Competitive Portfolios for Advanced Scientific Computing Research, MAGMA

Former sponsors:

- [ASCR Bluestone X-Stack](https://csmd.ornl.gov/Bluestone)
- The Exascale Computing Project -
  [PROTEAS-TUNE](https://www.ornl.gov/project/proteas-tune) 
