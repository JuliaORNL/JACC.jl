# JACC.jl

[![CI-CPU](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-cpu.yaml/badge.svg)](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-cpu.yaml)
[![CI-GPU-NVIDIA](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-gpu-NVIDIA.yaml/badge.svg)](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-gpu-NVIDIA.yaml)
[![CI-GPU-AMD](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-gpu-AMD.yaml/badge.svg)](https://github.com/JuliaORNL/JACC.jl/actions/workflows/ci-gpu-AMD.yaml)


CPU/GPU performance portable layer for Julia

JACC.jl follows a function as a argument approach in combination with the power of Julia's ecosystem for multiple dispatch, GPU access via [JuliaGPU back ends](https://juliagpu.org/), and [package extensions](https://julialang.org/blog/2023/04/julia-1.9-highlights/#package_extensions) since Julia v1.9 . Similar to portable layers like Kokkos, users would pass a size and a function including its arguments to a `parallel_for` or `parallel_reduce` function.
The overall goal is to write a single source code that can be executed on multiple vendor CPU and GPU parallel programming environments. JACC meant to simplify CPU/GPU kernel programming using a simple application programming interface (API).

## Quick start

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


## Resources

- For an app integration example see the [GrayScott.jl JACC branch](https://github.com/JuliaORNL/GrayScott.jl/tree/GrayScott-JACC) and the [Simulation.jl code](https://github.com/JuliaORNL/GrayScott.jl/blob/GrayScott-JACC/src/simulation/Simulation.jl) for writing kernels with JACC.jl and selecting specific vendor back ends in Julia.
- JuliaCon 2023 presentation [video](https://live.juliacon.org/talk/AY8EUX).
- SC24 JACC [Best Research Poster Finalist](https://sc24.supercomputing.org/proceedings/poster/poster_pages/post113.html)
- SC24 WACCPD [presentation](https://sc24.conference-program.com/presentation/?id=ws_waccpd101&sess=sess760) and [paper](https://conferences.computer.org/sc-wpub/pdfs/SC-W2024-6oZmigAQfgJ1GhPL0yE3pS/555400b955/555400b955.pdf)
- MiniVATES.jl proxy application [repository](https://github.com/JuliaORNL/MiniVATES.jl) and SC24 XLOOP [best paper using JACC.jl](https://conferences.computer.org/sc-wpub/pdfs/SC-W2024-6oZmigAQfgJ1GhPL0yE3pS/555400c107/555400c107.pdf)

## Citation

If you find JACC.jl useful please cite the following paper:

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

Funded by the US Department of Energy Advanced Scientific Computing Research (ASCR) projects:

- S4PST and PESO as part of the Next Generation of Scientific Software Technologies (NGSST)
- ASCR Competitive Portfolios for Advanced Scientific Computing Research

Former sponsors:

- [ASCR Bluestone X-Stack](https://csmd.ornl.gov/Bluestone)
- The Exascale Computing Project - [PROTEAS-TUNE](https://www.ornl.gov/project/proteas-tune) 
