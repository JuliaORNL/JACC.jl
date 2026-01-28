## Acknowledgements

JACC is funded by the US Department of Energy Advanced Scientific Computing Research
(ASCR) projects:

- [S4PST](https://s4pst.org/) as part of the Next Generation of Scientific Software Technologies (NGSST) ASCR Program. 
- NGSST sponsors the Consortium for the Advancement of Scientific Software, [CASS](https://cass.community/)
- ASCR Competitive Portfolios for Advanced Scientific Computing Research, MAGMA/Fairbanks

Former sponsors:

- [ASCR Bluestone X-Stack](https://csmd.ornl.gov/Bluestone)
- The Exascale Computing Project -
  [PROTEAS-TUNE](https://www.ornl.gov/project/proteas-tune) 

JACC would not be possible without the contributions of the [Julia language](https://julialang.org/) and the [JuliaGPU](https://juliagpu.org/) community and the amazing GPU work of the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) and [Metal.jl](https://github.com/JuliaGPU/Metal.jl) backend developers.

## Citing JACC
Much of JACC is motivated by the Julia desire to make high-performance computing more accessible to a broader range of users. If you use JACC in your research or projects, we would appreciate it if you could cite our paper from [SC24-WAACPD](https://doi.org/10.1109/SCW63240.2024.00245), open version available [here](https://conferences.computer.org/sc-wpub/pdfs/SC-W2024-6oZmigAQfgJ1GhPL0yE3pS/555400b955/555400b955.pdf).

bib entry:

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

Other papers that contributed to JACC's exploratory research include:

- [JACC shared at IEEE HPEC](https://doi.org/10.1109/HPEC62836.2024.10938453)
- [JACC Multi-GPU IEEE eScience](https://www.escience-conference.org/2025/papers)

## License
JACC.jl is licensed under the permissive MIT License, UT-BATTELLE, LLC owns the Copyright. See the [LICENSE](https://github.com/JuliaGPU/JACC.jl/blob/main/LICENSE) file for details. We will remain open-source and welcome contributions from the community.

## Frequently Asked Questions

1. **How does JACC differ from other Julia GPU programming packages?**

   JACC focuses on providing a high-level, backend-agnostic API for parallel computing that abstracts away the complexities of different CPU/GPU architectures. It leverages Julia's metaprogramming capabilities to offer a unified interface for various backends, making it easier for developers to write portable code without having to learn the nuances of each backend or GPU programming model. Hence, JACC.jl is complimentary to existing programming models in the Julia ecosystem and targets the needs of users seeking a unified, portable approach for HPC codes. We invite users to explore the [JuliaGPU](https://juliagpu.org/) ecosystem for more specialized GPU programming needs. One important thing to note is that JACC is not a new backend itself but rather a layer on top of existing backends like CUDA.jl, AMDGPU.jl, Metal.jl, and oneAPI.jl. We welcome collaborations with other JuliaGPU backend developers to ensure compatibility and leverage their strengths.

2. **What is the performance overhead of using JACC compared to writing backend-specific code?**

   While JACC introduces some abstraction layers, it is designed to minimize performance overhead. The just-in-time (JIT) compilation and LLVM-based optimizations in Julia help ensure that the generated code is efficient. Our SC24-WAACPD paper includes performance benchmarks demonstrating that JACC can achieve performance comparable to backend-specific implementations for many workloads. However, for highly specialized or performance-critical applications, users may still choose to write backend-specific code when necessary. This is ongoing exploratory research with actual science applications, see [JACC-applications](https://github.com/JuliaORNL/JACC-applications).

3. **Who uses JACC?**

    JACC is primarily targeted at scientists, engineers, and researchers. We thank the early adopters for their feedback, in particular CERFACS, the University of Tokyo, Riken [FugakuNext programming environments](https://www.dropbox.com/scl/fi/gxr4y51kmruajrjwzy43q/FugakuNEXT_7th_ApplicationSeminar.pdf?rlkey=awt4rh2lern0t8yb56a74jlmz&st=7d4c3t9s&dl=0),the [New Jersey Institute of Technology]() and Oak Ridge National Laboratory (where it originates) and anyone adopting Julia for their scietific computing capabilities. We aim to grow this community further. If you are using JACC in your projects, please let us know! We would love to hear about your experiences and use cases.
    
4. **What's JACC's governance model?**

    JACC is open-source software hosted on GitHub funded primarily by the US Department of Energy for HPC science needs with the core development done at ORNL. Hence, our roadmap is driven by both community needs and the evolving landscape of HPC and GPU computing. We welcome contributions from the community, including bug reports, feature requests, and code contributions. The core development team at ORNL oversees the project's direction, but we encourage collaboration and input from users and contributors worldwide. Open an issue on GitHub to get involved.