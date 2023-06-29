#!/bin/bash

# change these 3 lines accordingly
PROJ_DIR=/lustre/orion/proj-shared/csc383/$USER
export JULIA_DEPOT_PATH=$PROJ_DIR/julia_depot
# location where you cloned JACC.jl
JACC_DIR=$PROJ_DIR/ProgrammingModels.jl/JACC.jl

# remove existing generated Manifest.toml
rm -f $JACC_DIR/Manifest.toml
rm -f $JACC_DIR/LocalPreferences.toml

# good practice to avoid conflicts with existing default modules
module purge

# load required modules
module load PrgEnv-cray/8.3.3 # has required gcc
module load cray-mpich
module load rocm/5.4.0
module load julia # default is 1.9.0

# Required to point at underlying modules above
export JULIA_AMDGPU_DISABLE_ARTIFACTS=1

# MPIPreferences to use spectrum-mpi
julia --project=$JACC_DIR -e 'using Pkg; Pkg.add("MPIPreferences")'
julia --project=$JACC_DIR -e 'using MPIPreferences; MPIPreferences.use_system_binary(; library_names=["libmpi_cray"], mpiexec="srun")'

# Regression being fixed with CUDA v4.0.0. CUDA.jl does lazy loading for portability to systems without NVIDIA GPUs
# julia --project=$JACC_DIR -e 'using Pkg; Pkg.add(name="CUDA", version="v3.13.1")' 

# Instantiate the project by installing packages in Project.toml
julia --project=$JACC_DIR -e 'using Pkg; Pkg.instantiate()'
