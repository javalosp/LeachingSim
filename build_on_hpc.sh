#!/usr/bin/env bash

# Clean previous build
make clean

# Create release directory if it doesn't exist
mkdir -p release

# This is just to make sure that we are using the appropriate version of modules
module purge

## Load system modules
module load vtk/5.8.0 
#module load mpi/intel-2019
module load mpi/intel-2019.8.254 # This is prefered over the default 2019
# CX3 versions
#module load tools/prod
#module load iimpi/2021a
# Using CX3 version of mpi compilers makes boost unhappy
# Therefore we need to load intel suite
module load intel-suite/2019.4
module load boost/1.72.0 # This also loads intel-suite/2019.4 (boost requirement)
module load petsc/3.3-p3-intel-11

# This is required for linking the MPI version used for compiling petsc
export PATH=/apps/intel/ict/mpi/3.1.038/bin64:$PATH
export LD_LIBRARY_PATH=/apps/intel/ict/mpi/3.1.038/lib64:$LD_LIBRARY_PATH
# Find the appropriate MPI header files for the old petsc version used
# TODO: Update the code to a newer version of petsc
source /apps/intel/ict/mpi/3.1.038/bin64/mpivars.sh

make uint8 VERBOSE=1