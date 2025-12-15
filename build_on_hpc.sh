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


# New modules for cx3


# Old: mpi/intel-2019.8.254 + intel-suite/2019.4
# Alternatives: intel/2023a (or intel/2023b ?)
#
# Another alternative: iimpi/2023a (Compiler + MPI) + imkl/2023.1.0 (MKL (Math Kernel Library) for PETSc)
# Use one or the other alternative (second alternative implies two separate modules, first alternative has everything wrapped (?)

# Old: boost/1.72.0
# Alternative: Boost/1.82.0-intel-compilers-2023.1.0 (important! compatibility with imtel/MPI compilers)


# Old: petsc/3.3-p3-intel-11
# Alternative: PETSc/3.17.4-foss-2022a (or greater)

# Old: vtk/5.8.0
# Alternatives: VTK/9.2.2-foss-2022a (or greater) - Default VTK/9.3.1-foss-2024a


# Alternative build file
##!/usr/bin/env bash

## Clean previous build
#make clean
#
## Create release directory if it doesn't exist
#mkdir -p release
#
## 1. Purge Old Environment
## This ensures no conflicts with the old 2019 modules
#module purge
#
## 2. Load the Toolchain
## We use 'intel/2023a' because it contains the 2023.1.0 compiler required by Boost.
#echo "Loading Toolchain intel/2023a..."
#module load intel/2023a
# Alternatively, we can use iimpi, but it requires an additional module
#module load iimpi/2023a       # Loads Compiler + MPI
#module load imkl/2023.1.0     # <--- You MUST add this manually for PETSc
#
## 3. Load Compatible Libraries
#echo "Loading Dependencies..."
#
## Boost: The specific version you selected that matches the compiler in 'intel/2023a'
#module load Boost/1.82.0-intel-compilers-2023.1.0
#
## PETSc & VTK: 
## Note: You must check 'module avail' to find the exact PETSc/VTK versions 
## that act as siblings to the Boost module (i.e., built with 2023a).
## I have put placeholders below—you must verify the exact versions on the system.
#module load PETSc/3.19.1-intel-2023a   # <--- Verify this version exists
#module load VTK/9.2.2-intel-2023a      # <--- Verify this version exists
#
## 4. Diagnostics
#echo "----------------------------------------"
#echo "Environment Loaded:"
#module list
#echo "----------------------------------------"
#echo "Checking compiler..."
#which mpiicpc
#echo "Boost Root: $EBROOTBOOST"
#echo "PETSc Dir:  $PETSC_DIR"
#
## 5. Compile
## 'VERBOSE=1' helps you see if the correct -I and -L paths are being used
#make uint8 VERBOSE=1