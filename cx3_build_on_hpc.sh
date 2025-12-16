#!/usr/bin/env bash

# Clean previous build
make clean

# Create release directory if it doesn't exist
mkdir -p release

# This is just to make sure that we are using the appropriate version of modules
module purge

module load tools/prod

# Load system modules
# Use 'intel/2023a' because it contains the 2023.1.0 compiler required by Boost.
#echo "Loading Toolchain intel/2023a..."
module load intel/2023a
# Alternatively, we can use iimpi, but it requires an additional module
#module load iimpi/2023a       # Loads Compiler + MPI
#module load imkl/2023.1.0     # Loads MKL library for PETSc

# Load Compatible Libraries
#echo "Loading Dependencies..."
# Boost: use the specific version that matches the compiler in 'intel/2023a'
module load Boost/1.82.0-intel-compilers-2023.1.0

# PETSc & VTK: 
module load PETSc/3.17.4-foss-2022a
module load VTK/9.2.2-foss-2022a

# Some checks
echo "----------------------------------------"
echo "Environment Loaded:"
module list
echo "----------------------------------------"
echo "Checking compiler..."
which mpiicpc
echo "Boost Root: $EBROOTBOOST"
echo "PETSc Dir:  $PETSC_DIR"

# Compile
#'VERBOSE=1' helps to see if the correct -I and -L paths are being used
make uint8 VERBOSE=1