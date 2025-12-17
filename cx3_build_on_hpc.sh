#!/usr/bin/env bash

# Clean previous build artifacts
make clean

# Create release directory
mkdir -p release

# 1. Purge the old Intel environment
module purge

module load tools/prod

# 2. Load the FOSS Toolchain
# This loads GCC (compiler) and OpenMPI (mpi)
echo "Loading Toolchain foss/2023a..."
module load foss/2023a

# 3. Load Compatible Libraries
echo "Loading Dependencies..."

# Boost (Must match GCC version from foss-2023a)
module load Boost/1.82.0-GCC-12.3.0

# PETSc (Matches foss-2023a)
module load PETSc/3.20.3-foss-2023a

# VTK (Matches foss-2023a)
module load VTK/9.3.0-foss-2023a

# 4. Diagnostics
echo "----------------------------------------"
echo "Environment Loaded:"
module list
echo "----------------------------------------"
echo "Checking compiler..."
which mpicxx
echo "PETSc Dir:  $PETSC_DIR"
echo "VTK Root:   $EBROOTVTK"

# 5. Compile
# We assume the Makefile uses 'mpicxx' which now points to the GCC wrapper
make uint8 VERBOSE=1