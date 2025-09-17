# Dynamic Leaching Simulator (LeachingSim)

This repository contains a C++ application for simulating chemical leaching processes in 3D porous media. The code uses a dynamic flow model to simulate pressure-driven fluid flow (Darcy's Law) and the transport of chemical via an advection-diffusion-reaction equation.

The simulation is designed for parallel execution on multi-core workstations or HPC clusters using MPI and the PETSc library. The geometry of the porous medium is defined by a 3D `.raw` image file from a micro-CT scanner. This geometry is discretised as a uniform grid, therefore, simple finite diferences schemes are suitable for quantities calculations.

## Features

* **Parallel Execution** The code is parallelised using MPI.
* **Advanced Numerical Solvers** Uses the PETSc library for high-performance, scalable linear solves.
* **Dynamic Flow Model** Simulates fluid flow based on Darcy's Law by solving a pressure equation at each time step.
* **Coupled Physics** Models species transport using an advection-diffusion-reaction equation, where the advection term is driven by the calculated fluid flow.
* **Flexible Input** Accepts 3D `.raw` binary files as input for the domain geometry and supports both 8-bit and 16-bit data types.
* **Standardised Output** Generates parallel VTK (`.pvti` and `.vti`) files, which can be easily visualised in Paraview.

---

## Prerequisites

To compile and run this project, you will need the following libraries and tools installed on your system:

* **C++ Compiler:** A modern compiler that supports C++17 (e.g., GCC, Clang, Intel C++).
* **MPI Implementation:** A standard MPI library such as [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/). The `mpicxx` compiler wrapper must be in your PATH.
* **PETSc**
* **Boost:** Specifically **Program Options** and **Filesystem** libraries. Your system's package manager can usually provide these (e.g., `libboost-program-options-dev`, `libboost-filesystem-dev`).
* **VTK:** The development libraries for VTK are required for writing the output files.

---

## Directory Structure

The repository is organised as follows. The `release/` (or `debug/`)and `output/` directories will be created during the build and run process and are ignored by Git.

```bash
.
├── Makefile           # Build script for the project
├── build_on_hpc.sh    # Bash script for building on Imperial's HPC
├── hpc_test.pbs       # Example script for running on Imperial's HPC (requires an image file)
└── src/               # Directory for all C++ source (.cpp) and header (.h) files
    ├── main.cpp
    ├── Simulation.cpp
    ├── Domain.cpp
    ├── MPIDetails.cpp
    ├── MPIDomain.cpp
    ├── MPIRawLoader.cpp
    ├── utils.cpp
    ├── Simulation.h
    ├── Domain.h
    ├── MPIDetails.h
    ├── MPIDomain.h
    ├── MPIRawLoader.h
    ├── utils.h
    └── compiler_opts.h
```

---

## Compilation

The project is built using the provided `Makefile`.

1.  **Configure Library Paths:** Before compiling, you may need to edit the top of the `Makefile` to point to the correct include and library directories for **Boost** and **VTK** on your system.

2.  **Build the Executable:** The `Makefile` provides two targets to build the program for different input data types.

    * To compile for **8-bit** (`unsigned char`) raw files, run:
        ```bash
        make uint8
        ```
    * To compile for **16-bit** (`unsigned short`) raw files, run:
        ```bash
        make uint16
        ```

    The compiled executable (e.g., `LeachingSim_uint8`) will be placed in the `release/` directory.

For building on Imperial's HPC use the `build_on_hpc.sh` script provided

---

## Usage

The program is executed via the `mpirun` or `mpiexec` command. You must provide the path to the raw file and its dimensions.

### Example

Here is an example of processing a 338x338x283 8-bit raw image file using 4 parallel processes:

```bash
mpiexec -n 4 ./release/LeachingSim_uint8 \
       --raw-file /path/to/your/image.raw \
       --xext 338 \
       --yext 338 \
       --zext 283 \
       --dt 1.0 \
       --tmax 50.0 \
       --nout 10 \
       --out_dir ./simulation_output
```

## Command-Line Arguments

| Argument | Description | Required |
| :--- | :--- | :---: |
| `--raw-file` | The path to the input `.raw` binary file. | **Yes** |
| `--xext` | The extent (number of voxels) of the domain in the X dimension. | **Yes** |
| `--yext` | The extent (number of voxels) of the domain in the Y dimension. | **Yes** |
| `--zext` | The extent (number of voxels) of the domain in the Z dimension. | **Yes** |
| `--dt` | The simulation time step size in seconds. Defaults to `1.0`. | No |
| `--tmax` | The maximum simulation time in seconds. Defaults to `10.0`. | No |
| `--nout` | The frequency of output (writes files every N-th time step). Defaults to `1`. | No |
| `--out_dir` | The directory where output VTK files will be saved. Defaults to `./output`. | No |
| `--header_size`| The size of the file header in bytes to skip. Defaults to `0`. | No |
| `--D` | The diffusivity ($m^2/s$). Defaults to `1.0`. | No |
| `--kreac` | The reaction rate coefficient. Defaults to `1.0`. | No |
| `--help, -h` | Prints this help message and exits. | No |

## Input and Output
### Input
The tool expects a single, headerless (or with a skippable header) binary .raw file containing voxel data. The data should be either 8-bit unsigned char or 16-bit unsigned short per voxel, matching the version of the program you compiled.

### Output
The program generates a set of files in the specified output directory for each saved time step:

* **A master file**: `output_000000.pvti`, `output_000010.pvti`, etc., for each time step.

* **Part files**: `output_0_000000.vti`, `output_1_000000.vti`, etc., with one file for each MPI process.



You can open the single `.pvti` file in ParaView to visualise results on the unified domain.