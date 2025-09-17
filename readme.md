# raw2vtk - RAW to VTK Parallel Preprocessor

This project provides a tool for converting large 3D `.raw` image files into a parallel VTK format uniform grid suitable for futher processing and visualisation. The tool uses MPI for domain decomposition, allowing it to efficiently process datasets that are too large for a single process to handle.


## Features

* **Data Type Support:** Natively handles both 8-bit (`unsigned char`) and 16-bit (`unsigned short`) raw data.
* **Standard Output Format:** Generates VTK Image Data (`.vti`) files for each process and a master Parallel VTK Image Data (`.pvti`) file that groups them for easy loading.
* **Configurability:** All parameters, including file paths, domain dimensions, and data types, are configurable via command-line arguments.

---

## Prerequisites

To compile and run this project, you will need the following libraries and tools installed on your system:

* **C++ Compiler:** A modern compiler that supports C++17 (e.g., GCC, Clang, Intel C++).
* **MPI Implementation:** A standard MPI library such as [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/). The `mpicxx` compiler wrapper must be in your PATH.
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
├── src/               # Directory for all C++ source (.cpp) files
│   ├── main.cpp
│   ├── Preprocessor.cpp
│   ├── Domain.cpp
│   ├── MPIDetails.cpp
│   ├── MPIDomain.cpp
│   └── MPIRawLoader.cpp
├── Preprocessor.h     # Header files are in the root directory
├── Domain.h
├── MPIDetails.h
├── MPIDomain.h
├── MPIRawLoader.h
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

    The compiled executable (e.g., `raw2vtk_uint8`) will be placed in the `release/` directory.

---

## Usage

The program is executed via the `mpirun` or `mpiexec` command. You must provide the path to the raw file and its dimensions.

### Example

Here is an example of processing a 338x338x283 8-bit raw image file using 4 parallel processes:

```bash
mpiexec -n 4 ./release/raw2vtk_uint8 \
       --raw-file /path/to/your/image.raw \
       --x-ext 338 \
       --y-ext 338 \
       --z-ext 283 \
```

## Command-Line Arguments

| Argument       | Description                                                                    | Required |
| :------------- | :----------------------------------------------------------------------------- | :------: |
| `--raw-file`   | The path to the input `.raw` binary file.                                      |  **Yes** |
| `--x-ext`      | The extent (number of voxels) of the domain in the X dimension.                |  **Yes** |
| `--y-ext`      | The extent (number of voxels) of the domain in the Y dimension.                |  **Yes** |
| `--z-ext`      | The extent (number of voxels) of the domain in the Z dimension.                |  **Yes** |
| `--header-size`| The size of the file header in bytes to skip. Defaults to `0`.                 |    No    |
| `--output-dir` | The directory where the output VTK files will be saved. Defaults to `./output`. |    No    |
| `--help, -h`   | Prints the help message and exits.                                             |    No    |

## Input and Output
### Input
The tool expects a single, headerless (or with a skippable header) binary .raw file containing voxel data. The data should be either 8-bit unsigned char or 16-bit unsigned short per voxel, matching the version of the program you compiled.

### Output
The program generates a set of files in the specified output directory:

* **A master file**: material_domain.pvti (or a similar name based on the output path).

* **Part files**: material_domain_0.vti, material_domain_1.vti, etc., with one file for each MPI process.

You can open the single .pvti file in ParaView to visualise the unified domain.