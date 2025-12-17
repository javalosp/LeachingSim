SHELL = /bin/sh

# -----------------------------------------------------------------------------
# Module Environment Variables
# -----------------------------------------------------------------------------
# These variables are should be set by 'module load'.
# We use ?= to allow them to be overridden by the environment, but provide a fallback/check structure.

# BOOST
# Expects BOOST_ROOT to be set by the module.
BOOST_ROOT ?= /apps/boost/latest
BOOST_INC = $(BOOST_ROOT)/include
BOOST_LIB = $(BOOST_ROOT)/lib

# PETSC
# Expects PETSC_DIR to be set by the module.
PETSC_DIR ?= /apps/petsc/latest
PETSC_INC = $(PETSC_DIR)/include
PETSC_LIB = $(PETSC_DIR)/lib

# VTK
# Expects VTK_ROOT to be set by the module.
VTK_ROOT ?= $(EBROOTVTK)
# Fallback if the module doesn't set EBROOTVTK
ifeq ($(VTK_ROOT),)
    VTK_ROOT = /sw-eb/software/VTK/9.3.0-foss-2023a
endif
VTK_LIB = $(VTK_ROOT)/lib64

# VTK 9 puts headers in a versioned subdirectory (e.g., include/vtk-9.2).
# Use a shell command to find it automatically.
VTK_INC := $(shell find $(VTK_ROOT)/include -maxdepth 1 -name "vtk-*" | head -n 1)
ifeq ($(VTK_INC),)
    VTK_INC = $(VTK_ROOT)/include
endif

# -----------------------------------------------------------------------------
# Compiler & Flags
# -----------------------------------------------------------------------------
CXX = mpicxx

# Standard Flags
# -I flags now point to the variables defined above
#CXXFLAGS = -I./ -I$(PETSC_INC) -I$(VTK_INC) -I$(BOOST_INC) \
#           -O3 -Wall -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-deprecated
CXXFLAGS = -I./ -I$(PETSC_INC) -I$(VTK_INC) -I$(BOOST_INC) \
           -O3 -Wall -std=c++17 -Wno-deprecated

CXXFLAGS_DEBUG = $(CXXFLAGS) -g

# -----------------------------------------------------------------------------
# Linker Libraries
# -----------------------------------------------------------------------------

# VTK 9 MIGRATION NOTE:
#VTK_LIBS = -lvtkCommonCore -lvtkCommonDataModel -lvtkIOXML -lvtkIOLegacy -lvtkFiltersCore
VTK_LIBS = -lvtkCommonCore-9.3 \
           -lvtkCommonDataModel-9.3 \
           -lvtkIOXML-9.3 \
           -lvtkIOLegacy-9.3 \
           -lvtkFiltersCore-9.3 \
           -lvtkImagingCore-9.3 \
           -lvtkCommonExecutionModel-9.3 \
           -lvtksys-9.3

LIBS = -L$(PETSC_LIB) -lpetsc \
       -L$(VTK_LIB) $(VTK_LIBS) \
       -L$(BOOST_LIB) -lboost_program_options -lboost_random -lboost_filesystem -lboost_system

MAKE = make
AR = ar
ARFLAGS = cr

# Executable name
TARGET = LeachingSim

OBJS = main.o\
		Simulation.o\
		Domain.o\
		MPIDomain.o\
		MPIDetails.o

# Directories
DBG_DIR = debug
REL_DIR = release
SRC_DIR = src

U16_OBJS = $(OBJS:%=$(REL_DIR)/%_U16)
U8_OBJS = $(OBJS:%=$(REL_DIR)/%_U8)
SOURCE = $(OBJS:%.o=$(SRC_DIR)/%.cpp)

# -----------------------------------------------------------------------------
# Build Rules
# -----------------------------------------------------------------------------

# Rule for 16-bit (Standard)
$(REL_DIR)/%.o_U16: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ -DMPI_RAW_TYPE=MPI_UNSIGNED_SHORT -DDATA_TYPE=16 -DX_ZY=1

# Rule for 8-bit
# Added -DUSE_UINT8 to activate the logic in compiler_opts.h
$(REL_DIR)/%.o_U8: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ -DMPI_RAW_TYPE=MPI_UNSIGNED_CHAR -DDATA_TYPE=8 -DUSE_UINT8

# Link Targets
uint16: $(U16_OBJS)
	$(CXX) $(LFLAGS) $(U16_OBJS) -o $(REL_DIR)/$(TARGET)_uint16 $(LIBS) 
	
uint8: $(U8_OBJS)
	$(CXX) $(LFLAGS) $(U8_OBJS) -o $(REL_DIR)/$(TARGET)_uint8 $(LIBS) 
	
clean:
	rm -f $(REL_DIR)/*.o
	rm -f $(REL_DIR)/*.o_U8
	rm -f $(REL_DIR)/*.o_U16