SHELL = /bin/sh

BOOST_DIR = /apps/boost/1.72.0/build/boost_1_72_0
BOOST_LIB_DIR = /apps/boost/1.72.0/build/boost_1_72_0/stage/lib

VTK_INCLUDE_DIR = /apps/vtk/5.8.0/include/vtk-5.8
VTK_LIB_DIR = /apps/vtk/5.8.0/lib/vtk-5.8

PETSC_INCLUDE_DIR = /apps/petsc/3.3-p3-intel-11.1/include
PETSC_LIB_DIR = /apps/petsc/3.3-p3-intel-11.1/lib

CXXFLAGS = -I./ -I${PETSC_INCLUDE_DIR} -I$(VTK_INCLUDE_DIR) -I$(BOOST_DIR) -O3 -Wall -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-deprecated
CXXFLAGS_DEBUG = $(CXXFLAGS) -g

LIBS = -L$(PETSC_LIB_DIR) -lpetsc -L$(VTK_LIB_DIR) -lvtkIO -lvtkFiltering -lvtkCommon -L$(BOOST_LIB_DIR) -lboost_program_options -lboost_random -lboost_filesystem -lboost_system

MAKE = make
AR = ar
ARFLAGS = cr
CXX = mpicxx
#CXX = icpc

# executable file name
TARGET = LeachingSim

OBJS = main.o\
		Simulation.o\
		Domain.o\
		MPIDomain.o\
		MPIDetails.o

# underdirectories for binaries and source respectively
DBG_DIR = debug
REL_DIR = release
SRC_DIR = src

U16_OBJS = $(OBJS:%=$(REL_DIR)/%_U16)
U8_OBJS = $(OBJS:%=$(REL_DIR)/%_U8)
SOURCE = $(OBJS:%.o=$(SRC_DIR)/%.cpp)

#.SUFFIXES: .cpp .o
#
#.cpp.o:
#	$(CXX) $(CXXFLAGS) -c $< -o $@

$(REL_DIR)/%.o_U16: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ -DMPI_RAW_TYPE=MPI_UNSIGNED_SHORT -DX_ZY=1

$(REL_DIR)/%.o_U8: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ -DMPI_RAW_TYPE=MPI_UNSIGNED_CHAR

uint16: $(U16_OBJS)
	$(CXX) $(LFLAGS) $(U16_OBJS) -o $(REL_DIR)/$(TARGET)_uint16 $(LIBS) 
	
uint8: $(U8_OBJS)
	$(CXX) $(LFLAGS) $(U8_OBJS) -o $(REL_DIR)/$(TARGET)_uint8 $(LIBS) 
	
clean:
	rm -f $(REL_DIR)/*.o
	rm -f $(REL_DIR)/*.o_U8
	rm -f $(REL_DIR)/*.o_U16
