#include "MPIDetails.h"

int MPIDetails::mpi_rank = -1;
int MPIDetails::mpi_comm_size = -1;
bool MPIDetails::init = false;

MPIDetails::MPIDetails()
{
}

MPIDetails::~MPIDetails()
{
}

void MPIDetails::Init()
{
	MPI_Comm_size(PETSC_COMM_WORLD,&mpi_comm_size);
	MPI_Comm_rank(PETSC_COMM_WORLD,&mpi_rank);

	init=true;
}

int MPIDetails::Rank()
{
	if(!init) Init();

	return mpi_rank;
}

int MPIDetails::CommSize()
{
	if(!init) Init();

	return mpi_comm_size;
}

