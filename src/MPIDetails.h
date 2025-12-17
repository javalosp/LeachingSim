#ifndef MPIDETAILS_H_
#define MPIDETAILS_H_

// #include <mpi.h>
#include <petsc.h>

class MPIDetails
{
public:
	static int Rank();
	static int CommSize();

private:
	MPIDetails();
	virtual ~MPIDetails();

	static void Init();

	static int mpi_rank;
	static int mpi_comm_size;

	static bool init;
};

#endif /* MPIDETAILS_H_ */
