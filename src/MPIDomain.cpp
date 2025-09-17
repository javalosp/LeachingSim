#include "MPIDomain.h"
#include <stdexcept>

using namespace std;

Domain global;

void HandleMPIErr(int err)
{
	switch(err)
	{
	case MPI_SUCCESS:
		// do nothing
		break;
	case MPI_ERR_COMM:
		throw runtime_error("MPI_ERR_COMM detected.");
	case MPI_ERR_TYPE:
		throw runtime_error("MPI_ERR_TYPE detected.");
	case MPI_ERR_COUNT:
		throw runtime_error("MPI_ERR_COUNT detected.");
	case MPI_ERR_TAG:
		throw runtime_error("MPI_ERR_TAG detected.");
	case MPI_ERR_RANK:
		throw runtime_error("MPI_ERR_RANK detected.");
	default:
		throw runtime_error("Unknown MPI error received.");
	}
}
