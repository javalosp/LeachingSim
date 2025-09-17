#ifndef COMPILER_OPTS_H_
#define COMPILER_OPTS_H_

/*
 * Options which are set at compile time.
 */

#include <memory>
#include <mpi.h>

#ifndef MPI_RAW_TYPE
#define MPI_RAW_TYPE MPI_UNSIGNED_SHORT
#endif

// convert our preprocessor macro into a real C++ type
template <MPI_Datatype T>
struct CPP_type
{
	typedef unsigned short type;
}; // default
template <>
struct CPP_type<MPI_UNSIGNED_SHORT>
{
	typedef unsigned short type;
};
template <>
struct CPP_type<MPI_UNSIGNED_CHAR>
{
	typedef unsigned char type;
};
template <>
struct CPP_type<MPI_UNSIGNED>
{
	typedef unsigned int type;
};
template <>
struct CPP_type<MPI_DOUBLE>
{
	typedef double type;
};

typedef CPP_type<MPI_RAW_TYPE>::type RAWType;

typedef std::unique_ptr<RAWType[]> pRAWType;

enum PixelType
{
	// Numbering starting from 0
	// Rock = 2,
	// Air = 0,
	// Pore = 1,
	// Sulphide = 3

	Rock = 3,
	Air = 1,
	Pore = 2,
	Sulphide = 4
};

#define MPIOUT(a) \
	if (a == 0)   \
	std::cout

// handle cx1's flakey support for C++11 std library (26/03/2014)
#ifdef _MSC_VER
#include <random>
namespace rng = std;
#else
#include <boost/random.hpp>
namespace rng = boost::random;
#endif

#endif /* COMPILER_OPTS_H_ */
