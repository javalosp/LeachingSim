#ifndef COMPILER_OPTS_H_
#define COMPILER_OPTS_H_

/*
 * Options which are set at compile time.
 */

#include <memory>
#include <mpi.h>
#include <iostream>
#include <cstdint>

#ifndef MPI_RAW_TYPE
#define MPI_RAW_TYPE MPI_UNSIGNED_SHORT
#endif

/*
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
*/

// Change the template logic
// The old template <MPI_Datatype T> could fail on OpenMPI because
// MPI handles (like MPI_UNSIGNED_SHORT) are often pointers, not constants.
// Replace this with a preprocessor check.

// If your Makefile defines 'USE_UINT8', we use char. Otherwise default to short.
#ifdef USE_UINT8
typedef uint8_t RAWType;
// We define the MPI type that matches our C++ type
#ifndef MPI_RAW_TYPE
#define MPI_RAW_TYPE MPI_UNSIGNED_CHAR
#endif
#else
// Default to 16-bit unsigned integer (Rock/Pore/Sulphide usually need >255)
typedef uint16_t RAWType;
#ifndef MPI_RAW_TYPE
#define MPI_RAW_TYPE MPI_UNSIGNED_SHORT
#endif
#endif

typedef std::unique_ptr<RAWType[]> pRAWType;

enum PixelType
{
	// Numbering starting from 0
	// Rock = 2,
	// Air = 0,
	// Pore = 1,
	// Sulphide = 3

	// These are the actual labels from CT scan .raw files
	Rock = 3,
	Air = 1,
	Pore = 2,
	Sulphide = 4,
	// Secondary mineral formation
	// this doesn't come from CT scan labels
	// ensure that the value for this new material type is not
	// already in use in the .raw
	Precipitate = 5
};

#define MPIOUT(a) \
	if (a == 0)   \
	std::cout

#include <random>
namespace rng = std;

#endif /* COMPILER_OPTS_H_ */
