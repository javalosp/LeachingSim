#ifndef DOMAIN_H_
#define DOMAIN_H_

#include <iostream>
#include <sstream>
// #include <mpi.h>
#include <petsc.h>
#include "compiler_opts.h"

struct int3
{
	int i, j, k;
	int3()
		: i(0), j(0), k(0) {
		  };

	int3(int i, int j, int k)
		: i(i), j(j), k(k)
	{
	}

	// virtual ~int3()
	~int3()
	{
	}

	virtual bool operator==(const int3 &b)
	{
		return (i == b.i && j == b.j && k == b.k);
	}

	virtual unsigned int size() const
	{
		return i * j * k;
	}

	int3 operator+(const int3 &b)
	{
		return int3(i + b.i, j + b.j, k + b.k);
	}

	int3 operator-(const int3 &b)
	{
		return int3(i - b.i, j - b.j, k - b.k);
	}
};

std::ostream &operator<<(std::ostream &out, const int3 &v);

extern MPI_Datatype MPI_DOMAIN;

class Domain
{
public:
	Domain();
	virtual ~Domain();

	virtual void setup(int3 origin, int3 extent);

	static void BuildMPIDataType();

	int3 origin;
	int3 extent;
};

std::ostream &operator<<(std::ostream &out, const Domain &d);

/*
 * Class for handling the conversion between 3D (subscript) indices
 * and 1D (array) indices.
 */

// indexing conventions
enum IndexScheme
{
	XFastest,
	ZFastest,
};

// the class
template <IndexScheme S>
class SubIndex : public int3
{
public:
	virtual ~SubIndex()
	{
	}

	SubIndex(int i, int j, int k)
		: int3(i, j, k)
	{
	}

	SubIndex(const Domain &dom, int array_id);
	int arrayId(const Domain &dom);

	virtual bool valid(const Domain &dom)
	{
		return !(i < dom.origin.i || i >= (dom.origin.i + dom.extent.i) || j < dom.origin.j || j >= (dom.origin.j + dom.extent.j) || k < dom.origin.k || k >= (dom.origin.k + dom.extent.k));
	}
};

#endif /* DOMAIN_H_ */
