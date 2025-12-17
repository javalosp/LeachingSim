#ifndef RAWLOADERMPI_H_
#define RAWLOADERMPI_H_

#include <fstream>
#include <string>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include "MPIDomain.h"

/*
 * Class which loads distinct segments of a RAW voxel
 * image into memory on each process. The class does not
 * actually call any MPI routines as the file is read
 * on each process rather than sending data.
 */
template <typename T, int Padding, IndexScheme S>
class MPIRawLoader : public MPIDomain<T, Padding, S>
{
public:
	MPIRawLoader(std::string fname);
	virtual ~MPIRawLoader();

	void read(size_t header);

private:
	std::string fname;
};

template <typename T, int Padding, IndexScheme S>
MPIRawLoader<T, Padding, S>::MPIRawLoader(std::string fname)
	: fname(fname)
{
}

template <typename T, int Padding, IndexScheme S>
MPIRawLoader<T, Padding, S>::~MPIRawLoader()
{
}

/**
 * @brief Reads a designated segment of a binary .raw file into memory.
 * @details Each MPI process opens the same file but only reads and stores the
 * portion of the data corresponding to its assigned local domain.
 * @param header The size of the file header in bytes to skip before reading voxel data.
 * @throws std::runtime_error if the file cannot be opened.
 */
template <typename T, int Padding, IndexScheme S>
void MPIRawLoader<T, Padding, S>::read(size_t header)
{
	std::ifstream fin(fname.c_str(), std::ios::binary);
	if (!fin.is_open())
	{
		throw std::runtime_error("Cannot open file: " + fname);
	}

	// This function now reads data in contiguous scanlines.
	switch (S)
	{
	case ZFastest:
	{
		// In ZFastest, data is contiguous along the k-axis (a k-scanline).
		// The domain is decomposed along the i-axis.
		for (int i = this->origin.i; i < (this->origin.i + this->extent.i); ++i)
		{
			for (int j = this->origin.j; j < (this->origin.j + this->extent.j); ++j)
			{
				// Calculate the starting position of the scanline in the file.
				// long long offset = header + (long long)(i * global.extent.j * global.extent.k + j * global.extent.k) * sizeof(T);
				long long offset = header + (long long)i * global.extent.j * global.extent.k * sizeof(T) + (long long)j * global.extent.k * sizeof(T);
				fin.seekg(offset);

				// Calculate the destination address in this process's padded memory array.
				// We get the index of the first element (k=origin.k) of the scanline.
				int dest_idx = Index(i, j, this->origin.k).arrayId(this->padded);
				char *dest_addr = (char *)(this->data.get() + dest_idx);

				// Read the entire scanline (all k-values for this i,j) in a single operation.
				fin.read(dest_addr, (std::streamsize)this->extent.k * sizeof(T));
			}
		}
		break;
	}
	case XFastest:
	{
		// In XFastest, data is contiguous along the i-axis (an i-scanline).
		// The domain is decomposed along the k-axis.
		for (int k = this->origin.k; k < (this->origin.k + this->extent.k); ++k)
		{
			for (int j = this->origin.j; j < (this->origin.j + this->extent.j); ++j)
			{
				// Calculate the starting position of the scanline in the file.
				long long offset = header + (long long)(k * global.extent.i * global.extent.j + j * global.extent.i) * sizeof(T);
				fin.seekg(offset);

				// Calculate the destination address in this process's padded memory array.
				int dest_idx = Index(this->origin.i, j, k).arrayId(this->padded);
				char *dest_addr = (char *)(this->data.get() + dest_idx);

				// Read the entire scanline (all i-values for this j,k) in a single operation.
				fin.read(dest_addr, (std::streamsize)this->extent.i * sizeof(T));
			}
		}
		break;
	}
	}
}

#endif /* RAWLOADERMPI_H_ */
