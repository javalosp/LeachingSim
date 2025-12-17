#ifndef MPIDOMAIN_H_
#define MPIDOMAIN_H_

#include <memory>
#include <fstream>
// #include <mpi.h>
#include <cassert>
#include "MPIDetails.h"
#include "Domain.h"

// typedef
#define IDX_SCHEME ZFastest
typedef SubIndex<IDX_SCHEME> Index;

extern Domain global;

void HandleMPIErr(int MPI_ERR);

template <typename T, int Padding, IndexScheme S>
class MPIDomain : public Domain
{
public:
	MPIDomain();
	virtual ~MPIDomain();

	virtual void setup(int3 origin, int3 extent);
	static void SetGlobal(int3 origin, int3 extent);

	T &operator[](int arrayId);
	T &operator[](SubIndex<S> idx);

	std::unique_ptr<T[]> &getData();
	const std::unique_ptr<T[]> &getData() const;
	void take(std::unique_ptr<T[]> &data);

	void exchangePadding(MPI_Datatype exch_type);

	void serialize(std::ostream &fout);
	void deserialize(std::istream &fin);

	void debugPrint(std::ostream &fout);

	Domain padded;
	int pad_size;

protected:
	std::unique_ptr<T[]> data;
};

template <typename T, int Padding, IndexScheme S>
MPIDomain<T, Padding, S>::MPIDomain()
{
}

/**
 * @brief Sets up the local domain for an MPI process.
 * @details This function initialises the local domain's origin and extent, calculates the
 * dimensions of the padded "ghost cell" region, clips the padded region to the
 * global boundaries, and allocates memory for the data array.
 * @param orig The origin of this process's local (unpadded) domain.
 * @param ext The extent of this process's local (unpadded) domain.
 */
template <typename T, int Padding, IndexScheme S>
void MPIDomain<T, Padding, S>::setup(int3 orig, int3 ext)
{
	origin = orig;
	extent = ext;

	padded.origin = origin;
	padded.extent = extent;

	switch (S)
	{
	case ZFastest: // Decomposed along 'i'
		padded.origin.i = origin.i - Padding;
		padded.extent.i = extent.i + 2 * Padding;
		pad_size = Padding * extent.j * extent.k;
		break;
	case XFastest: // Decomposed along 'k'
		padded.origin.k = origin.k - Padding;
		padded.extent.k = extent.k + 2 * Padding;
		pad_size = Padding * extent.i * extent.j;
		break;
	}

	// ensure the padding does not go outside of the global domain
	// *Use 'else if' to make boundary clipping mutually exclusive
	// i-dimension clipping
	if (padded.origin.i < global.origin.i)
	{
		padded.extent.i -= (global.origin.i - padded.origin.i);
		padded.origin.i = global.origin.i;
	}
	else if (padded.origin.i + padded.extent.i > global.origin.i + global.extent.i)
	{
		padded.extent.i = (global.origin.i + global.extent.i) - padded.origin.i;
	}

	// j-dimension clipping
	if (padded.origin.j < global.origin.j)
	{
		padded.extent.j -= (global.origin.j - padded.origin.j);
		padded.origin.j = global.origin.j;
	}
	else if (padded.origin.j + padded.extent.j > global.origin.j + global.extent.j)
	{
		padded.extent.j = (global.origin.j + global.extent.j) - padded.origin.j;
	}

	// k-dimension clipping
	if (padded.origin.k < global.origin.k)
	{
		padded.extent.k -= (global.origin.k - padded.origin.k);
		padded.origin.k = global.origin.k;
	}
	else if (padded.origin.k + padded.extent.k > global.origin.k + global.extent.k)
	{
		padded.extent.k = (global.origin.k + global.extent.k) - padded.origin.k;
	}

	size_t n = 2;
	if (MPIDetails::Rank() == 0 || MPIDetails::Rank() == (MPIDetails::CommSize() - 1))
		n = 1;

	assert(pad_size * n + extent.size() == padded.extent.size() && "The local extent plus padding does not match the padded extent!");

	// allocate storage
	data = std::unique_ptr<T[]>(new T[padded.extent.size()]);
}

/**
 * @brief Exchanges padding (ghost) cells with neighboring MPI processes.
 * @details Uses non-blocking sends and receives (MPI_Isend/MPI_Irecv) to transfer
 * boundary data required for stencil-based computations.
 * @param exch_type The MPI_Datatype of the elements being exchanged.
 */
template <typename T, int Padding, IndexScheme S>
void MPIDomain<T, Padding, S>::exchangePadding(MPI_Datatype exch_type)
{
	MPI_Request reqs[4];
	// MPI_Status stats[4];

	int count = 0;
	if (MPIDetails::Rank() > 0)
	{
		MPI_Irecv(data.get(), pad_size, exch_type, MPIDetails::Rank() - 1, 0, PETSC_COMM_WORLD, &reqs[count]);
		count++;
		MPI_Isend(data.get() + pad_size, pad_size, exch_type, MPIDetails::Rank() - 1, 1, PETSC_COMM_WORLD, &reqs[count]);
		count++;
	}
	if (MPIDetails::Rank() < MPIDetails::CommSize() - 1)
	{
		MPI_Irecv(&(data.get()[padded.extent.size() - pad_size]), pad_size, exch_type, MPIDetails::Rank() + 1, 1, PETSC_COMM_WORLD, &reqs[count]);
		count++;
		MPI_Isend(&(data.get()[padded.extent.size() - 2 * pad_size]), pad_size, exch_type, MPIDetails::Rank() + 1, 0, PETSC_COMM_WORLD, &reqs[count]);
		count++;
	}

	int status = MPI_Waitall(count, reqs, MPI_STATUS_IGNORE);
	if (status == MPI_ERR_IN_STATUS)
	{
		throw std::runtime_error("MPI error exchanging padding.");
	}
	MPI_Barrier(PETSC_COMM_WORLD);
}

/**
 * @brief Sets the static global domain dimensions used for boundary checks.
 * @param origin The origin of the entire global domain (usually (0,0,0)).
 * @param extent The extent of the entire global domain.
 */
template <typename T, int Padding, IndexScheme S>
void MPIDomain<T, Padding, S>::SetGlobal(int3 origin, int3 extent)
{
	global.origin = origin;
	global.extent = extent;
}

template <typename T, int Padding, IndexScheme S>
MPIDomain<T, Padding, S>::~MPIDomain()
{
}

template <typename T, int Padding, IndexScheme S>
std::unique_ptr<T[]> &MPIDomain<T, Padding, S>::getData()
{
	return data;
}

template <typename T, int Padding, IndexScheme S>
const std::unique_ptr<T[]> &MPIDomain<T, Padding, S>::getData() const
{
	return data;
}

template <typename T, int Padding, IndexScheme S>
void MPIDomain<T, Padding, S>::take(std::unique_ptr<T[]> &data_in)
{
	// take ownership (data_in is invalid after this)
	data = std::move(data_in);
}

template <typename T, int Padding, IndexScheme S>
T &MPIDomain<T, Padding, S>::operator[](int arrayId)
{
	return data[arrayId];
}

template <typename T, int Padding, IndexScheme S>
T &MPIDomain<T, Padding, S>::operator[](SubIndex<S> idx)
{
	if (idx.valid(padded)) // local index
	{
		return data[idx.arrayId(padded)];
	}

	std::stringstream msg;
	msg << "Index " << idx << " out of bounds for padded region.";
	throw std::runtime_error(msg.str());
}

template <typename T, int Padding, IndexScheme S>
void MPIDomain<T, Padding, S>::serialize(std::ostream &fout)
{
	// write header
	fout.write((char *)&origin.i, sizeof(int));
	fout.write((char *)&origin.j, sizeof(int));
	fout.write((char *)&origin.k, sizeof(int));
	fout.write((char *)&extent.i, sizeof(int));
	fout.write((char *)&extent.j, sizeof(int));
	fout.write((char *)&extent.k, sizeof(int));

	fout.write((char *)&padded.origin.i, sizeof(int));
	fout.write((char *)&padded.origin.j, sizeof(int));
	fout.write((char *)&padded.origin.k, sizeof(int));
	fout.write((char *)&padded.extent.i, sizeof(int));
	fout.write((char *)&padded.extent.j, sizeof(int));
	fout.write((char *)&padded.extent.k, sizeof(int));

	// write data
	fout.write((char *)data.get(), sizeof(T) * padded.extent.size());
}

template <typename T, int Padding, IndexScheme S>
void MPIDomain<T, Padding, S>::deserialize(std::istream &in)
{
	// read header
	in.read((char *)&origin.i, sizeof(int));
	in.read((char *)&origin.j, sizeof(int));
	in.read((char *)&origin.k, sizeof(int));
	in.read((char *)&extent.i, sizeof(int));
	in.read((char *)&extent.j, sizeof(int));
	in.read((char *)&extent.k, sizeof(int));

	in.read((char *)&padded.origin.i, sizeof(int));
	in.read((char *)&padded.origin.j, sizeof(int));
	in.read((char *)&padded.origin.k, sizeof(int));
	in.read((char *)&padded.extent.i, sizeof(int));
	in.read((char *)&padded.extent.j, sizeof(int));
	in.read((char *)&padded.extent.k, sizeof(int));

	// allocate space
	data = std::unique_ptr<T[]>(new T[padded.extent.size()]);

	switch (S)
	{
	case XFastest:
		pad_size = Padding * extent.j * extent.i;
		break;
	case ZFastest:
		pad_size = Padding * extent.j * extent.k;
		break;
	}

	// read data
	in.read((char *)data.get(), sizeof(T) * padded.extent.size());
}

template <typename T, int Padding, IndexScheme S>
void MPIDomain<T, Padding, S>::debugPrint(std::ostream &fout)
{
	using namespace std;

	for (int i = padded.origin.i; i < (padded.origin.i + padded.extent.i); ++i)
	{
		for (int j = padded.origin.j; j < (padded.origin.j + padded.extent.j); ++j)
		{
			for (int k = padded.origin.k; k < (padded.origin.k + padded.extent.k); ++k)
			{
				fout << (double)data[Index(i, j, k).arrayId(padded)] << "\t";
				//  fout << Index(i,j,k).arrayId(padded) << "\t";
				//  fout << Index(i,j,k).arrayId(global) << "\t";
			}
			fout << endl;
		}
		fout << endl
			 << "-------------------------------------------------------------------------------------------------------------------------" << endl
			 << endl;
	}
}

/*
 * Class for converting between 3D (subscript) indices and
 * 1D (array) indices while keeping the local array index
 * continuous on a single MPI processes.
 */
template <IndexScheme S>
class MPISubIndex : public SubIndex<S>
{
public:
	MPISubIndex()
		: SubIndex<S>()
	{
	}

	MPISubIndex(int i, int j, int k)
		: SubIndex<S>(i, j, k)
	{
	}

	MPISubIndex(const Domain &dom, int local_array_idx)
		: SubIndex<S>(dom, local_array_idx)
	{
	}

	/**
	 * @brief Initialises static variables required for mapping between local and global indices.
	 * @details This function must be called once by all processes. It gathers the local domain
	 * information from all processes to compute the offsets needed for global index calculation.
	 * @param local_dom The unpadded local domain of the calling process.
	 * @param mpi_rank The rank of the calling process.
	 * @param mpi_comm_size The total number of MPI processes.
	 */
	static void Init(const Domain &local_dom, int mpi_rank, int mpi_comm_size)
	{
		// allocate storage for static variables
		offsets = std::move(std::unique_ptr<unsigned int[]>(new unsigned int[mpi_comm_size]));
		all_local_domains = std::move(std::unique_ptr<Domain[]>(new Domain[mpi_comm_size]));

		// distribute unpadded local domains
		Domain tmp = local_dom;
		/*
		 * Due to data padding MPI_Allgather does not behave as one would expect for MPI_DOMAIN
		 * so we must send the objects one-by-one to avoid this.
		 *
		 * MPI_Allgather(&tmp,1,MPI_DOMAIN,all_local_domains.get(),1,MPI_DOMAIN,PETSC_COMM_WORLD);
		 */
		for (int p = 0; p < mpi_comm_size; ++p)
		{
			if (p == mpi_rank)
				tmp = local_dom;
			MPI_Bcast(&tmp, 1, MPI_DOMAIN, p, PETSC_COMM_WORLD);
			all_local_domains[p] = tmp;
		}

		// calculate offsets by distributed extent of each
		offsets[0] = 0;
		for (int p = 1; p < mpi_comm_size; ++p)
		{
			offsets[p] = offsets[p - 1] + all_local_domains[p - 1].extent.size();
		}

		// store rank and comm_size
		MPISubIndex<S>::mpi_comm_size = mpi_comm_size;
		MPISubIndex<S>::mpi_rank = mpi_rank;
	}

	/**
	 * @brief Converts a 1D array index local to this process into a 1D global array index.
	 * @param local_idx The 1D index in the local process's data array.
	 * @return The corresponding 1D index in the global array.
	 */
	static int LocalToGlobal(int local_idx)
	{
		return offsets[mpi_rank] + local_idx;
	}

	/**
	 * @brief Calculates the global 1D array index for the current 3D index.
	 * @details Determines which process owns the 3D index and computes the global
	 * 1D index by adding that process's offset to its local 1D index.
	 * @param local_dom The MPIDomain of the current process.
	 * @return The calculated global 1D array index.
	 * @throws std::runtime_error if the 3D index is not found in any process's domain.
	 */
	template <typename T, int Padding>
	int globalArrayId(const MPIDomain<T, Padding, S> &local_dom)
	{
		if (SubIndex<S>::valid(local_dom)) // in our unpadded region
		{
			return LocalToGlobal(SubIndex<S>::arrayId(local_dom));
		}
		else
		{
			for (int p = 0; p < mpi_comm_size; ++p)
				if (SubIndex<S>::valid(all_local_domains[p]))
					return offsets[p] + SubIndex<S>::arrayId(all_local_domains[p]);

			std::stringstream msg;
			msg << "A Matching domain for global index " << *this << " was not found.";
			throw std::runtime_error(msg.str());
		}
	}

	/**
	 * @brief Gets the global array offset for the current MPI process.
	 * @return The number of elements preceding this process's data in the global 1D array.
	 */
	static unsigned int GetOffset()
	{
		return offsets[mpi_rank];
	}

	// protected:
	//  variables for getting global indices - common across all instances on an MPI process
	static int mpi_rank;
	static int mpi_comm_size;
	static std::unique_ptr<unsigned int[]> offsets;
	static std::unique_ptr<Domain[]> all_local_domains;
};

// static variable definitions
template <IndexScheme S>
std::unique_ptr<unsigned int[]> MPISubIndex<S>::offsets;
template <IndexScheme S>
std::unique_ptr<Domain[]> MPISubIndex<S>::all_local_domains;
template <IndexScheme S>
int MPISubIndex<S>::mpi_rank;
template <IndexScheme S>
int MPISubIndex<S>::mpi_comm_size;

#endif /* MPIDOMAIN_H_ */
