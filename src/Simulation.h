#ifndef SIMULATION_H_
#define SIMULATION_H_

// #include <mpi.h>
#include <petsc.h>
#include <string>
#include <random>
#include "compiler_opts.h"
#include "MPIRawLoader.h"
#include "MPIDomain.h"

// Enum for selecting which data arrays to write to VTK files
enum VTKOutput
{
	Conc = 0x001,
	Frac = 0x004,
	Type = 0x008,
	Pressure = 0x010,
	Permeability = 0x020, // This corresponds to effective permeability
	Saturation = 0x040,
	Flux_Vec = 0x080, // New flag for the flux vector
	CapPressure = 0x100,
	RelPermeability = 0x200
};

class Simulation
{
public:
	Simulation(size_t seed);
	virtual ~Simulation();

	// Core Simulation Setup & I/O
	void readRAW(std::string, size_t);
	void setupDomain(int3 global_extent);
	void setupPETSc(bool zero_conc);
	void writeVTKFile(std::string fname_root, size_t tstep, size_t data_flags);

	// Physics Initialisation & Time-Stepping
	void initProperties();
	void initFrac();
	void initSaturation();
	void setupReactionRates();
	int updateFrac(double dt);
	void doExchange();

	void updateSaturation(double dt);
	void updateCapillaryPressure();
	void updateRelativePermeability();
	void handleSurfaceEffects();
	void handlePrecipitation();

	// Solvers for Dynamic Model
	void setupPressureEqns();
	void solvePressure();
	void setupConcentrationEqns(double dt);
	void solveConc();
	void calculateFlux();

	// Template & Helper Functions
	template <IndexScheme S>
	void decomposeDomain();

	template <typename T, int P, IndexScheme S>
	void writeData(const MPIDomain<T, P, S> &data_dom, std::string fname, MPI_Datatype data_type);

	// Public Parameters
	double D;
	double dx;
	double Dpore_fac;
	double kext;
	double kreac;
	bool use_gamma;
	double gamma_alpha;
	double gamma_beta;
	double c_sat;			 // Saturation concentration limit
	double evaporative_flux; // Rate of evaporation at the air interface (m/s)
	// van Genuchten model parameters
	double vg_alpha; // van Genuchten alpha parameter (1/m)
	double vg_n;	 // van Genuchten n parameter ()
	double s_res;	 // Residual saturation ()

	// Public Data Fields
	// Input & Material Properties
	MPIDomain<RAWType, 1, IDX_SCHEME> img_data;
	MPIDomain<float, 1, IDX_SCHEME> frac;
	MPIDomain<double, 1, IDX_SCHEME> permeability;
	MPIDomain<double, 1, IDX_SCHEME> viscosity;
	// Solution Fields
	MPIDomain<double, 1, IDX_SCHEME> pressure;
	MPIDomain<double, 1, IDX_SCHEME> conc;
	MPIDomain<double, 1, IDX_SCHEME> flux_x;
	MPIDomain<double, 1, IDX_SCHEME> flux_y;
	MPIDomain<double, 1, IDX_SCHEME> flux_z;
	// Add/update data fields
	MPIDomain<double, 1, IDX_SCHEME> saturation;
	MPIDomain<double, 1, IDX_SCHEME> cap_pressure;
	MPIDomain<double, 1, IDX_SCHEME> rel_permeability;

	friend std::ostream &operator<<(std::ostream &, const Simulation &);

private:
	int mpi_rank;
	int mpi_comm_size;

	Domain local;
	Domain global;

	// Reaction rates
	size_t num_grains;
	std::vector<double> ks;

	// RNG
	size_t seed_val;
	rng::mt19937_64 generator;

	// PETSc data
	Vec sources_vec;
	Vec conc_vec;
	Mat coeff_mat;

	// PETSc solver variables
	KSP ksp;
	PC pc;
	MatNullSpace nullsp;
};

std::ostream &operator<<(std::ostream &, const Simulation &);

// Implementation for template functions must remain in the header
template <typename T, int P, IndexScheme S>
void Simulation::writeData(const MPIDomain<T, P, S> &dom, std::string fname, MPI_Datatype data_type)
{
	MPIOUT(mpi_rank) << "Writing data to " << fname << std::endl;

	MPI_File file;
	MPI_Status status;

	assert(MPISubIndex<IDX_SCHEME>::GetOffset() == MPISubIndex<IDX_SCHEME>(dom, 0).globalArrayId(dom) && "The index offset and (global) inverted local origin do not match.");
	if (mpi_rank == mpi_comm_size - 1)
		assert((MPISubIndex<IDX_SCHEME>::GetOffset() + dom.extent.size()) == global.extent.size() && "Calculated filesize in writeData does not match global size.");

	MPI_File_open(PETSC_COMM_WORLD, const_cast<char *>(fname.c_str()), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
	MPI_Offset offset = MPISubIndex<IDX_SCHEME>::GetOffset() * sizeof(T);
	MPI_File_seek(file, offset, MPI_SEEK_SET);
	MPI_File_write(file, dom.getData().get() + dom.pad_size, dom.extent.size(), data_type, &status);
	MPI_File_close(&file);
}

#endif /* SIMULATION_H_ */