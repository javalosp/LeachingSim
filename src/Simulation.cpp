#include "Simulation.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include <cassert>
#include <vtkSmartPointer.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkUnsignedShortArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkFloatArray.h>

using namespace std;

Simulation::Simulation(size_t seed)
	: seed_val(seed), generator(seed)
{
	mpi_rank = MPIDetails::Rank();
	mpi_comm_size = MPIDetails::CommSize();
}

Simulation::~Simulation()
{
	VecDestroy(&sources_vec);
	VecDestroy(&conc_vec);
	MatDestroy(&coeff_mat);
	KSPDestroy(&ksp);
}

/**
 * @brief Decomposes the global domain among processes.
 * @details This template is specialised for different IndexScheme enums (e.g., ZFastest)
 * to perform a 1D decomposition along the appropriate axis (i.e. ZFastest implies decomposing along X axis,
 * XFastest implies decomposing along Z axis).
 */
template <>
void Simulation::decomposeDomain<ZFastest>()
{
	local = global;
	int block_size = global.extent.i / mpi_comm_size;
	local.origin.i = mpi_rank * block_size;

	// This ensures the last process gets all remaining voxels
	local.extent.i = (mpi_rank == mpi_comm_size - 1) ? (global.extent.i - local.origin.i) : block_size;
}

template <>
void Simulation::decomposeDomain<XFastest>()
{
	local = global;
	int block_size = global.extent.k / mpi_comm_size;
	local.origin.k = mpi_rank * block_size;
	local.extent.k = (mpi_rank == mpi_comm_size - 1) ? (global.extent.k - local.origin.k) : block_size;
}

/**
 * @brief Reads the 3D domain geometry from a binary .raw file in parallel.
 * * Each MPI process reads only its assigned slice of the domain from the file.
 * @param fname The path to the input .raw file.
 * @param header The size of the file header in bytes to skip before reading data.
 */
void Simulation::readRAW(string fname, size_t header)
{
	MPIRawLoader<RAWType, 1, IDX_SCHEME> reader(fname);
	reader.setup(local.origin, local.extent);
	reader.read(header);
	img_data.take(reader.getData());
}

/**
 * @brief Sets up the global simulation domain, decomposes it across MPI processes, and
 * finally sets up the fields for quantities to be calculated
 * @param gextent The dimensions (i, j, k) of the entire dataset.
 */
void Simulation::setupDomain(int3 gextent)
{
	global.origin = int3();
	global.extent = gextent;
	decomposeDomain<IDX_SCHEME>();
	MPIDomain<double, 0, IDX_SCHEME>::SetGlobal(int3(), gextent);
	MPISubIndex<IDX_SCHEME>::Init(local, mpi_rank, mpi_comm_size);
	img_data.setup(local.origin, local.extent);
	conc.setup(local.origin, local.extent);
	frac.setup(local.origin, local.extent);
	pressure.setup(local.origin, local.extent);
	permeability.setup(local.origin, local.extent);
	viscosity.setup(local.origin, local.extent);
	flux_x.setup(local.origin, local.extent);
	flux_y.setup(local.origin, local.extent);
	flux_z.setup(local.origin, local.extent);
	saturation.setup(local.origin, local.extent);
}

/**
 * @brief Sets up the reaction rate coefficients for the different sulphide grains.
 * * Can use a single global rate or a gamma distribution for varied rates.
 */
void Simulation::setupReactionRates()
{
	int max_grain_id = 0;
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				RAWType voxel = img_data[Index(i, j, k)];
				if (voxel >= Sulphide && voxel > max_grain_id)
					max_grain_id = voxel;
			}
	int tmp;
	MPI_Allreduce(&max_grain_id, &tmp, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);
	max_grain_id = tmp;
	num_grains = max_grain_id - Sulphide + 1;
	ks.assign(num_grains, kreac);
	cout << "\nin Simulation::setupReactionRates" << endl;
	cout << "\nSulphide: " << Sulphide << endl;
}

/**
 * @brief Creates and configures all necessary PETSc objects.
 * * Initialises the parallel matrix, vectors, and the KSP solver.
 */
void Simulation::setupPETSc()
{
	VecCreate(PETSC_COMM_WORLD, &sources_vec);
	VecSetSizes(sources_vec, local.extent.size(), PETSC_DECIDE);
	VecSetFromOptions(sources_vec);
	VecDuplicate(sources_vec, &conc_vec);
	MatCreate(PETSC_COMM_WORLD, &coeff_mat);
	MatSetSizes(coeff_mat, local.extent.size(), local.extent.size(), PETSC_DETERMINE, PETSC_DETERMINE);
	MatSetType(coeff_mat, MATMPIAIJ);
	MatMPIAIJSetPreallocation(coeff_mat, 8, PETSC_NULL, 8, PETSC_NULL);
	MatSetFromOptions(coeff_mat);
	KSPCreate(PETSC_COMM_WORLD, &ksp);
	KSPSetType(ksp, KSPGMRES);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCSOR);
	KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, 10000);
	KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
	KSPSetFromOptions(ksp);
}

/**
 * @brief Initialises the physical properties of the domain.
 * * Sets the permeability and viscosity for each voxel based on its material type
 * (e.g., Pore, Rock).
 */
void Simulation::initProperties()
{
	const double pore_permeability = 1.0e-9;
	const double rock_permeability = 1.0e-18;
	const double fluid_viscosity = 1.0e-3;
	// const double air_viscosity = 1.0e-5;

	// TODO: Set material properties parameters in a settings file or command-line
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				RAWType voxel_type = img_data[idx];
				switch (voxel_type)
				{
				case Pore:
					permeability[idx] = pore_permeability;
					viscosity[idx] = fluid_viscosity;
					break;
				case Rock:
					permeability[idx] = rock_permeability;
					viscosity[idx] = fluid_viscosity;
					break;
				case Sulphide:
					permeability[idx] = rock_permeability;
					viscosity[idx] = fluid_viscosity;
					break;
				case Air:
				default:
					permeability[idx] = 0.0;
					viscosity[idx] = 0.0;
					break;
				}
			}
}

/**
 * @brief Initialises the fraction of reactive material in each voxel.
 * * Sets the 'frac' field to 1.0 for sulphide voxels and 0.0 for all others.
 */
void Simulation::initFrac()
{
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				frac[idx] = (img_data[idx] >= Sulphide) ? 1.0 : 0.0;
			}
}

/**
 * @brief Updates the fraction of reactive material based on the local concentration.
 * * Simulates the leaching process over a single time step. When a voxel's fraction
 * reaches zero, its material type is changed to Pore.
 * @param dt The current time step size.
 * @return The number of voxels that were fully leached in this step.
 */
int Simulation::updateFrac(double dt)
{
	int leached_locally = 0;
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				if (img_data[idx] >= Sulphide)
				{
					frac[idx] -= kreac * conc[idx] * dt;
					if (frac[idx] < 0.0)
					{
						frac[idx] = 0.0;
						img_data[idx] = (RAWType)Pore;
						leached_locally++;
					}
				}
			}
	return leached_locally;
}

/**
 * @brief Manages the exchange of halo/padding data between MPI processes.
 * * This blocking call ensures that the ghost cells for fieldsare updated before
 * further calculations.
 */
void Simulation::doExchange()
{
	img_data.exchangePadding(MPI_RAW_TYPE);
	conc.exchangePadding(MPI_DOUBLE);
	frac.exchangePadding(MPI_FLOAT);
	pressure.exchangePadding(MPI_DOUBLE);
	permeability.exchangePadding(MPI_DOUBLE);
	viscosity.exchangePadding(MPI_DOUBLE);
	flux_x.exchangePadding(MPI_DOUBLE);
	flux_y.exchangePadding(MPI_DOUBLE);
	flux_z.exchangePadding(MPI_DOUBLE);
	saturation.exchangePadding(MPI_DOUBLE);
}

/**
 * @brief Builds the PETSc linear system (Ax=b) for the steady-state pressure field.
 *
 * This function discretizes the continuity equation combined with Darcy's Law,
 * This results in a Poisson-like equation for pressure:
 * $$ \nabla \cdot (-\frac{K}{\mu} \cdot \nabla P) = 0 $$
 *
 * Since this formulation is for steady-state, changes in pressure at different
 * time steps depend on changes on permeability, related to changes in concentration
 * or in material type (change of Sulphide to Pore in leached voxels)
 *
 * A 7-point finite difference stencil is used to construct the matrix A, where the
 * coefficients are based on the hydraulic transmissivity (permeability divided by
 * viscosity) between adjacent voxels.
 *
 * Boundary conditions are set based on the material type:
 * - Air: Fixed-pressure (Dirichlet) boundary.
 * - Rock/Sulphide: No-flow boundary.
 */
void Simulation::setupPressureEqns()
{
	MatZeroEntries(coeff_mat);
	VecZeroEntries(sources_vec);
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				int arridx = idx.arrayId(global);
				RAWType voxel_type = img_data[idx];
				// Set values the linear system (Ax = b) as 1*P=0 for air, rock and sulphide voxels
				if (voxel_type == Air)
				{
					MatSetValue(coeff_mat, arridx, arridx, 1.0, INSERT_VALUES);
					VecSetValue(sources_vec, arridx, 0.0, INSERT_VALUES);
				}
				else if (voxel_type == Rock || voxel_type >= Sulphide)
				{
					MatSetValue(coeff_mat, arridx, arridx, 1.0, INSERT_VALUES);
					VecSetValue(sources_vec, arridx, 0.0, INSERT_VALUES);
				}
				// For the pore, calculate the coefficients for the matrix
				else if (voxel_type == Pore)
				{
					double diagonal_term = 0.0;
					// Lambda function for setting pressure coefficients in the matrix
					auto set_neighbor_val = [&](Index neighbor_idx)
					{
						if (neighbor_idx.valid(global))
						{
							double K_avg = (permeability[idx] + permeability[neighbor_idx]) / 2.0;
							double coeff = K_avg / viscosity[idx];
							MatSetValue(coeff_mat, arridx, neighbor_idx.arrayId(global), coeff, INSERT_VALUES);
							diagonal_term -= coeff;
						}
					};
					// Evaluate pressure on each voxel face
					set_neighbor_val(Index(i + 1, j, k));
					set_neighbor_val(Index(i - 1, j, k));
					set_neighbor_val(Index(i, j + 1, k));
					set_neighbor_val(Index(i, j - 1, k));
					set_neighbor_val(Index(i, j, k + 1));
					set_neighbor_val(Index(i, j, k - 1));
					MatSetValue(coeff_mat, arridx, arridx, diagonal_term, INSERT_VALUES);
					VecSetValue(sources_vec, arridx, 0.0, INSERT_VALUES);
				}
			}
	MatAssemblyBegin(coeff_mat, MAT_FINAL_ASSEMBLY);
	VecAssemblyBegin(sources_vec);
	MatAssemblyEnd(coeff_mat, MAT_FINAL_ASSEMBLY);
	VecAssemblyEnd(sources_vec);
}

/**
 * @brief Calls the PETSc solver to solve the linear system for pressure.
 * * The solution is written directly into the 'pressure' data field.
 */
void Simulation::solvePressure()
{
	PetscInt its;
	VecPlaceArray(conc_vec, pressure.getData().get() + pressure.pad_size);
	KSPSetOperators(ksp, coeff_mat, coeff_mat, DIFFERENT_NONZERO_PATTERN);
	KSPSolve(ksp, sources_vec, conc_vec);
	VecResetArray(conc_vec);
	KSPGetIterationNumber(ksp, &its);
	if (mpi_rank == 0)
		cout << "Pressure solve converged in " << its << " iterations." << endl;
}

/**
 * @brief Calculates the fluid flux vector based on the pressure gradient.
 *
 * This function is a direct numerical implementation of Darcy's Law:
 * $$ \mathbf{q} = -\frac{K}{\mu} \nabla P $$
 * Using a finite difference scheme to approximate the pressure gradient
 * Central differences are used for interior voxels and forward/backward
 * differences are used at the boundaries of the local MPI domain.
 */
void Simulation::calculateFlux()
{
	const double dx = this->dx;
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				double gradP_x, gradP_y, gradP_z;
				if (i == local.origin.i)
					gradP_x = (pressure[Index(i + 1, j, k)] - pressure[idx]) / dx;
				else if (i == local.origin.i + local.extent.i - 1)
					gradP_x = (pressure[idx] - pressure[Index(i - 1, j, k)]) / dx;
				else
					gradP_x = (pressure[Index(i + 1, j, k)] - pressure[Index(i - 1, j, k)]) / (2.0 * dx);
				if (j == local.origin.j)
					gradP_y = (pressure[Index(i, j + 1, k)] - pressure[idx]) / dx;
				else if (j == local.origin.j + local.extent.j - 1)
					gradP_y = (pressure[idx] - pressure[Index(i, j - 1, k)]) / dx;
				else
					gradP_y = (pressure[Index(i, j + 1, k)] - pressure[Index(i, j - 1, k)]) / (2.0 * dx);
				if (k == local.origin.k)
					gradP_z = (pressure[Index(i, j, k + 1)] - pressure[idx]) / dx;
				else if (k == local.origin.k + local.extent.k - 1)
					gradP_z = (pressure[idx] - pressure[Index(i, j, k - 1)]) / dx;
				else
					gradP_z = (pressure[Index(i, j, k + 1)] - pressure[Index(i, j, k - 1)]) / (2.0 * dx);
				double transmissivity = permeability[idx] / viscosity[idx];
				flux_x[idx] = -transmissivity * gradP_x;
				flux_y[idx] = -transmissivity * gradP_y;
				flux_z[idx] = -transmissivity * gradP_z;
			}
}

/**
 * @brief Builds the PETSc linear system for the time-dependent advection-diffusion-reaction equation.
 *
 * This function discretizes the governing equation for chemical transport:
 * $$ \frac{\partial c}{\partial t} + \nabla \cdot (\mathbf{q}c) = \nabla \cdot (D_{eff} \nabla c) + R $$
 * It uses an implicit Euler method for the time derivative and a first-order
 * upwind differencing scheme for the advection term (\nabla \cdot (\mathbf{q}c).
 * The fluid flux vector, $$ \mathbf{q} $$, is provided by the preceding pressure solve.
 *
 * @param dt The current time step size (s), used for the time-derivative term.
 */
void Simulation::setupConcentrationEqns(double dt)
{
	MatDestroy(&coeff_mat);
	MatCreate(PETSC_COMM_WORLD, &coeff_mat);
	MatSetSizes(coeff_mat, local.extent.size(), local.extent.size(), PETSC_DETERMINE, PETSC_DETERMINE);
	MatSetType(coeff_mat, MATMPIAIJ);
	MatMPIAIJSetPreallocation(coeff_mat, 8, PETSC_NULL, 8, PETSC_NULL);
	MatSetFromOptions(coeff_mat);
	VecZeroEntries(sources_vec);
	const double inv_dt = 1.0 / dt;
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				int arridx = idx.arrayId(global);
				RAWType voxel_type = img_data[idx];
				// Set values the linear system (Ax = b) for air, sulphide and rock voxels
				if (voxel_type == Air)
				{
					MatSetValue(coeff_mat, arridx, arridx, 1.0, INSERT_VALUES);
					VecSetValue(sources_vec, arridx, 0.0, INSERT_VALUES);
					continue;
				}
				else if (voxel_type >= Sulphide)
				{
					// Set the Sulphide source value (1.0)
					MatSetValue(coeff_mat, arridx, arridx, 1.0, INSERT_VALUES);
					VecSetValue(sources_vec, arridx, 1.0, INSERT_VALUES);
					continue;
				}
				else if (voxel_type == Rock)
				{
					MatSetValue(coeff_mat, arridx, arridx, 1.0, INSERT_VALUES);
					VecSetValue(sources_vec, arridx, conc[idx], INSERT_VALUES);
					continue;
				}
				MatSetValue(coeff_mat, arridx, arridx, inv_dt, ADD_VALUES);
				VecSetValue(sources_vec, arridx, conc[idx] * inv_dt, ADD_VALUES);
				// Lambda function for setting advection and diffussion rates
				auto set_link = [&](Index neighbor_idx, double q_face)
				{
					double D_eff = (img_data[neighbor_idx] == Pore) ? this->D * this->Dpore_fac : this->D;
					double diffusion_coeff = -D_eff / (dx * dx);
					MatSetValue(coeff_mat, arridx, neighbor_idx.arrayId(global), diffusion_coeff, ADD_VALUES);
					MatSetValue(coeff_mat, arridx, arridx, -diffusion_coeff, ADD_VALUES);
					double adv_coeff = q_face / dx;
					// upwind advection
					if (adv_coeff > 0)
					{
						MatSetValue(coeff_mat, arridx, arridx, adv_coeff, ADD_VALUES);
					}
					else
					{
						MatSetValue(coeff_mat, arridx, neighbor_idx.arrayId(global), -adv_coeff, ADD_VALUES);
					}
				};
				// Evaluate advection-diffusion on each voxel face
				Index neighbor_idx(0, 0, 0);
				neighbor_idx = Index(i + 1, j, k);
				if (neighbor_idx.valid(global))
					set_link(neighbor_idx, (flux_x[idx] + flux_x[neighbor_idx]) / 2.0);
				neighbor_idx = Index(i - 1, j, k);
				if (neighbor_idx.valid(global))
					set_link(neighbor_idx, -(flux_x[idx] + flux_x[neighbor_idx]) / 2.0);
				neighbor_idx = Index(i, j + 1, k);
				if (neighbor_idx.valid(global))
					set_link(neighbor_idx, (flux_y[idx] + flux_y[neighbor_idx]) / 2.0);
				neighbor_idx = Index(i, j - 1, k);
				if (neighbor_idx.valid(global))
					set_link(neighbor_idx, -(flux_y[idx] + flux_y[neighbor_idx]) / 2.0);
				neighbor_idx = Index(i, j, k + 1);
				if (neighbor_idx.valid(global))
					set_link(neighbor_idx, (flux_z[idx] + flux_z[neighbor_idx]) / 2.0);
				neighbor_idx = Index(i, j, k - 1);
				if (neighbor_idx.valid(global))
					set_link(neighbor_idx, -(flux_z[idx] + flux_z[neighbor_idx]) / 2.0);
			}
	MatAssemblyBegin(coeff_mat, MAT_FINAL_ASSEMBLY);
	VecAssemblyBegin(sources_vec);
	MatAssemblyEnd(coeff_mat, MAT_FINAL_ASSEMBLY);
	VecAssemblyEnd(sources_vec);
}

/**
 * @brief Calls the PETSc solver to solve the linear system for concentration.
 * * The solution is written directly into the 'conc' data field.
 */
void Simulation::solveConc()
{
	PetscInt its;
	VecPlaceArray(conc_vec, conc.getData().get() + conc.pad_size);
	KSPReset(ksp);
	KSPSetOperators(ksp, coeff_mat, coeff_mat, DIFFERENT_NONZERO_PATTERN);
	KSPSolve(ksp, sources_vec, conc_vec);
	VecResetArray(conc_vec);
	KSPGetIterationNumber(ksp, &its);
	MPIOUT(mpi_rank) << "Concentration solve converged in " << its << " iterations." << endl;
}

/**
 * @brief Writes the data to a set of VTK files.
 * @details The root process writes a master .pvti file that references individual .vti
 * part files written by each process.
 * @param fname_root The base path and filename for the output files.
 * @param tstep The current time step number.
 * @param data_flags A bitmask specifying the data fields to include in the output.
 */

void Simulation::writeVTKFile(std::string fname_root, size_t tstep, size_t data_flags)
{
	// Write the master .pvti file with the OVERLAPPING extent logic
	if (mpi_rank == 0)
	{
		stringstream fname;
		fname << fname_root << "_" << setw(6) << setfill('0') << tstep << ".pvti";
		ofstream fout(fname.str().c_str());

		fout << "<?xml version=\"1.0\"?>" << endl;
		fout << "<VTKFile type=\"PImageData\" version=\"0.1\">" << endl;
		fout << "\t<PImageData WholeExtent=\"0 " << global.extent.i - 1 << " 0 " << global.extent.j - 1 << " 0 " << global.extent.k - 1 << "\" ";
		fout << "GhostLevel=\"0\" Origin=\"0 0 0\" Spacing=\"1 1 1\">" << endl;
		fout << "\t\t<PPointData>" << endl;

#if DATA_TYPE == 16
		if (data_flags & VTKOutput::Type)
			fout << "\t\t\t<PDataArray type=\"UInt16\" Name=\"MaterialType\"/>" << endl;
#elif DATA_TYPE == 8
		if (data_flags & VTKOutput::Type)
			fout << "\t\t\t<PDataArray type=\"UInt8\" Name=\"MaterialType\"/>" << endl;
#endif
		if (data_flags & VTKOutput::Conc)
			fout << "\t\t\t<PDataArray type=\"Float64\" Name=\"Concentration\"/>" << endl;
		if (data_flags & VTKOutput::Pressure)
			fout << "\t\t\t<PDataArray type=\"Float64\" Name=\"Pressure\"/>" << endl;
		if (data_flags & VTKOutput::Frac)
			fout << "\t\t\t<PDataArray type=\"Float32\" Name=\"SulphideFrac\"/>" << endl;
		if (data_flags & VTKOutput::Flux_Vec)
			fout << "\t\t\t<PDataArray type=\"Float64\" Name=\"Flux\" NumberOfComponents=\"3\"/>" << endl;
		fout << "\t\t</PPointData>" << endl;

		for (int proc = 0; proc < mpi_comm_size; ++proc)
		{
			stringstream piece_fname;
			string root_basename = fname_root.substr(fname_root.find_last_of("/\\") + 1);
			piece_fname << root_basename << "_" << proc << "_" << setw(6) << setfill('0') << tstep << ".vti";

			const Domain &piece_dom = MPISubIndex<IDX_SCHEME>::all_local_domains[proc];
			int max_i = piece_dom.origin.i + piece_dom.extent.i - 1;

			// Add the one-voxel overlap for all pieces except the very last one
			if (proc != mpi_comm_size - 1)
			{
				max_i++;
			}

			fout << "\t\t<Piece Extent=\"";
			fout << piece_dom.origin.i << " " << max_i << " ";
			fout << piece_dom.origin.j << " " << piece_dom.origin.j + piece_dom.extent.j - 1 << " ";
			fout << piece_dom.origin.k << " " << piece_dom.origin.k + piece_dom.extent.k - 1 << "\" ";
			fout << "Source=\"" << piece_fname.str() << "\"/>" << endl;
		}

		fout << "\t</PImageData>" << endl;
		fout << "</VTKFile>" << endl;
		fout.close();
		cout << "Wrote master VTK file: " << fname.str() << endl;
	}

	// Each process writes its own .vti piece file, including data for the overlap
	stringstream fname;
	fname << fname_root << "_" << mpi_rank << "_" << setw(6) << setfill('0') << tstep << ".vti";

	int off_pos = (mpi_rank == mpi_comm_size - 1) ? 0 : 1;

	vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
	imageData->SetExtent(local.origin.i, local.origin.i + local.extent.i - 1 + off_pos,
						 local.origin.j, local.origin.j + local.extent.j - 1,
						 local.origin.k, local.origin.k + local.extent.k - 1);

	size_t num_voxels_to_write = (size_t)(local.extent.i + off_pos) * local.extent.j * local.extent.k;

	// This method (maybe less efficient) ensures consistency.
	// We create all arrays, then populate them in a single loop.

	// if (data_flags & VTKOutput::Type)
	//{
	//	auto arr = vtkSmartPointer<vtkDoubleArray>::New();
	//	arr->SetName("MaterialType");
	//	arr->SetNumberOfValues(num_voxels_to_write);
	//	imageData->GetPointData()->AddArray(arr);
	// }

	if (data_flags & VTKOutput::Type)
	{
		// This block now correctly compiles the right version based on the Makefile flag (-DDATA_TYPE)
		// and writes the correct VTK output
#if DATA_TYPE == 16
		auto arr = vtkSmartPointer<vtkUnsignedShortArray>::New();
		arr->SetName("MaterialType");
		arr->SetNumberOfValues(num_voxels_to_write);
		imageData->GetPointData()->AddArray(arr);
#elif DATA_TYPE == 8
		auto arr = vtkSmartPointer<vtkUnsignedCharArray>::New();
		arr->SetName("MaterialType");
		arr->SetNumberOfValues(num_voxels_to_write);
		imageData->GetPointData()->AddArray(arr);
#endif
	}
	if (data_flags & VTKOutput::Conc)
	{
		auto arr = vtkSmartPointer<vtkDoubleArray>::New();
		arr->SetName("Concentration");
		arr->SetNumberOfValues(num_voxels_to_write);
		imageData->GetPointData()->AddArray(arr);
	}
	if (data_flags & VTKOutput::Pressure)
	{
		auto arr = vtkSmartPointer<vtkDoubleArray>::New();
		arr->SetName("Pressure");
		arr->SetNumberOfValues(num_voxels_to_write);
		imageData->GetPointData()->AddArray(arr);
	}
	if (data_flags & VTKOutput::Frac)
	{
		auto arr = vtkSmartPointer<vtkFloatArray>::New();
		arr->SetName("SulphideFrac");
		arr->SetNumberOfValues(num_voxels_to_write);
		imageData->GetPointData()->AddArray(arr);
	}
	if (data_flags & VTKOutput::Flux_Vec)
	{
		auto arr = vtkSmartPointer<vtkDoubleArray>::New();
		arr->SetName("Flux");
		arr->SetNumberOfComponents(3);
		arr->SetNumberOfTuples(num_voxels_to_write);
		imageData->GetPointData()->SetVectors(arr);
	}

	size_t count = 0;
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
	{
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
		{
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i + off_pos); ++i)
			{
				Index idx(i, j, k);
				if (data_flags & VTKOutput::Type)
					imageData->GetPointData()->GetArray("MaterialType")->SetTuple1(count, img_data[idx]);
				if (data_flags & VTKOutput::Conc)
					imageData->GetPointData()->GetArray("Concentration")->SetTuple1(count, conc[idx]);
				if (data_flags & VTKOutput::Pressure)
					imageData->GetPointData()->GetArray("Pressure")->SetTuple1(count, pressure[idx]);
				if (data_flags & VTKOutput::Frac)
					imageData->GetPointData()->GetArray("SulphideFrac")->SetTuple1(count, frac[idx]);
				if (data_flags & VTKOutput::Flux_Vec)
					imageData->GetPointData()->GetVectors("Flux")->SetTuple3(count, flux_x[idx], flux_y[idx], flux_z[idx]);
				count++;
			}
		}
	}

	vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
	writer->SetFileName(fname.str().c_str());
	// writer->SetInputData(imageData); // SetInputData is simpler here but requires a different version of VTK
	writer->SetInputConnection(imageData->GetProducerPort());
	writer->Write();
}
