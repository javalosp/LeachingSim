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
	rel_permeability.setup(local.origin, local.extent);
	cap_pressure.setup(local.origin, local.extent);
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
void Simulation::setupPETSc(bool zero_conc)
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
				/*
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
					*/

				Index idx(i, j, k);
				RAWType voxel_type = img_data[idx];

				double intrinsic_K;
				switch (voxel_type)
				{
				case Pore:
					intrinsic_K = pore_permeability;
					break;
				case Rock:
				case Sulphide:
					intrinsic_K = rock_permeability;
					break;
				case Air:
				default:
					intrinsic_K = 0.0;
					break;
				}

				// The 'permeability' field now stores the effective permeability
				permeability[idx] = intrinsic_K * rel_permeability[idx];
				viscosity[idx] = fluid_viscosity;
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
 * @brief Updates the fraction of reactive material using a Resistance-in-Series model.
 */
int Simulation::updateFrac(double dt)
{
	int leached_locally = 0;

	// Assume Saturation Concentration (solubility limit) is 1.0 (normalized)
	// Or pass this as a parameter if you have a specific value in mol/m3
	// const double C_sat = 1.0;

	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
	{
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
		{
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);

				// Only process reactive solids
				if (img_data[idx] >= Sulphide)
				{
					double total_mass_loss = 0.0;

					// Lambda to calculate chemical flux to one liquid neighbor
					auto calc_surface_flux = [&](Index neighbor_idx)
					{
						if (neighbor_idx.valid(global))
						{
							RAWType neighbor_type = img_data[neighbor_idx];

							// Reaction only happens at interface with Pore
							if (neighbor_type == Pore)
							{
								// 1. Calculate Transport Coefficient (k_trans ~ D/dx)
								// We can also add advection velocity if flow is strong
								double k_trans = D / dx;

								// 2. Calculate Reaction Coefficient (k_reac)
								double k_rxn = kreac;

								// 3. Calculate Effective Coefficient (Series Resistance)
								// This is the logic from "Other Code" adapted to yours
								double k_eff = 1.0 / ((1.0 / k_trans) + (1.0 / k_rxn));

								// 4. Calculate Driving Force (c_sat - C_pore)
								double driving_force = c_sat - conc[neighbor_idx];

								// Ensure we don't leach backwards if pore is supersaturated
								if (driving_force < 0)
									driving_force = 0;

								// 5. Return Flux (mol/m^2/s)
								return k_eff * driving_force;
							}
						}
						return 0.0;
					};

					// Sum flux for all 6 faces
					double flux_sum = 0.0;
					flux_sum += calc_surface_flux(Index(i + 1, j, k));
					flux_sum += calc_surface_flux(Index(i - 1, j, k));
					flux_sum += calc_surface_flux(Index(i, j + 1, k));
					flux_sum += calc_surface_flux(Index(i, j - 1, k));
					flux_sum += calc_surface_flux(Index(i, j, k + 1));
					flux_sum += calc_surface_flux(Index(i, j, k - 1));

					// Update fraction
					// Mass Loss = Flux * Area * dt
					// (Assuming frac is normalized mass)
					double area_per_face = dx * dx;
					total_mass_loss = flux_sum * area_per_face * dt;

					// Apply loss.
					// Note: You might need to divide by Molar Density if frac isn't moles.
					frac[idx] -= total_mass_loss;

					if (frac[idx] <= 0.0)
					{
						frac[idx] = 0.0;
						img_data[idx] = (RAWType)Pore;

						// IMPORTANT: Initialise the new pore concentration
						// usually to c_sat because it just dissolved
						conc[idx] = c_sat;

						leached_locally++;
					}
				}
			}
		}
	}
	return leached_locally;
}

/**
 * @brief Updates the fraction of reactive material based on flux from the grain surface.
 *
 * This function models the leaching process. It calculates the total flux of the
 * dissolved species leaving each sulphide voxel and uses this flux to decrease the
 * remaining fraction 'frac'. When a voxel's fraction reaches zero, its material
 * type is changed from Sulphide to Pore.
 *
 * @param dt The current time step size.
 * @return The number of voxels that were fully leached in this step.
 */
/*
int Simulation::updateFrac(double dt)
{
	int leached_locally = 0;

	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
	{
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
		{
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);

				if (img_data[idx] >= Sulphide)
				{
					double total_flux_out = 0.0;

					// Lambda to calculate flux to one neighbor
					auto get_flux_to_neighbor = [&](Index neighbor_idx)
					{
						if (neighbor_idx.valid(global))
						{
							RAWType neighbor_type = img_data[neighbor_idx];
							if (neighbor_type == Pore || neighbor_type == Rock)
							{
								// Diffusive flux is proportional to concentration gradient
								double diffusive_flux = D * (conc[idx] - conc[neighbor_idx]) / dx;

								// Advective flux is q * c (using upwinding)
								// We need the flux at the face, which we approximate by averaging
								double q_face_x = (flux_x[idx] + flux_x[neighbor_idx]) / 2.0;
								double advective_flux = std::max(0.0, q_face_x) * conc[idx] + std::min(0.0, q_face_x) * conc[neighbor_idx];

								return diffusive_flux + advective_flux;
							}
						}
						return 0.0;
					};

					// Sum the flux out of all 6 faces of the voxel
					total_flux_out += get_flux_to_neighbor(Index(i + 1, j, k));
					total_flux_out += get_flux_to_neighbor(Index(i - 1, j, k));
					total_flux_out += get_flux_to_neighbor(Index(i, j + 1, k));
					total_flux_out += get_flux_to_neighbor(Index(i, j - 1, k));
					total_flux_out += get_flux_to_neighbor(Index(i, j, k + 1));
					total_flux_out += get_flux_to_neighbor(Index(i, j, k - 1));

					// Update frac based on the total flux leaving the grain voxel
					// (Flux has units of mol/m^2/s, multiply by area and dt, divide by molar volume)
					// We use 'kreac' as a scaling factor for this complex term.
					frac[idx] -= (kreac * total_flux_out * dx * dx) * dt;

					if (frac[idx] < 0.0)
					{
						frac[idx] = 0.0;
						img_data[idx] = (RAWType)Pore;
						leached_locally++;
					}
				}
			}
		}
	}
	return leached_locally;
}
*/

/**
 * @brief Updates the fraction of reactive material based on the local concentration.
 * * Simulates the leaching process over a single time step. When a voxel's fraction
 * reaches zero, its material type is changed to Pore.
 * @param dt The current time step size.
 * @return The number of voxels that were fully leached in this step.

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
	 */

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
					double source_term = 0.0; // Source term for this voxel

					// Lambda function for setting pressure coefficients in the matrix considering connections to neighbours
					auto set_neighbor_link = [&](Index neighbor_idx, int dimension)
					{
						if (neighbor_idx.valid(global))
						{
							RAWType neighbor_type = img_data[neighbor_idx];
							if (neighbor_type == Air)
							{
								// Evaporation (Neumann) Boundary Condition
								// This is a surface voxel. Add the evaporative flux to the source term.
								// Flux is outward, so it's a sink (negative source).
								source_term -= evaporative_flux * dx; // flux * area (dx*dx) / (dx)

								// Also need a diagonal contribution for the open face
								double K_self = permeability[idx];
								double coeff = K_self / viscosity[idx]; // Note: uses K of the pore, not an average
								diagonal_term -= coeff;
							}
							else
							{
								// This is an internal face (Pore-Pore or Pore-Rock).
								// Standard connection based on average permeability.
								double K_avg = (permeability[idx] + permeability[neighbor_idx]) / 2.0;
								double coeff = K_avg / viscosity[idx];
								MatSetValue(coeff_mat, arridx, neighbor_idx.arrayId(global), coeff, INSERT_VALUES);
								diagonal_term -= coeff;
							}
						}
					};

					set_neighbor_link(Index(i + 1, j, k), 0);
					set_neighbor_link(Index(i - 1, j, k), 0);
					set_neighbor_link(Index(i, j + 1, k), 1);
					set_neighbor_link(Index(i, j - 1, k), 1);
					set_neighbor_link(Index(i, j, k + 1), 2);
					set_neighbor_link(Index(i, j, k - 1), 2);

					MatSetValue(coeff_mat, arridx, arridx, diagonal_term, INSERT_VALUES);
					VecSetValue(sources_vec, arridx, source_term, INSERT_VALUES);
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
		if (data_flags & VTKOutput::Saturation)
			fout << "\t\t\t<PDataArray type=\"Float64\" Name=\"Saturation\"/>" << endl;
		if (data_flags & VTKOutput::CapPressure)
			fout << "\t\t\t<PDataArray type=\"Float64\" Name=\"CapillaryPressure\"/>" << endl;
		if (data_flags & VTKOutput::RelPermeability)
			fout << "\t\t\t<PDataArray type=\"Float64\" Name=\"RelativePermeability\"/>" << endl;
		if (data_flags & VTKOutput::Permeability)
			fout << "\t\t\t<PDataArray type=\"Float64\" Name=\"EffectivePermeability\"/>" << endl;

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

	if (data_flags & VTKOutput::Saturation)
	{
		auto arr = vtkSmartPointer<vtkDoubleArray>::New();
		arr->SetName("Saturation");
		arr->SetNumberOfValues(num_voxels_to_write);
		// arr->SetArray(saturation.getData().get() + saturation.pad_size, local.extent.size(), 1);
		imageData->GetPointData()->AddArray(arr);
	}
	if (data_flags & VTKOutput::CapPressure)
	{
		auto arr = vtkSmartPointer<vtkDoubleArray>::New();
		arr->SetName("CapillaryPressure");
		arr->SetNumberOfValues(num_voxels_to_write);
		// arr->SetArray(cap_pressure.getData().get() + cap_pressure.pad_size, local.extent.size(), 1);
		imageData->GetPointData()->AddArray(arr);
	}
	if (data_flags & VTKOutput::RelPermeability)
	{
		auto arr = vtkSmartPointer<vtkDoubleArray>::New();
		arr->SetName("RelativePermeability");
		arr->SetNumberOfValues(num_voxels_to_write);
		// arr->SetArray(rel_permeability.getData().get() + rel_permeability.pad_size, local.extent.size(), 1);
		imageData->GetPointData()->AddArray(arr);
	}
	if (data_flags & VTKOutput::Permeability)
	{
		auto arr = vtkSmartPointer<vtkDoubleArray>::New();
		arr->SetName("EffectivePermeability");
		arr->SetNumberOfValues(num_voxels_to_write);
		// arr->SetArray(permeability.getData().get() + permeability.pad_size, local.extent.size(), 1);
		imageData->GetPointData()->AddArray(arr);
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
				if (data_flags & VTKOutput::Saturation)
					imageData->GetPointData()->GetArray("Saturation")->SetTuple1(count, saturation[idx]);
				if (data_flags & VTKOutput::CapPressure)
					imageData->GetPointData()->GetArray("CapillaryPressure")->SetTuple1(count, cap_pressure[idx]);
				if (data_flags & VTKOutput::RelPermeability)
					imageData->GetPointData()->GetArray("RelativePermeability")->SetTuple1(count, rel_permeability[idx]);
				if (data_flags & VTKOutput::Permeability)
					imageData->GetPointData()->GetArray("EffectivePermeability")->SetTuple1(count, permeability[idx]);
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

/**
 * @brief Initializes the fluid saturation field.
 * * Sets the saturation to 1.0 (fully saturated) in pore and sulphide voxels
 * and 0.0 (dry) in rock and air voxels.
 */
void Simulation::initSaturation()
{
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				RAWType voxel_type = img_data[idx];
				if (voxel_type == Pore || voxel_type >= Sulphide)
				{
					saturation[idx] = 1.0;
				}
				else
				{
					saturation[idx] = 0.0;
				}
			}
}

/**
 * @brief Updates the capillary pressure field based on the current saturation.
 * * Implements the van Genuchten capillary pressure model.
 */
void Simulation::updateCapillaryPressure()
{
	const double m = 1.0 - (1.0 / vg_n);

	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				if (img_data[idx] == Pore || img_data[idx] >= Sulphide)
				{
					// Calculate effective saturation, Se, ensuring it's within a valid range
					double Se = (saturation[idx] - s_res) / (1.0 - s_res);
					Se = std::max(0.0, std::min(1.0, Se));

					if (Se >= 1.0)
					{
						cap_pressure[idx] = 0.0;
					}
					else
					{
						double Se_inv_m = pow(Se, -1.0 / m);
						cap_pressure[idx] = (1.0 / vg_alpha) * pow(Se_inv_m - 1.0, 1.0 / vg_n);
					}
				}
				else
				{
					cap_pressure[idx] = 0.0;
				}
			}
}

/**
 * @brief Updates the relative permeability field based on the current saturation.
 * * Implements the Mualem-van Genuchten relative permeability model.
 */
void Simulation::updateRelativePermeability()
{
	const double m = 1.0 - (1.0 / vg_n);

	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				if (img_data[idx] == Pore || img_data[idx] >= Sulphide)
				{
					// Calculate effective saturation, Se
					double Se = (saturation[idx] - s_res) / (1.0 - s_res);
					Se = std::max(0.0, std::min(1.0, Se));

					if (Se <= 0.0)
					{
						rel_permeability[idx] = 0.0;
					}
					else
					{
						double term1 = 1.0 - pow(1.0 - pow(Se, 1.0 / m), m);
						rel_permeability[idx] = sqrt(Se) * term1 * term1;
					}
				}
				else
				{
					rel_permeability[idx] = 0.0;
				}
			}
}

/**
 * @brief Evolves the fluid saturation field over a single time step.
 * * Implements an explicit Euler time-step for the mass conservation equation:
 * $$ \phi * \frac{\partial s}{\partial t} + \nabla \cdot \mathbf{q} = 0. $$
 * * @param dt The current time step size.
 */
void Simulation::updateSaturation(double dt)
{
	// A temporary field to store the new saturation values before updating the main one.
	MPIDomain<double, 1, IDX_SCHEME> saturation_new;
	saturation_new.setup(local.origin, local.extent);

	const double porosity = 0.1; // This should be a configurable parameter
	const double dt_over_phi = dt / porosity;

	// Loop through the interior of the local domain
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
	{
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
		{
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);

				//// Approximate flux at the faces between voxels by averaging cell-centered values
				// double q_face_xp = (flux_x[idx] + flux_x[Index(i + 1, j, k)]) / 2.0;
				// double q_face_xm = (flux_x[idx] + flux_x[Index(i - 1, j, k)]) / 2.0;
				// double q_face_yp = (flux_y[idx] + flux_y[Index(i, j + 1, k)]) / 2.0;
				// double q_face_ym = (flux_y[idx] + flux_y[Index(i, j - 1, k)]) / 2.0;
				// double q_face_zp = (flux_z[idx] + flux_z[Index(i, j, k + 1)]) / 2.0;
				// double q_face_zm = (flux_z[idx] + flux_z[Index(i, j, k - 1)]) / 2.0;

				// Add boundary checks for all neighbor accesses ---
				// +X face
				Index neighbor_xp(i + 1, j, k);
				double q_face_xp = neighbor_xp.valid(global) ? (flux_x[idx] + flux_x[neighbor_xp]) / 2.0 : 0.0;
				// -X face
				Index neighbor_xm(i - 1, j, k);
				double q_face_xm = neighbor_xm.valid(global) ? (flux_x[idx] + flux_x[neighbor_xm]) / 2.0 : 0.0;

				// +Y face
				Index neighbor_yp(i, j + 1, k);
				double q_face_yp = neighbor_yp.valid(global) ? (flux_y[idx] + flux_y[neighbor_yp]) / 2.0 : 0.0;
				// -Y face
				Index neighbor_ym(i, j - 1, k);
				double q_face_ym = neighbor_ym.valid(global) ? (flux_y[idx] + flux_y[neighbor_ym]) / 2.0 : 0.0;

				// +Z face
				Index neighbor_zp(i, j, k + 1);
				double q_face_zp = neighbor_zp.valid(global) ? (flux_z[idx] + flux_z[neighbor_zp]) / 2.0 : 0.0;
				// -Z face
				Index neighbor_zm(i, j, k - 1);
				double q_face_zm = neighbor_zm.valid(global) ? (flux_z[idx] + flux_z[neighbor_zm]) / 2.0 : 0.0;

				// Calculate the divergence of the flux vector using a central difference
				double div_q = ((q_face_xp - q_face_xm) + (q_face_yp - q_face_ym) + (q_face_zp - q_face_zm)) / dx;

				// Update saturation using the explicit time-step formula: S_new = S_old - (dt/phi) * div(q)
				double new_sat = saturation[idx] - dt_over_phi * div_q;

				// Clamp the result to the physical bounds [0, 1]
				saturation_new[idx] = std::max(0.0, std::min(1.0, new_sat));
			}
		}
	}

	// Atomically swap the new data into the main saturation field
	saturation.take(saturation_new.getData());
}

/**
 * @brief Models precipitation at the surface due to evaporation.
 *
 * This function identifies pore voxels adjacent to air that contain fluid (S > 0).
 * At these surface locations, it simulates precipitation by setting the chemical
 * concentration to its maximum value and changing the material type from Pore to Rock,
 * effectively clogging the pore.
 */
void Simulation::handleSurfaceEffects()
{
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
	{
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
		{
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);

				// Proceed only if the voxel is a pore containing some fluid
				if (img_data[idx] == Pore && saturation[idx] > 0.0)
				{
					bool is_surface_voxel = false;

					// Check all 6 neighbors to see if any are Air
					Index neighbors[6] = {
						Index(i + 1, j, k), Index(i - 1, j, k),
						Index(i, j + 1, k), Index(i, j - 1, k),
						Index(i, j, k + 1), Index(i, j, k - 1)};

					for (int n = 0; n < 6; ++n)
					{
						if (neighbors[n].valid(global) && img_data[neighbors[n]] == Air)
						{
							is_surface_voxel = true;
							break; // Found an air neighbor, no need to check further
						}
					}

					// If it's a surface voxel, apply the precipitation effect
					if (is_surface_voxel)
					{
						conc[idx] = 1.0;			   // Set concentration to maximum (saturated)
						img_data[idx] = (RAWType)Rock; // Clog the pore with precipitate
					}
				}
			}
		}
	}
}
