#include "Simulation.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include <cassert>
#include <algorithm>

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
	VecDestroy(&solution_vec);
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
	conc_acid.setup(local.origin, local.extent);
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
	precipitate_inventory.setup(local.origin, local.extent);
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
				if (voxel == Sulphide && voxel > max_grain_id)
					max_grain_id = voxel;
			}
	int tmp;
	MPI_Allreduce(&max_grain_id, &tmp, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);
	max_grain_id = tmp;
	num_grains = max_grain_id - Sulphide + 1;
	ks.assign(num_grains, kreac);
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
	VecDuplicate(sources_vec, &solution_vec);
	MatCreate(PETSC_COMM_WORLD, &coeff_mat);
	MatSetSizes(coeff_mat, local.extent.size(), local.extent.size(), PETSC_DETERMINE, PETSC_DETERMINE);
	MatSetType(coeff_mat, MATMPIAIJ);
	// MatMPIAIJSetPreallocation(coeff_mat, 8, PETSC_NULL, 8, PETSC_NULL);
	MatMPIAIJSetPreallocation(coeff_mat, 8, PETSC_NULLPTR, 8, PETSC_NULLPTR);
	// Allow sulphide to pore transitions in the matrix structure
	MatSetOption(coeff_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
	MatSetFromOptions(coeff_mat);
	KSPCreate(PETSC_COMM_WORLD, &ksp);
	KSPSetType(ksp, KSPGMRES);
	KSPGetPC(ksp, &pc);
	// PCSetType(pc, PCSOR);
	PCSetType(pc, PCJACOBI);
	KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, 10000);
	KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
	KSPSetFromOptions(ksp);
}

/**
 * @brief Initialises all dynamic solution arrays to prevent anything in memory
 * from corrupting the PETSc initial guess.
 */
void Simulation::initFields()
{
	size_t total_size = pressure.padded.extent.size();
	for (size_t i = 0; i < total_size; ++i)
	{
		pressure.getData()[i] = 0.0;
		conc.getData()[i] = 0.0;
		flux_x.getData()[i] = 0.0;
		flux_y.getData()[i] = 0.0;
		flux_z.getData()[i] = 0.0;
		saturation.getData()[i] = 0.0;
		cap_pressure.getData()[i] = 0.0;
		rel_permeability.getData()[i] = 0.0;
		precipitate_inventory.getData()[i] = 0.0;
	}
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

	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
	{
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
		{
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				RAWType type = img_data[idx];

				double K_intrinsic = rock_permeability; // Default for solid Rock

				double log_k_rock = std::log10(rock_permeability);
				double log_k_pore = std::log10(pore_permeability);

				if (type == Air)
				{
					K_intrinsic = pore_permeability;
				}
				else if (type == Pore)
				{
					// Dynamic Clogging due to Precipitation
					double p_frac = precipitate_inventory[idx];

					// Clamp between 0.0 (clean) and 1.0 (fully clogged) for safety
					p_frac = std::max(0.0, std::min(1.0, p_frac));

					// As precipitate fills the pore (p_frac -> 1.0), K drops to rock level
					double log_k_eff = (p_frac * log_k_rock) + ((1.0 - p_frac) * log_k_pore);
					K_intrinsic = std::pow(10.0, log_k_eff);
				}
				else if (type == Sulphide)
				{
					double f = frac[idx];

					// As rock dissolves (f -> 0.0), K rises to pore level
					double log_k_eff = (f * log_k_rock) + ((1.0 - f) * log_k_pore);
					K_intrinsic = std::pow(10.0, log_k_eff);
				}

				// Set dynamic viscosity based on material
				if (type == Air)
				{
					viscosity[idx] = 1.8e-5;
				}
				else
				{
					viscosity[idx] = 1.0e-3;
				}

				// Apply multiphase relative permeability scaling
				double K_rel = std::max(rel_permeability[idx], 1e-15);
				permeability[idx] = K_intrinsic * K_rel;
			}
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
				frac[idx] = (img_data[idx] == Sulphide) ? 1.0 : 0.0;
			}
}

/**
 * @brief Initialises the fraction of reactant material in each voxel.
 * * Sets the 'frac' field to 1.0 for sulphide voxels and 0.0 for all others.
 */
void Simulation::initAcid()
{
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
	{
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
		{
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				RAWType voxel_type = img_data[idx];

				// Initially, pores and the top Air boundary are full of acid (1.0)
				if (voxel_type == Pore || (voxel_type == Air && i == 0))
				{
					conc_acid[idx] = 1.0;
				}
				else
				{
					conc_acid[idx] = 0.0;
				}
			}
		}
	}
}

/**
 * @brief Updates the fraction of reactive material using a Resistance-in-Series model.
 */
int Simulation::updateFrac(double dt)
{
	int leached_locally = 0;

	// 1. Calculate maximum fractional loss rate to determine safe sub-steps
	double max_rate = 0.0;
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
	{
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
		{
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				if (img_data[idx] == Sulphide)
				{
					double flux_sum = 0.0;

					auto calc_surface_flux = [&](Index neighbor_idx)
					{
						if (neighbor_idx.valid(global) && img_data[neighbor_idx] == Pore)
						{
							double k_trans = D / dx;
							double k_eff = 1.0 / ((1.0 / k_trans) + (1.0 / kreac));
							// double driving_force = std::max(0.0, c_sat - conc[neighbor_idx]);
							double driving_force = conc_acid[neighbor_idx];
							// For testing purposes set driving_force to 1
							// double driving_force = 1.0;
							return k_eff * driving_force;
						}
						return 0.0;
					};

					flux_sum += calc_surface_flux(Index(i + 1, j, k));
					flux_sum += calc_surface_flux(Index(i - 1, j, k));
					flux_sum += calc_surface_flux(Index(i, j + 1, k));
					flux_sum += calc_surface_flux(Index(i, j - 1, k));
					flux_sum += calc_surface_flux(Index(i, j, k + 1));
					flux_sum += calc_surface_flux(Index(i, j, k - 1));

					double area_per_face = dx * dx;
					double rate = flux_sum * area_per_face;
					if (rate > max_rate)
						max_rate = rate;
				}
			}
		}
	}

	double global_max_rate = 0.0;
	MPI_Allreduce(&max_rate, &global_max_rate, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);

	// Determine sub-steps. Limit fractional loss to a maximum of 10% (0.1) per sub-step.
	double dt_safe = dt;
	if (global_max_rate > 1e-20)
	{
		dt_safe = 0.1 / global_max_rate;
	}

	// int num_steps = std::ceil(dt / dt_safe);
	// Use a double for the ceiling calculation, then cap it, then cast it.
	double calculated_steps = std::ceil(dt / dt_safe);
	int num_steps = 1;
	if (calculated_steps > 1000)
	{
		num_steps = 1000; // Hard cap to prevent infinite loops
	}
	else if (calculated_steps > 1.0)
	{
		num_steps = (int)calculated_steps;
	}

	double dt_sub = dt / num_steps;

	// Print to the log to monitor the physical rates
	if (mpi_rank == 0)
	{
		std::cout << "    [Leaching] Max rate: " << global_max_rate
				  << " | Sub-steps: " << num_steps << " (" << dt_sub << "s)" << std::endl;
	}

	if (mpi_rank == 0 && num_steps > 1)
	{
		std::cout << "    [Leaching] Sub-stepping required: " << num_steps << " steps of " << dt_sub << "s." << std::endl;
	}

	// Perform Sub-stepped updates
	for (int step = 0; step < num_steps; ++step)
	{
		for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
		{
			for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
			{
				for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
				{
					Index idx(i, j, k);

					if (img_data[idx] >= Sulphide && frac[idx] > 0.0)
					{
						double flux_sum = 0.0;

						// Re-evaluate flux (in case neighbors changed to Pore in previous sub-steps)
						auto calc_surface_flux = [&](Index neighbor_idx)
						{
							if (neighbor_idx.valid(global) && img_data[neighbor_idx] == Pore)
							{
								double k_trans = D / dx;
								double k_eff = 1.0 / ((1.0 / k_trans) + (1.0 / kreac));
								// double driving_force = std::max(0.0, c_sat - conc[neighbor_idx]);
								double driving_force = conc_acid[neighbor_idx];
								return k_eff * driving_force;
							}
							return 0.0;
						};

						// Calculate fluxes for all 6 faces
						double f_xp = calc_surface_flux(Index(i + 1, j, k));
						double f_xm = calc_surface_flux(Index(i - 1, j, k));
						double f_yp = calc_surface_flux(Index(i, j + 1, k));
						double f_ym = calc_surface_flux(Index(i, j - 1, k));
						double f_zp = calc_surface_flux(Index(i, j, k + 1));
						double f_zm = calc_surface_flux(Index(i, j, k - 1));

						flux_sum = f_xp + f_xm + f_yp + f_ym + f_zp + f_zm;

						double area_per_face = dx * dx;
						double mass_loss = flux_sum * area_per_face * dt_sub;

						frac[idx] -= mass_loss;

						// Consume the acid from the adjacent pores
						// Distribute the consumption across the neighboring pores proportional to the flux from each face.
						if (flux_sum > 0.0)
						{
							if (Index(i + 1, j, k).valid(global))
								conc_acid[Index(i + 1, j, k)] -= mass_loss * (f_xp / flux_sum);
							if (Index(i - 1, j, k).valid(global))
								conc_acid[Index(i - 1, j, k)] -= mass_loss * (f_xm / flux_sum);
							if (Index(i, j + 1, k).valid(global))
								conc_acid[Index(i, j + 1, k)] -= mass_loss * (f_yp / flux_sum);
							if (Index(i, j - 1, k).valid(global))
								conc_acid[Index(i, j - 1, k)] -= mass_loss * (f_ym / flux_sum);
							if (Index(i, j, k + 1).valid(global))
								conc_acid[Index(i, j, k + 1)] -= mass_loss * (f_zp / flux_sum);
							if (Index(i, j, k - 1).valid(global))
								conc_acid[Index(i, j, k - 1)] -= mass_loss * (f_zm / flux_sum);
						}

						// Check if voxel has fully dissolved
						// using a leaching threshold (e.g. 1e-4)
						if (frac[idx] <= 1e-4)
						{
							frac[idx] = 0.0;
							if (img_data[idx] == Sulphide)
							{
								img_data[idx] = Pore;
								// Ensure the new pore starts clean
								precipitate_inventory[idx] = 0.0;
								conc[idx] = c_sat; // Instant saturation in the newly formed pore
								leached_locally++;
							}
						}
					}
				}
			}
		}

		// If a voxel turns into a Pore, its MPI neighbors need to know before the next sub-step
		if (num_steps > 1 && step < num_steps - 1)
		{
			img_data.exchangePadding(MPI_RAW_TYPE);
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
	conc_acid.exchangePadding(MPI_DOUBLE);
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

				// Set values the linear system (Ax = b) as 1*P=0 for air, rock, sulphide, and precipitate voxels
				if (voxel_type == Air || voxel_type == Rock || voxel_type == Sulphide || voxel_type == Precipitate)
				{
					// Force the diagonal to 1.0 to prevent PETSc divide by zero crashes
					MatSetValue(coeff_mat, arridx, arridx, 1.0, ADD_VALUES);

					// Set RHS accordingly
					if (voxel_type == Air)
					{
						// Apply the pressure head to ANY Air voxel in the top half of the domain.
						// This guarantees the high pressure actually touches the porous rock.
						if (i < global.extent.i / 2)
						{
							// VecSetValue(sources_vec, arridx, 10000.0, ADD_VALUES);
							VecSetValue(sources_vec, arridx, top_pressure, ADD_VALUES);
						}
						else
						{
							VecSetValue(sources_vec, arridx, 0.0, ADD_VALUES);
						}
						continue;
					}
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
							double K_neighbor;

							if (neighbor_type == Air)
							{
								// Evaporation (Neumann) Boundary Condition
								// This is a surface voxel. Add the evaporative flux to the source term.
								// Flux is outward, so it's a sink (negative source).
								source_term -= evaporative_flux * dx; // flux * area (dx*dx) / (dx)
								// Air offers no fluid resistance; permeability is governed by the pore itself
								// K_neighbor = permeability[idx];
								// FIX: Give Air the same base permeability as a pore,
								// rather than dynamically copying the current voxel.
								// This prevents numerical shocks when Rock turns to Pore.
								K_neighbor = 1.0e-9;
							}
							else
							{
								K_neighbor = permeability[neighbor_idx];
							}
							// Calculate average permeability and enforce the 1e-25 regularization
							// double K_avg = (permeability[idx] + K_neighbor) / 2.0;
							// K_avg = std::max(K_avg, 1e-25);

							// Calculate the harmonic mean for permeability
							double K_self = std::max(permeability[idx], 1e-25);
							double K_neigh = std::max(K_neighbor, 1e-25);

							// Harmonic mean: 2 * (K1 * K2) / (K1 + K2)
							double K_avg = (2.0 * K_self * K_neigh) / (K_self + K_neigh);

							double coeff = K_avg / viscosity[idx];

							// Link the neighbour in the matrix, even if it's an Air boundary
							MatSetValue(coeff_mat, arridx, neighbor_idx.arrayId(global), coeff, ADD_VALUES);
							diagonal_term -= coeff;
						}
					};

					set_neighbor_link(Index(i + 1, j, k), 0);
					set_neighbor_link(Index(i - 1, j, k), 0);
					set_neighbor_link(Index(i, j + 1, k), 1);
					set_neighbor_link(Index(i, j - 1, k), 1);
					set_neighbor_link(Index(i, j, k + 1), 2);
					set_neighbor_link(Index(i, j, k - 1), 2);

					// MatSetValue(coeff_mat, arridx, arridx, diagonal_term, INSERT_VALUES);
					// VecSetValue(sources_vec, arridx, source_term, INSERT_VALUES);
					MatSetValue(coeff_mat, arridx, arridx, diagonal_term, ADD_VALUES);
					VecSetValue(sources_vec, arridx, source_term, ADD_VALUES);
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
	KSPConvergedReason reason;
	VecPlaceArray(solution_vec, pressure.getData().get() + pressure.pad_size);
	// KSPSetOperators(ksp, coeff_mat, coeff_mat, DIFFERENT_NONZERO_PATTERN);
	KSPSetOperators(ksp, coeff_mat, coeff_mat);
	KSPSolve(ksp, sources_vec, solution_vec);
	PetscInt min_loc, max_loc;
	PetscReal min_val, max_val;
	VecMin(solution_vec, &min_loc, &min_val);
	VecMax(solution_vec, &max_loc, &max_val);
	VecResetArray(solution_vec);
	KSPGetIterationNumber(ksp, &its);
	KSPGetConvergedReason(ksp, &reason);

	if (mpi_rank == 0)
	{
		if (reason > 0)
		{
			cout << "    [Pressure] Converged in " << its << " iterations. (Reason Code: " << reason << ")" << endl;
			// Print the exact bounds of the pressure field
			cout << "    [Pressure] Min Value: " << min_val << " Pa | Max Value: " << max_val << " Pa" << endl;
		}
		else
		{
			cout << "    [Pressure] DIVERGED/FAILED in " << its << " iterations! (Error Code: " << reason << ")" << endl;
		}
	}
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
				double gradP_x, gradP_y, gradP_z;	 // global pressure gradients
				double gradPc_x, gradPc_y, gradPc_z; // capillary pressure gradients

				// Calculate global pressure gradients
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

				// Calculate capillary pressure gradients
				if (i == local.origin.i)
					gradPc_x = (cap_pressure[Index(i + 1, j, k)] - cap_pressure[idx]) / dx;
				else if (i == local.origin.i + local.extent.i - 1)
					gradPc_x = (cap_pressure[idx] - cap_pressure[Index(i - 1, j, k)]) / dx;
				else
					gradPc_x = (cap_pressure[Index(i + 1, j, k)] - cap_pressure[Index(i - 1, j, k)]) / (2.0 * dx);

				if (j == local.origin.j)
					gradPc_y = (cap_pressure[Index(i, j + 1, k)] - cap_pressure[idx]) / dx;
				else if (j == local.origin.j + local.extent.j - 1)
					gradPc_y = (cap_pressure[idx] - cap_pressure[Index(i, j - 1, k)]) / dx;
				else
					gradPc_y = (cap_pressure[Index(i, j + 1, k)] - cap_pressure[Index(i, j - 1, k)]) / (2.0 * dx);

				if (k == local.origin.k)
					gradPc_z = (cap_pressure[Index(i, j, k + 1)] - cap_pressure[idx]) / dx;
				else if (k == local.origin.k + local.extent.k - 1)
					gradPc_z = (cap_pressure[idx] - cap_pressure[Index(i, j, k - 1)]) / dx;
				else
					gradPc_z = (cap_pressure[Index(i, j, k + 1)] - cap_pressure[Index(i, j, k - 1)]) / (2.0 * dx);

				// capillary regularization (gradient clamping)

				// Prevent extreme microscopic suction spikes from crashing the explicit solver.
				// use a very strong (physically stable) suction limit, e.g. 50000 Pa/m
				// this value is defined through command-line --max_cap_grad 50000.0
				// lower values: fast but less accurate drying (e.g. 10000)
				// higher values: slowt but accurate drying (e.g. 200000)

				gradPc_x = std::max(-max_cap_grad, std::min(max_cap_grad, gradPc_x));
				gradPc_y = std::max(-max_cap_grad, std::min(max_cap_grad, gradPc_y));
				gradPc_z = std::max(-max_cap_grad, std::min(max_cap_grad, gradPc_z));

				// Darcy flux
				double transmissivity = permeability[idx] / viscosity[idx];

				// Calculate gravity head: rho (1000 kg/m^3) * g (9.81 m/s^2)
				// Assuming +i direction is downwards.
				const double rho_g = 1000.0 * 9.81;

				// Liquid Pressure Gradient = gradP - gradPc
				double liq_grad_x = gradP_x - gradPc_x;
				double liq_grad_y = gradP_y - gradPc_y;
				double liq_grad_z = gradP_z - gradPc_z;

				// Darcy's law with gravity and capillary suction combined
				// flux_x corresponds to the 'i' index (Z-axis, vertical flow)
				// Darcy's law with gravity: q = -(k/mu) * (gradP - rho*g)
				flux_x[idx] = -transmissivity * (liq_grad_x - rho_g);
				// Horizontal flows aren't affected by gravity
				flux_y[idx] = -transmissivity * liq_grad_y;
				flux_z[idx] = -transmissivity * liq_grad_z;
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
void Simulation::setupConcentrationEqns(double dt, MPIDomain<double, 1, IDX_SCHEME> &field, double inlet_bc, double sulphide_bc)
{
	MatDestroy(&coeff_mat);
	MatCreate(PETSC_COMM_WORLD, &coeff_mat);
	MatSetSizes(coeff_mat, local.extent.size(), local.extent.size(), PETSC_DETERMINE, PETSC_DETERMINE);
	MatSetType(coeff_mat, MATMPIAIJ);
	// MatMPIAIJSetPreallocation(coeff_mat, 8, PETSC_NULL, 8, PETSC_NULL);
	MatMPIAIJSetPreallocation(coeff_mat, 8, PETSC_NULLPTR, 8, PETSC_NULLPTR);
	// Allow sulphide to pore transitions in the matrix structure
	MatSetOption(coeff_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
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

				if (voxel_type == Rock || voxel_type == Precipitate || voxel_type == Sulphide)
				{
					flux_x[idx] = 0.0;
					flux_y[idx] = 0.0;
					flux_z[idx] = 0.0;
					continue;
				}

				// Set values the linear system (Ax = b) for air, sulphide and rock voxels
				if (voxel_type == Air)
				{
					// Air boundaries (top/bottom) act as fresh water inlet or open outlet (C = 0.0)
					MatSetValue(coeff_mat, arridx, arridx, 1.0, ADD_VALUES);
					// VecSetValue(sources_vec, arridx, 0.0, ADD_VALUES);
					VecSetValue(sources_vec, arridx, inlet_bc, ADD_VALUES);
					continue;
				}
				else if (voxel_type == Sulphide)
				{
					// Sulphide voxels act as a saturated source (C = 1.0)
					MatSetValue(coeff_mat, arridx, arridx, 1.0, ADD_VALUES);
					// VecSetValue(sources_vec, arridx, 1.0, ADD_VALUES);
					VecSetValue(sources_vec, arridx, sulphide_bc, ADD_VALUES);
					continue;
				}
				else if (voxel_type == Rock || voxel_type == Precipitate)
				{
					// Rock & Precipitate are impermeable; they retain their current concentration
					MatSetValue(coeff_mat, arridx, arridx, 1.0, ADD_VALUES);
					VecSetValue(sources_vec, arridx, field[idx], ADD_VALUES);
					continue;
				}

				MatSetValue(coeff_mat, arridx, arridx, inv_dt, ADD_VALUES);
				// VecSetValue(sources_vec, arridx, conc[idx] * inv_dt, ADD_VALUES);
				VecSetValue(sources_vec, arridx, field[idx] * inv_dt, ADD_VALUES);

				// Lambda function for setting advection and diffussion rates
				auto set_link = [&](Index neighbor_idx, double q_face)
				{
					double D_eff = (img_data[neighbor_idx] == Pore) ? this->D * this->Dpore_fac : this->D;
					double diffusion_coeff = -D_eff / (dx * dx);

					// Group the physical coefficients for the (-L) operator
					double coeff_neighbor = diffusion_coeff;
					double coeff_self = -diffusion_coeff;

					double adv_coeff = q_face / dx;
					if (adv_coeff > 0)
					{
						coeff_self += adv_coeff; // upwind: flux out
					}
					else
					{
						coeff_neighbor -= adv_coeff; // upwind: flux in
					}

					// Build LHS Matrix A (weighted by theta)
					MatSetValue(coeff_mat, arridx, neighbor_idx.arrayId(global), theta * coeff_neighbor, ADD_VALUES);
					MatSetValue(coeff_mat, arridx, arridx, theta * coeff_self, ADD_VALUES);

					// Build RHS Vector b (weighted by 1.0 - theta)
					// Mathematically: L(C^n) = -(coeff_self * C_self + coeff_neighbor * C_neighbor)
					// double old_C_self = conc[idx];
					double old_C_self = field[idx];
					// double old_C_neighbor = conc[neighbor_idx];
					double old_C_neighbor = field[neighbor_idx];

					double explicit_rhs_term = -(coeff_self * old_C_self + coeff_neighbor * old_C_neighbor);
					VecSetValue(sources_vec, arridx, (1.0 - theta) * explicit_rhs_term, ADD_VALUES);
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
	KSPConvergedReason reason;
	VecPlaceArray(solution_vec, conc.getData().get() + conc.pad_size);
	KSPReset(ksp);
	// KSPSetOperators(ksp, coeff_mat, coeff_mat, DIFFERENT_NONZERO_PATTERN);
	KSPSetOperators(ksp, coeff_mat, coeff_mat);
	KSPSolve(ksp, sources_vec, solution_vec);
	PetscInt min_loc, max_loc;
	PetscReal min_val, max_val;
	VecMin(solution_vec, &min_loc, &min_val);
	VecMax(solution_vec, &max_loc, &max_val);
	VecResetArray(solution_vec);
	KSPGetIterationNumber(ksp, &its);
	KSPGetConvergedReason(ksp, &reason);
	// MPIOUT(mpi_rank) << "Concentration solve converged in " << its << " iterations." << endl;

	if (mpi_rank == 0)
	{
		if (reason > 0)
		{
			cout << "    [Concentration] Converged in " << its << " iterations. (Reason Code: " << reason << ")" << endl;
			// Print the exact bounds of the concentration field
			cout << "    [Concentration] Min Value: " << min_val << " Pa | Max Value: " << max_val << " Pa" << endl;
		}
		else
		{
			cout << "    [Concentration] DIVERGED/FAILED in " << its << " iterations! (Error Code: " << reason << ")" << endl;
		}
	}
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
		if (data_flags & VTKOutput::AcidConc)
			fout << "\t\t\t<PDataArray type=\"Float64\" Name=\"AcidConcentration\"/>" << endl;

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
	if (data_flags & VTKOutput::AcidConc)
	{
		auto arr = vtkSmartPointer<vtkDoubleArray>::New();
		arr->SetName("AcidConcentration");
		arr->SetNumberOfValues(num_voxels_to_write);
		imageData->GetPointData()->AddArray(arr);
	}

	// NaN check loop
	int nan_count_pressure = 0;
	int nan_count_cap = 0;
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
	{
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
		{
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i + off_pos); ++i)
			{
				Index idx(i, j, k);
				// NaN check
				if (std::isnan(pressure[idx]) || std::isinf(pressure[idx]))
				{
					pressure[idx] = 0.0;
					nan_count_pressure++;
				}
				if (std::isnan(cap_pressure[idx]) || std::isinf(cap_pressure[idx]))
				{
					cap_pressure[idx] = 0.0;
					nan_count_cap++;
				}
			}
		}
	}

	if (nan_count_pressure > 0 || nan_count_cap > 0)
	{
		std::cout << "    [VTK Writer Rank " << mpi_rank << "] checked "
				  << nan_count_pressure << " Pressure NaNs and "
				  << nan_count_cap << " CapPressure NaNs from boundary padding." << std::endl;
	}
	/*
	// This is for debugging
	else
	{
		std::cout << "    [VTK Writer Rank " << mpi_rank << "] checked "
				  << " No NaN values found." << std::endl;
	}
	*/

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
				if (data_flags & VTKOutput::AcidConc)
					imageData->GetPointData()->GetArray("AcidConcentration")->SetTuple1(count, conc_acid[idx]);
				count++;
			}
		}
	}

	vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
	writer->SetFileName(fname.str().c_str());
	writer->SetInputData(imageData); // SetInputData is simpler here but requires a newer version of VTK
	// writer->SetInputConnection(imageData->GetProducerPort());  // For older versions of VTK
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
				if (voxel_type == Pore || voxel_type == Sulphide)
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
				if (img_data[idx] == Pore || img_data[idx] == Sulphide)
				{
					// Calculate effective saturation, Se, ensuring it's within a valid range
					double Se = (saturation[idx] - s_res) / (1.0 - s_res);
					// Se = std::max(0.0, std::min(1.0, Se));
					//  Fix the lower bound to 1e-6 to prevent Capillary Pressure Infinity
					Se = std::max(1e-6, std::min(1.0, Se));

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
				if (img_data[idx] == Pore || img_data[idx] == Sulphide)
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
	// const double porosity = 0.1; // This should ideally be a configurable parameter

	// Calculate maximum divergence of flux to determine safe sub-steps
	double max_div_q = 0.0;
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
	{
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
		{
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				RAWType voxel_type = img_data[idx];
				if (voxel_type == Air || voxel_type == Rock || voxel_type == Sulphide || voxel_type == Precipitate)
				{
					continue;
				}

				Index neighbor_xp(i + 1, j, k);
				double q_face_xp = neighbor_xp.valid(global) ? (flux_x[idx] + flux_x[neighbor_xp]) / 2.0 : 0.0;
				Index neighbor_xm(i - 1, j, k);
				double q_face_xm = neighbor_xm.valid(global) ? (flux_x[idx] + flux_x[neighbor_xm]) / 2.0 : 0.0;

				Index neighbor_yp(i, j + 1, k);
				double q_face_yp = neighbor_yp.valid(global) ? (flux_y[idx] + flux_y[neighbor_yp]) / 2.0 : 0.0;
				Index neighbor_ym(i, j - 1, k);
				double q_face_ym = neighbor_ym.valid(global) ? (flux_y[idx] + flux_y[neighbor_ym]) / 2.0 : 0.0;

				Index neighbor_zp(i, j, k + 1);
				double q_face_zp = neighbor_zp.valid(global) ? (flux_z[idx] + flux_z[neighbor_zp]) / 2.0 : 0.0;
				Index neighbor_zm(i, j, k - 1);
				double q_face_zm = neighbor_zm.valid(global) ? (flux_z[idx] + flux_z[neighbor_zm]) / 2.0 : 0.0;

				double div_q = ((q_face_xp - q_face_xm) + (q_face_yp - q_face_ym) + (q_face_zp - q_face_zm)) / dx;

				// Calculate evaporative sink for stability check
				double evap_sink = 0.0;
				if (img_data[idx] == Pore)
				{
					int air_faces = 0;
					if (neighbor_xp.valid(global) && img_data[neighbor_xp] == Air)
						air_faces++;
					if (neighbor_xm.valid(global) && img_data[neighbor_xm] == Air)
						air_faces++;
					if (neighbor_yp.valid(global) && img_data[neighbor_yp] == Air)
						air_faces++;
					if (neighbor_ym.valid(global) && img_data[neighbor_ym] == Air)
						air_faces++;
					if (neighbor_zp.valid(global) && img_data[neighbor_zp] == Air)
						air_faces++;
					if (neighbor_zm.valid(global) && img_data[neighbor_zm] == Air)
						air_faces++;

					evap_sink = (air_faces * evaporative_flux) / dx;
				}

				// The total rate of change includes both flux divergence and evaporation
				double total_rate = std::abs(div_q) + evap_sink;
				if (total_rate > max_div_q)
					max_div_q = total_rate;
			}
		}
	}

	double global_max_div_q = 0.0;
	MPI_Allreduce(&max_div_q, &global_max_div_q, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);

	// Determine sub-steps. Limit saturation change to max 10% (0.1) per sub-step.
	double max_dsat = 0.1;
	double dt_safe = dt;
	if (global_max_div_q > 1e-20)
	{
		dt_safe = (max_dsat * porosity) / global_max_div_q;
	}

	// int num_steps = std::ceil(dt / dt_safe);
	// Use a double for the ceiling calculation, then cap it, then cast it.
	double calculated_steps = std::ceil(dt / dt_safe);
	int num_steps = 1;
	// Check for Infinity or NaN just in case dt_safe is 0.0
	if (std::isinf(calculated_steps) || std::isnan(calculated_steps))
	{
		num_steps = 1000;
	}
	else if (calculated_steps > 1000)
	{
		num_steps = 1000; // Hard cap
	}
	else if (calculated_steps > 1.0)
	{
		num_steps = (int)calculated_steps;
	}

	double dt_sub = dt / num_steps;
	double dt_sub_over_phi = dt_sub / porosity;

	// Print this to the log to monitor
	if (mpi_rank == 0)
	{
		std::cout << "    [Saturation] Max div(q): " << global_max_div_q
				  << " | Sub-steps: " << num_steps << " (" << dt_sub << "s)" << std::endl;
	}

	if (mpi_rank == 0 && num_steps > 1)
	{
		std::cout << "    [Saturation] Sub-stepping required: " << num_steps << " steps of " << dt_sub << "s." << std::endl;
	}

	MPIDomain<double, 1, IDX_SCHEME> saturation_new;
	saturation_new.setup(local.origin, local.extent);

	// Perform Sub-stepped updates
	for (int step = 0; step < num_steps; ++step)
	{
		for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
		{
			for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
			{
				for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
				{
					Index idx(i, j, k);
					RAWType voxel_type = img_data[idx];
					if (voxel_type == Air || voxel_type == Rock || voxel_type == Sulphide || voxel_type == Precipitate)
					{
						continue;
					}

					Index neighbor_xp(i + 1, j, k);
					double q_face_xp = neighbor_xp.valid(global) ? (flux_x[idx] + flux_x[neighbor_xp]) / 2.0 : 0.0;
					Index neighbor_xm(i - 1, j, k);
					double q_face_xm = neighbor_xm.valid(global) ? (flux_x[idx] + flux_x[neighbor_xm]) / 2.0 : 0.0;

					Index neighbor_yp(i, j + 1, k);
					double q_face_yp = neighbor_yp.valid(global) ? (flux_y[idx] + flux_y[neighbor_yp]) / 2.0 : 0.0;
					Index neighbor_ym(i, j - 1, k);
					double q_face_ym = neighbor_ym.valid(global) ? (flux_y[idx] + flux_y[neighbor_ym]) / 2.0 : 0.0;

					Index neighbor_zp(i, j, k + 1);
					double q_face_zp = neighbor_zp.valid(global) ? (flux_z[idx] + flux_z[neighbor_zp]) / 2.0 : 0.0;
					Index neighbor_zm(i, j, k - 1);
					double q_face_zm = neighbor_zm.valid(global) ? (flux_z[idx] + flux_z[neighbor_zm]) / 2.0 : 0.0;

					double div_q = ((q_face_xp - q_face_xm) + (q_face_yp - q_face_ym) + (q_face_zp - q_face_zm)) / dx;

					// apply evaporative sink at boundaries
					double evap_sink = 0.0;
					if (img_data[idx] == Pore)
					{
						int air_faces = 0;
						if (neighbor_xp.valid(global) && img_data[neighbor_xp] == Air)
							air_faces++;
						if (neighbor_xm.valid(global) && img_data[neighbor_xm] == Air)
							air_faces++;
						if (neighbor_yp.valid(global) && img_data[neighbor_yp] == Air)
							air_faces++;
						if (neighbor_ym.valid(global) && img_data[neighbor_ym] == Air)
							air_faces++;
						if (neighbor_zp.valid(global) && img_data[neighbor_zp] == Air)
							air_faces++;
						if (neighbor_zm.valid(global) && img_data[neighbor_zm] == Air)
							air_faces++;

						evap_sink = (air_faces * evaporative_flux) / dx;
					}

					// Update saturation using the combined sink term
					double new_sat = saturation[idx] - dt_sub_over_phi * (div_q + evap_sink);

					// Clamp the saturation to physical bounds
					saturation_new[idx] = std::max(0.0, std::min(1.0, new_sat));
				}
			}
		}

		// Atomically swap the new data into the main saturation field
		saturation.take(saturation_new.getData());

		// Setup saturation_new again for the next sub-step iteration
		if (step < num_steps - 1)
		{
			saturation_new.setup(local.origin, local.extent);
		}
	}
}

void Simulation::setupSaturationEqns(double dt)
{
	MatDestroy(&coeff_mat);
	MatCreate(PETSC_COMM_WORLD, &coeff_mat);
	MatSetSizes(coeff_mat, local.extent.size(), local.extent.size(), PETSC_DETERMINE, PETSC_DETERMINE);
	MatSetType(coeff_mat, MATMPIAIJ);
	MatMPIAIJSetPreallocation(coeff_mat, 8, PETSC_NULLPTR, 8, PETSC_NULLPTR);
	MatSetOption(coeff_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
	MatSetFromOptions(coeff_mat);
	VecZeroEntries(sources_vec);

	const double inv_dt = 1.0 / dt;

	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
	{
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
		{
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);
				int arridx = idx.arrayId(global);
				RAWType voxel_type = img_data[idx];

				// Inactive Regions (Rock, Air, Sulphide, Precipitate)
				if (voxel_type == Air || voxel_type == Rock || voxel_type == Sulphide || voxel_type == Precipitate)
				{
					// These boundaries maintain their current saturation state
					MatSetValue(coeff_mat, arridx, arridx, 1.0, ADD_VALUES);
					VecSetValue(sources_vec, arridx, saturation[idx], ADD_VALUES);
					continue;
				}

				// Active pore regions
				// Calculate evaporation sink
				double evap_sink = 0.0;
				Index neighbor_xp(i + 1, j, k);
				Index neighbor_xm(i - 1, j, k);
				Index neighbor_yp(i, j + 1, k);
				Index neighbor_ym(i, j - 1, k);
				Index neighbor_zp(i, j, k + 1);
				Index neighbor_zm(i, j, k - 1);

				int air_faces = 0;
				if (neighbor_xp.valid(global) && img_data[neighbor_xp] == Air)
					air_faces++;
				if (neighbor_xm.valid(global) && img_data[neighbor_xm] == Air)
					air_faces++;
				if (neighbor_yp.valid(global) && img_data[neighbor_yp] == Air)
					air_faces++;
				if (neighbor_ym.valid(global) && img_data[neighbor_ym] == Air)
					air_faces++;
				if (neighbor_zp.valid(global) && img_data[neighbor_zp] == Air)
					air_faces++;
				if (neighbor_zm.valid(global) && img_data[neighbor_zm] == Air)
					air_faces++;

				evap_sink = (air_faces * evaporative_flux) / dx;

				// Setup central diagonal (phi / dt)
				double coeff_self = porosity * inv_dt;
				double explicit_rhs = (porosity * inv_dt * saturation[idx]) - evap_sink;

				// Lambda for upwinded mobility advection
				auto set_link = [&](Index neighbor_idx, double q_face_outward)
				{
					if (neighbor_idx.valid(global))
					{
						if (q_face_outward > 0)
						{
							// Flow is leaving this cell -> upwind depends on local saturation
							// Max limit prevents division by zero on dry cells
							double v_eff = q_face_outward / std::max(saturation[idx], 1e-6);
							coeff_self += v_eff / dx;
						}
						else
						{
							// Flow is entering this cell -> upwind depends on neighbor saturation
							double v_eff = q_face_outward / std::max(saturation[neighbor_idx], 1e-6);
							// v_eff is negative, off-diagonal matrix entries for entering flux are negative
							MatSetValue(coeff_mat, arridx, neighbor_idx.arrayId(global), v_eff / dx, ADD_VALUES);
						}
					}
				};

				// Evaluate flux on all 6 faces (Calculate outward flux specifically)
				if (neighbor_xp.valid(global))
					set_link(neighbor_xp, (flux_x[idx] + flux_x[neighbor_xp]) / 2.0);
				if (neighbor_xm.valid(global))
					set_link(neighbor_xm, -(flux_x[idx] + flux_x[neighbor_xm]) / 2.0);
				if (neighbor_yp.valid(global))
					set_link(neighbor_yp, (flux_y[idx] + flux_y[neighbor_yp]) / 2.0);
				if (neighbor_ym.valid(global))
					set_link(neighbor_ym, -(flux_y[idx] + flux_y[neighbor_ym]) / 2.0);
				if (neighbor_zp.valid(global))
					set_link(neighbor_zp, (flux_z[idx] + flux_z[neighbor_zp]) / 2.0);
				if (neighbor_zm.valid(global))
					set_link(neighbor_zm, -(flux_z[idx] + flux_z[neighbor_zm]) / 2.0);

				// Insert local coefficients into PETSc
				MatSetValue(coeff_mat, arridx, arridx, coeff_self, ADD_VALUES);
				VecSetValue(sources_vec, arridx, explicit_rhs, ADD_VALUES);
			}
		}
	}

	MatAssemblyBegin(coeff_mat, MAT_FINAL_ASSEMBLY);
	VecAssemblyBegin(sources_vec);
	MatAssemblyEnd(coeff_mat, MAT_FINAL_ASSEMBLY);
	VecAssemblyEnd(sources_vec);
}

void Simulation::solveSaturation()
{
	PetscInt its;
	KSPConvergedReason reason;

	// Point the PETSc vector wrapper directly to the C++ Saturation array
	VecPlaceArray(solution_vec, saturation.getData().get() + saturation.pad_size);

	// Execute the Implicit Matrix Solve
	KSPReset(ksp);
	KSPSetOperators(ksp, coeff_mat, coeff_mat);
	KSPSolve(ksp, sources_vec, solution_vec);

	// Extract un-clamped min/max for logging
	PetscInt min_loc, max_loc;
	PetscReal min_val, max_val;
	VecMin(solution_vec, &min_loc, &min_val);
	VecMax(solution_vec, &max_loc, &max_val);

	// Unlink the PETSc memory
	VecResetArray(solution_vec);

	KSPGetIterationNumber(ksp, &its);
	KSPGetConvergedReason(ksp, &reason);

	// Thermodynamic clamp (prevent numerical undershoot/overshoot)
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
	{
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
		{
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);

				// Force saturation to remain strictly within [0.0, 1.0]
				saturation[idx] = std::max(0.0, std::min(1.0, saturation[idx]));
			}
		}
	}

	// Output Solver Statistics
	if (mpi_rank == 0)
	{
		if (reason > 0)
		{
			std::cout << "    [Implicit Saturation] Converged in " << its << " iterations. (Reason Code: " << reason << ")" << std::endl;
			std::cout << "    [Implicit Saturation] Raw Min: " << min_val << " | Raw Max: " << max_val << std::endl;
		}
		else
		{
			std::cout << "    [Implicit Saturation] DIVERGED/FAILED in " << its << " iterations. (Error Code: " << reason << ")" << std::endl;
		}
	}
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
						conc[idx] = 1.0;			 // Set concentration to maximum (saturated)
						img_data[idx] = Precipitate; // Clog the pore with precipitate
					}
				}
			}
		}
	}
}

/**
 * @brief Models precipitation in the bulk domain due to supersaturation.
 *
 * This function identifies pore voxels where the chemical concentration 'c' has
 * exceeded the saturation limit 'c_sat'. At these locations, it simulates
 * precipitation by changing the material type from Pore to Rock and resetting the
 * local concentration to the saturation limit.
 */

void Simulation::handlePrecipitation()
{
	for (int k = local.origin.k; k < (local.origin.k + local.extent.k); ++k)
	{
		for (int j = local.origin.j; j < (local.origin.j + local.extent.j); ++j)
		{
			for (int i = local.origin.i; i < (local.origin.i + local.extent.i); ++i)
			{
				Index idx(i, j, k);

				if (img_data[idx] == Pore && conc[idx] > c_sat)
				{
					// Calculate excess concentration (kg/m^3 of fluid)
					double excess_conc = conc[idx] - c_sat;

					if (use_instant_precipitation)
					{
						img_data[idx] = Precipitate;
					}
					else
					{
						// Convert excess concentration to solid volume.
						// mass = conc * (voxel_volume * porosity * saturation)
						// solid_volume = mass / solid_density
						// For simplicity, tracking it as a volume fraction [0.0 to 1.0]:

						double fluid_vol_fraction = porosity * saturation[idx];
						double precipitated_vol_fraction = (excess_conc * fluid_vol_fraction) / solid_density;

						precipitate_inventory[idx] += precipitated_vol_fraction;

						// If the pore is more than 90% full of solid, blind it.
						double critical_limit = 0.90;
						if (precipitate_inventory[idx] > critical_limit)
						{
							img_data[idx] = Precipitate;
						}
					}

					// Reset the local concentration to the saturation limit
					conc[idx] = c_sat;
				}
			}
		}
	}
}

void Simulation::solveAcid()
{
	PetscInt its;
	KSPConvergedReason reason;

	// Point the PETSc vector wrapper to the acid concentration array
	VecPlaceArray(solution_vec, conc_acid.getData().get() + conc_acid.pad_size);

	KSPReset(ksp);
	KSPSetOperators(ksp, coeff_mat, coeff_mat);
	KSPSolve(ksp, sources_vec, solution_vec);

	// Extract min/max for logging and stability checking
	PetscInt min_loc, max_loc;
	PetscReal min_val, max_val;
	VecMin(solution_vec, &min_loc, &min_val);
	VecMax(solution_vec, &max_loc, &max_val);

	// Unlink the memory
	VecResetArray(solution_vec);

	KSPGetIterationNumber(ksp, &its);
	KSPGetConvergedReason(ksp, &reason);

	if (mpi_rank == 0)
	{
		if (reason > 0)
		{
			cout << "    [Acid Transport] Converged in " << its << " iterations. (Reason Code: " << reason << ")" << endl;
			cout << "    [Acid Transport] Min Value: " << min_val << " | Max Value: " << max_val << endl;
		}
		else
		{
			cout << "    [Acid Transport] DIVERGED/FAILED in " << its << " iterations. (Error Code: " << reason << ")" << endl;
		}
	}
}

/**
 * @brief Writes checkpoint files with the values of primary state variables.
 *
 * This function uses a file-per-rank writing.
 * Each MPI process will directly dump its local memory slice
 * (including its padded ghost cells) to a raw binary file, such that
 * when you restart a simulation from a checkpoint,
 * each rank simply reads its own file back into memory
 */
void Simulation::writeCheckpoint(std::string out_dir, int step)
{
	// Create a unique filename for each MPI rank
	std::stringstream fname;
	fname << out_dir << "/checkpoint_rank" << mpi_rank << "_" << std::setw(6) << std::setfill('0') << step << ".bin";

	// Open the file in binary write mode
	std::ofstream fout(fname.str(), std::ios::out | std::ios::binary);
	if (!fout.is_open())
	{
		std::cerr << "    [Error] Rank " << mpi_rank << " failed to open checkpoint file for writing!" << std::endl;
		return;
	}

	// Get the total local array size (including padded boundaries)
	size_t total_size = pressure.padded.extent.size(); //

	// Dump the raw memory of the 7 primary state variables
	fout.write(reinterpret_cast<char *>(img_data.getData().get()), total_size * sizeof(RAWType));
	fout.write(reinterpret_cast<char *>(frac.getData().get()), total_size * sizeof(float));
	fout.write(reinterpret_cast<char *>(precipitate_inventory.getData().get()), total_size * sizeof(double));
	fout.write(reinterpret_cast<char *>(saturation.getData().get()), total_size * sizeof(double));
	fout.write(reinterpret_cast<char *>(pressure.getData().get()), total_size * sizeof(double));
	fout.write(reinterpret_cast<char *>(conc.getData().get()), total_size * sizeof(double));
	fout.write(reinterpret_cast<char *>(conc_acid.getData().get()), total_size * sizeof(double));

	fout.close();

	if (mpi_rank == 0)
	{
		std::cout << "    [Checkpoint] Successfully saved binary state for step " << step << " to " << out_dir << std::endl;
	}
}

/**
 * @brief Loads checkpoint files with the values of primary state variables.
 *
 * Just do the reverse process of writeCheckpoint
 */
void Simulation::loadCheckpoint(std::string out_dir, int step)
{
	// Locate the specific file for this MPI rank
	std::stringstream fname;
	fname << out_dir << "/checkpoint_rank" << mpi_rank << "_" << std::setw(6) << std::setfill('0') << step << ".bin";

	// Open the file in binary read mode
	std::ifstream fin(fname.str(), std::ios::in | std::ios::binary);
	if (!fin.is_open())
	{
		std::cerr << "    [Fatal Error] Rank " << mpi_rank << " failed to locate checkpoint file: " << fname.str() << std::endl;
		MPI_Abort(PETSC_COMM_WORLD, 1); // Abort the whole simulation if a piece is missing
		return;
	}

	// Get the total local array size
	size_t total_size = pressure.padded.extent.size(); //

	// Read the raw bytes directly back into the memory arrays
	fin.read(reinterpret_cast<char *>(img_data.getData().get()), total_size * sizeof(RAWType));
	fin.read(reinterpret_cast<char *>(frac.getData().get()), total_size * sizeof(float));
	fin.read(reinterpret_cast<char *>(precipitate_inventory.getData().get()), total_size * sizeof(double));
	fin.read(reinterpret_cast<char *>(saturation.getData().get()), total_size * sizeof(double));
	fin.read(reinterpret_cast<char *>(pressure.getData().get()), total_size * sizeof(double));
	fin.read(reinterpret_cast<char *>(conc.getData().get()), total_size * sizeof(double));
	fin.read(reinterpret_cast<char *>(conc_acid.getData().get()), total_size * sizeof(double));

	fin.close();

	if (mpi_rank == 0)
	{
		std::cout << "    [Checkpoint] Successfully loaded binary state from step " << step << std::endl;
	}
}
