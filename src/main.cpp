#include <iostream>
#include <iomanip>
#include <stdexcept>
// #include <mpi.h>
#include <petsc.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "compiler_opts.h"
#include "Simulation.h"

using namespace std;
namespace opts = boost::program_options;

const char *help = "PETSc help message for the leaching simulator.";

/**
 * @brief The main entry point and driver for the simulation.
 * @details This function executes the workflow:
 * 1. Initialising and finalising PETSc and MPI.
 * 2. Parsing all command-line arguments using Boost Program Options.
 * 3. Creating and setting up the main Simulation object.
 * 4. Executing the main time-stepping loop.
 * 5. Handling any exceptions that occur during the run.
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return 0 on successful completion, non-zero on error.
 */

int main(int argc, char *argv[])
{
	try
	{
		int mpi_rank;
		PetscInitialize(&argc, &argv, 0, help);
		MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);

		// Command line argument parsing
		opts::options_description cmd_opts("Command line arguments");
		cmd_opts.add_options()("help,h", "Print this message and exit.")("raw-file", opts::value<string>()->required(), "Input RAW file specifying the domain.")("xext", opts::value<int>()->required(), "The x extent of the domain")("yext", opts::value<int>()->required(), "The y extent of the domain")("zext", opts::value<int>()->required(), "The z extent of the domain")("out_dir", opts::value<string>()->default_value("./output"), "The output directory")("header_size", opts::value<size_t>()->default_value(0), "RAW file header size in bytes.")("voxel_size", opts::value<double>()->default_value(1.), "Size of voxels (m).")("D", opts::value<double>()->default_value(1.), "Diffusion constant (m^2/s).")("kext", opts::value<double>()->default_value(1.), "Mass transfer to exterior.")("kreac", opts::value<double>()->default_value(1.), "Reaction transfer from sulphide grains.")("Dpore_fac", opts::value<double>()->default_value(1.), "Pore diffusivity enhancement factor")("dt", opts::value<double>()->default_value(1.), "Simulation time step (s).")("tmax", opts::value<double>()->default_value(10.), "Maximum simulation time (s).")("nout", opts::value<int>()->default_value(1), "Output every N-th time step.")("seed", opts::value<size_t>()->default_value(42), "Pseudo-RNG seed value.")("csat", opts::value<double>()->default_value(0.95), "Saturation concentration limit.")("evap_flux", opts::value<double>()->default_value(0.0), "Evaporative flux (m/s)")("theta", opts::value<double>()->default_value(1.0), "Time scheme (0=Explicit, 0.5=Crank-Nicolson, 1.0=Implicit)")("instant_precip", opts::value<bool>()->default_value(false), "Enable instant pore blinding upon supersaturation")("porosity", opts::value<double>()->default_value(0.1), "Material porosity")("top_pressure", opts::value<double>()->default_value(10000.0), "Applied pressure head at the top boundary (Pa, simulates irrigation)")("max_cap_grad", opts::value<double>()->default_value(50000.0), "Maximum capillary pressure gradient (Pa/m) for explicit solver stability")("implicit_sat", opts::value<bool>()->default_value(false), "Toggle: True = Implicit PETSc Solver, False = Explicit Sub-stepping")("restart_step", opts::value<int>()->default_value(0), "Step to restart from (0 = fresh start)")("chk_freq", opts::value<int>()->default_value(1000), "Number of steps between saving binary checkpoints");
		/* Simluation cases can vary depending on some options values, e.g.
		Leaching case
			./LeachingSim --top_pressure 10000.0 --evap_flux 0.0 ...
			The material is "irrigated" keeping it wet. Evaporation is negligible.
		Drying case
			./LeachingSim --top_pressure 0.0 --evap_flux 1.0e-6 ...
			There is no downward push.
			Evaporation pulls liquid volume out of the surface-connected pores and surface pores are converted into solid
		*/

		opts::variables_map cmd;
		opts::store(opts::parse_command_line(argc, argv, cmd_opts), cmd);

		if (cmd.count("help") && mpi_rank == 0)
		{
			cout << "Dynamic Advection-Diffusion Leaching Simulator" << endl;
			cout << cmd_opts << endl;
			PetscFinalize();
			return 0;
		}
		opts::notify(cmd);

		// Simulation Setup
		Domain::BuildMPIDataType();
		Simulation the_simulation(cmd["seed"].as<size_t>());
		the_simulation.D = cmd["D"].as<double>();
		the_simulation.dx = cmd["voxel_size"].as<double>();
		the_simulation.kext = cmd["kext"].as<double>();
		the_simulation.kreac = cmd["kreac"].as<double>();
		the_simulation.Dpore_fac = cmd["Dpore_fac"].as<double>();
		the_simulation.c_sat = cmd["csat"].as<double>();
		the_simulation.evaporative_flux = cmd["evap_flux"].as<double>();
		the_simulation.top_pressure = cmd["top_pressure"].as<double>();
		the_simulation.max_cap_grad = cmd["max_cap_grad"].as<double>();
		the_simulation.theta = cmd["theta"].as<double>();
		the_simulation.use_instant_precipitation = cmd["instant_precip"].as<bool>();
		the_simulation.porosity = cmd["porosity"].as<double>();

		// Initialise some constants
		the_simulation.vg_n = 2.0; // Must be strictly > 1.0
		the_simulation.vg_alpha = 1.0;
		the_simulation.s_res = 0.05; // 5% residual saturation

		// Maps arguments to the code's (i, j, k) = (Z, Y, X) internal indexing
		int3 global_extent(cmd["zext"].as<int>(), cmd["yext"].as<int>(), cmd["xext"].as<int>());
		string out_dir = cmd["out_dir"].as<string>();

		// Initialise the computational domain
		the_simulation.setupDomain(global_extent);
		the_simulation.setupPETSc(true);

		// Read the file with the physical domain
		the_simulation.readRAW(cmd["raw-file"].as<string>(), cmd["header_size"].as<size_t>());

		// Time Step Stability Check (Diffusive)
		double dx = the_simulation.dx;
		double D_coeff = the_simulation.D;
		double dt_user = cmd["dt"].as<double>();

		// Calculate the maximum stable time step based on diffusion
		// dt_stable = dx^2 / (2 * D)
		double dt_stable_max = (dx * dx) / (2.0 * D_coeff + 1e-30);

		// Choose the actual dt to use
		double dt_actual;

		// If Theta >= 0.5, the scheme is unconditionally stable for transport
		if (the_simulation.theta >= 0.5)
		{
			dt_actual = dt_user;
			if (mpi_rank == 0)
			{
				cout << "Using Theta=" << the_simulation.theta
					 << " (Unconditionally Stable). Using user dt = " << dt_actual << "s." << endl;
			}
		}
		else
		{
			// Explicit scheme (Theta < 0.5), requires stability check
			double stability_factor = 0.9;
			if (dt_user > dt_stable_max)
			{
				dt_actual = stability_factor * dt_stable_max;
				if (mpi_rank == 0)
				{
					cout << "[Warning] Explicit scheme requested. User dt (" << dt_user
						 << "s) exceeds stability limit (" << dt_stable_max
						 << "s). Using adjusted dt = " << dt_actual << "s." << endl;
				}
			}
			else
			{
				dt_actual = dt_user;
				if (mpi_rank == 0)
				{
					cout << "Using user-specified dt = " << dt_actual << "s (stable)." << endl;
				}
			}
		}

		// Main Simulation Loop
		double t = 0.0;
		size_t n = 0;

		// Initialisation or restart from checkpoint
		int start_step = cmd["restart_step"].as<int>();

		if (start_step > 0)
		{
			if (mpi_rank == 0)
				cout << "Restarting simulation from step " << start_step << "..." << endl;

			// Load the primary state variables from checkpoint
			the_simulation.loadCheckpoint(out_dir, start_step);

			// Re-initialise reaction rates based on the loaded geometry
			the_simulation.setupReactionRates();

			// Share the loaded state across MPI boundaries
			the_simulation.doExchange();

			// Rebuild the derived variables from the loaded state
			the_simulation.updateRelativePermeability();
			the_simulation.updateCapillaryPressure();
			the_simulation.initProperties();

			// Ensure the pressure solver has valid neighbor permeabilities
			the_simulation.permeability.exchangePadding(MPI_DOUBLE);

			// Fast-forward the time tracking variables
			n = start_step;
			t = start_step * dt_actual;
		}
		else
		{
			if (mpi_rank == 0)
				cout << "Initialising new simulation..." << endl;

			// Initialise fields
			the_simulation.initFields();

			// Initialise physical state
			the_simulation.initFrac();
			the_simulation.initSaturation();
			the_simulation.initAcid();
			the_simulation.setupReactionRates();

			// Share the initial state across MPI boundaries
			the_simulation.doExchange();

			// Precalculate initial properties before iteration 0 begins
			the_simulation.updateRelativePermeability();
			the_simulation.updateCapillaryPressure();
			the_simulation.initProperties();

			// Ensure coeff_mat is correctly built at t=0
			the_simulation.permeability.exchangePadding(MPI_DOUBLE);
		}

		double tmax = cmd["tmax"].as<double>();
		int nout = cmd["nout"].as<int>();

		// PARAMETER VERIFICATION LOG
		if (mpi_rank == 0)
		{
			std::cout << "\n======================================" << std::endl;
			std::cout << "  SIMULATION PARAMETERS" << std::endl;
			std::cout << "========================================" << std::endl;
			std::cout << "  Input file                       : " << cmd["raw-file"].as<string>() << std::endl;
			std::cout << "  Output folder                    : " << out_dir << std::endl;
			std::cout << "  Voxel size (dx)                  : " << the_simulation.dx << " m" << std::endl;
			std::cout << "  File size in voxels (x, y, z)    : " << "(" << cmd["zext"].as<int>() << ")" << "(" << cmd["yext"].as<int>() << ")" << "(" << cmd["zext"].as<int>() << ")" << std::endl;
			std::cout << "  Porosity                         : " << the_simulation.porosity << std::endl;
			std::cout << "  Diffusion (D)                    : " << the_simulation.D << std::endl;
			std::cout << "  Reaction rate (kreac)            : " << the_simulation.kreac << std::endl;
			std::cout << "  Saturation limit (csat)          : " << the_simulation.c_sat << std::endl;
			std::cout << "  Top Pressure (Irrigation)        : " << the_simulation.top_pressure << " Pa" << std::endl;
			std::cout << "  Evaporative flux                 : " << the_simulation.evaporative_flux << std::endl;
			std::cout << "  Max Capillary Grad               : " << the_simulation.max_cap_grad << " Pa/m" << std::endl;
			std::cout << "  Time step (dt)                   : " << dt_user << " s" << std::endl;
			std::cout << "  Time discretisation (Theta)      : " << the_simulation.theta << std::endl;
			std::cout << "  Precipitation mode               : " << (the_simulation.use_instant_precipitation ? "True (Instant)" : "False (Gradual)") << std::endl;
			std::cout << "==================================================\n"
					  << std::endl;
		}
		// =====================================================================

		if (mpi_rank == 0)
		{
			boost::filesystem::create_directories(out_dir); // create output directory
			cout << "\n*** Starting Simulation ***" << endl;
		}
		do
		{
			if (mpi_rank == 0)
				cout << "\nIteration: " << n << " | Time: " << t << "s" << endl;

			// 1. Hydrodynamics (Picard iteration loop)
			// If explicit use 1 pass, if implicit use 3 passes to converge non-linear physics.
			int max_picard_iters = the_simulation.use_implicit_saturation ? 3 : 1;

			for (int picard = 0; picard < max_picard_iters; ++picard)
			{
				if (mpi_rank == 0 && the_simulation.use_implicit_saturation)
					cout << "    --- Picard Iteration " << picard + 1 << " ---" << endl;

				// A. Update physical properties based on the latest saturation and materials
				the_simulation.updateRelativePermeability();
				the_simulation.updateCapillaryPressure();
				the_simulation.initProperties();
				the_simulation.permeability.exchangePadding(MPI_DOUBLE);

				// B. Solve pressure using the updated effective permeability
				the_simulation.setupPressureEqns();
				the_simulation.solvePressure();
				the_simulation.pressure.exchangePadding(MPI_DOUBLE);

				// C. Calculate Darcy flux and exchange MPI boundaries
				the_simulation.calculateFlux();
				the_simulation.flux_x.exchangePadding(MPI_DOUBLE);
				the_simulation.flux_y.exchangePadding(MPI_DOUBLE);
				the_simulation.flux_z.exchangePadding(MPI_DOUBLE);

				// D. Solve saturation using the updated flux
				if (the_simulation.use_implicit_saturation)
				{
					// 1-Step Implicit Solver (No sub-stepping limits)
					the_simulation.setupSaturationEqns(dt_actual);
					the_simulation.solveSaturation();
				}
				else
				{
					// Explicit sub-stepping solver (CFL limited)
					the_simulation.updateSaturation(dt_actual);
				}
				// Exchange saturation so the next Picard iteration or transport step (at last Picard iteration) is synced
				the_simulation.saturation.exchangePadding(MPI_DOUBLE);
			}

			// Apply evaporation clogging after hydrodynamics have settled
			the_simulation.handleSurfaceEffects();

			// 2. Dissolved mineral transport (conc)
			the_simulation.setupConcentrationEqns(dt_actual, the_simulation.conc, 0.0, 1.0);
			the_simulation.solveConc();
			the_simulation.conc.exchangePadding(MPI_DOUBLE);

			// 3. Acid transport (conc_acid)
			the_simulation.setupConcentrationEqns(dt_actual, the_simulation.conc_acid, 1.0, 0.0);
			the_simulation.solveAcid();
			the_simulation.conc_acid.exchangePadding(MPI_DOUBLE);

			// 4. Reactions and precipitation
			the_simulation.handlePrecipitation();

			// If precipitation alters the rock matrix, tell the neighbours
#if DATA_TYPE == 16
			the_simulation.img_data.exchangePadding(MPI_UNSIGNED_SHORT);
#elif DATA_TYPE == 8
			the_simulation.img_data.exchangePadding(MPI_UNSIGNED_CHAR);
#endif

			// Update remaining reactive fraction
			int leached_this_step = the_simulation.updateFrac(dt_actual);
			int leached_total = 0;
			MPI_Allreduce(&leached_this_step, &leached_total, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);

			// 5 Property sync and file output
			// Only update these properties if we are actually writing a file
			if (n % cmd["nout"].as<int>() == 0)
			{
				if (mpi_rank == 0)
					cout << "Syncing physical properties for VTK output..." << endl;

				the_simulation.updateRelativePermeability();
				the_simulation.updateCapillaryPressure();
				the_simulation.initProperties();

				// Execute the MPI exchange so the ParaView boundary stitching is seamless
				the_simulation.permeability.exchangePadding(MPI_DOUBLE);

				if (mpi_rank == 0)
					cout << "Writing output for time step " << n << "..." << endl;

				size_t output_flags = VTKOutput::Conc | VTKOutput::Pressure | VTKOutput::Frac |
									  VTKOutput::Type | VTKOutput::Flux_Vec | VTKOutput::Saturation |
									  VTKOutput::CapPressure | VTKOutput::RelPermeability |
									  VTKOutput::Permeability | VTKOutput::AcidConc;

				the_simulation.writeVTKFile(out_dir + "/output", n, output_flags);
			}

			// Checkpoint Output
			if (n > 0 && n % cmd["chk_freq"].as<int>() == 0)
			{
				the_simulation.writeCheckpoint(out_dir, n);
			}

			t += dt_actual;
			n++;
			if (mpi_rank == 0)
				cout << "Done iteration. [" << leached_total << " voxels fully leached this step.]" << endl;

		} while (t < tmax);

		if (mpi_rank == 0)
			cout << "\n*** Simulation Finished ***" << endl;

		PetscFinalize();
	}
	catch (const std::exception &e)
	{
		int rank = -1;
		int initialized = 0;
		MPI_Initialized(&initialized);
		if (initialized)
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		cerr << "An unhandled std::exception occurred on rank " << rank << ":" << endl;
		cerr << "\t" << e.what() << endl;
		if (initialized)
			MPI_Abort(MPI_COMM_WORLD, 1);
		return 1;
	}
	catch (...)
	{
		int rank = -1;
		int initialized = 0;
		MPI_Initialized(&initialized);
		if (initialized)
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		cerr << "An unknown exception occurred on rank " << rank << "." << endl;
		if (initialized)
			MPI_Abort(MPI_COMM_WORLD, 1);
		return 1;
	}

	return 0;
}

/*
int main(int argc, char *argv[])
{
   try
   {
	   int mpi_rank;
	   PetscInitialize(&argc, &argv, 0, help);
	   MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);

	   // Command line argument parsing
	   opts::options_description cmd_opts("Command line arguments");
	   cmd_opts.add_options()("help,h", "Print this message and exit.")("raw-file", opts::value<string>()->required(), "Input RAW file specifying the domain.")("xext", opts::value<int>()->required(), "The x extent of the domain")("yext", opts::value<int>()->required(), "The y extent of the domain")("zext", opts::value<int>()->required(), "The z extent of the domain")("out_dir", opts::value<string>()->default_value("./output"), "The output directory")("header_size", opts::value<size_t>()->default_value(0), "RAW file header size in bytes.")("voxel_size", opts::value<double>()->default_value(1.), "Size of voxels (m).")("D", opts::value<double>()->default_value(1.), "Diffusion constant (m^2/s).")("kext", opts::value<double>()->default_value(1.), "Mass transfer to exterior.")("kreac", opts::value<double>()->default_value(1.), "Reaction transfer from sulphide grains.")("Dpore_fac", opts::value<double>()->default_value(1.), "Pore diffusivity enhancement factor")("dt", opts::value<double>()->default_value(1.), "Simulation time step (s).")("tmax", opts::value<double>()->default_value(10.), "Maximum simulation time (s).")("nout", opts::value<int>()->default_value(1), "Output every N-th time step.")("seed", opts::value<size_t>()->default_value(42), "Pseudo-RNG seed value.")("csat", opts::value<double>()->default_value(0.95), "Saturation concentration limit.")("evap_flux", opts::value<double>()->default_value(1e-7), "Evaporative flux (m/s)");

	   opts::variables_map cmd;
	   opts::store(opts::parse_command_line(argc, argv, cmd_opts), cmd);

	   if (cmd.count("help") && mpi_rank == 0)
	   {
		   cout << "Dynamic Advection-Diffusion Leaching Simulator" << endl;
		   cout << cmd_opts << endl;
		   PetscFinalize();
		   return 0;
	   }
	   opts::notify(cmd);

	   // Simulation Setup
	   Domain::BuildMPIDataType();
	   Simulation the_simulation(cmd["seed"].as<size_t>());
	   the_simulation.D = cmd["D"].as<double>();
	   the_simulation.dx = cmd["voxel_size"].as<double>();
	   the_simulation.kext = cmd["kext"].as<double>();
	   the_simulation.kreac = cmd["kreac"].as<double>();
	   the_simulation.Dpore_fac = cmd["Dpore_fac"].as<double>();
	   the_simulation.c_sat = cmd["csat"].as<double>();
	   the_simulation.evaporative_flux = cmd["evap_flux"].as<double>();

	   // Maps arguments to the code's (i, j, k) = (Z, Y, X) internal indexing
	   int3 global_extent(cmd["zext"].as<int>(), cmd["yext"].as<int>(), cmd["xext"].as<int>());
	   string out_dir = cmd["out_dir"].as<string>();

	   the_simulation.setupDomain(global_extent);
	   the_simulation.setupPETSc(true);

	   the_simulation.readRAW(cmd["raw-file"].as<string>(), cmd["header_size"].as<size_t>());

	   the_simulation.doExchange();
	   the_simulation.initFrac();
	   the_simulation.initSaturation();
	   the_simulation.setupReactionRates();

	   // Time Step Stability Check (Diffusive)
	   double dx = the_simulation.dx;
	   double D_coeff = the_simulation.D;
	   double dt_user = cmd["dt"].as<double>();

	   // Calculate the maximum stable time step based on diffusion
	   // dt_stable = dx^2 / (2 * D)
	   // (Add a small value to avoid division by zero if D is zero,
	   // considering that values of the can be of the order 1e-15).
	   double dt_stable_max = (dx * dx) / (2.0 * D_coeff + 1e-30);

	   // Choose the actual dt to use
	   double dt_actual;
	   double stability_factor = 0.9; // Use 90% of the max stable step for safety
	   if (dt_user > dt_stable_max)
	   {
		   dt_actual = stability_factor * dt_stable_max;
		   if (mpi_rank == 0)
		   {
			   cout << "[Warning] User requested dt (" << dt_user
					<< "s) exceeds stability limit (" << dt_stable_max
					<< "s). Using adjusted dt = " << dt_actual << "s." << endl;
		   }
	   }
	   else
	   {
		   dt_actual = dt_user;
		   if (mpi_rank == 0)
		   {
			   cout << "Using user-specified dt = " << dt_actual << "s (stable)." << endl;
		   }
	   }

	   // Main Simulation Loop
	   double t = 0.0;
	   size_t n = 0;
	   double dt = cmd["dt"].as<double>();
	   double tmax = cmd["tmax"].as<double>();
	   int nout = cmd["nout"].as<int>();

	   if (mpi_rank == 0)
	   {
		   boost::filesystem::create_directories(out_dir); // create output directory
		   cout << "\n*** Starting Simulation ***" << endl;
	   }
	   do
	   {
		   if (mpi_rank == 0)
			   cout << "\nIteration: " << n << " | Time: " << t << "s" << endl;

		   the_simulation.updateRelativePermeability();
		   the_simulation.updateCapillaryPressure(); //

		   the_simulation.initProperties();
		   the_simulation.setupPressureEqns();
		   the_simulation.solvePressure();
		   the_simulation.pressure.exchangePadding(MPI_DOUBLE);
		   the_simulation.calculateFlux();
		   the_simulation.updateSaturation(dt);
		   the_simulation.handleSurfaceEffects();
		   the_simulation.flux_x.exchangePadding(MPI_DOUBLE);
		   the_simulation.flux_y.exchangePadding(MPI_DOUBLE);
		   the_simulation.flux_z.exchangePadding(MPI_DOUBLE);
		   the_simulation.setupConcentrationEqns(dt);
		   the_simulation.solveConc();
		   the_simulation.handlePrecipitation();

		   // int leached_this_step = the_simulation.updateFrac(dt);
		   int leached_this_step = the_simulation.updateFrac(dt_actual);
		   int leached_total = 0;
		   MPI_Allreduce(&leached_this_step, &leached_total, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);

		   if (n % nout == 0)
		   {
			   if (mpi_rank == 0)
				   cout << "Writing output for time step " << n << "..." << endl;
			   the_simulation.writeVTKFile(out_dir + "/output", n, (VTKOutput)(Conc | Pressure | Frac | Type | Flux_Vec | Saturation | CapPressure | RelPermeability | Permeability));
		   }

		   // t += dt;
		   t += dt_actual;
		   n++;
		   if (mpi_rank == 0)
			   cout << "Done iteration. [" << leached_total << " voxels fully leached this step.]" << endl;

	   } while (t < tmax);

	   if (mpi_rank == 0)
		   cout << "\n*** Simulation Finished ***" << endl;

	   PetscFinalize();
   }
   catch (const std::exception &e)
   {
	   int rank = -1;
	   int initialized = 0;
	   MPI_Initialized(&initialized);
	   if (initialized)
		   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	   cerr << "An unhandled std::exception occurred on rank " << rank << ":" << endl;
	   cerr << "\t" << e.what() << endl;
	   if (initialized)
		   MPI_Abort(MPI_COMM_WORLD, 1);
	   return 1;
   }
   catch (...)
   {
	   int rank = -1;
	   int initialized = 0;
	   MPI_Initialized(&initialized);
	   if (initialized)
		   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	   cerr << "An unknown exception occurred on rank " << rank << "." << endl;
	   if (initialized)
		   MPI_Abort(MPI_COMM_WORLD, 1);
	   return 1;
   }

   return 0;
}
*/
