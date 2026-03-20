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
		cmd_opts.add_options()("help,h", "Print this message and exit.")("raw-file", opts::value<string>()->required(), "Input RAW file specifying the domain.")("xext", opts::value<int>()->required(), "The x extent of the domain")("yext", opts::value<int>()->required(), "The y extent of the domain")("zext", opts::value<int>()->required(), "The z extent of the domain")("out_dir", opts::value<string>()->default_value("./output"), "The output directory")("header_size", opts::value<size_t>()->default_value(0), "RAW file header size in bytes.")("voxel_size", opts::value<double>()->default_value(1.), "Size of voxels (m).")("D", opts::value<double>()->default_value(1.), "Diffusion constant (m^2/s).")("kext", opts::value<double>()->default_value(1.), "Mass transfer to exterior.")("kreac", opts::value<double>()->default_value(1.), "Reaction transfer from sulphide grains.")("Dpore_fac", opts::value<double>()->default_value(1.), "Pore diffusivity enhancement factor")("dt", opts::value<double>()->default_value(1.), "Simulation time step (s).")("tmax", opts::value<double>()->default_value(10.), "Maximum simulation time (s).")("nout", opts::value<int>()->default_value(1), "Output every N-th time step.")("seed", opts::value<size_t>()->default_value(42), "Pseudo-RNG seed value.")("csat", opts::value<double>()->default_value(0.95), "Saturation concentration limit.")("evap_flux", opts::value<double>()->default_value(0.0), "Evaporative flux (m/s)")("theta", opts::value<double>()->default_value(1.0), "Time scheme (0=Explicit, 0.5=Crank-Nicolson, 1.0=Implicit)")("instant_precip", opts::value<bool>()->default_value(false), "Enable instant pore blinding upon supersaturation")("porosity", opts::value<double>()->default_value(0.1), "Material porosity")("top_pressure", opts::value<double>()->default_value(10000.0), "Applied pressure head at the top boundary (Pa, simulates irrigation)")("max_cap_grad", opts::value<double>()->default_value(50000.0), "Maximum capillary pressure gradient (Pa/m) for explicit solver stability")("implicit_sat", opts::value<bool>()->default_value(false), "Toggle: True = Implicit PETSc Solver, False = Explicit Sub-stepping")("restart_step", opts::value<int>()->default_value(0), "Step to restart from (0 = fresh start)")("chk_freq", opts::value<int>()->default_value(1000), "Number of steps between saving binary checkpoints")("debug", opts::value<bool>()->default_value(false), "Enable detailed runtime debugging and NaN checks");

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
		// opts::store(opts::parse_command_line(argc, argv, cmd_opts), cmd);
		// Allow command line options not defined before
		// This allos passing some PETSc options like preconditioners
		opts::store(opts::command_line_parser(argc, argv).options(cmd_opts).allow_unregistered().run(), cmd);

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
		the_simulation.use_implicit_saturation = cmd["implicit_sat"].as<bool>();
		the_simulation.debug_mode = cmd["debug"].as<bool>();

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
			std::cout << "  Saturation Solver                : " << (the_simulation.use_implicit_saturation ? "Implicit (PETSc)" : "Explicit (Sub-stepping)") << std::endl;
			std::cout << "  Debug Mode                       : " << (the_simulation.debug_mode ? "True (Active)" : "False (Inactive)") << std::endl;
			std::cout << "==================================================\n"
					  << std::endl;
		}
		// =====================================================================

		if (mpi_rank == 0)
		{
			boost::filesystem::create_directories(out_dir); // create output directory
			cout << "\n*** Starting Simulation ***" << endl;
		}

		// TEMPORAL RAMPING SETUP
		double dt_target = dt_actual;
		double current_dt = dt_target;
		// TODO: Change this to a cmd option
		bool use_ramping = true;

		if (use_ramping && start_step == 0) // Only ramp on a fresh start, not restarts
		{
			current_dt = std::min(60.0, dt_target); // Start with a 60-second step
			if (mpi_rank == 0)
				cout << "Temporal Ramping Enabled: Starting at dt = " << current_dt << "s" << endl;
		}

		// TIME TRACKING VECTORS FOR PARAVIEW
		std::vector<double> history_t;
		std::vector<size_t> history_n;

		do
		{
			bool step_successful = false;

			// =================================================================
			// THE ADAPTIVE TIME-STEPPING (RETRY) LOOP
			// =================================================================
			while (!step_successful)
			{
				// 1. Snapshot the memory before calculating any physics
				the_simulation.saveState();
				step_successful = true; // Assume success until proven otherwise

				if (mpi_rank == 0)
					cout << "\nIteration: " << n << " | Time: " << t << "s | dt: " << current_dt << "s" << endl;

				// -------------------------------------------------------------
				// A. Hydrodynamics (Picard iteration loop)
				// -------------------------------------------------------------
				int max_picard_iters = the_simulation.use_implicit_saturation ? 3 : 1;

				for (int picard = 0; picard < max_picard_iters; ++picard)
				{
					if (mpi_rank == 0 && the_simulation.use_implicit_saturation)
						cout << "    --- Picard Iteration " << picard + 1 << " ---" << endl;

					// Update properties and sync boundaries
					the_simulation.updateRelativePermeability();
					the_simulation.updateCapillaryPressure();
					the_simulation.initProperties();
					the_simulation.permeability.exchangePadding(MPI_DOUBLE);

					// Solve pressure
					the_simulation.setupPressureEqns();
					if (!the_simulation.solvePressure())
					{
						step_successful = false;
						break;
					} // Abort Picard
					the_simulation.pressure.exchangePadding(MPI_DOUBLE);

					// Calculate fluxes
					the_simulation.calculateFlux();
					the_simulation.flux_x.exchangePadding(MPI_DOUBLE);
					the_simulation.flux_y.exchangePadding(MPI_DOUBLE);
					the_simulation.flux_z.exchangePadding(MPI_DOUBLE);

					// Solve saturation
					if (the_simulation.use_implicit_saturation)
					{
						the_simulation.setupSaturationEqns(current_dt);
						if (!the_simulation.solveSaturation())
						{
							step_successful = false;
							break;
						} // Abort Picard
					}
					else
					{
						the_simulation.updateSaturation(current_dt);
					}
					the_simulation.saturation.exchangePadding(MPI_DOUBLE);
				}

				// -------------------------------------------------------------
				// B. Post-Picard Physics (Only executes if hydrodynamics succeeded)
				// -------------------------------------------------------------
				if (step_successful)
				{
					the_simulation.handleSurfaceEffects();

					// Dissolved mineral transport
					the_simulation.setupConcentrationEqns(current_dt, the_simulation.conc, 0.0, 1.0);
					if (!the_simulation.solveConc())
					{
						step_successful = false;
					}
					else
					{
						the_simulation.conc.exchangePadding(MPI_DOUBLE);
					}
				}

				if (step_successful)
				{
					// Acid transport
					the_simulation.setupConcentrationEqns(current_dt, the_simulation.conc_acid, 1.0, 0.0);
					if (!the_simulation.solveAcid())
					{
						step_successful = false;
					}
					else
					{
						the_simulation.conc_acid.exchangePadding(MPI_DOUBLE);
					}
				}

				if (step_successful)
				{
					// Reactions and precipitation
					the_simulation.handlePrecipitation();

#if DATA_TYPE == 16
					the_simulation.img_data.exchangePadding(MPI_UNSIGNED_SHORT);
#elif DATA_TYPE == 8
					the_simulation.img_data.exchangePadding(MPI_UNSIGNED_CHAR);
#endif
					// Update remaining reactive fraction
					int leached_this_step = the_simulation.updateFrac(current_dt);
					int leached_total = 0;
					MPI_Allreduce(&leached_this_step, &leached_total, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
					if (mpi_rank == 0)
						cout << "Done iteration. [" << leached_total << " voxels fully leached this step.]" << endl;
				}

				// -------------------------------------------------------------
				// C. The Retry Manager Trigger
				// -------------------------------------------------------------
				if (!step_successful)
				{
					if (mpi_rank == 0)
						cout << "    [Retry Manager] Physics diverged! Rolling back memory and halving dt..." << endl;

					the_simulation.restoreState(); // Wipe the garbage data
					current_dt *= 0.5;			   // Cut the time step in half

					// Hard limit to prevent infinite loops on irreversibly unstable matrices
					if (current_dt < 1e-3)
					{
						if (mpi_rank == 0)
							cout << "    [FATAL ERROR] Time step shrank below 1ms. Simulation is irreversibly unstable. Aborting." << endl;
						MPI_Abort(PETSC_COMM_WORLD, 1);
					}
				}
			} // End of Adaptive Retry Loop

			// =================================================================
			// OUTPUT AND TIME ADVANCEMENT
			// =================================================================

			// Property sync and file output
			if (n % cmd["nout"].as<int>() == 0)
			{
				if (mpi_rank == 0)
					cout << "Syncing physical properties for VTK output..." << endl;

				the_simulation.updateRelativePermeability();
				the_simulation.updateCapillaryPressure();
				the_simulation.initProperties();
				the_simulation.permeability.exchangePadding(MPI_DOUBLE);

				if (mpi_rank == 0)
					cout << "Writing output for time step " << n << "..." << endl;

				size_t output_flags = VTKOutput::Conc | VTKOutput::Pressure | VTKOutput::Frac |
									  VTKOutput::Type | VTKOutput::Flux_Vec | VTKOutput::Saturation |
									  VTKOutput::CapPressure | VTKOutput::RelPermeability |
									  VTKOutput::Permeability | VTKOutput::AcidConc;

				the_simulation.writeVTKFile(out_dir + "/output", n, output_flags);

				// PVD Time-tracking file
				history_t.push_back(t);
				history_n.push_back(n);

				if (mpi_rank == 0)
				{
					std::ofstream pvd(out_dir + "/leaching_simulation.pvd");
					pvd << "<?xml version=\"1.0\"?>\n";
					pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
					pvd << "  <Collection>\n";

					for (size_t i = 0; i < history_t.size(); ++i)
					{
						pvd << "    <DataSet timestep=\"" << history_t[i]
							<< "\" file=\"output_" << std::setw(6) << std::setfill('0') << history_n[i] << ".pvti\"/>\n";
					}

					pvd << "  </Collection>\n";
					pvd << "</VTKFile>\n";
					pvd.close();
				}
			}

			// Checkpoint output
			if (n > 0 && n % cmd["chk_freq"].as<int>() == 0)
			{
				the_simulation.writeCheckpoint(out_dir, n);
			}

			// Fast-forward simulation time
			t += current_dt;
			n++;

			// ==========================================
			// ACCELERATE THE TIME STEP (TEMPORAL RAMPING)
			// ==========================================
			if (use_ramping && current_dt < dt_target)
			{
				current_dt *= 1.5; // Grow the time step by 50% each iteration
				if (current_dt > dt_target)
				{
					current_dt = dt_target; // Cap it strictly at the user's target
					if (mpi_rank == 0)
						cout << "    [Time Manager] Ramping complete. Reached target dt = " << dt_target << "s." << endl;
				}
				else
				{
					if (mpi_rank == 0)
						cout << "    [Time Manager] Accelerating next dt to " << current_dt << "s." << endl;
				}
			}

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
