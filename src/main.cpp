#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <mpi.h>
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
 * 1. Initialising and finalizing PETSc and MPI.
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
		cmd_opts.add_options()("help,h", "Print this message and exit.")("raw-file", opts::value<string>()->required(), "Input RAW file specifying the domain.")("xext", opts::value<int>()->required(), "The x extent of the domain")("yext", opts::value<int>()->required(), "The y extent of the domain")("zext", opts::value<int>()->required(), "The z extent of the domain")("out_dir", opts::value<string>()->default_value("./output"), "The output directory")("header_size", opts::value<size_t>()->default_value(0), "RAW file header size in bytes.")("voxel_size", opts::value<double>()->default_value(1.), "Size of voxels (m).")("D", opts::value<double>()->default_value(1.), "Diffusion constant (m^2/s).")("kext", opts::value<double>()->default_value(1.), "Mass transfer to exterior.")("kreac", opts::value<double>()->default_value(1.), "Reaction transfer from sulphide grains.")("Dpore_fac", opts::value<double>()->default_value(1.), "Pore diffusivity enhancement factor")("dt", opts::value<double>()->default_value(1.), "Simulation time step (s).")("tmax", opts::value<double>()->default_value(10.), "Maximum simulation time (s).")("nout", opts::value<int>()->default_value(1), "Output every N-th time step.")("seed", opts::value<size_t>()->default_value(42), "Pseudo-RNG seed value.");

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

		// Maps arguments to the code's (i, j, k) = (Z, Y, X) internal indexing
		int3 global_extent(cmd["zext"].as<int>(), cmd["yext"].as<int>(), cmd["xext"].as<int>());
		string out_dir = cmd["out_dir"].as<string>();

		the_simulation.setupDomain(global_extent);
		the_simulation.setupPETSc();

		the_simulation.readRAW(cmd["raw-file"].as<string>(), cmd["header_size"].as<size_t>());

		the_simulation.doExchange();
		the_simulation.initFrac();
		the_simulation.setupReactionRates();

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

			the_simulation.initProperties();
			the_simulation.setupPressureEqns();
			the_simulation.solvePressure();
			the_simulation.pressure.exchangePadding(MPI_DOUBLE);
			the_simulation.calculateFlux();
			the_simulation.flux_x.exchangePadding(MPI_DOUBLE);
			the_simulation.flux_y.exchangePadding(MPI_DOUBLE);
			the_simulation.flux_z.exchangePadding(MPI_DOUBLE);
			the_simulation.setupConcentrationEqns(dt);
			the_simulation.solveConc();

			int leached_this_step = the_simulation.updateFrac(dt);
			int leached_total = 0;
			MPI_Allreduce(&leached_this_step, &leached_total, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);

			if (n % nout == 0)
			{
				if (mpi_rank == 0)
					cout << "Writing output for time step " << n << "..." << endl;
				the_simulation.writeVTKFile(out_dir + "/output", n, (VTKOutput)(Conc | Pressure | Frac | Type | Flux_Vec));
			}

			t += dt;
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