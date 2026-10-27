#include "NCFV/NCFVApp.hpp"

/**
 * @brief Run the ncfv_euler2D solver command-line entry point.
 * @param argc Number of command-line arguments, including the executable name.
 * @param argv Argument strings selecting a case or schema/check mode and optional JSON-pointer/value overrides.
 * @return Exit status from the dimension/model-specific application driver; zero denotes successful completion.
 */
int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const int errorCode = DNDS::NCFV::RunConsoleApp<2>(argc, argv);
    if (errorCode != 0)
        MPI_Abort(MPI_COMM_WORLD, errorCode);
    MPI_Finalize();
    return errorCode;
}
