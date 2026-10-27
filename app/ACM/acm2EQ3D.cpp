/** @file acm2EQ3D.cpp
 *  @brief Compiled 3-D ACM 2EQ conservative-state executable.
 */
#include "ACM/SingleBlockApp.hpp"

/**
 * @brief Run the acm2EQ3D solver command-line entry point.
 * @param argc Number of command-line arguments, including the executable name.
 * @param argv Argument strings selecting a case or schema/check mode and optional JSON-pointer/value overrides.
 * @return Exit status from the dimension/model-specific application driver; zero denotes successful completion.
 */
int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const int error = DNDS::ACM::RunSingleBlockConsoleApp<
        DNDS::ACM::ACMModel::ConstantDensity2EQ3D>(argc, argv);
    if (error != 0)
        MPI_Abort(MPI_COMM_WORLD, error);
    MPI_Finalize();
    return error;
}
