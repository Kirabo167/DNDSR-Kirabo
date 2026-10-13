/** @file acmSA2D.cpp
 *  @brief Compiled 2-D ACM SA conservative-state executable.
 */
#include "ACM/SingleBlockApp.hpp"

int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const int error = DNDS::ACM::RunSingleBlockConsoleApp<
        DNDS::ACM::ACMModel::ConstantDensitySA2D>(argc, argv);
    if (error != 0)
        MPI_Abort(MPI_COMM_WORLD, error);
    MPI_Finalize();
    return error;
}
