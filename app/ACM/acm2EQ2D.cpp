/** @file acm2EQ2D.cpp
 *  @brief Compiled 2-D ACM 2EQ conservative-state executable.
 */
#include "ACM/SingleBlockApp.hpp"

int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const int error = DNDS::ACM::RunSingleBlockConsoleApp<
        DNDS::ACM::ACMModel::ConstantDensity2EQ2D>(argc, argv);
    if (error != 0)
        MPI_Abort(MPI_COMM_WORLD, error);
    MPI_Finalize();
    return error;
}
