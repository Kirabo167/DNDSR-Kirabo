/**
 * @file acm3D.cpp
 * @brief Minimal three-dimensional ACM executable entry point.
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#include "ACMGPU/SingleBlockApp.hpp"

/**
 * @brief Initialize MPI and dispatch the independent three-dimensional ACM solver.
 * @param argc Command-line argument count.
 * @param argv Command-line argument array.
 * @return Process exit code; MPI is aborted collectively on solver failure.
 */
int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const int errorCode = DNDS::ACMGPU::RunSingleBlockConsoleApp<
        DNDS::ACMGPU::ACMModel::ConstantDensity3D>(argc, argv);
    if (errorCode != 0)
        MPI_Abort(MPI_COMM_WORLD, errorCode);
    MPI_Finalize();
    return errorCode;
}
