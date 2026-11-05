#include "ACMGPU/SingleBlockApp.hpp"

int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const int error = DNDS::ACMGPU::RunSingleBlockConsoleApp<DNDS::ACMGPU::ACMModel::ConstantDensity2EQ3D>(argc, argv);
    if (error != 0)
        MPI_Abort(MPI_COMM_WORLD, error);
    MPI_Finalize();
    return error;
}
