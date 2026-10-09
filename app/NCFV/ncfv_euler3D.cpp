#include "NCFV/NCFVApp.hpp"

int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const int errorCode = DNDS::NCFV::RunConsoleApp<3>(argc, argv);
    if (errorCode != 0)
        MPI_Abort(MPI_COMM_WORLD, errorCode);
    MPI_Finalize();
    return errorCode;
}
