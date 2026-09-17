#include "UnifiedSolverDispatch.hpp"

#include "ACM/SingleBlockApp.hpp"

namespace DNDS::App
{
    int RunACMConstantDensity2D(int argc, char *argv[])
    {
        return ACM::RunSingleBlockConsoleApp<ACM::ACMModel::ConstantDensity2D>(argc, argv);
    }

    int RunACMConstantDensity3D(int argc, char *argv[])
    {
        return ACM::RunSingleBlockConsoleApp<ACM::ACMModel::ConstantDensity3D>(argc, argv);
    }
}
