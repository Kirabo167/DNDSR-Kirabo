#include "UnifiedSolverDispatch.hpp"

#include "ACMVariable/SingleBlockApp.hpp"

namespace DNDS::App
{
    int RunACMVariableDensity2D(int argc, char *argv[])
    {
        return ACMVariable::RunSingleBlockConsoleApp<
            ACMVariable::ACMModel::VariableDensity2D>(argc, argv);
    }

    int RunACMVariableDensity3D(int argc, char *argv[])
    {
        return ACMVariable::RunSingleBlockConsoleApp<
            ACMVariable::ACMModel::VariableDensity3D>(argc, argv);
    }
}
