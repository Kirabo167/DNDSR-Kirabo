#include "UnifiedSolverDispatch.hpp"

#include "Euler/SingleBlockApp.hpp"

namespace DNDS::App
{
    int RunEulerNSEX3D(int argc, char *argv[], int fieldNVariables)
    {
        return Euler::RunSingleBlockConsoleApp<Euler::NS_EX_3D>(argc, argv, fieldNVariables);
    }
}
