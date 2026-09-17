#include "UnifiedSolverDispatch.hpp"

#include "Euler/SingleBlockApp.hpp"

namespace DNDS::App
{
    int RunEulerNSEX(int argc, char *argv[], int fieldNVariables)
    {
        return Euler::RunSingleBlockConsoleApp<Euler::NS_EX>(argc, argv, fieldNVariables);
    }
}
