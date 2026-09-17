#include "UnifiedSolverDispatch.hpp"

#include "Euler/SingleBlockApp.hpp"

namespace DNDS::App
{
    int RunEulerNSSA(int argc, char *argv[])
    {
        return Euler::RunSingleBlockConsoleApp<Euler::NS_SA>(argc, argv);
    }
}
