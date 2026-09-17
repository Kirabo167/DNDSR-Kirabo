#include "UnifiedSolverDispatch.hpp"

#include "Euler/SingleBlockApp.hpp"

namespace DNDS::App
{
    int RunEulerNS2D(int argc, char *argv[])
    {
        return Euler::RunSingleBlockConsoleApp<Euler::NS_2D>(argc, argv);
    }
}
