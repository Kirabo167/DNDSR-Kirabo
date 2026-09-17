#include "UnifiedSolverDispatch.hpp"

#include "Euler/SingleBlockApp.hpp"

namespace DNDS::App
{
    int RunEulerNS(int argc, char *argv[])
    {
        return Euler::RunSingleBlockConsoleApp<Euler::NS>(argc, argv);
    }
}
