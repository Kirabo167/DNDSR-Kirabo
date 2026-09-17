#include "UnifiedSolverDispatch.hpp"

#include "Euler/SingleBlockApp.hpp"

namespace DNDS::App
{
    int RunEulerNS2EQ(int argc, char *argv[])
    {
        return Euler::RunSingleBlockConsoleApp<Euler::NS_2EQ>(argc, argv);
    }
}
