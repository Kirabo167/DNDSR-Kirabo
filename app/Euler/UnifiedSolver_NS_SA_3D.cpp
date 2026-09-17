#include "UnifiedSolverDispatch.hpp"

#include "Euler/SingleBlockApp.hpp"

namespace DNDS::App
{
    int RunEulerNSSA3D(int argc, char *argv[])
    {
        return Euler::RunSingleBlockConsoleApp<Euler::NS_SA_3D>(argc, argv);
    }
}
