#include "UnifiedSolverDispatch.hpp"

#include "Euler/SingleBlockApp.hpp"

namespace DNDS::App
{
    int RunEulerNS3D(int argc, char *argv[])
    {
        return Euler::RunSingleBlockConsoleApp<Euler::NS_3D>(argc, argv);
    }
}
