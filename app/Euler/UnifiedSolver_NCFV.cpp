#include "UnifiedSolverDispatch.hpp"

#include "NCFV/NCFVApp.hpp"

namespace DNDS::App
{
    int RunNCFVIdealGas(int argc, char *argv[])
    {
        return NCFV::RunConsoleApp(argc, argv);
    }
}
