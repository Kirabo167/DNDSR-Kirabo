/** @file NCFVApp.hpp @brief Command-line entry shared by runtime dimensions. */
#pragma once

#include "NCFVSolver.hpp"

#include "DNDS/Config/SolverArguments.hpp"

#include <filesystem>
#include <iostream>

namespace DNDS::NCFV
{
    /**
     * @brief Run the dimension-specific NCFV command-line application with MPI communicator setup and error
     *     handling.
     * @param argc Number of command-line arguments, including the executable name.
     * @param argv Argument strings: case path or schema/config-check mode, followed by optional JSON-pointer/value
     *     overrides.
     * @return Zero after a successful run or configuration/schema request, or one after a caught top-level
     *     exception.
     * @tparam dimension Spatial dimension compiled into this executable, either 2 or 3.
     * @pre MPI has been initialized by the application entry point.
     */
    template <int dimension>
    int RunConsoleApp(int argc, char *argv[])
    {
        static_assert(dimension == 2 || dimension == 3);
        const std::string appName = "ncfv_euler" + std::to_string(dimension) + "D";
        MPIInfo mpi;
        mpi.setWorld();
        argparse::ArgumentParser parser(appName, DNDS_VERSION_STRING);
        parser.add_description(std::string("NCFV: ") + MethodName);
        parser.add_argument("config").default_value("");
        DNDS::Config::AddSolverArguments(parser);

        try
        {
            parser.parse_args(argc, argv);
            if (parser.get<bool>("--emit-schema"))
            {
                if (mpi.rank == 0)
                {
                    auto schema = Configuration::schema(
                        std::string("DNDSR NCFV (") + MethodName + ") configuration");
                    schema["$schema"] = "http://json-schema.org/draft-07/schema#";
                    SolverSelection::ConstrainSchema(schema,
                        {"ncfv_euler", "NCFV", "IdealGas", dimension + 2}, true);
                    auto &variables = schema["properties"]["solver"]["properties"]["fieldNVariables"];
                    variables["enum"] = {dimension + 2, dimension + 3, dimension + 4};
                    variables["description"] = "dim+2 flow components plus zero, one (SA), or two turbulence components";
                    schema["properties"]["dimension"]["const"] = dimension;
                    schema["properties"]["dimension"]["default"] = dimension;
                    if constexpr (dimension == 2)
                        schema["properties"]["physics"]["properties"]["boundaryMode"]["enum"] = PhysicalBoundaryModeNames();
                    std::cout << schema.dump(4) << std::endl;
                }
                return 0;
            }

            std::filesystem::path configurationPath =
                std::filesystem::path("../cases/ncfv_euler") /
                (std::to_string(dimension) + "D") / (appName + ".json");
            const std::string requested = parser.get<std::string>("config");
            if (!requested.empty())
                configurationPath = requested;
            const auto loaded = LoadConfiguration(
                configurationPath.string(),
                parser.get<std::vector<std::string>>("--overwrite_key"),
                parser.get<std::vector<std::string>>("--overwrite_value"));

            const int nTurbulence = loaded.configuration.physics.MakeTurbulenceModel().ComponentNumber();
            loaded.configuration.solver.Require(
                {"ncfv_euler", "NCFV", "IdealGas", dimension + 2 + nTurbulence});
            DNDS_check_throw_info(loaded.configuration.dimension == dimension,
                                  "NCFV configuration dimension does not match " + appName);
            if (parser.get<bool>("--check-config"))
                return 0;
            const auto run = [&](auto components)
            {
                Solver<dimension, decltype(components)::value> solver(mpi, loaded.configuration);
                solver.Initialize();
                solver.Run();
            };
            switch (nTurbulence)
            {
            case 0:
                run(std::integral_constant<int, 0>{});
                break;
            case 1:
                run(std::integral_constant<int, 1>{});
                break;
            case 2:
                run(std::integral_constant<int, 2>{});
                break;
            }
        }
        catch (const std::exception &exception)
        {
            std::cerr << "DNDS NCFV top-level exception on rank " << mpi.rank
                      << ": " << exception.what() << std::endl;
            return 1;
        }
        return 0;
    }
}
