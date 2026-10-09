/** @file NCFVApp.hpp @brief Command-line entry shared by runtime dimensions. */
#pragma once

#include "NCFVSolver.hpp"

#include <argparse.hpp>

#include <filesystem>
#include <iostream>

namespace DNDS::NCFV
{
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
        parser.add_argument("-k", "--overwrite_key")
            .append()
            .default_value<std::vector<std::string>>({});
        parser.add_argument("-v", "--overwrite_value")
            .append()
            .default_value<std::vector<std::string>>({});
        parser.add_argument("--emit-schema").flag().default_value(false);
        parser.add_argument("--check-config").flag().default_value(false);

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
                        {"ncfv_euler", "NCFV", "IdealGas", dimension + 2});
                    schema["properties"]["dimension"]["const"] = dimension;
                    schema["properties"]["dimension"]["default"] = dimension;
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

            loaded.configuration.solver.Require(
                {"ncfv_euler", "NCFV", "IdealGas", dimension + 2});
            DNDS_check_throw_info(loaded.configuration.dimension == dimension,
                                  "NCFV configuration dimension does not match " + appName);
            if (parser.get<bool>("--check-config"))
                return 0;
            Solver<dimension> solver(mpi, loaded.configuration);
            solver.Initialize();
            solver.Run();
        }
        catch (const std::exception &exception)
        {
            if (mpi.rank == 0)
                std::cerr << "DNDS NCFV top-level exception: "
                          << exception.what() << std::endl;
            return 1;
        }
        return 0;
    }
}
