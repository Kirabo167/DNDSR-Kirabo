/**
 * @file SingleBlockApp.hpp
 * @brief Compact command-line driver shared by ACM two- and three-dimensional applications.
 *
 * @details Each run reads one self-contained case JSON, then applies optional command-line
 * JSON-pointer overrides. Construction and solve operations are delegated to ACMSolver.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMSolver.hpp"

#include <argparse.hpp>

#include <filesystem>
#include <iostream>

namespace DNDS::ACM
{
    /**
     * @brief Return the conventional executable/case directory name for an ACM model.
     * @param model Compile-time ACM model value.
     * @return `acm2D` or `acm3D`.
     */
    constexpr const char *GetSingleBlockAppName(ACMModel model)
    {
        return model == ACMModel::ConstantDensity2D ? "acm2D" : "acm3D";
    }

    /**
     * @brief Parse one complete case JSON, initialize an ACM solver, and run time marching.
     * @tparam model Two- or three-dimensional constant-density model.
     * @param argc Command-line argument count.
     * @param argv Command-line argument array.
     * @return Zero on success; exceptions are converted to a nonzero return code.
     */
    template <ACMModel model>
    int RunSingleBlockConsoleApp(int argc, char *argv[])
    {
        MPIInfo mpi;
        mpi.setWorld();

        argparse::ArgumentParser parser(GetSingleBlockAppName(model), DNDS_VERSION_STRING);
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
                    auto schema = KernelConfiguration::schema(
                        std::string("DNDSR ") + GetSingleBlockAppName(model) + " configuration");
                    schema["$schema"] = "http://json-schema.org/draft-07/schema#";
                    SolverSelection::ConstrainSchema(schema,
                        {"ACM", "CFV", model == ACMModel::ConstantDensity2D ? "ConstantDensity2D" : "ConstantDensity3D", 4});
                    std::cout << schema.dump(4) << std::endl;
                }
                return 0;
            }

            const std::string requestedConfiguration = parser.get<std::string>("config");
            std::filesystem::path caseConfiguration =
                std::filesystem::path("../cases/acm") /
                (model == ACMModel::ConstantDensity2D ? "2D" : "3D") /
                (std::string(GetSingleBlockAppName(model)) + ".json");
            if (!requestedConfiguration.empty())
                caseConfiguration = requestedConfiguration;

            if (mpi.rank == 0)
                log() << "Reading complete ACM case configuration from "
                      << caseConfiguration.string() << std::endl;

            const LoadedConfiguration loaded = LoadConfiguration(
                caseConfiguration.string(),
                parser.get<std::vector<std::string>>("--overwrite_key"),
                parser.get<std::vector<std::string>>("--overwrite_value"));
            loaded.configuration.solver.Require(
                {"ACM", "CFV", model == ACMModel::ConstantDensity2D ? "ConstantDensity2D" : "ConstantDensity3D", 4});
            if (parser.get<bool>("--check-config"))
                return 0;
            ACMSolver<model> solver(mpi, loaded.configuration);
            solver.ReadMeshAndInitialize();
            solver.Run();
        }
        catch (const std::exception &exception)
        {
            if (mpi.rank == 0)
                std::cerr << "DNDS ACM top-level exception: " << exception.what() << std::endl;
            return 1;
        }
        return 0;
    }
}
