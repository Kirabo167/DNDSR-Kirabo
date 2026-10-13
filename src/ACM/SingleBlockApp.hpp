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
        switch (model)
        {
        case ACMModel::ConstantDensity2D: return "acm2D";
        case ACMModel::ConstantDensity3D: return "acm3D";
        case ACMModel::ConstantDensitySA2D: return "acmSA2D";
        case ACMModel::ConstantDensitySA3D: return "acmSA3D";
        case ACMModel::ConstantDensity2EQ2D: return "acm2EQ2D";
        case ACMModel::ConstantDensity2EQ3D: return "acm2EQ3D";
        }
        return "unknownACM";
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
                    SolverSelection::ConstrainSchema(schema, ModelSelection<model>());
                    auto &initial = schema["properties"]["initialState"];
                    initial["minItems"] = ModelTraits<model>::nVarsFixed;
                    initial["maxItems"] = ModelTraits<model>::nVarsFixed;
                    auto &turbulence = schema["properties"]["turbulenceSettings"]["properties"];
                    if constexpr (ModelTraits<model>::nExtraVars == 0)
                    {
                        turbulence["model"]["const"] = "Laminar";
                        turbulence["model"]["default"] = "Laminar";
                    }
                    else
                    {
                        const bool sa = ModelTraits<model>::nExtraVars == 1;
                        auto &physical = schema["properties"]["acmSettings"]["properties"];
                        physical["enableViscousFlux"]["const"] = true;
                        physical["enableViscousFlux"]["default"] = true;
                        physical["dynamicViscosity"]["minimum"] = std::numeric_limits<real>::min();
                        physical["dynamicViscosity"]["default"] = 0.001;
                        turbulence["model"]["enum"] = sa
                            ? nlohmann::ordered_json::array({"SpalartAllmaras"})
                            : nlohmann::ordered_json::array({"KOmegaWilcox", "KOmegaSST", "RealizableKEpsilon"});
                        turbulence["model"]["default"] = sa ? "SpalartAllmaras" : "KOmegaWilcox";
                        turbulence["transportSubsteps"]["const"] = 1;
                        turbulence["transportSubsteps"]["default"] = 1;
                        nlohmann::ordered_json state = nlohmann::ordered_json::array({1, 0, 0, 0});
                        for (int variable = 0; variable < ModelTraits<model>::nExtraVars; variable++)
                            state.push_back(TurbulenceSettings{}.initialValue[variable]);
                        initial["default"] = state;
                    }
                    std::cout << schema.dump(4) << std::endl;
                }
                return 0;
            }

            const std::string requestedConfiguration = parser.get<std::string>("config");
            std::filesystem::path caseConfiguration =
                std::filesystem::path("../cases/acm") /
                (ModelTraits<model>::gDim == 2 ? "2D" : "3D") /
                (ModelTraits<model>::nExtraVars == 1 ? "SA" : ModelTraits<model>::nExtraVars == 2 ? "2EQ" : "") /
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
            ValidateModelConfiguration<model>(loaded.configuration);
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
