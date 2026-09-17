/**
 * @file euler.cpp
 * @brief Unified run-time entry point for all DNDSR flow solvers.
 */
#include "UnifiedSolverDispatch.hpp"

#include "DNDS/Config/SolverSelection.hpp"
#include "DNDS/MPI.hpp"

#include <argparse.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    struct LaunchArguments
    {
        std::string configurationPath;
        std::vector<std::string> overwriteKeys;
        std::vector<std::string> overwriteValues;
        bool emitSchema = false;
        bool listSolvers = false;
    };

    LaunchArguments ParseLaunchArguments(int argc, char *argv[])
    {
        argparse::ArgumentParser parser("euler", DNDS_VERSION_STRING);
        parser.add_description(
            "Unified DNDSR flow solver; solver family and model are selected by JSON");
        parser.add_argument("config").default_value("");
        parser.add_argument("-k", "--overwrite_key")
            .append()
            .default_value<std::vector<std::string>>({});
        parser.add_argument("-v", "--overwrite_value")
            .append()
            .default_value<std::vector<std::string>>({});
        parser.add_argument("--debug").flag().default_value(false);
        parser.add_argument("--emit-schema").flag().default_value(false);
        parser.add_argument("--list-solvers").flag().default_value(false);
        parser.parse_args(argc, argv);

        LaunchArguments arguments;
        arguments.configurationPath = parser.get<std::string>("config");
        arguments.overwriteKeys =
            parser.get<std::vector<std::string>>("--overwrite_key");
        arguments.overwriteValues =
            parser.get<std::vector<std::string>>("--overwrite_value");
        arguments.emitSchema = parser.get<bool>("--emit-schema");
        arguments.listSolvers = parser.get<bool>("--list-solvers");
        if (arguments.overwriteKeys.size() != arguments.overwriteValues.size())
            throw std::runtime_error("overwrite keys and values do not match");
        return arguments;
    }

    DNDS::SolverSelection LoadSolverSelection(const LaunchArguments &arguments)
    {
        if (arguments.configurationPath.empty())
            return {};

        std::ifstream input(arguments.configurationPath);
        if (!input.good())
            throw std::runtime_error(
                "solver configuration file does not exist: " +
                arguments.configurationPath);
        auto document = nlohmann::ordered_json::parse(input, nullptr, true, true);

        for (std::size_t i = 0; i < arguments.overwriteKeys.size(); ++i)
        {
            const nlohmann::ordered_json::json_pointer key(arguments.overwriteKeys[i]);
            try
            {
                document[key] =
                    nlohmann::ordered_json::parse(arguments.overwriteValues[i]);
            }
            catch (const nlohmann::ordered_json::parse_error &)
            {
                document[key] = arguments.overwriteValues[i];
            }
        }

        if (!document.contains("solver"))
            throw std::runtime_error(
                "configuration is missing the required top-level 'solver' object");
        DNDS::SolverSelection selection =
            document.at("solver").get<DNDS::SolverSelection>();
        for (const auto &failure : selection.validate())
            if (!failure.passed)
                throw std::runtime_error(failure.message);
        return selection;
    }

    int RunEulerModel(const DNDS::SolverSelection &selection, int argc, char *argv[])
    {
        if (selection.model == "NS")
            return DNDS::App::RunEulerNS(argc, argv);
        if (selection.model == "NS_2D")
            return DNDS::App::RunEulerNS2D(argc, argv);
        if (selection.model == "NS_3D")
            return DNDS::App::RunEulerNS3D(argc, argv);
        if (selection.model == "NS_SA")
            return DNDS::App::RunEulerNSSA(argc, argv);
        if (selection.model == "NS_SA_3D")
            return DNDS::App::RunEulerNSSA3D(argc, argv);
        if (selection.model == "NS_2EQ")
            return DNDS::App::RunEulerNS2EQ(argc, argv);
        if (selection.model == "NS_2EQ_3D")
            return DNDS::App::RunEulerNS2EQ3D(argc, argv);
        if (selection.model == "NS_EX")
            return DNDS::App::RunEulerNSEX(argc, argv, selection.fieldNVariables);
        if (selection.model == "NS_EX_3D")
            return DNDS::App::RunEulerNSEX3D(argc, argv, selection.fieldNVariables);
        throw std::runtime_error("unknown Euler solver.model: " + selection.model);
    }

    int RunSelectedSolver(
        const DNDS::SolverSelection &selection, int argc, char *argv[])
    {
        if (selection.discretization == "NCFV")
        {
            if (selection.type != "Euler")
                throw std::runtime_error(
                    "NCFV currently supports solver.type='Euler' only");
            return DNDS::App::RunNCFVIdealGas(argc, argv);
        }

        if (selection.discretization != "CFV")
            throw std::runtime_error(
                "unknown solver.discretization: " + selection.discretization);
        if (selection.type == "Euler")
            return RunEulerModel(selection, argc, argv);
        if (selection.type == "ACM")
        {
            if (selection.model == "ConstantDensity2D")
                return DNDS::App::RunACMConstantDensity2D(argc, argv);
            if (selection.model == "ConstantDensity3D")
                return DNDS::App::RunACMConstantDensity3D(argc, argv);
            throw std::runtime_error("unknown ACM solver.model: " + selection.model);
        }
        if (selection.type == "ACMVariable")
        {
            if (selection.model == "VariableDensity2D")
                return DNDS::App::RunACMVariableDensity2D(argc, argv);
            if (selection.model == "VariableDensity3D")
                return DNDS::App::RunACMVariableDensity3D(argc, argv);
            throw std::runtime_error(
                "unknown ACMVariable solver.model: " + selection.model);
        }
        throw std::runtime_error("unknown solver.type: " + selection.type);
    }

    void PrintSolverList()
    {
        std::cout
            << "Euler/CFV models: NS, NS_2D, NS_3D, NS_SA, NS_SA_3D, "
               "NS_2EQ, NS_2EQ_3D, NS_EX, NS_EX_3D\n"
            << "ACM/CFV models: ConstantDensity2D, ConstantDensity3D\n"
            << "ACMVariable/CFV models: VariableDensity2D, VariableDensity3D\n"
            << "Euler/NCFV model: IdealGas (dimension is selected by JSON)\n";
    }
}

int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    int errorCode = 0;
    try
    {
        const LaunchArguments arguments = ParseLaunchArguments(argc, argv);
        if (arguments.listSolvers)
        {
            if (mpi.rank == 0)
                PrintSolverList();
        }
        else
        {
            const DNDS::SolverSelection selection = LoadSolverSelection(arguments);
            if (mpi.rank == 0 && !arguments.emitSchema)
                DNDS::log() << "Unified solver dispatch: type=" << selection.type
                            << ", discretization=" << selection.discretization
                            << ", model=" << selection.model << std::endl;
            errorCode = RunSelectedSolver(selection, argc, argv);
        }
    }
    catch (const std::exception &exception)
    {
        if (mpi.rank == 0)
            std::cerr << "DNDS unified solver error: " << exception.what() << std::endl;
        errorCode = 1;
    }

    if (errorCode != 0)
        MPI_Abort(MPI_COMM_WORLD, errorCode);
    MPI_Finalize();
    return errorCode;
}
