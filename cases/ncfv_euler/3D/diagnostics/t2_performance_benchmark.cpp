/**
 * @file t2_performance_benchmark.cpp
 * @brief Time-to-physical-t=2 and process-memory benchmark for NCFV modes.
 *
 * The probe deliberately calls the production Solver::Run() path.  It therefore
 * includes the SSPRK3 vector updates, all three RHS evaluations per physical
 * step, CFL selection and the final shortened step.  Output/restart I/O is
 * disabled so the reported time is solver computation rather than filesystem
 * throughput.
 */
#include "NCFV/NCFVSolver.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace
{
    using DNDS::index;
    using DNDS::NCFV::Configuration;
    using DNDS::NCFV::IntegrationMode;

    struct ProcessMemory
    {
        index rssKiB = 0;
        index hwmKiB = 0;
        index pssKiB = 0;
    };

    struct AggregatedMemory
    {
        ProcessMemory sum;
        ProcessMemory maximum;
    };

    index ReadProcValueKiB(const std::filesystem::path &path,
                           const std::string &field)
    {
        std::ifstream input(path);
        DNDS_check_throw_info(input.good(), "Cannot read " + path.string());
        const std::string wanted = field + ':';
        std::string line;
        while (std::getline(input, line))
        {
            std::istringstream parser(line);
            std::string name, unit;
            index value = -1;
            parser >> name;
            if (name != wanted)
                continue;
            parser >> value >> unit;
            DNDS_check_throw_info((parser.good() || parser.eof()) && value >= 0 && unit == "kB",
                                  "Cannot parse " + field + " from " + path.string());
            return value;
        }
        DNDS_check_throw_info(false, "Cannot find " + field + " in " + path.string());
        return 0;
    }

    ProcessMemory SampleLocalMemory()
    {
        return {ReadProcValueKiB("/proc/self/status", "VmRSS"),
                ReadProcValueKiB("/proc/self/status", "VmHWM"),
                ReadProcValueKiB("/proc/self/smaps_rollup", "Pss")};
    }

    AggregatedMemory SampleMemory(const DNDS::MPIInfo &mpi)
    {
        const ProcessMemory local = SampleLocalMemory();
        const std::array<index, 3> localValues{local.rssKiB, local.hwmKiB, local.pssKiB};
        std::array<index, 3> sums{}, maxima{};
        MPI_Allreduce(localValues.data(), sums.data(), static_cast<int>(localValues.size()),
                      DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        MPI_Allreduce(localValues.data(), maxima.data(), static_cast<int>(localValues.size()),
                      DNDS::DNDS_MPI_INDEX, MPI_MAX, mpi.comm);
        return {{sums[0], sums[1], sums[2]}, {maxima[0], maxima[1], maxima[2]}};
    }

    nlohmann::ordered_json Quantity(index sumKiB, index maximumKiB)
    {
        return {{"sum_kib", sumKiB},
                {"max_kib", maximumKiB},
                {"sum_mib", static_cast<double>(sumKiB) / 1024.0},
                {"max_mib", static_cast<double>(maximumKiB) / 1024.0}};
    }

    nlohmann::ordered_json MemoryToJson(const AggregatedMemory &memory)
    {
        return {{"rss", Quantity(memory.sum.rssKiB, memory.maximum.rssKiB)},
                {"hwm", Quantity(memory.sum.hwmKiB, memory.maximum.hwmKiB)},
                {"pss", Quantity(memory.sum.pssKiB, memory.maximum.pssKiB)}};
    }

    IntegrationMode ParseMode(const std::string &name)
    {
        if (name == "EfficientDifferential" || name == "efficient")
            return IntegrationMode::EfficientDifferential;
        if (name == "TraditionalQuadrature" || name == "traditional")
            return IntegrationMode::TraditionalQuadrature;
        DNDS_check_throw_info(false, "mode must be efficient or traditional");
        return IntegrationMode::EfficientDifferential;
    }

    std::string ModeName(IntegrationMode mode)
    {
        return mode == IntegrationMode::EfficientDifferential ? "EfficientDifferential"
                                                               : "TraditionalQuadrature";
    }

    template <int dimension>
    nlohmann::ordered_json RunToEndTime(const DNDS::MPIInfo &mpi,
                                        const Configuration &configuration,
                                        const std::string &sourceConfiguration,
                                        const std::string &checkpoint)
    {
        DNDS::NCFV::Solver<dimension> solver(mpi, configuration);
        const auto beforeInitialize = SampleMemory(mpi);

        MPI_Barrier(mpi.comm);
        const double initializeStart = MPI_Wtime();
        solver.Initialize();
        const double localInitializeSeconds = MPI_Wtime() - initializeStart;
        double initializeSeconds = 0;
        MPI_Allreduce(&localInitializeSeconds, &initializeSeconds, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        const auto afterInitialize = SampleMemory(mpi);

        MPI_Barrier(mpi.comm);
        const double marchStart = MPI_Wtime();
        solver.Run();
        const double localMarchSeconds = MPI_Wtime() - marchStart;
        double marchSeconds = 0;
        MPI_Allreduce(&localMarchSeconds, &marchSeconds, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        const auto afterTime2 = SampleMemory(mpi);

        DNDS_check_throw_info(std::abs(solver.SimulationTime() - configuration.time.endTime) < 1e-12,
                              "Solver did not reach requested physical end time");
        DNDS_check_throw_info(solver.CurrentIteration() > 0, "Solver made no physical time steps");

        const auto mesh = solver.Mesh();
        return {{"schema_version", 1},
                {"source_configuration", sourceConfiguration},
                {"mesh_file", configuration.mesh.meshFile},
                {"mode", ModeName(configuration.algorithm.mode)},
                {"mpi_ranks", mpi.size},
                {"nodes", mesh->NumNodeGlobal()},
                {"cells", mesh->NumCellGlobal()},
                {"end_time", solver.SimulationTime()},
                {"physical_steps", solver.CurrentIteration()},
                {"initialize_seconds", initializeSeconds},
                {"time_march_seconds", marchSeconds},
                {"total_seconds", initializeSeconds + marchSeconds},
                {"memory", {{"pre_initialize", MemoryToJson(beforeInitialize)},
                            {"after_initialize", MemoryToJson(afterInitialize)},
                            {"after_time_2", MemoryToJson(afterTime2)}}},
                {"checkpoint_prefix", checkpoint},
                {"checkpoint_base", checkpoint.empty() ? "" : checkpoint + "_" + fmt::format("{:08d}", solver.CurrentIteration())},
                {"measurement_scope", checkpoint.empty() ? "production Solver::Run() to t=endTime; solver output and restart I/O disabled" : "production Solver::Run() to t=endTime; final checkpoint I/O included solely for segmented continuation"}};
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        try
        {
            DNDS_check_throw_info(argc == 7,
                                  "Usage: t2_performance_benchmark config.json efficient|traditional result.json endTime restartInput checkpointBase");
            auto configuration = DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
            configuration.algorithm.mode = ParseMode(argv[2]);
            configuration.time.iterations = 1000000;
            configuration.time.endTime = std::stod(argv[4]);
            configuration.time.useCFLTimeStep = true;
            configuration.time.useLocalTimeStep = false;
            configuration.time.maximumTimeStep = 1e30;
            configuration.time.minimumTimeStep = 1e-30;
            configuration.time.reportInterval = 1000000;
            configuration.io.writeVTK = false;
            configuration.io.writeInitial = false;
            configuration.io.writeFinal = false;
            configuration.io.outputInterval = 0;
            configuration.io.writeFinalRestart = false;
            configuration.io.restartInterval = 0;
            configuration.io.writeResolvedConfiguration = false;
            configuration.io.restartInput = argv[5];
            configuration.io.writeFinalRestart = argv[6][0] != '\0';
            configuration.io.restartPrefix = argv[6];

            nlohmann::ordered_json result;
            if (configuration.dimension == 3)
                result = RunToEndTime<3>(mpi, configuration, argv[1], argv[6]);
            else if (configuration.dimension == 2)
                result = RunToEndTime<2>(mpi, configuration, argv[1], argv[6]);
            else
                DNDS_check_throw_info(false, "Unsupported dimension");

            if (mpi.rank == 0)
            {
                const std::filesystem::path output(argv[3]);
                DNDS_check_throw_info(!std::filesystem::exists(output),
                                      "Refusing to overwrite result " + output.string());
                std::ofstream stream(output);
                DNDS_check_throw_info(stream.good(), "Cannot write " + output.string());
                stream << result.dump(2) << '\n';
                DNDS_check_throw_info(stream.good(), "Cannot finish " + output.string());
                DNDS::log() << "t=2 performance complete: " << result.dump() << std::endl;
            }
        }
        catch (const std::exception &exception)
        {
            std::cerr << "NCFV t=2 performance benchmark: " << exception.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
