/**
 * @file performance_benchmark.cpp
 * @brief Initialization, residual, and process-memory benchmark for both NCFV modes.
 *
 * This diagnostic deliberately does not call Solver::Run().  It disables all
 * solver output, initializes one of the existing IV configurations, performs
 * one untimed-state-changing warm-up residual evaluation, and then times
 * repeated evaluations of the same spatial residual.
 */
#include "NCFV/NCFVSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

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
            std::string name;
            parser >> name;
            if (name != wanted)
                continue;

            index value = -1;
            std::string unit;
            parser >> value >> unit;
            DNDS_check_throw_info(parser.good() || parser.eof(),
                                  "Cannot parse " + field + " from " + path.string());
            DNDS_check_throw_info(value >= 0 && unit == "kB",
                                  "Unexpected unit for " + field + " in " + path.string());
            return value;
        }
        DNDS_check_throw_info(false,
                              "Cannot find " + field + " in " + path.string());
        return 0;
    }

    ProcessMemory SampleLocalMemory()
    {
        return {
            ReadProcValueKiB("/proc/self/status", "VmRSS"),
            ReadProcValueKiB("/proc/self/status", "VmHWM"),
            ReadProcValueKiB("/proc/self/smaps_rollup", "Pss")};
    }

    AggregatedMemory SampleMemory(const DNDS::MPIInfo &mpi)
    {
        const ProcessMemory local = SampleLocalMemory();
        const std::array<index, 3> values{local.rssKiB, local.hwmKiB, local.pssKiB};
        std::array<index, 3> sums{}, maxima{};
        MPI_Allreduce(values.data(), sums.data(), static_cast<int>(values.size()),
                      DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        MPI_Allreduce(values.data(), maxima.data(), static_cast<int>(values.size()),
                      DNDS::DNDS_MPI_INDEX, MPI_MAX, mpi.comm);
        return {{sums[0], sums[1], sums[2]},
                {maxima[0], maxima[1], maxima[2]}};
    }

    nlohmann::ordered_json MemoryQuantity(index sumKiB, index maxKiB)
    {
        constexpr double kiBPerMiB = 1024.0;
        return {{"sum_kib", sumKiB},
                {"max_kib", maxKiB},
                {"sum_mib", static_cast<double>(sumKiB) / kiBPerMiB},
                {"max_mib", static_cast<double>(maxKiB) / kiBPerMiB}};
    }

    nlohmann::ordered_json MemoryToJson(const AggregatedMemory &memory)
    {
        return {
            {"rss", MemoryQuantity(memory.sum.rssKiB, memory.maximum.rssKiB)},
            {"hwm", MemoryQuantity(memory.sum.hwmKiB, memory.maximum.hwmKiB)},
            {"pss", MemoryQuantity(memory.sum.pssKiB, memory.maximum.pssKiB)}};
    }

    IntegrationMode ParseMode(const std::string &input)
    {
        if (input == "EfficientDifferential" || input == "efficient")
            return IntegrationMode::EfficientDifferential;
        if (input == "TraditionalQuadrature" || input == "traditional")
            return IntegrationMode::TraditionalQuadrature;
        DNDS_check_throw_info(
            false,
            "mode must be EfficientDifferential/efficient or TraditionalQuadrature/traditional");
        return IntegrationMode::EfficientDifferential;
    }

    std::string ModeName(IntegrationMode mode)
    {
        return mode == IntegrationMode::EfficientDifferential
                   ? "EfficientDifferential"
                   : "TraditionalQuadrature";
    }

    std::string MeshLabel(const Configuration &configuration,
                          const std::string &sourceConfiguration)
    {
        const std::regex pattern("iv[0-9]+", std::regex::icase);
        std::smatch match;
        if (std::regex_search(configuration.mesh.meshFile, match, pattern))
            return match.str();
        if (std::regex_search(sourceConfiguration, match, pattern))
            return match.str();
        return std::filesystem::path(configuration.mesh.meshFile).stem().string();
    }

    nlohmann::ordered_json TimingSummary(const std::vector<double> &samples)
    {
        DNDS_check_throw_info(!samples.empty(), "Timing sample list is empty");
        const auto extrema = std::minmax_element(samples.begin(), samples.end());
        const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                            static_cast<double>(samples.size());
        double squaredDifference = 0;
        for (double sample : samples)
            squaredDifference += (sample - mean) * (sample - mean);

        std::vector<double> ordered = samples;
        std::sort(ordered.begin(), ordered.end());
        const std::size_t middle = ordered.size() / 2;
        const double median = ordered.size() % 2 == 0
                                  ? 0.5 * (ordered[middle - 1] + ordered[middle])
                                  : ordered[middle];
        return {{"minimum_seconds", *extrema.first},
                {"maximum_seconds", *extrema.second},
                {"mean_seconds", mean},
                {"median_seconds", median},
                {"population_stddev_seconds",
                 std::sqrt(squaredDifference / static_cast<double>(samples.size()))}};
    }

    nlohmann::ordered_json AddMemoryAliases(
        nlohmann::ordered_json memory,
        const AggregatedMemory &finalMemory)
    {
        constexpr double kiBPerMiB = 1024.0;
        memory["rss_sum_mib"] = static_cast<double>(finalMemory.sum.rssKiB) / kiBPerMiB;
        memory["rss_max_mib"] = static_cast<double>(finalMemory.maximum.rssKiB) / kiBPerMiB;
        memory["pss_sum_mib"] = static_cast<double>(finalMemory.sum.pssKiB) / kiBPerMiB;
        memory["pss_max_mib"] = static_cast<double>(finalMemory.maximum.pssKiB) / kiBPerMiB;
        memory["hwm_sum_mib"] = static_cast<double>(finalMemory.sum.hwmKiB) / kiBPerMiB;
        memory["hwm_max_mib"] = static_cast<double>(finalMemory.maximum.hwmKiB) / kiBPerMiB;
        return memory;
    }

    template <int dimension>
    nlohmann::ordered_json RunBenchmark(
        const DNDS::MPIInfo &mpi,
        const Configuration &configuration,
        const std::string &sourceConfiguration,
        int repetitions)
    {
        std::vector<double> localCallSeconds(static_cast<std::size_t>(repetitions));
        std::vector<double> maximumCallSeconds(static_cast<std::size_t>(repetitions));

        DNDS::NCFV::Solver<dimension> solver(mpi, configuration);
        const AggregatedMemory beforeInitialize = SampleMemory(mpi);

        MPI_Barrier(mpi.comm);
        const double initializeStart = MPI_Wtime();
        solver.Initialize();
        const double localInitializeSeconds = MPI_Wtime() - initializeStart;
        double initializeSeconds = 0;
        MPI_Allreduce(&localInitializeSeconds, &initializeSeconds, 1,
                      MPI_DOUBLE, MPI_MAX, mpi.comm);
        const AggregatedMemory afterInitialize = SampleMemory(mpi);

        const auto mesh = solver.Mesh();
        const auto &topology = solver.EdgeTopology();
        const auto &geometry = solver.Geometry();
        index localQuadrature[3]{};
        for (index iNode = 0; iNode < mesh->NumNode(); ++iNode)
        {
            const auto &volume = geometry.NodeVolume(iNode);
            localQuadrature[0] += static_cast<index>(volume.volumeQuadrature.size());
            for (const auto &piece : volume.boundaryPieces)
                localQuadrature[2] += static_cast<index>(piece.quadrature.size());
        }
        for (index iEdge = 0; iEdge < topology.NumEdge(); ++iEdge)
            localQuadrature[1] +=
                static_cast<index>(geometry.EdgeSurface(iEdge).quadrature.size());
        index globalQuadrature[3]{};
        MPI_Allreduce(localQuadrature, globalQuadrature, 3,
                      DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);

        MPI_Barrier(mpi.comm);
        const double warmupStart = MPI_Wtime();
        const double warmupResidual = solver.EvaluateResidual();
        const double localWarmupSeconds = MPI_Wtime() - warmupStart;
        double warmupSeconds = 0;
        MPI_Allreduce(&localWarmupSeconds, &warmupSeconds, 1,
                      MPI_DOUBLE, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(std::isfinite(warmupResidual),
                              "Warm-up residual is not finite");
        const AggregatedMemory afterWarmup = SampleMemory(mpi);

        MPI_Barrier(mpi.comm);
        const double totalStart = MPI_Wtime();
        double lastResidual = warmupResidual;
        for (int repetition = 0; repetition < repetitions; ++repetition)
        {
            const double callStart = MPI_Wtime();
            lastResidual = solver.EvaluateResidual();
            localCallSeconds[static_cast<std::size_t>(repetition)] =
                MPI_Wtime() - callStart;
        }
        const double localTotalSeconds = MPI_Wtime() - totalStart;
        double totalSeconds = 0;
        MPI_Allreduce(&localTotalSeconds, &totalSeconds, 1,
                      MPI_DOUBLE, MPI_MAX, mpi.comm);
        MPI_Allreduce(localCallSeconds.data(), maximumCallSeconds.data(), repetitions,
                      MPI_DOUBLE, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(std::isfinite(lastResidual),
                              "Measured residual is not finite");
        const AggregatedMemory afterMeasurement = SampleMemory(mpi);

        double warmupResidualMinimum = 0, warmupResidualMaximum = 0;
        double lastResidualMinimum = 0, lastResidualMaximum = 0;
        MPI_Allreduce(&warmupResidual, &warmupResidualMinimum, 1,
                      MPI_DOUBLE, MPI_MIN, mpi.comm);
        MPI_Allreduce(&warmupResidual, &warmupResidualMaximum, 1,
                      MPI_DOUBLE, MPI_MAX, mpi.comm);
        MPI_Allreduce(&lastResidual, &lastResidualMinimum, 1,
                      MPI_DOUBLE, MPI_MIN, mpi.comm);
        MPI_Allreduce(&lastResidual, &lastResidualMaximum, 1,
                      MPI_DOUBLE, MPI_MAX, mpi.comm);

        nlohmann::ordered_json memory = {
            {"pre_initialize", MemoryToJson(beforeInitialize)},
            {"after_initialize", MemoryToJson(afterInitialize)},
            {"after_warmup", MemoryToJson(afterWarmup)},
            {"after_measurement", MemoryToJson(afterMeasurement)}};
        memory = AddMemoryAliases(std::move(memory), afterMeasurement);

        const index totalQuadrature =
            globalQuadrature[0] + globalQuadrature[1] + globalQuadrature[2];
        return {
            {"schema_version", 1},
            {"source_configuration", sourceConfiguration},
            {"mesh", MeshLabel(configuration, sourceConfiguration)},
            {"mesh_file", configuration.mesh.meshFile},
            {"mode", ModeName(configuration.algorithm.mode)},
            {"mpi_ranks", mpi.size},
            {"repetitions", repetitions},
            {"nodes", mesh->NumNodeGlobal()},
            {"cells", mesh->NumCellGlobal()},
            {"edges", topology.NumEdgeGlobal()},
            {"initialize_seconds", initializeSeconds},
            {"warmup_seconds", warmupSeconds},
            {"rhs_total_seconds", totalSeconds},
            {"rhs_seconds_per_call", maximumCallSeconds},
            {"rhs_summary", TimingSummary(maximumCallSeconds)},
            {"residual",
             {{"warmup", warmupResidual},
              {"warmup_rank_min", warmupResidualMinimum},
              {"warmup_rank_max", warmupResidualMaximum},
              {"last", lastResidual},
              {"last_rank_min", lastResidualMinimum},
              {"last_rank_max", lastResidualMaximum}}},
            {"quadrature_points",
             {{"volume", globalQuadrature[0]},
              {"surface", globalQuadrature[1]},
              {"boundary", globalQuadrature[2]},
              {"total", totalQuadrature}}},
            {"memory", std::move(memory)},
            {"timing_scope",
             "Initialize is timed after a rank barrier; RHS samples exclude one warm-up and report the maximum elapsed time across MPI ranks."}};
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
            DNDS_check_throw_info(
                argc == 5,
                "Usage: performance_benchmark config mode repetitions output.json");
            const std::string sourceConfiguration = argv[1];
            Configuration configuration =
                DNDS::NCFV::LoadConfiguration(sourceConfiguration, {}, {}).configuration;
            configuration.algorithm.mode = ParseMode(argv[2]);
            configuration.algorithm.retainMicroGeometry = false;

            const long long repetitionsLong = std::stoll(argv[3]);
            DNDS_check_throw_info(
                repetitionsLong > 0 &&
                    repetitionsLong <= std::numeric_limits<int>::max(),
                "repetitions must be in [1, INT_MAX]");
            const int repetitions = static_cast<int>(repetitionsLong);

            // Keep mesh/physics/reconstruction settings from the IV case, while
            // preventing every production-output path from affecting the benchmark.
            configuration.time.iterations = 0;
            configuration.io.writeVTK = false;
            configuration.io.writeInitial = false;
            configuration.io.writeFinal = false;
            configuration.io.outputInterval = 0;
            configuration.io.restartInput.clear();
            configuration.io.restartInterval = 0;
            configuration.io.writeFinalRestart = false;
            configuration.io.writeResolvedConfiguration = false;

            nlohmann::ordered_json result;
            if (configuration.dimension == 2)
                result = RunBenchmark<2>(mpi, configuration,
                                         sourceConfiguration, repetitions);
            else if (configuration.dimension == 3)
                result = RunBenchmark<3>(mpi, configuration,
                                         sourceConfiguration, repetitions);
            else
                DNDS_check_throw_info(false, "NCFV benchmark dimension must be 2 or 3");

            if (mpi.rank == 0)
            {
                const std::filesystem::path output = argv[4];
                DNDS_check_throw_info(!std::filesystem::exists(output),
                                      "Refusing to overwrite benchmark output: " +
                                          output.string());
                if (!output.parent_path().empty())
                    std::filesystem::create_directories(output.parent_path());
                std::ofstream file(output);
                DNDS_check_throw_info(file.good(),
                                      "Cannot open benchmark output: " + output.string());
                file << result.dump(2) << '\n';
                file.close();
                DNDS_check_throw_info(file.good(),
                                      "Cannot finish benchmark output: " + output.string());
                DNDS::log() << "NCFV PERFORMANCE BENCHMARK COMPLETE: mode="
                            << result["mode"] << ", mesh=" << result["mesh"]
                            << ", ranks=" << mpi.size
                            << ", initialize_seconds=" << result["initialize_seconds"]
                            << ", rhs_mean_seconds="
                            << result["rhs_summary"]["mean_seconds"] << std::endl;
            }
            MPI_Barrier(mpi.comm);
        }
        catch (const std::exception &exception)
        {
            std::cerr << "NCFV performance benchmark: "
                      << exception.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
