/**
 * @file integration_initialization_benchmark.cpp
 * @brief Single-rank timing of method-specific NCFV integration-data construction.
 *
 * The timers are embedded only around differential-weight construction or
 * quadrature-point/physical-weight construction in DualGeometry.  Mesh I/O,
 * topology, common geometric measures, reconstruction, allocation, and flow
 * initialization are executed by Solver::Initialize() but are outside every
 * reported timer.
 */
#include "NCFV/NCFVSolver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

namespace
{
    using DNDS::index;
    using DNDS::NCFV::Configuration;
    using DNDS::NCFV::IntegrationMode;

    IntegrationMode ParseMode(const std::string &name)
    {
        if (name == "efficient" || name == "EfficientDifferential")
            return IntegrationMode::EfficientDifferential;
        if (name == "traditional" || name == "TraditionalQuadrature")
            return IntegrationMode::TraditionalQuadrature;
        DNDS_check_throw_info(false, "Mode must be efficient or traditional");
        return IntegrationMode::EfficientDifferential;
    }

    std::string ModeName(IntegrationMode mode)
    {
        return mode == IntegrationMode::EfficientDifferential
                   ? "EfficientDifferential"
                   : "TraditionalQuadrature";
    }

    double EstimateTimerPairSeconds()
    {
        constexpr index repetitions = 1000000;
        double elapsedSum = 0;
        for (index repetition = 0; repetition < repetitions; repetition++)
        {
            const double start = MPI_Wtime();
            elapsedSum += MPI_Wtime() - start;
        }
        return elapsedSum / static_cast<double>(repetitions);
    }

    nlohmann::ordered_json CorrectedTiming(double rawSeconds, index calls,
                                            double timerPairSeconds)
    {
        const double estimatedOverhead = timerPairSeconds * static_cast<double>(calls);
        return {
            {"raw_seconds", rawSeconds},
            {"timed_blocks", calls},
            {"estimated_timer_overhead_seconds", estimatedOverhead},
            {"corrected_seconds", std::max(0.0, rawSeconds - estimatedOverhead)}};
    }

    template <int dimension>
    nlohmann::ordered_json Run(
        const DNDS::MPIInfo &mpi,
        const Configuration &configuration,
        const std::string &sourceConfiguration)
    {
        DNDS::NCFV::Solver<dimension> solver(mpi, configuration);
        solver.Initialize();

        const auto &geometry = solver.Geometry();
        const auto &profile = geometry.IntegrationInitializationProfile();
        const auto &topology = solver.EdgeTopology();
        const auto &mesh = solver.Mesh();

        index volumeQuadraturePoints = 0;
        index internalSurfaceQuadraturePoints = 0;
        index boundarySurfaceQuadraturePoints = 0;
        index volumeRecoveryWeightEntries = 0;
        index internalSurfaceStateWeightEntries = 0;
        index internalSurfaceFluxWeightEntries = 0;
        index internalSurfaceGradientWeightEntries = 0;
        index boundaryGradientWeightEntries = 0;
        index internalSurfaceUniqueNodeEntries = 0;
        index boundaryUniqueNodeEntries = 0;

        for (index iNode = 0; iNode < mesh->NumNode(); iNode++)
        {
            const auto &volume = geometry.NodeVolume(iNode);
            volumeQuadraturePoints += static_cast<index>(volume.volumeQuadrature.size());
            volumeRecoveryWeightEntries +=
                static_cast<index>(volume.pointRecoveryStencil.size());
            for (const auto &piece : volume.boundaryPieces)
            {
                boundarySurfaceQuadraturePoints +=
                    static_cast<index>(piece.quadrature.size());
                boundaryGradientWeightEntries +=
                    static_cast<index>(piece.efficientStencil.size());
                boundaryUniqueNodeEntries +=
                    static_cast<index>(piece.efficientStencil.size());
            }
        }
        for (index iEdge = 0; iEdge < topology.NumEdge(); iEdge++)
        {
            const auto &surface = geometry.EdgeSurface(iEdge);
            internalSurfaceQuadraturePoints +=
                static_cast<index>(surface.quadrature.size());
            internalSurfaceUniqueNodeEntries +=
                static_cast<index>(surface.efficientStencil.size());
            for (const DNDS::NCFV::EfficientSurfaceNode &entry : surface.efficientStencil)
            {
                internalSurfaceGradientWeightEntries +=
                    std::abs(entry.gradientWeight) > DNDS::verySmallReal;
                for (int side = 0; side < 2; side++)
                {
                    internalSurfaceStateWeightEntries +=
                        entry.stateGradientWeights[static_cast<std::size_t>(side)]
                                .squaredNorm() >
                            DNDS::sqr(DNDS::verySmallReal);
                    internalSurfaceFluxWeightEntries +=
                        entry.fluxGradientWeights[static_cast<std::size_t>(side)]
                                .squaredNorm() >
                            DNDS::sqr(DNDS::verySmallReal);
                }
            }
        }

        const double timerPairSeconds = EstimateTimerPairSeconds();
        const auto volume = CorrectedTiming(
            profile.volumeSeconds, profile.volumeCalls, timerPairSeconds);
        const auto internalSurface = CorrectedTiming(
            profile.internalSurfaceSeconds, profile.internalSurfaceCalls,
            timerPairSeconds);
        const auto boundarySurface = CorrectedTiming(
            profile.boundarySurfaceSeconds, profile.boundarySurfaceCalls,
            timerPairSeconds);
        const auto normalization = CorrectedTiming(
            profile.volumeNormalizationSeconds,
            profile.volumeNormalizationCalls, timerPairSeconds);
        const double rawTotal = profile.TotalSeconds();
        const index totalCalls = profile.TimedBlockCalls();

        return {
            {"schema_version", 1},
            {"source_configuration", sourceConfiguration},
            {"mesh_file", configuration.mesh.meshFile},
            {"mode", ModeName(configuration.algorithm.mode)},
            {"mpi_ranks", mpi.size},
            {"omp_threads_per_rank", 1},
            {"nodes", mesh->NumNodeGlobal()},
            {"cells", mesh->NumCellGlobal()},
            {"edges", topology.NumEdgeGlobal()},
            {"quadrature_order", configuration.algorithm.quadratureOrder},
            {"timer_pair_seconds", timerPairSeconds},
            {"timing",
             {{"volume", volume},
              {"internal_surface", internalSurface},
              {"boundary_surface", boundarySurface},
              {"volume_weight_normalization", normalization},
              {"total", CorrectedTiming(rawTotal, totalCalls, timerPairSeconds)}}},
            {"stored_data",
             {{"volume_quadrature_points", volumeQuadraturePoints},
              {"internal_surface_quadrature_points", internalSurfaceQuadraturePoints},
              {"boundary_surface_quadrature_points", boundarySurfaceQuadraturePoints},
              {"volume_recovery_weight_entries", volumeRecoveryWeightEntries},
              {"internal_surface_state_weight_entries", internalSurfaceStateWeightEntries},
              {"internal_surface_flux_weight_entries", internalSurfaceFluxWeightEntries},
              {"internal_surface_gradient_weight_entries", internalSurfaceGradientWeightEntries},
              {"boundary_gradient_weight_entries", boundaryGradientWeightEntries},
              {"internal_surface_unique_node_entries", internalSurfaceUniqueNodeEntries},
              {"boundary_unique_node_entries", boundaryUniqueNodeEntries}}},
            {"timing_scope",
             "Only method-specific DualGeometry integration-data construction is timed. Efficient mode includes differential weight generation/merging and recovery-weight normalization. Traditional mode includes reference-rule construction plus physical Gauss-point and physical-weight generation/storage. Mesh I/O, topology, common geometry, quadrature moment accumulation, reconstruction, allocation, and flow-field initialization are excluded."}};
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
                argc == 4,
                "Usage: integration_initialization_benchmark config mode output.json");
            DNDS_check_throw_info(mpi.size == 1,
                                  "Integration initialization benchmark requires one MPI rank");

            const std::string sourceConfiguration = argv[1];
            Configuration configuration =
                DNDS::NCFV::LoadConfiguration(sourceConfiguration, {}, {}).configuration;
            configuration.algorithm.mode = ParseMode(argv[2]);
            configuration.algorithm.retainMicroGeometry = false;
            configuration.algorithm.profileIntegrationInitialization = true;
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
                result = Run<2>(mpi, configuration, sourceConfiguration);
            else if (configuration.dimension == 3)
                result = Run<3>(mpi, configuration, sourceConfiguration);
            else
                DNDS_check_throw_info(false, "Benchmark dimension must be 2 or 3");

            if (mpi.rank == 0)
            {
                const std::filesystem::path output = argv[3];
                DNDS_check_throw_info(!std::filesystem::exists(output),
                                      "Refusing to overwrite " + output.string());
                if (!output.parent_path().empty())
                    std::filesystem::create_directories(output.parent_path());
                std::ofstream stream(output);
                DNDS_check_throw_info(stream.good(), "Cannot open " + output.string());
                stream << result.dump(2) << '\n';
                DNDS_check_throw_info(stream.good(), "Cannot finish " + output.string());
                DNDS::log() << "NCFV integration initialization benchmark complete: "
                            << result.dump() << std::endl;
            }
        }
        catch (const std::exception &exception)
        {
            std::cerr << "NCFV integration initialization benchmark: "
                      << exception.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
