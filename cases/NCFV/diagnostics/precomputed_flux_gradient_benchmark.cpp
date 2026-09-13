/**
 * @file precomputed_flux_gradient_benchmark.cpp
 * @brief Compare cached and edge-recomputed physical-flux gradients on IV10.
 */
#include "NCFV/NCFVSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    using DNDS::index;
    using DNDS::real;
    using DNDS::NCFV::Configuration;
    using DNDS::NCFV::IntegrationMode;
    using DNDS::NCFV::RhsPhaseTiming;
    using DNDS::NCFV::Solver;

    int MaximumOpenMPThreads()
    {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    struct Sample
    {
        real rhs = 0;
        real reconstruction = 0;
        real fluxGradientCompute = 0;
        real edgeFlux = 0;
    };

    Sample GlobalMaximum(const DNDS::MPIInfo &mpi, const RhsPhaseTiming &timing)
    {
        const std::array<real, 4> local{
            timing.totalSeconds,
            timing.reconstructionSeconds,
            timing.physicalFluxGradientComputeSeconds,
            timing.edgeFluxSeconds};
        std::array<real, 4> global{};
        MPI_Allreduce(local.data(), global.data(), static_cast<int>(local.size()),
                      DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        return {global[0], global[1], global[2], global[3]};
    }

    real Median(std::vector<real> values)
    {
        DNDS_check_throw_info(!values.empty(), "Cannot take median of no samples");
        std::sort(values.begin(), values.end());
        const std::size_t middle = values.size() / 2;
        if (values.size() % 2 == 1)
            return values[middle];
        return 0.5 * (values[middle - 1] + values[middle]);
    }

    nlohmann::ordered_json Summarize(const std::vector<Sample> &samples)
    {
        std::vector<real> rhs;
        std::vector<real> reconstruction;
        std::vector<real> fluxGradientCompute;
        std::vector<real> edgeFlux;
        rhs.reserve(samples.size());
        reconstruction.reserve(samples.size());
        fluxGradientCompute.reserve(samples.size());
        edgeFlux.reserve(samples.size());
        for (const auto &sample : samples)
        {
            rhs.push_back(sample.rhs);
            reconstruction.push_back(sample.reconstruction);
            fluxGradientCompute.push_back(sample.fluxGradientCompute);
            edgeFlux.push_back(sample.edgeFlux);
        }
        return {
            {"samples", samples.size()},
            {"median_seconds", {
                {"rhs_total", Median(rhs)},
                {"reconstruction", Median(reconstruction)},
                {"physical_flux_gradient_compute", Median(fluxGradientCompute)},
                {"internal_edge_flux", Median(edgeFlux)}}},
            {"minimum_seconds", {
                {"rhs_total", *std::min_element(rhs.begin(), rhs.end())},
                {"internal_edge_flux", *std::min_element(edgeFlux.begin(), edgeFlux.end())}}},
            {"maximum_seconds", {
                {"rhs_total", *std::max_element(rhs.begin(), rhs.end())},
                {"internal_edge_flux", *std::max_element(edgeFlux.begin(), edgeFlux.end())}}}};
    }

    std::vector<real> CaptureOwnedResidual(const Solver<3> &solver)
    {
        constexpr int nVars = 5;
        std::vector<real> values(
            static_cast<std::size_t>(solver.Mesh()->NumNode()) * nVars);
        for (index iNode = 0; iNode < solver.Mesh()->NumNode(); iNode++)
            for (int iVar = 0; iVar < nVars; iVar++)
                values[static_cast<std::size_t>(iNode) * nVars + iVar] =
                    solver.ResidualField()[iNode](iVar, 0);
        return values;
    }

    std::array<real, 2> CompareResiduals(
        const DNDS::MPIInfo &mpi,
        const std::vector<real> &reference,
        const std::vector<real> &candidate)
    {
        DNDS_check_throw_info(reference.size() == candidate.size(),
                              "Residual field sizes differ");
        real localAbsolute = 0;
        real localScale = 0;
        for (std::size_t i = 0; i < reference.size(); i++)
        {
            localAbsolute = std::max(
                localAbsolute, std::abs(candidate[i] - reference[i]));
            localScale = std::max(localScale, std::abs(reference[i]));
        }
        std::array<real, 2> local{localAbsolute, localScale};
        std::array<real, 2> global{};
        MPI_Allreduce(local.data(), global.data(), static_cast<int>(local.size()),
                      DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        return {global[0], global[0] / std::max(global[1], DNDS::verySmallReal)};
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
                argc == 3,
                "Usage: precomputed_flux_gradient_benchmark 3d-configuration.json result.json");
            constexpr int warmupPairs = 2;
            constexpr int measuredPairs = 20;
            const std::filesystem::path resultPath = argv[2];
            Configuration configuration =
                DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
            DNDS_check_throw_info(
                configuration.dimension == 3,
                "Precomputed flux-gradient benchmark currently expects dimension=3");
            configuration.algorithm.mode = IntegrationMode::EfficientDifferential;
            configuration.io.writeVTK = false;
            configuration.io.writeInitial = false;
            configuration.io.writeFinal = false;
            configuration.io.outputInterval = 0;
            configuration.io.restartInput.clear();
            configuration.io.restartInterval = 0;
            configuration.io.writeFinalRestart = false;
            configuration.io.writeResolvedConfiguration = false;

            if (mpi.rank == 0)
            {
                DNDS_check_throw_info(!std::filesystem::exists(resultPath),
                                      "Refusing to overwrite result: " +
                                          resultPath.string());
                if (!resultPath.parent_path().empty())
                    std::filesystem::create_directories(resultPath.parent_path());
            }
            MPI_Barrier(mpi.comm);

            Solver<3> solver(mpi, configuration);
            solver.Initialize();
            solver.EnableDetailedFluxTiming(false);

            for (int warmup = 0; warmup < warmupPairs; warmup++)
                for (bool precomputed : {false, true})
                {
                    solver.UsePrecomputedPhysicalFluxGradients(precomputed);
                    MPI_Barrier(mpi.comm);
                    solver.EvaluateResidual();
                }

            solver.UsePrecomputedPhysicalFluxGradients(false);
            const real legacyResidualNorm = solver.EvaluateResidual();
            const std::vector<real> legacyResidual = CaptureOwnedResidual(solver);
            solver.UsePrecomputedPhysicalFluxGradients(true);
            const real precomputedResidualNorm = solver.EvaluateResidual();
            const auto residualDifference = CompareResiduals(
                mpi, legacyResidual, CaptureOwnedResidual(solver));
            DNDS_check_throw_info(
                residualDifference[1] < 5e-13,
                "Precomputed physical-flux gradients changed the residual field");

            std::vector<Sample> legacySamples;
            std::vector<Sample> precomputedSamples;
            legacySamples.reserve(measuredPairs);
            precomputedSamples.reserve(measuredPairs);
            for (int iPair = 0; iPair < measuredPairs; iPair++)
            {
                const std::array<bool, 2> order =
                    iPair % 2 == 0 ? std::array<bool, 2>{true, false}
                                   : std::array<bool, 2>{false, true};
                for (bool precomputed : order)
                {
                    solver.UsePrecomputedPhysicalFluxGradients(precomputed);
                    MPI_Barrier(mpi.comm);
                    solver.EvaluateResidual();
                    Sample sample = GlobalMaximum(mpi, solver.LastRhsTiming());
                    (precomputed ? precomputedSamples : legacySamples)
                        .push_back(sample);
                }
            }

            const auto legacy = Summarize(legacySamples);
            const auto precomputed = Summarize(precomputedSamples);
            const real legacyRhs = legacy["median_seconds"]["rhs_total"];
            const real precomputedRhs = precomputed["median_seconds"]["rhs_total"];
            const real legacyFluxWork =
                legacy["median_seconds"]["internal_edge_flux"];
            const real precomputedFluxWork =
                precomputed["median_seconds"]["physical_flux_gradient_compute"].get<real>() +
                precomputed["median_seconds"]["internal_edge_flux"].get<real>();

            nlohmann::ordered_json result = {
                {"source_configuration", argv[1]},
                {"mpi_ranks", mpi.size},
                {"omp_threads", MaximumOpenMPThreads()},
                {"detailed_edge_timing_enabled", false},
                {"warmup_pairs", warmupPairs},
                {"measured_pairs", measuredPairs},
                {"physical_flux_gradient_cache", {
                    {"scalars_per_node", 5 * 3 * 3},
                    {"bytes_per_node", 5 * 3 * 3 * sizeof(real)},
                    {"owned_nodes_on_rank_0", solver.Mesh()->NumNode()},
                    {"ghost_nodes_on_rank_0", solver.Mesh()->NumNodeGhost()}}},
                {"correctness", {
                    {"legacy_residual_norm", legacyResidualNorm},
                    {"precomputed_residual_norm", precomputedResidualNorm},
                    {"maximum_absolute_rhs_difference", residualDifference[0]},
                    {"maximum_relative_rhs_difference", residualDifference[1]}}},
                {"legacy_edge_recomputed", legacy},
                {"node_precomputed", precomputed},
                {"comparison", {
                    {"rhs_speedup", legacyRhs / precomputedRhs},
                    {"flux_work_speedup", legacyFluxWork / precomputedFluxWork},
                    {"rhs_time_reduction_fraction", 1.0 - precomputedRhs / legacyRhs},
                    {"flux_work_time_reduction_fraction",
                     1.0 - precomputedFluxWork / legacyFluxWork}}}};

            if (mpi.rank == 0)
            {
                std::ofstream output(resultPath);
                DNDS_check_throw_info(output.good(),
                                      "Cannot open result: " + resultPath.string());
                output << result.dump(2) << '\n';
                output.close();
                DNDS_check_throw_info(output.good(),
                                      "Cannot finish result: " + resultPath.string());
            }
            MPI_Barrier(mpi.comm);
        }
        catch (const std::exception &exception)
        {
            std::cerr << "NCFV precomputed-flux-gradient benchmark: "
                      << exception.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
