/** Independent one-RHS audit of the production shared-gradient face sweep. */
#include "NCFV/NCFVSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

using DNDS::DNDS_MPI_INDEX;
using DNDS::DNDS_MPI_REAL;

namespace
{
    using Json = nlohmann::ordered_json;
    using State = Eigen::Matrix<DNDS::real, 5, 1>;
    using Gradient = Eigen::Matrix<DNDS::real, 3, 5>;

    struct PhysicalStatistics
    {
        DNDS::index samples = 0, nonfinite = 0, nonpositiveDensity = 0, nonpositivePressure = 0;
        DNDS::real minimumDensity = std::numeric_limits<DNDS::real>::max();
        DNDS::real minimumPressure = std::numeric_limits<DNDS::real>::max();

        void Add(const State &state, DNDS::real gamma)
        {
            samples++;
            nonfinite += !state.allFinite();
            nonpositiveDensity += !(state(0) > 0);
            if (std::isfinite(state(0)))
                minimumDensity = std::min(minimumDensity, state(0));
            const DNDS::real pressure = (gamma - 1) *
                (state(4) - .5 * state.segment<3>(1).squaredNorm() / state(0));
            nonpositivePressure += !(pressure > 0);
            nonfinite += state.allFinite() && !std::isfinite(pressure);
            if (std::isfinite(pressure))
                minimumPressure = std::min(minimumPressure, pressure);
        }

        PhysicalStatistics Global(const DNDS::MPIInfo &mpi) const
        {
            const std::array<DNDS::index, 4> counts{
                samples, nonfinite, nonpositiveDensity, nonpositivePressure};
            const std::array<DNDS::real, 2> minima{minimumDensity, minimumPressure};
            std::array<DNDS::index, 4> globalCounts{};
            std::array<DNDS::real, 2> globalMinima{};
            MPI_Allreduce(counts.data(), globalCounts.data(), 4, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            MPI_Allreduce(minima.data(), globalMinima.data(), 2, DNDS_MPI_REAL, MPI_MIN, mpi.comm);
            PhysicalStatistics result;
            result.samples = globalCounts[0];
            result.nonfinite = globalCounts[1];
            result.nonpositiveDensity = globalCounts[2];
            result.nonpositivePressure = globalCounts[3];
            result.minimumDensity = globalMinima[0];
            result.minimumPressure = globalMinima[1];
            return result;
        }

        bool Physical() const
        {
            return samples > 0 && nonfinite == 0 && nonpositiveDensity == 0 && nonpositivePressure == 0;
        }

        Json ToJson() const
        {
            return {{"samples", samples}, {"nonfinite", nonfinite},
                    {"nonpositive_density", nonpositiveDensity},
                    {"nonpositive_pressure", nonpositivePressure},
                    {"minimum_density", minimumDensity}, {"minimum_pressure", minimumPressure},
                    {"all_physical", Physical()}};
        }
    };
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    int returnCode = 0;
    try
    {
        DNDS_check_throw_info(argc >= 3 && argc <= 6,
                              "usage: config output-prefix [mesh-override|-] [node-field-override|-] [node-table-prefix|-]");
        auto cfg = DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
        if (argc > 3 && std::string(argv[3]) != "-")
            cfg.mesh.meshFile = argv[3];
        if (argc > 4 && std::string(argv[4]) != "-")
            cfg.initialField.nodeFile = argv[4];
        cfg.reconstruction.enableLimiter = true;
        cfg.reconstruction.sharedGradientFaceSweep = true;
        cfg.reconstruction.limitFluxCorrection = false;
        cfg.reconstruction.fluxCorrectionPairBounds = false;
        cfg.reconstruction.directionalPairBounds = false;
        cfg.time.iterations = 0;
        cfg.io.restartInput.clear();
        cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
        cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
        cfg.io.restartInterval = 0;
        DNDS_check_throw_info(cfg.dimension == 3 &&
                                  cfg.algorithm.mode == DNDS::NCFV::IntegrationMode::EfficientDifferential,
                              "Shared-gradient runtime audit requires 3-D efficient integration");
        cfg.Validate();
        std::filesystem::path outputPath(argv[2]);
        if (outputPath.extension() != ".json")
            outputPath += ".json";
        int outputExists = 0;
        if (mpi.rank == 0)
            outputExists = std::filesystem::exists(outputPath);
        MPI_Bcast(&outputExists, 1, MPI_INT, 0, mpi.comm);
        DNDS_check_throw_info(!outputExists, "Refusing to overwrite runtime audit JSON");
        const std::string nodeTablePrefix = argc > 5 && std::string(argv[5]) != "-" ? argv[5] : "";
        std::ofstream nodeTable;
        if (!nodeTablePrefix.empty())
        {
            std::ostringstream suffix;
            suffix << ".rank" << std::setfill('0') << std::setw(6) << mpi.rank << ".csv";
            const std::filesystem::path tablePath(nodeTablePrefix + suffix.str());
            if (mpi.rank == 0 && !tablePath.parent_path().empty())
                std::filesystem::create_directories(tablePath.parent_path());
            MPI_Barrier(mpi.comm);
            DNDS_check_throw_info(!std::filesystem::exists(tablePath), "Refusing to overwrite node audit CSV");
            nodeTable.open(tablePath);
            DNDS_check_throw_info(nodeTable.good(), "Cannot open node audit CSV");
            nodeTable << std::setprecision(17) << "original_id,x,y,z,shared_factor";
            for (const std::string group : {"raw", "final"})
                for (int direction = 0; direction < 3; direction++)
                    for (int component = 0; component < 5; component++)
                        nodeTable << ',' << group << "_d" << direction << "_c" << component;
            nodeTable << '\n';
        }

        const double initializeStart = MPI_Wtime();
        DNDS::NCFV::Solver<3> solver(mpi, cfg);
        solver.Initialize();
        const auto &mesh = solver.Mesh();
        const auto &halo = solver.NodeCommunication();
        DNDS::NCFV::NodeStatePair means, rhs, rawRecoveredPoints;
        for (auto *field : {&means, &rhs, &rawRecoveredPoints})
        {
            field->InitPair(field == &means ? "sweepRuntime.means" :
                            field == &rhs ? "sweepRuntime.rhs" : "sweepRuntime.rawRecoveredPoints", mpi);
            field->father->Resize(mesh->NumNode(), 5, 1);
            field->son->Resize(halo.NumNodeGhost(), 5, 1);
            field->BorrowSetup(halo.Layout());
            field->trans.initPersistentPull();
            for (DNDS::index node = 0; node < field->Size(); node++)
                (*field)[node].setZero();
        }
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
            means[node] = solver.StateField()[node];
        means.trans.startPersistentPull();
        means.trans.waitPersistentPull();

        DNDS::NCFV::NodeMatrixPair rawGradients, rawCoefficients;
        for (auto *field : {&rawGradients, &rawCoefficients})
        {
            const int rows = field == &rawGradients ? 3 : 9;
            field->InitPair(field == &rawGradients ? "sweepRuntime.rawGradients" : "sweepRuntime.rawCoefficients", mpi);
            field->father->Resize(mesh->NumNode(), rows, 5);
            field->son->Resize(halo.NumNodeGhost(), rows, 5);
            field->BorrowSetup(halo.Layout());
            field->trans.initPersistentPull();
            for (DNDS::index node = 0; node < field->Size(); node++)
                (*field)[node].setZero();
        }
        const double initializeSeconds = MPI_Wtime() - initializeStart;
        const double rawStart = MPI_Wtime();
        solver.ReconstructionData().ComputeCoefficients(means, rawGradients, rawCoefficients);
        rawGradients.trans.startPersistentPull();
        rawGradients.trans.waitPersistentPull();
        const double rawSeconds = MPI_Wtime() - rawStart;

        DNDS::NCFV::SpatialOperator<3> spatial(
            mpi, mesh, solver.EdgeTopology(), solver.Geometry(), solver.ReconstructionData(),
            solver.Boundaries(), cfg.algorithm.mode, cfg.reconstruction, cfg.physics, cfg.time, halo);
        spatial.Initialize();
        spatial.ResetRiemannSolverCallCount();
        MPI_Barrier(mpi.comm);
        const double rhsStart = MPI_Wtime();
        const DNDS::real residual = spatial.EvaluateRHS(means, rhs, false, false);
        const double rhsSeconds = MPI_Wtime() - rhsStart;
        const auto &gradients = spatial.Gradients();
        const auto &fluxGradients = spatial.PhysicalFluxGradients();
        const auto &points = spatial.PointValues();
        const auto &sideFactors = spatial.LimiterFactors();
        solver.ReconstructionData().RecoverPointValues(
            means, gradients, rawCoefficients, rawRecoveredPoints);

        // Owned nodes only contribute norms/counts; faces are owned once globally.
        // maxima: rawG, finalG, isotropic absolute/relative residual, ratio spread,
        // alpha outside [0,1], mass-H absolute/relative mismatch, side-factor error,
        // mean change, RHS absolute maximum, zero-raw-node final maximum,
        // PreservePhysical point-value modification.
        std::array<DNDS::real, 13> maxima{};
        std::array<DNDS::real, 4> squares{};
        std::array<DNDS::real, 2> scales{1, 0};
        // counts: nodes, nonzero-raw nodes, scaled nodes, zeroed nodes,
        // nonfinite G/H/RHS nodes, nonzero raw components, modified point values,
        // owned edges, calls.
        std::array<DNDS::index, 9> counts{};
        PhysicalStatistics meanStatistics, pointStatistics, rawPointStatistics, faceStatistics;
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
        {
            const Gradient raw = rawGradients[node], limited = gradients[node];
            counts[0]++;
            counts[4] += !raw.allFinite() || !limited.allFinite() || !fluxGradients[node].allFinite() || !rhs[node].allFinite();
            Eigen::Index largestRow = 0, largestColumn = 0;
            const DNDS::real rawMaximum = raw.cwiseAbs().maxCoeff(&largestRow, &largestColumn);
            const DNDS::real finalMaximum = limited.cwiseAbs().maxCoeff();
            maxima[0] = std::max(maxima[0], rawMaximum);
            maxima[1] = std::max(maxima[1], finalMaximum);
            const DNDS::real scale = spatial.SharedGradientFactor(node);
            counts[4] += !std::isfinite(scale);
            scales[0] = std::min(scales[0], scale);
            scales[1] = std::max(scales[1], scale);
            maxima[5] = std::max({maxima[5], -scale, scale - 1});
            counts[2] += scale < 1 - 1e-12;
            squares[0] += raw.squaredNorm();
            squares[1] += limited.squaredNorm();
            const DNDS::real volume = halo.Volume(node);
            squares[2] += volume * raw.squaredNorm();
            squares[3] += volume * limited.squaredNorm();
            if (rawMaximum > 0)
            {
                counts[1]++;
                counts[3] += finalMaximum == 0;
                const DNDS::real error = (limited - scale * raw).cwiseAbs().maxCoeff();
                maxima[2] = std::max(maxima[2], error);
                maxima[3] = std::max(maxima[3], error / std::max(rawMaximum, DNDS::real(1e-300)));
                for (int direction = 0; direction < 3; direction++)
                    for (int component = 0; component < 5; component++)
                    {
                        counts[5] += raw(direction, component) != 0;
                        if (std::abs(raw(direction, component)) > 1e-12 * rawMaximum)
                            maxima[4] = std::max(maxima[4], std::abs(
                                limited(direction, component) / raw(direction, component) - scale));
                    }
            }
            else
                maxima[11] = std::max(maxima[11], finalMaximum);

            for (int derivativeDirection = 0; derivativeDirection < 3; derivativeDirection++)
                for (int fluxDirection = 0; fluxDirection < 3; fluxDirection++)
                {
                    const DNDS::real expected = limited(derivativeDirection, 1 + fluxDirection);
                    const DNDS::real actual = fluxGradients[node](0, derivativeDirection * 3 + fluxDirection);
                    const DNDS::real error = std::abs(actual - expected);
                    maxima[6] = std::max(maxima[6], error);
                    maxima[7] = std::max(maxima[7], error / std::max(DNDS::real(1), std::abs(expected)));
                }
            maxima[9] = std::max(maxima[9], (means[node] - solver.StateField()[node]).cwiseAbs().maxCoeff());
            maxima[10] = std::max(maxima[10], rhs[node].cwiseAbs().maxCoeff());
            meanStatistics.Add(State(means[node]), cfg.physics.gamma);
            pointStatistics.Add(State(points[node]), cfg.physics.gamma);
            const State rawPoint = rawRecoveredPoints[node];
            rawPointStatistics.Add(rawPoint, cfg.physics.gamma);
            const DNDS::real pointModification = (rawPoint - points[node]).cwiseAbs().maxCoeff();
            counts[6] += pointModification > 0;
            maxima[12] = std::max(maxima[12], pointModification);
            if (nodeTable.is_open())
            {
                const auto coordinate = halo.Coordinate(node);
                nodeTable << mesh->node2nodeOrig(node, 0) << ',' << coordinate(0) << ','
                          << coordinate(1) << ',' << coordinate(2) << ',' << scale;
                for (const Gradient *field : {&raw, &limited})
                    for (int direction = 0; direction < 3; direction++)
                        for (int component = 0; component < 5; component++)
                            nodeTable << ',' << (*field)(direction, component);
                nodeTable << '\n';
            }
        }
        nodeTable.close();
        // Ghost factors also feed interface evaluation and must remain at one.
        for (DNDS::index node = 0; node < sideFactors.Size(); node++)
        {
            const DNDS::real factor = sideFactors[node](0, 0);
            maxima[8] = std::max(maxima[8], std::isfinite(factor) ? std::abs(factor - 1) :
                                std::numeric_limits<DNDS::real>::infinity());
        }
        for (DNDS::index edge = 0; edge < solver.EdgeTopology().NumEdge(); edge++)
        {
            const auto &surface = solver.Geometry().EdgeSurface(edge);
            for (int side = 0; side < 2; side++)
            {
                State correction = State::Zero();
                for (const auto &entry : surface.efficientStencil)
                    correction += gradients[entry.node].transpose() *
                                  entry.stateGradientWeights[side].head<3>();
                const State face = State(points[surface.nodes[static_cast<std::size_t>(side)]]) +
                                   correction / surface.measure;
                // Deliberately no old side factor and no PreservePhysical call.
                faceStatistics.Add(face, cfg.physics.gamma);
            }
        }
        counts[7] = solver.EdgeTopology().NumEdge();
        counts[8] = spatial.RiemannSolverCallCount();
        const auto &timing = spatial.LastRhsTiming();
        const auto *sweepDiagnostics = spatial.SharedGradientSweepDiagnostics();
        DNDS_check_throw_info(sweepDiagnostics != nullptr, "Shared-gradient sweep diagnostics are missing");
        const std::array<DNDS::real, 13> timings{
            initializeSeconds, rawSeconds, rhsSeconds, timing.reconstructionSeconds,
            timing.limiterSeconds, timing.physicalFluxGradientComputeSeconds,
            timing.edgeFluxSeconds, timing.edgeHaloSeconds, timing.totalSeconds,
            sweepDiagnostics->inputHaloSeconds, sweepDiagnostics->localSweepSeconds,
            sweepDiagnostics->blockCommunicationSeconds, sweepDiagnostics->totalSeconds};
        std::array<DNDS::real, 13> globalMaxima{};
        std::array<DNDS::real, 4> globalSquares{};
        std::array<DNDS::index, 9> globalCounts{};
        std::array<DNDS::real, 13> globalTimings{};
        DNDS::real minimumScale = 1, maximumScale = 0;
        MPI_Allreduce(maxima.data(), globalMaxima.data(), 13, DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        MPI_Allreduce(squares.data(), globalSquares.data(), 4, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        MPI_Allreduce(counts.data(), globalCounts.data(), 9, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        MPI_Allreduce(timings.data(), globalTimings.data(), 13, DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        MPI_Allreduce(&scales[0], &minimumScale, 1, DNDS_MPI_REAL, MPI_MIN, mpi.comm);
        MPI_Allreduce(&scales[1], &maximumScale, 1, DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        meanStatistics = meanStatistics.Global(mpi);
        pointStatistics = pointStatistics.Global(mpi);
        rawPointStatistics = rawPointStatistics.Global(mpi);
        faceStatistics = faceStatistics.Global(mpi);
        const DNDS::real retained = globalSquares[0] > 0 ? std::sqrt(globalSquares[1] / globalSquares[0]) : 1;
        const DNDS::real weightedRetained = globalSquares[2] > 0 ? std::sqrt(globalSquares[3] / globalSquares[2]) : 1;
        const bool algebraPassed = globalCounts[4] == 0 && globalMaxima[3] <= 1e-11 &&
            globalMaxima[4] <= 1e-10 && globalMaxima[5] <= 1e-12 && globalMaxima[7] <= 1e-12 &&
            globalMaxima[8] == 0 && globalMaxima[9] == 0 && globalMaxima[11] == 0 &&
            std::isfinite(residual) && globalCounts[8] >= globalCounts[7];
        const bool physicalPassed = meanStatistics.Physical() && pointStatistics.Physical() && faceStatistics.Physical();
        returnCode = algebraPassed && physicalPassed ? 0 : 2;
        if (mpi.rank == 0)
        {
            const Json result{
                {"format", "DNDSR shared-gradient one-RHS runtime audit v1"},
                {"source_configuration", argv[1]}, {"effective_configuration", cfg},
                {"optional_node_table_prefix", nodeTablePrefix},
                {"node_table_layout", "<prefix>.rankNNNNNN.csv: original_id,x,y,z,shared_factor,raw_d0_c0..raw_d2_c4,final_d0_c0..final_d2_c4"},
                {"mpi_ranks", mpi.size}, {"owned_nodes_global", globalCounts[0]},
                {"owned_edges_global", globalCounts[7]}, {"rhs_evaluations", 1},
                {"time_steps_taken", 0}, {"restart_input", ""},
                {"algebra_checks_passed", algebraPassed}, {"physical_checks_passed", physicalPassed},
                {"all_checks_passed", returnCode == 0},
                {"gradient_scaling", {
                    {"raw_L2", std::sqrt(globalSquares[0])}, {"final_L2", std::sqrt(globalSquares[1])},
                    {"retained_L2_fraction", retained}, {"retained_volume_weighted_L2_fraction", weightedRetained},
                    {"raw_max_abs", globalMaxima[0]}, {"final_max_abs", globalMaxima[1]},
                    {"nodes_with_nonzero_raw_gradient", globalCounts[1]}, {"scaled_nodes", globalCounts[2]},
                    {"zeroed_nodes", globalCounts[3]}, {"nonzero_raw_components", globalCounts[5]},
                    {"common_scale_min", minimumScale}, {"common_scale_max", maximumScale},
                    {"common_scale_range_violation_max", globalMaxima[5]},
                    {"isotropic_residual_abs_max", globalMaxima[2]}, {"isotropic_residual_relative_max", globalMaxima[3]},
                    {"resolved_component_scale_deviation_max", globalMaxima[4]},
                    {"zero_raw_gradient_final_abs_max", globalMaxima[11]},
                    {"ratio_note", "Recorded SharedGradientFactor is checked against all 15 raw/final components. Ratio deviation checks components above 1e-12 of the node maximum."}}},
                {"sweep_diagnostics", {
                    {"updated_faces_global", sweepDiagnostics->updatedFaces},
                    {"zero_factor_faces_global", sweepDiagnostics->zeroFactorFaces},
                    {"minimum_face_factor", sweepDiagnostics->minimumFaceFactor},
                    {"rank_blocks", sweepDiagnostics->rankBlocks}}},
                {"physical_flux_gradient_consistency", {
                    {"identity", "H(0,3*derivativeDirection+fluxDirection)=G(derivativeDirection,1+fluxDirection)"},
                    {"mass_row_abs_error_max", globalMaxima[6]}, {"mass_row_relative_error_max", globalMaxima[7]}}},
                {"old_side_limiter_factor_deviation_from_one_max", globalMaxima[8]},
                {"conservative_means_change_max", globalMaxima[9]}, {"nonfinite_gradient_flux_or_rhs_nodes", globalCounts[4]},
                {"rhs_abs_max", globalMaxima[10]}, {"rhs_finite", globalCounts[4] == 0},
                {"riemann_solver_calls_global", globalCounts[8]},
                {"conservative_means", meanStatistics.ToJson()}, {"runtime_points", pointStatistics.ToJson()},
                {"unprotected_recovered_points", rawPointStatistics.ToJson()},
                {"point_physical_protection", {
                    {"modified_owned_points", globalCounts[6]},
                    {"maximum_component_change", globalMaxima[12]},
                    {"comparison", "RecoverPointValues(means,finalG) versus actual runtime PointValues after PreservePhysical"}}},
                {"unprotected_interface_means_from_final_gradients", faceStatistics.ToJson()},
                {"face_check_note", "Reassembled from final synchronized gradients and runtime point anchors, with no old side theta and no physical-state fallback."},
                {"timing_max_over_ranks_seconds", {
                    {"initialize_and_allocate", globalTimings[0]}, {"raw_reconstruction", globalTimings[1]},
                    {"RHS_wall", globalTimings[2]}, {"RHS_reconstruction", globalTimings[3]},
                    {"limiter_sweep_and_point_recovery", globalTimings[4]},
                    {"physical_flux_gradient_compute", globalTimings[5]},
                    {"edge_flux", globalTimings[6]}, {"edge_halo", globalTimings[7]}, {"RHS_reported_total", globalTimings[8]},
                    {"sweep_input_halo", globalTimings[9]}, {"sweep_local_pass", globalTimings[10]},
                    {"sweep_block_communication", globalTimings[11]}, {"sweep_total", globalTimings[12]}}},
                {"scope", "One fresh-state RHS only; no solver Run(), no time integration, no production result output, no geometry export."}};
            if (!outputPath.parent_path().empty())
                std::filesystem::create_directories(outputPath.parent_path());
            std::ofstream output(outputPath);
            DNDS_check_throw_info(output.good(), "Cannot open runtime audit JSON");
            output << result.dump(2) << '\n';
            std::cout << "Shared-gradient audit: algebra=" << algebraPassed << ", physical=" << physicalPassed
                      << ", retained G L2=" << retained << ", isotropic error=" << globalMaxima[3]
                      << ", mass-H error=" << globalMaxima[6] << ", Riemann calls=" << globalCounts[8]
                      << "; report=" << outputPath << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "shared-gradient runtime audit: " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    MPI_Finalize();
    return returnCode;
}
