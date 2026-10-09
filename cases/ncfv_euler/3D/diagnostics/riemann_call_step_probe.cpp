/**
 * @file riemann_call_step_probe.cpp
 * @brief Count selected-Riemann-solver calls in one production SSPRK3 IV10 step.
 */
#include "NCFV/NCFVSolver.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <string>

namespace
{
    using DNDS::index;
    using DNDS::NCFV::BoundaryMode;
    using DNDS::NCFV::Configuration;
    using DNDS::NCFV::IntegrationMode;
    using DNDS::NCFV::Solver;

    struct FluxCallBreakdown
    {
        index internal = 0;
        index boundary = 0;
    };

    std::string ModeName(IntegrationMode mode)
    {
        return mode == IntegrationMode::EfficientDifferential
                   ? "EfficientDifferential"
                   : "TraditionalQuadrature";
    }

    FluxCallBreakdown ExpectedCallsPerRHS(
        const Solver<3> &solver,
        IntegrationMode mode)
    {
        FluxCallBreakdown result;
        const auto &topology = solver.EdgeTopology();
        const auto &geometry = solver.Geometry();
        const auto &boundaries = solver.Boundaries();
        for (index iEdge = 0; iEdge < topology.NumEdge(); ++iEdge)
        {
            const auto &surface = geometry.EdgeSurface(iEdge);
            result.internal += mode == IntegrationMode::TraditionalQuadrature
                                   ? static_cast<index>(surface.quadrature.size())
                                   : 1;
        }
        for (index iNode = 0; iNode < solver.Mesh()->NumNode(); ++iNode)
        {
            for (const auto &piece : geometry.NodeVolume(iNode).boundaryPieces)
            {
                if (boundaries.Get(piece.zone).mode == BoundaryMode::Periodic)
                    continue;
                result.boundary += mode == IntegrationMode::TraditionalQuadrature
                                       ? static_cast<index>(piece.quadrature.size())
                                       : static_cast<index>(piece.nPoints);
            }
        }
        return result;
    }

    FluxCallBreakdown GlobalSum(
        const DNDS::MPIInfo &mpi,
        const FluxCallBreakdown &local)
    {
        const std::array<index, 2> localValues{local.internal, local.boundary};
        std::array<index, 2> globalValues{};
        MPI_Allreduce(localValues.data(), globalValues.data(),
                      static_cast<int>(localValues.size()), DNDS::DNDS_MPI_INDEX,
                      MPI_SUM, mpi.comm);
        return {globalValues[0], globalValues[1]};
    }

    DNDS::NCFV::StepPhaseTiming GlobalMaximum(
        const DNDS::MPIInfo &mpi,
        const DNDS::NCFV::StepPhaseTiming &local)
    {
        const std::array<DNDS::real, 33> localValues{
            local.rhs.reconstructionSeconds,
            local.rhs.meanHaloSeconds,
            local.rhs.coefficientComputeSeconds,
            local.rhs.coefficientHaloSeconds,
            local.rhs.pointRecoverySeconds,
            local.rhs.limiterSeconds,
            local.rhs.pointValueHaloSeconds,
            local.rhs.physicalFluxGradientComputeSeconds,
            local.rhs.edgeFluxSeconds,
            local.rhs.edgeLeftStatePreparationSeconds,
            local.rhs.edgeRightStatePreparationSeconds,
            local.rhs.edgeLeftPhysicalFluxIntegralSeconds,
            local.rhs.edgeRightPhysicalFluxIntegralSeconds,
            local.rhs.edgeNumericalFluxAndAssemblySeconds,
            local.rhs.edgeLeftPhysicalFluxDetail.zeroOrderFluxSeconds,
            local.rhs.edgeLeftPhysicalFluxDetail.precomputedGradientIntegralSeconds,
            local.rhs.edgeLeftPhysicalFluxDetail.finalAssemblySeconds,
            local.rhs.edgeRightPhysicalFluxDetail.zeroOrderFluxSeconds,
            local.rhs.edgeRightPhysicalFluxDetail.precomputedGradientIntegralSeconds,
            local.rhs.edgeRightPhysicalFluxDetail.finalAssemblySeconds,
            local.rhs.edgeRiemannFluxSeconds,
            local.rhs.edgeCentralFluxSeconds,
            local.rhs.edgeDissipationAssemblySeconds,
            local.rhs.edgeInviscidFinalAssemblySeconds,
            local.rhs.edgeViscousFluxSeconds,
            local.rhs.edgeHaloSeconds,
            local.rhs.localTimeStepSeconds,
            local.rhs.residualAssemblySeconds,
            local.rhs.residualNormSeconds,
            local.rhs.totalSeconds,
            local.baseStateCopySeconds,
            local.stageUpdateSeconds,
            local.totalSeconds};
        std::array<DNDS::real, 33> globalValues{};
        MPI_Allreduce(localValues.data(), globalValues.data(),
                      static_cast<int>(localValues.size()), DNDS::DNDS_MPI_REAL,
                      MPI_MAX, mpi.comm);

        DNDS::NCFV::StepPhaseTiming result;
        result.rhs.reconstructionSeconds = globalValues[0];
        result.rhs.meanHaloSeconds = globalValues[1];
        result.rhs.coefficientComputeSeconds = globalValues[2];
        result.rhs.coefficientHaloSeconds = globalValues[3];
        result.rhs.pointRecoverySeconds = globalValues[4];
        result.rhs.limiterSeconds = globalValues[5];
        result.rhs.pointValueHaloSeconds = globalValues[6];
        result.rhs.physicalFluxGradientComputeSeconds = globalValues[7];
        result.rhs.edgeFluxSeconds = globalValues[8];
        result.rhs.edgeLeftStatePreparationSeconds = globalValues[9];
        result.rhs.edgeRightStatePreparationSeconds = globalValues[10];
        result.rhs.edgeLeftPhysicalFluxIntegralSeconds = globalValues[11];
        result.rhs.edgeRightPhysicalFluxIntegralSeconds = globalValues[12];
        result.rhs.edgeNumericalFluxAndAssemblySeconds = globalValues[13];
        result.rhs.edgeLeftPhysicalFluxDetail.zeroOrderFluxSeconds = globalValues[14];
        result.rhs.edgeLeftPhysicalFluxDetail.precomputedGradientIntegralSeconds =
            globalValues[15];
        result.rhs.edgeLeftPhysicalFluxDetail.finalAssemblySeconds = globalValues[16];
        result.rhs.edgeRightPhysicalFluxDetail.zeroOrderFluxSeconds = globalValues[17];
        result.rhs.edgeRightPhysicalFluxDetail.precomputedGradientIntegralSeconds =
            globalValues[18];
        result.rhs.edgeRightPhysicalFluxDetail.finalAssemblySeconds = globalValues[19];
        result.rhs.edgeRiemannFluxSeconds = globalValues[20];
        result.rhs.edgeCentralFluxSeconds = globalValues[21];
        result.rhs.edgeDissipationAssemblySeconds = globalValues[22];
        result.rhs.edgeInviscidFinalAssemblySeconds = globalValues[23];
        result.rhs.edgeViscousFluxSeconds = globalValues[24];
        result.rhs.edgeHaloSeconds = globalValues[25];
        result.rhs.localTimeStepSeconds = globalValues[26];
        result.rhs.residualAssemblySeconds = globalValues[27];
        result.rhs.residualNormSeconds = globalValues[28];
        result.rhs.totalSeconds = globalValues[29];
        result.baseStateCopySeconds = globalValues[30];
        result.stageUpdateSeconds = globalValues[31];
        result.totalSeconds = globalValues[32];
        return result;
    }

    nlohmann::ordered_json RunOneStep(
        const DNDS::MPIInfo &mpi,
        const Configuration &base,
        IntegrationMode mode,
        const std::filesystem::path &scratchDirectory)
    {
        Configuration configuration = base;
        configuration.algorithm.mode = mode;
        configuration.time.iterations = 1;
        configuration.io.writeVTK = false;
        configuration.io.writeInitial = false;
        configuration.io.writeFinal = false;
        configuration.io.outputInterval = 0;
        configuration.io.restartInput.clear();
        configuration.io.restartInterval = 0;
        configuration.io.writeFinalRestart = false;
        configuration.io.writeResolvedConfiguration = false;
        configuration.io.outputPrefix =
            (scratchDirectory / (ModeName(mode) + "_solution")).string();
        configuration.io.vtkSeriesName =
            (scratchDirectory / (ModeName(mode) + "_series")).string();
        configuration.io.restartPrefix =
            (scratchDirectory / (ModeName(mode) + "_restart")).string();

        Solver<3> solver(mpi, configuration);
        solver.Initialize();
        solver.EnableDetailedFluxTiming(
            mode == IntegrationMode::EfficientDifferential);
        const FluxCallBreakdown expected = GlobalSum(
            mpi, ExpectedCallsPerRHS(solver, mode));
        solver.Run();
        const auto timing = GlobalMaximum(mpi, solver.LastStepTiming());

        const index localCalls = solver.LastStepRiemannSolverCalls();
        index globalCalls = 0;
        index minimumCalls = 0;
        index maximumCalls = 0;
        MPI_Allreduce(&localCalls, &globalCalls, 1, DNDS::DNDS_MPI_INDEX,
                      MPI_SUM, mpi.comm);
        MPI_Allreduce(&localCalls, &minimumCalls, 1, DNDS::DNDS_MPI_INDEX,
                      MPI_MIN, mpi.comm);
        MPI_Allreduce(&localCalls, &maximumCalls, 1, DNDS::DNDS_MPI_INDEX,
                      MPI_MAX, mpi.comm);

        constexpr index sspRkStages = 3;
        const index expectedPerRHS = expected.internal + expected.boundary;
        const index expectedPerStep = sspRkStages * expectedPerRHS;
        DNDS_check_throw_info(
            solver.CurrentIteration() == 1 && solver.SimulationTime() > 0,
            "Expected exactly one positive-size SSPRK3 step");
        DNDS_check_throw_info(
            globalCalls == expectedPerStep,
            "Instrumented Riemann call count does not match the executed flux topology");

        return {
            {"mode", ModeName(mode)},
            {"time_steps", solver.CurrentIteration()},
            {"simulation_time", solver.SimulationTime()},
            {"ssprk_stages", sspRkStages},
            {"rhs_evaluations_per_step", sspRkStages},
            {"riemann_solver", "Roe_M2"},
            {"riemann_calls", {
                {"per_rhs", expectedPerRHS},
                {"per_step", globalCalls},
                {"internal_per_rhs", expected.internal},
                {"boundary_per_rhs", expected.boundary},
                {"global_expected_per_step", expectedPerStep},
                {"local_rank_min_per_step", minimumCalls},
                {"local_rank_max_per_step", maximumCalls}}},
            {"timing_seconds_rank_max", {
                {"step_total", timing.totalSeconds},
                {"base_state_copy", timing.baseStateCopySeconds},
                {"reconstruction", timing.rhs.reconstructionSeconds},
                {"mean_halo", timing.rhs.meanHaloSeconds},
                {"coefficient_compute", timing.rhs.coefficientComputeSeconds},
                {"coefficient_halo", timing.rhs.coefficientHaloSeconds},
                {"point_recovery", timing.rhs.pointRecoverySeconds},
                {"limiter", timing.rhs.limiterSeconds},
                {"point_value_halo", timing.rhs.pointValueHaloSeconds},
                {"physical_flux_gradient_compute",
                 timing.rhs.physicalFluxGradientComputeSeconds},
                {"internal_edge_flux", timing.rhs.edgeFluxSeconds},
                {"left_state_preparation", timing.rhs.edgeLeftStatePreparationSeconds},
                {"right_state_preparation", timing.rhs.edgeRightStatePreparationSeconds},
                {"left_physical_flux_integral", timing.rhs.edgeLeftPhysicalFluxIntegralSeconds},
                {"right_physical_flux_integral", timing.rhs.edgeRightPhysicalFluxIntegralSeconds},
                {"numerical_flux_and_assembly", timing.rhs.edgeNumericalFluxAndAssemblySeconds},
                {"efficient_left_zero_order_physical_flux",
                 timing.rhs.edgeLeftPhysicalFluxDetail.zeroOrderFluxSeconds},
                {"efficient_left_precomputed_flux_gradient_integral",
                 timing.rhs.edgeLeftPhysicalFluxDetail
                     .precomputedGradientIntegralSeconds},
                {"efficient_left_physical_flux_final_assembly",
                 timing.rhs.edgeLeftPhysicalFluxDetail.finalAssemblySeconds},
                {"efficient_right_zero_order_physical_flux",
                 timing.rhs.edgeRightPhysicalFluxDetail.zeroOrderFluxSeconds},
                {"efficient_right_precomputed_flux_gradient_integral",
                 timing.rhs.edgeRightPhysicalFluxDetail
                     .precomputedGradientIntegralSeconds},
                {"efficient_right_physical_flux_final_assembly",
                 timing.rhs.edgeRightPhysicalFluxDetail.finalAssemblySeconds},
                {"efficient_riemann_flux", timing.rhs.edgeRiemannFluxSeconds},
                {"efficient_central_flux", timing.rhs.edgeCentralFluxSeconds},
                {"efficient_dissipative_correction_assembly",
                 timing.rhs.edgeDissipationAssemblySeconds},
                {"efficient_inviscid_final_assembly",
                 timing.rhs.edgeInviscidFinalAssemblySeconds},
                {"efficient_viscous_flux", timing.rhs.edgeViscousFluxSeconds},
                {"edge_halo", timing.rhs.edgeHaloSeconds},
                {"cfl_and_local_time_step", timing.rhs.localTimeStepSeconds},
                {"boundary_and_residual_assembly", timing.rhs.residualAssemblySeconds},
                {"residual_norm", timing.rhs.residualNormSeconds},
                {"rhs_total", timing.rhs.totalSeconds},
                {"ssprk_state_update_and_check", timing.stageUpdateSeconds}}},
            {"mesh", {
                {"nodes", solver.Mesh()->NumNodeGlobal()},
                {"cells", solver.Mesh()->NumCellGlobal()},
                {"edges", solver.EdgeTopology().NumEdgeGlobal()}}},
            {"vortex_diagnostic_rhs_excluded_from_step_count", 2}};
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
                "Usage: riemann_call_step_probe NCFV_iv10.json result.json scratch-directory");
            const std::filesystem::path resultPath = argv[2];
            const std::filesystem::path scratchDirectory = argv[3];
            const Configuration configuration =
                DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
            DNDS_check_throw_info(
                configuration.dimension == 3 && configuration.initialField.isentropicVortex &&
                    !configuration.physics.viscous.enabled &&
                    configuration.physics.riemannSolver == DNDS::Euler::Gas::Roe_M2,
                "Expected the fresh inviscid 3-D Roe_M2 isentropic-vortex configuration");

            if (mpi.rank == 0)
            {
                DNDS_check_throw_info(!std::filesystem::exists(resultPath),
                                      "Refusing to overwrite result: " + resultPath.string());
                std::filesystem::create_directories(resultPath.parent_path());
                std::filesystem::create_directories(scratchDirectory);
            }
            MPI_Barrier(mpi.comm);

            nlohmann::ordered_json result = {
                {"source_configuration", argv[1]},
                {"mpi_ranks", mpi.size},
                {"measurement_scope",
                 "The locked counter covers only the three EvaluateRHS calls in one SSPRK3 step."},
                {"results", nlohmann::ordered_json::array()}};
            for (const IntegrationMode mode : {
                     IntegrationMode::EfficientDifferential,
                     IntegrationMode::TraditionalQuadrature})
                result["results"].push_back(RunOneStep(mpi, configuration, mode, scratchDirectory));

            if (mpi.rank == 0)
            {
                std::ofstream output(resultPath);
                DNDS_check_throw_info(output.good(),
                                      "Cannot open result: " + resultPath.string());
                output << result.dump(2) << '\n';
                output.close();
                DNDS_check_throw_info(output.good(),
                                      "Cannot finish result: " + resultPath.string());
                DNDS::log() << "NCFV one-step Riemann-call probe complete: "
                            << resultPath << std::endl;
            }
            MPI_Barrier(mpi.comm);
        }
        catch (const std::exception &exception)
        {
            std::cerr << "NCFV one-step Riemann-call probe: "
                      << exception.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
