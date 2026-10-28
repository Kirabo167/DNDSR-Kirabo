/** Read-only limiter audit of a saved periodic RP1 state. */
#include "NCFV/NCFVSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using DNDS::DNDS_MPI_INDEX;
using DNDS::DNDS_MPI_REAL;

namespace
{
    using State = Eigen::Matrix<DNDS::real, 5, 1>;
    constexpr DNDS::real tolerance = 1e-9;

    DNDS::real Pressure(const State &state, DNDS::real gamma)
    {
        return (gamma - 1) *
               (state(4) - .5 * state.segment<3>(1).squaredNorm() / state(0));
    }

    struct Statistics
    {
        DNDS::index samples = 0, nonfinite = 0, nonpositiveDensity = 0, nonpositivePressure = 0;
        DNDS::index pressureBoundViolations = 0;
        std::array<DNDS::index, 5> boundViolations{};
        std::array<DNDS::real, 5> maximumViolation{};
        DNDS::real minimumDensity = 1e300, minimumPressure = 1e300;
        DNDS::real maximumPressureViolation = 0;

        DNDS::real Add(const State &state, const State &lower, const State &upper,
                       DNDS::real pressureLower, DNDS::real pressureUpper,
                       DNDS::real gamma)
        {
            samples++;
            nonfinite += !state.allFinite();
            const DNDS::real pressure = Pressure(state, gamma);
            nonpositiveDensity += !(state(0) > 0);
            nonpositivePressure += !(pressure > 0);
            minimumDensity = std::min(minimumDensity, state(0));
            minimumPressure = std::min(minimumPressure, pressure);
            DNDS::real largest = 0;
            for (int variable = 0; variable < 5; variable++)
            {
                const DNDS::real excess = std::max(
                    {DNDS::real(0), lower(variable) - state(variable), state(variable) - upper(variable)});
                boundViolations[variable] += excess > tolerance;
                maximumViolation[variable] = std::max(maximumViolation[variable], excess);
                largest = std::max(largest, excess);
            }
            const DNDS::real pressureExcess = std::max(
                {DNDS::real(0), pressureLower - pressure, pressure - pressureUpper});
            pressureBoundViolations += pressureExcess > tolerance;
            maximumPressureViolation = std::max(maximumPressureViolation, pressureExcess);
            return largest;
        }

        nlohmann::ordered_json Reduce(const DNDS::MPIInfo &mpi) const
        {
            const std::array<DNDS::index, 5> counts{
                samples, nonfinite, nonpositiveDensity, nonpositivePressure, pressureBoundViolations};
            std::array<DNDS::index, 5> globalCounts{}, globalBoundViolations{};
            std::array<DNDS::real, 5> globalMaximumViolation{};
            const std::array<DNDS::real, 2> minima{minimumDensity, minimumPressure};
            std::array<DNDS::real, 2> globalMinima{};
            DNDS::real globalPressureViolation = 0;
            MPI_Reduce(counts.data(), globalCounts.data(), 5, DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
            MPI_Reduce(boundViolations.data(), globalBoundViolations.data(), 5,
                       DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
            MPI_Reduce(maximumViolation.data(), globalMaximumViolation.data(), 5,
                       DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
            MPI_Reduce(minima.data(), globalMinima.data(), 2,
                       DNDS_MPI_REAL, MPI_MIN, 0, mpi.comm);
            MPI_Reduce(&maximumPressureViolation, &globalPressureViolation, 1,
                       DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
            return {{"samples", globalCounts[0]}, {"nonfinite", globalCounts[1]},
                    {"nonpositive_density", globalCounts[2]}, {"nonpositive_pressure", globalCounts[3]},
                    {"conservative_bound_violation_count", globalBoundViolations},
                    {"conservative_bound_violation_max", globalMaximumViolation},
                    {"minimum_density", globalMinima[0]}, {"minimum_pressure", globalMinima[1]},
                    {"pressure_neighbor_bound_violation_count", globalCounts[4]},
                    {"pressure_neighbor_bound_violation_max", globalPressureViolation}};
        }
    };

    std::array<DNDS::real, 5> Components(const State &state)
    {
        return {state(0), state(1), state(2), state(3), state(4)};
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    try
    {
        DNDS_check_throw_info(argc == 4, "usage: config restart-base output-summary.json");
        auto cfg = DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
        cfg.io.restartInput = argv[2];
        cfg.time.iterations = 0;
        cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
        cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
        cfg.io.restartInterval = 0;
        DNDS_check_throw_info(
            cfg.algorithm.mode == DNDS::NCFV::IntegrationMode::EfficientDifferential &&
                cfg.reconstruction.enableLimiter && cfg.reconstruction.limitPointRecovery,
            "Final limiter audit requires the mean-anchored efficient limiter");
        DNDS::NCFV::Solver<3> solver(mpi, cfg);
        solver.Initialize();
        const auto &mesh = solver.Mesh();
        const auto &halo = solver.NodeCommunication();
        DNDS::NCFV::NodeStatePair means, rhs;
        const auto allocate = [&](DNDS::NCFV::NodeStatePair &field, const std::string &name)
        {
            field.InitPair(name, mpi);
            field.father->Resize(mesh->NumNode(), 5, 1);
            field.son->Resize(halo.NumNodeGhost(), 5, 1);
            field.BorrowSetup(halo.Layout());
            field.trans.initPersistentPull();
            for (DNDS::index i = 0; i < field.Size(); i++)
            {
                field[i].setZero();
            }
        };
        allocate(means, "limiterFinal.means");
        allocate(rhs, "limiterFinal.rhs");
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            means[i] = solver.StateField()[i];
        }
        DNDS::NCFV::SpatialOperator<3> spatial(
            mpi, mesh, solver.EdgeTopology(), solver.Geometry(), solver.ReconstructionData(),
            solver.Boundaries(), cfg.algorithm.mode, cfg.reconstruction, cfg.physics, cfg.time, halo);
        spatial.Initialize();
        const DNDS::real residual = spatial.EvaluateRHS(means, rhs);
        const auto &points = spatial.PointValues();
        const auto &gradients = spatial.Gradients();
        const auto &theta = spatial.LimiterFactors();
        std::array<Statistics, 3> stats;
        // Owned nodes, theta==0, theta<1e-12, theta<.1, theta<.5, theta==1,
        // zero-theta point samples, zero-theta interface-side samples.
        std::array<DNDS::index, 8> thetaCounts{};
        std::array<DNDS::real, 2> zeroFallbackMaximum{};
        std::array<DNDS::real, 2> worstBoundExcess{-1, -1};
        std::array<nlohmann::ordered_json, 2> examples;
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            const State mean = means[i];
            State lower = mean, upper = mean;
            DNDS::real pressureLower = Pressure(mean, cfg.physics.gamma);
            DNDS::real pressureUpper = pressureLower;
            for (const auto &incidence : solver.EdgeTopology().Node2Edge(i))
            {
                const auto &surface = solver.Geometry().EdgeSurface(incidence.edge);
                const DNDS::index neighbor = surface.nodes[surface.nodes[0] == i ? 1 : 0];
                const State neighborMean = means[neighbor];
                lower = lower.cwiseMin(neighborMean);
                upper = upper.cwiseMax(neighborMean);
                const DNDS::real pressure = Pressure(neighborMean, cfg.physics.gamma);
                pressureLower = std::min(pressureLower, pressure);
                pressureUpper = std::max(pressureUpper, pressure);
            }
            const DNDS::real factor = theta[i](0);
            thetaCounts[0]++;
            thetaCounts[1] += factor == 0;
            thetaCounts[2] += factor < 1e-12;
            thetaCounts[3] += factor < .1;
            thetaCounts[4] += factor < .5;
            thetaCounts[5] += factor == 1;
            stats[0].Add(mean, lower, upper, pressureLower, pressureUpper, cfg.physics.gamma);
            const auto record = [&](int type, const State &state,
                                    DNDS::real excess, DNDS::index neighborGlobal)
            {
                if (excess <= worstBoundExcess[type])
                    return;
                worstBoundExcess[type] = excess;
                const auto coordinate = halo.Coordinate(i);
                examples[type] = {
                    {"rank", mpi.rank}, {"original_node", mesh->node2nodeOrig(i, 0)},
                    {"global_node", halo.LocalToGlobal(i)}, {"neighbor_global_node", neighborGlobal},
                    {"coordinate", {coordinate(0), coordinate(1), coordinate(2)}},
                    {"theta", factor}, {"maximum_conservative_bound_excess", excess},
                    {"mean", Components(mean)}, {"sample", Components(state)},
                    {"lower", Components(lower)}, {"upper", Components(upper)},
                    {"pressure", Pressure(state, cfg.physics.gamma)},
                    {"neighbor_pressure_lower", pressureLower}, {"neighbor_pressure_upper", pressureUpper}};
            };
            const State point = points[i];
            const DNDS::real pointExcess = stats[1].Add(
                point, lower, upper, pressureLower, pressureUpper, cfg.physics.gamma);
            record(0, point, pointExcess, DNDS::UnInitIndex);
            if (factor == 0)
            {
                thetaCounts[6]++;
                zeroFallbackMaximum[0] = std::max(zeroFallbackMaximum[0], (point - mean).cwiseAbs().maxCoeff());
            }
            for (const auto &incidence : solver.EdgeTopology().Node2Edge(i))
            {
                const auto &surface = solver.Geometry().EdgeSurface(incidence.edge);
                const int side = surface.nodes[0] == i ? 0 : 1;
                const DNDS::index neighbor = surface.nodes[1 - side];
                State correction = State::Zero();
                for (const auto &entry : surface.efficientStencil)
                {
                    correction += gradients[entry.node].transpose() *
                                  entry.stateGradientWeights[side].head<3>();
                }
                // Exactly the actual state before the final PreservePhysical call.
                const State face = point + factor * correction / surface.measure;
                const DNDS::real faceExcess = stats[2].Add(
                    face, lower, upper, pressureLower, pressureUpper, cfg.physics.gamma);
                record(1, face, faceExcess, halo.LocalToGlobal(neighbor));
                if (factor == 0)
                {
                    thetaCounts[7]++;
                    zeroFallbackMaximum[1] = std::max(zeroFallbackMaximum[1], (face - mean).cwiseAbs().maxCoeff());
                }
            }
        }
        std::array<DNDS::index, 8> globalThetaCounts{};
        std::array<DNDS::real, 2> globalZeroFallbackMaximum{};
        MPI_Reduce(thetaCounts.data(), globalThetaCounts.data(), 8,
                   DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(zeroFallbackMaximum.data(), globalZeroFallbackMaximum.data(), 2,
                   DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        nlohmann::ordered_json reduced;
        const std::array<std::string, 3> names{"mean", "limited_point", "limited_interface_before_final_PP"};
        for (int i = 0; i < 3; i++)
        {
            reduced[names[i]] = stats[i].Reduce(mpi);
        }
        const std::string rankPath = std::string(argv[3]) + ".rank" + std::to_string(mpi.rank) + ".json";
        std::ofstream rankOutput(rankPath);
        DNDS_check_throw_info(rankOutput.good(), "Cannot open final audit rank JSON");
        rankOutput << nlohmann::ordered_json{
            {"rank", mpi.rank}, {"worst_point_bound_example", examples[0]},
            {"worst_interface_bound_example", examples[1]}}.dump(2) << std::endl;
        if (mpi.rank == 0)
        {
            nlohmann::ordered_json result{
                {"mpi_ranks", mpi.size}, {"restart_input", cfg.io.restartInput},
                {"mesh_file", cfg.mesh.meshFile}, {"periodic_lengths", cfg.mesh.periodicLengths},
                {"gamma", cfg.physics.gamma},
                {"limiter_settings", {{"enableLimiter", cfg.reconstruction.enableLimiter},
                                      {"limitPointRecovery", cfg.reconstruction.limitPointRecovery},
                                      {"limitPressure", cfg.reconstruction.limitPressure}}},
                {"iteration", solver.CurrentIteration()}, {"physical_time", solver.SimulationTime()},
                {"residual_rms", residual}, {"absolute_bound_tolerance", tolerance},
                {"conservative_component_order", {"rho", "rhou", "rhov", "rhow", "rhoE"}},
                {"theta_count_order", {"nodes", "equal_zero", "below_1e_minus_12", "below_0.1",
                                        "below_0.5", "equal_one", "zero_point_samples", "zero_interface_sides"}},
                {"theta_counts", globalThetaCounts},
                {"zero_theta_point_vs_mean_abs_max", globalZeroFallbackMaximum[0]},
                {"zero_theta_interface_vs_mean_abs_max", globalZeroFallbackMaximum[1]},
                {"samples", reduced},
                {"pressure_bounds_note", "Diagnostic only: a conservative component box does not imply neighbor pressure bounds"},
                {"interface_sampling_note", "One sample for every owned anchor side; both sides of an internal edge are included"},
                {"rank_examples_pattern", std::string(argv[3]) + ".rank<RANK>.json"}};
            std::ofstream output(argv[3]);
            DNDS_check_throw_info(output.good(), "Cannot open final limiter audit summary");
            output << result.dump(2) << std::endl;
            std::cout << result.dump(2) << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "final limiter audit rank " << mpi.rank << ": " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    MPI_Finalize();
}
