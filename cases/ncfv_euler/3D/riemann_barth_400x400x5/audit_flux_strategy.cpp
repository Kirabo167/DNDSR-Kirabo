/** Read-only audit of conservative LLF/high-order convex flux limiting.
 *
 * This audits one saved-state RHS and an optional supplied forward-Euler dt.
 * Production SSPRK3 holds the first-stage dt fixed for later stages, and the
 * spatial operator now checks the convex CFL condition at every RHS call.
 * This independent audit records the weight sum at one saved state only.
 */
#include "NCFV/NCFVSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

using DNDS::DNDS_MPI_INDEX;
using DNDS::DNDS_MPI_REAL;

namespace
{
    using State = Eigen::Matrix<DNDS::real, 5, 1>;
    using Vector = Eigen::Matrix<DNDS::real, 3, 1>;
    constexpr DNDS::real tolerance = 1e-9;

    DNDS::real Pressure(const State &state, DNDS::real gamma)
    {
        return (gamma - 1) * (state(4) - .5 * state.segment<3>(1).squaredNorm() / state(0));
    }

    DNDS::real Speed(const State &left, const State &right, const Vector &normal, DNDS::real gamma)
    {
        const auto speed = [&](const State &state)
        {
            return std::abs(state.segment<3>(1).dot(normal) / state(0)) +
                   std::sqrt(gamma * Pressure(state, gamma) / state(0));
        };
        return std::max(speed(left), speed(right));
    }

    State PhysicalFlux(const State &state, const Vector &normal, DNDS::real gamma)
    {
        const DNDS::real velocity = state.segment<3>(1).dot(normal) / state(0);
        const DNDS::real pressure = Pressure(state, gamma);
        State flux = velocity * state;
        flux.segment<3>(1) += pressure * normal;
        flux(4) += pressure * velocity;
        return flux;
    }

    State PreservePhysical(const State &candidate, const State &anchor, DNDS::real gamma)
    {
        const auto physical = [&](const State &state)
        {
            return state.allFinite() && state(0) > 1e-12 && Pressure(state, gamma) > 1e-12;
        };
        if (physical(candidate))
            return candidate;
        DNDS_check_throw_info(physical(anchor), "Nonphysical high-flux audit anchor");
        DNDS::real lower = 0, upper = 1;
        for (int iteration = 0; iteration < 40; iteration++)
        {
            const DNDS::real middle = .5 * (lower + upper);
            if (physical(anchor + middle * (candidate - anchor)))
                lower = middle;
            else
                upper = middle;
        }
        return anchor + (.95 * lower) * (candidate - anchor);
    }

    State RebuildHighFlux(const DNDS::NCFV::EdgeControlSurface &surface,
                          const DNDS::NCFV::SpatialOperator<3> &spatial,
                          const DNDS::NCFV::PhysicsSettings &physics)
    {
        std::array<State, 2> integrated{}, surfaceMeans{};
        for (int side = 0; side < 2; side++)
        {
            const auto node = surface.nodes[side];
            const State point = spatial.PointValues()[node];
            State fluxCorrection = State::Zero(), stateCorrection = State::Zero();
            for (const auto &entry : surface.efficientStencil)
            {
                for (int derivative = 0; derivative < 3; derivative++)
                    for (int fluxDirection = 0; fluxDirection < 3; fluxDirection++)
                        fluxCorrection += entry.fluxGradientWeights[side](derivative, fluxDirection) *
                                          spatial.PhysicalFluxGradients()[entry.node].col(3 * derivative + fluxDirection);
                stateCorrection += spatial.Gradients()[entry.node].transpose() *
                                   entry.stateGradientWeights[side].head<3>();
            }
            const DNDS::real theta = spatial.LimiterFactors()[node](0);
            integrated[side] = PhysicalFlux(point, surface.vectorMeasure.head<3>(), physics.gamma) +
                               theta * fluxCorrection;
            surfaceMeans[side] = PreservePhysical(
                (surface.measure * point + theta * stateCorrection) / surface.measure, point, physics.gamma);
        }
        const Vector normal = surface.vectorMeasure.head<3>().normalized();
        State numerical = State::Zero();
        Vector gridVelocity = Vector::Zero();
        DNDS::real acousticMinus = 0, contact = 0, acousticPlus = 0;
        DNDS::Euler::Gas::InviscidFlux_IdealGas_Dispatcher<3>(
            physics.riemannSolver, surfaceMeans[0], surfaceMeans[1], surfaceMeans[0], surfaceMeans[1],
            gridVelocity, normal, physics.gamma, physics.gamma, numerical, 0.0, 1.0, 1.0,
            []() {}, acousticMinus, contact, acousticPlus);
        const State central = .5 * (PhysicalFlux(surfaceMeans[0], normal, physics.gamma) +
                                    PhysicalFlux(surfaceMeans[1], normal, physics.gamma));
        return .5 * (integrated[0] + integrated[1]) + surface.measure * (numerical - central);
    }

    State Primitive(const std::vector<DNDS::real> &value, DNDS::real gamma)
    {
        State state;
        state(0) = value[0];
        for (int i = 0; i < 3; i++)
            state(i + 1) = value[0] * value[i + 1];
        state(4) = value[4] / (gamma - 1) + .5 * state.segment<3>(1).squaredNorm() / state(0);
        return state;
    }

    State Exterior(const State &inside, const Vector &normal,
                   const DNDS::NCFV::BoundaryZoneSettings &boundary, DNDS::real gamma)
    {
        using DNDS::NCFV::BoundaryMode;
        switch (boundary.mode)
        {
        case BoundaryMode::FarField:
        case BoundaryMode::SupersonicInlet:
            return Primitive(boundary.primitive, gamma);
        case BoundaryMode::SupersonicOutlet:
        case BoundaryMode::Periodic:
            return inside;
        case BoundaryMode::PressureOutlet:
        {
            State outside = inside;
            outside(4) += (boundary.staticPressure - Pressure(inside, gamma)) / (gamma - 1);
            return outside;
        }
        case BoundaryMode::SlipWall:
        case BoundaryMode::Symmetry:
        {
            State outside = inside;
            outside.segment<3>(1) -= 2 * inside.segment<3>(1).dot(normal) * normal;
            return outside;
        }
        case BoundaryMode::NoSlipAdiabaticWall:
        case BoundaryMode::NoSlipIsothermalWall:
        {
            State outside = inside;
            for (int i = 0; i < 3; i++)
                outside(i + 1) = 2 * inside(0) * boundary.wallVelocity[i] - inside(i + 1);
            outside(4) += .5 * (outside.segment<3>(1).squaredNorm() -
                               inside.segment<3>(1).squaredNorm()) / inside(0);
            return outside;
        }
        }
        DNDS_check_throw_info(false, "Unsupported boundary mode in flux audit");
        return inside;
    }

    std::array<DNDS::real, 5> Components(const State &state)
    {
        return {state(0), state(1), state(2), state(3), state(4)};
    }

    struct Statistics
    {
        DNDS::index samples = 0, nonfinite = 0, nonpositiveDensity = 0, nonpositivePressure = 0;
        DNDS::index pressureLowerViolations = 0;
        std::array<DNDS::index, 5> boundViolations{};
        std::array<DNDS::real, 5> maximumViolation{};
        DNDS::real minimumDensity = 1e300, minimumPressure = 1e300, maximumPressureViolation = 0;

        void Add(const State &state, const State &lower, const State &upper,
                 DNDS::real pressureLower, DNDS::real gamma)
        {
            samples++;
            nonfinite += !state.allFinite();
            const DNDS::real pressure = Pressure(state, gamma);
            nonpositiveDensity += !(state(0) > 0);
            nonpositivePressure += !(pressure > 0);
            minimumDensity = std::min(minimumDensity, state(0));
            minimumPressure = std::min(minimumPressure, pressure);
            for (int component = 0; component < 5; component++)
            {
                const DNDS::real excess = std::max(
                    {DNDS::real(0), lower(component) - state(component), state(component) - upper(component)});
                boundViolations[component] += excess > tolerance;
                maximumViolation[component] = std::max(maximumViolation[component], excess);
            }
            const DNDS::real excess = std::max(DNDS::real(0), pressureLower - pressure);
            pressureLowerViolations += excess > tolerance;
            maximumPressureViolation = std::max(maximumPressureViolation, excess);
        }

        nlohmann::ordered_json Reduce(const DNDS::MPIInfo &mpi) const
        {
            const std::array<DNDS::index, 5> counts{
                samples, nonfinite, nonpositiveDensity, nonpositivePressure, pressureLowerViolations};
            std::array<DNDS::index, 5> globalCounts{}, globalViolations{};
            std::array<DNDS::real, 5> globalMaximum{};
            const std::array<DNDS::real, 2> minima{minimumDensity, minimumPressure};
            std::array<DNDS::real, 2> globalMinima{};
            DNDS::real globalPressureViolation = 0;
            MPI_Reduce(counts.data(), globalCounts.data(), 5, DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
            MPI_Reduce(boundViolations.data(), globalViolations.data(), 5, DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
            MPI_Reduce(maximumViolation.data(), globalMaximum.data(), 5, DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
            MPI_Reduce(minima.data(), globalMinima.data(), 2, DNDS_MPI_REAL, MPI_MIN, 0, mpi.comm);
            MPI_Reduce(&maximumPressureViolation, &globalPressureViolation, 1, DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
            return {{"samples", globalCounts[0]}, {"nonfinite", globalCounts[1]},
                    {"nonpositive_density", globalCounts[2]}, {"nonpositive_pressure", globalCounts[3]},
                    {"conservative_bound_violation_count", globalViolations},
                    {"conservative_bound_violation_max", globalMaximum},
                    {"minimum_density", globalMinima[0]}, {"minimum_pressure", globalMinima[1]},
                    {"pressure_lower_violation_count", globalCounts[4]},
                    {"pressure_lower_violation_max", globalPressureViolation}};
        }
    };
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    try
    {
        DNDS_check_throw_info(argc == 4 || argc == 5,
                              "usage: config restart-base-or-minus-for-initial output-summary.json [forward-euler-dt]");
        auto cfg = DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
        cfg.io.restartInput = std::string(argv[2]) == "-" ? "" : argv[2];
        cfg.time.iterations = 0;
        cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
        cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
        cfg.io.restartInterval = 0;
        DNDS_check_throw_info(cfg.reconstruction.limitFluxCorrection,
                              "Flux audit requires reconstruction.limitFluxCorrection=true");
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
                field[i].setZero();
        };
        allocate(means, "fluxAudit.means");
        allocate(rhs, "fluxAudit.rhs");
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
            means[i] = solver.StateField()[i];
        DNDS::NCFV::SpatialOperator<3> spatial(
            mpi, mesh, solver.EdgeTopology(), solver.Geometry(), solver.ReconstructionData(),
            solver.Boundaries(), cfg.algorithm.mode, cfg.reconstruction, cfg.physics, cfg.time, halo);
        spatial.Initialize();
        const DNDS::real residual = spatial.EvaluateRHS(means, rhs);
        const DNDS::real recomputedTimeStep = spatial.LastMinimumTimeStep();
        const DNDS::real auditTimeStep = argc == 5 ? std::stod(argv[4]) : recomputedTimeStep;
        DNDS_check_throw_info(auditTimeStep > 0 && std::isfinite(auditTimeStep), "Invalid audit dt");
        const auto &bounds = spatial.FluxCorrectionBounds();
        const auto &factors = spatial.EdgeFluxCorrectionFactors();
        std::array<DNDS::real, 15> gradientMaximum{}, globalGradientMaximum{};
        std::ofstream initialNodes;
        if (cfg.io.restartInput.empty())
        {
            initialNodes.open(std::string(argv[3]) + ".initial_nodes.rank" + std::to_string(mpi.rank) + ".csv");
            DNDS_check_throw_info(initialNodes.good(), "Cannot open initial gradient diagnostic CSV");
            initialNodes << std::setprecision(17) << "original_node,x,y,z";
            for (const std::string prefix : {"mean_", "point_"})
                for (const std::string component : {"rho", "rhou", "rhov", "rhow", "rhoE"})
                    initialNodes << ',' << prefix << component;
            for (const std::string direction : {"x", "y", "z"})
                for (const std::string component : {"rho", "rhou", "rhov", "rhow", "rhoE"})
                    initialNodes << ",d_" << component << "_d" << direction;
            initialNodes << '\n';
        }
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
        {
            const auto gradient = spatial.Gradients()[node];
            for (int direction = 0; direction < 3; direction++)
                for (int component = 0; component < 5; component++)
                    gradientMaximum[5 * direction + component] = std::max(
                        gradientMaximum[5 * direction + component], std::abs(gradient(direction, component)));
            if (initialNodes.is_open())
            {
                const auto coordinate = halo.Coordinate(node);
                initialNodes << mesh->node2nodeOrig(node, 0) << ',' << coordinate(0) << ',' << coordinate(1) << ',' << coordinate(2);
                for (int component = 0; component < 5; component++)
                    initialNodes << ',' << means[node](component);
                for (int component = 0; component < 5; component++)
                    initialNodes << ',' << spatial.PointValues()[node](component);
                for (int direction = 0; direction < 3; direction++)
                    for (int component = 0; component < 5; component++)
                        initialNodes << ',' << gradient(direction, component);
                initialNodes << '\n';
            }
        }
        MPI_Reduce(gradientMaximum.data(), globalGradientMaximum.data(), 15,
                   DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        Statistics correctedBars, lowBars, eulerUpdate;
        // edges, exactly zero, exactly one, below .1, below .5.
        std::array<DNDS::index, 5> counts{};
        DNDS::real factorSum = 0, maximumFluxAssemblyError = 0, maximumNormalNorm = 0;
        for (DNDS::index edge = 0; edge < solver.EdgeTopology().NumEdge(); edge++)
        {
            const auto &surface = solver.Geometry().EdgeSurface(edge);
            const State left = means[surface.nodes[0]], right = means[surface.nodes[1]];
            const Vector normal = surface.vectorMeasure.head<3>() / surface.measure;
            const DNDS::real speed = Speed(left, right, normal, cfg.physics.gamma);
            const DNDS::real dissipation = speed * surface.measure;
            const State lowFlux = .5 * surface.measure *
                                  (PhysicalFlux(left, normal, cfg.physics.gamma) +
                                   PhysicalFlux(right, normal, cfg.physics.gamma) - speed * (right - left));
            const State correction = RebuildHighFlux(surface, spatial, cfg.physics) - lowFlux;
            const State bar = .5 * (left + right) -
                              .5 / speed * (PhysicalFlux(right, normal, cfg.physics.gamma) -
                                            PhysicalFlux(left, normal, cfg.physics.gamma));
            const DNDS::real factor = factors[edge](0);
            counts[0]++;
            counts[1] += factor == 0;
            counts[2] += factor == 1;
            counts[3] += factor < .1;
            counts[4] += factor < .5;
            factorSum += factor;
            maximumNormalNorm = std::max(maximumNormalNorm, normal.norm());
            maximumFluxAssemblyError = std::max(maximumFluxAssemblyError,
                (State(spatial.EdgeFluxes()[edge]) - lowFlux - factor * correction).cwiseAbs().maxCoeff());
            for (int side = 0; side < 2; side++)
            {
                const auto node = surface.nodes[side];
                const State lower = cfg.reconstruction.fluxCorrectionPairBounds ?
                                        State(left.cwiseMin(right).cwiseMin(bar)) :
                                        State(bounds[node].topRows<5>().col(0));
                const State upper = cfg.reconstruction.fluxCorrectionPairBounds ?
                                        State(left.cwiseMax(right).cwiseMax(bar)) :
                                        State(bounds[node].topRows<5>().col(1));
                const DNDS::real pressureLower = cfg.reconstruction.fluxCorrectionPairBounds ?
                    std::min({Pressure(left, cfg.physics.gamma), Pressure(right, cfg.physics.gamma), Pressure(bar, cfg.physics.gamma)}) :
                    bounds[node](5, 0);
                const State corrected = bar + (side == 0 ? -1.0 : 1.0) * factor * correction / dissipation;
                correctedBars.Add(corrected, lower, upper, pressureLower, cfg.physics.gamma);
                lowBars.Add(bar, lower, upper, pressureLower, cfg.physics.gamma);
            }
        }
        State rhsIntegral = State::Zero(), boundaryIntegral = State::Zero();
        DNDS::real maximumCFL = 0, maximumRecomputedCFL = 0, maximumClosureError = 0;
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
        {
            const State self = means[node];
            const auto &volumeData = solver.Geometry().NodeVolume(node);
            const DNDS::real volume = volumeData.moments.measure;
            DNDS::real dissipationSum = 0, measureSum = 0;
            Vector closure = Vector::Zero();
            for (const auto &incidence : solver.EdgeTopology().Node2Edge(node))
            {
                const auto &surface = solver.Geometry().EdgeSurface(incidence.edge);
                const auto otherNode = surface.nodes[surface.nodes[0] == node ? 1 : 0];
                const Vector normal = incidence.outwardSign * surface.vectorMeasure.head<3>() / surface.measure;
                dissipationSum += surface.measure * Speed(self, State(means[otherNode]), normal, cfg.physics.gamma);
                measureSum += surface.measure;
                closure += incidence.outwardSign * surface.vectorMeasure.head<3>();
            }
            for (const auto &piece : volumeData.boundaryPieces)
            {
                const auto &boundary = solver.Boundaries().Get(piece.zone);
                if (boundary.mode == DNDS::NCFV::BoundaryMode::Periodic)
                    continue;
                const Vector normal = piece.vectorMeasure.head<3>() / piece.measure;
                const State other = Exterior(self, normal, boundary, cfg.physics.gamma);
                const DNDS::real speed = Speed(self, other, normal, cfg.physics.gamma);
                const State bar = .5 * (self + other) -
                                  .5 / speed * (PhysicalFlux(other, normal, cfg.physics.gamma) -
                                                PhysicalFlux(self, normal, cfg.physics.gamma));
                const State lower = bounds[node].topRows<5>().col(0);
                const State upper = bounds[node].topRows<5>().col(1);
                lowBars.Add(bar, lower, upper, bounds[node](5, 0), cfg.physics.gamma);
                correctedBars.Add(bar, lower, upper, bounds[node](5, 0), cfg.physics.gamma);
                boundaryIntegral += .5 * piece.measure *
                                    (PhysicalFlux(self, normal, cfg.physics.gamma) +
                                     PhysicalFlux(other, normal, cfg.physics.gamma) - speed * (other - self));
                dissipationSum += piece.measure * speed;
                measureSum += piece.measure;
                closure += piece.vectorMeasure.head<3>();
                maximumNormalNorm = std::max(maximumNormalNorm, normal.norm());
            }
            rhsIntegral += volume * State(rhs[node]);
            maximumCFL = std::max(maximumCFL, auditTimeStep * dissipationSum / volume);
            maximumRecomputedCFL = std::max(maximumRecomputedCFL, recomputedTimeStep * dissipationSum / volume);
            maximumClosureError = std::max(maximumClosureError, closure.norm() / measureSum);
            eulerUpdate.Add(self + auditTimeStep * State(rhs[node]),
                            State(bounds[node].topRows<5>().col(0)), State(bounds[node].topRows<5>().col(1)),
                            bounds[node](5, 0), cfg.physics.gamma);
        }
        std::array<DNDS::index, 5> globalCounts{};
        DNDS::real globalFactorSum = 0;
        const std::array<DNDS::real, 5> maxima{maximumCFL, maximumRecomputedCFL,
            maximumFluxAssemblyError, maximumNormalNorm, maximumClosureError};
        std::array<DNDS::real, 5> globalMaxima{};
        State globalRhs = State::Zero(), globalBoundary = State::Zero();
        MPI_Reduce(counts.data(), globalCounts.data(), 5, DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(&factorSum, &globalFactorSum, 1, DNDS_MPI_REAL, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(maxima.data(), globalMaxima.data(), 5, DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        MPI_Reduce(rhsIntegral.data(), globalRhs.data(), 5, DNDS_MPI_REAL, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(boundaryIntegral.data(), globalBoundary.data(), 5, DNDS_MPI_REAL, MPI_SUM, 0, mpi.comm);
        nlohmann::ordered_json statistics{
            {"low_order_bars", lowBars.Reduce(mpi)}, {"corrected_bars", correctedBars.Reduce(mpi)},
            {"forward_euler_update", eulerUpdate.Reduce(mpi)}};
        if (mpi.rank == 0)
        {
            nlohmann::ordered_json result{
                {"mpi_ranks", mpi.size}, {"restart_input", cfg.io.restartInput},
                {"iteration", solver.CurrentIteration()}, {"physical_time", solver.SimulationTime()},
                {"limiter_settings", {{"limitFluxCorrection", cfg.reconstruction.limitFluxCorrection},
                                      {"maximumFluxCorrection", cfg.reconstruction.maximumFluxCorrection},
                                      {"fluxCorrectionPairBounds", cfg.reconstruction.fluxCorrectionPairBounds},
                                      {"completeStencilRings", cfg.reconstruction.completeStencilRings},
                                      {"enableLimiter", cfg.reconstruction.enableLimiter},
                                      {"limitPointRecovery", cfg.reconstruction.limitPointRecovery},
                                      {"limitPressure", cfg.reconstruction.limitPressure}}},
                {"residual_rms", residual}, {"absolute_bound_tolerance", tolerance},
                {"conservative_component_order", {"rho", "rhou", "rhov", "rhow", "rhoE"}},
                {"raw_gradient_max_abs_flat_xyz_then_components", globalGradientMaximum},
                {"factor_count_order", {"owned_edges", "equal_zero", "equal_one", "below_0.1", "below_0.5"}},
                {"factor_counts", globalCounts}, {"factor_mean", globalFactorSum / globalCounts[0]},
                {"snapshot_recomputed_dt", recomputedTimeStep}, {"audit_forward_euler_dt", auditTimeStep},
                {"maximum_cfl_weight_sum", globalMaxima[0]},
                {"maximum_recomputed_cfl_weight_sum", globalMaxima[1]},
                {"maximum_flux_assembly_error", globalMaxima[2]},
                {"maximum_normal_norm", globalMaxima[3]}, {"maximum_relative_geometry_closure_error", globalMaxima[4]},
                {"sum_volume_rhs", Components(globalRhs)}, {"sum_boundary_flux", Components(globalBoundary)},
                {"conservation_closure_sum_volume_rhs_plus_boundary_flux", Components(globalRhs + globalBoundary)},
                {"samples", statistics},
                {"cfl_note", "Independent weight sum for one saved state. Production SSPRK3 keeps first-stage dt fixed, and the spatial operator separately rejects any later RHS whose actual convex CFL weight sum exceeds one."},
                {"boundary_note", "Includes every nonperiodic boundary piece with the same mean-state LLF flux used by convex mode."},
                {"pressure_note", "Pressure lower bounds only; pressure upper sets are not convex in conservative variables."}};
            std::ofstream output(argv[3]);
            DNDS_check_throw_info(output.good(), "Cannot open flux strategy audit summary");
            output << result.dump(2) << std::endl;
            std::cout << result.dump(2) << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "flux strategy audit rank " << mpi.rank << ": " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    MPI_Finalize();
}
