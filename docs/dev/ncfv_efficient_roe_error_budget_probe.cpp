/**
 * @file efficient_roe_error_budget_probe.cpp
 * @brief Isolate the Roe dissipative correction in either NCFV mode without
 *        modifying the production solver.
 *
 * The diagnostic calls the production reconstruction and RHS at every RK
 * stage.  It independently reproduces the selected mode's Roe correction,
 * checks RHS = central + correction, and scales that correction by alpha.
 */
#include "NCFV/NCFVSolver.hpp"
#include "NCFV/NCFVAnalytic.hpp"
#include "Geom/Quadrature.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

using DNDS::real;
using Index = DNDS::index;
using namespace DNDS::NCFV;
using State = Eigen::Matrix<real, 5, 1>;
using Vector = Eigen::Matrix<real, 3, 1>;

namespace
{
    State PhysicalFlux(const State &state, const Vector &normal, real gamma)
    {
        const Vector momentum = state.segment<3>(1);
        const Vector velocity = momentum / state(0);
        const real pressure = (gamma - 1) *
                              (state(4) - 0.5 * momentum.squaredNorm() / state(0));
        const real normalVelocity = velocity.dot(normal);
        State flux;
        flux(0) = momentum.dot(normal);
        flux.segment<3>(1) = momentum * normalVelocity + pressure * normal;
        flux(4) = (state(4) + pressure) * normalVelocity;
        return flux;
    }

    real Pressure(const State &state, real gamma)
    {
        return (gamma - 1) *
               (state(4) - 0.5 * state.segment<3>(1).squaredNorm() / state(0));
    }

    State RoeFlux(const State &left, const State &right,
                  const Vector &normal, real gamma)
    {
        State flux = State::Zero();
        real acousticMinus = 0, contact = 0, acousticPlus = 0;
        DNDS::Euler::Gas::InviscidFlux_IdealGas_Dispatcher<3>(
            DNDS::Euler::Gas::Roe, left, right, left, right,
            Vector::Zero(), normal, gamma, gamma, flux,
            0.0, 1.0, 1.0, []() {}, acousticMinus, contact, acousticPlus);
        return flux;
    }

    void Allocate(NodeStatePair &field, const char *name, int nVars,
                  const DNDS::MPIInfo &mpi, Index owned, const NodeHalo &halo)
    {
        field.InitPair(name, mpi);
        field.father->Resize(owned, nVars, 1);
        field.son->Resize(halo.NumNodeGhost(), nVars, 1);
        field.BorrowSetup(halo.Layout());
        field.trans.initPersistentPull();
        for (Index i = 0; i < field.Size(); i++)
            field[i].setZero();
    }

    struct DissipationAssembler
    {
        const DNDS::MPIInfo &mpi;
        const DNDS::Geom::UnstructuredMesh &mesh;
        const Topology &topology;
        const DualGeometry &geometry;
        const SpatialOperator<3> &spatial;
        const Reconstruction &reconstruction;
        const NodeHalo &halo;
        IntegrationMode mode;
        real gamma;
        NodeStatePair edgeFlux;
        NodeStatePair edgeCentral;
        real lastRhsClosure = 0;

        DissipationAssembler(const DNDS::MPIInfo &mpiIn,
                             const DNDS::Geom::UnstructuredMesh &meshIn,
                             const Topology &topologyIn,
                             const DualGeometry &geometryIn,
                             const SpatialOperator<3> &spatialIn,
                             const Reconstruction &reconstructionIn,
                             const NodeHalo &haloIn,
                             IntegrationMode modeIn,
                             real gammaIn)
            : mpi(mpiIn), mesh(meshIn), topology(topologyIn),
              geometry(geometryIn), spatial(spatialIn),
              reconstruction(reconstructionIn), halo(haloIn),
              mode(modeIn), gamma(gammaIn)
        {
            edgeFlux.InitPair("roeBudget.edgeFlux", mpi);
            edgeFlux.father->Resize(topology.NumEdge(), 5, 1);
            edgeFlux.son->Resize(topology.NumEdgeGhost(), 5, 1);
            edgeFlux.BorrowSetup(
                const_cast<DNDS::Geom::tAdjPair &>(topology.Edge2Node()));
            edgeFlux.trans.initPersistentPull();
            edgeCentral.InitPair("roeBudget.edgeCentral", mpi);
            edgeCentral.father->Resize(topology.NumEdge(), 5, 1);
            edgeCentral.son->Resize(topology.NumEdgeGhost(), 5, 1);
            edgeCentral.BorrowSetup(
                const_cast<DNDS::Geom::tAdjPair &>(topology.Edge2Node()));
            edgeCentral.trans.initPersistentPull();
        }

        std::vector<State> Evaluate(const NodeStatePair &rhs)
        {
            const auto &points = spatial.PointValues();
            const auto &gradients = spatial.Gradients();
            const auto &fluxGradients = spatial.PhysicalFluxGradients();
            const auto &coefficients = spatial.Coefficients();
            for (Index e = 0; e < topology.NumEdge(); e++)
            {
                const EdgeControlSurface &surface = geometry.EdgeSurface(e);
                if (mode == IntegrationMode::TraditionalQuadrature)
                {
                    edgeCentral[e].setZero();
                    for (const auto &point : surface.quadrature)
                    {
                        State states[2];
                        for (int side = 0; side < 2; side++)
                        {
                            const Index node = surface.nodes[side];
                            states[side] = State(points[node]) +
                                coefficients[node].transpose() *
                                Reconstruction::EvaluateBasis(
                                    halo.DisplacementToPoint(node, point.coordinate),
                                    reconstruction.ReferenceLengths(node), 3);
                            DNDS_check_throw_info(
                                states[side](0) > 0 &&
                                    Pressure(states[side], gamma) > 0,
                                "Traditional Roe budget found a nonphysical quadrature state");
                        }
                        const Vector normal = point.vectorWeight / point.weight;
                        const State centered = 0.5 *
                            (PhysicalFlux(states[0], normal, gamma) +
                             PhysicalFlux(states[1], normal, gamma));
                        edgeCentral[e] += point.weight * centered;
                    }
                    edgeFlux[e] = State(spatial.EdgeFluxes()[e]) -
                                  State(edgeCentral[e]);
                    continue;
                }
                const Vector normal = surface.vectorMeasure.normalized();
                State means[2];
                State integrated[2];
                for (int side = 0; side < 2; side++)
                {
                    means[side] = surface.measure * State(points[surface.nodes[side]]);
                    for (const EfficientSurfaceNode &entry : surface.efficientStencil)
                        means[side] += gradients[entry.node].transpose() *
                                       entry.stateGradientWeights[side].head<3>();
                    means[side] /= surface.measure;
                    DNDS_check_throw_info(
                        means[side](0) > 0 && Pressure(means[side], gamma) > 0,
                        "Roe budget found a nonphysical smooth-vortex face mean");
                    integrated[side] = PhysicalFlux(
                        State(points[surface.nodes[side]]),
                        surface.vectorMeasure, gamma);
                    for (const EfficientSurfaceNode &entry : surface.efficientStencil)
                        for (int derivative = 0; derivative < 3; derivative++)
                            for (int fluxDirection = 0; fluxDirection < 3; fluxDirection++)
                                integrated[side] +=
                                    entry.fluxGradientWeights[side](derivative, fluxDirection) *
                                    fluxGradients[entry.node].col(
                                        derivative * 3 + fluxDirection);
                }
                edgeCentral[e] = 0.5 * (integrated[0] + integrated[1]);
                const State centered = 0.5 *
                    (PhysicalFlux(means[0], normal, gamma) +
                     PhysicalFlux(means[1], normal, gamma));
                edgeFlux[e] = surface.measure *
                    (RoeFlux(means[0], means[1], normal, gamma) - centered);
            }
            edgeFlux.trans.startPersistentPull();
            edgeFlux.trans.waitPersistentPull();
            edgeCentral.trans.startPersistentPull();
            edgeCentral.trans.waitPersistentPull();

            std::vector<State> result(static_cast<std::size_t>(mesh.NumNode()),
                                      State::Zero());
            real localMismatch = 0;
            for (Index i = 0; i < mesh.NumNode(); i++)
            {
                State central = State::Zero();
                for (const auto &incidence : topology.Node2Edge(i))
                {
                    result[static_cast<std::size_t>(i)] -=
                        incidence.outwardSign * State(edgeFlux[incidence.edge]) /
                        geometry.NodeVolume(i).moments.measure;
                    central -= incidence.outwardSign *
                               State(edgeCentral[incidence.edge]) /
                               geometry.NodeVolume(i).moments.measure;
                }
                localMismatch = std::max(
                    localMismatch,
                    (State(rhs[i]) - central - result[static_cast<std::size_t>(i)])
                        .cwiseAbs().maxCoeff());
            }
            MPI_Allreduce(&localMismatch, &lastRhsClosure, 1,
                          DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(lastRhsClosure < 1e-8,
                                  "Independent Roe/central decomposition does not match production RHS");
            return result;
        }
    };

    void SetAnalyticState(NodeStatePair &field, const DNDS::Geom::UnstructuredMesh &mesh,
                          const DualGeometry &geometry, const NodeHalo &halo,
                          const Configuration &configuration, real time)
    {
        std::vector<State> exactPoints(static_cast<std::size_t>(halo.NumNodeProc()));
        std::vector<Eigen::Matrix<real, 3, 5>> exactGradients(
            static_cast<std::size_t>(halo.NumNodeProc()));
        for (Index i = 0; i < halo.NumNodeProc(); i++)
            std::tie(exactPoints[static_cast<std::size_t>(i)],
                     exactGradients[static_cast<std::size_t>(i)]) =
                IsentropicVortex<3>(configuration, halo.Coordinate(i), time);
        for (Index i = 0; i < mesh.NumNode(); i++)
        {
            if (configuration.algorithm.mode == IntegrationMode::TraditionalQuadrature)
            {
                field[i].setZero();
                const auto &volume = geometry.NodeVolume(i);
                for (const auto &point : volume.volumeQuadrature)
                    field[i] += point.weight / volume.moments.measure *
                        IsentropicVortex<3>(configuration, point.coordinate, time).first;
                continue;
            }
            field[i] = exactPoints[static_cast<std::size_t>(i)];
            for (const auto &entry : geometry.NodeVolume(i).pointRecoveryStencil)
                field[i] += exactGradients[static_cast<std::size_t>(entry.node)].transpose() *
                            entry.gradientWeight;
        }
    }

    nlohmann::ordered_json StaticBudget(
        real time, NodeStatePair &field, NodeStatePair &rhs,
        SpatialOperator<3> &spatial, DissipationAssembler &assembler,
        const DNDS::Geom::UnstructuredMesh &mesh, const DualGeometry &geometry,
        const NodeHalo &halo, const Configuration &configuration,
        const DNDS::MPIInfo &mpi)
    {
        SetAnalyticState(field, mesh, geometry, halo, configuration, time);
        spatial.EvaluateRHS(field, rhs);
        const auto dissipation = assembler.Evaluate(rhs);
        DNDS::Geom::Elem::Quadrature quadrature({DNDS::Geom::Elem::Tet4}, 5);
        real local[12]{}, global[12]{};
        for (Index i = 0; i < mesh.NumNode(); i++)
        {
            const auto &volume = geometry.NodeVolume(i);
            const real measure = volume.moments.measure;
            real exactDerivative = 0;
            for (const auto &micro : volume.microVolumes)
                for (int q = 0; q < quadrature.GetNumPoints(); q++)
                {
                    const auto [p, weight] = quadrature.GetQuadraturePointInfo(q);
                    const Vector x = micro.points[0] +
                        p[0] * (micro.points[1] - micro.points[0]) +
                        p[1] * (micro.points[2] - micro.points[0]) +
                        p[2] * (micro.points[3] - micro.points[0]);
                    const auto [exactPoint, exactGradient] =
                        IsentropicVortex<3>(configuration, x, time);
                    static_cast<void>(exactPoint);
                    exactDerivative -= 6 * micro.measure * weight / measure *
                                       (exactGradient(0, 0) + exactGradient(1, 0));
                }
            const real centerError = rhs[i](0) - dissipation[static_cast<std::size_t>(i)](0) -
                                     exactDerivative;
            const real diss = dissipation[static_cast<std::size_t>(i)](0);
            const real totalError = centerError + diss;
            const Vector coordinate = mesh.coords[i];
            const real dx = std::remainder(coordinate.x() - 5.0 - time, 10.0);
            const real dy = std::remainder(coordinate.y() - 5.0 - time, 10.0);
            const bool core = dx * dx + dy * dy < 9.0;
            local[0] += measure;
            local[1] += measure * centerError * centerError;
            local[2] += measure * diss * diss;
            local[3] += measure * totalError * totalError;
            local[4] += measure * centerError * diss;
            local[5] += measure * diss;
            if (core)
            {
                local[6] += measure;
                local[7] += measure * centerError * centerError;
                local[8] += measure * diss * diss;
                local[9] += measure * totalError * totalError;
                local[10] += measure * centerError * diss;
                local[11] += measure * diss;
            }
        }
        MPI_Allreduce(local, global, 12, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        nlohmann::ordered_json result;
        result["time"] = time;
        result["rhs_decomposition_max_mismatch"] = assembler.lastRhsClosure;
        for (int region = 0; region < 2; region++)
        {
            const int base = region * 6;
            const real v = global[base];
            auto &section = result[region ? "core_r_lt_3" : "full_domain"];
            section["volume"] = v;
            section["center_error_L2V"] = std::sqrt(global[base + 1] / v);
            section["roe_dissipation_L2V"] = std::sqrt(global[base + 2] / v);
            section["total_error_L2V"] = std::sqrt(global[base + 3] / v);
            section["center_dissipation_dot_per_volume"] = global[base + 4] / v;
            section["center_dissipation_cosine"] =
                global[base + 4] /
                std::sqrt(std::max(global[base + 1] * global[base + 2], 1e-300));
            section["dissipation_signed_integral"] = global[base + 5];
        }
        return result;
    }

    nlohmann::ordered_json FinalError(
        const NodeStatePair &field, const SpatialOperator<3> &spatial,
        const DNDS::Geom::UnstructuredMesh &mesh, const DualGeometry &geometry,
        const Configuration &configuration, const DNDS::MPIInfo &mpi,
        real time, real initialMass)
    {
        DNDS::Geom::Elem::Quadrature referenceQuadrature(
            {DNDS::Geom::Elem::Tet4}, 6);
        real local[11]{}, global[11]{};
        real localMaximum = 0, globalMaximum = 0;
        real localMeanMaximum = 0, globalMeanMaximum = 0;
        real regionalLocal[16]{}, regionalGlobal[16]{};
        for (Index i = 0; i < mesh.NumNode(); i++)
        {
            const State state = spatial.PointValues()[i];
            const State exact = IsentropicVortex<3>(configuration, mesh.coords[i], time).first;
            const real volume = geometry.NodeVolume(i).moments.measure;
            const real rhoError = state(0) - exact(0);
            const real pressureError = Pressure(state, configuration.physics.gamma) -
                                       Pressure(exact, configuration.physics.gamma);
            State exactMean = State::Zero();
            for (const auto &micro : geometry.NodeVolume(i).microVolumes)
                for (int q = 0; q < referenceQuadrature.GetNumPoints(); q++)
                {
                    const auto [p, weight] =
                        referenceQuadrature.GetQuadraturePointInfo(q);
                    const Vector x = micro.points[0] +
                        p[0] * (micro.points[1] - micro.points[0]) +
                        p[1] * (micro.points[2] - micro.points[0]) +
                        p[2] * (micro.points[3] - micro.points[0]);
                    exactMean += 6 * micro.measure * weight / volume *
                                 IsentropicVortex<3>(configuration, x, time).first;
                }
            const real meanRhoError = field[i](0) - exactMean(0);
            const real meanPressureError =
                Pressure(State(field[i]), configuration.physics.gamma) -
                Pressure(exactMean, configuration.physics.gamma);
            DNDS_check_throw_info(state.allFinite() && state(0) > 0 &&
                                  Pressure(state, configuration.physics.gamma) > 0,
                                  "Roe budget transient produced a nonphysical recovered state");
            local[0] += volume;
            local[1] += volume * std::abs(rhoError);
            local[2] += volume * rhoError * rhoError;
            local[3] += volume * std::abs(pressureError);
            local[4] += volume * pressureError * pressureError;
            local[5] += volume * field[i](0);
            local[6] += volume * rhoError;
            local[7] += volume * std::abs(meanRhoError);
            local[8] += volume * meanRhoError * meanRhoError;
            local[9] += volume * meanPressureError * meanPressureError;
            local[10] += volume * meanRhoError;
            localMaximum = std::max(localMaximum, std::abs(rhoError));
            localMeanMaximum = std::max(localMeanMaximum, std::abs(meanRhoError));
            const real dx = std::remainder(
                mesh.coords[i](0) - configuration.initialField.vortexCenter[0] -
                    time * configuration.physics.initialPrimitive[1],
                configuration.mesh.periodicLengths[0]);
            const real dy = std::remainder(
                mesh.coords[i](1) - configuration.initialField.vortexCenter[1] -
                    time * configuration.physics.initialPrimitive[2],
                configuration.mesh.periodicLengths[1]);
            const bool regions[4]{
                dx * dx + dy * dy < 9.0,
                std::abs(dx) > configuration.mesh.periodicLengths[0] / 2 - 0.5 ||
                    std::abs(dy) > configuration.mesh.periodicLengths[1] / 2 - 0.5,
                dx * dx + dy * dy >= 9.0,
                std::min({mesh.coords[i](0),
                          configuration.mesh.periodicLengths[0] - mesh.coords[i](0),
                          mesh.coords[i](1),
                          configuration.mesh.periodicLengths[1] - mesh.coords[i](1)}) < 0.5,
            };
            for (int region = 0; region < 4; region++)
                if (regions[region])
                {
                    const int offset = 4 * region;
                    regionalLocal[offset] += volume;
                    regionalLocal[offset + 1] += volume * meanRhoError * meanRhoError;
                    regionalLocal[offset + 2] += volume * rhoError * rhoError;
                    regionalLocal[offset + 3] += volume * std::abs(meanRhoError);
                }
        }
        MPI_Allreduce(local, global, 11, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        MPI_Allreduce(regionalLocal, regionalGlobal, 16,
                      DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        MPI_Allreduce(&localMaximum, &globalMaximum, 1,
                      DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        MPI_Allreduce(&localMeanMaximum, &globalMeanMaximum, 1,
                      DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        nlohmann::ordered_json regions;
        const char *regionNames[4]{"core_r_lt_3", "splice_strip_width_0p5",
                                   "outside_core", "physical_boundary_width_0p5"};
        for (int region = 0; region < 4; region++)
        {
            const int offset = 4 * region;
            const real regionVolume = regionalGlobal[offset];
            regions[regionNames[region]] = {
                {"volume_fraction", regionVolume / global[0]},
                {"rho_mean_L1V", regionalGlobal[offset + 3] / regionVolume},
                {"rho_mean_L2V", std::sqrt(regionalGlobal[offset + 1] / regionVolume)},
                {"rho_point_L2V", std::sqrt(regionalGlobal[offset + 2] / regionVolume)},
                {"rho_mean_error_energy_fraction", regionalGlobal[offset + 1] / global[8]},
                {"rho_point_error_energy_fraction", regionalGlobal[offset + 2] / global[2]},
            };
        }
        return {
            {"time", time}, {"volume", global[0]},
            {"rho_point_L1V", global[1] / global[0]},
            {"rho_point_L2V", std::sqrt(global[2] / global[0])},
            {"rho_point_Linf", globalMaximum},
            {"pressure_point_L1V", global[3] / global[0]},
            {"pressure_point_L2V", std::sqrt(global[4] / global[0])},
            {"mass_drift", global[5] - initialMass},
            {"rho_signed_bias", global[6] / global[0]},
            {"rho_mean_L1V", global[7] / global[0]},
            {"rho_mean_L2V", std::sqrt(global[8] / global[0])},
            {"rho_mean_Linf", globalMeanMaximum},
            {"pressure_of_mean_L2V", std::sqrt(global[9] / global[0])},
            {"rho_mean_signed_bias", global[10] / global[0]},
            {"regions", regions},
            {"mean_reference_quadrature_order", 6},
            {"mean_reference_points_per_micro_tet",
             referenceQuadrature.GetNumPoints()},
        };
    }

    nlohmann::ordered_json FinalPointError(
        const NodeStatePair &field,
        const SpatialOperator<3> &spatial,
        const DNDS::Geom::UnstructuredMesh &mesh,
        const DualGeometry &geometry,
        const Configuration &configuration,
        const DNDS::MPIInfo &mpi,
        real time,
        real initialMass)
    {
        real local[6]{}, global[6]{};
        real localMaximum = 0, globalMaximum = 0;
        for (Index i = 0; i < mesh.NumNode(); i++)
        {
            const State state = spatial.PointValues()[i];
            const State exact =
                IsentropicVortex<3>(configuration, mesh.coords[i], time).first;
            DNDS_check_throw_info(state.allFinite() && state(0) > 0 &&
                                      Pressure(state, configuration.physics.gamma) > 0,
                                  "Roe component transient produced a nonphysical recovered state");
            const real volume = geometry.NodeVolume(i).moments.measure;
            const real rhoError = state(0) - exact(0);
            const real pressureError =
                Pressure(state, configuration.physics.gamma) -
                Pressure(exact, configuration.physics.gamma);
            local[0] += volume;
            local[1] += volume * std::abs(rhoError);
            local[2] += volume * rhoError * rhoError;
            local[3] += volume * std::abs(pressureError);
            local[4] += volume * pressureError * pressureError;
            local[5] += volume * field[i](0);
            localMaximum = std::max(localMaximum, std::abs(rhoError));
        }
        MPI_Allreduce(local, global, 6, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        MPI_Allreduce(&localMaximum, &globalMaximum, 1,
                      DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        return {
            {"time", time}, {"volume", global[0]},
            {"rho_point_L1V", global[1] / global[0]},
            {"rho_point_L2V", std::sqrt(global[2] / global[0])},
            {"rho_point_Linf", globalMaximum},
            {"pressure_point_L1V", global[3] / global[0]},
            {"pressure_point_L2V", std::sqrt(global[4] / global[0])},
            {"mass_drift", global[5] - initialMass},
        };
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    int status = 0;
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        try
        {
            DNDS_check_throw_info(argc >= 8 && argc <= 12,
                "usage: config.json output.json LeastSquares|Variational stencil-count dt final-time roe-scale [Traditional|Efficient [mesh-file [rho|mx|my|mz|E|all|none [lean]]]]");
            const std::filesystem::path output = argv[2];
            DNDS_check_throw_info(!std::filesystem::exists(output),
                                  "Roe budget output already exists");
            std::ifstream rawInput(argv[1]);
            DNDS_check_throw_info(rawInput.good(), "Cannot open input config");
            const auto raw = nlohmann::ordered_json::parse(rawInput);
            nlohmann::ordered_json resolved = Configuration{};
            resolved.merge_patch(raw);
            for (auto &boundary : resolved["physics"]["boundaryZones"])
            {
                nlohmann::ordered_json normalized = BoundaryZoneSettings{};
                normalized.merge_patch(boundary);
                boundary = std::move(normalized);
            }
            auto cfg = resolved.get<Configuration>();
            DNDS_check_throw_info(!cfg.reconstruction.enableLimiter,
                                  "Input config must disable the limiter");
            const std::string modeName = argc >= 9 ? argv[8] : "Efficient";
            DNDS_check_throw_info(modeName == "Traditional" || modeName == "Efficient",
                                  "Unknown integration mode");
            cfg.algorithm.mode = modeName == "Traditional" ?
                IntegrationMode::TraditionalQuadrature : IntegrationMode::EfficientDifferential;
            if (argc >= 10)
                cfg.mesh.meshFile = argv[9];
            cfg.algorithm.retainMicroGeometry = true;
            cfg.physics.riemannSolver = DNDS::Euler::Gas::Roe;
            cfg.physics.viscous.enabled = false;
            cfg.reconstruction.enableLimiter = false;
            const std::string method = argv[3];
            DNDS_check_throw_info(method == "LeastSquares" || method == "Variational",
                                  "Unknown reconstruction method");
            cfg.reconstruction.method = method == "Variational" ?
                ReconstructionMethod::Variational : ReconstructionMethod::LeastSquares;
            const int stencil = std::stoi(argv[4]);
            if (method == "Variational")
            {
                DNDS_check_throw_info(stencil == 0,
                                      "Use stencil-count 0 for benchmark variational settings");
                cfg.reconstruction.stencilSizeFactor = 1.7;
                cfg.reconstruction.variationalWeight = 5.0;
                cfg.reconstruction.variationalIterations = 3;
                cfg.reconstruction.variationalRelaxation = 1.0;
            }
            else
            {
                DNDS_check_throw_info(stencil == 0 || stencil >= 9,
                                      "Need zero for the benchmark stencil factor or at least nine stencil nodes");
                cfg.reconstruction.stencilSizeFactor = stencil == 0 ? 1.7 :
                    static_cast<real>(stencil) / 9.0;
            }
            const real dt = std::stod(argv[5]);
            const real finalTime = std::stod(argv[6]);
            const real alpha = std::stod(argv[7]);
            DNDS_check_throw_info(dt > 0 && finalTime >= 0 && alpha >= 0 && alpha <= 1,
                                  "Invalid dt, final time, or Roe scale");
            const std::string ablatedComponent = argc >= 11 ? argv[10] : "none";
            const bool lean = argc == 12;
            DNDS_check_throw_info(!lean || std::string(argv[11]) == "lean",
                                  "Unknown Roe budget diagnostic mode");
            State correctionScales = State::Ones();
            if (ablatedComponent != "none")
            {
                DNDS_check_throw_info(alpha == 1.0,
                                      "Component ablation requires roe-scale 1");
                if (ablatedComponent == "all")
                    correctionScales.setZero();
                else
                {
                    const std::vector<std::string> names{"rho", "mx", "my", "mz", "E"};
                    const auto match = std::find(names.begin(), names.end(),
                                                 ablatedComponent);
                    DNDS_check_throw_info(match != names.end(),
                                          "Unknown Roe correction component");
                    correctionScales[std::distance(names.begin(), match)] = 0;
                }
            }
            const int steps = static_cast<int>(std::llround(finalTime / dt));
            DNDS_check_throw_info(std::abs(steps * dt - finalTime) < 1e-12,
                                  "final-time/dt must be an integer");
            cfg.time.timeStep = dt;
            cfg.time.endTime = finalTime;
            cfg.time.iterations = steps;
            cfg.time.useCFLTimeStep = false;
            cfg.time.useLocalTimeStep = false;
            cfg.io.writeVTK = false;
            cfg.io.writeInitial = false;
            cfg.io.writeFinal = false;
            cfg.io.writeFinalRestart = false;
            cfg.io.writeResolvedConfiguration = false;
            cfg.io.outputInterval = 0;
            cfg.io.restartInterval = 0;
            cfg.io.outputPrefix = (output.parent_path() /
                                   (output.stem().string() + "_solution")).string();

            Solver<3> solver(mpi, cfg);
            solver.Initialize();
            const auto &mesh = *solver.Mesh();
            const auto &geometry = solver.Geometry();
            const auto &topology = solver.EdgeTopology();
            const auto &halo = solver.NodeCommunication();
            SpatialOperator<3> spatial(mpi, solver.Mesh(), topology, geometry,
                                       solver.ReconstructionData(), solver.Boundaries(),
                                       cfg.algorithm.mode, cfg.reconstruction,
                                       cfg.physics, cfg.time, halo);
            spatial.Initialize();
            NodeStatePair state, base, stage, rhs;
            Allocate(state, "roeBudget.state", 5, mpi, mesh.NumNode(), halo);
            Allocate(base, "roeBudget.base", 5, mpi, mesh.NumNode(), halo);
            Allocate(stage, "roeBudget.stage", 5, mpi, mesh.NumNode(), halo);
            Allocate(rhs, "roeBudget.rhs", 5, mpi, mesh.NumNode(), halo);
            DissipationAssembler assembler(mpi, mesh, topology, geometry,
                                           spatial, solver.ReconstructionData(), halo,
                                           cfg.algorithm.mode, cfg.physics.gamma);

            nlohmann::ordered_json initialBudget;
            if (lean)
            {
                SetAnalyticState(state, mesh, geometry, halo, cfg, 0.0);
                spatial.EvaluateRHS(state, rhs);
            }
            else
                initialBudget = StaticBudget(0.0, state, rhs, spatial,
                    assembler, mesh, geometry, halo, cfg, mpi);
            real componentRhsMismatch = 0;
            if (ablatedComponent != "none")
            {
                const auto fullCorrection = assembler.Evaluate(rhs);
                std::vector<State> expected(static_cast<std::size_t>(mesh.NumNode()));
                for (Index i = 0; i < mesh.NumNode(); i++)
                    expected[static_cast<std::size_t>(i)] = State(rhs[i]) -
                        (State::Ones() - correctionScales).cwiseProduct(
                            fullCorrection[static_cast<std::size_t>(i)]);
                spatial.SetInteriorInviscidCorrectionScalesForDiagnostics(
                    correctionScales);
                spatial.EvaluateRHS(state, rhs);
                real localMismatch = 0;
                for (Index i = 0; i < mesh.NumNode(); i++)
                    localMismatch = std::max(localMismatch,
                        (State(rhs[i]) - expected[static_cast<std::size_t>(i)])
                            .cwiseAbs().maxCoeff());
                MPI_Allreduce(&localMismatch, &componentRhsMismatch, 1,
                              DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
                DNDS_check_throw_info(componentRhsMismatch < 1e-8,
                                      "Component Roe ablation does not match the independent RHS decomposition");
            }
            real initialMassLocal = 0, initialMass = 0;
            for (Index i = 0; i < mesh.NumNode(); i++)
                initialMassLocal += geometry.NodeVolume(i).moments.measure * state[i](0);
            MPI_Allreduce(&initialMassLocal, &initialMass, 1,
                          DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);

            const auto evaluateStage = [&](NodeStatePair &input)
            {
                spatial.EvaluateRHS(input, rhs, false, false);
                if (alpha != 1.0)
                {
                    const auto dissipation = assembler.Evaluate(rhs);
                    for (Index i = 0; i < mesh.NumNode(); i++)
                        rhs[i] += (alpha - 1.0) *
                                  dissipation[static_cast<std::size_t>(i)];
                }
            };
            for (int step = 0; step < steps; step++)
            {
                for (Index i = 0; i < mesh.NumNode(); i++)
                    base[i] = state[i];
                evaluateStage(state);
                for (Index i = 0; i < mesh.NumNode(); i++)
                    stage[i] = base[i] + dt * rhs[i];
                evaluateStage(stage);
                for (Index i = 0; i < mesh.NumNode(); i++)
                    stage[i] = 0.75 * base[i] +
                               0.25 * (stage[i] + dt * rhs[i]);
                evaluateStage(stage);
                for (Index i = 0; i < mesh.NumNode(); i++)
                    state[i] = (1.0 / 3.0) * base[i] +
                               (2.0 / 3.0) * (stage[i] + dt * rhs[i]);
            }
            if (ablatedComponent != "none")
                spatial.SetInteriorInviscidCorrectionScalesForDiagnostics(
                    State::Ones());
            spatial.EvaluateRHS(state, rhs);
            const auto final = lean ?
                FinalPointError(state, spatial, mesh, geometry, cfg, mpi,
                                finalTime, initialMass) :
                FinalError(state, spatial, mesh, geometry, cfg, mpi,
                           finalTime, initialMass);
            nlohmann::ordered_json finalStaticBudget;
            if (!lean)
                finalStaticBudget = StaticBudget(finalTime, state, rhs,
                    spatial, assembler, mesh, geometry, halo, cfg, mpi);
            if (mpi.rank == 0)
            {
                nlohmann::ordered_json result = {
                    {"mesh", cfg.mesh.meshFile}, {"mode", modeName},
                    {"method", method},
                    {"stencil", stencil}, {"nodes", solver.Mesh()->NumNodeGlobal()},
                    {"mpi_ranks", mpi.size}, {"dt", dt}, {"steps", steps},
                    {"final_time", finalTime}, {"roe_scale", alpha},
                    {"diagnostics", lean ? "point_only" : "full"},
                    {"ablated_component", ablatedComponent},
                    {"component_rhs_validation_max_mismatch_t0", componentRhsMismatch},
                    {"transient", final},
                };
                if (!lean)
                {
                    result["initial_static"] = initialBudget;
                    result["analytic_final_static"] = finalStaticBudget;
                }
                std::ofstream stream(output);
                DNDS_check_throw_info(stream.good(), "Cannot open Roe budget output");
                stream << std::setw(2) << result << '\n';
                DNDS_check_throw_info(stream.good(), "Cannot write Roe budget output");
                std::cout << result.dump(2) << std::endl;
            }
        }
        catch (const std::exception &error)
        {
            std::cerr << "NCFV efficient Roe budget: " << error.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
            status = 1;
        }
    }
    MPI_Finalize();
    return status;
}
