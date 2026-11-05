/**
 * @file first_interface_flux_probe.cpp
 * @brief Interface-flux integration error immediately after the first reconstruction.
 *
 * The production efficient surface integral is compared with a higher-order
 * analytic integral over every retained micro-surface.  No time step is taken.
 */
#include "NCFV/NCFVAnalytic.hpp"
#include "NCFV/NCFVSolver.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <map>

using DNDS::real;
using Index = DNDS::index;
using namespace DNDS::NCFV;
using State = Eigen::Matrix<real, 5, 1>;
using Gradient = Eigen::Matrix<real, 3, 5>;

namespace
{
    real Pressure(const State &state, real gamma)
    {
        return (gamma - 1) *
               (state(4) -
                0.5 * state.segment<3>(1).squaredNorm() / state(0));
    }

    State PhysicalFlux(const State &state, const Vector3 &normal, real gamma)
    {
        const real normalVelocity =
            state.segment<3>(1).dot(normal) / state(0);
        const real pressure = Pressure(state, gamma);
        State flux;
        flux << state(0) * normalVelocity,
            state.segment<3>(1) * normalVelocity + pressure * normal,
            (state(4) + pressure) * normalVelocity;
        return flux;
    }

    State FluxDerivative(
        const State &state,
        const State &stateDerivative,
        int direction,
        real gamma)
    {
        const Vector3 momentum = state.segment<3>(1);
        const Vector3 momentumDerivative =
            stateDerivative.segment<3>(1);
        const Vector3 velocity = momentum / state(0);
        const Vector3 velocityDerivative =
            (momentumDerivative - velocity * stateDerivative(0)) /
            state(0);
        const real pressure = Pressure(state, gamma);
        const real pressureDerivative =
            (gamma - 1) *
            (stateDerivative(4) -
             0.5 * (momentumDerivative.dot(velocity) +
                    momentum.dot(velocityDerivative)));
        State derivative;
        derivative(0) = momentumDerivative(direction);
        derivative.segment<3>(1) =
            momentumDerivative * velocity(direction) +
            momentum * velocityDerivative(direction);
        derivative(direction + 1) += pressureDerivative;
        derivative(4) =
            (stateDerivative(4) + pressureDerivative) *
                velocity(direction) +
            (state(4) + pressure) * velocityDerivative(direction);
        return derivative;
    }

    State NumericalFlux(
        const State &left,
        const State &right,
        const Vector3 &normal,
        real gamma)
    {
        State flux = State::Zero();
        const Vector3 gridVelocity = Vector3::Zero();
        real acousticMinus = 0;
        real contact = 0;
        real acousticPlus = 0;
        DNDS::Euler::Gas::InviscidFlux_IdealGas_Dispatcher<3>(
            DNDS::Euler::Gas::Roe,
            left, right, left, right,
            gridVelocity, normal, gamma, gamma, flux,
            0.0, 1.0, 1.0,
            []() {}, acousticMinus, contact, acousticPlus);
        return flux;
    }

    struct Norms
    {
        // Weight, weighted absolute value, weighted square, maximum.
        std::map<std::string, std::array<real, 4>> values;

        void Add(const std::string &key, real value, real weight)
        {
            DNDS_check_throw_info(
                std::isfinite(value) && weight > 0,
                "Invalid interface-flux diagnostic sample");
            auto &entry = values[key];
            entry[0] += weight;
            entry[1] += weight * std::abs(value);
            entry[2] += weight * value * value;
            entry[3] = std::max(entry[3], std::abs(value));
        }

        nlohmann::ordered_json Reduce(const DNDS::MPIInfo &mpi) const
        {
            nlohmann::ordered_json result;
            for (const auto &[key, local] : values)
            {
                real sums[3]{};
                real maximum = 0;
                MPI_Allreduce(
                    local.data(), sums, 3,
                    DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                MPI_Allreduce(
                    &local[3], &maximum, 1,
                    DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
                result[key] = {
                    {"L1", sums[1] / sums[0]},
                    {"L2", std::sqrt(sums[2] / sums[0])},
                    {"Linf", maximum},
                    {"weight", sums[0]}};
            }
            return result;
        }
    };
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
            DNDS_check_throw_info(
                argc >= 3 && argc <= 4,
                "Usage: first_interface_flux_probe config.json output.json [reference-order]");
            auto configuration =
                LoadConfiguration(argv[1], {}, {}).configuration;
            const int referenceOrder =
                argc == 4 ? std::stoi(argv[3]) : 6;
            DNDS_check_throw_info(
                configuration.dimension == 3 &&
                    configuration.initialField.isentropicVortex &&
                    configuration.algorithm.mode ==
                        IntegrationMode::EfficientDifferential &&
                    !configuration.reconstruction.enableLimiter &&
                    !configuration.physics.viscous.enabled &&
                    configuration.io.restartInput.empty() &&
                    configuration.physics.riemannSolver ==
                        DNDS::Euler::Gas::Roe,
                "Requires a fresh, inviscid, unlimited efficient Roe vortex");
            DNDS_check_throw_info(
                referenceOrder >= 1 &&
                    referenceOrder <= DNDS::Geom::Elem::INT_ORDER_MAX,
                "Unsupported reference quadrature order");

            configuration.algorithm.retainMicroGeometry = true;
            configuration.time.iterations = 0;
            configuration.time.endTime = 0;
            configuration.io.writeVTK = false;
            configuration.io.writeInitial = false;
            configuration.io.writeFinal = false;
            configuration.io.writeFinalRestart = false;
            configuration.io.writeResolvedConfiguration = false;
            configuration.io.restartInterval = 0;

            Solver<3> solver(mpi, configuration);
            solver.Initialize();
            const auto mesh = solver.Mesh();
            const auto &geometry = solver.Geometry();
            const auto &topology = solver.EdgeTopology();
            const auto &reconstruction = solver.ReconstructionData();
            const NodeHalo &nodeHalo = solver.NodeCommunication();

            NodeStatePair means;
            NodeStatePair rhs;
            const auto allocate = [&](NodeStatePair &field,
                                      const std::string &name)
            {
                field.InitPair(name, mpi);
                field.father->Resize(mesh->NumNode(), 5, 1);
                field.son->Resize(nodeHalo.NumNodeGhost(), 5, 1);
                field.BorrowSetup(nodeHalo.Layout());
                field.trans.initPersistentPull();
                for (Index i = 0; i < field.Size(); i++)
                    field[i].setZero();
            };
            allocate(means, "interfaceFlux.means");
            allocate(rhs, "interfaceFlux.rhs");
            for (Index i = 0; i < mesh->NumNode(); i++)
                means[i] = solver.StateField()[i];

            SpatialOperator<3> spatial(
                mpi, mesh, topology, geometry, reconstruction,
                solver.Boundaries(), configuration.algorithm.mode,
                configuration.reconstruction, configuration.physics,
                configuration.time, nodeHalo);
            spatial.Initialize();
            // Exactly one coefficient build and point-value recovery; no step.
            spatial.EvaluateRHS(means, rhs);
            const auto &reconstructedPoints = spatial.PointValues();
            const auto &reconstructedGradients = spatial.Gradients();

            std::vector<State> exactPoints(nodeHalo.NumNodeProc());
            std::vector<Gradient> exactGradients(nodeHalo.NumNodeProc());
            for (Index i = 0; i < nodeHalo.NumNodeProc(); i++)
                std::tie(exactPoints[i], exactGradients[i]) =
                    IsentropicVortex<3>(
                        configuration, nodeHalo.Coordinate(i), 0);

            const DNDS::Geom::Elem::Quadrature triangleQuadrature(
                {DNDS::Geom::Elem::Tri3}, referenceOrder);
            const std::array<std::string, 5> variableNames{
                "rho", "rhou", "rhov", "rhow", "rhoE"};
            const std::array<std::string, 4> modeNames{
                "reconstructed_point_reconstructed_gradient",
                "exact_point_reconstructed_gradient",
                "exact_point_exact_gradient",
                "reconstructed_point_exact_gradient"};
            Norms norms;
            real localVolume = 0;
            for (Index iNode = 0; iNode < mesh->NumNode(); iNode++)
                localVolume +=
                    geometry.NodeVolume(iNode).moments.measure;
            real localArea = 0;
            Index localFaces = 0;
            Index localMicroSurfaces = 0;
            real localMaximumAreaClosure = 0;

            for (Index iEdge = 0;
                 iEdge < topology.NumEdge(); iEdge++)
            {
                const auto &surface = geometry.EdgeSurface(iEdge);
                DNDS_check_throw_info(
                    !surface.microSurfaces.empty(),
                    "Reference micro-surfaces were not retained");
                const real area = surface.measure;
                const Vector3 normal =
                    surface.vectorMeasure.normalized();
                State exactIntegral = State::Zero();
                real recoveredArea = 0;
                Vector3 recoveredVectorMeasure = Vector3::Zero();
                for (const MicroSurface &micro : surface.microSurfaces)
                {
                    DNDS_check_throw_info(
                        micro.nPoints == 3 && micro.measure > 0,
                        "Invalid retained micro-surface");
                    localMicroSurfaces++;
                    recoveredArea += micro.measure;
                    recoveredVectorMeasure += micro.vectorMeasure;
                    const Vector3 microNormal =
                        micro.vectorMeasure / micro.measure;
                    for (int iQuadrature = 0;
                         iQuadrature <
                         triangleQuadrature.GetNumPoints();
                         iQuadrature++)
                    {
                        const auto [parametric, referenceWeight] =
                            triangleQuadrature.GetQuadraturePointInfo(
                                iQuadrature);
                        const Vector3 coordinate =
                            micro.points[0] +
                            parametric[0] *
                                (micro.points[1] - micro.points[0]) +
                            parametric[1] *
                                (micro.points[2] - micro.points[0]);
                        const auto [state, gradient] =
                            IsentropicVortex<3>(
                                configuration, coordinate, 0);
                        exactIntegral +=
                            2.0 * micro.measure * referenceWeight *
                            PhysicalFlux(
                                state, microNormal,
                                configuration.physics.gamma);
                    }
                }
                localMaximumAreaClosure = std::max(
                    localMaximumAreaClosure,
                    std::max(
                        std::abs(recoveredArea - area) / area,
                        (recoveredVectorMeasure -
                         surface.vectorMeasure)
                                .norm() /
                            area));

                for (int mode = 0; mode < 4; mode++)
                {
                    const auto pointAt = [&](Index node) -> State
                    {
                        return mode == 1 || mode == 2
                                   ? exactPoints[node]
                                   : State(reconstructedPoints[node]);
                    };
                    const auto gradientAt = [&](Index node) -> Gradient
                    {
                        return mode >= 2
                                   ? exactGradients[node]
                                   : Gradient(
                                         reconstructedGradients[node]);
                    };
                    const auto surfaceMean = [&](int side) -> State
                    {
                        const std::size_t sideIndex =
                            static_cast<std::size_t>(side);
                        const Index anchor = surface.nodes[sideIndex];
                        State integral = area * pointAt(anchor);
                        for (const EfficientSurfaceNode &entry :
                             surface.efficientStencil)
                            integral +=
                                gradientAt(entry.node).transpose() *
                                entry.stateGradientWeights[sideIndex];
                        return integral / area;
                    };
                    const auto integratedPhysicalFlux =
                        [&](int side) -> State
                    {
                        const std::size_t sideIndex =
                            static_cast<std::size_t>(side);
                        const Index anchor = surface.nodes[sideIndex];
                        State integral = PhysicalFlux(
                            pointAt(anchor), surface.vectorMeasure,
                            configuration.physics.gamma);
                        for (const EfficientSurfaceNode &entry :
                             surface.efficientStencil)
                            for (int derivativeDirection = 0;
                                 derivativeDirection < 3;
                                 derivativeDirection++)
                                for (int fluxDirection = 0;
                                     fluxDirection < 3;
                                     fluxDirection++)
                                    integral +=
                                        entry.fluxGradientWeights[sideIndex](
                                            derivativeDirection,
                                            fluxDirection) *
                                        FluxDerivative(
                                            pointAt(entry.node),
                                            gradientAt(entry.node)
                                                .row(derivativeDirection)
                                                .transpose(),
                                            fluxDirection,
                                            configuration.physics.gamma);
                        return integral;
                    };

                    const State leftMean = surfaceMean(0);
                    const State rightMean = surfaceMean(1);
                    DNDS_check_throw_info(
                        leftMean(0) > 0 && rightMean(0) > 0 &&
                            Pressure(
                                leftMean,
                                configuration.physics.gamma) > 0 &&
                            Pressure(
                                rightMean,
                                configuration.physics.gamma) > 0,
                        "Nonphysical reconstructed surface mean");
                    const State leftIntegral =
                        integratedPhysicalFlux(0);
                    const State rightIntegral =
                        integratedPhysicalFlux(1);
                    const State centralIntegral =
                        0.5 * (leftIntegral + rightIntegral);
                    const State centralMean =
                        0.5 *
                        (PhysicalFlux(
                             leftMean, normal,
                             configuration.physics.gamma) +
                         PhysicalFlux(
                             rightMean, normal,
                             configuration.physics.gamma));
                    const State numericalIntegral =
                        centralIntegral +
                        area *
                            (NumericalFlux(
                                 leftMean, rightMean, normal,
                                 configuration.physics.gamma) -
                             centralMean);
                    const State dissipativeIntegral =
                        numericalIntegral - centralIntegral;

                    for (int variable = 0; variable < 5; variable++)
                    {
                        const std::string prefix =
                            "mode" + std::to_string(mode) + ".";
                        const std::string suffix =
                            "." + variableNames[variable];
                        const real leftError =
                            leftIntegral(variable) -
                            exactIntegral(variable);
                        const real rightError =
                            rightIntegral(variable) -
                            exactIntegral(variable);
                        const real centralError =
                            centralIntegral(variable) -
                            exactIntegral(variable);
                        const real numericalError =
                            numericalIntegral(variable) -
                            exactIntegral(variable);
                        norms.Add(
                            prefix + "left.mean_flux_error" + suffix,
                            leftError / area, area);
                        norms.Add(
                            prefix + "right.mean_flux_error" + suffix,
                            rightError / area, area);
                        norms.Add(
                            prefix + "central.mean_flux_error" + suffix,
                            centralError / area, area);
                        norms.Add(
                            prefix + "roe_total.mean_flux_error" + suffix,
                            numericalError / area, area);
                        norms.Add(
                            prefix + "roe_dissipation.mean_flux" + suffix,
                            dissipativeIntegral(variable) / area, area);
                        norms.Add(
                            prefix + "surface_mean_jump" + suffix,
                            rightMean(variable) - leftMean(variable),
                            area);
                        norms.Add(
                            prefix + "central.integral_error" + suffix,
                            centralError, 1.0);
                        norms.Add(
                            prefix + "roe_total.integral_error" + suffix,
                            numericalError, 1.0);
                    }
                }
                localArea += area;
                localFaces++;
            }

            real globalArea = 0;
            real globalVolume = 0;
            Index globalFaces = 0;
            Index globalMicroSurfaces = 0;
            real globalMaximumAreaClosure = 0;
            MPI_Allreduce(
                &localVolume, &globalVolume, 1,
                DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(
                &localArea, &globalArea, 1,
                DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(
                &localFaces, &globalFaces, 1,
                DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            MPI_Allreduce(
                &localMicroSurfaces, &globalMicroSurfaces, 1,
                DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            MPI_Allreduce(
                &localMaximumAreaClosure,
                &globalMaximumAreaClosure, 1,
                DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            const auto metrics = norms.Reduce(mpi);
            if (mpi.rank == 0)
            {
                nlohmann::ordered_json output{
                    {"method", MethodName},
                    {"configuration", configuration},
                    {"source_configuration", argv[1]},
                    {"reference_quadrature_order", referenceOrder},
                    {"time", 0},
                    {"time_steps", 0},
                    {"reconstruction_calls", 1},
                    {"rhs_evaluations", 1},
                    {"mpi_ranks", mpi.size},
                    {"nodes", mesh->NumNodeGlobal()},
                    {"cells", mesh->NumCellGlobal()},
                    {"owned_interfaces", globalFaces},
                    {"micro_surfaces", globalMicroSurfaces},
                    {"volume", globalVolume},
                    {"surface_area", globalArea},
                    {"h_3d", std::cbrt(
                         globalVolume /
                         static_cast<real>(mesh->NumNodeGlobal()))},
                    {"surface_geometry_closure_max",
                     globalMaximumAreaClosure},
                    {"modes", modeNames},
                    {"errors", metrics}};
                std::ofstream file(argv[2]);
                file << std::setw(2) << output << '\n';
                DNDS_check_throw_info(
                    file.good(),
                    "Cannot write interface-flux metrics");
                DNDS::log()
                    << "First interface flux: rho central L2="
                    << metrics[
                           "mode0.central.mean_flux_error.rho"]
                                  ["L2"]
                    << ", Roe total L2="
                    << metrics[
                           "mode0.roe_total.mean_flux_error.rho"]
                                  ["L2"]
                    << std::endl;
            }
        }
        catch (const std::exception &exception)
        {
            std::cerr << "NCFV first interface-flux diagnostic: "
                      << exception.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
            status = 1;
        }
    }
    MPI_Finalize();
    return status;
}
