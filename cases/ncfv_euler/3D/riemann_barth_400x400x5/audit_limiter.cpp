/** Read-only audit of production point recovery and efficient Barth limiting. */
#include "NCFV/NCFVSolver.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

using State = Eigen::Matrix<DNDS::real, 5, 1>;
using Vector = Eigen::Matrix<DNDS::real, 3, 1>;
using DNDS::DNDS_MPI_INDEX;
using DNDS::DNDS_MPI_REAL;

namespace
{
    DNDS::real Pressure(const State &u, DNDS::real gamma)
    {
        return (gamma - 1) * (u(4) - .5 * u.segment<3>(1).squaredNorm() / u(0));
    }

    State Roe(const State &left, const State &right, const Vector &normal, DNDS::real gamma)
    {
        State flux = State::Zero();
        DNDS::real acousticMinus = 0, contact = 0, acousticPlus = 0;
        DNDS::Euler::Gas::InviscidFlux_IdealGas_Dispatcher<3>(
            DNDS::Euler::Gas::Roe, left, right, left, right, Vector::Zero(), normal,
            gamma, gamma, flux, 0., 1., 1., []() {}, acousticMinus, contact, acousticPlus);
        return flux;
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    try
    {
        DNDS_check_throw_info(argc == 7,
                              "usage: config output.json restart-base-or-- mesh-or-- node-file-or-- label");
        auto cfg = DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
        if (std::string(argv[3]) != "-")
            cfg.io.restartInput = argv[3];
        if (std::string(argv[4]) != "-")
            cfg.mesh.meshFile = argv[4];
        if (std::string(argv[5]) != "-")
            cfg.initialField.nodeFile = argv[5];
        cfg.time.iterations = 0;
        cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
        cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
        cfg.io.restartInterval = 0;
        DNDS_check_throw_info(cfg.algorithm.mode == DNDS::NCFV::IntegrationMode::EfficientDifferential &&
                                  cfg.reconstruction.enableLimiter,
                              "Audit requires efficient mode with limiting enabled");
        DNDS::NCFV::Solver<3> solver(mpi, cfg);
        solver.Initialize();
        const auto &mesh = solver.Mesh();
        const auto &halo = solver.NodeCommunication();
        const auto &topology = solver.EdgeTopology();
        const auto &geometry = solver.Geometry();
        DNDS::NCFV::NodeStatePair means, rhs, raw;
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
        allocate(means, "audit.means");
        allocate(rhs, "audit.rhs");
        allocate(raw, "audit.rawPoints");
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
            means[i] = solver.StateField()[i];
        DNDS::NCFV::SpatialOperator<3> spatial(
            mpi, mesh, topology, geometry, solver.ReconstructionData(), solver.Boundaries(),
            cfg.algorithm.mode, cfg.reconstruction, cfg.physics, cfg.time, halo);
        spatial.Initialize();
        const DNDS::real residual = spatial.EvaluateRHS(means, rhs);
        solver.ReconstructionData().RecoverPointValues(means, spatial.Gradients(), spatial.Coefficients(), raw);
        const auto &points = spatial.PointValues();
        const auto &gradients = spatial.Gradients();
        const auto &theta = spatial.LimiterFactors();
        std::ofstream nodeDump;
        if (mesh->NumNodeGlobal() < 100000)
        {
            nodeDump.open(std::string(argv[2]) + ".rank" + std::to_string(mpi.rank) + ".csv");
            nodeDump << "node,x,y,z,mean_rho,raw_rho,protected_rho,mean_lower,mean_upper,theta,pressure,rhs_rho\n"
                     << std::setprecision(17);
        }
        // nodes, raw overshoots, protected overshoots, theta-zero nodes,
        // overshoots at theta zero, positivity repairs, sides, pair violations,
        // sides beyond mean bounds, edges with both theta zero.
        std::array<DNDS::index, 10> counts{};
        // raw/physical density overshoot, endpoint bound violation,
        // interface density overshoot, zero-theta vs mean/point Roe differences.
        std::array<DNDS::real, 6> maxima{};
        std::array<DNDS::real, 4> minima{1e300, 1e300, 1e300, 1e300};
        std::array<DNDS::real, 4> upper{-1e300, -1e300, -1e300, -1e300};
        std::array<DNDS::real, 12> example{};
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            State lo = means[i], hi = means[i];
            for (const auto &incidence : topology.Node2Edge(i))
            {
                const auto &surface = geometry.EdgeSurface(incidence.edge);
                const DNDS::index neighbor = surface.nodes[surface.nodes[0] == i ? 1 : 0];
                lo = lo.cwiseMin(State(means[neighbor]));
                hi = hi.cwiseMax(State(means[neighbor]));
            }
            const auto violation = [&](DNDS::real rho)
            { return std::max({0., lo(0) - rho, rho - hi(0)}); };
            const DNDS::real rawViolation = violation(raw[i](0));
            const DNDS::real pointViolation = violation(points[i](0));
            const bool zero = theta[i](0) < 1e-12;
            std::ofstream trace;
            if (mesh->NumNodeGlobal() < 100000 && mesh->node2nodeOrig(i, 0) == 4612)
            {
                trace.open(std::string(argv[2]) + ".trace_rank" + std::to_string(mpi.rank) + ".csv");
                trace << "node,neighbor,variable,point,neighbor_point,candidate,increment,lower,upper,ratio,scaled_roundoff\n"
                      << std::setprecision(17);
            }
            if (nodeDump.is_open())
            {
                const Vector x = halo.Coordinate(i);
                nodeDump << mesh->node2nodeOrig(i, 0) << ',' << x(0) << ',' << x(1) << ',' << x(2)
                         << ',' << means[i](0) << ',' << raw[i](0) << ',' << points[i](0)
                         << ',' << lo(0) << ',' << hi(0) << ',' << theta[i](0)
                         << ',' << Pressure(State(points[i]), cfg.physics.gamma) << ',' << rhs[i](0) << '\n';
            }
            counts[0]++;
            counts[1] += rawViolation > 1e-11;
            counts[2] += pointViolation > 1e-11;
            counts[3] += zero;
            counts[4] += pointViolation > 1e-11 && zero;
            counts[5] += (points[i] - raw[i]).norm() > 1e-12;
            maxima[0] = std::max(maxima[0], rawViolation);
            if (pointViolation > maxima[1])
            {
                maxima[1] = pointViolation;
                const Vector x = halo.Coordinate(i);
                example = {pointViolation, static_cast<DNDS::real>(mesh->node2nodeOrig(i, 0)),
                           x(0), x(1), x(2), means[i](0), raw[i](0), points[i](0),
                           lo(0), hi(0), theta[i](0), Pressure(State(points[i]), cfg.physics.gamma)};
            }
            const std::array<DNDS::real, 4> values{
                means[i](0), raw[i](0), points[i](0), Pressure(State(points[i]), cfg.physics.gamma)};
            for (int k = 0; k < 4; k++)
            {
                minima[k] = std::min(minima[k], values[k]);
                upper[k] = std::max(upper[k], values[k]);
            }
            for (const auto &incidence : topology.Node2Edge(i))
            {
                const auto &surface = geometry.EdgeSurface(incidence.edge);
                const int side = surface.nodes[0] == i ? 0 : 1;
                const auto j = surface.nodes[1 - side];
                if (trace.is_open())
                {
                    Eigen::VectorXd candidate = surface.measure * points[i];
                    for (const auto &entry : surface.efficientStencil)
                        candidate += gradients[entry.node].transpose() * entry.stateGradientWeights[side].head<3>();
                    candidate /= surface.measure;
                    for (int variable = 0; variable < 5; variable++)
                    {
                        const DNDS::real point = points[i](variable), neighbor = points[j](variable);
                        const DNDS::real increment = candidate(variable) - point;
                        const DNDS::real lower = std::min(point, neighbor), upperBound = std::max(point, neighbor);
                        DNDS::real ratio = 1;
                        if (increment > DNDS::verySmallReal)
                            ratio = (upperBound - point) / increment;
                        else if (increment < -DNDS::verySmallReal)
                            ratio = (lower - point) / increment;
                        const DNDS::real tolerance = 64 * std::numeric_limits<DNDS::real>::epsilon() *
                                                     std::max({1., std::abs(point), std::abs(neighbor), std::abs(candidate(variable))});
                        trace << mesh->node2nodeOrig(i, 0) << ',' << mesh->node2nodeOrig(j, 0) << ',' << variable
                              << ',' << point << ',' << neighbor << ',' << candidate(variable) << ',' << increment
                              << ',' << lower << ',' << upperBound << ',' << ratio << ',' << tolerance << '\n';
                    }
                }
                State correction = State::Zero();
                for (const auto &entry : surface.efficientStencil)
                    correction += gradients[entry.node].transpose() * entry.stateGradientWeights[side].head<3>();
                const State limited = State(points[i]) + theta[i](0) * correction / surface.measure;
                const State pairLo = State(points[i]).cwiseMin(State(points[j]));
                const State pairHi = State(points[i]).cwiseMax(State(points[j]));
                const DNDS::real pairViolation = std::max({0., (pairLo - limited).maxCoeff(), (limited - pairHi).maxCoeff()});
                counts[6]++;
                counts[7] += pairViolation > 1e-11;
                counts[8] += violation(limited(0)) > 1e-11;
                maxima[2] = std::max(maxima[2], pairViolation);
                maxima[3] = std::max(maxima[3], violation(limited(0)));
            }
        }
        for (DNDS::index edge = 0; edge < topology.NumEdge(); edge++)
        {
            const auto &surface = geometry.EdgeSurface(edge);
            const auto a = surface.nodes[0], b = surface.nodes[1];
            if (theta[a](0) > 1e-12 || theta[b](0) > 1e-12)
                continue;
            counts[9]++;
            const Vector normal = surface.vectorMeasure.normalized();
            const State actual = spatial.EdgeFluxes()[edge];
            const State meanFlux = surface.measure * Roe(State(means[a]), State(means[b]), normal, cfg.physics.gamma);
            const State pointFlux = surface.measure * Roe(State(points[a]), State(points[b]), normal, cfg.physics.gamma);
            maxima[4] = std::max(maxima[4], (actual - meanFlux).norm() / surface.measure);
            maxima[5] = std::max(maxima[5], (actual - pointFlux).norm() / surface.measure);
        }
        std::array<DNDS::index, 10> globalCounts{};
        std::array<DNDS::real, 6> globalMaxima{};
        std::array<DNDS::real, 4> globalMinima{}, globalUpper{};
        MPI_Reduce(counts.data(), globalCounts.data(), 10, DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(maxima.data(), globalMaxima.data(), 6, DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        MPI_Reduce(minima.data(), globalMinima.data(), 4, DNDS_MPI_REAL, MPI_MIN, 0, mpi.comm);
        MPI_Reduce(upper.data(), globalUpper.data(), 4, DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        std::vector<DNDS::real> examples(static_cast<std::size_t>(mpi.size) * example.size());
        MPI_Gather(example.data(), 12, DNDS_MPI_REAL, examples.data(), 12, DNDS_MPI_REAL, 0, mpi.comm);
        if (mpi.rank == 0)
        {
            int best = 0;
            for (int rank = 1; rank < mpi.size; rank++)
                if (examples[12 * rank] > examples[12 * best])
                    best = rank;
            nlohmann::ordered_json result{
                {"label", argv[6]}, {"mpi_ranks", mpi.size}, {"iteration", solver.CurrentIteration()},
                {"physical_time", solver.SimulationTime()}, {"residual_rms", residual},
                {"nodes", globalCounts[0]}, {"raw_density_outside_neighbor_means", globalCounts[1]},
                {"protected_density_outside_neighbor_means", globalCounts[2]}, {"theta_zero_nodes", globalCounts[3]},
                {"outside_means_with_theta_zero", globalCounts[4]}, {"positivity_repaired_points", globalCounts[5]},
                {"interface_sides", globalCounts[6]}, {"endpoint_bound_violations", globalCounts[7]},
                {"interface_density_outside_neighbor_means", globalCounts[8]}, {"both_theta_zero_edges", globalCounts[9]},
                {"max_raw_density_overshoot", globalMaxima[0]}, {"max_protected_density_overshoot", globalMaxima[1]},
                {"max_endpoint_bound_violation", globalMaxima[2]}, {"max_interface_density_overshoot", globalMaxima[3]},
                {"zero_theta_flux_vs_mean_roe_max_per_area", globalMaxima[4]},
                {"zero_theta_flux_vs_point_roe_max_per_area", globalMaxima[5]},
                {"range_order", {"mean_rho", "raw_point_rho", "protected_point_rho", "protected_point_pressure"}},
                {"minima", globalMinima}, {"maxima", globalUpper},
                {"worst_point_fields", {"overshoot", "original_node", "x", "y", "z", "mean_rho", "raw_rho", "protected_rho", "mean_lower", "mean_upper", "theta", "pressure"}},
                {"worst_point", std::vector<DNDS::real>(examples.begin() + 12 * best, examples.begin() + 12 * (best + 1))}};
            std::ofstream output(argv[2]);
            output << result.dump(2) << std::endl;
            std::cout << result.dump(2) << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "audit rank " << mpi.rank << ": " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    MPI_Finalize();
}
