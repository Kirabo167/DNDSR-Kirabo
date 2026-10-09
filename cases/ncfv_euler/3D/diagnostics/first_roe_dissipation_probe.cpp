/** Read-only first-reconstruction Roe dissipation diagnostic. Never calls Run(). */
#include "NCFV/NCFVSolver.hpp"
#include "NCFV/NCFVAnalytic.hpp"

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
    real Pressure(const State &u, real gamma)
    {
        return (gamma - 1) * (u(4) - 0.5 * u.segment<3>(1).squaredNorm() / u(0));
    }

    State PhysicalFlux(const State &u, const Vector3 &n, real gamma)
    {
        const real un = u.segment<3>(1).dot(n) / u(0);
        const real p = Pressure(u, gamma);
        State f;
        f << u(0) * un, u.segment<3>(1) * un + p * n, (u(4) + p) * un;
        return f;
    }

    State FluxDerivative(const State &u, const State &du, int d, real gamma)
    {
        const Vector3 m = u.segment<3>(1), dm = du.segment<3>(1);
        const Vector3 v = m / u(0), dv = (dm - v * du(0)) / u(0);
        const real p = Pressure(u, gamma);
        const real dp = (gamma - 1) * (du(4) - 0.5 * (dm.dot(v) + m.dot(dv)));
        State f;
        f(0) = dm(d);
        f.segment<3>(1) = dm * v(d) + m * dv(d);
        f(d + 1) += dp;
        f(4) = (du(4) + dp) * v(d) + (u(4) + p) * dv(d);
        return f;
    }

    State NumericalFlux(const State &l, const State &r, const Vector3 &n,
                        real gamma, DNDS::Euler::Gas::RiemannSolverType scheme)
    {
        State f = State::Zero();
        const Vector3 grid = Vector3::Zero();
        real lm = 0, lc = 0, lp = 0;
        DNDS::Euler::Gas::InviscidFlux_IdealGas_Dispatcher<3>(
            scheme, l, r, l, r, grid, n, gamma, gamma, f, 0.0, 1.0, 1.0,
            []() {}, lm, lc, lp);
        return f;
    }

    struct Norms
    {
        // Weight, weighted absolute value, weighted square, maximum.
        std::map<std::string, std::array<real, 4>> values;

        void Add(const std::string &key, real x, real weight)
        {
            DNDS_check_throw_info(std::isfinite(x) && weight > 0, "Invalid diagnostic sample");
            auto &v = values[key];
            v[0] += weight;
            v[1] += weight * std::abs(x);
            v[2] += weight * x * x;
            v[3] = std::max(v[3], std::abs(x));
        }

        nlohmann::ordered_json Reduce(const DNDS::MPIInfo &mpi) const
        {
            nlohmann::ordered_json result;
            for (const auto &[key, v] : values)
            {
                real sum[3]{}, maximum = 0;
                MPI_Allreduce(v.data(), sum, 3, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                MPI_Allreduce(&v[3], &maximum, 1, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
                result[key] = {{"L1", sum[1] / sum[0]}, {"L2", std::sqrt(sum[2] / sum[0])}, {"Linf", maximum}, {"weight", sum[0]}};
            }
            return result;
        }
    };
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        try
        {
            DNDS_check_throw_info(argc == 3, "Usage: first_roe_dissipation_probe config.json output-directory");
            auto cfg = LoadConfiguration(argv[1], {}, {}).configuration;
            DNDS_check_throw_info(cfg.dimension == 3 && cfg.initialField.isentropicVortex &&
                                      cfg.algorithm.mode == IntegrationMode::EfficientDifferential &&
                                      !cfg.reconstruction.enableLimiter && !cfg.physics.viscous.enabled &&
                                      cfg.io.restartInput.empty() && cfg.physics.riemannSolver == DNDS::Euler::Gas::Roe_M2,
                                  "Requires the fresh, inviscid, unlimited efficient Roe_M2 vortex");
            cfg.time.iterations = 0;
            cfg.time.endTime = 0;
            cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
            cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
            cfg.io.restartInterval = 0;
            const std::filesystem::path output = argv[2];
            if (mpi.rank == 0)
            {
                DNDS_check_throw_info(!std::filesystem::exists(output), "Refusing to overwrite diagnostic output");
                std::filesystem::create_directories(output);
            }
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            Solver<3> solver(mpi, cfg);
            solver.Initialize();
            const auto mesh = solver.Mesh();
            const auto &geometry = solver.Geometry();
            const auto &topology = solver.EdgeTopology();
            const auto &reconstruction = solver.ReconstructionData();
            const NodeHalo &nodeHalo = solver.NodeCommunication();
            NodeStatePair means, rhs, parts;
            auto allocate = [&](NodeStatePair &field, const std::string &name, int size)
            {
                field.InitPair(name, mpi);
                field.father->Resize(mesh->NumNode(), size, 1);
                field.son->Resize(nodeHalo.NumNodeGhost(), size, 1);
                field.BorrowSetup(nodeHalo.Layout());
                field.trans.initPersistentPull();
                for (Index i = 0; i < field.Size(); i++)
                    field[i].setZero();
            };
            allocate(means, "firstRoe.means", 5);
            allocate(rhs, "firstRoe.rhs", 5);
            // Actual M2 dissipation, standard Roe dissipation, integrated central flux.
            allocate(parts, "firstRoe.parts", 15);
            for (Index i = 0; i < mesh->NumNode(); i++)
                means[i] = solver.StateField()[i];
            SpatialOperator<3> spatial(mpi, mesh, topology, geometry, reconstruction,
                                       solver.Boundaries(), cfg.algorithm.mode, cfg.reconstruction,
                                       cfg.physics, cfg.time, nodeHalo);
            spatial.Initialize();
            // One RHS call performs exactly one coefficient build and one point recovery.
            // It evaluates the static spatial operator but does not update the state or time.
            spatial.EvaluateRHS(means, rhs);
            const auto &points = spatial.PointValues();
            const auto &gradients = spatial.Gradients();
            std::vector<State> exact(nodeHalo.NumNodeProc());
            std::vector<Gradient> exactGradient(nodeHalo.NumNodeProc());
            for (Index i = 0; i < nodeHalo.NumNodeProc(); i++)
                std::tie(exact[i], exactGradient[i]) =
                    IsentropicVortex<3>(cfg, nodeHalo.Coordinate(i), 0);

            NodeStatePair edges;
            edges.InitPair("firstRoe.edges", mpi);
            edges.father->Resize(topology.NumEdge(), 15, 1);
            edges.son->Resize(topology.NumEdgeGhost(), 15, 1);
            edges.BorrowSetup(const_cast<DNDS::Geom::tAdjPair &>(topology.Edge2Node()));
            edges.trans.initPersistentPull();
            const std::array<std::string, 5> names{"rho", "rhou", "rhov", "rhow", "rhoE"};
            Norms faceNorms, nodeNorms;
            real maxOracle = 0, maxConsistency = 0, maxReverse = 0;
            real minRho = DNDS::veryLargeReal, minPressure = DNDS::veryLargeReal;
            real alphaMin = DNDS::veryLargeReal, alphaMax = 0;
            Index gauss = 0;
            std::ofstream faceFile(output / fmt::format("edges.rank{:04d}.csv", mpi.rank));
            DNDS_check_throw_info(faceFile.good(), "Cannot write edge diagnostics");
            faceFile << "node0,node1,area,alpha,jump_rho,d_m2_rho,d_roe_rho,jump_rhoE,d_m2_rhoE,d_roe_rhoE\n"
                     << std::setprecision(17);
            for (Index e = 0; e < topology.NumEdge(); e++)
            {
                const auto &surface = geometry.EdgeSurface(e);
                const real area = surface.measure;
                const Vector3 normal = surface.vectorMeasure.normalized();
                gauss += surface.quadrature.size();
                auto surfaceMean = [&](int side, int mode) -> State
                {
                    // Counterfactuals share the very same geometry/weights; no new reconstruction.
                    // 0 actual, 1 exact points only, 2 exact gradients only, 3 exact both.
                    const std::size_t sideIndex = static_cast<std::size_t>(side);
                    const Index anchor = surface.nodes[sideIndex];
                    State sum = area *
                                (mode == 1 || mode == 3
                                     ? exact[anchor]
                                     : State(points[anchor]));
                    for (const EfficientSurfaceNode &entry : surface.efficientStencil)
                    {
                        sum += (mode >= 2
                                    ? exactGradient[entry.node]
                                    : Gradient(gradients[entry.node]))
                                   .transpose() *
                               entry.stateGradientWeights[sideIndex];
                    }
                    const State u = sum / area;
                    // Fail rather than silently omitting production's positivity fallback.
                    DNDS_check_throw_info(u.allFinite() && u(0) > 1e-12 && Pressure(u, cfg.physics.gamma) > 1e-12,
                                          "Surface positivity fallback would be required");
                    minRho = std::min(minRho, u(0));
                    minPressure = std::min(minPressure, Pressure(u, cfg.physics.gamma));
                    return u;
                };
                auto integrated = [&](int side) -> State
                {
                    const std::size_t sideIndex = static_cast<std::size_t>(side);
                    const Index anchor = surface.nodes[sideIndex];
                    State sum = PhysicalFlux(
                        points[anchor], surface.vectorMeasure,
                        cfg.physics.gamma);
                    for (const EfficientSurfaceNode &entry : surface.efficientStencil)
                    {
                        for (int d = 0; d < 3; d++)
                            for (int f = 0; f < 3; f++)
                                sum += entry.fluxGradientWeights[sideIndex](d, f) *
                                       FluxDerivative(
                                           points[entry.node],
                                           gradients[entry.node].row(d).transpose(),
                                           f, cfg.physics.gamma);
                    }
                    return sum;
                };
                const State left = surfaceMean(0, 0);
                const State right = surfaceMean(1, 0);
                const State jump = right - left;
                const State center = 0.5 * (PhysicalFlux(left, normal, cfg.physics.gamma) +
                                            PhysicalFlux(right, normal, cfg.physics.gamma));
                const State m2 = NumericalFlux(left, right, normal, cfg.physics.gamma, DNDS::Euler::Gas::Roe_M2);
                const State roe = NumericalFlux(left, right, normal, cfg.physics.gamma, DNDS::Euler::Gas::Roe);
                // Positive D convention: F_num = F_center - D.
                const State dM2 = center - m2, dRoe = center - roe;
                auto signal = [&](const State &u)
                {
                    return std::abs(u.segment<3>(1).dot(normal) / u(0)) +
                           std::sqrt(cfg.physics.gamma * Pressure(u, cfg.physics.gamma) / u(0));
                };
                const real alpha = std::max(signal(left), signal(right));
                alphaMin = std::min(alphaMin, alpha);
                alphaMax = std::max(alphaMax, alpha);
                maxOracle = std::max(maxOracle, (dM2 - 0.5 * alpha * jump).cwiseAbs().maxCoeff());
                maxConsistency = std::max(maxConsistency,
                                          (NumericalFlux(left, left, normal, cfg.physics.gamma, DNDS::Euler::Gas::Roe) -
                                           PhysicalFlux(left, normal, cfg.physics.gamma))
                                              .cwiseAbs()
                                              .maxCoeff());
                maxReverse = std::max(maxReverse,
                                      (roe + NumericalFlux(right, left, -normal, cfg.physics.gamma, DNDS::Euler::Gas::Roe)).cwiseAbs().maxCoeff());
                edges[e].segment<5>(0) = area * dM2;
                edges[e].segment<5>(5) = area * dRoe;
                edges[e].segment<5>(10) = 0.5 *
                                          (integrated(0) + integrated(1));
                for (int v = 0; v < 5; v++)
                {
                    faceNorms.Add("jump." + names[v], jump(v), area);
                    faceNorms.Add("M2.D." + names[v], dM2(v), area);
                    faceNorms.Add("Roe.D." + names[v], dRoe(v), area);
                    faceNorms.Add("M2.integrated_D." + names[v], area * dM2(v), area);
                    faceNorms.Add("Roe.integrated_D." + names[v], area * dRoe(v), area);
                }
                for (int mode = 1; mode < 4; mode++)
                {
                    const State l = surfaceMean(0, mode);
                    const State r = surfaceMean(1, mode);
                    const real a = std::max(signal(l), signal(r));
                    for (int v = 0; v < 5; v++)
                    {
                        const std::string prefix = "control" + std::to_string(mode) + ".";
                        faceNorms.Add(prefix + "jump." + names[v], r(v) - l(v), area);
                        faceNorms.Add(prefix + "M2.D." + names[v], 0.5 * a * (r(v) - l(v)), area);
                    }
                }
                faceFile << nodeHalo.LocalToGlobal(surface.nodes[0]) << ','
                         << nodeHalo.LocalToGlobal(surface.nodes[1]) << ',' << area << ',' << alpha;
                for (int v : {0, 4})
                    faceFile << ',' << jump(v) << ',' << dM2(v) << ',' << dRoe(v);
                faceFile << '\n';
            }
            faceFile.close();
            DNDS_check_throw_info(faceFile.good(), "Incomplete edge diagnostics");
            edges.trans.startPersistentPull();
            edges.trans.waitPersistentPull();
            for (Index i = 0; i < mesh->NumNode(); i++)
            {
                parts[i].setZero();
                const auto &volume = geometry.NodeVolume(i);
                gauss += volume.volumeQuadrature.size();
                for (const auto &piece : volume.boundaryPieces)
                {
                    gauss += piece.quadrature.size();
                    DNDS_check_throw_info(solver.Boundaries().Get(piece.zone).mode == BoundaryMode::Periodic,
                                          "This probe requires exclusively periodic boundaries");
                }
                for (const auto &incidence : topology.Node2Edge(i))
                    parts[i] += incidence.outwardSign * edges[incidence.edge] / volume.moments.measure;
                // D enters the RHS with +div(D); central flux with -div(C).
                parts[i].segment<5>(10) *= -1;
            }
            std::ofstream nodeFile(output / fmt::format("nodes.rank{:04d}.csv", mpi.rank));
            DNDS_check_throw_info(nodeFile.good(), "Cannot write node diagnostics");
            nodeFile << "original_node,x,y,z,partial_volume";
            for (const auto &group : {"M2_Rd", "Roe_Rd", "central_rhs", "production_rhs"})
                for (const auto &name : names)
                    nodeFile << ',' << group << '_' << name;
            nodeFile << '\n'
                     << std::setprecision(17);
            real volumeSum = 0, periodicCount = 0, maxMatch = 0, maxStateChange = 0;
            State conservationM2 = State::Zero(), conservationRoe = State::Zero();
            for (Index i = 0; i < mesh->NumNode(); i++)
            {
                const real volume = geometry.NodeVolume(i).moments.measure;
                volumeSum += volume;
                periodicCount += volume / nodeHalo.Volume(i);
                const State m2 = parts[i].segment<5>(0), roe = parts[i].segment<5>(5);
                const State center = parts[i].segment<5>(10);
                const State mismatch = m2 + center - State(rhs[i]);
                maxMatch = std::max(maxMatch, mismatch.cwiseAbs().maxCoeff());
                maxStateChange = std::max(maxStateChange,
                                          (State(means[i]) - State(solver.StateField()[i])).cwiseAbs().maxCoeff());
                conservationM2 += volume * m2;
                conservationRoe += volume * roe;
                for (int v = 0; v < 5; v++)
                {
                    nodeNorms.Add("M2.Rd." + names[v], m2(v), volume);
                    nodeNorms.Add("Roe.Rd." + names[v], roe(v), volume);
                    nodeNorms.Add("central_rhs." + names[v], center(v), volume);
                    nodeNorms.Add("production_rhs." + names[v], rhs[i](v), volume);
                    nodeNorms.Add("decomposition_mismatch." + names[v], mismatch(v), volume);
                    nodeNorms.Add("point_error." + names[v], points[i](v) - exact[i](v), volume);
                    nodeNorms.Add("gradient_error." + names[v],
                                  (Gradient(gradients[i]).col(v) - exactGradient[i].col(v)).norm(), volume);
                }
                nodeFile << mesh->node2nodeOrig(i, 0);
                for (int d = 0; d < 3; d++)
                    nodeFile << ',' << mesh->coords[i](d);
                nodeFile << ',' << volume;
                for (int v = 0; v < 15; v++)
                    nodeFile << ',' << parts[i](v);
                for (int v = 0; v < 5; v++)
                    nodeFile << ',' << rhs[i](v);
                nodeFile << '\n';
            }
            nodeFile.close();
            DNDS_check_throw_info(nodeFile.good(), "Incomplete node diagnostics");
            DNDS_check_throw_info(solver.CurrentIteration() == 0 && solver.SimulationTime() == 0,
                                  "Unexpected time advancement");
            real sumsLocal[]{volumeSum, periodicCount, real(topology.NumEdge()), real(gauss)};
            real sums[4]{};
            MPI_Allreduce(sumsLocal, sums, 4, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            real maximaLocal[]{maxOracle, maxConsistency, maxReverse, maxMatch, maxStateChange,
                               alphaMax, MPI_Wtime() - start};
            real maxima[7]{};
            MPI_Allreduce(maximaLocal, maxima, 7, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            real minimaLocal[]{minRho, minPressure, alphaMin}, minima[3]{};
            MPI_Allreduce(minimaLocal, minima, 3, DNDS::DNDS_MPI_REAL, MPI_MIN, mpi.comm);
            State sumM2, sumRoe;
            MPI_Allreduce(conservationM2.data(), sumM2.data(), 5, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(conservationRoe.data(), sumRoe.data(), 5, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            DNDS_check_throw_info(sums[3] == 0 && maxima[4] == 0 && maxima[0] < 1e-12 &&
                                      maxima[1] < 1e-12 && maxima[2] < 1e-12 && maxima[3] < 1e-10,
                                  "Static dissipation verification failed");
            const auto faceMetrics = faceNorms.Reduce(mpi), nodeMetrics = nodeNorms.Reduce(mpi);
            if (mpi.rank == 0)
            {
                const Index unknowns = std::llround(sums[1]);
                nlohmann::ordered_json metadata = {
                    {"configuration", cfg}, {"source_configuration", argv[1]}, {"time", 0}, {"time_steps", 0}, {"reconstruction_calls", 1}, {"rhs_evaluations", 1}, {"mpi_ranks", mpi.size}, {"nodes", mesh->NumNodeGlobal()}, {"cells", mesh->NumCellGlobal()}, {"owned_edges_global", sums[2]}, {"periodic_unknowns", unknowns}, {"volume", sums[0]}, {"h_3d", std::cbrt(sums[0] / unknowns)}, {"gauss_points_stored", sums[3]}, {"M2_formula_max_mismatch", maxima[0]}, {"Roe_consistency_max_mismatch", maxima[1]}, {"Roe_orientation_max_mismatch", maxima[2]}, {"rhs_decomposition_max_mismatch", maxima[3]}, {"mean_state_max_change", maxima[4]}, {"surface_rho_min", minima[0]}, {"surface_pressure_min", minima[1]}, {"alpha_min", minima[2]}, {"alpha_max", maxima[5]}, {"M2_dissipation_global_integral", std::vector<real>(sumM2.data(), sumM2.data() + 5)}, {"Roe_dissipation_global_integral", std::vector<real>(sumRoe.data(), sumRoe.data() + 5)}, {"wall_seconds", maxima[6]}, {"face_norms", faceMetrics}, {"node_norms", nodeMetrics}};
                std::ofstream out(output / "metrics.json");
                out << metadata.dump(2) << '\n';
                DNDS_check_throw_info(out.good(), "Cannot write final metrics");
                DNDS::log() << "STATIC COMPLETE: time=0, reconstructions=1, M2 rho D L2="
                            << faceMetrics["M2.D.rho"]["L2"] << ", M2 rho Rd L2="
                            << nodeMetrics["M2.Rd.rho"]["L2"] << std::endl;
            }
        }
        catch (const std::exception &exception)
        {
            std::cerr << "NCFV first Roe dissipation diagnostic: " << exception.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
