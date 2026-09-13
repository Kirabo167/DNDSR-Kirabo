// Read-only diagnostic of the existing NCFV library. No solver source changes.
// Extra quadrature is used only for analytic reference integrals in this probe.
#include "NCFV/NCFVSolver.hpp"
#include "NCFV/NCFVAnalytic.hpp"
#include "Geom/Quadrature.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

using DNDS::real;
using Index = DNDS::index;
using namespace DNDS::NCFV;
using State = Eigen::Matrix<real, 5, 1>;
using Gradient = Eigen::Matrix<real, 3, 5>;

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    int status = 0;
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        try
        {
            DNDS_check_throw_info(argc >= 3, "config.json output.json [time] [reference-order]");
            auto cfg = LoadConfiguration(argv[1], {}, {}).configuration;
            const real time = argc > 3 ? std::stod(argv[3]) : 0.0;
            const int referenceOrder = argc > 4 ? std::stoi(argv[4]) : 5;
            const int compactTarget = argc > 5 ? std::stoi(argv[5]) : 0;
            const bool evolve = argc > 6 && std::string(argv[6]) == "evolve";
            cfg.algorithm.retainMicroGeometry = true;
            cfg.io.writeVTK = false;
            cfg.io.writeFinalRestart = evolve;
            cfg.io.restartInterval = 0;
            cfg.io.writeResolvedConfiguration = evolve;
            if (evolve)
            {
                const std::string prefix = std::filesystem::path(argv[2]).replace_extension().string() + "_run/";
                cfg.io.outputPrefix = prefix + "solution";
                cfg.io.restartPrefix = prefix + "restart";
                cfg.io.outputInterval = 0;
                DNDS_check_throw_info(!std::filesystem::exists(prefix + "solution.resolved.json"),
                                      "Refusing to overwrite an existing diagnostic evolution");
            }
            Solver<3> solver(mpi, cfg);
            solver.Initialize();
            const auto mesh = solver.Mesh();
            const auto &geom = solver.Geometry();
            const auto &topo = solver.EdgeTopology();
            PeriodicNodes periodic(mpi, mesh, geom, cfg.mesh);
            periodic.Build(topo, cfg.mesh.periodicTolerance);
            // Diagnostic counterfactual ONLY: replace the in-memory operators of
            // this private probe instance; existing libraries/files are unchanged.
            // Keep all face neighbours, then add nearest remaining nodes until
            // the target size and full quadratic rank are both attained.
            real localStencilSum = 0, globalStencilSum = 0;
            int localStencilMin = 1000000, localStencilMax = 0, globalStencilMin = 0, globalStencilMax = 0;
            for (Index i = 0; i < mesh->NumNode(); i++)
            {
                auto &op = const_cast<ReconstructionOperator &>(solver.ReconstructionData().Operator(i));
                const Index rep = periodic.Representative(i);
                if (compactTarget > 0)
                {
                    const auto &direct = periodic.Graph()[rep];
                    auto candidates = op.stencil;
                    const auto isDirect = [&](Index j)
                    {
                        return std::find(direct.begin(), direct.end(), j) != direct.end();
                    };
                    std::sort(candidates.begin(), candidates.end(), [&](Index a, Index b)
                              {
                        if (isDirect(a) != isDirect(b)) return isDirect(a);
                        const real da = periodic.Displacement(rep, a).squaredNorm();
                        const real db = periodic.Displacement(rep, b).squaredNorm();
                        return da != db ? da < db : a < b; });
                    bool accepted = false;
                    for (int count = std::max(compactTarget, static_cast<int>(direct.size()));
                         count <= static_cast<int>(candidates.size()); count++)
                    {
                        Eigen::MatrixXd matrix(count, 9);
                        Eigen::VectorXd weights(count);
                        for (int j = 0; j < count; j++)
                        {
                            const Vector3 delta = periodic.Displacement(rep, candidates[j]);
                            matrix.row(j) = (Reconstruction::MeanBasis(
                                                 periodic.Moments(candidates[j], delta), Vector3::Zero(),
                                                 op.referenceLengths, 3) -
                                             op.targetBasisMean)
                                                .transpose();
                            weights(j) = std::pow(std::max(delta.norm() / op.lengthScale,
                                                           cfg.reconstruction.distanceWeightFloor),
                                                  -cfg.reconstruction.distanceWeightPower);
                        }
                        const Eigen::MatrixXd weighted = weights.asDiagonal() * matrix;
                        Eigen::JacobiSVD<Eigen::MatrixXd> svd(weighted, Eigen::ComputeThinU | Eigen::ComputeThinV);
                        const auto s = svd.singularValues();
                        if (s(8) <= cfg.reconstruction.svdTolerance * s(0))
                            continue;
                        op.inverseRows = (svd.matrixV() * s.cwiseInverse().asDiagonal() *
                                          svd.matrixU().transpose() * weights.asDiagonal())
                                             .topRows(3);
                        op.stencil.assign(candidates.begin(), candidates.begin() + count);
                        accepted = true;
                        break;
                    }
                    DNDS_check_throw_info(accepted, "Compact diagnostic stencil was rank deficient");
                }
                const int count = static_cast<int>(op.stencil.size());
                localStencilSum += count;
                localStencilMin = std::min(localStencilMin, count);
                localStencilMax = std::max(localStencilMax, count);
            }
            MPI_Allreduce(&localStencilSum, &globalStencilSum, 1, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(&localStencilMin, &globalStencilMin, 1, MPI_INT, MPI_MIN, mpi.comm);
            MPI_Allreduce(&localStencilMax, &globalStencilMax, 1, MPI_INT, MPI_MAX, mpi.comm);
            if (evolve)
                solver.Run();
            SpatialOperator<3> spatial(mpi, mesh, topo, geom, solver.ReconstructionData(),
                                       solver.Boundaries(), cfg.algorithm.mode,
                                       cfg.reconstruction, cfg.physics, cfg.time, &periodic);
            spatial.Initialize();
            auto nodeField = [&](NodeStatePair &a, const std::string &name, int rows)
            {
                DNDS::CFV::BuildUDofOnMesh(a, name, mpi, mesh, rows, true, true, DNDS::Geom::MeshLoc::Node);
            };
            NodeStatePair means, exactMeans, exactRhs, rhs, referenceRhs, contributions;
            nodeField(means, "probe.means", 5);
            nodeField(exactMeans, "probe.exactMeans", 5);
            nodeField(exactRhs, "probe.exactRhs", 5);
            nodeField(rhs, "probe.rhs", 5);
            nodeField(referenceRhs, "probe.referenceRhs", 5);
            nodeField(contributions, "probe.contributions", 8);
            std::vector<State> exactPoints(mesh->NumNodeProc());
            std::vector<Gradient> exactGradients(mesh->NumNodeProc());
            for (Index i = 0; i < mesh->NumNodeProc(); i++)
                std::tie(exactPoints[i], exactGradients[i]) = IsentropicVortex<3>(cfg, mesh->coords[i], time);
            DNDS::Geom::Elem::Quadrature quadrature({DNDS::Geom::Elem::Tet4}, referenceOrder);
            for (Index i = 0; i < mesh->NumNode(); i++)
            {
                const auto &volume = geom.NodeVolume(i);
                means[i] = exactPoints[i];
                for (const auto &w : volume.pointRecoveryWeights)
                    means[i] += exactGradients[w.node].transpose() * w.value;
                exactMeans[i].setZero();
                exactRhs[i].setZero();
                for (const auto &micro : volume.microVolumes)
                    for (int q = 0; q < quadrature.GetNumPoints(); q++)
                    {
                        const auto [p, w] = quadrature.GetQuadraturePointInfo(q);
                        const Vector3 x = micro.points[0] +
                                          p[0] * (micro.points[1] - micro.points[0]) +
                                          p[1] * (micro.points[2] - micro.points[0]) +
                                          p[2] * (micro.points[3] - micro.points[0]);
                        const auto [u, g] = IsentropicVortex<3>(cfg, x, time);
                        const real weight = 6 * micro.measure * w / volume.moments.measure;
                        exactMeans[i] += weight * u;
                        exactRhs[i] -= weight * (g.row(0) + g.row(1)).transpose();
                    }
            }
            periodic.Average(means);
            periodic.Average(exactMeans);
            periodic.Average(exactRhs);
            spatial.EvaluateRHS(exactMeans, referenceRhs);
            real referencePointLocal = 0;
            for (Index i = 0; i < mesh->NumNode(); i++)
                referencePointLocal += geom.NodeVolume(i).moments.measure *
                                       DNDS::sqr(spatial.PointValues()[i](0) - exactPoints[i](0));
            spatial.EvaluateRHS(means, rhs);

            NodeStatePair edgeParts;
            edgeParts.InitPair("probe.edgeParts", mpi);
            edgeParts.father->Resize(topo.NumEdge(), 8, 1);
            edgeParts.son->Resize(topo.NumEdgeGhost(), 8, 1);
            edgeParts.BorrowSetup(const_cast<DNDS::Geom::tAdjPair &>(topo.Edge2Node()));
            edgeParts.trans.initPersistentPull();
            auto physicalMass = [](const State &u, const Vector3 &normal)
            {
                return u.segment<3>(1).dot(normal);
            };
            auto numericalMass = [&](const State &left, const State &right, const Vector3 &normal)
            {
                State flux = State::Zero();
                real am = 0, ac = 0, ap = 0;
                const Vector3 grid = Vector3::Zero();
                DNDS::Euler::Gas::InviscidFlux_IdealGas_Dispatcher<3>(
                    cfg.physics.riemannSolver, left, right, left, right,
                    grid, normal, cfg.physics.gamma, cfg.physics.gamma,
                    flux, 0.0, 1.0, 1.0,
                    []() {}, am, ac, ap);
                return flux(0);
            };
            // Modes: actual point/gradient, exact point/computed gradient,
            // exact point/exact gradient, computed point/exact gradient.
            for (Index e = 0; e < topo.NumEdge(); e++)
            {
                const auto &face = geom.EdgeSurface(e);
                const Vector3 normal = face.vectorMeasure.normalized();
                for (int mode = 0; mode < 4; mode++)
                {
                    auto pointAt = [&](Index i) -> State
                    {
                        return mode == 1 || mode == 2 ? exactPoints[i] : State(spatial.PointValues()[i]);
                    };
                    auto gradientAt = [&](Index i) -> Gradient
                    {
                        return mode >= 2 ? exactGradients[i] : Gradient(spatial.Gradients()[i]);
                    };
                    auto surfaceMean = [&](Index anchor, const auto &weights) -> State
                    {
                        State value = pointAt(anchor);
                        for (const auto &w : weights)
                            value += gradientAt(w.node).transpose() * w.value / face.measure;
                        return value;
                    };
                    auto integratedMass = [&](Index anchor, const auto &weights)
                    {
                        real value = physicalMass(pointAt(anchor), face.vectorMeasure);
                        for (const auto &w : weights)
                        {
                            const Gradient g = gradientAt(w.node);
                            for (int d = 0; d < 3; d++)
                                for (int f = 0; f < 3; f++)
                                    value += w.value(d, f) * g(d, 1 + f);
                        }
                        return value;
                    };
                    const State left = surfaceMean(face.nodes[0], face.leftStateWeights);
                    const State right = surfaceMean(face.nodes[1], face.rightStateWeights);
                    edgeParts[e](2 * mode) = 0.5 *
                                             (integratedMass(face.nodes[0], face.leftFluxWeights) +
                                              integratedMass(face.nodes[1], face.rightFluxWeights));
                    edgeParts[e](2 * mode + 1) = face.measure *
                                                 (numericalMass(left, right, normal) -
                                                  0.5 * (physicalMass(left, normal) + physicalMass(right, normal)));
                }
            }
            edgeParts.trans.startPersistentPull();
            edgeParts.trans.waitPersistentPull();
            for (Index i = 0; i < mesh->NumNode(); i++)
            {
                contributions[i].setZero();
                for (const auto &incidence : topo.Node2Edge(i))
                    contributions[i] -= incidence.outwardSign * edgeParts[incidence.edge] /
                                        geom.NodeVolume(i).moments.measure;
            }
            periodic.Average(contributions);
            std::vector<std::string> names{
                "mean_projection_L2", "point_recovery_L2", "gradient_rho_L2",
                "gradient_momentum_L2", "rhs_density_L2", "rhs_exact_means_L2",
                "decomposition_match_L2", "point_recovery_exact_means_L2"};
            for (int mode = 0; mode < 4; mode++)
                for (const auto &s : {"central_error", "dissipation", "total_error"})
                    names.push_back("mode" + std::to_string(mode) + "_" + s + "_L2");
            std::vector<real> local(names.size(), 0), global(names.size(), 0);
            local[7] = referencePointLocal;
            real localVolume = 0, globalVolume = 0;
            for (Index i = 0; i < mesh->NumNode(); i++)
            {
                const real v = geom.NodeVolume(i).moments.measure;
                localVolume += v;
                local[0] += v * DNDS::sqr(means[i](0) - exactMeans[i](0));
                local[1] += v * DNDS::sqr(spatial.PointValues()[i](0) - exactPoints[i](0));
                const Gradient dg = Gradient(spatial.Gradients()[i]) - exactGradients[i];
                local[2] += v * dg.col(0).squaredNorm();
                local[3] += v * dg.middleCols<3>(1).squaredNorm();
                local[4] += v * DNDS::sqr(rhs[i](0) - exactRhs[i](0));
                local[5] += v * DNDS::sqr(referenceRhs[i](0) - exactRhs[i](0));
                local[6] += v * DNDS::sqr(rhs[i](0) - contributions[i](0) - contributions[i](1));
                for (int mode = 0; mode < 4; mode++)
                {
                    const real central = contributions[i](2 * mode) - exactRhs[i](0);
                    const real diss = contributions[i](2 * mode + 1);
                    local[8 + 3 * mode] += v * central * central;
                    local[9 + 3 * mode] += v * diss * diss;
                    local[10 + 3 * mode] += v * DNDS::sqr(central + diss);
                }
            }
            MPI_Allreduce(local.data(), global.data(), static_cast<int>(local.size()), DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(&localVolume, &globalVolume, 1, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            if (mpi.rank == 0)
            {
                nlohmann::ordered_json out;
                out["mesh"] = cfg.mesh.meshFile;
                out["analytic_time"] = time;
                out["reference_quadrature_order"] = referenceOrder;
                out["compact_stencil_target"] = compactTarget;
                out["stencil_min"] = globalStencilMin;
                out["stencil_max"] = globalStencilMax;
                out["stencil_mean"] = globalStencilSum / mesh->NumNodeGlobal();
                out["evolution_time"] = solver.SimulationTime();
                out["evolution_output_prefix"] = evolve ? cfg.io.outputPrefix : "";
                out["nodes"] = mesh->NumNodeGlobal();
                out["volume"] = globalVolume;
                for (std::size_t i = 0; i < names.size(); i++)
                    out[names[i]] = std::sqrt(global[i] / globalVolume);
                std::ofstream(argv[2]) << std::setw(4) << out << '\n';
                std::cout << out.dump(2) << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
            status = 1;
        }
    }
    MPI_Finalize();
    return status;
}
