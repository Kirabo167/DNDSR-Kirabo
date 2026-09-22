/** First NCFV reconstruction only: no RHS evaluation, flux or time stepping. */
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
    std::pair<State, Gradient> Primitive(const State &u, const Gradient &g, real gamma)
    {
        State q;
        Gradient dq;
        const Vector3 velocity = u.segment<3>(1) / u(0);
        q << u(0), velocity, (gamma - 1) * (u(4) - 0.5 * u(0) * velocity.squaredNorm());
        dq.col(0) = g.col(0);
        for (int d = 0; d < 3; d++)
        {
            dq.row(d).segment<3>(1) =
                (g.row(d).segment<3>(1) - g(d, 0) * velocity.transpose()) / u(0);
            dq(d, 4) = (gamma - 1) *
                (g(d, 4) - velocity.dot(g.row(d).segment<3>(1)) +
                 0.5 * velocity.squaredNorm() * g(d, 0));
        }
        return {q, dq};
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
            DNDS_check_throw_info(argc == 3, "Usage: first_reconstruction_probe config.json output-directory");
            auto cfg = LoadConfiguration(argv[1], {}, {}).configuration;
            DNDS_check_throw_info(cfg.dimension == 3 && cfg.initialField.isentropicVortex,
                                  "This diagnostic requires the 3-D analytic vortex");
            DNDS_check_throw_info(cfg.algorithm.mode == IntegrationMode::EfficientDifferential &&
                                      !cfg.reconstruction.enableLimiter && cfg.io.restartInput.empty(),
                                  "Use the efficient, unlimited fresh-field configuration");
            cfg.time.iterations = 0;
            cfg.time.endTime = 0;
            cfg.io.writeVTK = false;
            cfg.io.writeInitial = false;
            cfg.io.writeFinal = false;
            cfg.io.writeFinalRestart = false;
            cfg.io.restartInterval = 0;
            cfg.io.writeResolvedConfiguration = false;
            const std::filesystem::path output = argv[2];
            if (mpi.rank == 0)
            {
                DNDS_check_throw_info(!std::filesystem::exists(output),
                                      "Refusing to overwrite a diagnostic output directory");
                std::filesystem::create_directories(output);
            }
            MPI_Barrier(mpi.comm);

            const double start = MPI_Wtime();
            Solver<3> solver(mpi, cfg);
            solver.Initialize(); // Builds geometry/operators and analytic dual means; no reconstruction.
            const auto mesh = solver.Mesh();
            const auto &geometry = solver.Geometry();
            const auto &reconstruction = solver.ReconstructionData();
            const NodeHalo &nodeHalo = solver.NodeCommunication();
            const auto &means = solver.StateField(); // Already synchronized by Solver::Initialize.
            NodeMatrixPair gradients, coefficients;
            gradients.InitPair("NCFV.firstReconstruction.gradients", mpi);
            gradients.father->Resize(mesh->NumNode(), 3, 5);
            gradients.son->Resize(nodeHalo.NumNodeGhost(), 3, 5);
            gradients.BorrowSetup(nodeHalo.Layout());
            gradients.trans.initPersistentPull();
            NodeStatePair points;
            points.InitPair("NCFV.firstReconstruction.points", mpi);
            points.father->Resize(mesh->NumNode(), 5, 1);
            points.son->Resize(nodeHalo.NumNodeGhost(), 5, 1);
            points.BorrowSetup(nodeHalo.Layout());
            points.trans.initPersistentPull();

            // Exactly the unlimited efficient SpatialOperator::Reconstruct sequence.
            // Call each public reconstruction kernel once, on the real initialized solver state.
            reconstruction.ComputeCoefficients(means, gradients, coefficients);
            gradients.trans.startPersistentPull();
            gradients.trans.waitPersistentPull();
            reconstruction.RecoverPointValues(means, gradients, coefficients, points);
            points.trans.startPersistentPull();
            points.trans.waitPersistentPull();
            DNDS_check_throw_info(solver.CurrentIteration() == 0 && solver.SimulationTime() == 0,
                                  "Unexpected time advancement in reconstruction diagnostic");

            std::ofstream nodes(output / fmt::format("nodes.rank{:04d}.csv", mpi.rank));
            DNDS_check_throw_info(nodes.good(), "Cannot open reconstruction node output");
            const std::array<std::string, 5> names{"rho", "rhou", "rhov", "rhow", "rhoE"};
            const std::array<std::string, 5> primitiveNames{"rho", "u", "v", "w", "p"};
            const std::array<std::string, 3> axes{"x", "y", "z"};
            nodes << "original_node,x,y,z,partial_volume";
            for (const auto &name : names) nodes << ",mean_" << name;
            for (const auto &name : names) nodes << ",point_" << name;
            for (const auto &name : names)
                for (const auto &axis : axes) nodes << ",d_" << name << "_d" << axis;
            nodes << '\n' << std::setprecision(17);

            std::map<std::string, std::array<real, 3>> accumulators;
            auto accumulate = [&](const std::string &name, real error, real volume)
            {
                auto &a = accumulators[name];
                a[0] += volume * std::abs(error);
                a[1] += volume * error * error;
                a[2] = std::max(a[2], std::abs(error));
            };
            real localVolume = 0, localPeriodicCount = 0, localStencilSum = 0;
            real localStencilMin = DNDS::veryLargeReal, localStencilMax = 0;
            real localConditionMax = 0;
            Index localGauss = 0;
            for (Index i = 0; i < mesh->NumNode(); i++)
            {
                const Vector3 xyz = mesh->coords[i];
                const real volume = geometry.NodeVolume(i).moments.measure;
                const State value = points[i];
                const Gradient derivative = gradients[i];
                DNDS_check_throw_info(value.allFinite() && derivative.allFinite(), "Nonfinite reconstruction");
                const auto [exact, exactGradient] = IsentropicVortex<3>(cfg, xyz, 0);
                const auto [primitive, primitiveGradient] = Primitive(value, derivative, cfg.physics.gamma);
                const auto [exactPrimitive, exactPrimitiveGradient] = Primitive(exact, exactGradient, cfg.physics.gamma);
                for (int v = 0; v < 5; v++)
                {
                    const std::string c = "conservative." + names[v];
                    const std::string p = "primitive." + primitiveNames[v];
                    accumulate(c + ".value", value(v) - exact(v), volume);
                    accumulate(p + ".value", primitive(v) - exactPrimitive(v), volume);
                    accumulate(c + ".gradient", (derivative.col(v) - exactGradient.col(v)).norm(), volume);
                    accumulate(p + ".gradient", (primitiveGradient.col(v) - exactPrimitiveGradient.col(v)).norm(), volume);
                    for (int d = 0; d < 3; d++)
                    {
                        accumulate(c + ".d" + axes[d], derivative(d, v) - exactGradient(d, v), volume);
                        accumulate(p + ".d" + axes[d], primitiveGradient(d, v) - exactPrimitiveGradient(d, v), volume);
                    }
                }
                localVolume += volume;
                const auto &op = reconstruction.Operator(i);
                localPeriodicCount += volume / std::pow(op.lengthScale, 3);
                localStencilSum += op.stencil.size();
                localStencilMin = std::min(localStencilMin, real(op.stencil.size()));
                localStencilMax = std::max(localStencilMax, real(op.stencil.size()));
                localConditionMax = std::max(localConditionMax, op.conditionNumber);
                localGauss += geometry.NodeVolume(i).volumeQuadrature.size();
                for (const auto &piece : geometry.NodeVolume(i).boundaryPieces)
                    localGauss += piece.quadrature.size();
                nodes << mesh->node2nodeOrig(i, 0) << ',' << xyz(0) << ',' << xyz(1) << ',' << xyz(2) << ',' << volume;
                for (int v = 0; v < 5; v++) nodes << ',' << means[i](v);
                for (int v = 0; v < 5; v++) nodes << ',' << value(v);
                for (int v = 0; v < 5; v++)
                    for (int d = 0; d < 3; d++) nodes << ',' << derivative(d, v);
                nodes << '\n';
            }
            nodes.close();
            DNDS_check_throw_info(nodes.good(), "Failed to finish reconstruction node output");
            for (Index e = 0; e < solver.EdgeTopology().NumEdge(); e++)
                localGauss += geometry.EdgeSurface(e).quadrature.size();
            real localSums[]{localVolume, localPeriodicCount, localStencilSum};
            real sums[3]{};
            MPI_Allreduce(localSums, sums, 3, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            real minimumStencil = 0;
            MPI_Allreduce(&localStencilMin, &minimumStencil, 1, DNDS::DNDS_MPI_REAL, MPI_MIN, mpi.comm);
            real localMaxima[]{localStencilMax, localConditionMax, real(localGauss), MPI_Wtime() - start};
            real maxima[4]{};
            MPI_Allreduce(localMaxima, maxima, 4, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(maxima[2] == 0, "Efficient diagnostic stored Gauss points");
            nlohmann::ordered_json errors;
            for (const auto &[name, local] : accumulators)
            {
                real sum[2]{}, maximum = 0;
                MPI_Allreduce(local.data(), sum, 2, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                MPI_Allreduce(&local[2], &maximum, 1, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
                errors[name] = {{"L1", sum[0] / sums[0]}, {"L2", std::sqrt(sum[1] / sums[0])}, {"Linf", maximum}};
            }
            if (mpi.rank == 0)
            {
                const Index periodicCount = std::llround(sums[1]);
                nlohmann::ordered_json result = {
                    {"method", MethodName}, {"configuration", cfg}, {"source_configuration", argv[1]},
                    {"time", solver.SimulationTime()}, {"iteration", solver.CurrentIteration()},
                    {"compute_coefficients_calls", 1}, {"recover_point_values_calls", 1},
                    {"rhs_evaluations", 0}, {"time_steps", 0}, {"gauss_points_stored", 0},
                    {"mpi_ranks", mpi.size}, {"nodes", mesh->NumNodeGlobal()}, {"cells", mesh->NumCellGlobal()},
                    {"periodic_unknowns", periodicCount}, {"volume", sums[0]},
                    {"h_3d", std::cbrt(sums[0] / periodicCount)},
                    {"stencil_min", minimumStencil}, {"stencil_max", maxima[0]},
                    {"stencil_mean_owned_nodes", sums[2] / mesh->NumNodeGlobal()},
                    {"condition_max", maxima[1]}, {"wall_seconds", maxima[3]}, {"errors", errors}};
                std::ofstream metadata(output / "metrics.json");
                metadata << result.dump(2) << '\n';
                DNDS_check_throw_info(metadata.good(), "Cannot write reconstruction metrics");
                DNDS::log() << "First reconstruction only: rho point L2="
                            << errors["conservative.rho.value"]["L2"] << ", rho gradient L2="
                            << errors["conservative.rho.gradient"]["L2"] << std::endl;
            }
        }
        catch (const std::exception &exception)
        {
            std::cerr << "NCFV first-reconstruction diagnostic: " << exception.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
