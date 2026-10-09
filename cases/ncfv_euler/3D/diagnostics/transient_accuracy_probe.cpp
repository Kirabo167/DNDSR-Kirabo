/** Run the unmodified NCFV solver and export recovered states at t=0 and t=end. */
#include "NCFV/NCFVSolver.hpp"
#include "NCFV/NCFVAnalytic.hpp"

#include <array>
#include <fstream>
#include <iomanip>

namespace
{
    using namespace DNDS::NCFV;
    using State = Eigen::Matrix<DNDS::real, 5, 1>;

    nlohmann::ordered_json Snapshot(
        const DNDS::MPIInfo &mpi, const Solver<3> &solver,
        const Configuration &cfg, const std::filesystem::path &directory,
        const std::string &label, bool writePoints)
    {
        const auto mesh = solver.Mesh();
        const auto &geometry = solver.Geometry();
        const auto &reconstruction = solver.ReconstructionData();
        const auto &nodeHalo = solver.NodeCommunication();
        const auto &means = solver.StateField();
        NodeMatrixPair gradients, coefficients;
        gradients.InitPair("NCFV.transientAudit.gradients", mpi);
        gradients.father->Resize(mesh->NumNode(), 3, 5);
        gradients.son->Resize(nodeHalo.NumNodeGhost(), 3, 5);
        gradients.BorrowSetup(nodeHalo.Layout());
        gradients.trans.initPersistentPull();
        NodeStatePair points;
        points.InitPair("NCFV.transientAudit.points", mpi);
        points.father->Resize(mesh->NumNode(), 5, 1);
        points.son->Resize(nodeHalo.NumNodeGhost(), 5, 1);
        points.BorrowSetup(nodeHalo.Layout());
        points.trans.initPersistentPull();
        reconstruction.ComputeCoefficients(means, gradients, coefficients);
        gradients.trans.startPersistentPull();
        gradients.trans.waitPersistentPull();
        reconstruction.RecoverPointValues(means, gradients, coefficients, points);

        const auto pressure = [&](const State &u)
        {
            return (cfg.physics.gamma - 1) * (u(4) - 0.5 * u.segment<3>(1).squaredNorm() / u(0));
        };
        std::ofstream out;
        if (writePoints)
        {
            out.open(directory / fmt::format(
                                     "points_{}.rank{:04d}.csv",
                                     label, mpi.rank));
            DNDS_check_throw_info(out.good(), "Cannot open transient audit output");
            out << "original_node,x,y,z,partial_volume,mean_rho,mean_rhou,mean_rhov,mean_rhow,mean_rhoE,"
                   "point_rho,point_rhou,point_rhov,point_rhow,point_rhoE,rho_exact,p_point,p_exact\n"
                << std::setprecision(17);
        }
        DNDS::real sums[5]{}, maxima[3]{};
        DNDS::index gauss = 0;
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            const State point = points[i];
            const State exact = IsentropicVortex<3>(cfg, mesh->coords[i], solver.SimulationTime()).first;
            const DNDS::real p = pressure(point), pExact = pressure(exact);
            DNDS_check_throw_info(point.allFinite() && std::isfinite(p) && point(0) > 0 && p > 0,
                                  "Nonphysical recovered state in transient audit");
            const DNDS::real volume = geometry.NodeVolume(i).moments.measure;
            sums[0] += volume;
            for (int v = 0; v < 2; v++)
            {
                const DNDS::real error = v == 0 ? point(0) - exact(0) : p - pExact;
                sums[1 + 2 * v] += volume * std::abs(error);
                sums[2 + 2 * v] += volume * error * error;
                maxima[v] = std::max(maxima[v], std::abs(error));
            }
            maxima[2] = std::max(maxima[2], reconstruction.Operator(i).conditionNumber);
            gauss += geometry.NodeVolume(i).volumeQuadrature.size();
            for (const auto &piece : geometry.NodeVolume(i).boundaryPieces)
                gauss += piece.quadrature.size();
            if (writePoints)
            {
                out << mesh->node2nodeOrig(i, 0);
                for (int d = 0; d < 3; d++)
                    out << ',' << mesh->coords[i](d);
                out << ',' << volume;
                for (int v = 0; v < 5; v++)
                    out << ',' << means[i](v);
                for (int v = 0; v < 5; v++)
                    out << ',' << point(v);
                out << ',' << exact(0) << ',' << p << ',' << pExact << '\n';
            }
        }
        if (writePoints)
        {
            out.close();
            DNDS_check_throw_info(out.good(), "Transient audit output failed");
        }
        for (const auto &surface : geometry.EdgeSurfaces())
            gauss += surface.quadrature.size();
        DNDS::real globalSums[5]{}, globalMaxima[3]{};
        DNDS::index globalGauss = 0;
        MPI_Allreduce(sums, globalSums, 5, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        MPI_Allreduce(maxima, globalMaxima, 3, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        MPI_Allreduce(&gauss, &globalGauss, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        DNDS_check_throw_info(globalGauss == 0, "Efficient solver stored Gauss points");
        nlohmann::ordered_json result = {{"time", solver.SimulationTime()},
                                         {"iteration", solver.CurrentIteration()},
                                         {"volume", globalSums[0]},
                                         {"condition_max", globalMaxima[2]},
                                         {"gauss_points_stored", globalGauss}};
        for (int v = 0; v < 2; v++)
            result[v == 0 ? "rho_point" : "pressure_point"] = {
                {"L1", globalSums[1 + 2 * v] / globalSums[0]},
                {"L2", std::sqrt(globalSums[2 + 2 * v] / globalSums[0])},
                {"Linf", globalMaxima[v]}};
        return result;
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
                argc == 2 || argc == 5 || argc == 6,
                "Usage: transient_accuracy_probe config.json [output-directory reconstruction-method fixed-dt [target-stencil-size]]");
            auto cfg = DNDS::NCFV::LoadConfiguration(
                           argv[1], {}, {})
                           .configuration;
            const bool compactBatch = argc >= 5;
            if (compactBatch)
            {
                cfg.io.outputPrefix =
                    (std::filesystem::path(argv[2]) / "solution").string();
                cfg.reconstruction.method =
                    nlohmann::ordered_json(argv[3])
                        .get<DNDS::NCFV::ReconstructionMethod>();
                cfg.time.timeStep = std::stod(argv[4]);
                if (argc == 6)
                {
                    constexpr int basisSize = 9;
                    const int targetStencil = std::stoi(argv[5]);
                    DNDS_check_throw_info(
                        targetStencil >= basisSize,
                        "Target stencil is smaller than the quadratic basis");
                    cfg.reconstruction.stencilSizeFactor = std::nextafter(
                        static_cast<DNDS::real>(targetStencil) / basisSize,
                        0.0);
                    DNDS_check_throw_info(
                        static_cast<int>(std::ceil(
                            cfg.reconstruction.stencilSizeFactor *
                            basisSize)) == targetStencil,
                        "Failed to encode the requested stencil size");
                }
                cfg.time.useCFLTimeStep = false;
                cfg.time.useLocalTimeStep = false;
                cfg.time.endTime = 1.6;
                cfg.time.iterations = 100000;
                cfg.io.writeVTK = false;
                cfg.io.writeInitial = false;
                cfg.io.writeFinal = false;
                cfg.io.writeFinalRestart = false;
                cfg.io.restartInterval = 0;
                cfg.io.writeResolvedConfiguration = false;
                cfg.Validate();
            }
            DNDS_check_throw_info(cfg.dimension == 3 && cfg.initialField.isentropicVortex &&
                                      cfg.algorithm.mode == DNDS::NCFV::IntegrationMode::EfficientDifferential &&
                                      !cfg.reconstruction.enableLimiter && cfg.io.restartInput.empty(),
                                  "Expected fresh unlimited efficient 3-D vortex");
            DNDS_check_throw_info(!cfg.time.useLocalTimeStep && cfg.time.endTime >= 0 &&
                                      (cfg.time.useCFLTimeStep ? cfg.time.cfl > 0 : cfg.time.timeStep > 0),
                                  "Expected a global positive CFL or fixed physical time step");
            const auto directory = std::filesystem::path(cfg.io.outputPrefix).parent_path();
            if (mpi.rank == 0)
            {
                DNDS_check_throw_info(!std::filesystem::exists(directory), "Refusing to overwrite a run directory");
                std::filesystem::create_directories(directory);
            }
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            DNDS::NCFV::Solver<3> solver(mpi, cfg);
            solver.Initialize();
            const auto initial = Snapshot(
                mpi, solver, cfg, directory, "initial", !compactBatch);
            solver.EvaluateResidual(); // Initial selected step, before end-time clipping.
            DNDS::real localMin = DNDS::veryLargeReal, localMax = 0, minStep = 0, maxStep = 0;
            for (DNDS::index i = 0; i < solver.Mesh()->NumNode(); i++)
            {
                localMin = std::min(localMin, solver.LocalTimeStep(i));
                localMax = std::max(localMax, solver.LocalTimeStep(i));
            }
            MPI_Allreduce(&localMin, &minStep, 1, DNDS::DNDS_MPI_REAL, MPI_MIN, mpi.comm);
            MPI_Allreduce(&localMax, &maxStep, 1, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(minStep > 0 && std::abs(maxStep - minStep) < 1e-13,
                                  "Initial physical step is nonpositive or nonuniform");
            if (!cfg.time.useCFLTimeStep)
                DNDS_check_throw_info(std::abs(minStep - cfg.time.timeStep) < 1e-13,
                                      "Initial fixed physical step differs from configured timeStep");
            if (mpi.rank == 0)
                DNDS::log() << "Transient audit: "
                            << (cfg.time.useCFLTimeStep ? "CFL=" : "fixed dt=")
                            << (cfg.time.useCFLTimeStep ? cfg.time.cfl : cfg.time.timeStep)
                            << ", initial dt="
                            << std::setprecision(17) << minStep << ", target t=" << cfg.time.endTime << std::endl;
            solver.Run(); // Existing production SSPRK3, reconstruction and flux path unchanged.
            DNDS_check_throw_info(std::abs(solver.SimulationTime() - cfg.time.endTime) < 1e-12,
                                  "Solver stopped before requested physical time");
            const auto final = Snapshot(
                mpi, solver, cfg, directory, "final", !compactBatch);
            const double localSeconds = MPI_Wtime() - start;
            double seconds = 0;
            MPI_Allreduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            if (mpi.rank == 0)
            {
                nlohmann::ordered_json result = {{"configuration", cfg}, {"mpi_ranks", mpi.size}, {"normalization", "dual-bounds-half-span (thesis 3-34)"}, {"initial_cfl_step", minStep}, {"wall_seconds", seconds}, {"initial", initial}, {"final", final}};
                std::ofstream out(directory / "accuracy.json");
                out << result.dump(2) << '\n';
                out.close();
                DNDS_check_throw_info(out.good(), "Cannot finish transient accuracy metadata");
                DNDS::log() << "Transient audit complete: " << final.dump() << ", wall_seconds=" << seconds << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "NCFV transient audit: " << e.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
