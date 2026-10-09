/** Audit production NCFV time integration with selectable fixed step and surface quadrature. */
#include "NCFV/NCFVSolver.hpp"
#include "NCFV/NCFVAnalytic.hpp"
#include "Geom/Quadrature.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

namespace
{
    using namespace DNDS::NCFV;
    using State = Eigen::Matrix<DNDS::real, 5, 1>;

    nlohmann::ordered_json Snapshot(
        const DNDS::MPIInfo &mpi, const Solver<3> &solver,
        const Configuration &cfg, const std::filesystem::path &directory,
        const std::string &label)
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
        coefficients.InitPair("NCFV.traditionalTransientAudit.coefficients", mpi);
        coefficients.father->Resize(mesh->NumNode(), 9, 5);
        coefficients.son->Resize(nodeHalo.NumNodeGhost(), 9, 5);
        coefficients.BorrowSetup(nodeHalo.Layout());
        coefficients.trans.initPersistentPull();
        NodeStatePair points;
        points.InitPair("NCFV.transientAudit.points", mpi);
        points.father->Resize(mesh->NumNode(), 5, 1);
        points.son->Resize(nodeHalo.NumNodeGhost(), 5, 1);
        points.BorrowSetup(nodeHalo.Layout());
        points.trans.initPersistentPull();
        reconstruction.ComputeCoefficients(means, gradients, coefficients);
        coefficients.trans.startPersistentPull();
        coefficients.trans.waitPersistentPull();
        reconstruction.RecoverPointValues(means, gradients, coefficients, points);

        const auto pressure = [&](const State &u)
        {
            return (cfg.physics.gamma - 1) * (u(4) - 0.5 * u.segment<3>(1).squaredNorm() / u(0));
        };
        std::ofstream out(directory / fmt::format("points_{}.rank{:04d}.csv", label, mpi.rank));
        DNDS_check_throw_info(out.good(), "Cannot open transient audit output");
        out << "original_node,x,y,z,partial_volume,mean_rho,mean_rhou,mean_rhov,mean_rhow,mean_rhoE,"
               "point_rho,point_rhou,point_rhov,point_rhow,point_rhoE,rho_exact,p_point,p_exact\n"
            << std::setprecision(17);
        DNDS::real sums[5]{}, maxima[3]{};
        DNDS::index gauss = 0;
        int localMinimumStencil = std::numeric_limits<int>::max();
        int localMaximumStencil = 0;
        DNDS::index localStencilSum = 0;
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
            if (cfg.algorithm.mode == IntegrationMode::TraditionalQuadrature)
                DNDS_check_throw_info(reconstruction.Operator(i).inverseRows.rows() == 9,
                                      "Traditional reconstruction must retain all nine basis rows");
            const int stencilSize = static_cast<int>(reconstruction.Operator(i).stencil.size());
            localMinimumStencil = std::min(localMinimumStencil, stencilSize);
            localMaximumStencil = std::max(localMaximumStencil, stencilSize);
            localStencilSum += stencilSize;
            maxima[2] = std::max(maxima[2], reconstruction.Operator(i).conditionNumber);
            gauss += geometry.NodeVolume(i).volumeQuadrature.size();
            for (const auto &piece : geometry.NodeVolume(i).boundaryPieces)
                gauss += piece.quadrature.size();
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
        out.close();
        DNDS_check_throw_info(out.good(), "Transient audit output failed");
        for (const auto &surface : geometry.EdgeSurfaces())
            gauss += surface.quadrature.size();
        DNDS::real globalSums[5]{}, globalMaxima[3]{};
        DNDS::index globalGauss = 0;
        int globalMinimumStencil = 0, globalMaximumStencil = 0;
        DNDS::index globalStencilSum = 0;
        MPI_Allreduce(sums, globalSums, 5, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        MPI_Allreduce(maxima, globalMaxima, 3, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        MPI_Allreduce(&gauss, &globalGauss, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        MPI_Allreduce(&localMinimumStencil, &globalMinimumStencil, 1, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(&localMaximumStencil, &globalMaximumStencil, 1, MPI_INT, MPI_MAX, mpi.comm);
        MPI_Allreduce(&localStencilSum, &globalStencilSum, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        if (cfg.algorithm.mode == IntegrationMode::TraditionalQuadrature)
            DNDS_check_throw_info(globalGauss > 0, "Traditional solver has no quadrature storage");
        else
            DNDS_check_throw_info(globalGauss == 0, "Efficient solver stored Gauss points");
        nlohmann::ordered_json result = {{"time", solver.SimulationTime()},
                                         {"iteration", solver.CurrentIteration()},
                                         {"volume", globalSums[0]},
                                         {"condition_max", globalMaxima[2]},
                                         {"stencil_size_min", globalMinimumStencil},
                                         {"stencil_size_mean", static_cast<DNDS::real>(globalStencilSum) / mesh->NumNodeGlobal()},
                                         {"stencil_size_max", globalMaximumStencil},
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
                argc == 3 || argc == 6 || argc == 7 || argc == 8 || argc == 9,
                "Usage: traditional_transient_accuracy_probe source-config.json output-directory "
                "[traditional|efficient fixed-time-step end-time [surface-order [target-stencil-size [reconstruction-method]]]]");
            auto cfg = DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
            DNDS_check_throw_info(cfg.dimension == 3 && cfg.initialField.isentropicVortex &&
                                      !cfg.reconstruction.enableLimiter && cfg.io.restartInput.empty() &&
                                      !cfg.physics.viscous.enabled &&
                                      cfg.physics.riemannSolver == DNDS::Euler::Gas::Roe,
                                  "Expected a fresh unlimited inviscid Roe 3-D vortex source configuration");
            DNDS_check_throw_info(!cfg.time.useLocalTimeStep && cfg.time.endTime >= 0 &&
                                      (cfg.time.useCFLTimeStep ? cfg.time.cfl > 0 : cfg.time.timeStep > 0),
                                  "Expected a global positive CFL or fixed physical time step");
            const bool fixedStep = argc >= 6;
            if (fixedStep)
            {
                const std::string mode = argv[3];
                DNDS_check_throw_info(mode == "traditional" || mode == "efficient",
                                      "Fixed-step mode must be traditional or efficient");
                cfg.algorithm.mode = mode == "traditional" ?
                                         DNDS::NCFV::IntegrationMode::TraditionalQuadrature :
                                         DNDS::NCFV::IntegrationMode::EfficientDifferential;
            }
            else
                cfg.algorithm.mode = DNDS::NCFV::IntegrationMode::TraditionalQuadrature;
            cfg.algorithm.quadratureOrder = 4;
            cfg.algorithm.surfaceQuadratureOrder = argc == 7 ? std::stoi(argv[6]) : 3;
            if (argc >= 8)
                cfg.algorithm.surfaceQuadratureOrder = std::stoi(argv[6]);
            DNDS_check_throw_info(cfg.algorithm.surfaceQuadratureOrder >= 3 &&
                                      cfg.algorithm.surfaceQuadratureOrder <= 7,
                                  "Surface order must be between 3 and 7");
            cfg.algorithm.retainMicroGeometry = false;
            if (argc >= 8)
            {
                const int targetStencilSize = std::stoi(argv[7]);
                const int basisSize = Reconstruction::QuadraticBasisSize(3);
                DNDS_check_throw_info(targetStencilSize >= basisSize,
                                      "Target stencil size is smaller than the quadratic basis");
                cfg.reconstruction.stencilSizeFactor = std::nextafter(
                    static_cast<DNDS::real>(targetStencilSize) / basisSize,
                    0.0);
                DNDS_check_throw_info(
                    static_cast<int>(std::ceil(
                        cfg.reconstruction.stencilSizeFactor * basisSize)) ==
                        targetStencilSize,
                    "Failed to encode the requested target stencil size");
            }
            if (argc == 9)
                cfg.reconstruction.method =
                    nlohmann::ordered_json(argv[8]).get<DNDS::NCFV::ReconstructionMethod>();
            // Check the actual dispatched rule, not just the configuration label.
            const int requestedDegree = cfg.algorithm.SurfaceQuadraturePolynomialDegree();
            DNDS::Geom::Elem::Quadrature triangle({DNDS::Geom::Elem::Tri3}, requestedDegree);
            std::vector<DNDS::real> degreeErrors;
            for (int degree = 0; degree <= requestedDegree; degree++)
            {
                DNDS::real maximumError = 0;
                for (int xPower = 0; xPower <= degree; xPower++)
                {
                    const int yPower = degree - xPower;
                    DNDS::real integral = 0;
                    for (int q = 0; q < triangle.GetNumPoints(); q++)
                    {
                        const auto [point, weight] = triangle.GetQuadraturePointInfo(q);
                        integral += weight * std::pow(point(0), xPower) * std::pow(point(1), yPower);
                    }
                    const DNDS::real exact = std::tgamma(xPower + 1.0) * std::tgamma(yPower + 1.0) /
                                             std::tgamma(degree + 3.0);
                    maximumError = std::max(maximumError, std::abs(integral - exact));
                }
                degreeErrors.push_back(maximumError);
                DNDS_check_throw_info(maximumError < 1e-13, "Surface rule failed polynomial exactness check");
            }
            if (fixedStep)
            {
                const DNDS::real timeStep = std::stod(argv[4]);
                const DNDS::real endTime = std::stod(argv[5]);
                DNDS_check_throw_info(timeStep > 0 && endTime > 0,
                                      "Fixed time step and end time must be positive");
                const auto iterations = static_cast<DNDS::index>(std::llround(endTime / timeStep));
                DNDS_check_throw_info(iterations > 0 &&
                                          std::abs(endTime - iterations * timeStep) < 1e-13,
                                      "End time must be an integer multiple of the fixed time step");
                cfg.time.iterations = iterations;
                cfg.time.endTime = endTime;
                cfg.time.timeStep = timeStep;
                cfg.time.useCFLTimeStep = false;
                cfg.time.useLocalTimeStep = false;
            }
            cfg.io.writeVTK = false;
            cfg.io.writeInitial = false;
            cfg.io.writeFinal = false;
            cfg.io.outputInterval = 0;
            cfg.io.restartInterval = 0;
            cfg.io.writeFinalRestart = false;
            cfg.io.writeResolvedConfiguration = true;
            const auto directory = std::filesystem::absolute(argv[2]);
            cfg.io.outputPrefix = (directory / "solution").string();
            cfg.io.vtkSeriesName = (directory / "series").string();
            cfg.io.restartPrefix = (directory / "restart").string();
            if (mpi.rank == 0)
            {
                DNDS_check_throw_info(!std::filesystem::exists(directory), "Refusing to overwrite a run directory");
                std::filesystem::create_directories(directory);
            }
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            DNDS::NCFV::Solver<3> solver(mpi, cfg);
            solver.Initialize();
            const auto initial = Snapshot(mpi, solver, cfg, directory, "initial");
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
            const auto final = Snapshot(mpi, solver, cfg, directory, "final");
            const double localSeconds = MPI_Wtime() - start;
            double seconds = 0;
            MPI_Allreduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            if (mpi.rank == 0)
            {
                nlohmann::ordered_json result = {{"configuration", cfg}, {"mpi_ranks", mpi.size}, {"normalization", "dual-bounds-half-span (thesis 3-34)"}, {"initial_time_step", minStep}, {"wall_seconds", seconds}, {"initial", initial}, {"final", final},
                    {"surface_quadrature_check", {{"requested_polynomial_degree", requestedDegree},
                                                  {"points_per_micro_triangle", triangle.GetNumPoints()},
                                                  {"maximum_absolute_moment_error_by_degree", degreeErrors}}}};
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
