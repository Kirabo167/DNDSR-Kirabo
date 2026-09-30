/** Compare Roe NCFV reconstruction/integration modes on uniformly refined prisms. */
#include "NCFV/NCFVSolver.hpp"
#include "NCFV/NCFVAnalytic.hpp"
#include "Geom/Quadrature.hpp"

#include <sys/resource.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace DNDS::NCFV;
using State = Eigen::Matrix<DNDS::real, 5, 1>;
using Vector = Eigen::Matrix<DNDS::real, 3, 1>;

namespace
{
    DNDS::real Pressure(const State &u, DNDS::real gamma)
    {
        return (gamma - 1) *
               (u(4) - 0.5 * u.segment<3>(1).squaredNorm() / u(0));
    }

    nlohmann::ordered_json FinalError(
        const DNDS::MPIInfo &mpi, const Solver<3> &solver,
        const Configuration &cfg)
    {
        const auto &mesh = *solver.Mesh();
        const auto &geometry = solver.Geometry();
        const auto &field = solver.StateField();
        const DNDS::real time = solver.SimulationTime();
        DNDS::Geom::Elem::Quadrature quadrature({DNDS::Geom::Elem::Tet4}, 6);
        DNDS_check_throw_info(quadrature.GetNumPoints() == 24,
                              "Expected the 24-point micro-tetrahedron reference rule");
        DNDS::real local[15]{}, global[15]{};
        DNDS::real localMax = 0, globalMax = 0;
        for (DNDS::index i = 0; i < mesh.NumNode(); i++)
        {
            const auto &volume = geometry.NodeVolume(i);
            const DNDS::real measure = volume.moments.measure;
            DNDS_check_throw_info(measure > 0 && !volume.microVolumes.empty(),
                                  "Missing positive dual volume or retained micro-tetrahedra");
            State exactMean = State::Zero();
            for (const auto &micro : volume.microVolumes)
                for (int q = 0; q < quadrature.GetNumPoints(); q++)
                {
                    const auto [p, weight] = quadrature.GetQuadraturePointInfo(q);
                    const Vector x = micro.points[0] +
                        p[0] * (micro.points[1] - micro.points[0]) +
                        p[1] * (micro.points[2] - micro.points[0]) +
                        p[2] * (micro.points[3] - micro.points[0]);
                    exactMean += 6 * micro.measure * weight / measure *
                                 IsentropicVortex<3>(cfg, x, time).first;
                }
            const State numerical = field[i];
            DNDS_check_throw_info(numerical.allFinite() && numerical(0) > 0 &&
                                  Pressure(numerical, cfg.physics.gamma) > 0,
                                  "Nonphysical or nonfinite final control-volume mean");
            const State error = numerical - exactMean;
            local[0] += measure;
            local[1] += measure * std::abs(error(0));
            local[2] += measure * error(0) * error(0);
            for (int v = 0; v < 5; v++)
                local[3 + v] += measure * error(v) * error(v);
            const DNDS::real pressureError = Pressure(numerical, cfg.physics.gamma) -
                                             Pressure(exactMean, cfg.physics.gamma);
            local[8] += measure * pressureError * pressureError;
            local[9] += measure * error(0);
            local[10] += measure * numerical(0);
            local[11] += measure * exactMean(0);
            localMax = std::max(localMax, std::abs(error(0)));
        }
        MPI_Allreduce(local, global, 15, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        MPI_Allreduce(&localMax, &globalMax, 1, DNDS::DNDS_MPI_REAL, MPI_MAX,
                      mpi.comm);
        nlohmann::ordered_json conservedL2 = nlohmann::ordered_json::array();
        for (int v = 0; v < 5; v++)
            conservedL2.push_back(std::sqrt(global[3 + v] / global[0]));
        return {
            {"time", time}, {"volume", global[0]},
            {"rho_mean_L1V", global[1] / global[0]},
            {"rho_mean_L2V", std::sqrt(global[2] / global[0])},
            {"rho_mean_Linf", globalMax},
            {"conserved_mean_L2V", conservedL2},
            {"pressure_of_mean_L2V", std::sqrt(global[8] / global[0])},
            {"rho_mean_signed_bias", global[9] / global[0]},
            {"mass_numerical", global[10]},
            {"mass_exact_reference", global[11]},
            {"reference_quadrature_order", 6},
            {"reference_points_per_micro_tetrahedron", quadrature.GetNumPoints()},
        };
    }

    DNDS::real GlobalMemoryHighWaterMiB(const DNDS::MPIInfo &mpi)
    {
        struct rusage usage{};
        DNDS_check_throw_info(getrusage(RUSAGE_SELF, &usage) == 0,
                              "getrusage failed");
        const DNDS::real local = static_cast<DNDS::real>(usage.ru_maxrss) / 1024;
        DNDS::real total = 0;
        MPI_Allreduce(&local, &total, 1, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        return total;
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
            DNDS_check_throw_info(argc == 9 || argc == 10,
                "usage: base-config mesh-file output.json Variational|LeastSquares "
                "Traditional|Efficient dt final-time target-stencil [variational-w]");
            const std::filesystem::path output(argv[3]);
            DNDS_check_throw_info(!std::filesystem::exists(output),
                                  "Accuracy output already exists");
            std::ifstream stream(argv[1]);
            DNDS_check_throw_info(stream.good(), "Cannot read base configuration");
            const auto raw = nlohmann::ordered_json::parse(stream);
            nlohmann::ordered_json resolved = Configuration{};
            resolved.merge_patch(raw);
            for (auto &boundary : resolved["physics"]["boundaryZones"])
            {
                nlohmann::ordered_json normalized = BoundaryZoneSettings{};
                normalized.merge_patch(boundary);
                boundary = std::move(normalized);
            }
            auto cfg = resolved.get<Configuration>();
            cfg.mesh.meshFile = argv[2];
            const std::string method = argv[4];
            DNDS_check_throw_info(method == "Variational" || method == "LeastSquares",
                                  "Unknown reconstruction method");
            cfg.reconstruction.method = method == "Variational" ?
                ReconstructionMethod::Variational : ReconstructionMethod::LeastSquares;
            if (argc == 10)
                cfg.reconstruction.variationalWeight = std::stod(argv[9]);
            const std::string mode = argv[5];
            DNDS_check_throw_info(mode == "Traditional" || mode == "Efficient",
                                  "Unknown integration mode");
            DNDS_check_throw_info(method != "Variational" || mode != "Efficient",
                                  "Variational reconstruction requires Traditional mode");
            cfg.algorithm.mode = mode == "Traditional" ?
                IntegrationMode::TraditionalQuadrature : IntegrationMode::EfficientDifferential;
            cfg.algorithm.retainMicroGeometry = true;
            cfg.algorithm.quadratureOrder = 4;
            cfg.algorithm.surfaceQuadratureOrder = 3;
            cfg.physics.riemannSolver = DNDS::Euler::Gas::Roe;
            cfg.physics.viscous.enabled = false;
            cfg.reconstruction.enableLimiter = false;
            const int targetStencil = std::stoi(argv[8]);
            DNDS_check_throw_info(targetStencil >= 9, "Need at least nine stencil nodes");
            cfg.reconstruction.stencilSizeFactor = std::nextafter(
                static_cast<DNDS::real>(targetStencil) / 9.0, 0.0);
            const DNDS::real dt = std::stod(argv[6]);
            const DNDS::real finalTime = std::stod(argv[7]);
            const auto steps = static_cast<DNDS::index>(std::llround(finalTime / dt));
            DNDS_check_throw_info(dt > 0 && finalTime > 0 && steps > 0 &&
                                  std::abs(steps * dt - finalTime) < 1e-12,
                                  "Invalid fixed time step or end time");
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
            const auto runtimeDirectory =
                std::filesystem::temp_directory_path() /
                "ncfv_refined_prism_accuracy" /
                output.parent_path().filename() / output.stem();
            if (mpi.rank == 0)
            {
                DNDS_check_throw_info(!std::filesystem::exists(runtimeDirectory),
                                      "Runtime diagnostics already exist");
                std::filesystem::create_directories(runtimeDirectory);
            }
            MPI_Barrier(mpi.comm);
            cfg.io.outputPrefix = (runtimeDirectory / "solution").string();
            cfg.Validate();

            const double start = MPI_Wtime();
            Solver<3> solver(mpi, cfg);
            solver.Initialize();
            const DNDS::real initializationSeconds = MPI_Wtime() - start;
            const DNDS::real initializedHwmMiB = GlobalMemoryHighWaterMiB(mpi);
            const double marchStart = MPI_Wtime();
            solver.Run();
            const DNDS::real marchSeconds = MPI_Wtime() - marchStart;
            DNDS_check_throw_info(std::abs(solver.SimulationTime() - finalTime) < 1e-12,
                                  "Solver stopped before the requested final time");
            const auto final = FinalError(mpi, solver, cfg);
            const DNDS::real finalHwmMiB = GlobalMemoryHighWaterMiB(mpi);
            if (mpi.rank == 0)
            {
                nlohmann::ordered_json result = {
                    {"configuration", cfg},
                    {"mesh", cfg.mesh.meshFile}, {"method", method}, {"mode", mode},
                    {"variational_weight", cfg.reconstruction.variationalWeight},
                    {"time_integrator", "SSPRK3"},
                    {"runtime_diagnostics_directory", runtimeDirectory.string()},
                    {"nodes", solver.Mesh()->NumNodeGlobal()},
                    {"cells", solver.Mesh()->NumCellGlobal()},
                    {"mpi_ranks", mpi.size}, {"stencil_target", targetStencil},
                    {"variational_definition", method == "Variational"
                         ? "traditional_midpoint" : "none"},
                    {"dt", dt}, {"steps", steps}, {"final_time", finalTime},
                    {"initialization_seconds", initializationSeconds},
                    {"march_seconds", marchSeconds},
                    {"initialized_hwm_sum_mib", initializedHwmMiB},
                    {"final_hwm_sum_mib", finalHwmMiB},
                    {"final", final},
                };
                std::ofstream outputStream(output);
                DNDS_check_throw_info(outputStream.good(), "Cannot write accuracy JSON");
                outputStream << std::setw(2) << result << '\n';
                DNDS_check_throw_info(outputStream.good(), "Accuracy JSON write failed");
                std::cout << result.dump(2) << std::endl;
            }
        }
        catch (const std::exception &error)
        {
            std::cerr << "NCFV refined accuracy: " << error.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
