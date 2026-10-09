/** Production 3-D vortex benchmark, with optional Riemann trace/RHS diagnostics. */
#include "NCFV/NCFVSolver.hpp"
#include "riemann_flux_budget.hpp"

#include <sys/resource.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace DNDS::NCFV;

namespace
{
    double RankMaximum(const DNDS::MPIInfo &mpi, double local)
    {
        double global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        return global;
    }

    double RankSum(const DNDS::MPIInfo &mpi, double local)
    {
        double global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
        return global;
    }

    double PeakRSSMiB()
    {
        struct rusage usage{};
        DNDS_check_throw_info(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
        return static_cast<double>(usage.ru_maxrss) / 1024.0;
    }

    nlohmann::ordered_json ParseDiagnosticsRow(const std::string &row)
    {
        std::stringstream stream(row);
        std::string token;
        std::vector<double> values;
        while (std::getline(stream, token, ','))
            values.push_back(std::stod(token));
        DNDS_check_throw_info(values.size() == 12, "Unexpected vortex diagnostics CSV");
        return {{"time", values[1]}, {"volume", values[2]},
                {"rho_point_L1V", values[3]}, {"rho_point_L2V", values[4]},
                {"rho_point_Linf", values[5]}, {"mass", values[6]},
                {"entropy_L1V", values[11]}};
    }

    std::pair<nlohmann::ordered_json, nlohmann::ordered_json>
    ReadDiagnostics(const std::filesystem::path &path)
    {
        std::ifstream file(path);
        DNDS_check_throw_info(file.good(), "Missing vortex diagnostics CSV");
        std::string header, row, first, last;
        std::getline(file, header);
        while (std::getline(file, row))
            if (!row.empty())
            {
                if (first.empty())
                    first = row;
                last = row;
            }
        DNDS_check_throw_info(!last.empty(), "Empty vortex diagnostics CSV");
        auto initial = ParseDiagnosticsRow(first);
        auto final = ParseDiagnosticsRow(last);
        final["mass_drift"] = final["mass"].get<double>() - initial["mass"].get<double>();
        return {initial, final};
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
            DNDS_check_throw_info(argc >= 8 && argc <= 10,
                "usage: base-config mesh-file result.json Traditional|Efficient "
                "LeastSquares|SVDLeastSquares|Variational dt end-time "
                "[riemann-solver [distance-weight-power]]");
            const std::filesystem::path output = std::filesystem::absolute(argv[3]);
            DNDS_check_throw_info(!std::filesystem::exists(output),
                                  "Benchmark result already exists");
            std::ifstream base(argv[1]);
            DNDS_check_throw_info(base.good(), "Cannot open base configuration");
            nlohmann::ordered_json resolved = Configuration{};
            resolved.merge_patch(nlohmann::ordered_json::parse(base));
            if (argc >= 9)
                resolved["physics"]["riemannSolver"] = argv[8];
            for (auto &boundary : resolved["physics"]["boundaryZones"])
            {
                nlohmann::ordered_json normalized = BoundaryZoneSettings{};
                normalized.merge_patch(boundary);
                boundary = std::move(normalized);
            }
            auto cfg = resolved.get<Configuration>();
            cfg.mesh.meshFile = argv[2];
            const std::string mode = argv[4], method = argv[5];
            DNDS_check_throw_info(mode == "Traditional" || mode == "Efficient",
                                  "Unknown integration mode");
            DNDS_check_throw_info(method == "LeastSquares" ||
                                      method == "SVDLeastSquares" ||
                                      method == "Variational",
                                  "Unknown reconstruction method");
            cfg.algorithm.mode = mode == "Traditional" ?
                IntegrationMode::TraditionalQuadrature : IntegrationMode::EfficientDifferential;
            cfg.reconstruction.method = method == "LeastSquares" ?
                ReconstructionMethod::LeastSquares : method == "SVDLeastSquares" ?
                ReconstructionMethod::SVDLeastSquares : ReconstructionMethod::Variational;
            cfg.algorithm.retainMicroGeometry = false;
            cfg.algorithm.quadratureOrder = 4;
            cfg.algorithm.surfaceQuadratureOrder = 3;
            cfg.reconstruction.enableLimiter = false;
            cfg.reconstruction.stencilSizeFactor = 1.7;
            cfg.reconstruction.variationalWeight = 5.0;
            cfg.reconstruction.variationalIterations = 3;
            if (argc == 10)
                cfg.reconstruction.distanceWeightPower = std::stod(argv[9]);
            const double dt = std::stod(argv[6]);
            const double endTime = std::stod(argv[7]);
            const auto steps = static_cast<DNDS::index>(std::llround(endTime / dt));
            DNDS_check_throw_info(dt > 0 && endTime > 0 && steps > 0 &&
                                      std::abs(steps * dt - endTime) < 1e-12,
                                  "End time must be an integer multiple of dt");
            cfg.time.timeStep = dt;
            cfg.time.endTime = endTime;
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
            const auto runtime = output.parent_path() / output.stem();
            if (mpi.rank == 0)
            {
                DNDS_check_throw_info(!std::filesystem::exists(runtime),
                                      "Benchmark runtime directory already exists");
                std::filesystem::create_directories(runtime);
            }
            MPI_Barrier(mpi.comm);
            cfg.io.outputPrefix = (runtime / "solution").string();
            cfg.Validate();

            const double initializeStart = MPI_Wtime();
            Solver<3> solver(mpi, cfg);
            solver.Initialize();
            const double initializeSeconds = RankMaximum(mpi, MPI_Wtime() - initializeStart);
            const double initializedRSS = RankSum(mpi, PeakRSSMiB());
            const bool auditFlux = std::getenv("NCFV_AUDIT_FLUX_BUDGET") != nullptr;
            nlohmann::ordered_json initialBudget, finalBudget;
            if (auditFlux)
                initialBudget = RiemannFluxBudget(mpi, cfg, solver);
            const double marchStart = MPI_Wtime();
            solver.Run();
            const double marchSeconds = RankMaximum(mpi, MPI_Wtime() - marchStart);
            const double lastStepSeconds = RankMaximum(mpi, solver.LastStepTiming().totalSeconds);
            const double finalRSS = RankSum(mpi, PeakRSSMiB());
            if (auditFlux)
                finalBudget = RiemannFluxBudget(mpi, cfg, solver);
            DNDS_check_throw_info(std::abs(solver.SimulationTime() - endTime) < 1e-12,
                                  "Solver stopped before requested end time");
            if (mpi.rank == 0)
            {
                const auto [initial, final] =
                    ReadDiagnostics(cfg.io.outputPrefix + ".diagnostics.csv");
                nlohmann::ordered_json result = {
                    {"mesh", cfg.mesh.meshFile}, {"mode", mode}, {"method", method},
                    {"riemann_solver", nlohmann::ordered_json(cfg.physics.riemannSolver)},
                    {"configuration", nlohmann::ordered_json(cfg)},
                    {"nodes", solver.Mesh()->NumNodeGlobal()},
                    {"cells", solver.Mesh()->NumCellGlobal()}, {"mpi_ranks", mpi.size},
                    {"dt", dt}, {"steps", steps}, {"end_time", endTime},
                    {"variational_weight", cfg.reconstruction.variationalWeight},
                    {"variational_iterations", cfg.reconstruction.variationalIterations},
                    {"initialization_seconds", initializeSeconds},
                    {"march_seconds", marchSeconds},
                    {"last_step_core_seconds", lastStepSeconds},
                    {"initialization_peak_rss_sum_mib", initializedRSS},
                    {"final_peak_rss_sum_mib", finalRSS},
                    {"initial", initial}, {"final", final}};
                if (auditFlux)
                {
                    result["initial_flux_budget"] = initialBudget;
                    result["final_flux_budget"] = finalBudget;
                }
                std::ofstream out(output);
                DNDS_check_throw_info(out.good(), "Cannot write benchmark result");
                out << std::setw(2) << result << '\n';
                DNDS_check_throw_info(out.good(), "Benchmark result write failed");
                std::cout << result.dump(2) << std::endl;
            }
        }
        catch (const std::exception &error)
        {
            std::cerr << "NCFV benchmark: " << error.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
}
