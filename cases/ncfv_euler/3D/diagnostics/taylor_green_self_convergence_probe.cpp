/** Periodic compressible Taylor--Green vortex self-convergence for weighted LS NCFV. */
#include "NCFV/NCFVSolver.hpp"
#include "Geom/Quadrature.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace DNDS::NCFV;
using DNDS::real;
using Index = DNDS::index;
using State = Eigen::Matrix<real, 5, 1>;
using Vector = Eigen::Matrix<real, 3, 1>;

namespace
{
    struct TaylorGreen
    {
        Vector k;
        real gamma = 1.4, mach = 0.1;

        State Conservative(const Vector &x) const
        {
            const real X = k.x() * x.x();
            const real Y = k.y() * x.y();
            const real Z = k.z() * x.z();
            const Vector velocity{
                std::sin(X) * std::cos(Y) * std::cos(Z),
                -std::cos(X) * std::sin(Y) * std::cos(Z),
                0};
            const real pressure = 1 / (gamma * mach * mach) +
                ((std::cos(2 * X) + std::cos(2 * Y)) *
                 (2 + std::cos(2 * Z))) / 16;
            const real rho = gamma * mach * mach * pressure;
            State state;
            state << rho, rho * velocity,
                pressure / (gamma - 1) + 0.5 * rho * velocity.squaredNorm();
            return state;
        }

        State Primitive(const State &u) const
        {
            State result;
            result(0) = u(0);
            result.segment<3>(1) = u.segment<3>(1) / u(0);
            result(4) = (gamma - 1) *
                (u(4) - 0.5 * u.segment<3>(1).squaredNorm() / u(0));
            return result;
        }
    };


    void Allocate(NodeStatePair &field, const char *name,
                  const DNDS::MPIInfo &mpi, Index owned, const NodeHalo &halo)
    {
        field.InitPair(name, mpi);
        field.father->Resize(owned, 5, 1);
        field.son->Resize(halo.NumNodeGhost(), 5, 1);
        field.BorrowSetup(halo.Layout());
        field.trans.initPersistentPull();
        for (Index i = 0; i < field.Size(); i++)
            field[i].setZero();
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
            DNDS_check_throw_info(argc == 10 || argc == 11,
                "usage: config.json output.json mach dt final-time stencil-size Traditional|Efficient distance-weight-power mesh-path [LeastSquares|Variational]");
            const std::filesystem::path output(argv[2]);
            DNDS_check_throw_info(!std::filesystem::exists(output),
                                  "Wave output already exists");
            std::ifstream input(argv[1]);
            DNDS_check_throw_info(input.good(), "Cannot read configuration");
            const auto raw = nlohmann::ordered_json::parse(input);
            nlohmann::ordered_json resolved = Configuration{};
            resolved.merge_patch(raw);
            for (auto &boundary : resolved["physics"]["boundaryZones"])
            {
                nlohmann::ordered_json normalized = BoundaryZoneSettings{};
                normalized.merge_patch(boundary);
                boundary = std::move(normalized);
            }
            auto cfg = resolved.get<Configuration>();
            const real mach = std::stod(argv[3]);
            DNDS_check_throw_info(std::isfinite(mach) && mach > 0,
                                  "Mach number must be finite and positive");
            const int stencil = std::stoi(argv[6]);
            DNDS_check_throw_info(stencil >= 9, "Need at least nine stencil nodes");
            const std::string mode(argv[7]);
            DNDS_check_throw_info(mode == "Traditional" || mode == "Efficient",
                                  "Mode must be Traditional or Efficient");
            const real distanceWeightPower = std::stod(argv[8]);
            DNDS_check_throw_info(std::isfinite(distanceWeightPower) &&
                                  distanceWeightPower >= 0,
                                  "Distance weight power must be finite and nonnegative");
            cfg.mesh.meshFile = argv[9];
            const std::string method = argc == 11 ? argv[10] : "LeastSquares";
            DNDS_check_throw_info(method == "LeastSquares" || method == "Variational",
                                  "Unknown reconstruction method");
            cfg.reconstruction.method = method == "Variational" ?
                ReconstructionMethod::Variational : ReconstructionMethod::LeastSquares;
            cfg.reconstruction.variationalWeight = 5;
            cfg.reconstruction.variationalIterations = 3;
            cfg.reconstruction.stencilSizeFactor =
                std::nextafter(static_cast<real>(stencil) / 9.0, 0.0);
            cfg.reconstruction.distanceWeightPower = distanceWeightPower;
            cfg.reconstruction.enableLimiter = false;
            cfg.algorithm.mode = mode == "Traditional" ?
                IntegrationMode::TraditionalQuadrature : IntegrationMode::EfficientDifferential;
            cfg.algorithm.retainMicroGeometry = true;
            cfg.algorithm.surfaceQuadratureOrder = 3;
            cfg.algorithm.quadratureOrder = 4;
            cfg.physics.riemannSolver = DNDS::Euler::Gas::Roe;
            cfg.physics.viscous.enabled = false;
            cfg.initialField.isentropicVortex = false;
            const real dt = std::stod(argv[4]);
            const real endTime = std::stod(argv[5]);
            const Index steps = std::llround(endTime / dt);
            DNDS_check_throw_info(dt > 0 && endTime > 0 && steps > 0 &&
                                  std::abs(steps * dt - endTime) < 1e-12,
                                  "Invalid fixed time step or end time");
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

            const Vector k{
                2 * DNDS::pi / cfg.mesh.periodicLengths[0],
                2 * DNDS::pi / cfg.mesh.periodicLengths[1],
                2 * DNDS::pi / cfg.mesh.periodicLengths[2],
            };
            const TaylorGreen wave{k, cfg.physics.gamma, mach};
            real periodicMismatch = 0;
            for (int d = 0; d < 3; d++)
            {
                const Vector x{0.371, 0.619, 0.913};
                Vector shifted = x;
                shifted(d) += cfg.mesh.periodicLengths[d];
                periodicMismatch = std::max(periodicMismatch,
                    (wave.Conservative(x) - wave.Conservative(shifted))
                        .cwiseAbs().maxCoeff());
            }
            DNDS_check_throw_info(periodicMismatch < 1e-12,
                                  "Taylor-Green initial field is not periodic");

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
            const bool midpointFlux = std::getenv("NCFV_TRADITIONAL_EDGE_MIDPOINT") != nullptr;
            spatial.SetTraditionalEdgeMidpointFluxForDiagnostics(midpointFlux);
            const bool referenceRoe = std::getenv("NCFV_REFERENCE_ROE_HARTEN") != nullptr;
            spatial.SetReferenceRoeHartenForDiagnostics(referenceRoe);
            const bool referenceDocument = std::getenv("NCFV_REFERENCE_DOCUMENT_FLUX") != nullptr;
            spatial.SetReferenceDocumentFluxForDiagnostics(referenceDocument);
            NodeStatePair state, base, stage, rhs;
            Allocate(state, "exactWave.state", mpi, mesh.NumNode(), halo);
            Allocate(base, "exactWave.base", mpi, mesh.NumNode(), halo);
            Allocate(stage, "exactWave.stage", mpi, mesh.NumNode(), halo);
            Allocate(rhs, "exactWave.rhs", mpi, mesh.NumNode(), halo);
            DNDS::Geom::Elem::Quadrature reference(
                {DNDS::Geom::Elem::Tet4}, 6);
            DNDS_check_throw_info(reference.GetNumPoints() == 24,
                                  "Reference must use 24 points per micro tetrahedron");
            const auto exactMeanAt = [&](const decltype(geometry.NodeVolume(0)) &volume)
            {
                const real measure = volume.moments.measure;
                State exactMean = State::Zero();
                for (const auto &micro : volume.microVolumes)
                    for (int q = 0; q < reference.GetNumPoints(); q++)
                    {
                        const auto [p, weight] = reference.GetQuadraturePointInfo(q);
                        const Vector x = micro.points[0] +
                            p[0] * (micro.points[1] - micro.points[0]) +
                            p[1] * (micro.points[2] - micro.points[0]) +
                            p[2] * (micro.points[3] - micro.points[0]);
                        exactMean += 6 * micro.measure * weight / measure *
                                     wave.Conservative(x);
                    }
                return exactMean;
            };
            for (Index i = 0; i < mesh.NumNode(); i++)
                state[i] = exactMeanAt(geometry.NodeVolume(i));
            State localInitial = State::Zero(), initialConserved = State::Zero();
            for (Index i = 0; i < mesh.NumNode(); i++)
                localInitial += geometry.NodeVolume(i).moments.measure * State(state[i]);
            MPI_Allreduce(localInitial.data(), initialConserved.data(), 5,
                          DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);

            const double start = MPI_Wtime();
            for (Index step = 0; step < steps; step++)
            {
                for (Index i = 0; i < mesh.NumNode(); i++)
                    base[i] = state[i];
                spatial.EvaluateRHS(state, rhs);
                for (Index i = 0; i < mesh.NumNode(); i++)
                    stage[i] = base[i] + dt * rhs[i];
                spatial.EvaluateRHS(stage, rhs);
                for (Index i = 0; i < mesh.NumNode(); i++)
                    stage[i] = 0.75 * base[i] + 0.25 * (stage[i] + dt * rhs[i]);
                spatial.EvaluateRHS(stage, rhs);
                for (Index i = 0; i < mesh.NumNode(); i++)
                    state[i] = (1.0 / 3.0) * base[i] +
                               (2.0 / 3.0) * (stage[i] + dt * rhs[i]);
            }
            spatial.ResetRiemannSolverCallCount();
            spatial.EvaluateRHS(state, rhs);
            const Index localCalls = spatial.RiemannSolverCallCount();
            Index globalCalls = 0;
            MPI_Allreduce(&localCalls, &globalCalls, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            const double localMarchSeconds = MPI_Wtime() - start;
            double marchSeconds = 0;
            MPI_Allreduce(&localMarchSeconds, &marchSeconds, 1,
                          MPI_DOUBLE, MPI_MAX, mpi.comm);

            real localMinimum[2]{DNDS::veryLargeReal, DNDS::veryLargeReal};
            Index localNonphysical = 0;
            State localFinal = State::Zero(), finalConserved = State::Zero();
            real localKinetic = 0, kinetic = 0;
            real localVolume = 0, totalVolume = 0;
            std::ostringstream fieldName;
            fieldName << output.string() << ".rank" << std::setw(4)
                      << std::setfill('0') << mpi.rank << ".csv";
            DNDS_check_throw_info(!std::filesystem::exists(fieldName.str()),
                                  "Taylor-Green field output already exists");
            std::ofstream field(fieldName.str());
            DNDS_check_throw_info(field.good(), "Cannot write Taylor-Green nodal field");
            field << std::setprecision(17);
            field << "x,y,z,rho,u,v,w,p,dual_volume\n";
            for (Index i = 0; i < mesh.NumNode(); i++)
            {
                const auto &volume = geometry.NodeVolume(i);
                const real measure = volume.moments.measure;
                const State numerical = state[i];
                const State point = spatial.PointValues()[i];
                const State primitive = wave.Primitive(point);
                if (!numerical.allFinite() || !point.allFinite() ||
                    primitive(0) <= 0 || primitive(4) <= 0)
                    localNonphysical++;
                localMinimum[0] = std::min(localMinimum[0], primitive(0));
                localMinimum[1] = std::min(localMinimum[1], primitive(4));
                localFinal += measure * numerical;
                localKinetic += 0.5 * measure * primitive(0) *
                    primitive.segment<3>(1).squaredNorm();
                localVolume += measure;
                field << mesh.coords[i].x() << ',' << mesh.coords[i].y() << ','
                      << mesh.coords[i].z();
                for (int component = 0; component < 5; component++)
                    field << ',' << primitive(component);
                field << ',' << measure << '\n';
            }
            DNDS_check_throw_info(field.good(), "Taylor-Green nodal field write failed");
            MPI_Allreduce(localFinal.data(), finalConserved.data(), 5,
                          DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(&localKinetic, &kinetic, 1,
                          DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(&localVolume, &totalVolume, 1,
                          DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            real minimum[2]{};
            Index nonphysical = 0;
            MPI_Allreduce(localMinimum, minimum, 2,
                          DNDS::DNDS_MPI_REAL, MPI_MIN, mpi.comm);
            MPI_Allreduce(&localNonphysical, &nonphysical, 1,
                          DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            DNDS_check_throw_info(nonphysical == 0,
                                  "Taylor-Green vortex produced nonphysical state");
            const Index localFallbacks = spatial.ReferenceMeanFallbackCount();
            Index globalFallbacks = 0;
            MPI_Allreduce(&localFallbacks, &globalFallbacks, 1,
                          DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            if (mpi.rank == 0)
            {
                nlohmann::ordered_json result = {
                    {"problem", "compressible_taylor_green"},
                    {"mesh", cfg.mesh.meshFile}, {"mach", mach},
                    {"nodes", solver.Mesh()->NumNodeGlobal()}, {"mpi_ranks", mpi.size},
                    {"stencil", stencil}, {"mode", mode},
                    {"method", method},
                    {"variational_weight", cfg.reconstruction.variationalWeight},
                    {"variational_iterations", cfg.reconstruction.variationalIterations},
                    {"traditional_edge_midpoint_flux", midpointFlux},
                    {"reference_roe_harten", referenceRoe},
                    {"reference_document_flux", referenceDocument},
                    {"mean_fallback_count", globalFallbacks},
                    {"final_rhs_riemann_calls", globalCalls},
                    {"distance_weight_power", distanceWeightPower},
                    {"dt", dt}, {"steps", steps},
                    {"final_time", endTime}, {"march_seconds", marchSeconds},
                    {"periodic_mismatch", periodicMismatch},
                    {"initial_conserved", std::vector<real>(initialConserved.data(), initialConserved.data() + 5)},
                    {"final_conserved", std::vector<real>(finalConserved.data(), finalConserved.data() + 5)},
                    {"kinetic_energy_point_integral", kinetic},
                    {"total_dual_volume", totalVolume},
                    {"minimum_point_density", minimum[0]},
                    {"minimum_point_pressure", minimum[1]},
                    {"nonphysical_point_count", nonphysical},
                    {"reference_quadrature_order", 6},
                    {"reference_points_per_micro_tet", reference.GetNumPoints()},
                    {"point_field_pattern", output.filename().string() + ".rank%04d.csv"},
                };
                std::ofstream file(output);
                DNDS_check_throw_info(file.good(), "Cannot write Taylor-Green output");
                file << std::setw(2) << result << '\n';
                DNDS_check_throw_info(file.good(), "Taylor-Green output write failed");
                std::cout << result.dump(2) << std::endl;
            }
        }
        catch (const std::exception &error)
        {
            std::cerr << "NCFV Taylor-Green vortex: " << error.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
}
