/** Native efficient/traditional mean-flux and Roe-dissipation crossover test.
 * No production flux formula is reimplemented here. Zeroing the native
 * numerical-minus-central correction extracts C; native full minus C gives D.
 * Divergence is linear, so C_a + D_b is assembled from native owned-node RHSs.
 */
#include "NCFV/NCFVSolver.hpp"
#include "NCFV/NCFVAnalytic.hpp"
#include "Geom/Quadrature.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sys/resource.h>

using namespace DNDS::NCFV;
using State = Eigen::Vector<DNDS::real, 5>;
using Matrix = Eigen::Matrix<DNDS::real, Eigen::Dynamic, 5, Eigen::RowMajor>;

namespace
{
    double Reduce(const DNDS::MPIInfo &mpi, double value, MPI_Op operation)
    {
        double result = 0;
        MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, operation, mpi.comm);
        return result;
    }

    State Primitive(const State &u, double gamma)
    {
        State w;
        w(0) = u(0);
        w.segment<3>(1) = u.segment<3>(1) / u(0);
        w(4) = (gamma - 1) * (u(4) - 0.5 * u.segment<3>(1).squaredNorm() / u(0));
        return w;
    }

    struct NativeOperator
    {
        const DNDS::MPIInfo &mpi;
        const DualGeometry &geometry;
        const NodeHalo &halo;
        std::unique_ptr<SpatialOperator<3>> spatial;
        NodeStatePair input, output;
        DNDS::index owned;

        NativeOperator(const DNDS::MPIInfo &mpiIn, Solver<3> &solver,
                       const DualGeometry &geometryIn, const NodeHalo &haloIn,
                       const Reconstruction &reconstruction, const Configuration &cfg)
            : mpi(mpiIn), geometry(geometryIn), halo(haloIn), owned(solver.Mesh()->NumNode())
        {
            for (auto *field : {&input, &output})
            {
                field->InitPair("hybridNative", mpi);
                field->father->Resize(owned, 5, 1);
                field->son->Resize(halo.NumNodeGhost(), 5, 1);
                field->BorrowSetup(halo.Layout());
                field->trans.initPersistentPull();
            }
            spatial = std::make_unique<SpatialOperator<3>>(
                mpi, solver.Mesh(), solver.EdgeTopology(), geometry,
                reconstruction, solver.Boundaries(), cfg.algorithm.mode,
                cfg.reconstruction, cfg.physics, cfg.time, halo);
            spatial->Initialize();
        }

        Matrix Evaluate(const Matrix &state, bool full)
        {
            for (DNDS::index i = 0; i < owned; i++)
                input[i] = state.row(i).transpose();
            spatial->SetInteriorInviscidCorrectionScalesForDiagnostics(
                full ? State(State::Ones()) : State(State::Zero()));
            spatial->EvaluateRHS(input, output, false, false);
            Matrix result(owned, 5);
            for (DNDS::index i = 0; i < owned; i++)
                result.row(i) = output[i].transpose();
            return result;
        }

        Matrix PointValues() const
        {
            Matrix result(owned, 5);
            for (DNDS::index i = 0; i < owned; i++)
                result.row(i) = spatial->PointValues()[i].transpose();
            return result;
        }
    };

    nlohmann::ordered_json Norms(const DNDS::MPIInfo &mpi, const DualGeometry &geometry,
                                 const Matrix &numerical, const Matrix &exact)
    {
        State localL1 = State::Zero(), localL2 = State::Zero(), localMax = State::Zero();
        double localVolume = 0;
        for (DNDS::index i = 0; i < numerical.rows(); i++)
        {
            const double volume = geometry.NodeVolume(i).moments.measure;
            const State error = (numerical.row(i) - exact.row(i)).transpose();
            localL1 += volume * error.cwiseAbs();
            localL2 += volume * error.cwiseAbs2();
            localMax = localMax.cwiseMax(error.cwiseAbs());
            localVolume += volume;
        }
        State l1, l2, linf;
        MPI_Allreduce(localL1.data(), l1.data(), 5, MPI_DOUBLE, MPI_SUM, mpi.comm);
        MPI_Allreduce(localL2.data(), l2.data(), 5, MPI_DOUBLE, MPI_SUM, mpi.comm);
        MPI_Allreduce(localMax.data(), linf.data(), 5, MPI_DOUBLE, MPI_MAX, mpi.comm);
        const double volume = Reduce(mpi, localVolume, MPI_SUM);
        nlohmann::ordered_json result = {{"volume", volume}};
        for (int v = 0; v < 5; v++)
        {
            result["L1V"].push_back(l1(v) / volume);
            result["L2V"].push_back(std::sqrt(l2(v) / volume));
            result["Linf"].push_back(linf(v));
        }
        return result;
    }

    Matrix ToPrimitive(const Matrix &state, double gamma)
    {
        Matrix result(state.rows(), 5);
        for (DNDS::index i = 0; i < state.rows(); i++)
            result.row(i) = Primitive(state.row(i).transpose(), gamma).transpose();
        return result;
    }

    Matrix ExactMeans(const Solver<3> &solver, const Configuration &cfg, double time)
    {
        DNDS::Geom::Elem::Quadrature quadrature({DNDS::Geom::Elem::Tet4}, 6);
        DNDS_check_throw_info(quadrature.GetNumPoints() == 24, "Unexpected reference rule");
        Matrix result(solver.Mesh()->NumNode(), 5);
        for (DNDS::index i = 0; i < result.rows(); i++)
        {
            const auto &volume = solver.Geometry().NodeVolume(i);
            DNDS_check_throw_info(!volume.microVolumes.empty(), "Missing reference micro geometry");
            State mean = State::Zero();
            for (const auto &micro : volume.microVolumes)
                for (int q = 0; q < quadrature.GetNumPoints(); q++)
                {
                    const auto [p, weight] = quadrature.GetQuadraturePointInfo(q);
                    const auto x = (micro.points[0] +
                        p[0] * (micro.points[1] - micro.points[0]) +
                        p[1] * (micro.points[2] - micro.points[0]) +
                        p[2] * (micro.points[3] - micro.points[0])).eval();
                    mean += 6 * micro.measure * weight / volume.moments.measure *
                            IsentropicVortex<3>(cfg, x, time).first;
                }
            result.row(i) = mean.transpose();
        }
        return result;
    }

    double Mass(const DNDS::MPIInfo &mpi, const DualGeometry &geometry, const Matrix &state)
    {
        double local = 0;
        for (DNDS::index i = 0; i < state.rows(); i++)
            local += geometry.NodeVolume(i).moments.measure * state(i, 0);
        return Reduce(mpi, local, MPI_SUM);
    }

    void CheckPhysical(const DNDS::MPIInfo &mpi, const Matrix &state, double gamma)
    {
        double localBad = 0;
        for (DNDS::index i = 0; i < state.rows(); i++)
        {
            const State u = state.row(i).transpose();
            localBad += !u.allFinite() || u(0) <= 1e-12 || Primitive(u, gamma)(4) <= 1e-12;
        }
        DNDS_check_throw_info(Reduce(mpi, localBad, MPI_SUM) == 0, "Nonphysical RK stage");
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
            DNDS_check_throw_info(argc == 6 || argc == 7,
                "usage: base-config mesh output.json dt final-time [verify-production]");
            const std::filesystem::path output = std::filesystem::absolute(argv[3]);
            DNDS_check_throw_info(!std::filesystem::exists(output), "Output already exists");
            std::ifstream base(argv[1]);
            DNDS_check_throw_info(base.good(), "Missing configuration");
            nlohmann::ordered_json resolved = Configuration{};
            resolved.merge_patch(nlohmann::ordered_json::parse(base));
            for (auto &boundary : resolved["physics"]["boundaryZones"])
            {
                nlohmann::ordered_json normalized = BoundaryZoneSettings{};
                normalized.merge_patch(boundary);
                boundary = std::move(normalized);
            }
            Configuration cfg = resolved.get<Configuration>();
            cfg.mesh.meshFile = argv[2];
            cfg.algorithm.mode = IntegrationMode::TraditionalQuadrature;
            cfg.algorithm.quadratureOrder = 4;
            cfg.algorithm.surfaceQuadratureOrder = 3;
            cfg.algorithm.retainMicroGeometry = true;
            cfg.reconstruction.method = ReconstructionMethod::LeastSquares;
            cfg.reconstruction.stencilSizeFactor = std::nextafter(13.0 / 9.0, 0.0);
            cfg.reconstruction.enableLimiter = false;
            cfg.physics.riemannSolver = DNDS::Euler::Gas::Roe;
            cfg.physics.viscous.enabled = false;
            const double dt = std::stod(argv[4]), endTime = std::stod(argv[5]);
            const auto steps = static_cast<DNDS::index>(std::llround(endTime / dt));
            DNDS_check_throw_info(dt > 0 && steps > 0 && std::abs(steps * dt - endTime) < 1e-12,
                                  "Invalid fixed dt/end time");
            cfg.time.timeStep = dt;
            cfg.time.endTime = endTime;
            cfg.time.iterations = steps;
            cfg.time.useCFLTimeStep = false;
            cfg.time.useLocalTimeStep = false;
            cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
            cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
            cfg.io.outputInterval = cfg.io.restartInterval = 0;
            const auto runtime = output.parent_path() / output.stem();
            if (mpi.rank == 0)
                std::filesystem::create_directories(runtime);
            MPI_Barrier(mpi.comm);
            cfg.io.outputPrefix = (runtime / "production").string();
            cfg.Validate();
            const double initializeStart = MPI_Wtime();
            Solver<3> solver(mpi, cfg);
            solver.Initialize();
            Configuration efficientCfg = cfg;
            efficientCfg.algorithm.mode = IntegrationMode::EfficientDifferential;
            efficientCfg.algorithm.retainMicroGeometry = false;
            DualGeometry efficientGeometry(mpi, solver.Mesh(), solver.EdgeTopology(),
                                          efficientCfg.algorithm, false, &cfg.mesh);
            efficientGeometry.Build();
            NodeHalo efficientHalo(mpi, solver.Mesh(), cfg.mesh);
            efficientHalo.BuildPreliminary(efficientGeometry,
                cfg.reconstruction.maximumStencilRings, efficientGeometry.CollectNodeDependencies());
            Reconstruction efficientReconstruction(mpi, solver.Mesh(), solver.EdgeTopology(),
                efficientGeometry, efficientCfg.algorithm.mode, cfg.reconstruction, efficientHalo);
            efficientReconstruction.Build();
            auto dependencies = efficientGeometry.CollectNodeDependencies();
            const auto reconstructionDependencies = efficientReconstruction.CollectNodeDependencies();
            dependencies.insert(dependencies.end(), reconstructionDependencies.begin(), reconstructionDependencies.end());
            efficientHalo.Finalize(efficientGeometry, std::move(dependencies));
            efficientGeometry.RemapNodeIndices([&](DNDS::index i) { return efficientHalo.MeshLocalToLocal(i); });
            efficientReconstruction.RemapNodeIndices();

            NativeOperator traditional(mpi, solver, solver.Geometry(), solver.NodeCommunication(),
                                       solver.ReconstructionData(), cfg);
            NativeOperator efficient(mpi, solver, efficientGeometry, efficientHalo,
                                     efficientReconstruction, efficientCfg);
            const DNDS::index count = solver.Mesh()->NumNode();
            Matrix initial(count, 5), efficientInitial(count, 5);
            double localVolumeDifference = 0;
            for (DNDS::index i = 0; i < count; i++)
            {
                DNDS_check_throw_info(solver.NodeCommunication().LocalToGlobal(i) == efficientHalo.LocalToGlobal(i),
                                      "Native operator owned-node ordering differs");
                const auto &opT = solver.ReconstructionData().Operator(i);
                const auto &opE = efficientReconstruction.Operator(i);
                DNDS_check_throw_info(opT.stencilGlobals == opE.stencilGlobals && opT.stencil.size() == 13,
                                      "Native LS13 stencil selection differs");
                initial.row(i) = solver.StateField()[i].transpose();
                State value = IsentropicVortex<3>(cfg, solver.Mesh()->coords[i], 0).first;
                for (const auto &entry : efficientGeometry.NodeVolume(i).pointRecoveryStencil)
                    value += IsentropicVortex<3>(cfg, efficientHalo.Coordinate(entry.node), 0).second.transpose() *
                             entry.gradientWeight.head<3>();
                efficientInitial.row(i) = value.transpose();
                localVolumeDifference = std::max(localVolumeDifference,
                    std::abs(efficientGeometry.NodeVolume(i).moments.measure - solver.Geometry().NodeVolume(i).moments.measure));
                DNDS_check_throw_info(solver.Geometry().NodeVolume(i).boundaryPieces.empty() &&
                                      efficientGeometry.NodeVolume(i).boundaryPieces.empty(),
                                      "Crossover diagnostic requires fully stitched periodic volumes");
            }
            const double volumeDifference = Reduce(mpi, localVolumeDifference, MPI_MAX);
            DNDS_check_throw_info(volumeDifference < 1e-12, "Native dual volumes differ");

            // Full = C + D identity, checked independently on pulled edge fields.
            nlohmann::ordered_json checks = {{"max_dual_volume_difference", volumeDifference},
                                            {"identical_owned_order_and_LS13_stencils", true}};
            for (auto *native : {&traditional, &efficient})
            {
                const Matrix full = native->Evaluate(initial, true);
                std::vector<State> edgeFull;
                for (DNDS::index e = 0; e < solver.EdgeTopology().NumEdgeProc(); e++)
                    edgeFull.emplace_back(native->spatial->EdgeFluxes()[e]);
                const Matrix central = native->Evaluate(initial, false);
                Matrix independentDivergence(count, 5);
                for (DNDS::index i = 0; i < count; i++)
                {
                    State sum = State::Zero();
                    for (const auto &incidence : solver.EdgeTopology().Node2Edge(i))
                        sum += incidence.outwardSign *
                               (edgeFull[incidence.edge] - State(native->spatial->EdgeFluxes()[incidence.edge]));
                    independentDivergence.row(i) = (-sum / native->geometry.NodeVolume(i).moments.measure).transpose();
                }
                const double error = Reduce(mpi, (full - central - independentDivergence).cwiseAbs().maxCoeff(), MPI_MAX);
                DNDS_check_throw_info(error < 1e-10, "Flux part divergence identity failed");
                const char *name = native == &traditional ? "traditional" : "efficient";
                checks[std::string(name) + "_part_divergence_identity_max"] = error;
                Matrix zeros = Matrix::Zero(count, 5);
                checks[std::string(name) + "_initial_central_rhs"] = Norms(mpi, solver.Geometry(), central, zeros);
                checks[std::string(name) + "_initial_dissipation_rhs"] = Norms(mpi, solver.Geometry(), full - central, zeros);
            }
            const double initializationSeconds = Reduce(mpi, MPI_Wtime() - initializeStart, MPI_MAX);
            const Matrix exactFinal = ExactMeans(solver, cfg, endTime);
            const Matrix exactInitial = ExactMeans(solver, cfg, 0);
            Matrix exactPoint(count, 5);
            for (DNDS::index i = 0; i < count; i++)
                exactPoint.row(i) = IsentropicVortex<3>(cfg, solver.Mesh()->coords[i], endTime).first.transpose();
            nlohmann::ordered_json result = {
                {"configuration", cfg}, {"efficient_configuration", efficientCfg},
                {"mesh", cfg.mesh.meshFile}, {"nodes", solver.Mesh()->NumNodeGlobal()},
                {"cells", solver.Mesh()->NumCellGlobal()}, {"mpi_ranks", mpi.size},
                {"dt", dt}, {"steps", steps}, {"final_time", endTime}, {"time_integrator", "SSPRK3"},
                {"stencil_target", 13}, {"reference_quadrature_order", 6}, {"reference_points_per_micro_tet", 24},
                {"initialization_seconds", initializationSeconds}, {"checks", checks},
                {"traditional_initial_error", Norms(mpi, solver.Geometry(), initial, exactInitial)},
                {"efficient_initial_error", Norms(mpi, solver.Geometry(), efficientInitial, exactInitial)},
                {"initialization_difference", Norms(mpi, solver.Geometry(), efficientInitial, initial)},
                {"conserved_components", {"rho", "rho_u", "rho_v", "rho_w", "rho_E"}},
                {"primitive_components", {"rho", "u", "v", "w", "p"}},
                {"cases", nlohmann::ordered_json::object()}};

            const std::vector<std::string> cases{"TT", "EE", "ET", "TE", "EE_native_initialization"};
            for (const auto &name : cases)
            {
                Matrix state = name == "EE_native_initialization" ? efficientInitial : initial;
                const double initialMass = Mass(mpi, solver.Geometry(), state);
                const auto rhs = [&](const Matrix &u) -> Matrix
                {
                    if (name == "TT")
                        return traditional.Evaluate(u, true);
                    if (name == "EE" || name == "EE_native_initialization")
                        return efficient.Evaluate(u, true);
                    if (name == "ET")
                    {
                        Matrix centralE = efficient.Evaluate(u, false);
                        Matrix fullT = traditional.Evaluate(u, true);
                        Matrix centralT = traditional.Evaluate(u, false);
                        return centralE + fullT - centralT;
                    }
                    Matrix centralT = traditional.Evaluate(u, false);
                    Matrix fullE = efficient.Evaluate(u, true);
                    Matrix centralE = efficient.Evaluate(u, false);
                    return centralT + fullE - centralE;
                };
                const double start = MPI_Wtime();
                for (DNDS::index step = 0; step < steps; step++)
                {
                    const Matrix baseState = state;
                    Matrix stage = baseState + dt * rhs(baseState);
                    CheckPhysical(mpi, stage, cfg.physics.gamma);
                    stage = 0.75 * baseState + 0.25 * (stage + dt * rhs(stage));
                    CheckPhysical(mpi, stage, cfg.physics.gamma);
                    state = (1.0 / 3.0) * baseState + (2.0 / 3.0) * (stage + dt * rhs(stage));
                    CheckPhysical(mpi, state, cfg.physics.gamma);
                    if (mpi.rank == 0 && (step == 0 || (step + 1) % 8 == 0 || step + 1 == steps))
                        std::cout << "hybrid " << name << " step=" << step + 1 << "/" << steps
                                  << " elapsed=" << MPI_Wtime() - start << std::endl;
                }
                nlohmann::ordered_json errors = {
                    {"initialization", name == "EE_native_initialization" ? "native_efficient_differential" : "common_native_traditional_q4"},
                    {"average_flux", name.front() == 'E' ? "Efficient" : "Traditional"},
                    {"dissipation_flux", name[1] == 'E' ? "Efficient" : "Traditional"},
                    {"march_seconds", Reduce(mpi, MPI_Wtime() - start, MPI_MAX)},
                    {"mean_conserved", Norms(mpi, solver.Geometry(), state, exactFinal)},
                    {"primitive_of_mean", Norms(mpi, solver.Geometry(), ToPrimitive(state, cfg.physics.gamma),
                                                ToPrimitive(exactFinal, cfg.physics.gamma))},
                    {"mass_initial", initialMass}, {"mass_final", Mass(mpi, solver.Geometry(), state)}};
                errors["mass_drift"] = errors["mass_final"].get<double>() - initialMass;
                for (auto *native : {&traditional, &efficient})
                {
                    native->Evaluate(state, true);
                    const Matrix point = native->PointValues();
                    const char *prefix = native == &traditional ? "point_traditional" : "point_efficient";
                    errors[std::string(prefix) + "_conserved"] = Norms(mpi, solver.Geometry(), point, exactPoint);
                    errors[std::string(prefix) + "_primitive"] = Norms(mpi, solver.Geometry(), ToPrimitive(point, cfg.physics.gamma),
                                                                   ToPrimitive(exactPoint, cfg.physics.gamma));
                }
                if (name == "TT" && argc == 7)
                {
                    solver.Run();
                    double local = 0;
                    for (DNDS::index i = 0; i < count; i++)
                        local = std::max(local, (state.row(i).transpose() - State(solver.StateField()[i])).cwiseAbs().maxCoeff());
                    const double difference = Reduce(mpi, local, MPI_MAX);
                    DNDS_check_throw_info(difference < 1e-11, "Probe differs from production SSPRK3");
                    errors["production_final_state_max_difference"] = difference;
                }
                result["cases"][name] = errors;
                if (mpi.rank == 0)
                {
                    std::ofstream out(output);
                    DNDS_check_throw_info(out.good(), "Cannot write crossover result");
                    out << std::setw(2) << result << '\n';
                    std::cout << "completed " << name << " rho_mean_L2V="
                              << errors["mean_conserved"]["L2V"][0] << std::endl;
                }
            }
            struct rusage usage{};
            DNDS_check_throw_info(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
            result["peak_rss_sum_mib"] = Reduce(mpi, usage.ru_maxrss / 1024.0, MPI_SUM);
            result["complete"] = true;
            if (mpi.rank == 0)
            {
                std::ofstream out(output);
                out << std::setw(2) << result << '\n';
            }
        }
        catch (const std::exception &error)
        {
            std::cerr << "Hybrid flux diagnostic: " << error.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
}
