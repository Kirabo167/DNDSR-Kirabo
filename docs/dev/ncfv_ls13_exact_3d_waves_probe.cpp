/** Exact smooth 3-D periodic Euler wave accuracy for efficient LS NCFV. */
#include "NCFV/NCFVSolver.hpp"
#include "Geom/Quadrature.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace DNDS::NCFV;
using DNDS::real;
using Index = DNDS::index;
using State = Eigen::Matrix<real, 5, 1>;
using Vector = Eigen::Matrix<real, 3, 1>;

namespace
{
    struct ExactWave
    {
        Vector k, background, polarization;
        real gamma = 1.4, amplitude = 0.2;
        bool entropy = true;

        real Phase(const Vector &x, real time) const
        {
            return k.dot(x) - k.dot(background) * time;
        }

        State Conservative(const Vector &x, real time) const
        {
            const real sine = std::sin(Phase(x, time));
            const real rho = entropy ? 1 + amplitude * sine : 1;
            const Vector velocity = entropy ? background :
                background + amplitude * polarization * sine;
            State state;
            state << rho, rho * velocity,
                1 / (gamma - 1) + 0.5 * rho * velocity.squaredNorm();
            return state;
        }

        Eigen::Matrix<real, 3, 5> Gradient(const Vector &x, real time) const
        {
            const real sine = std::sin(Phase(x, time));
            const real cosine = std::cos(Phase(x, time));
            Eigen::Matrix<real, 3, 5> result =
                Eigen::Matrix<real, 3, 5>::Zero();
            const Vector velocity = entropy ? background :
                background + amplitude * polarization * sine;
            for (int d = 0; d < 3; d++)
            {
                const real factor = amplitude * k(d) * cosine;
                if (entropy)
                {
                    result(d, 0) = factor;
                    result.block<1, 3>(d, 1) =
                        factor * background.transpose();
                    result(d, 4) = 0.5 * factor * background.squaredNorm();
                }
                else
                {
                    const Vector velocityGradient = factor * polarization;
                    result.block<1, 3>(d, 1) = velocityGradient.transpose();
                    result(d, 4) = velocity.dot(velocityGradient);
                }
            }
            return result;
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

    struct Norm
    {
        State absolute = State::Zero();
        State squared = State::Zero();
        State maximum = State::Zero();
        real volume = 0;

        void Add(const State &error, real measure)
        {
            absolute += measure * error.cwiseAbs();
            squared += measure * error.cwiseAbs2();
            maximum = maximum.cwiseMax(error.cwiseAbs());
            volume += measure;
        }

        nlohmann::ordered_json Reduce(const DNDS::MPIInfo &mpi) const
        {
            State globalAbsolute, globalSquared, globalMaximum;
            real globalVolume = 0;
            MPI_Allreduce(absolute.data(), globalAbsolute.data(), 5,
                          DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(squared.data(), globalSquared.data(), 5,
                          DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(maximum.data(), globalMaximum.data(), 5,
                          DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            MPI_Allreduce(&volume, &globalVolume, 1,
                          DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            DNDS_check_throw_info(globalVolume > 0,
                                  "Exact 3-D wave has no dual-cell volume");
            globalAbsolute /= globalVolume;
            globalSquared = (globalSquared / globalVolume).cwiseSqrt();
            return {
                {"volume", globalVolume},
                {"L1", std::vector<real>(globalAbsolute.data(), globalAbsolute.data() + 5)},
                {"L2", std::vector<real>(globalSquared.data(), globalSquared.data() + 5)},
                {"Linf", std::vector<real>(globalMaximum.data(), globalMaximum.data() + 5)},
            };
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
            DNDS_check_throw_info(argc == 7,
                "usage: config.json output.json entropy|vortical dt final-time stencil-size");
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
            const std::string kind(argv[3]);
            DNDS_check_throw_info(kind == "entropy" || kind == "vortical",
                                  "Wave type must be entropy or vortical");
            const int stencil = std::stoi(argv[6]);
            DNDS_check_throw_info(stencil >= 9, "Need at least nine stencil nodes");
            cfg.reconstruction.method = ReconstructionMethod::LeastSquares;
            cfg.reconstruction.stencilSizeFactor =
                std::nextafter(static_cast<real>(stencil) / 9.0, 0.0);
            cfg.reconstruction.enableLimiter = false;
            cfg.algorithm.mode = IntegrationMode::EfficientDifferential;
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
            const Vector background{
                cfg.physics.initialPrimitive[1],
                cfg.physics.initialPrimitive[2],
                cfg.physics.initialPrimitive[3],
            };
            Vector polarization{
                k.x() * k.z(), k.y() * k.z(),
                -(k.x() * k.x() + k.y() * k.y()),
            };
            polarization.normalize();
            const ExactWave wave{k, background, polarization,
                                 cfg.physics.gamma, 0.2, kind == "entropy"};
            real periodicMismatch = 0;
            for (int d = 0; d < 3; d++)
            {
                const Vector x{0.371, 0.619, 0.913};
                Vector shifted = x;
                shifted(d) += cfg.mesh.periodicLengths[d];
                for (real t : {real(0), endTime})
                    periodicMismatch = std::max(periodicMismatch,
                        (wave.Conservative(x, t) - wave.Conservative(shifted, t))
                            .cwiseAbs().maxCoeff());
            }
            DNDS_check_throw_info(periodicMismatch < 1e-13 &&
                                  std::abs(k.dot(polarization)) < 1e-13,
                                  "Wave periodicity or transversality failed");

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
            NodeStatePair state, base, stage, rhs;
            Allocate(state, "exactWave.state", mpi, mesh.NumNode(), halo);
            Allocate(base, "exactWave.base", mpi, mesh.NumNode(), halo);
            Allocate(stage, "exactWave.stage", mpi, mesh.NumNode(), halo);
            Allocate(rhs, "exactWave.rhs", mpi, mesh.NumNode(), halo);
            for (Index i = 0; i < mesh.NumNode(); i++)
            {
                const auto &volume = geometry.NodeVolume(i);
                state[i] = wave.Conservative(mesh.coords[i], 0);
                for (const auto &entry : volume.pointRecoveryStencil)
                    state[i] += wave.Gradient(halo.Coordinate(entry.node), 0).transpose() *
                                entry.gradientWeight.head<3>();
            }
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
            spatial.EvaluateRHS(state, rhs);
            const double localMarchSeconds = MPI_Wtime() - start;
            double marchSeconds = 0;
            MPI_Allreduce(&localMarchSeconds, &marchSeconds, 1,
                          MPI_DOUBLE, MPI_MAX, mpi.comm);

            DNDS::Geom::Elem::Quadrature reference(
                {DNDS::Geom::Elem::Tet4}, 6);
            DNDS_check_throw_info(reference.GetNumPoints() == 24,
                                  "Reference must use 24 points per micro tetrahedron");
            Norm meanConservative, pointConservative, pointPrimitive;
            real localMinimum[2]{DNDS::veryLargeReal, DNDS::veryLargeReal};
            Index localNonphysical = 0;
            State localFinal = State::Zero(), finalConserved = State::Zero();
            for (Index i = 0; i < mesh.NumNode(); i++)
            {
                const auto &volume = geometry.NodeVolume(i);
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
                                     wave.Conservative(x, endTime);
                    }
                const State numerical = state[i];
                const State point = spatial.PointValues()[i];
                const State primitive = wave.Primitive(point);
                if (!numerical.allFinite() || !point.allFinite() ||
                    primitive(0) <= 0 || primitive(4) <= 0)
                    localNonphysical++;
                localMinimum[0] = std::min(localMinimum[0], primitive(0));
                localMinimum[1] = std::min(localMinimum[1], primitive(4));
                meanConservative.Add(numerical - exactMean, measure);
                const State exactPoint = wave.Conservative(mesh.coords[i], endTime);
                pointConservative.Add(point - exactPoint, measure);
                pointPrimitive.Add(primitive - wave.Primitive(exactPoint), measure);
                localFinal += measure * numerical;
            }
            MPI_Allreduce(localFinal.data(), finalConserved.data(), 5,
                          DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            real minimum[2]{};
            Index nonphysical = 0;
            MPI_Allreduce(localMinimum, minimum, 2,
                          DNDS::DNDS_MPI_REAL, MPI_MIN, mpi.comm);
            MPI_Allreduce(&localNonphysical, &nonphysical, 1,
                          DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            const auto meanError = meanConservative.Reduce(mpi);
            const auto pointError = pointConservative.Reduce(mpi);
            const auto primitiveError = pointPrimitive.Reduce(mpi);
            DNDS_check_throw_info(nonphysical == 0,
                                  "Exact 3-D wave produced nonphysical state");
            if (mpi.rank == 0)
            {
                nlohmann::ordered_json result = {
                    {"problem", kind}, {"mesh", cfg.mesh.meshFile},
                    {"nodes", solver.Mesh()->NumNodeGlobal()}, {"mpi_ranks", mpi.size},
                    {"stencil", stencil}, {"dt", dt}, {"steps", steps},
                    {"final_time", endTime}, {"march_seconds", marchSeconds},
                    {"periodic_mismatch", periodicMismatch},
                    {"k_dot_polarization", k.dot(polarization)},
                    {"initial_conserved", std::vector<real>(initialConserved.data(), initialConserved.data() + 5)},
                    {"final_conserved", std::vector<real>(finalConserved.data(), finalConserved.data() + 5)},
                    {"minimum_point_density", minimum[0]},
                    {"minimum_point_pressure", minimum[1]},
                    {"nonphysical_point_count", nonphysical},
                    {"reference_quadrature_order", 6},
                    {"reference_points_per_micro_tet", reference.GetNumPoints()},
                    {"mean_conservative", meanError},
                    {"point_conservative", pointError},
                    {"point_primitive", primitiveError},
                };
                std::ofstream file(output);
                DNDS_check_throw_info(file.good(), "Cannot write exact-wave output");
                file << std::setw(2) << result << '\n';
                DNDS_check_throw_info(file.good(), "Exact-wave output write failed");
                std::cout << result.dump(2) << std::endl;
            }
        }
        catch (const std::exception &error)
        {
            std::cerr << "NCFV exact 3-D wave: " << error.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
}
