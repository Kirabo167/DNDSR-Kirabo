/** Quadratic conservative-field patch test; no residual or time integration. */
#include "NCFV/NCFVSolver.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <map>

namespace
{
    using namespace DNDS::NCFV;
    using DNDS::real;
    using Index = DNDS::index;
    template <typename T> using StateT = Eigen::Matrix<T, 5, 1>;
    template <typename T> using GradientT = Eigen::Matrix<T, 3, 5>;
    using State = StateT<real>;
    using Gradient = GradientT<real>;
    using LongVector = Eigen::Matrix<long double, 3, 1>;
    using LongMatrix = Eigen::Matrix<long double, 3, 3>;

    // Exact rational coefficients, divided by 1000. All five conservative
    // fields are quadratic; primitive velocity and pressure need not be.
    const std::array<std::array<int, 10>, 5> Coefficients{{
        {{2000, 100, -60, 40, 30, 20, -15, 25, 12, 20}},
        {{600, -30, 50, 20, 12, -10, 8, -6, 9, 10}},
        {{400, 40, 20, -30, -8, 12, 6, 10, -7, 8}},
        {{200, 20, -10, 40, 6, 8, -10, 12, 5, -4}},
        {{5000, 150, -100, 80, 60, -25, 20, 40, 15, 30}},
    }};
    const std::array<std::string, 5> Names{"rho", "rhou", "rhov", "rhow", "rhoE"};
    const std::array<std::string, 5> PrimitiveNames{"rho", "u", "v", "w", "p"};
    const std::array<std::string, 3> Axes{"x", "y", "z"};
    const std::array<std::string, 3> Regions{"all", "interior", "boundary"};

    template <typename T>
    std::pair<StateT<T>, GradientT<T>> Polynomial(const Vector3 &point)
    {
        const T x = (T(point.x()) - 5) / 5;
        const T y = (T(point.y()) - 5) / 5;
        const T z = (T(point.z()) - 2) / 2;
        const std::array<T, 10> basis{1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z};
        StateT<T> value = StateT<T>::Zero();
        GradientT<T> gradient;
        for (int v = 0; v < 5; v++)
        {
            const auto &c = Coefficients[v];
            for (int k = 0; k < 10; k++)
                value(v) += T(c[k]) / 1000 * basis[k];
            gradient(0, v) = (c[1] + 2 * c[4] * x + c[5] * y + c[6] * z) / T(5000);
            gradient(1, v) = (c[2] + c[5] * x + 2 * c[7] * y + c[8] * z) / T(5000);
            gradient(2, v) = (c[3] + c[6] * x + c[8] * y + 2 * c[9] * z) / T(2000);
        }
        return {value, gradient};
    }

    template <typename T>
    std::pair<StateT<T>, GradientT<T>> Primitive(const StateT<T> &u, const GradientT<T> &g, T gamma)
    {
        StateT<T> q;
        GradientT<T> dq;
        const Eigen::Matrix<T, 3, 1> velocity = u.template segment<3>(1) / u(0);
        q << u(0), velocity, (gamma - 1) * (u(4) - T(0.5) * u(0) * velocity.squaredNorm());
        dq.col(0) = g.col(0);
        for (int d = 0; d < 3; d++)
        {
            dq.row(d).template segment<3>(1) =
                (g.row(d).template segment<3>(1) - g(d, 0) * velocity.transpose()) / u(0);
            dq(d, 4) = (gamma - 1) *
                (g(d, 4) - velocity.dot(g.row(d).template segment<3>(1)) +
                 T(0.5) * velocity.squaredNorm() * g(d, 0));
        }
        return {q, dq};
    }

    // Independent barycentric integral: no production RawMoments, weights,
    // or quadrature used. Recompute each tetrahedron's determinant as well.
    std::pair<StateT<long double>, long double> ExactAverage(const NodeControlVolume &volume)
    {
        StateT<long double> integral = StateT<long double>::Zero();
        long double measure = 0;
        DNDS_check_throw_info(!volume.microVolumes.empty(), "Patch test needs retained micro geometry");
        for (const auto &tet : volume.microVolumes)
        {
            DNDS_check_throw_info(tet.nPoints == 4, "Expected a tetrahedral dual micro-volume");
            LongVector sum = LongVector::Zero();
            LongMatrix squares = LongMatrix::Zero(), jacobian;
            for (int a = 0; a < 4; a++)
            {
                LongVector scaled = tet.points[a].cast<long double>();
                scaled -= LongVector(5, 5, 2);
                scaled.array() /= LongVector(5, 5, 2).array();
                sum += scaled;
                squares += scaled * scaled.transpose();
                if (a > 0)
                    jacobian.col(a - 1) = tet.points[a].cast<long double>() - tet.points[0].cast<long double>();
            }
            const long double weight = std::abs(jacobian.determinant()) / 6;
            const LongVector first = sum / 4;
            const LongMatrix second = (sum * sum.transpose() + squares) / 20;
            const std::array<long double, 10> basis{
                1, first(0), first(1), first(2), second(0, 0), second(0, 1),
                second(0, 2), second(1, 1), second(1, 2), second(2, 2)};
            for (int v = 0; v < 5; v++)
                for (int k = 0; k < 10; k++)
                    integral(v) += weight * (static_cast<long double>(Coefficients[v][k]) / 1000) * basis[k];
            measure += weight;
        }
        DNDS_check_throw_info(measure > 0, "Nonpositive independent dual volume");
        return {integral / measure, measure};
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
            DNDS_check_throw_info(argc == 3, "Usage: quadratic_reconstruction_probe base-config.json fresh-output-directory");
            auto cfg = LoadConfiguration(argv[1], {}, {}).configuration;
            DNDS_check_throw_info(cfg.dimension == 3 && cfg.io.restartInput.empty(), "Use a fresh 3-D configuration");
            const bool efficient = cfg.algorithm.mode == IntegrationMode::EfficientDifferential;
            cfg.mesh.periodicLengths = {0, 0, 0};
            cfg.algorithm.retainMicroGeometry = true; // Oracle vertices only; efficient Gauss storage remains zero.
            cfg.reconstruction.enableLimiter = false;
            cfg.physics.boundaryMode = BoundaryMode::FarField;
            for (auto &zone : cfg.physics.boundaryZones)
            {
                zone.mode = BoundaryMode::FarField;
                zone.strongState = false;
            }
            cfg.initialField = InitialFieldSettings{};
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
                DNDS_check_throw_info(!std::filesystem::exists(output), "Refusing to overwrite patch-test output");
                std::filesystem::create_directories(output);
            }
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            Solver<3> solver(mpi, cfg);
            solver.Initialize();
            const auto mesh = solver.Mesh();
            const auto &geometry = solver.Geometry();
            const auto &reconstruction = solver.ReconstructionData();
            const NodeHalo &nodeHalo = solver.NodeCommunication();

            // A separate diagnostic state: no const_cast or production-state edit.
            NodeStatePair means, points;
            const auto allocateState = [&](NodeStatePair &field,
                                           const std::string &name)
            {
                field.InitPair(name, mpi);
                field.father->Resize(mesh->NumNode(), 5, 1);
                field.son->Resize(nodeHalo.NumNodeGhost(), 5, 1);
                field.BorrowSetup(nodeHalo.Layout());
                field.trans.initPersistentPull();
            };
            allocateState(means, "NCFV.quadratic.means");
            allocateState(points, "NCFV.quadratic.points");
            NodeMatrixPair gradients, coefficients;
            gradients.InitPair("NCFV.quadratic.gradients", mpi);
            gradients.father->Resize(mesh->NumNode(), 3, 5);
            gradients.son->Resize(nodeHalo.NumNodeGhost(), 3, 5);
            gradients.BorrowSetup(nodeHalo.Layout());
            gradients.trans.initPersistentPull();
            if (!efficient)
            {
                coefficients.InitPair("NCFV.quadratic.coefficients", mpi);
                coefficients.father->Resize(mesh->NumNode(), 9, 5);
                coefficients.son->Resize(nodeHalo.NumNodeGhost(), 9, 5);
                coefficients.BorrowSetup(nodeHalo.Layout());
                coefficients.trans.initPersistentPull();
            }

            // Mirror InitializeVortex's two integration branches, replacing
            // only the analytic field. Exact gradients are used ONLY here.
            for (Index i = 0; i < mesh->NumNode(); i++)
            {
                const auto &volume = geometry.NodeVolume(i);
                if (efficient)
                {
                    means[i] = Polynomial<real>(mesh->coords[i]).first;
                    for (const auto &entry : volume.pointRecoveryStencil)
                        means[i] += Polynomial<real>(nodeHalo.Coordinate(entry.node)).second.transpose() *
                                    entry.gradientWeight;
                }
                else
                {
                    means[i].setZero();
                    for (const auto &q : volume.volumeQuadrature)
                        means[i] += q.weight / volume.moments.measure * Polynomial<real>(q.coordinate).first;
                }
            }
            means.trans.startPersistentPull();
            means.trans.waitPersistentPull();
            reconstruction.ComputeCoefficients(means, gradients, coefficients);
            if (efficient)
            {
                gradients.trans.startPersistentPull();
                gradients.trans.waitPersistentPull();
            }
            else
            {
                coefficients.trans.startPersistentPull();
                coefficients.trans.waitPersistentPull();
                for (Index i = 0; i < mesh->NumNode(); i++)
                    gradients[i] = Reconstruction::EvaluateBasisGradient(
                                       Vector3::Zero(), reconstruction.ReferenceLengths(i), 3) * coefficients[i];
            }
            reconstruction.RecoverPointValues(means, gradients, coefficients, points);
            DNDS_check_throw_info(solver.CurrentIteration() == 0 && solver.SimulationTime() == 0,
                                  "Patch test unexpectedly advanced time");

            std::ofstream nodes(output / fmt::format("nodes.rank{:04d}.csv", mpi.rank));
            DNDS_check_throw_info(nodes.good(), "Cannot open patch-test CSV");
            nodes << "original_node,x,y,z,volume,boundary";
            for (const auto &prefix : {"mean_", "exact_mean_", "point_"})
                for (const auto &name : Names)
                    nodes << ',' << prefix << name;
            for (const auto &name : Names)
                for (const auto &axis : Axes)
                    nodes << ",d_" << name << "_d" << axis;
            nodes << '\n' << std::setprecision(21);

            std::map<std::string, std::array<real, 3>> errors;
            // Predeclare every reduction key, even on ranks without boundary nodes.
            for (const auto &region : Regions)
                for (int v = 0; v < 5; v++)
                {
                    errors[region + ".conservative." + Names[v] + ".mean"] = {};
                    for (const auto &quantity : {"value", "gradient", "dx", "dy", "dz"})
                    {
                        errors[region + ".conservative." + Names[v] + "." + quantity] = {};
                        errors[region + ".primitive." + PrimitiveNames[v] + "." + quantity] = {};
                    }
                }
            real localVolumes[3]{}, localMaxima[5]{};
            Index localCounts[5]{}; // interior, boundary, stored quadrature, micro-volumes, rank minimum below.
            int localMinimumRank = 9;
            real localMinimumPrimitive[2]{DNDS::veryLargeReal, DNDS::veryLargeReal};
            for (Index i = 0; i < mesh->NumNode(); i++)
            {
                const auto &volume = geometry.NodeVolume(i);
                const bool boundary = !volume.boundaryPieces.empty();
                const int regionIndex = boundary ? 2 : 1;
                const real weight = volume.moments.measure;
                localVolumes[0] += weight;
                localVolumes[regionIndex] += weight;
                localCounts[boundary ? 1 : 0]++;
                localCounts[2] += volume.volumeQuadrature.size();
                localCounts[3] += volume.microVolumes.size();
                for (const auto &piece : volume.boundaryPieces)
                    localCounts[2] += piece.quadrature.size();
                const auto [exactMean, exactVolume] = ExactAverage(volume);
                const auto [exact, exactGradient] = Polynomial<long double>(mesh->coords[i]);
                const State value = points[i];
                const Gradient derivative = gradients[i];
                const auto [primitive, primitiveGradient] = Primitive(value, derivative, cfg.physics.gamma);
                const auto [exactPrimitive, exactPrimitiveGradient] = Primitive(
                    exact, exactGradient, static_cast<long double>(cfg.physics.gamma));
                DNDS_check_throw_info(value.allFinite() && derivative.allFinite() &&
                                          primitive(0) > 0 && primitive(4) > 0,
                                      "Nonfinite or nonphysical reconstructed polynomial state");
                const auto accumulate = [&](const std::string &name, long double difference)
                {
                    const real error = static_cast<real>(std::abs(difference));
                    for (int r : {0, regionIndex})
                    {
                        auto &e = errors.at(Regions[r] + "." + name);
                        e[0] += weight * error;
                        e[1] += weight * error * error;
                        e[2] = std::max(e[2], error);
                    }
                };
                for (int v = 0; v < 5; v++)
                {
                    const auto c = "conservative." + Names[v];
                    const auto p = "primitive." + PrimitiveNames[v];
                    accumulate(c + ".mean", static_cast<long double>(means[i](v)) - exactMean(v));
                    accumulate(c + ".value", static_cast<long double>(value(v)) - exact(v));
                    accumulate(p + ".value", static_cast<long double>(primitive(v)) - exactPrimitive(v));
                    accumulate(c + ".gradient", (derivative.col(v).cast<long double>() - exactGradient.col(v)).norm());
                    accumulate(p + ".gradient", (primitiveGradient.col(v).cast<long double>() - exactPrimitiveGradient.col(v)).norm());
                    for (int d = 0; d < 3; d++)
                    {
                        accumulate(c + ".d" + Axes[d], static_cast<long double>(derivative(d, v)) - exactGradient(d, v));
                        accumulate(p + ".d" + Axes[d], static_cast<long double>(primitiveGradient(d, v)) - exactPrimitiveGradient(d, v));
                    }
                }
                const auto &op = reconstruction.Operator(i);
                DNDS_check_throw_info(op.inverseRows.rows() == (efficient ? 3 : 9), "Incorrect stored inverse block size");
                localMinimumRank = std::min(localMinimumRank, op.numericalRank);
                localMaxima[0] = std::max(localMaxima[0], op.conditionNumber);
                localMaxima[1] = std::max(localMaxima[1], real(std::abs(weight - exactVolume) / exactVolume));
                localMaxima[2] = std::max(localMaxima[2], real(op.stencil.size()));
                localMaxima[3] = std::max(localMaxima[3], real(op.rings));
                localMinimumPrimitive[0] = std::min(localMinimumPrimitive[0], primitive(0));
                localMinimumPrimitive[1] = std::min(localMinimumPrimitive[1], primitive(4));
                nodes << mesh->node2nodeOrig(i, 0);
                for (int d = 0; d < 3; d++)
                    nodes << ',' << mesh->coords[i](d);
                nodes << ',' << weight << ',' << int(boundary);
                for (int v = 0; v < 5; v++) nodes << ',' << means[i](v);
                for (int v = 0; v < 5; v++) nodes << ',' << exactMean(v);
                for (int v = 0; v < 5; v++) nodes << ',' << value(v);
                for (int v = 0; v < 5; v++)
                    for (int d = 0; d < 3; d++) nodes << ',' << derivative(d, v);
                nodes << '\n';
            }
            nodes.close();
            DNDS_check_throw_info(nodes.good(), "Failed to finish patch-test CSV");
            for (const auto &surface : geometry.EdgeSurfaces())
                localCounts[2] += surface.quadrature.size();
            localMaxima[4] = MPI_Wtime() - start;
            real volumes[3]{}, maxima[5]{}, minimumPrimitive[2]{};
            Index counts[5]{};
            int minimumRank = 0;
            MPI_Allreduce(localVolumes, volumes, 3, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(localCounts, counts, 5, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            MPI_Allreduce(localMaxima, maxima, 5, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            MPI_Allreduce(localMinimumPrimitive, minimumPrimitive, 2, DNDS::DNDS_MPI_REAL, MPI_MIN, mpi.comm);
            MPI_Allreduce(&localMinimumRank, &minimumRank, 1, MPI_INT, MPI_MIN, mpi.comm);
            DNDS_check_throw_info(minimumRank == 9 && (efficient ? counts[2] == 0 : counts[2] > 0),
                                  "Invalid quadratic rank or quadrature mode");
            nlohmann::ordered_json norms;
            for (const auto &[name, local] : errors)
            {
                real sum[2]{}, maximum = 0;
                MPI_Allreduce(local.data(), sum, 2, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                MPI_Allreduce(&local[2], &maximum, 1, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
                const int r = name.rfind("all.", 0) == 0 ? 0 : (name.rfind("interior.", 0) == 0 ? 1 : 2);
                DNDS_check_throw_info(volumes[r] > 0, "Empty patch-test error region");
                norms[name] = {{"L1", sum[0] / volumes[r]}, {"L2", std::sqrt(sum[1] / volumes[r])}, {"Linf", maximum}};
            }
            if (mpi.rank == 0)
            {
                nlohmann::ordered_json result{
                    {"method", MethodName}, {"configuration", cfg}, {"source_configuration", argv[1]},
                    {"field_coefficients_numerator", Coefficients}, {"field_coefficient_denominator", 1000},
                    {"field_basis", {"1", "X", "Y", "Z", "X^2", "XY", "XZ", "Y^2", "YZ", "Z^2"}},
                    {"coordinate_center", {5, 5, 2}}, {"coordinate_scale", {5, 5, 2}},
                    {"initialization", efficient ? "production differential weights with analytic gradients" : "production volume quadrature"},
                    {"mean_oracle", "independent long-double barycentric tetrahedron integral and determinant"},
                    {"long_double_digits", std::numeric_limits<long double>::digits},
                    {"time", solver.SimulationTime()}, {"iteration", solver.CurrentIteration()},
                    {"compute_coefficients_calls", 1}, {"recover_point_values_calls", 1},
                    {"rhs_evaluations", 0}, {"time_steps", 0}, {"periodic", false},
                    {"mpi_ranks", mpi.size}, {"nodes", mesh->NumNodeGlobal()}, {"cells", mesh->NumCellGlobal()},
                    {"interior_nodes", counts[0]}, {"boundary_nodes", counts[1]}, {"volume", volumes[0]},
                    {"stored_quadrature_points", counts[2]}, {"micro_tetrahedra", counts[3]},
                    {"minimum_reconstruction_rank", minimumRank}, {"inverse_rows", efficient ? 3 : 9},
                    {"condition_max", maxima[0]}, {"volume_relative_error_max", maxima[1]},
                    {"stencil_max", maxima[2]}, {"rings_max", maxima[3]}, {"wall_seconds", maxima[4]},
                    {"rho_min", minimumPrimitive[0]}, {"pressure_min", minimumPrimitive[1]}, {"errors", norms}};
                std::ofstream metadata(output / "metrics.json");
                metadata << result.dump(2) << '\n';
                DNDS_check_throw_info(metadata.good(), "Cannot write patch-test metrics");
                DNDS::log() << "Quadratic patch test: rho point L2=" << norms["all.conservative.rho.value"]["L2"]
                            << ", rho gradient L2=" << norms["all.conservative.rho.gradient"]["L2"] << std::endl;
            }
        }
        catch (const std::exception &exception)
        {
            std::cerr << "NCFV quadratic patch test: " << exception.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
