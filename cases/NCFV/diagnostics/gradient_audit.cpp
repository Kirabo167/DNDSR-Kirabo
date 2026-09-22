/** Isolated gradient audit. No production operators are changed; no RHS or time stepping. */
#include "NCFV/NCFVSolver.hpp"
#include "NCFV/NCFVAnalytic.hpp"
#include "Geom/Quadrature.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>

using DNDS::real;
using Index = DNDS::index;
using namespace DNDS::NCFV;
using Vec9 = Eigen::Matrix<real, 9, 1>;
using Vec13 = Eigen::Matrix<real, 13, 1>;

namespace
{
    Vec9 ShiftedBasis(const Vec13 &m, const Vector3 &s, const Vector3 &lengths)
    {
        const Vector3 first = m.head<3>();
        Matrix3 second;
        second << 2 * m(3), m(4), m(5), m(4), 2 * m(6), m(7), m(5), m(7), 2 * m(8);
        second += s * first.transpose() + first * s.transpose() + s * s.transpose();
        second.array() /= (lengths * lengths.transpose()).array();
        Vec9 b;
        b.head<3>() = (first + s).cwiseQuotient(lengths);
        b.tail<6>() << 0.5 * second(0, 0), second(0, 1), second(0, 2),
            0.5 * second(1, 1), second(1, 2), 0.5 * second(2, 2);
        return b;
    }

    Eigen::Vector4d ShiftedCubic(const Vec13 &m, const Vector3 &s)
    {
        const real x = s(0), y = s(1), mx = m(0), my = m(1), xx = 2 * m(3), xy = m(4), yy = 2 * m(6);
        Eigen::Vector4d c;
        c << m(9) + 3 * x * xx + 3 * x * x * mx + x * x * x,
            m(10) + y * xx + 2 * x * xy + 2 * x * y * mx + x * x * my + x * x * y,
            m(11) + x * yy + 2 * y * xy + 2 * x * y * my + y * y * mx + x * y * y,
            m(12) + 3 * y * yy + 3 * y * y * my + y * y * y;
        return c;
    }

    Eigen::Vector4d DensityCubic(const Configuration &cfg, const Vector3 &position)
    {
        Vector3 r = position - Vector3(cfg.initialField.vortexCenter.data());
        for (int d = 0; d < 2; d++)
            r(d) -= cfg.mesh.periodicLengths[d] * std::round(r(d) / cfg.mesh.periodicLengths[d]);
        const real g = cfg.physics.gamma, m = 1 / (g - 1);
        const real a = cfg.initialField.vortexStrength / (2 * DNDS::pi) * std::exp((1 - r.head<2>().squaredNorm()) / 2);
        const real D = (g - 1) / (2 * g) * a * a, T = 1 - D;
        const real f2 = -m * D * std::pow(T, m - 1) + m * (m - 1) * D * D * std::pow(T, m - 2);
        const real f3 = m * D * std::pow(T, m - 1) - 3 * m * (m - 1) * D * D * std::pow(T, m - 2) + m * (m - 1) * (m - 2) * D * D * D * std::pow(T, m - 3);
        const real x = r(0), y = r(1);
        return Eigen::Vector4d{(12 * f2 * x + 8 * f3 * x * x * x) / 6, (4 * f2 * y + 8 * f3 * x * x * y) / 2,
                               (4 * f2 * x + 8 * f3 * x * y * y) / 2, (12 * f2 * y + 8 * f3 * y * y * y) / 6};
    }

    std::pair<real, Vector3> Wave(const Vector3 &x, int frequency)
    {
        const real k = 2 * DNDS::pi * frequency / 10;
        return {std::sin(k * x(0)) * std::cos(k * x(1)),
                Vector3{k * std::cos(k * x(0)) * std::cos(k * x(1)), -k * std::sin(k * x(0)) * std::sin(k * x(1)), 0}};
    }

    Eigen::MatrixXd GradientInverse(const Eigen::MatrixXd &a, const Eigen::VectorXd &weights, real tolerance)
    {
        const Eigen::MatrixXd weighted = weights.asDiagonal() * a;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(weighted, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto s = svd.singularValues();
        if (s.size() != 9 || s(8) <= tolerance * s(0))
            return {};
        return (svd.matrixV() * s.cwiseInverse().asDiagonal() * svd.matrixU().transpose() * weights.asDiagonal()).topRows(3);
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
            DNDS_check_throw_info(argc == 3, "Usage: gradient_audit config.json new-output-directory");
            auto cfg = LoadConfiguration(argv[1], {}, {}).configuration;
            DNDS_check_throw_info(cfg.dimension == 3 && cfg.initialField.isentropicVortex &&
                                      cfg.algorithm.mode == IntegrationMode::EfficientDifferential && !cfg.reconstruction.enableLimiter,
                                  "Expected unlimited efficient 3-D vortex configuration");
            cfg.algorithm.retainMicroGeometry = true; // Diagnostics only; quadrature is not part of the solver.
            cfg.time.iterations = 0;
            cfg.time.endTime = 0;
            cfg.io.writeVTK = false;
            cfg.io.writeFinalRestart = false;
            cfg.io.writeResolvedConfiguration = false;
            const std::filesystem::path output = argv[2];
            if (mpi.rank == 0)
            {
                DNDS_check_throw_info(!std::filesystem::exists(output), "Refusing to overwrite audit output");
                std::filesystem::create_directories(output);
            }
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            Solver<3> solver(mpi, cfg);
            solver.Initialize();
            const auto mesh = solver.Mesh();
            const auto &geom = solver.Geometry();
            const auto &reconstruction = solver.ReconstructionData();
            const auto &topology = solver.EdgeTopology();
            const NodeHalo &nodeHalo = solver.NodeCommunication();
            NodeStatePair fields, moments;
            const auto allocate = [&](NodeStatePair &field,
                                      const std::string &name,
                                      int rows)
            {
                field.InitPair(name, mpi);
                field.father->Resize(mesh->NumNode(), rows, 1);
                field.son->Resize(nodeHalo.NumNodeGhost(), rows, 1);
                field.BorrowSetup(nodeHalo.Layout());
                field.trans.initPersistentPull();
                for (Index i = 0; i < field.Size(); i++)
                    field[i].setZero();
            };
            allocate(fields, "audit.fields", 7);
            allocate(moments, "audit.centeredMoments", 13);
            for (Index i = 0; i < mesh->NumNode(); i++)
            {
                fields[i].setZero();
                moments[i].setZero();
                fields[i](0) = solver.StateField()[i](0);
                for (int f = 1; f <= 2; f++)
                {
                    fields[i](4 + f) = Wave(mesh->coords[i], f).first;
                    for (const auto &entry : geom.NodeVolume(i).pointRecoveryStencil)
                        fields[i](4 + f) += Wave(nodeHalo.Coordinate(entry.node), f).second.dot(
                            entry.gradientWeight);
                }
            }
            for (int order : {5, 6})
            {
                DNDS::Geom::Elem::Quadrature rule({DNDS::Geom::Elem::Tet4}, order);
                for (Index i = 0; i < mesh->NumNode(); i++)
                {
                    const auto &volume = geom.NodeVolume(i);
                    for (const auto &micro : volume.microVolumes)
                        for (int q = 0; q < rule.GetNumPoints(); q++)
                        {
                            const auto [a, w] = rule.GetQuadraturePointInfo(q);
                            const Vector3 x = micro.points[0] + a(0) * (micro.points[1] - micro.points[0]) +
                                              a(1) * (micro.points[2] - micro.points[0]) + a(2) * (micro.points[3] - micro.points[0]);
                            const real weight = 6 * micro.measure * w / volume.moments.measure;
                            fields[i](order == 6 ? 1 : 2) += weight * IsentropicVortex<3>(cfg, x, 0).first(0);
                            if (order == 6)
                            {
                                fields[i](3) += weight * Wave(x, 1).first;
                                fields[i](4) += weight * Wave(x, 2).first;
                                const Vector3 d = x - mesh->coords[i];
                                moments[i].head<9>() += weight * Reconstruction::EvaluateBasis(d, Vector3::Ones(), 3);
                                moments[i].tail<4>() += weight * Eigen::Vector4d{d(0) * d(0) * d(0), d(0) * d(0) * d(1), d(0) * d(1) * d(1), d(1) * d(1) * d(1)};
                            }
                        }
                }
            }
            fields.trans.startPersistentPull();
            moments.trans.startPersistentPull();
            fields.trans.waitPersistentPull();
            moments.trans.waitPersistentPull();
            NodeMatrixPair gradients, unused;
            gradients.InitPair("audit.gradients", mpi);
            gradients.father->Resize(mesh->NumNode(), 3, 7);
            gradients.son->Resize(nodeHalo.NumNodeGhost(), 3, 7);
            gradients.BorrowSetup(nodeHalo.Layout());
            gradients.trans.initPersistentPull();
            reconstruction.ComputeCoefficients(fields, gradients, unused);

            std::map<std::string, std::array<real, 3>> norms;
            std::map<std::string, real> maxima;
            auto add = [&](const std::string &key, const Vector3 &e, real v)
            {
                auto &n = norms[key];
                n[0] += v * e.norm();
                n[1] += v * e.squaredNorm();
                n[2] = std::max(n[2], e.norm());
            };
            const std::array<std::string, 7> names{"rho_initial", "rho_reference6", "rho_reference5", "wave1_reference", "wave2_reference", "wave1_initial", "wave2_initial"};
            std::ofstream nodes(output / fmt::format("nodes.rank{:04d}.csv", mpi.rank));
            nodes << "original_node,x,y,z,volume,radius_xy,radius_3d,tie_z,stencil,condition,rho_initial_error,rho_reference_error,cubic_error,higher_remainder_error,init_gradient_change,compact15_error,power4_error\n"
                  << std::setprecision(17);
            real localVolume = 0, localRadiusXY = 0, localRadius3D = 0, localTieVolume = 0, localCubicDotHigher = 0;
            real localPeriodicCount = 0;
            for (Index i = 0; i < mesh->NumNode(); i++)
            {
                const real volume = geom.NodeVolume(i).moments.measure;
                const auto &op = reconstruction.Operator(i);
                const Index rep = i;
                const real h = op.lengthScale;
                const Vector3 lengths = op.referenceLengths;
                const Matrix3 inverseLengths = lengths.cwiseInverse().asDiagonal();
                const Vec13 center = moments[rep];
                const Vec9 base = ShiftedBasis(center, Vector3::Zero(), lengths);
                const Vector3 exact = IsentropicVortex<3>(cfg, mesh->coords[i], 0).second.col(0);
                const Eigen::Vector4d cubic = DensityCubic(cfg, mesh->coords[i]);
                Eigen::MatrixXd matrix(op.stencil.size(), 9), differences(op.stencil.size(), 2);
                Eigen::VectorXd weights(op.stencil.size()), distance(op.stencil.size()), cubicRhs(op.stencil.size());
                real radiusXY = 0, radius3D = 0;
                bool tieZ = false;
                for (std::size_t row = 0; row < op.stencil.size(); row++)
                {
                    const Index j = op.stencil[row];
                    const Vector3 shift = nodeHalo.Displacement(rep, j);
                    matrix.row(row) = (ShiftedBasis(Vec13(moments[j]), shift, lengths) - base).transpose();
                    distance(row) = std::max(shift.norm() / h, cfg.reconstruction.distanceWeightFloor);
                    weights(row) = std::pow(distance(row), -cfg.reconstruction.distanceWeightPower);
                    differences(row, 0) = fields[j](0) - fields[i](0);
                    differences(row, 1) = fields[j](1) - fields[i](1);
                    cubicRhs(row) = cubic.dot(ShiftedCubic(Vec13(moments[j]), shift) - ShiftedCubic(center, Vector3::Zero()));
                    radiusXY = std::max(radiusXY, shift.head<2>().norm());
                    radius3D = std::max(radius3D, shift.norm());
                    tieZ = tieZ || std::abs(std::abs(shift(2)) - cfg.mesh.periodicLengths[2] / 2) < 1e-9;
                }
                Eigen::MatrixXd identity = Eigen::MatrixXd::Zero(3, 9);
                identity.leftCols<3>().setIdentity();
                const Eigen::MatrixXd defect = op.inverseRows * matrix - identity;
                maxima["independent_quadratic_identity_defect"] = std::max(maxima["independent_quadratic_identity_defect"], defect.cwiseAbs().maxCoeff());
                maxima["independent_quadratic_gradient_defect"] = std::max(maxima["independent_quadratic_gradient_defect"], (inverseLengths * defect).cwiseAbs().maxCoeff());
                const auto independent = GradientInverse(matrix, weights, cfg.reconstruction.svdTolerance);
                DNDS_check_throw_info(independent.rows() == 3, "Independent complete quadratic system lost rank");
                const Vector3 gIndependent = inverseLengths * independent * differences.col(0);
                maxima["independent_full_svd_gradient_difference"] = std::max(maxima["independent_full_svd_gradient_difference"], (gIndependent - gradients[i].col(0)).norm());
                Eigen::MatrixXd rescaled = matrix;
                rescaled.leftCols<3>() /= 2;
                rescaled.rightCols<6>() /= 4;
                const Vector3 gRescaled = 0.5 * inverseLengths * GradientInverse(rescaled, weights, cfg.reconstruction.svdTolerance) * differences.col(0);
                maxima["length_scale_invariance_difference"] = std::max(maxima["length_scale_invariance_difference"], (gRescaled - gIndependent).norm());
                maxima["condition_number"] = std::max(maxima["condition_number"], op.conditionNumber);
                const Eigen::MatrixXd weight4 = inverseLengths * GradientInverse(matrix, distance.array().pow(-4).matrix(), cfg.reconstruction.svdTolerance) * differences;
                add("rho_power4_initial", weight4.col(0) - exact, volume);
                add("rho_power4_reference", weight4.col(1) - exact, volume);

                std::vector<int> candidates(op.stencil.size());
                std::iota(candidates.begin(), candidates.end(), 0);
                std::vector<Index> direct;
                for (const auto &incidence : topology.Node2Edge(i))
                {
                    const auto &surface = geom.EdgeSurface(incidence.edge);
                    DNDS_check_throw_info(
                        surface.nodes[0] == i || surface.nodes[1] == i,
                        "Owned-node edge is not incident after exact-halo remapping");
                    direct.push_back(
                        surface.nodes[surface.nodes[0] == i ? 1 : 0]);
                }
                std::sort(direct.begin(), direct.end());
                direct.erase(std::unique(direct.begin(), direct.end()), direct.end());
                auto isDirect = [&](int row)
                { return std::find(direct.begin(), direct.end(), op.stencil[row]) != direct.end(); };
                // A geometric tie-break is independent of MPI partition/global numbering.
                auto geometricKey = [&](int row)
                {
                    const Vector3 delta = nodeHalo.Displacement(rep, op.stencil[row]);
                    return std::array<long long, 4>{std::llround(delta.squaredNorm() / 1e-12),
                                                    std::llround(delta(0) / 1e-10), std::llround(delta(1) / 1e-10), std::llround(delta(2) / 1e-10)};
                };
                std::sort(candidates.begin(), candidates.end(), [&](int a, int b)
                          {
                    if(isDirect(a)!=isDirect(b)) return isDirect(a);
                    return geometricKey(a)<geometricKey(b); });
                Eigen::MatrixXd compactGradient;
                int selected = std::max(15, int(direct.size()));
                for (; selected <= int(candidates.size()); selected++)
                {
                    Eigen::MatrixXd a(selected, 9), b(selected, 2);
                    Eigen::VectorXd w(selected);
                    for (int j = 0; j < selected; j++)
                    {
                        a.row(j) = matrix.row(candidates[j]);
                        b.row(j) = differences.row(candidates[j]);
                        w(j) = weights(candidates[j]);
                    }
                    const auto p = GradientInverse(a, w, cfg.reconstruction.svdTolerance);
                    if (p.rows() == 3)
                    {
                        compactGradient = inverseLengths * p * b;
                        break;
                    }
                }
                DNDS_check_throw_info(compactGradient.rows() == 3, "Compact diagnostic lost quadratic rank");
                maxima["compact_stencil_max"] = std::max(maxima["compact_stencil_max"], real(selected));
                add("rho_compact15_initial", compactGradient.col(0) - exact, volume);
                add("rho_compact15_reference", compactGradient.col(1) - exact, volume);
                for (int f = 0; f < 7; f++)
                {
                    const Vector3 reference = f < 3 ? exact : Wave(mesh->coords[i], (f == 3 || f == 5) ? 1 : 2).second;
                    add(names[f], gradients[i].col(f) - reference, volume);
                }
                const Vector3 e = gradients[i].col(1) - exact;
                const Vector3 leading = inverseLengths * op.inverseRows * cubicRhs;
                const Vector3 higher = e - leading;
                add("rho_cubic_taylor_term", leading, volume);
                add("rho_higher_taylor_remainder", higher, volume);
                add("rho_initialization_gradient_change", gradients[i].col(0) - gradients[i].col(1), volume);
                add("rho_reference_order_change", gradients[i].col(1) - gradients[i].col(2), volume);
                add(tieZ ? "rho_initial_halfbox_z" : "rho_initial_no_halfbox_z", gradients[i].col(0) - exact, volume);
                // Ensure every rank reduces the same named metrics even if its nodes contain only one subset.
                norms.try_emplace("rho_initial_halfbox_z", std::array<real, 3>{});
                norms.try_emplace("rho_initial_no_halfbox_z", std::array<real, 3>{});
                localCubicDotHigher += volume * leading.dot(higher);
                localVolume += volume;
                localRadiusXY += volume * radiusXY;
                localRadius3D += volume * radius3D;
                localTieVolume += volume * int(tieZ);
                localPeriodicCount += volume / std::pow(h, 3);
                nodes << mesh->node2nodeOrig(i, 0) << ',' << mesh->coords[i](0) << ',' << mesh->coords[i](1) << ',' << mesh->coords[i](2) << ',' << volume
                      << ',' << radiusXY << ',' << radius3D << ',' << tieZ << ',' << op.stencil.size() << ',' << op.conditionNumber << ',' << (gradients[i].col(0) - exact).norm()
                      << ',' << e.norm() << ',' << leading.norm() << ',' << higher.norm() << ',' << (gradients[i].col(0) - gradients[i].col(1)).norm()
                      << ',' << (compactGradient.col(0) - exact).norm() << ',' << (weight4.col(0) - exact).norm() << '\n';
            }
            nodes.close();
            real local[] = {localVolume, localRadiusXY, localRadius3D, localTieVolume, localCubicDotHigher, localPeriodicCount}, sum[6]{};
            MPI_Allreduce(local, sum, 6, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            nlohmann::ordered_json result;
            result["configuration"] = cfg;
            result["nodes"] = mesh->NumNodeGlobal();
            result["mpi_ranks"] = mpi.size;
            result["volume"] = sum[0];
            result["h"] = std::cbrt(sum[0] / std::round(sum[5]));
            result["time_steps"] = 0;
            result["rhs_evaluations"] = 0;
            result["reference_quadrature_orders"] = {5, 6};
            result["production_operator_modified"] = false;
            result["diagnostic_revision"] = 3;
            result["normalization"] = "dual-bounds-half-span (thesis 3-34)";
            result["compact_tie_break"] = "quantized geometric key, independent of MPI numbering";
            result["volume_weighted_stencil_radius_xy"] = sum[1] / sum[0];
            result["volume_weighted_stencil_radius_3d"] = sum[2] / sum[0];
            result["halfbox_z_volume_fraction"] = sum[3] / sum[0];
            result["cubic_higher_inner_product"] = sum[4] / sum[0];
            for (const auto &[key, n] : norms)
            {
                real sums[2]{}, maximum = 0;
                MPI_Allreduce(n.data(), sums, 2, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                MPI_Allreduce(&n[2], &maximum, 1, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
                result["errors"][key] = {{"L1", sums[0] / sum[0]}, {"L2", std::sqrt(sums[1] / sum[0])}, {"Linf", maximum}};
            }
            for (const auto &[key, value] : maxima)
            {
                real maximum = 0;
                MPI_Allreduce(&value, &maximum, 1, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
                result["checks"][key] = maximum;
            }
            real localTime = MPI_Wtime() - start, elapsed = 0;
            MPI_Allreduce(&localTime, &elapsed, 1, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            result["wall_seconds"] = elapsed;
            if (mpi.rank == 0)
            {
                std::ofstream out(output / "metrics.json");
                out << result.dump(2) << '\n';
                std::cout << result.dump(2) << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
