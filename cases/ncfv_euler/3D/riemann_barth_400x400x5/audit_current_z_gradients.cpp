/** Read-only reconstruction of z derivatives from a frozen NCFV restart. */
#include "NCFV/NCFVSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using DNDS::DNDS_MPI_INDEX;
using DNDS::DNDS_MPI_REAL;

namespace
{
    using State = Eigen::Matrix<DNDS::real, 5, 1>;
    constexpr int fieldCount = 17, recordWidth = 7, nodeWidth = 32;
    const std::array<std::string, 5> conservativeNames{"rho", "rhou", "rhov", "rhow", "rhoE"};
    const std::array<std::string, 6> primitiveNames{"rho", "p", "u", "v", "w", "rhoE"};

    DNDS::real Pressure(const State &state, DNDS::real gamma)
    {
        return (gamma - 1) * (state(4) - .5 * state.segment<3>(1).squaredNorm() / state(0));
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    try
    {
        DNDS_check_throw_info(argc == 4, "usage: saved-config restart-base output-summary.json");
        const std::uint16_t endianProbe = 1;
        DNDS_check_throw_info(sizeof(DNDS::real) == 8 &&
                                  *reinterpret_cast<const unsigned char *>(&endianProbe) == 1,
                              "Node-table format requires little-endian float64");
        auto cfg = DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
        cfg.io.restartInput = argv[2];
        cfg.time.iterations = 0;
        cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
        cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
        cfg.io.restartInterval = 0;
        DNDS_check_throw_info(cfg.algorithm.mode == DNDS::NCFV::IntegrationMode::EfficientDifferential,
                              "Current z-gradient audit requires efficient integration");
        DNDS::NCFV::Solver<3> solver(mpi, cfg);
        solver.Initialize();
        const auto &mesh = solver.Mesh();
        const auto &halo = solver.NodeCommunication();
        DNDS::NCFV::NodeStatePair means, rhs;
        const auto allocate = [&](DNDS::NCFV::NodeStatePair &field, const std::string &name)
        {
            field.InitPair(name, mpi);
            field.father->Resize(mesh->NumNode(), 5, 1);
            field.son->Resize(halo.NumNodeGhost(), 5, 1);
            field.BorrowSetup(halo.Layout());
            field.trans.initPersistentPull();
            for (DNDS::index node = 0; node < field.Size(); node++)
                field[node].setZero();
        };
        allocate(means, "currentZ.means");
        allocate(rhs, "currentZ.rhs");
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
            means[node] = solver.StateField()[node];
        DNDS::NCFV::SpatialOperator<3> spatial(
            mpi, mesh, solver.EdgeTopology(), solver.Geometry(), solver.ReconstructionData(),
            solver.Boundaries(), cfg.algorithm.mode, cfg.reconstruction, cfg.physics, cfg.time, halo);
        spatial.Initialize();
        // Reconstruct and evaluate a read-only RHS; never call Solver::Run().
        spatial.EvaluateRHS(means, rhs, false, false);
        std::array<DNDS::real, fieldCount * recordWidth> maxima{};
        for (int field = 0; field < fieldCount; field++)
            maxima[field * recordWidth] = -1;
        std::array<DNDS::index, fieldCount> aboveThreshold{};
        std::array<DNDS::real, fieldCount> squareSum{};
        std::array<DNDS::index, 5> thetaCounts{};
        std::vector<DNDS::real> table;
        table.reserve(static_cast<std::size_t>(mesh->NumNode()) * nodeWidth);
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
        {
            const auto coordinate = halo.Coordinate(node);
            const DNDS::index original = mesh->node2nodeOrig(node, 0);
            const State mean = means[node], point = spatial.PointValues()[node];
            const State derivative = spatial.Gradients()[node].row(2).transpose();
            const DNDS::real theta = spatial.LimiterFactors()[node](0);
            DNDS_check_throw_info(point.allFinite() && point(0) > 0 && Pressure(point, cfg.physics.gamma) > 0,
                                  "Primitive chain rule requires a physical recovered point");
            const Eigen::Vector3d velocity = point.segment<3>(1) / point(0);
            const Eigen::Vector3d velocityDerivative =
                (derivative.segment<3>(1) - velocity * derivative(0)) / point(0);
            const DNDS::real pressureDerivative = (cfg.physics.gamma - 1) *
                (derivative(4) - velocity.dot(derivative.segment<3>(1)) +
                 .5 * velocity.squaredNorm() * derivative(0));
            const std::array<DNDS::real, 6> primitiveDerivative{
                derivative(0), pressureDerivative, velocityDerivative(0), velocityDerivative(1),
                velocityDerivative(2), derivative(4)};
            std::array<DNDS::real, fieldCount> values{};
            for (int component = 0; component < 5; component++)
                values[component] = derivative(component);
            for (int component = 0; component < 6; component++)
            {
                values[5 + component] = primitiveDerivative[component];
                values[11 + component] = theta * primitiveDerivative[component];
            }
            for (int field = 0; field < fieldCount; field++)
            {
                const DNDS::real value = values[field];
                DNDS_check_throw_info(std::isfinite(value), "Nonfinite reconstructed derivative");
                aboveThreshold[field] += std::abs(value) > 1e-10;
                squareSum[field] += value * value;
                if (std::abs(value) > maxima[field * recordWidth])
                {
                    const std::array<DNDS::real, recordWidth> record{
                        std::abs(value), value, static_cast<DNDS::real>(original),
                        coordinate(0), coordinate(1), coordinate(2), theta};
                    std::copy(record.begin(), record.end(), maxima.begin() + field * recordWidth);
                }
            }
            thetaCounts[0]++;
            thetaCounts[1] += theta == 0;
            thetaCounts[2] += theta == 1;
            thetaCounts[3] += theta < .1;
            thetaCounts[4] += theta < .5;
            const std::array<DNDS::real, 5> prefix{
                static_cast<DNDS::real>(original), coordinate(0), coordinate(1), coordinate(2), theta};
            table.insert(table.end(), prefix.begin(), prefix.end());
            for (const State &state : {mean, point, derivative})
                for (int component = 0; component < 5; component++)
                    table.push_back(state(component));
            for (DNDS::real value : primitiveDerivative)
                table.push_back(value);
            for (DNDS::real value : primitiveDerivative)
                table.push_back(theta * value);
        }
        std::ostringstream rankSuffix;
        rankSuffix << std::setw(6) << std::setfill('0') << mpi.rank;
        const std::string tablePath = std::string(argv[3]) + ".rank" + rankSuffix.str() + ".bin";
        std::ofstream tableFile(tablePath, std::ios::binary);
        DNDS_check_throw_info(tableFile.good(), "Cannot open current derivative node table");
        tableFile.write(reinterpret_cast<const char *>(table.data()),
                        static_cast<std::streamsize>(table.size() * sizeof(DNDS::real)));
        tableFile.close();
        const DNDS::index localNodes = mesh->NumNode();
        DNDS::index totalNodes = 0;
        std::vector<DNDS::index> rankNodes(static_cast<std::size_t>(mpi.size));
        std::vector<DNDS::real> allMaxima(static_cast<std::size_t>(mpi.size) * maxima.size());
        std::array<DNDS::index, fieldCount> totalAbove{};
        std::array<DNDS::real, fieldCount> totalSquare{};
        std::array<DNDS::index, 5> totalTheta{};
        MPI_Reduce(&localNodes, &totalNodes, 1, DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
        MPI_Gather(&localNodes, 1, DNDS_MPI_INDEX, rankNodes.data(), 1, DNDS_MPI_INDEX, 0, mpi.comm);
        MPI_Gather(maxima.data(), static_cast<int>(maxima.size()), DNDS_MPI_REAL,
                   allMaxima.data(), static_cast<int>(maxima.size()), DNDS_MPI_REAL, 0, mpi.comm);
        MPI_Reduce(aboveThreshold.data(), totalAbove.data(), fieldCount, DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(squareSum.data(), totalSquare.data(), fieldCount, DNDS_MPI_REAL, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(thetaCounts.data(), totalTheta.data(), 5, DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
        if (mpi.rank == 0)
        {
            nlohmann::ordered_json statistics;
            for (int field = 0; field < fieldCount; field++)
            {
                int rank = 0;
                for (int candidate = 1; candidate < mpi.size; candidate++)
                    if (allMaxima[static_cast<std::size_t>(candidate) * maxima.size() + field * recordWidth] >
                        allMaxima[static_cast<std::size_t>(rank) * maxima.size() + field * recordWidth])
                        rank = candidate;
                const auto offset = static_cast<std::size_t>(rank) * maxima.size() + field * recordWidth;
                const std::string group = field < 5 ? "raw_conservative" : field < 11 ? "raw_primitive_at_protected_point" : "theta_scaled_primitive_reference";
                const std::string component = field < 5 ? conservativeNames[field] : primitiveNames[(field - 5) % 6];
                statistics[group][component] = {
                    {"max_abs_dz", allMaxima[offset]}, {"signed_dz", allMaxima[offset + 1]},
                    {"original_node", static_cast<DNDS::index>(std::llround(allMaxima[offset + 2]))},
                    {"coordinate", {allMaxima[offset + 3], allMaxima[offset + 4], allMaxima[offset + 5]}},
                    {"theta_at_maximum", allMaxima[offset + 6]}, {"owner_rank", rank},
                    {"nodes_above_1e_minus_10", totalAbove[field]},
                    {"unweighted_rms_dz", std::sqrt(totalSquare[field] / totalNodes)}};
            }
            std::vector<std::string> columns{"original_node", "x", "y", "z", "theta"};
            for (const std::string prefix : {"mean_", "point_", "raw_dz_"})
                for (const auto &component : conservativeNames)
                    columns.push_back(prefix + component);
            for (const std::string prefix : {"point_chain_dz_", "theta_scaled_point_chain_dz_"})
                for (const auto &component : primitiveNames)
                    columns.push_back(prefix + component);
            nlohmann::ordered_json result{
                {"configuration_file", argv[1]}, {"restart_input", cfg.io.restartInput},
                {"physical_time", solver.SimulationTime()}, {"iteration", solver.CurrentIteration()},
                {"mpi_ranks", mpi.size}, {"nodes", totalNodes}, {"nodes_per_rank", rankNodes},
                {"gamma", cfg.physics.gamma},
                {"limiter_configuration", {{"enableLimiter", cfg.reconstruction.enableLimiter},
                    {"limitPointRecovery", cfg.reconstruction.limitPointRecovery},
                    {"limitPressure", cfg.reconstruction.limitPressure},
                    {"directionalPairBounds", cfg.reconstruction.directionalPairBounds},
                    {"completeStencilRings", cfg.reconstruction.completeStencilRings},
                    {"limitFluxCorrection", cfg.reconstruction.limitFluxCorrection}}},
                {"derivative_origin", "Recomputed from saved conservative dual means and saved configuration; gradients are not stored in the restart. One RHS evaluation, no physical time advancement."},
                {"limiter_note", "Efficient mean-anchored limiting leaves shared conservative gradients unchanged. Theta limits complete mean-to-point and side corrections. Raw primitive derivatives use the chain rule at the actual protected recovered point. Theta-scaled derivatives are an anchor-local reference, not a unique post-limiter nodal derivative of the assembled macro-interface reconstruction."},
                {"theta_count_order", {"nodes", "equal_zero", "equal_one", "below_0.1", "below_0.5"}},
                {"theta_counts", totalTheta}, {"derivatives", statistics},
                {"node_table_format", "little-endian float64, row-major, no header"},
                {"node_table_columns", columns}, {"node_table_width", nodeWidth},
                {"node_table_pattern", std::string(argv[3]) + ".rank<RANK:06d>.bin"}};
            std::ofstream output(argv[3]);
            DNDS_check_throw_info(output.good(), "Cannot open current derivative summary");
            output << result.dump(2) << std::endl;
            std::cout << result.dump(2) << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "current z-gradient audit rank " << mpi.rank << ": " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    MPI_Finalize();
}
