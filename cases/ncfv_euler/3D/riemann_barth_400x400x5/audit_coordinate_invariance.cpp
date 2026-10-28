/** Read-only coordinate-invariance comparison, with all raw derivative components. */
#include "NCFV/NCFVSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <string>
#include <utility>

using DNDS::DNDS_MPI_INDEX;
using DNDS::DNDS_MPI_REAL;

namespace
{
    using State = Eigen::Matrix<DNDS::real, 5, 1>;
    using Vector = Eigen::Matrix<DNDS::real, 3, 1>;
    using ColumnKey = std::pair<long long, long long>;

    DNDS::real Pressure(const State &state, DNDS::real gamma)
    {
        return (gamma - 1) *
               (state(4) - .5 * state.segment<3>(1).squaredNorm() / state(0));
    }

    struct RegionStats
    {
        DNDS::index count = 0;
        std::array<DNDS::real, 5> gradientZMaximum{};
        std::array<DNDS::real, 5> rhsMaximum{};

        void Add(const Eigen::MatrixXd &gradient, const State &rhs)
        {
            count++;
            for (int variable = 0; variable < 5; variable++)
            {
                gradientZMaximum[variable] = std::max(
                    gradientZMaximum[variable], std::abs(gradient(2, variable)));
                rhsMaximum[variable] = std::max(
                    rhsMaximum[variable], std::abs(rhs(variable)));
            }
        }

        nlohmann::ordered_json Reduce(const DNDS::MPIInfo &mpi) const
        {
            RegionStats global;
            MPI_Reduce(&count, &global.count, 1, DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
            MPI_Reduce(gradientZMaximum.data(), global.gradientZMaximum.data(), 5,
                       DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
            MPI_Reduce(rhsMaximum.data(), global.rhsMaximum.data(), 5,
                       DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
            return {{"nodes", global.count}, {"raw_gradient_z_abs_max", global.gradientZMaximum},
                    {"rhs_abs_max", global.rhsMaximum}};
        }
    };
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    try
    {
        DNDS_check_throw_info(argc >= 3 && argc <= 5,
                              "usage: config out.json [meshOverride|-] [nodeFileOverride|-]");
        auto cfg = DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
        if (argc > 3 && std::string(argv[3]) != "-")
        {
            cfg.mesh.meshFile = argv[3];
        }
        if (argc > 4 && std::string(argv[4]) != "-")
        {
            cfg.initialField.nodeFile = argv[4];
        }
        cfg.time.iterations = 0;
        cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
        cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
        cfg.io.restartInterval = 0;
        DNDS_check_throw_info(
            cfg.algorithm.mode == DNDS::NCFV::IntegrationMode::EfficientDifferential &&
                cfg.reconstruction.method != DNDS::NCFV::ReconstructionMethod::Variational,
            "Spanwise operator audit requires efficient least-squares reconstruction");

        DNDS::NCFV::Solver<3> solver(mpi, cfg);
        solver.Initialize();
        const auto &mesh = solver.Mesh();
        const auto &halo = solver.NodeCommunication();
        const auto &reconstruction = solver.ReconstructionData();
        DNDS::NCFV::NodeStatePair means, rhs, raw, originalIDs;
        const auto allocate = [&](DNDS::NCFV::NodeStatePair &field,
                                  const std::string &name, int rows)
        {
            field.InitPair(name, mpi);
            field.father->Resize(mesh->NumNode(), rows, 1);
            field.son->Resize(halo.NumNodeGhost(), rows, 1);
            field.BorrowSetup(halo.Layout());
            field.trans.initPersistentPull();
            for (DNDS::index i = 0; i < field.Size(); i++)
            {
                field[i].setZero();
            }
        };
        allocate(means, "zAudit.means", 5);
        allocate(rhs, "zAudit.rhs", 5);
        allocate(raw, "zAudit.rawPoints", 5);
        allocate(originalIDs, "zAudit.originalIDs", 1);
        Vector localMinimum = Vector::Constant(1e300);
        Vector localMaximum = Vector::Constant(-1e300);
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            means[i] = solver.StateField()[i];
            originalIDs[i](0) = static_cast<DNDS::real>(mesh->node2nodeOrig(i, 0));
            localMinimum = localMinimum.cwiseMin(halo.Coordinate(i));
            localMaximum = localMaximum.cwiseMax(halo.Coordinate(i));
        }
        originalIDs.trans.startPersistentPull();
        originalIDs.trans.waitPersistentPull();
        Vector minimum, maximum;
        MPI_Allreduce(localMinimum.data(), minimum.data(), 3,
                      DNDS_MPI_REAL, MPI_MIN, mpi.comm);
        MPI_Allreduce(localMaximum.data(), maximum.data(), 3,
                      DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        const DNDS::real coordinateTolerance =
            1e-10 * std::max(DNDS::real(1), (maximum - minimum).norm());
        const auto columnKey = [&](const Vector &coordinate)
        {
            return ColumnKey{
                std::llround((coordinate(0) - minimum(0)) / coordinateTolerance),
                std::llround((coordinate(1) - minimum(1)) / coordinateTolerance)};
        };
        const auto columnWeights = [&](DNDS::index node)
        {
            const auto &op = reconstruction.Operator(node);
            std::map<ColumnKey, DNDS::real> sums;
            DNDS::real anchorWeight = 0;
            for (std::size_t j = 0; j < op.stencil.size(); j++)
            {
                const DNDS::real weight = op.inverseRows(2, static_cast<Eigen::Index>(j)) /
                                          op.referenceLengths(2);
                sums[columnKey(halo.Coordinate(op.stencil[j]))] += weight;
                anchorWeight -= weight;
            }
            sums[columnKey(halo.Coordinate(node))] += anchorWeight;
            return sums;
        };

        DNDS::NCFV::SpatialOperator<3> spatial(
            mpi, mesh, solver.EdgeTopology(), solver.Geometry(), reconstruction,
            solver.Boundaries(), cfg.algorithm.mode, cfg.reconstruction,
            cfg.physics, cfg.time, halo);
        spatial.Initialize();
        const DNDS::real residual = spatial.EvaluateRHS(means, rhs);
        reconstruction.RecoverPointValues(means, spatial.Gradients(),
                                           spatial.Coefficients(), raw);
        const auto &points = spatial.PointValues();
        const auto &gradients = spatial.Gradients();
        const auto &theta = spatial.LimiterFactors();
        const std::string nodePath = std::string(argv[2]) + ".rank" +
                                     std::to_string(mpi.rank) + ".csv";
        std::ofstream nodeDump(nodePath);
        DNDS_check_throw_info(nodeDump.good(), "Cannot open spanwise audit node CSV");
        nodeDump << "original_node,global_node,x,y,z,z_physical_boundary,z_end_coordinate_layer";
        for (const std::string prefix : {"mean", "raw_gradient_x", "raw_gradient_y", "raw_gradient_z", "rhs", "raw_point", "protected_point"})
        {
            for (const std::string component : {"rho", "rhou", "rhov", "rhow", "rhoE"})
            {
                nodeDump << ',' << prefix << '_' << component;
            }
        }
        nodeDump << ",theta,max_abs_z_column_weight_sum\n" << std::setprecision(17);

        std::array<RegionStats, 5> regions;
        std::array<DNDS::real, 15> gradientMaximum{};
        std::array<DNDS::real, 5> limitedGradientZMaximum{};
        std::array<DNDS::real, 18> stateMinimum, stateMaximum;
        stateMinimum.fill(1e300);
        stateMaximum.fill(-1e300);
        DNDS::index localWorstNode = DNDS::UnInitIndex;
        DNDS::real localWorstColumn = -1;
        DNDS::index nonzeroColumnNodes = 0;
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            const Vector coordinate = halo.Coordinate(i);
            bool physicalZBoundary = false;
            for (const auto &piece : solver.Geometry().NodeVolume(i).boundaryPieces)
            {
                if (piece.measure > 0 &&
                    std::abs(piece.vectorMeasure(2)) > (1 - 1e-10) * piece.measure)
                {
                    physicalZBoundary = true;
                }
            }
            const bool endCoordinateLayer =
                std::abs(coordinate(2) - minimum(2)) <= coordinateTolerance ||
                std::abs(coordinate(2) - maximum(2)) <= coordinateTolerance;
            const Eigen::MatrixXd gradient = gradients[i];
            regions[0].Add(gradient, State(rhs[i]));
            regions[physicalZBoundary ? 1 : 2].Add(gradient, State(rhs[i]));
            regions[endCoordinateLayer ? 3 : 4].Add(gradient, State(rhs[i]));
            DNDS::real maximumColumn = 0;
            for (const auto &entry : columnWeights(i))
            {
                maximumColumn = std::max(maximumColumn, std::abs(entry.second));
            }
            nonzeroColumnNodes += maximumColumn > 1e-10;
            if (maximumColumn > localWorstColumn)
            {
                localWorstColumn = maximumColumn;
                localWorstNode = i;
            }
            nodeDump << mesh->node2nodeOrig(i, 0) << ',' << halo.LocalToGlobal(i)
                     << ',' << coordinate(0) << ',' << coordinate(1) << ',' << coordinate(2)
                     << ',' << physicalZBoundary << ',' << endCoordinateLayer;
            for (int variable = 0; variable < 5; variable++)
            {
                nodeDump << ',' << means[i](variable);
                for (int direction = 0; direction < 3; direction++)
                {
                    gradientMaximum[direction * 5 + variable] = std::max(
                        gradientMaximum[direction * 5 + variable],
                        std::abs(gradient(direction, variable)));
                }
                limitedGradientZMaximum[variable] = std::max(
                    limitedGradientZMaximum[variable],
                    std::abs(theta[i](0) * gradient(2, variable)));
            }
            for (int direction = 0; direction < 3; direction++)
            {
                for (int variable = 0; variable < 5; variable++)
                {
                    nodeDump << ',' << gradient(direction, variable);
                }
            }
            for (int variable = 0; variable < 5; variable++)
            {
                nodeDump << ',' << rhs[i](variable);
            }
            for (const State state : {State(raw[i]), State(points[i])})
            {
                for (int variable = 0; variable < 5; variable++)
                {
                    nodeDump << ',' << state(variable);
                }
            }
            nodeDump << ',' << theta[i](0) << ',' << maximumColumn << '\n';
            const std::array<State, 3> states{State(means[i]), State(raw[i]), State(points[i])};
            for (int field = 0; field < 3; field++)
            {
                for (int variable = 0; variable < 6; variable++)
                {
                    const DNDS::real value = variable < 5 ? states[field](variable) :
                        Pressure(states[field], cfg.physics.gamma);
                    stateMinimum[6 * field + variable] = std::min(stateMinimum[6 * field + variable], value);
                    stateMaximum[6 * field + variable] = std::max(stateMaximum[6 * field + variable], value);
                }
            }
        }
        nodeDump.close();

        struct MaximumLocation
        {
            double value;
            int rank;
        } localLocation{localWorstColumn, mpi.rank}, globalLocation{};
        MPI_Allreduce(&localLocation, &globalLocation, 1, MPI_DOUBLE_INT, MPI_MAXLOC, mpi.comm);
        const std::string stencilPath = std::string(argv[2]) + ".worst_stencil.csv";
        std::array<DNDS::real, 7> worstNode{};
        if (mpi.rank == globalLocation.rank && localWorstNode != DNDS::UnInitIndex)
        {
            const DNDS::index i = localWorstNode;
            const auto &op = reconstruction.Operator(i);
            const auto sums = columnWeights(i);
            const Vector anchor = halo.Coordinate(i);
            worstNode = {static_cast<DNDS::real>(mesh->node2nodeOrig(i, 0)),
                         static_cast<DNDS::real>(halo.LocalToGlobal(i)),
                         anchor(0), anchor(1), anchor(2), op.conditionNumber,
                         static_cast<DNDS::real>(op.stencil.size())};
            std::ofstream dump(stencilPath);
            DNDS_check_throw_info(dump.good(), "Cannot open spanwise audit stencil CSV");
            dump << "role,original_node,global_node,x,y,z,dx,dy,dz,z_gradient_weight,"
                    "xy_column_weight_sum,mean_rho,mean_rhou,mean_rhov,mean_rhow,mean_rhoE\n"
                 << std::setprecision(17);
            const auto write = [&](const std::string &role, DNDS::index node, DNDS::real weight)
            {
                const Vector coordinate = halo.Coordinate(node);
                const Vector displacement = halo.Displacement(i, node);
                dump << role << ',' << static_cast<DNDS::index>(std::llround(originalIDs[node](0)))
                     << ',' << halo.LocalToGlobal(node) << ',' << coordinate(0) << ',' << coordinate(1)
                     << ',' << coordinate(2) << ',' << displacement(0) << ',' << displacement(1)
                     << ',' << displacement(2) << ',' << weight << ',' << sums.at(columnKey(coordinate));
                for (int variable = 0; variable < 5; variable++)
                {
                    dump << ',' << means[node](variable);
                }
                dump << '\n';
            };
            DNDS::real anchorWeight = 0;
            for (std::size_t j = 0; j < op.stencil.size(); j++)
            {
                const DNDS::real weight = op.inverseRows(2, static_cast<Eigen::Index>(j)) /
                                          op.referenceLengths(2);
                anchorWeight -= weight;
                write("stencil", op.stencil[j], weight);
            }
            write("anchor", i, anchorWeight);
        }
        MPI_Bcast(worstNode.data(), 7, DNDS_MPI_REAL, globalLocation.rank, mpi.comm);
        std::array<DNDS::real, 15> globalGradientMaximum{};
        std::array<DNDS::real, 5> globalLimitedGradientZMaximum{};
        std::array<DNDS::real, 18> globalStateMinimum{}, globalStateMaximum{};
        DNDS::index globalNonzeroColumnNodes = 0;
        MPI_Reduce(gradientMaximum.data(), globalGradientMaximum.data(), 15,
                   DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        MPI_Reduce(limitedGradientZMaximum.data(), globalLimitedGradientZMaximum.data(), 5,
                   DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        MPI_Reduce(stateMinimum.data(), globalStateMinimum.data(), 18,
                   DNDS_MPI_REAL, MPI_MIN, 0, mpi.comm);
        MPI_Reduce(stateMaximum.data(), globalStateMaximum.data(), 18,
                   DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        MPI_Reduce(&nonzeroColumnNodes, &globalNonzeroColumnNodes, 1,
                   DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
        nlohmann::ordered_json regional;
        const std::array<std::string, 5> names{
            "all", "physical_z_boundary", "not_physical_z_boundary",
            "z_end_coordinate_layers", "other_z_coordinate_layers"};
        for (int region = 0; region < 5; region++)
        {
            regional[names[region]] = regions[region].Reduce(mpi);
        }
        if (mpi.rank == 0)
        {
            nlohmann::ordered_json result{
                {"mpi_ranks", mpi.size}, {"iteration", solver.CurrentIteration()},
                {"physical_time", solver.SimulationTime()}, {"residual_rms", residual},
                {"conservative_order", {"rho", "rhou", "rhov", "rhow", "rhoE"}},
                {"gradient_order", "x five components, y five components, z five components"},
                {"raw_gradient_abs_max", globalGradientMaximum},
                {"theta_times_raw_gradient_z_abs_max", globalLimitedGradientZMaximum},
                {"state_range_order", "mean, raw_point, protected_point; each rho,rhou,rhov,rhow,rhoE,p"},
                {"state_minimum", globalStateMinimum}, {"state_maximum", globalStateMaximum},
                {"coordinate_minimum", {minimum(0), minimum(1), minimum(2)}},
                {"coordinate_maximum", {maximum(0), maximum(1), maximum(2)}},
                {"coordinate_group_tolerance", coordinateTolerance},
                {"periodic_lengths", cfg.mesh.periodicLengths},
                {"preserveCoordinateInvariance", cfg.reconstruction.preserveCoordinateInvariance},
                {"regions", regional},
                {"max_abs_z_column_weight_sum", globalLocation.value},
                {"nodes_with_z_column_weight_sum_above_1e_minus_10", globalNonzeroColumnNodes},
                {"worst_node_rank", globalLocation.rank},
                {"worst_node_order", {"original_node", "global_node", "x", "y", "z", "condition", "stencil_size"}},
                {"worst_node", worstNode}, {"worst_stencil_csv", stencilPath},
                {"node_csv_pattern", std::string(argv[2]) + ".rank<RANK>.csv"},
                {"physical_z_boundary_definition", "An existing boundary piece has normal parallel to z"},
                {"z_end_coordinate_layer_note", "Coordinate extrema of quotient nodes; not physical boundaries in a z-periodic mesh"},
                {"column_test", "Sum the z-gradient row, including its anchor coefficient, separately for each identical (x,y) column"}};
            std::ofstream output(argv[2]);
            DNDS_check_throw_info(output.good(), "Cannot open spanwise audit JSON");
            output << result.dump(2) << std::endl;
            std::cout << result.dump(2) << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "z invariance audit rank " << mpi.rank << ": " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    MPI_Finalize();
}
