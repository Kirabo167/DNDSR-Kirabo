/** Read-only column annihilation audit at an actual RP1 maximum-dz node. */
#include "NCFV/NCFVSolver.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>

namespace
{
    using State = Eigen::Matrix<DNDS::real, 5, 1>;
    using Vector = Eigen::Matrix<DNDS::real, 3, 1>;
    using Key = std::pair<long long, long long>;
    struct Column
    {
        DNDS::real weight = 0;
        State value = State::Zero();
        State minimum = State::Constant(1e300), maximum = State::Constant(-1e300);
        DNDS::real x = 0, y = 0;
        int points = 0;
    };
    std::array<DNDS::real, 5> Components(const State &state)
    {
        return {state(0), state(1), state(2), state(3), state(4)};
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    try
    {
        DNDS_check_throw_info(mpi.size == 1 && (argc == 3 || argc == 4),
                              "usage (np1): config output.json [original-node-id]");
        auto cfg = DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
        cfg.io.restartInput.clear();
        cfg.time.iterations = 0;
        cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
        cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
        cfg.io.restartInterval = 0;
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
        allocate(means, "columnActual.means");
        allocate(rhs, "columnActual.rhs");
        Vector minimum = Vector::Constant(1e300), maximum = Vector::Constant(-1e300);
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
        {
            means[node] = solver.StateField()[node];
            minimum = minimum.cwiseMin(halo.Coordinate(node));
            maximum = maximum.cwiseMax(halo.Coordinate(node));
        }
        DNDS::NCFV::SpatialOperator<3> spatial(
            mpi, mesh, solver.EdgeTopology(), solver.Geometry(), solver.ReconstructionData(),
            solver.Boundaries(), cfg.algorithm.mode, cfg.reconstruction, cfg.physics, cfg.time, halo);
        spatial.Initialize();
        spatial.EvaluateRHS(means, rhs);
        DNDS::index selected = DNDS::UnInitIndex;
        DNDS::real selectedMaximum = -1, allMaximum = 0;
        const DNDS::index requested = argc == 4 ? std::stoll(argv[3]) : DNDS::UnInitIndex;
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
        {
            const DNDS::real magnitude = std::abs(spatial.Gradients()[node](2, 0));
            allMaximum = std::max(allMaximum, magnitude);
            const Vector coordinate = halo.Coordinate(node);
            const bool coordinateInterior = ((coordinate - minimum).array() > 1e-10).all() &&
                                            ((maximum - coordinate).array() > 1e-10).all();
            if ((requested != DNDS::UnInitIndex && mesh->node2nodeOrig(node, 0) == requested) ||
                (requested == DNDS::UnInitIndex && coordinateInterior && magnitude > selectedMaximum))
            {
                selected = node;
                selectedMaximum = magnitude;
            }
        }
        DNDS_check_throw_info(selected != DNDS::UnInitIndex, "Cannot select an interior RP1 audit node");
        const auto &op = solver.ReconstructionData().Operator(selected);
        const Vector anchor = halo.Coordinate(selected);
        const State anchorValue = means[selected];
        const DNDS::real coordinateTolerance = 1e-10;
        const auto columnKey = [&](const Vector &coordinate)
        {
            return Key{std::llround(coordinate(0) / coordinateTolerance),
                       std::llround(coordinate(1) / coordinateTolerance)};
        };
        std::map<Key, Column> columns;
        State directDifference = State::Zero(), directFullRow = State::Zero();
        DNDS::real anchorWeight = 0;
        const std::string stencilPath = std::string(argv[2]) + ".stencil.csv";
        std::ofstream stencil(stencilPath);
        DNDS_check_throw_info(stencil.good(), "Cannot open actual maximum stencil CSV");
        stencil << std::setprecision(17)
                << "role,original_node,x,y,z,dx,dy,dz,z_gradient_weight,mean_rho,mean_rhou,mean_rhov,mean_rhow,mean_rhoE\n";
        const auto add = [&](const std::string &role, DNDS::index node, DNDS::real weight)
        {
            const Vector coordinate = halo.Coordinate(node), displacement = halo.Displacement(selected, node);
            const State value = means[node];
            auto &column = columns[columnKey(coordinate)];
            column.weight += weight;
            column.value = value;
            column.minimum = column.minimum.cwiseMin(value);
            column.maximum = column.maximum.cwiseMax(value);
            column.x = coordinate(0);
            column.y = coordinate(1);
            column.points++;
            directFullRow += weight * value;
            stencil << role << ',' << mesh->node2nodeOrig(node, 0);
            for (int direction = 0; direction < 3; direction++)
                stencil << ',' << coordinate(direction);
            for (int direction = 0; direction < 3; direction++)
                stencil << ',' << displacement(direction);
            stencil << ',' << weight;
            for (int component = 0; component < 5; component++)
                stencil << ',' << value(component);
            stencil << '\n';
        };
        for (std::size_t entry = 0; entry < op.stencil.size(); entry++)
        {
            const DNDS::real weight = op.inverseRows(2, static_cast<Eigen::Index>(entry)) / op.referenceLengths(2);
            anchorWeight -= weight;
            directDifference += weight * (State(means[op.stencil[entry]]) - anchorValue);
            add("stencil", op.stencil[entry], weight);
        }
        add("anchor", selected, anchorWeight);
        stencil.close();
        const std::string columnsPath = std::string(argv[2]) + ".columns.csv";
        std::ofstream columnFile(columnsPath);
        DNDS_check_throw_info(columnFile.good(), "Cannot open actual maximum column CSV");
        columnFile << std::setprecision(17)
                   << "x,y,points,column_weight_sum,mean_rho,mean_rhou,mean_rhov,mean_rhow,mean_rhoE,contribution_rho,contribution_rhou,contribution_rhov,contribution_rhow,contribution_rhoE\n";
        State columnSum = State::Zero(), maximumColumnDataSpan = State::Zero();
        DNDS::real maximumColumnWeight = 0;
        for (const auto &[key, column] : columns)
        {
            const State contribution = column.weight * column.value;
            columnSum += contribution;
            maximumColumnDataSpan = maximumColumnDataSpan.cwiseMax(column.maximum - column.minimum);
            maximumColumnWeight = std::max(maximumColumnWeight, std::abs(column.weight));
            columnFile << column.x << ',' << column.y << ',' << column.points << ',' << column.weight;
            for (int component = 0; component < 5; component++)
                columnFile << ',' << column.value(component);
            for (int component = 0; component < 5; component++)
                columnFile << ',' << contribution(component);
            columnFile << '\n';
        }
        const State actual = spatial.Gradients()[selected].row(2).transpose();
        nlohmann::ordered_json result{
            {"mpi_ranks", mpi.size}, {"configuration", argv[1]},
            {"completeStencilRings", cfg.reconstruction.completeStencilRings},
            {"selection", requested == DNDS::UnInitIndex ? "maximum actual abs(dz rho), all three coordinate directions interior" : "requested original node"},
            {"original_node", mesh->node2nodeOrig(selected, 0)},
            {"coordinate", {anchor(0), anchor(1), anchor(2)}},
            {"global_actual_dz_rho_abs_max", allMaximum}, {"selected_actual_dz_rho_abs", selectedMaximum},
            {"stencil_size_without_anchor", op.stencil.size()}, {"stencil_rings", op.rings},
            {"condition_number", op.conditionNumber}, {"reference_lengths", {op.referenceLengths(0), op.referenceLengths(1), op.referenceLengths(2)}},
            {"conservative_component_order", {"rho", "rhou", "rhov", "rhow", "rhoE"}},
            {"actual_dz", Components(actual)}, {"sum_weight_times_neighbor_minus_anchor", Components(directDifference)},
            {"sum_full_row_including_anchor", Components(directFullRow)}, {"sum_column_weight_times_column_value", Components(columnSum)},
            {"difference_form_vs_actual_abs_max", (directDifference - actual).cwiseAbs().maxCoeff()},
            {"column_form_vs_actual_abs_max", (columnSum - actual).cwiseAbs().maxCoeff()},
            {"column_data_span_max", Components(maximumColumnDataSpan)},
            {"maximum_abs_column_weight_sum", maximumColumnWeight},
            {"stencil_csv", stencilPath}, {"column_csv", columnsPath},
            {"column_condition", "For arbitrary z-flat nodal means, dz vanishes exactly iff each projected (x,y) column has zero total row weight, including anchor."}};
        std::ofstream output(argv[2]);
        DNDS_check_throw_info(output.good(), "Cannot open actual maximum column audit JSON");
        output << result.dump(2) << std::endl;
        std::cout << result.dump(2) << std::endl;
    }
    catch (const std::exception &error)
    {
        std::cerr << "actual column audit: " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    MPI_Finalize();
}
