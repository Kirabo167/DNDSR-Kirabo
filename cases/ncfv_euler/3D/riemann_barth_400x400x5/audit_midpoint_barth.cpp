/** Check the actual frozen solver's midpoint states against one-ring bounds. */
#include "NCFV/NCFVSolver.hpp"

#include <array>
#include <fstream>

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    try
    {
        DNDS_check_throw_info(argc == 5, "usage: config mesh initial.csv output.json");
        auto cfg = DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
        cfg.algorithm.mode = DNDS::NCFV::IntegrationMode::TraditionalQuadrature;
        cfg.reconstruction.limitPointRecovery = false;
        cfg.mesh.meshFile = argv[2];
        cfg.initialField.nodeFile = argv[3];
        cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
        cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
        cfg.io.restartInterval = 0;
        DNDS::NCFV::Solver<3> solver(mpi, cfg);
        solver.Initialize();
        const auto &mesh = solver.Mesh();
        const auto &halo = solver.NodeCommunication();
        const auto &reconstruction = solver.ReconstructionData();
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
        allocate(means, "midpointAudit.means");
        allocate(rhs, "midpointAudit.rhs");
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
            means[node] = solver.StateField()[node];
        DNDS::NCFV::SpatialOperator<3> spatial(mpi, mesh, solver.EdgeTopology(),
            solver.Geometry(), reconstruction, solver.Boundaries(), cfg.algorithm.mode,
            cfg.reconstruction, cfg.physics, cfg.time, halo);
        spatial.Initialize();
        spatial.EvaluateRHS(means, rhs);
        std::array<DNDS::index, 3> counts{}, globalCounts{};
        DNDS::real maximum = 0, globalMaximum = 0;
        const auto &points = spatial.PointValues();
        const auto &coefficients = spatial.Coefficients();
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
        {
            Eigen::VectorXd lower = means[node], upper = means[node];
            for (DNDS::index neighbor : reconstruction.DirectNeighbors(node))
            {
                lower = lower.cwiseMin(means[neighbor]);
                upper = upper.cwiseMax(means[neighbor]);
            }
            const auto check = [&](const Eigen::VectorXd &state)
            {
                counts[0]++;
                const DNDS::real excess = std::max((lower - state).maxCoeff(), (state - upper).maxCoeff());
                maximum = std::max(maximum, excess);
                if (excess > 1e-9)
                    counts[1]++;
                const DNDS::real pressure = .4 * (state(4) - .5 * state.segment<3>(1).squaredNorm() / state(0));
                if (!state.allFinite() || state(0) <= 0 || pressure <= 0)
                    counts[2]++;
            };
            check(points[node]);
            for (DNDS::index neighbor : reconstruction.DirectNeighbors(node))
                check(points[node] + coefficients[node].transpose() * DNDS::NCFV::Reconstruction::EvaluateBasis(
                    .5 * halo.Displacement(node, neighbor), reconstruction.ReferenceLengths(node), 3));
        }
        MPI_Allreduce(counts.data(), globalCounts.data(), 3, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        MPI_Allreduce(&maximum, &globalMaximum, 1, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        if (mpi.rank == 0)
        {
            nlohmann::json result{{"samples", globalCounts[0]}, {"outside_one_ring_bounds", globalCounts[1]},
                {"nonphysical_samples", globalCounts[2]}, {"maximum_bound_excess", globalMaximum}, {"ranks", mpi.size}};
            std::ofstream output(argv[4]);
            output << result.dump(2) << std::endl;
        }
        DNDS_check_throw_info(globalCounts[1] == 0 && globalCounts[2] == 0,
                              "Frozen midpoint limiter failed its actual-state bounds audit");
    }
    catch (const std::exception &error)
    {
        std::cerr << "Midpoint audit: " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    MPI_Finalize();
}
