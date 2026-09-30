/** Inspect the eight prism nodes whose initial recovery depends on MPI partitioning. */
#include "NCFV/NCFVSolver.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>

using namespace DNDS::NCFV;

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        try
        {
            DNDS_check_throw_info(argc == 4,
                                  "usage: base-config mesh-file output-prefix");
            std::ifstream source(argv[1]);
            DNDS_check_throw_info(source.good(), "Cannot read base configuration");
            nlohmann::ordered_json resolved = Configuration{};
            resolved.merge_patch(nlohmann::ordered_json::parse(source));
            for (auto &boundary : resolved["physics"]["boundaryZones"])
            {
                nlohmann::ordered_json normalized = BoundaryZoneSettings{};
                normalized.merge_patch(boundary);
                boundary = std::move(normalized);
            }
            auto cfg = resolved.get<Configuration>();
            cfg.mesh.meshFile = argv[2];
            cfg.algorithm.mode = IntegrationMode::TraditionalQuadrature;
            cfg.algorithm.retainMicroGeometry = false;
            cfg.reconstruction.method = ReconstructionMethod::LeastSquares;
            cfg.reconstruction.stencilSizeFactor = 1.7;
            cfg.reconstruction.enableLimiter = false;
            cfg.time.iterations = 0;
            cfg.io.writeVTK = false;
            cfg.io.writeInitial = false;
            cfg.io.writeFinal = false;
            cfg.io.writeFinalRestart = false;
            cfg.io.writeResolvedConfiguration = false;
            cfg.io.outputPrefix = std::string(argv[3]) + ".unused";
            cfg.Validate();
            Solver<3> solver(mpi, cfg);
            solver.Initialize();
            const std::set<DNDS::index> targets{
                400, 887, 1374, 1861, 2348, 2835, 3322, 3809};
            nlohmann::ordered_json result = nlohmann::ordered_json::array();
            for (DNDS::index i = 0; i < solver.Mesh()->NumNode(); i++)
            {
                const DNDS::index original = solver.Mesh()->node2nodeOrig(i, 0);
                if (!targets.count(original))
                    continue;
                const auto &op = solver.ReconstructionData().Operator(i);
                std::vector<DNDS::index> runtimeStencilGlobals;
                nlohmann::ordered_json stencilDisplacements =
                    nlohmann::ordered_json::array();
                for (DNDS::index local : op.stencil)
                {
                    runtimeStencilGlobals.push_back(
                        solver.NodeCommunication().LocalToGlobal(local));
                    const auto displacement =
                        solver.NodeCommunication().Displacement(i, local);
                    stencilDisplacements.push_back({displacement.x(),
                                                    displacement.y(),
                                                    displacement.z()});
                }
                nlohmann::ordered_json matrix = nlohmann::ordered_json::array();
                for (Eigen::Index row = 0; row < op.inverseRows.rows(); row++)
                {
                    auto values = nlohmann::ordered_json::array();
                    for (Eigen::Index col = 0; col < op.inverseRows.cols(); col++)
                        values.push_back(op.inverseRows(row, col));
                    matrix.push_back(std::move(values));
                }
                result.push_back({
                    {"original_node", original}, {"rank", mpi.rank},
                    {"stencil_globals", runtimeStencilGlobals},
                    {"stencil_displacements", stencilDisplacements},
                    {"inverse_rows", matrix},
                    {"direct_neighbors", op.directNeighborCount},
                    {"condition", op.conditionNumber},
                    {"reference_lengths", {op.referenceLengths.x(),
                                           op.referenceLengths.y(),
                                           op.referenceLengths.z()}}});
            }
            const auto output = std::filesystem::path(
                std::string(argv[3]) + ".rank" + std::to_string(mpi.rank) + ".json");
            std::ofstream file(output);
            DNDS_check_throw_info(file.good(), "Cannot open probe output");
            file << std::setw(2) << result << '\n';
            DNDS_check_throw_info(file.good(), "Cannot write probe output");
        }
        catch (const std::exception &error)
        {
            std::cerr << "Partition stencil probe: " << error.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
}
