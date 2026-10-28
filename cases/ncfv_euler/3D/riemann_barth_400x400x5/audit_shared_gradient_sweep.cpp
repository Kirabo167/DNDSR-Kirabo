/** Export raw reconstruction and exact surface weights for sequential replay. */
#include "NCFV/NCFVSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <set>
#include <string>

namespace
{
    using Json = nlohmann::ordered_json;

    template <typename Derived>
    Json VectorJson(const Eigen::MatrixBase<Derived> &vector)
    {
        Json result = Json::array();
        for (Eigen::Index row = 0; row < vector.size(); row++)
            result.push_back(vector(row));
        return result;
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    try
    {
        DNDS_check_throw_info(mpi.size == 1 && argc >= 3 && argc <= 6,
                              "usage (np1): config output.json [mesh-override|-] [node-field-override|-] [restart-prefix|-]");
        auto cfg = DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
        if (argc > 3 && std::string(argv[3]) != "-")
            cfg.mesh.meshFile = argv[3];
        if (argc > 4 && std::string(argv[4]) != "-")
            cfg.initialField.nodeFile = argv[4];
        cfg.io.restartInput = argc > 5 && std::string(argv[5]) != "-" ? argv[5] : "";
        cfg.time.iterations = 0;
        cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
        cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
        cfg.io.restartInterval = 0;
        cfg.reconstruction.enableLimiter = false;
        cfg.reconstruction.limitPointRecovery = false;
        cfg.reconstruction.limitPressure = false;
        cfg.reconstruction.directionalPairBounds = false;
        cfg.reconstruction.limitFluxCorrection = false;
        cfg.reconstruction.fluxCorrectionPairBounds = false;
        DNDS_check_throw_info(
            cfg.algorithm.mode == DNDS::NCFV::IntegrationMode::EfficientDifferential &&
                cfg.reconstruction.method != DNDS::NCFV::ReconstructionMethod::Variational,
            "Shared-gradient replay export requires efficient least-squares reconstruction");
        DNDS::NCFV::Solver<3> solver(mpi, cfg);
        solver.Initialize();
        const auto &mesh = solver.Mesh();
        const auto &halo = solver.NodeCommunication();
        DNDS_check_throw_info(halo.NumNodeGhost() == 0,
                              "The np1 replay exporter expects all referenced nodes to be owned");
        DNDS::NCFV::NodeStatePair means, points;
        for (auto *field : {&means, &points})
        {
            field->InitPair(field == &means ? "sweepExport.means" : "sweepExport.points", mpi);
            field->father->Resize(mesh->NumNode(), 5, 1);
            field->son->Resize(0, 5, 1);
            field->BorrowSetup(halo.Layout());
            field->trans.initPersistentPull();
        }
        DNDS::NCFV::NodeMatrixPair gradients, coefficients;
        for (auto *field : {&gradients, &coefficients})
        {
            const int rows = field == &gradients ? 3 : 9;
            field->InitPair(field == &gradients ? "sweepExport.gradients" : "sweepExport.coefficients", mpi);
            field->father->Resize(mesh->NumNode(), rows, 5);
            field->son->Resize(0, rows, 5);
            field->BorrowSetup(halo.Layout());
            field->trans.initPersistentPull();
            for (DNDS::index node = 0; node < field->Size(); node++)
                (*field)[node].setZero();
        }
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
            means[node] = solver.StateField()[node];
        means.trans.startPersistentPull();
        means.trans.waitPersistentPull();
        solver.ReconstructionData().ComputeCoefficients(means, gradients, coefficients);
        gradients.trans.startPersistentPull();
        gradients.trans.waitPersistentPull();
        solver.ReconstructionData().RecoverPointValues(means, gradients, coefficients, points);
        const DNDS::real coordinateTolerance = 1e-10;
        Json nodes = Json::array(), faces = Json::array();
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
        {
            Json recovery = Json::array(), gradient = Json::array();
            for (int direction = 0; direction < 3; direction++)
                gradient.push_back(VectorJson(gradients[node].row(direction).transpose()));
            for (const auto &entry : solver.Geometry().NodeVolume(node).pointRecoveryStencil)
            {
                DNDS_check_throw_info(entry.node >= 0 && entry.node < mesh->NumNode(),
                                      "Point-recovery reference outside exported nodes");
                recovery.push_back({{"node", entry.node}, {"weight", VectorJson(entry.gradientWeight)}});
            }
            nodes.push_back({{"local_index", node}, {"original_id", mesh->node2nodeOrig(node, 0)},
                             {"coordinate", VectorJson(halo.Coordinate(node))},
                             {"mean", VectorJson(means[node])}, {"gradient", gradient},
                             {"point", VectorJson(points[node])}, {"recovery", recovery}});
        }
        std::set<std::array<DNDS::index, 5>> seenKeys;
        for (DNDS::index edge = 0; edge < solver.EdgeTopology().NumEdge(); edge++)
        {
            const auto &surface = solver.Geometry().EdgeSurface(edge);
            const DNDS::index first = mesh->node2nodeOrig(surface.nodes[0], 0);
            const DNDS::index second = mesh->node2nodeOrig(surface.nodes[1], 0);
            const bool forward = first < second;
            const DNDS::NCFV::Vector3 displacement = (forward ? 1.0 : -1.0) *
                halo.Displacement(surface.nodes[0], surface.nodes[1]);
            const std::array<DNDS::index, 5> key{
                std::min(first, second), std::max(first, second),
                std::llround(displacement(0) / coordinateTolerance),
                std::llround(displacement(1) / coordinateTolerance),
                std::llround(displacement(2) / coordinateTolerance)};
            DNDS_check_throw_info(seenKeys.insert(key).second, "Duplicate canonical macro-interface key");
            Json stencil = Json::array();
            for (const auto &entry : surface.efficientStencil)
            {
                DNDS_check_throw_info(entry.node >= 0 && entry.node < mesh->NumNode(),
                                      "Surface reference outside exported nodes");
                stencil.push_back({{"node", entry.node},
                    {"weights", {VectorJson(entry.stateGradientWeights[0]), VectorJson(entry.stateGradientWeights[1])}}});
            }
            faces.push_back({{"local_index", edge}, {"anchors", surface.nodes}, {"measure", surface.measure},
                             {"vector_measure", VectorJson(surface.vectorMeasure)}, {"key", key},
                             {"canonical_displacement", VectorJson(displacement)}, {"stencil", stencil}});
        }
        Json result{
            {"format", "DNDSR shared-gradient sequential replay export v1"}, {"mpi_ranks", mpi.size},
            {"source_configuration", argv[1]}, {"mesh_file", cfg.mesh.meshFile},
            {"initial_node_file", cfg.initialField.nodeFile}, {"gamma", cfg.physics.gamma},
            {"periodic_lengths", cfg.mesh.periodicLengths}, {"coordinate_tolerance", coordinateTolerance},
            {"complete_stencil_rings", cfg.reconstruction.completeStencilRings},
            {"node_count", nodes.size()}, {"face_count", faces.size()},
            {"component_order", {"rho", "rhou", "rhov", "rhow", "rhoE"}},
            {"point_formula", "point_i = mean_i - sum_{k in recovery_i} gradient_k^T weight_ik"},
            {"face_formula", "face_mean_side = point_anchor + sum_{j in stencil} gradient_j^T weights_j_side / measure"},
            {"state_note", "Unmodified initialization; all limiters disabled. Only ComputeCoefficients and RecoverPointValues called; no RHS or time advancement."},
            {"nodes", nodes}, {"faces", faces}};
        std::ofstream output(argv[2]);
        DNDS_check_throw_info(output.good(), "Cannot open shared-gradient replay export JSON");
        output << result.dump() << '\n';
        std::cout << "Exported " << nodes.size() << " nodes and " << faces.size()
                  << " faces to " << argv[2] << std::endl;
    }
    catch (const std::exception &error)
    {
        std::cerr << "shared-gradient replay export: " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    MPI_Finalize();
}
