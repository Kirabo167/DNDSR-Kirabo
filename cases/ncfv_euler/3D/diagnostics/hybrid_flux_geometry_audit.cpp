/** Read-only macro-area/normal audit, using the same native Roe dispatcher. */
#include "NCFV/NCFVSolver.hpp"
#include <fstream>
#include <iomanip>

using namespace DNDS::NCFV;
using State = Eigen::Vector<double, 5>;

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        DNDS_check_throw_info(argc == 3 && mpi.size == 1, "usage: base-config output.json (serial audit)");
        std::ifstream base(argv[1]);
        nlohmann::ordered_json resolved = Configuration{};
        resolved.merge_patch(nlohmann::ordered_json::parse(base));
        for (auto &boundary : resolved["physics"]["boundaryZones"])
        {
            nlohmann::ordered_json normalized = BoundaryZoneSettings{};
            normalized.merge_patch(boundary);
            boundary = std::move(normalized);
        }
        auto cfg = resolved.get<Configuration>();
        cfg.algorithm.mode = IntegrationMode::TraditionalQuadrature;
        cfg.algorithm.retainMicroGeometry = false;
        cfg.algorithm.quadratureOrder = 4;
        cfg.algorithm.surfaceQuadratureOrder = 3;
        cfg.reconstruction.method = ReconstructionMethod::LeastSquares;
        cfg.reconstruction.enableLimiter = false;
        cfg.reconstruction.stencilSizeFactor = std::nextafter(13.0 / 9.0, 0.0);
        cfg.physics.riemannSolver = DNDS::Euler::Gas::Roe;
        cfg.physics.viscous.enabled = false;
        cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
        cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
        Solver<3> solver(mpi, cfg);
        solver.Initialize();
        SpatialOperator<3> spatial(mpi, solver.Mesh(), solver.EdgeTopology(), solver.Geometry(),
            solver.ReconstructionData(), solver.Boundaries(), cfg.algorithm.mode,
            cfg.reconstruction, cfg.physics, cfg.time, solver.NodeCommunication());
        spatial.Initialize();
        const Eigen::Vector3d velocity{1, 1, 0};
        const auto state = [&](double rho)
        {
            State u;
            u << rho, rho, rho, 0, 1 / (cfg.physics.gamma - 1) + rho;
            return u;
        };
        const State left = state(0.9), right = state(1.1);
        const auto massCentral = [&](const Eigen::Vector3d &normal)
        {
            return 0.5 * (left.segment<3>(1) + right.segment<3>(1)).dot(normal);
        };
        double sumArea = 0, sumProjected = 0;
        double minRatio = 1e100, maxRatio = 0, maxIdentity = 0;
        double sumTraditionalD = 0, sumEfficientD = 0;
        DNDS::index nonplanar = 0, under = 0, over = 0;
        std::size_t minQ = 1000000, maxQ = 0;
        for (DNDS::index e = 0; e < solver.EdgeTopology().NumEdge(); e++)
        {
            const auto &surface = solver.Geometry().EdgeSurface(e);
            const auto normal = surface.vectorMeasure.normalized().eval();
            const double projected = surface.vectorMeasure.norm();
            const double ratio = surface.measure / projected;
            sumArea += surface.measure;
            sumProjected += projected;
            minRatio = std::min(minRatio, ratio);
            maxRatio = std::max(maxRatio, ratio);
            nonplanar += ratio > 1 + 1e-10;
            minQ = std::min(minQ, surface.quadrature.size());
            maxQ = std::max(maxQ, surface.quadrature.size());
            double integratedD = 0, scalarContact = 0;
            for (const auto &q : surface.quadrature)
            {
                const auto n = (q.vectorWeight / q.weight).eval();
                integratedD += q.weight * (spatial.EvaluateNumericalFluxForDiagnostics(left, right, n)(0) - massCentral(n));
                scalarContact += q.weight * std::abs(velocity.dot(n));
            }
            const double macroD = surface.measure *
                (spatial.EvaluateNumericalFluxForDiagnostics(left, right, normal)(0) - massCentral(normal));
            maxIdentity = std::max(maxIdentity, std::abs(integratedD + 0.1 * scalarContact));
            maxIdentity = std::max(maxIdentity, std::abs(macroD + 0.1 * surface.measure * std::abs(velocity.dot(normal))));
            sumTraditionalD += std::abs(integratedD);
            sumEfficientD += std::abs(macroD);
            over += std::abs(macroD) > std::abs(integratedD) + 1e-12;
            under += std::abs(macroD) < std::abs(integratedD) - 1e-12;
        }
        nlohmann::ordered_json result = {
            {"mesh", cfg.mesh.meshFile}, {"macro_edges", solver.EdgeTopology().NumEdge()},
            {"area_over_vector_measure_min", minRatio}, {"area_over_vector_measure_max", maxRatio},
            {"sum_area_over_sum_vector_measure", sumArea / sumProjected},
            {"nonplanar_macro_edges", nonplanar}, {"quadrature_points_per_macro_min", minQ},
            {"quadrature_points_per_macro_max", maxQ}, {"constant_jump_left_rho", 0.9},
            {"constant_jump_right_rho", 1.1}, {"constant_jump_pressure", 1}, {"velocity", {1, 1, 0}},
            {"traditional_sum_abs_mass_dissipation", sumTraditionalD},
            {"macro_sum_abs_mass_dissipation", sumEfficientD},
            {"macro_over_traditional_dissipation_ratio", sumEfficientD / sumTraditionalD},
            {"macro_overdissipative_edges", over}, {"macro_underdissipative_edges", under},
            {"pure_contact_identity_max_error", maxIdentity},
            {"interpretation", "Constant traces isolate geometry/normal aggregation; not a globally evolving manufactured solution."}};
        DNDS_check_throw_info(maxIdentity < 1e-11, "Native Roe contact identity failed");
        DNDS_check_throw_info(!std::filesystem::exists(argv[2]), "Audit output already exists");
        std::ofstream output(argv[2]);
        output << std::setw(2) << result << '\n';
    }
    MPI_Finalize();
}
