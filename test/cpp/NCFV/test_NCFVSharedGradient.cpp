#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "NCFV/NCFVSolver.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>

namespace
{
    using namespace DNDS;
    using namespace DNDS::NCFV;
    MPIInfo gMPI;
    using State = Eigen::Vector<double, 5>;
    using Gradient = Eigen::Matrix<double, 3, 5>;

    std::filesystem::path Root()
    {
        return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    }

    Configuration MakeConfiguration()
    {
        Configuration c;
        c.dimension = 3;
        c.mesh.meshFile = (Root() / "cases/ncfv_euler/3D/generated_periodic_meshes/periodic_hex_iv10.cgns").string();
        c.mesh.periodicLengths = {0, 0, 4};
        c.mesh.periodicBoundaryPairs = {"bc-4", "bc-4-1"};
        c.algorithm.mode = IntegrationMode::EfficientDifferential;
        c.reconstruction.enableLimiter = true;
        c.physics.initialPrimitive = {1, 0, 0, 0, 1};
        c.physics.farFieldPrimitive = c.physics.initialPrimitive;
        c.physics.boundaryMode = BoundaryMode::SupersonicOutlet;
        c.physics.requireBoundaryZoneCoverage = true;
        for (const std::string name : {"bc-2", "bc-2-1", "bc-3", "bc-3-1", "bc-4", "bc-4-1"})
        {
            BoundaryZoneSettings zone;
            zone.name = name;
            zone.mode = name.find("bc-4") == 0 ? BoundaryMode::Periodic : BoundaryMode::SupersonicOutlet;
            c.physics.boundaryZones.push_back(zone);
        }
        c.time.iterations = 0;
        c.io.writeVTK = false;
        c.io.writeInitial = false;
        c.io.writeFinal = false;
        c.io.writeFinalRestart = false;
        c.io.writeResolvedConfiguration = false;
        c.Validate();
        return c;
    }

    nlohmann::json AuditJson(const SharedGradientFaceSweep<3>::Audit &a)
    {
        return {{"minimum_density", a.minimumDensity}, {"minimum_pressure", a.minimumPressure},
                {"maximum_conservative_excess", a.maximumConservativeExcess},
                {"maximum_pressure_excess", a.maximumPressureExcess},
                {"nonphysical_sides", a.nonphysicalSides}, {"out_of_bounds_sides", a.outOfBoundsSides}};
    }
}

TEST_CASE("Full-support groups arbitrate remote conflicts and multiply every face factor once")
{
    // All ranks have a face whose extra dependency is node zero on rank zero.
    // Looking only at face endpoints would miss this deliberately remote conflict.
    const DNDS::index owned = 3;
    std::vector<MPI_int> owners;
    std::vector<DNDS::index> ownerRows;
    if (gMPI.rank != 0)
    {
        owners.push_back(0);
        ownerRows.push_back(0);
    }
    const DNDS::index shared = gMPI.rank == 0 ? 0 : owned;
    std::vector<std::vector<DNDS::index>> supports{{1, 2, shared}, {0, 1}};
    for (auto &support : supports)
    {
        std::sort(support.begin(), support.end());
        support.erase(std::unique(support.begin(), support.end()), support.end());
    }
    // Also exercise a rank with no faces but authoritative nodes and ghost subscribers.
    if (gMPI.size > 2 && gMPI.rank == gMPI.size - 1)
        supports.clear();
    SharedGradientSchedule schedule(gMPI);
    schedule.Initialize(owned, owners, ownerRows, supports);
    std::vector<double> localProduct(owned * gMPI.size, 1), expected(localProduct.size());
    for (std::size_t face = 0; face < supports.size(); face++)
        for (DNDS::index row : supports[face])
            localProduct[row < owned ? owned * gMPI.rank + row : 0] *= .7 + .05 * face;
    MPI_Allreduce(localProduct.data(), expected.data(), static_cast<int>(expected.size()), DNDS_MPI_REAL, MPI_PROD, gMPI.comm);
    std::vector<double> values(owned, 1), ghostFactors(owners.size(), 1);
    for (int color = 0; color < schedule.ColorCount(); color++)
    {
        std::vector<int> occupancy(expected.size()), globalOccupancy(expected.size());
        std::fill(ghostFactors.begin(), ghostFactors.end(), 1);
        for (std::size_t face : schedule.Faces(color))
            for (DNDS::index row : supports[face])
            {
                occupancy[row < owned ? owned * gMPI.rank + row : 0]++;
                if (row < owned)
                    values[row] *= .7 + .05 * face;
                else
                    ghostFactors[row - owned] *= .7 + .05 * face;
            }
        MPI_Allreduce(occupancy.data(), globalOccupancy.data(), static_cast<int>(occupancy.size()), MPI_INT, MPI_SUM, gMPI.comm);
        CHECK(*std::max_element(globalOccupancy.begin(), globalOccupancy.end()) <= 1);
        schedule.ExchangeFactors(color, ghostFactors, [&](DNDS::index row, double factor) { values[row] *= factor; });
    }
    for (DNDS::index row = 0; row < owned; row++)
        CHECK(values[row] == doctest::Approx(expected[owned * gMPI.rank + row]).epsilon(2e-14));
}

TEST_CASE("Colored shared gradients match a serial color reference on physical periodic stencils")
{
    const auto configuration = MakeConfiguration();
    Solver<3> solver(gMPI, configuration);
    solver.Initialize();
    const auto &mesh = solver.Mesh();
    const auto &halo = solver.NodeCommunication();
    NodeStatePair means;
    means.InitPair("NCFV.test.coloredMeans", gMPI);
    means.father->Resize(mesh->NumNode(), 5, 1);
    means.son->Resize(halo.NumNodeGhost(), 5, 1);
    means.BorrowSetup(halo.Layout());
    means.trans.initPersistentPull();
    NodeMatrixPair gradients, coefficients;
    for (auto *field : {&gradients, &coefficients})
    {
        field->InitPair(field == &gradients ? "NCFV.test.coloredGradients" : "NCFV.test.coloredCoefficients", gMPI);
        field->father->Resize(mesh->NumNode(), field == &gradients ? 3 : 9, 5);
        field->son->Resize(halo.NumNodeGhost(), field == &gradients ? 3 : 9, 5);
        field->BorrowSetup(halo.Layout());
        field->trans.initPersistentPull();
    }
    SharedGradientFaceSweep<3> sweep(gMPI, mesh, solver.EdgeTopology(), solver.Geometry(), halo);
    sweep.Initialize(true);
    sweep.EnableEventAudit();
    SharedGradientFaceSweep<3> legacy(gMPI, mesh, solver.EdgeTopology(), solver.Geometry(), halo);
    legacy.Initialize();
    legacy.EnableEventAudit();
    DNDS::index globalNodes = 0, localNodes = mesh->NumNode();
    MPI_Allreduce(&localNodes, &globalNodes, 1, DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
    nlohmann::json reports = nlohmann::json::array();
    for (int kind = 0; kind < 4; kind++)
    {
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
        {
            const auto x = mesh->coords[node];
            const bool high = kind == 1 ? (x(0) < 5 && x(1) < 5) : mesh->node2nodeOrig(node, 0) % 5 == 0;
            const double rho = kind == 3 ? 1 + .1 * std::sin(.5*x(0) + .2*x(1)) + .1 * std::cos(.5*std::acos(-1.)*x(2))
                                        : kind == 0 ? 1 : high ? 2 : .2;
            const double pressure = kind == 3 ? 1.2 + .1 * std::cos(x(0)) * std::cos(.5*std::acos(-1.)*x(2))
                                             : kind == 0 ? 1 : high ? 3 : .05;
            const Vector3 velocity = kind == 2 ? Vector3(.1, -.2, .15) : Vector3(.1, -.2, 0);
            means[node] << rho, rho * velocity, pressure / .4 + .5 * rho * velocity.squaredNorm();
        }
        means.trans.startPersistentPull();
        means.trans.waitPersistentPull();
        solver.ReconstructionData().ComputeCoefficients(means, gradients, coefficients);
        gradients.trans.startPersistentPull();
        gradients.trans.waitPersistentPull();
        std::vector<Gradient> raw(mesh->NumNode()), parallel(mesh->NumNode());
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
            raw[node] = gradients[node];
        sweep.Apply(means, gradients, 1.4, kind == 2);
        const auto event = sweep.EventAudit();
        const auto final = sweep.FinalAudit(1.4);
        CHECK(event.nonphysicalSides == 0);
        CHECK(event.maximumConservativeExcess <= 1.1e-10);
        if (kind == 2)
            CHECK(event.maximumPressureExcess <= 1.1e-10);
        const auto records = sweep.FaceRecords();
        DNDS::index localFractional = 0, fractional = 0;
        std::vector<double> localProduct(globalNodes, 1), product(globalNodes);
        const int colors = sweep.LastDiagnostics().colorGroups;
        std::vector<int> occupancy(globalNodes * colors), globalOccupancy(occupancy.size());
        for (const auto &face : records)
        {
            CHECK(face.factor >= 0);
            CHECK(face.factor <= 1);
            localFractional += face.factor > 1e-12 && face.factor < 1 - 1e-12;
            for (DNDS::index node : face.globalSupport)
            {
                localProduct[node] *= face.factor;
                occupancy[face.color * globalNodes + node]++;
            }
        }
        MPI_Allreduce(localProduct.data(), product.data(), static_cast<int>(globalNodes), DNDS_MPI_REAL, MPI_PROD, gMPI.comm);
        MPI_Allreduce(&localFractional, &fractional, 1, DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
        if (kind == 3)
            CHECK(fractional > 0);
        MPI_Allreduce(occupancy.data(), globalOccupancy.data(), static_cast<int>(occupancy.size()), MPI_INT, MPI_SUM, gMPI.comm);
        CHECK(*std::max_element(globalOccupancy.begin(), globalOccupancy.end()) <= 1);
        double localErrors[4]{};
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
        {
            const double factor = sweep.OwnedFactor(node);
            localErrors[0] = std::max(localErrors[0], std::abs(factor - product[halo.LocalToGlobal(node)]));
            localErrors[1] = std::max(localErrors[1], (gradients[node] - product[halo.LocalToGlobal(node)] * raw[node]).cwiseAbs().maxCoeff());
            localErrors[2] = std::max(localErrors[2], raw[node].row(2).cwiseAbs().maxCoeff());
            localErrors[3] = std::max(localErrors[3], gradients[node].row(2).cwiseAbs().maxCoeff());
            parallel[node] = gradients[node];
            gradients[node] = raw[node];
        }
        double errors[4]{};
        MPI_Allreduce(localErrors, errors, 4, DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        CHECK(errors[0] < 2e-14);
        CHECK(errors[1] < 2e-12);
        sweep.Apply(means, gradients, 1.4, kind == 2, true);
        double localDifference = 0, difference = 0;
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
        {
            localDifference = std::max(localDifference, (gradients[node] - parallel[node]).cwiseAbs().maxCoeff());
            gradients[node] = raw[node];
        }
        MPI_Allreduce(&localDifference, &difference, 1, DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        CHECK(difference < 2e-12);
        legacy.Apply(means, gradients, 1.4, kind == 2);
        const auto legacyFinal = legacy.FinalAudit(1.4);
        if (kind == 0)
        {
            CHECK(sweep.LastDiagnostics().updatedFaces == 0);
            CHECK(final.outOfBoundsSides == 0);
            CHECK(final.nonphysicalSides == 0);
        }
        reports.push_back({{"field_kind", kind}, {"mpi_ranks", gMPI.size}, {"colors", colors},
                           {"factor_product_error", errors[0]}, {"gradient_product_error", errors[1]},
                           {"raw_maximum_z_derivative", errors[2]}, {"limited_maximum_z_derivative", errors[3]},
                           {"parallel_serial_color_difference", difference}, {"event", AuditJson(event)},
                           {"fractional_face_factors", fractional},
                           {"final", AuditJson(final)}, {"legacy_final", AuditJson(legacyFinal)}});
    }
    if (gMPI.rank == 0)
    {
        std::cout << "SHARED_GRADIENT_AUDIT " << reports.dump() << std::endl;
        if (const char *path = std::getenv("DNDS_SHARED_GRADIENT_AUDIT"))
            std::ofstream(path) << reports.dump(2) << '\n';
    }
}

TEST_CASE("Frozen shared-gradient budgets bound all traces after independent contractions")
{
    const auto c = MakeConfiguration();
    Solver<3> solver(gMPI, c);
    solver.Initialize();
    const auto &mesh = solver.Mesh();
    const auto &halo = solver.NodeCommunication();
    NodeStatePair means, points;
    for (auto *field : {&means, &points})
    {
        field->InitPair(field == &means ? "NCFV.test.budgetMeans" : "NCFV.test.budgetPoints", gMPI);
        field->father->Resize(mesh->NumNode(), 5, 1);
        field->son->Resize(halo.NumNodeGhost(), 5, 1);
        field->BorrowSetup(halo.Layout());
        field->trans.initPersistentPull();
    }
    NodeMatrixPair gradients, coefficients;
    for (auto *field : {&gradients, &coefficients})
    {
        field->InitPair(field == &gradients ? "NCFV.test.budgetGradients" : "NCFV.test.budgetCoefficients", gMPI);
        field->father->Resize(mesh->NumNode(), field == &gradients ? 3 : 9, 5);
        field->son->Resize(halo.NumNodeGhost(), field == &gradients ? 3 : 9, 5);
        field->BorrowSetup(halo.Layout());
        field->trans.initPersistentPull();
    }
    for (DNDS::index node = 0; node < mesh->NumNode(); node++)
    {
        const auto id = mesh->node2nodeOrig(node, 0);
        const auto x = mesh->coords[node];
        const bool smoothDiagnostic = std::getenv("DNDS_BUDGET_CONSTRAINT_EXPORT") != nullptr;
        const double rho = smoothDiagnostic ? 1 + .1 * std::sin(.7 * x(0) + .3 * x(1)) + .05 * std::cos(1.57079632679 * x(2))
                                            : id % 5 == 0 ? 2 : .2;
        const double p = smoothDiagnostic ? 1 + .1 * std::cos(.4 * x(0)) * std::sin(.3 * x(1) + .5 * x(2))
                                          : id % 7 == 0 ? 3 : .05;
        const Vector3 v(.2, -.1, .15);
        means[node] << rho, rho * v, p / .4 + .5 * rho * v.squaredNorm();
    }
    means.trans.startPersistentPull();
    means.trans.waitPersistentPull();
    solver.ReconstructionData().ComputeCoefficients(means, gradients, coefficients);
    SharedGradientFaceSweep<3> budget(gMPI, mesh, solver.EdgeTopology(), solver.Geometry(), halo);
    budget.Initialize(true, true);
    budget.Apply(means, gradients, 1.4);
    const auto audit = budget.FinalAudit(1.4);
    if (const char *directory = std::getenv("DNDS_BUDGET_CONSTRAINT_EXPORT"))
        std::ofstream(std::filesystem::path(directory) / ("constraints_rank" + std::to_string(gMPI.rank) + ".json"))
            << budget.ConstraintSystem().dump() << '\n';
    CHECK(audit.nonphysicalSides == 0);
    CHECK(audit.maximumConservativeExcess <= 1.1e-10);
    CHECK(budget.LastDiagnostics().rankBlocks == 0);
    CHECK(budget.LastDiagnostics().colorGroups == 0);
    for (DNDS::index node = 0; node < mesh->NumNode(); node++)
    {
        CHECK(budget.OwnedFactor(node) >= 0);
        CHECK(budget.OwnedFactor(node) <= 1);
        gradients[node] *= (mesh->node2nodeOrig(node, 0) % 13) / 13.;
    }
    gradients.trans.startPersistentPull();
    gradients.trans.waitPersistentPull();
    solver.ReconstructionData().RecoverPointValues(means, gradients, coefficients, points);
    const auto check = [&](const State &candidate, const State &minimum, const State &maximum)
    {
        CHECK(candidate.allFinite());
        CHECK((candidate - maximum).maxCoeff() <= 1.1e-10);
        CHECK((minimum - candidate).maxCoeff() <= 1.1e-10);
        CHECK(candidate(0) > 1e-12);
        CHECK(.4 * (candidate(4) - .5 * candidate.segment<3>(1).squaredNorm() / candidate(0)) > 1e-12);
    };
    for (DNDS::index node = 0; node < mesh->NumNode(); node++)
    {
        State minimum = means[node], maximum = means[node];
        for (const auto &incidence : solver.EdgeTopology().Node2Edge(node))
        {
            const auto &s = solver.Geometry().EdgeSurface(incidence.edge);
            const State other = means[s.nodes[s.nodes[0] == node ? 1 : 0]];
            minimum = minimum.cwiseMin(other);
            maximum = maximum.cwiseMax(other);
        }
        check(State(points[node]), minimum, maximum);
        for (const auto &incidence : solver.EdgeTopology().Node2Edge(node))
        {
            const auto &s = solver.Geometry().EdgeSurface(incidence.edge);
            const int side = s.nodes[0] == node ? 0 : 1;
            State candidate = points[node];
            for (const auto &entry : s.efficientStencil)
                candidate += gradients[entry.node].transpose() * entry.stateGradientWeights[side].head<3>() / s.measure;
            check(candidate, minimum, maximum);
        }
    }
    if (gMPI.rank == 0)
        std::cout << "BUDGET_AUDIT " << AuditJson(audit).dump() << std::endl;
}

TEST_CASE("Complete binary and continuous flux limiting enforce physical entropy bar states")
{
    auto c = MakeConfiguration();
    c.reconstruction.enableLimiter = false;
    c.reconstruction.limitFluxCorrection = true;
    c.reconstruction.fluxCorrectionEntropy = true;
    c.time.useCFLTimeStep = true;
    c.time.useLocalTimeStep = false;
    c.algorithm.checkGeometryClosure = true;
    c.time.cfl = .5;
    c.Validate();
    Solver<3> solver(gMPI, c);
    solver.Initialize();
    const auto &mesh = solver.Mesh();
    const auto &halo = solver.NodeCommunication();
    NodeStatePair means, rhs;
    for (auto *field : {&means, &rhs})
    {
        field->InitPair(field == &means ? "NCFV.test.idpMeans" : "NCFV.test.idpRhs", gMPI);
        field->father->Resize(mesh->NumNode(), 5, 1);
        field->son->Resize(halo.NumNodeGhost(), 5, 1);
        field->BorrowSetup(halo.Layout());
        field->trans.initPersistentPull();
    }
    for (DNDS::index node = 0; node < mesh->NumNode(); node++)
    {
        const bool high = mesh->node2nodeOrig(node, 0) % 7 == 0;
        const double rho = high ? 2 : .2, pressure = high ? 3 : .05;
        const Vector3 v(.2, -.1, .15);
        means[node] << rho, rho * v, pressure / .4 + .5 * rho * v.squaredNorm();
    }
    for (bool binary : {false, true})
    {
        c.reconstruction.fluxCorrectionBinary = binary;
        SpatialOperator<3> spatial(gMPI, mesh, solver.EdgeTopology(), solver.Geometry(), solver.ReconstructionData(),
                                   solver.Boundaries(), c.algorithm.mode, c.reconstruction, c.physics, c.time, halo);
        spatial.Initialize();
        CHECK(std::isfinite(spatial.EvaluateRHS(means, rhs)));
        for (DNDS::index edge = 0; edge < solver.EdgeTopology().NumEdge(); edge++)
        {
            const auto &s = solver.Geometry().EdgeSurface(edge);
            const State left = means[s.nodes[0]], right = means[s.nodes[1]];
            const Vector3 normal = s.vectorMeasure / s.measure;
            const auto pressure = [](const State &u) { return .4 * (u(4) - .5 * u.segment<3>(1).squaredNorm() / u(0)); };
            const auto flux = [&](const State &u)
            {
                const double vn = u.segment<3>(1).dot(normal) / u(0);
                State f = vn * u;
                f.segment<3>(1) += pressure(u) * normal;
                f(4) += pressure(u) * vn;
                return f;
            };
            const double speed = std::max(std::abs(left.segment<3>(1).dot(normal) / left(0)) + std::sqrt(1.4 * pressure(left) / left(0)),
                                          std::abs(right.segment<3>(1).dot(normal) / right(0)) + std::sqrt(1.4 * pressure(right) / right(0)));
            const State bar = .5 * (left + right) - .5 / speed * (flux(right) - flux(left));
            const State low = .5 * s.measure * (flux(left) + flux(right) - speed * (right - left));
            const State correction = State(spatial.EdgeFluxes()[edge]) - low;
            const double factor = spatial.EdgeFluxCorrectionFactors()[edge](0);
            CHECK(factor >= 0);
            CHECK(factor <= 1);
            if (binary) CHECK((factor == 0 || factor == 1));
            for (int side = 0; side < 2; side++)
            {
                const State candidate = bar + (side == 0 ? -1. : 1.) * correction / (speed * s.measure);
                const auto bounds = spatial.FluxCorrectionBounds()[s.nodes[side]];
                CHECK(candidate(0) > 1e-12);
                CHECK(pressure(candidate) > 1e-12);
                CHECK((candidate - bounds.topRows<5>().col(1)).maxCoeff() <= 1e-10);
                CHECK((bounds.topRows<5>().col(0) - candidate).maxCoeff() <= 1e-10);
                CHECK(pressure(candidate) >= bounds(6, 0) * std::pow(candidate(0), 1.4) - 1e-10);
            }
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    gMPI.setWorld();
    doctest::Context context(argc, argv);
    const int localResult = context.run();
    int result = 0;
    MPI_Allreduce(&localResult, &result, 1, MPI_INT, MPI_MAX, gMPI.comm);
    MPI_Finalize();
    return result;
}
