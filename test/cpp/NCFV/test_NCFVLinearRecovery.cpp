/**
 * @file test_NCFVLinearRecovery.cpp
 * @brief P1 specialization of thesis (3-59), independent quadrature and MPI locality checks.
 */
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "NCFV/NCFVSolver.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>

namespace
{
    using namespace DNDS;
    using namespace DNDS::NCFV;
    MPIInfo gMPI;

    std::filesystem::path ProjectRoot()
    {
        return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    }

    Configuration ConfigurationForTest()
    {
        Configuration configuration;
        configuration.dimension = 3;
        configuration.mesh.meshFile = (ProjectRoot() / "data/mesh/ACMVariable_verify3D.cgns").string();
        configuration.algorithm.mode = IntegrationMode::EfficientDifferential;
        configuration.algorithm.retainMicroGeometry = true;
        configuration.reconstruction.enableLimiter = false;
        configuration.physics.initialPrimitive = {1, .1, 0, 0, 1};
        configuration.physics.farFieldPrimitive = configuration.physics.initialPrimitive;
        configuration.physics.requireBoundaryZoneCoverage = true;
        BoundaryZoneSettings far;
        far.name = "FAR";
        far.mode = BoundaryMode::FarField;
        far.primitive = configuration.physics.initialPrimitive;
        configuration.physics.boundaryZones = {far};
        configuration.time.iterations = 0;
        configuration.io.writeInitial = false;
        configuration.io.writeFinal = false;
        configuration.io.writeFinalRestart = false;
        configuration.io.writeResolvedConfiguration = false;
        configuration.Validate();
        return configuration;
    }

    template <class Function>
    void Quadrature(const NodeControlVolume &volume, Function &&function)
    {
        const DNDS::real a = (5 + 3 * std::sqrt(5.0)) / 20;
        const DNDS::real b = (5 - std::sqrt(5.0)) / 20;
        for (const auto &tet : volume.microVolumes)
            for (int p = 0; p < 4; p++)
            {
                Vector3 x = Vector3::Zero();
                for (int j = 0; j < 4; j++)
                    x += (p == j ? a : b) * tet.points[j];
                function(x, tet.measure / 4);
            }
    }
}

TEST_CASE("NCFV P1 point recovery removes the quadratic constant before contraction")
{
    // On [-1/2,1/2]^2, u=0.9+1.2*x^2 has mean 1 and zero gradient at the anchor.
    // The P2 point is 0.9, while the P1 zero-mean truncation has point 1.
    const Eigen::Vector2d mean{1, 3};
    const Eigen::Matrix<double, 2, 2> gradient = Eigen::Matrix<double, 2, 2>::Zero();
    const Vector3 weight = Vector3::Zero();
    const Eigen::Vector2d oldPoint{.9, 2.8};
    const auto recovered = Reconstruction::RecoverLinearPointValue(mean, gradient, weight, .5);
    CHECK((recovered - mean).norm() == 0);
    CHECK((mean + .5 * (oldPoint - mean) - recovered).norm() > .1);
    CHECK_THROWS(static_cast<void>(Reconstruction::RecoverLinearPointValue(mean, gradient, weight, -0.1)));
    CHECK_THROWS(static_cast<void>(Reconstruction::RecoverLinearPointValue(mean, gradient, weight, 1.1)));
    CHECK_THROWS(static_cast<void>(Reconstruction::RecoverLinearPointValue(mean, gradient, weight,
                                                        std::numeric_limits<double>::quiet_NaN())));

    // Nonzero first moments in 2-D, including a third coordinate that must be ignored.
    Eigen::Matrix<double, 2, 2> slope;
    slope << .3, -.7, .8, .2;
    const std::array<Vector3, 3> vertices{Vector3{0, 0, 2}, Vector3{2, 0, 2}, Vector3{0, 3, 2}};
    const Vector3 centroid = (vertices[0] + vertices[1] + vertices[2]) / 3;
    for (double theta : {0., .37, 1.})
    {
        const auto point = Reconstruction::RecoverLinearPointValue(mean, slope, centroid, theta);
        Eigen::Vector2d integrated = Eigen::Vector2d::Zero();
        for (const auto &x : vertices)
            integrated += (point + theta * slope.transpose() * x.head<2>()) / 3;
        CHECK((integrated - mean).norm() < 1e-14);
        const auto fromLimitedGradient = Reconstruction::RecoverLinearPointValue(mean, theta * slope, centroid);
        CHECK((fromLimitedGradient - point).norm() < 1e-14);
    }
}

TEST_CASE("NCFV P1 Barth bounds use R1 and gradient as one increment")
{
    const Eigen::Vector2d mean{10, 20}, lower{9, 18}, upper{10.1, 21};
    Eigen::Matrix2d gradient;
    gradient << 1, 2, 0, 0;
    const Vector3 weight{1, 0, 0};
    const std::vector<Vector3> offsets{Vector3{.5, 0, 0}, Vector3{1.5, 0, 0},
                                       Vector3{1, .5, 0}, Vector3{1, -.5, 0}};
    const auto limited = LinearBarthLimiter::LimitWithWeight(mean, gradient, weight, offsets, lower, upper);
    CHECK(limited.factor == doctest::Approx(.2).epsilon(1e-13));
    CHECK((limited.linearConstant - Eigen::Vector2d{-1, -2}).norm() == 0);
    for (const auto &offset : offsets)
    {
        const Eigen::Vector2d trace = limited.point + limited.gradient.transpose() * offset.head<2>();
        CHECK((trace.array() >= lower.array() - 1e-13).all());
        CHECK((trace.array() <= upper.array() + 1e-13).all());
    }
    CHECK((limited.point + limited.gradient.transpose() * weight.head<2>() - mean).norm() < 1e-13);
    // R2=-0.4 masks the upper violation if the old point is used to find theta.
    const Eigen::Vector2d oldP2Point{8.6, 18};
    for (const auto &offset : offsets)
    {
        const Eigen::Vector2d trace = oldP2Point + gradient.transpose() * offset.head<2>();
        CHECK((trace.array() >= lower.array() - 1e-13).all());
        CHECK((trace.array() <= upper.array() + 1e-13).all());
    }
    const auto recoveredAfterWrongFactor = Reconstruction::RecoverLinearPointValue(mean, gradient, weight, 1);
    CHECK((recoveredAfterWrongFactor + gradient.transpose() * offsets[1].head<2>())(0) > upper(0) + .3);
}

TEST_CASE("NCFV cached P1 weights preserve means on sheared cells without neighbor gradients")
{
    Configuration configuration = ConfigurationForTest();
    Solver<3> original(gMPI, configuration);
    original.Initialize();
    CHECK_FALSE(original.ReconstructionData().LinearLimiter().Enabled());
    CHECK(original.ReconstructionData().LinearLimiter().WeightStorageBytes() == 0);
    configuration.reconstruction.enableLimiter = true;
    const auto mesh = original.Mesh();
    Matrix3 affine;
    affine << .7, .21, .03, .04, 1.3, .17, .02, .06, .45;
    for (DNDS::index i = 0; i < mesh->NumNodeProc(); i++)
        mesh->coords[i] = (affine * mesh->coords[i] + Vector3{.3, -.2, .7}).eval();
    const auto &topology = original.EdgeTopology();

    Eigen::Matrix<double, 3, 2> linear;
    linear << .2, -.1, -.3, .4, .7, -.2;
    std::array<Matrix3, 2> hessian;
    hessian[0] << .8, .11, -.09, .11, .5, .13, -.09, .13, .6;
    hessian[1] << .4, -.12, .07, -.12, .7, .16, .07, .16, .9;
    const auto value = [&](const Vector3 &x) -> Eigen::Vector2d
    {
        Eigen::Vector2d result = Eigen::Vector2d{2, 4} + linear.transpose() * x;
        for (int v = 0; v < 2; v++) result(v) += .5 * x.dot(hessian[v] * x);
        return result;
    };
    const auto gradient = [&](const Vector3 &x) -> Eigen::Matrix<double, 3, 2>
    {
        Eigen::Matrix<double, 3, 2> result = linear;
        for (int v = 0; v < 2; v++) result.col(v) += hessian[v] * x;
        return result;
    };

    for (IntegrationMode mode : {IntegrationMode::EfficientDifferential, IntegrationMode::TraditionalQuadrature})
    {
        configuration.algorithm.mode = mode;
        configuration.algorithm.retainMicroGeometry = true;
        DualGeometry retained(gMPI, mesh, topology, configuration.algorithm);
        retained.Build();
        configuration.algorithm.retainMicroGeometry = false;
        DualGeometry geometry(gMPI, mesh, topology, configuration.algorithm);
        geometry.Build();
        auto dependencies = geometry.CollectNodeDependencies();
        NodeHalo halo(gMPI, mesh, configuration.mesh);
        halo.BuildPreliminary(geometry, configuration.reconstruction.maximumStencilRings, dependencies);
        Reconstruction reconstruction(gMPI, mesh, topology, geometry, mode, configuration.reconstruction, halo);
        reconstruction.Build();
        const auto reconstructionDependencies = reconstruction.CollectNodeDependencies();
        dependencies.insert(dependencies.end(), reconstructionDependencies.begin(), reconstructionDependencies.end());
        halo.Finalize(geometry, std::move(dependencies));
        geometry.RemapNodeIndices([&](DNDS::index node) { return halo.MeshLocalToLocal(node); });
        reconstruction.RemapNodeIndices();

        NodeStatePair means, points, p2Points;
        const auto allocateState = [&](NodeStatePair &field, const std::string &name)
        {
            field.InitPair(name, gMPI);
            field.father->Resize(mesh->NumNode(), 2, 1);
            field.son->Resize(halo.NumNodeGhost(), 2, 1);
            field.BorrowSetup(halo.Layout());
        };
        allocateState(means, "NCFV.P1.means");
        allocateState(points, "NCFV.P1.points");
        allocateState(p2Points, "NCFV.P1.p2Points");
        NodeMatrixPair gradients, unusedCoefficients;
        gradients.InitPair("NCFV.P1.gradients", gMPI);
        gradients.father->Resize(mesh->NumNode(), 3, 2);
        gradients.son->Resize(halo.NumNodeGhost(), 3, 2);
        gradients.BorrowSetup(halo.Layout());
        for (DNDS::index i = 0; i < halo.NumNodeProc(); i++)
            gradients[i] = gradient(halo.Coordinate(i));
        std::vector<Vector3> independentWeights(static_cast<std::size_t>(mesh->NumNode()));
        DNDS::real localSelfWeightDifference = 0, localWeightError = 0;
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            const auto &volume = retained.NodeVolume(i);
            const Vector3 anchor = mesh->coords[i];
            Eigen::Vector2d mean = Eigen::Vector2d::Zero();
            Vector3 first = Vector3::Zero();
            DNDS::real measure = 0;
            Quadrature(volume, [&](const Vector3 &x, DNDS::real w)
            {
                mean += w * value(x);
                first += w * (x - anchor);
                measure += w;
            });
            means[i] = mean / measure;
            independentWeights[i] = first / measure;
            const auto &cached = geometry.NodeVolume(i);
            const auto &weight = reconstruction.LinearLimiter().Weight(i);
            CHECK(cached.microVolumes.empty());
            localWeightError = std::max(localWeightError, (weight - first / measure).norm());
            if (mode == IntegrationMode::EfficientDifferential)
            {
                Vector3 sum = Vector3::Zero(), self = Vector3::Zero();
                for (const auto &entry : cached.pointRecoveryStencil)
                {
                    sum += entry.gradientWeight;
                    if (entry.node == i) self = entry.gradientWeight;
                }
                CHECK((sum - weight).norm() < 2e-13);
                localSelfWeightDifference = std::max(localSelfWeightDifference, (self - weight).norm());
            }
        }
        if (mode == IntegrationMode::EfficientDifferential)
        {
            reconstruction.RecoverPointValues(means, gradients, unusedCoefficients, p2Points);
            for (DNDS::index i = 0; i < mesh->NumNode(); i++)
                CHECK((p2Points[i] - value(mesh->coords[i])).norm() < 2e-12);
            DNDS::real globalDifference = 0;
            MPI_Allreduce(&localSelfWeightDifference, &globalDifference, 1, DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
            CHECK(globalDifference > 1e-4);
        }

        // P1 must not use ghost gradients, even when the old P2 recovery does.
        for (DNDS::index i = mesh->NumNode(); i < halo.NumNodeProc(); i++)
            gradients[i].setConstant(std::numeric_limits<double>::quiet_NaN());
        std::vector<DNDS::real> factors(static_cast<std::size_t>(mesh->NumNode()), .37);
        reconstruction.RecoverLinearPointValues(means, gradients, points, factors);
        DNDS::real localMeanError = 0, localPointError = 0, localWrongFormulaError = 0;
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            const Vector3 anchor = mesh->coords[i];
            const auto rawGradient = gradient(anchor);
            const Eigen::Vector2d expected = means[i] - .37 * rawGradient.transpose() * independentWeights[i];
            localPointError = std::max(localPointError, (points[i] - expected).norm());
            Eigen::Vector2d average = Eigen::Vector2d::Zero();
            DNDS::real measure = 0;
            Quadrature(retained.NodeVolume(i), [&](const Vector3 &x, DNDS::real w)
            {
                average += w * (points[i] + .37 * rawGradient.transpose() * (x - anchor));
                measure += w;
            });
            localMeanError = std::max(localMeanError, (average / measure - means[i]).norm());
            CHECK((gradients[i] - rawGradient).norm() == 0);
            const Eigen::Vector2d wrong = means[i] + .37 * (value(anchor) - means[i]);
            localWrongFormulaError = std::max(localWrongFormulaError, (wrong - points[i]).norm());
        }
        DNDS::real local[4]{localWeightError, localMeanError, localPointError, localWrongFormulaError}, global[4]{};
        MPI_Allreduce(local, global, 4, DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        CHECK(global[0] < 2e-13);
        CHECK(global[1] < 2e-12);
        CHECK(global[2] < 2e-13);
        CHECK(global[3] > 1e-4);
        if (gMPI.rank == 0)
            std::cout << "P1_RECOVERY mode=" << static_cast<int>(mode) << " ranks=" << gMPI.size
                      << " weight_error=" << global[0] << " mean_error=" << global[1]
                      << " point_error=" << global[2] << " old_P2_formula_error=" << global[3] << std::endl;

        std::vector<Eigen::Vector2d> before;
        for (DNDS::index i = 0; i < mesh->NumNode(); i++) before.emplace_back(points[i]);
        DNDS::index selected = UnInitIndex;
        if (gMPI.rank == 0)
            for (DNDS::index i = 0; i < mesh->NumNode(); i++)
                if ((gradient(mesh->coords[i]).transpose() * independentWeights[i]).norm() > 1e-4)
                {
                    selected = i;
                    factors[i] = .73;
                    break;
                }
        reconstruction.RecoverLinearPointValues(means, gradients, points, factors);
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
            if (gMPI.rank != 0 || i != selected) CHECK((points[i] - before[i]).norm() == 0);
            else CHECK((points[i] - before[i]).norm() > 1e-5);
        if (gMPI.rank == 0) CHECK(selected != UnInitIndex);
        factors.assign(static_cast<std::size_t>(mesh->NumNode()), 0);
        reconstruction.RecoverLinearPointValues(means, gradients, points, factors);
        for (DNDS::index i = 0; i < mesh->NumNode(); i++) CHECK((points[i] - means[i]).norm() == 0);
        reconstruction.RecoverLinearPointValues(means, gradients, points);
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
            CHECK((points[i] - means[i] + gradient(mesh->coords[i]).transpose() * independentWeights[i]).norm() < 2e-13);
        LinearBarthLimiter released;
        released.Initialize(geometry, true);
        CHECK(released.WeightCount() == static_cast<std::size_t>(mesh->NumNode()));
        released.Initialize(geometry, false);
        CHECK(released.WeightStorageBytes() == 0);
    }
}

TEST_CASE("NCFV P1 recovery weights use periodic local coordinates")
{
    Configuration configuration = ConfigurationForTest();
    configuration.mesh.meshFile = (ProjectRoot() / "cases/ncfv_euler/3D/generated_periodic_meshes/periodic_hex_iv10.cgns").string();
    configuration.mesh.periodicLengths = {10, 10, 4};
    configuration.mesh.periodicBoundaryPairs = {"bc-2", "bc-2-1", "bc-3", "bc-3-1", "bc-4", "bc-4-1"};
    configuration.physics.boundaryZones.clear();
    for (const auto &name : configuration.mesh.periodicBoundaryPairs)
    {
        BoundaryZoneSettings boundary;
        boundary.name = name;
        boundary.mode = BoundaryMode::Periodic;
        configuration.physics.boundaryZones.push_back(boundary);
    }
    configuration.algorithm.retainMicroGeometry = false;
    configuration.reconstruction.enableLimiter = true;
    configuration.reconstruction.maximumStencilRings = 4;
    Solver<3> solver(gMPI, configuration);
    solver.Initialize();
    for (const auto &volume : solver.Geometry().NodeVolumes())
    {
        CHECK(volume.microVolumes.empty());
        CHECK(solver.ReconstructionData().LinearLimiter().Weight(volume.node).norm() < 1e-12);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    gMPI.setWorld();
    doctest::Context context(argc, argv);
    const int localResult = context.run();
    int globalResult = 0;
    MPI_Allreduce(&localResult, &globalResult, 1, MPI_INT, MPI_MAX, gMPI.comm);
    MPI_Finalize();
    return globalResult;
}
