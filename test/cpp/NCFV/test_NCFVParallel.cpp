/**
 * @file test_NCFVParallel.cpp
 * @brief MPI integration test for both standalone NCFV algorithms.
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

    struct QuadraticField
    {
        DNDS::real constant = 0;
        Vector3 linear = Vector3::Zero();
        Matrix3 hessian = Matrix3::Zero();

        [[nodiscard]] DNDS::real Value(const Vector3 &coordinate) const
        {
            return constant + linear.dot(coordinate) +
                   0.5 * coordinate.dot(hessian * coordinate);
        }

        [[nodiscard]] Vector3 Gradient(const Vector3 &coordinate) const
        {
            return linear + hessian * coordinate;
        }

        [[nodiscard]] DNDS::real Integral(const RawMoments &moments) const
        {
            return constant * moments.measure + linear.dot(moments.first) +
                   0.5 * (hessian.cwiseProduct(moments.second)).sum();
        }
    };

    std::filesystem::path ProjectRoot()
    {
        return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    }

    Configuration MakeConfiguration(IntegrationMode mode)
    {
        Configuration configuration;
        configuration.dimension = 3;
        configuration.mesh.meshFile =
            (ProjectRoot() / "data/mesh/ACMVariable_verify3D.cgns").string();
        configuration.algorithm.mode = mode;
        configuration.algorithm.quadratureOrder = 4;
        configuration.algorithm.retainMicroGeometry = true;
        configuration.physics.initialPrimitive = {1.0, 0.1, 0.0, 0.0, 1.0};
        configuration.physics.farFieldPrimitive = {1.0, 0.1, 0.0, 0.0, 1.0};
        configuration.physics.requireBoundaryZoneCoverage = true;
        BoundaryZoneSettings farField;
        farField.name = "FAR";
        farField.mode = BoundaryMode::FarField;
        farField.primitive = configuration.physics.farFieldPrimitive;
        configuration.physics.boundaryZones = {farField};
        configuration.physics.viscous.enabled = true;
        configuration.physics.viscous.model = ViscosityModel::Constant;
        configuration.physics.viscous.dynamicViscosity = 1e-3;
        configuration.time.iterations = 0;
        configuration.time.useCFLTimeStep = true;
        configuration.time.useLocalTimeStep = true;
        configuration.time.cfl = 0.5;
        configuration.Validate();
        return configuration;
    }

    Configuration MakeExactPeriodicConfiguration(IntegrationMode mode)
    {
        Configuration configuration = MakeConfiguration(mode);
        configuration.mesh.meshFile =
            (ProjectRoot() / "cases/NCFV/generated_periodic_meshes/periodic_hex_iv10.cgns").string();
        configuration.mesh.periodicLengths = {10, 10, 4};
        configuration.mesh.periodicBoundaryPairs = {
            "bc-2", "bc-2-1", "bc-3", "bc-3-1", "bc-4", "bc-4-1"};
        configuration.algorithm.retainMicroGeometry = false;
        configuration.reconstruction.enableLimiter = false;
        configuration.reconstruction.maximumStencilRings = 4;
        configuration.physics.initialPrimitive = {1.0, 0.2, -0.1, 0.05, 1.0};
        configuration.physics.farFieldPrimitive = configuration.physics.initialPrimitive;
        configuration.physics.viscous.enabled = false;
        configuration.physics.boundaryZones.clear();
        for (const std::string &name : configuration.mesh.periodicBoundaryPairs)
        {
            BoundaryZoneSettings periodic;
            periodic.name = name;
            periodic.mode = BoundaryMode::Periodic;
            configuration.physics.boundaryZones.push_back(periodic);
        }
        configuration.time.iterations = 1;
        configuration.time.useCFLTimeStep = true;
        configuration.time.useLocalTimeStep = false;
        configuration.time.cfl = 0.1;
        configuration.time.maximumTimeStep = 1e-3;
        configuration.Validate();
        return configuration;
    }

    void VerifyConstructionAverages(const Solver<3> &solver)
    {
        const auto &mesh = solver.Mesh();
        const auto &topology = solver.EdgeTopology();
        const auto &geometry = solver.Geometry();
        DNDS::real localMaximumError = 0;

        for (DNDS::index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            const auto cell = mesh->GetCellElement(iCell);
            Vector3 expected = Vector3::Zero();
            for (int iNode = 0; iNode < cell.GetNumVertices(); iNode++)
                expected += mesh->coords[mesh->cell2node(iCell, iNode)];
            expected /= static_cast<DNDS::real>(cell.GetNumVertices());
            localMaximumError = std::max(
                localMaximumError,
                (expected - geometry.CellConstructionPoints()[static_cast<std::size_t>(iCell)])
                    .norm());
        }

        for (DNDS::index iFace = 0; iFace < topology.NumFaceProc(); iFace++)
        {
            const auto face = topology.GetFaceElement(iFace);
            Vector3 expected = Vector3::Zero();
            for (int iNode = 0; iNode < face.GetNumVertices(); iNode++)
                expected += mesh->coords[topology.Face2Node()(iFace, iNode)];
            expected /= static_cast<DNDS::real>(face.GetNumVertices());
            localMaximumError = std::max(
                localMaximumError,
                (expected - geometry.FaceConstructionPoints()[static_cast<std::size_t>(iFace)])
                    .norm());
        }

        for (DNDS::index iEdge = 0; iEdge < topology.NumEdgeProc(); iEdge++)
        {
            const Vector3 expected = 0.5 *
                                     (mesh->coords[topology.Edge2Node()(iEdge, 0)] +
                                      mesh->coords[topology.Edge2Node()(iEdge, 1)]);
            localMaximumError = std::max(
                localMaximumError,
                (expected - geometry.EdgeConstructionPoints()[static_cast<std::size_t>(iEdge)])
                    .norm());
        }

        DNDS::real globalMaximumError = 0;
        MPI_Allreduce(&localMaximumError, &globalMaximumError, 1,
                      DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        CHECK(globalMaximumError < 2e-14);
    }

    void VerifyEfficientGradientWeights(const Solver<3> &solver)
    {
        DNDS::real localMaximumError = 0;
        DNDS::index localWeightSets = 0;
        for (const auto &surface : solver.Geometry().EdgeSurfaces())
        {
            DNDS::real sum = 0;
            for (const auto &entry : surface.efficientStencil)
                sum += entry.gradientWeight;
            localMaximumError = std::max(
                localMaximumError,
                std::abs(sum - surface.measure) / surface.measure);
            localWeightSets += !surface.efficientStencil.empty();
        }
        for (const auto &volume : solver.Geometry().NodeVolumes())
            for (const auto &piece : volume.boundaryPieces)
            {
                DNDS::real sum = 0;
                for (const auto &entry : piece.efficientStencil)
                    sum += entry.integrationWeight;
                localMaximumError = std::max(
                    localMaximumError,
                    std::abs(sum - piece.measure) / piece.measure);
                localWeightSets += !piece.efficientStencil.empty();
            }

        DNDS::real globalMaximumError = 0;
        DNDS::index globalWeightSets = 0;
        MPI_Allreduce(&localMaximumError, &globalMaximumError, 1,
                      DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        MPI_Allreduce(&localWeightSets, &globalWeightSets, 1,
                      DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
        CHECK(globalWeightSets > 0);
        CHECK(globalMaximumError < 2e-14);
    }

    void VerifyEfficientStencilLayout(const Solver<3> &solver)
    {
        const auto &mesh = solver.Mesh();
        const NodeHalo &nodeHalo = solver.NodeCommunication();
        const DNDS::index processNodeCount = nodeHalo.NumNodeProc();
        DNDS::real localMaximumMomentError = 0;
        DNDS::index localInvalidOrdering = 0;
        DNDS::index localGhostReferences = 0;
        DNDS::index localStencilEntries = 0;

        const auto auditNode = [&](DNDS::index node, DNDS::index &previous)
        {
            localInvalidOrdering +=
                node < 0 || node >= processNodeCount ||
                (previous != UnInitIndex && node <= previous);
            localGhostReferences += node >= mesh->NumNode();
            localStencilEntries++;
            previous = node;
        };

        for (const auto &volume : solver.Geometry().NodeVolumes())
        {
            DNDS::index previous = UnInitIndex;
            for (const auto &entry : volume.pointRecoveryStencil)
                auditNode(entry.node, previous);
            for (const auto &piece : volume.boundaryPieces)
            {
                previous = UnInitIndex;
                DNDS::real valueSum = 0;
                DNDS::real gradientSum = 0;
                for (const EfficientBoundaryNode &entry : piece.efficientStencil)
                {
                    auditNode(entry.node, previous);
                    valueSum += entry.integrationWeight;
                    gradientSum += entry.integrationWeight;
                }
                localMaximumMomentError = std::max(
                    localMaximumMomentError,
                    std::abs(valueSum - piece.measure) / piece.measure);
                localMaximumMomentError = std::max(
                    localMaximumMomentError,
                    std::abs(gradientSum - piece.measure) / piece.measure);
            }
        }

        for (const auto &surface : solver.Geometry().EdgeSurfaces())
        {
            DNDS::index previous = UnInitIndex;
            for (const EfficientSurfaceNode &entry : surface.efficientStencil)
            {
                auditNode(entry.node, previous);
            }
            for (DNDS::index anchor : surface.nodes)
                localInvalidOrdering +=
                    anchor < 0 || anchor >= processNodeCount;
            localInvalidOrdering +=
                !(surface.measure > 0) || !surface.vectorMeasure.allFinite();
        }

        DNDS::real globalMaximumMomentError = 0;
        DNDS::index localCounts[3]{
            localInvalidOrdering, localGhostReferences, localStencilEntries};
        DNDS::index globalCounts[3]{};
        MPI_Allreduce(&localMaximumMomentError, &globalMaximumMomentError, 1,
                      DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        MPI_Allreduce(localCounts, globalCounts, 3,
                      DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
        CHECK(globalMaximumMomentError < 2e-14);
        CHECK(globalCounts[0] == 0);
        CHECK(globalCounts[2] > 0);
        if (gMPI.size > 1)
            CHECK(globalCounts[1] > 0);
    }

    void VerifyEfficientPolynomialWeights(const Solver<3> &solver)
    {
        const auto &mesh = solver.Mesh();
        const auto &geometry = solver.Geometry();
        const NodeHalo &nodeHalo = solver.NodeCommunication();
        const auto coordinate = [&](DNDS::index node) -> Vector3
        {
            return nodeHalo.Coordinate(node);
        };

        QuadraticField state;
        state.constant = 0.73;
        state.linear = Vector3{0.31, -0.47, 0.29};
        state.hessian << 0.8, -0.2, 0.13,
            -0.2, -0.5, 0.17,
            0.13, 0.17, 0.4;

        std::array<QuadraticField, 3> fluxes;
        for (int f = 0; f < 3; f++)
        {
            fluxes[static_cast<std::size_t>(f)].constant = 0.2 + 0.3 * f;
            fluxes[static_cast<std::size_t>(f)].linear =
                Vector3{0.11 + 0.07 * f, -0.23 + 0.05 * f, 0.37 - 0.09 * f};
            Matrix3 &hessian = fluxes[static_cast<std::size_t>(f)].hessian;
            hessian << 0.4 + 0.1 * f, -0.08, 0.03 + 0.02 * f,
                -0.08, -0.3 + 0.04 * f, 0.06,
                0.03 + 0.02 * f, 0.06, 0.2 - 0.03 * f;
        }

        DNDS::real localPointError = 0;
        DNDS::real localStateMeanError = 0;
        DNDS::real localFluxError = 0;
        DNDS::index localSurfaceCount = 0;

        for (DNDS::index iNode = 0; iNode < mesh->NumNode(); iNode++)
        {
            const auto &volume = geometry.NodeVolume(iNode);
            const DNDS::real exactIntegral = state.Integral(volume.moments);
            DNDS::real recovered = exactIntegral / volume.moments.measure;
            for (const auto &entry : volume.pointRecoveryStencil)
                recovered -= entry.gradientWeight.dot(
                    state.Gradient(coordinate(entry.node)));
            localPointError = std::max(
                localPointError,
                std::abs(recovered - state.Value(mesh->coords[iNode])) /
                    std::max<DNDS::real>(1.0, std::abs(state.Value(mesh->coords[iNode]))));
        }

        for (const auto &surface : geometry.EdgeSurfaces())
        {
            DNDS_check_throw_info(!surface.microSurfaces.empty(),
                                  "NCFV polynomial-weight test requires retained micro surfaces");
            DNDS::real exactStateIntegral = 0;
            DNDS::real exactFluxIntegral = 0;
            for (const auto &micro : surface.microSurfaces)
            {
                std::vector<Vector3> coordinates;
                coordinates.reserve(static_cast<std::size_t>(micro.nPoints));
                for (int p = 0; p < micro.nPoints; p++)
                    coordinates.push_back(micro.points[static_cast<std::size_t>(p)]);
                const RawMoments moments =
                    DualGeometry::ExactSimplexMoments(coordinates, micro.measure);
                exactStateIntegral += state.Integral(moments);
                const Vector3 unitNormal = micro.vectorMeasure / micro.measure;
                for (int f = 0; f < 3; f++)
                    exactFluxIntegral += unitNormal(f) *
                                         fluxes[static_cast<std::size_t>(f)].Integral(moments);
            }

            const auto stateIntegral = [&](int side)
            {
                const std::size_t sideIndex = static_cast<std::size_t>(side);
                const DNDS::index anchor = surface.nodes[sideIndex];
                DNDS::real integral = surface.measure *
                                      state.Value(coordinate(anchor));
                for (const EfficientSurfaceNode &entry : surface.efficientStencil)
                {
                    integral += entry.stateGradientWeights[sideIndex].dot(
                        state.Gradient(coordinate(entry.node)));
                }
                return integral;
            };
            const auto fluxIntegral = [&](int side)
            {
                const std::size_t sideIndex = static_cast<std::size_t>(side);
                const DNDS::index anchor = surface.nodes[sideIndex];
                Vector3 anchorFlux;
                for (int f = 0; f < 3; f++)
                    anchorFlux(f) = fluxes[static_cast<std::size_t>(f)].Value(
                        coordinate(anchor));
                DNDS::real integral = anchorFlux.dot(
                    surface.vectorMeasure);
                for (const EfficientSurfaceNode &entry : surface.efficientStencil)
                {
                    for (int d = 0; d < 3; d++)
                        for (int f = 0; f < 3; f++)
                            integral += entry.fluxGradientWeights[sideIndex](d, f) *
                                        fluxes[static_cast<std::size_t>(f)]
                                            .Gradient(coordinate(entry.node))(d);
                }
                return integral;
            };

            for (int side = 0; side < 2; side++)
            {
                localStateMeanError = std::max(
                    localStateMeanError,
                    std::abs(stateIntegral(side) - exactStateIntegral) /
                        std::max<DNDS::real>(1.0, std::abs(exactStateIntegral)));
                localFluxError = std::max(
                    localFluxError,
                    std::abs(fluxIntegral(side) - exactFluxIntegral) /
                        std::max<DNDS::real>(1.0, std::abs(exactFluxIntegral)));
            }
            localSurfaceCount++;
        }

        DNDS::real localErrors[3]{
            localPointError, localStateMeanError, localFluxError};
        DNDS::real globalErrors[3]{};
        DNDS::index globalSurfaceCount = 0;
        MPI_Allreduce(localErrors, globalErrors, 3,
                      DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        MPI_Allreduce(&localSurfaceCount, &globalSurfaceCount, 1,
                      DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
        CHECK(globalSurfaceCount > 0);
        CHECK(globalErrors[0] < 3e-13);
        CHECK(globalErrors[1] < 3e-13);
        CHECK(globalErrors[2] < 3e-13);
    }

    void VerifyEfficientLimiterUsesMacroSurfaceMeans(const Solver<3> &solver)
    {
        const auto &mesh = solver.Mesh();
        const auto &topology = solver.EdgeTopology();
        const auto &geometry = solver.Geometry();
        const NodeHalo &nodeHalo = solver.NodeCommunication();
        const DNDS::index ghostCount = nodeHalo.NumNodeGhost();
        const auto coordinate = [&](DNDS::index node) -> Vector3
        {
            return nodeHalo.Coordinate(node);
        };

        NodeStatePair means, points;
        const auto allocateState = [&](NodeStatePair &field,
                                       const std::string &name)
        {
            field.InitPair(name, gMPI);
            field.father->Resize(mesh->NumNode(), 1, 1);
            field.son->Resize(ghostCount, 1, 1);
            field.BorrowSetup(nodeHalo.Layout());
            field.trans.initPersistentPull();
            for (DNDS::index iNode = 0; iNode < field.Size(); iNode++)
                field[iNode].setZero();
        };
        allocateState(means, "NCFV.test.limiterMeans");
        allocateState(points, "NCFV.test.limiterPoints");
        NodeMatrixPair gradients, coefficients;
        const auto allocate = [&](NodeMatrixPair &field,
                                  const std::string &name, int rows)
        {
            field.InitPair(name, gMPI);
            field.father->Resize(mesh->NumNode(), rows, 1);
            field.son->Resize(ghostCount, rows, 1);
            field.BorrowSetup(nodeHalo.Layout());
            field.trans.initPersistentPull();
            for (DNDS::index iNode = 0; iNode < field.Size(); iNode++)
                field[iNode].setZero();
        };
        allocate(gradients, "NCFV.test.limiterGradients", 3);
        allocate(coefficients, "NCFV.test.limiterCoefficients", 9);

        const Vector3 exactGradient{0.4, 0.7, 1.1};
        for (DNDS::index iNode = 0; iNode < mesh->NumNode(); iNode++)
        {
            const DNDS::real point = 2.0 + exactGradient.dot(mesh->coords[iNode]);
            points[iNode](0) = point;
            // Deliberately unrelated means make the old mean-bounded midpoint
            // implementation observably different from thesis (3-94)--(3-100).
            means[iNode](0) = point + 20.0 + 0.1 * mesh->coords[iNode].x();
            gradients[iNode].col(0) = 4.0 * exactGradient;
        }
        means.trans.startPersistentPull();
        points.trans.startPersistentPull();
        gradients.trans.startPersistentPull();
        means.trans.waitPersistentPull();
        points.trans.waitPersistentPull();
        gradients.trans.waitPersistentPull();

        const std::vector<DNDS::real> factors =
            solver.ReconstructionData().ComputeLimiterFactors(
                means, points, gradients, coefficients);
        DNDS::real localMaximumError = 0;
        DNDS::real localLegacyDifference = 0;
        DNDS::real localBoundViolation = 0;
        DNDS::index localNontrivial = 0;
        for (DNDS::index iNode = 0; iNode < mesh->NumNode(); iNode++)
        {
            DNDS::real expected = 1.0;
            DNDS::real legacy = 1.0;
            DNDS::real legacyMinimum = means[iNode](0);
            DNDS::real legacyMaximum = means[iNode](0);
            for (const auto &incidence : topology.Node2Edge(iNode))
            {
                const auto &surface = geometry.EdgeSurface(incidence.edge);
                const bool isLeft = surface.nodes[0] == iNode;
                const DNDS::index neighbor = surface.nodes[isLeft ? 1 : 0];
                legacyMinimum = std::min(legacyMinimum, means[neighbor](0));
                legacyMaximum = std::max(legacyMaximum, means[neighbor](0));
            }
            for (const auto &incidence : topology.Node2Edge(iNode))
            {
                const auto &surface = geometry.EdgeSurface(incidence.edge);
                const bool isLeft = surface.nodes[0] == iNode;
                const DNDS::index neighbor = surface.nodes[isLeft ? 1 : 0];
                const std::size_t side = isLeft ? 0 : 1;
                DNDS::real candidate =
                    surface.measure * points[iNode](0);
                for (const EfficientSurfaceNode &entry : surface.efficientStencil)
                {
                    candidate += entry.stateGradientWeights[side].dot(
                        gradients[entry.node].col(0));
                }
                candidate /= surface.measure;
                const DNDS::real increment = candidate - points[iNode](0);
                const DNDS::real minimum =
                    std::min(points[iNode](0), points[neighbor](0));
                const DNDS::real maximum =
                    std::max(points[iNode](0), points[neighbor](0));
                if (increment > verySmallReal)
                    expected = std::min(
                        expected, (maximum - points[iNode](0)) / increment);
                else if (increment < -verySmallReal)
                    expected = std::min(
                        expected, (minimum - points[iNode](0)) / increment);
                const DNDS::real limited =
                    points[iNode](0) + std::clamp(expected, 0.0, 1.0) * increment;
                localBoundViolation = std::max(
                    localBoundViolation,
                    std::max({minimum - limited, limited - maximum, 0.0}));

                const Vector3 midpoint =
                    0.5 * (mesh->coords[iNode] + coordinate(neighbor));
                const DNDS::real legacyCandidate =
                    points[iNode](0) + gradients[iNode].col(0).dot(
                                           midpoint - mesh->coords[iNode]);
                const DNDS::real legacyIncrement =
                    legacyCandidate - means[iNode](0);
                if (legacyIncrement > verySmallReal)
                    legacy = std::min(
                        legacy, (legacyMaximum - means[iNode](0)) /
                                    legacyIncrement);
                else if (legacyIncrement < -verySmallReal)
                    legacy = std::min(
                        legacy, (legacyMinimum - means[iNode](0)) /
                                    legacyIncrement);
            }
            expected = std::clamp(expected, 0.0, 1.0);
            legacy = std::clamp(legacy, 0.0, 1.0);
            localMaximumError = std::max(
                localMaximumError,
                std::abs(factors[static_cast<std::size_t>(iNode)] - expected));
            localLegacyDifference = std::max(
                localLegacyDifference, std::abs(expected - legacy));
            localNontrivial += expected > 1e-12 && expected < 1.0 - 1e-12;
        }

        DNDS::real localChecks[3]{
            localMaximumError, localLegacyDifference, localBoundViolation};
        DNDS::real globalChecks[3]{};
        DNDS::index globalNontrivial = 0;
        MPI_Allreduce(localChecks, globalChecks, 3,
                      DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        MPI_Allreduce(&localNontrivial, &globalNontrivial, 1,
                      DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
        CHECK(globalChecks[0] < 3e-14);
        CHECK(globalChecks[1] > 1e-3);
        CHECK(globalChecks[2] < 3e-14);
        CHECK(globalNontrivial > 0);
    }

    void VerifyMode(IntegrationMode mode)
    {
        Solver<3> solver(gMPI, MakeConfiguration(mode));
        solver.Initialize();
        VerifyConstructionAverages(solver);

        CHECK(solver.Mesh()->NumNodeGlobal() == 125);
        CHECK(solver.Mesh()->NumCellGlobal() == 64);
        CHECK(solver.EdgeTopology().NumEdgeGlobal() == 300);
        CHECK(solver.Geometry().MaximumClosureError() < 2e-13);

        DNDS::index localVolumeQuadrature = 0;
        DNDS::index localInternalSurfaceQuadrature = 0;
        DNDS::index localBoundaryQuadrature = 0;
        DNDS::index localMicroVolumes = 0;
        for (const auto &volume : solver.Geometry().NodeVolumes())
        {
            localVolumeQuadrature += static_cast<DNDS::index>(volume.volumeQuadrature.size());
            localMicroVolumes += static_cast<DNDS::index>(volume.microVolumes.size());
            for (const auto &piece : volume.boundaryPieces)
                localBoundaryQuadrature +=
                    static_cast<DNDS::index>(piece.quadrature.size());
        }
        for (const auto &surface : solver.Geometry().EdgeSurfaces())
            localInternalSurfaceQuadrature +=
                static_cast<DNDS::index>(surface.quadrature.size());

        DNDS::index localCounts[4]{
            localVolumeQuadrature,
            localInternalSurfaceQuadrature,
            localBoundaryQuadrature,
            localMicroVolumes};
        DNDS::index globalCounts[4]{};
        MPI_Allreduce(localCounts, globalCounts, 4,
                      DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
        CHECK(globalCounts[3] > 0);
        if (mode == IntegrationMode::EfficientDifferential)
        {
            CHECK(globalCounts[0] == 0);
            CHECK(globalCounts[1] == 0);
            CHECK(globalCounts[2] == 0);
            VerifyEfficientStencilLayout(solver);
            VerifyEfficientGradientWeights(solver);
            VerifyEfficientPolynomialWeights(solver);
            VerifyEfficientLimiterUsesMacroSurfaceMeans(solver);
        }
        else
        {
            CHECK(globalCounts[0] > 0);
            CHECK(globalCounts[1] > 0);
            CHECK(globalCounts[2] > 0);
        }

        const DNDS::real residual = solver.EvaluateResidual();
        CHECK(residual < 5e-13);

        DNDS::real localMinimumTimeStep = std::numeric_limits<DNDS::real>::max();
        DNDS::real localMaximumTimeStep = 0;
        for (DNDS::index iNode = 0; iNode < solver.Mesh()->NumNode(); iNode++)
        {
            const DNDS::real timeStep = solver.LocalTimeStep(iNode);
            CHECK(std::isfinite(timeStep));
            CHECK(timeStep > 0);
            localMinimumTimeStep = std::min(localMinimumTimeStep, timeStep);
            localMaximumTimeStep = std::max(localMaximumTimeStep, timeStep);
        }
        DNDS::real globalMinimumTimeStep = 0;
        DNDS::real globalMaximumTimeStep = 0;
        MPI_Allreduce(&localMinimumTimeStep, &globalMinimumTimeStep, 1,
                      DNDS_MPI_REAL, MPI_MIN, gMPI.comm);
        MPI_Allreduce(&localMaximumTimeStep, &globalMaximumTimeStep, 1,
                      DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        CHECK(globalMinimumTimeStep > 0);
        CHECK(globalMaximumTimeStep > globalMinimumTimeStep);
    }
}

TEST_CASE("NCFV 3-D efficient and traditional modes are MPI invariant")
{
    VerifyMode(IntegrationMode::EfficientDifferential);
    VerifyMode(IntegrationMode::TraditionalQuadrature);
}

TEST_CASE("NCFV exact sparse node halo preserves a truly paired periodic topology")
{
    for (IntegrationMode mode : {IntegrationMode::EfficientDifferential,
                                 IntegrationMode::TraditionalQuadrature})
    {
        Configuration configuration = MakeExactPeriodicConfiguration(mode);
        Solver<3> solver(gMPI, configuration);
        solver.Initialize();

        CHECK(solver.NodeCommunication().IsFinalized());
        CHECK(solver.Mesh()->isPeriodic);
        CHECK(solver.Mesh()->NumNodeGlobal() == 1000);
        CHECK(solver.Mesh()->NumCellGlobal() == 1000);
        CHECK(solver.EdgeTopology().NumEdgeGlobal() == 3000);
        CHECK(solver.Geometry().MaximumClosureError() < 2e-13);

        DNDS::index localHaloCounts[2]{
            solver.NodeCommunication().PreliminaryGhostCount(),
            solver.NodeCommunication().NumNodeGhost()};
        DNDS::index globalHaloCounts[2]{};
        MPI_Allreduce(localHaloCounts, globalHaloCounts, 2,
                      DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
        CHECK(globalHaloCounts[1] <= globalHaloCounts[0]);
        if (gMPI.size > 1)
            CHECK(globalHaloCounts[1] > 0);

        CHECK(solver.EvaluateResidual() < 2e-11);
        Eigen::Vector<double, 5> initial = solver.StateField()[0];
        solver.Run();
        DNDS::real localStateChange = 0;
        for (DNDS::index iNode = 0; iNode < solver.Mesh()->NumNode(); iNode++)
            localStateChange = std::max(
                localStateChange,
                (solver.StateField()[iNode] - initial).norm());
        DNDS::real globalStateChange = 0;
        MPI_Allreduce(&localStateChange, &globalStateChange, 1,
                      DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        CHECK(globalStateChange < 2e-12);

        configuration.initialField.expressions = {{{
            "inRegion := 1;",
            "UPrim[0] := 1 + 0.01 * sin(2*pi*x[0]/10);",
            "0;"}}};
        Solver<3> transported(gMPI, configuration);
        transported.Initialize();
        const auto totals = [&]()
        {
            Eigen::Vector<double, 5> local = Eigen::Vector<double, 5>::Zero();
            for (DNDS::index iNode = 0; iNode < transported.Mesh()->NumNode(); iNode++)
                local += transported.Geometry().NodeVolume(iNode).moments.measure *
                         transported.StateField()[iNode];
            Eigen::Vector<double, 5> global;
            MPI_Allreduce(local.data(), global.data(), 5,
                          DNDS_MPI_REAL, MPI_SUM, gMPI.comm);
            return global;
        };
        const auto initialTotals = totals();
        transported.Run();
        CHECK((totals() - initialTotals).norm() < 5e-11);
    }
}

TEST_CASE("NCFV directional normalization survives stretching, ghost exchange and geometry disposal")
{
    const Vector3 stretch{0.05, 2, 5};
    Configuration configuration = MakeConfiguration(IntegrationMode::EfficientDifferential);
    configuration.reconstruction.enableLimiter = false;
    Solver<3> original(gMPI, configuration);
    original.Initialize();
    const auto mesh = original.Mesh();
    for (DNDS::index i = 0; i < mesh->NumNodeProc(); i++)
        mesh->coords[i] = mesh->coords[i].cwiseProduct(stretch);
    const auto &topology = original.EdgeTopology();

    const Vector3 linear{0.2, -0.1, 0.3};
    Matrix3 hessian;
    hessian << 0.4, 0.1, -0.2, 0.1, -0.3, 0.15, -0.2, 0.15, 0.2;
    const auto value = [&](const Vector3 &x)
    {
        const Vector3 y = x.cwiseQuotient(stretch);
        return 3 + linear.dot(y) + 0.5 * y.dot(hessian * y);
    };
    const auto gradient = [&](const Vector3 &x) -> Vector3
    {
        return (linear + hessian * x.cwiseQuotient(stretch)).cwiseQuotient(stretch);
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
        const std::vector<DNDS::index> integrationDependencies =
            geometry.CollectNodeDependencies();
        NodeHalo nodeHalo(gMPI, mesh, configuration.mesh);
        nodeHalo.BuildPreliminary(
            geometry,
            configuration.reconstruction.maximumStencilRings,
            integrationDependencies);
        Reconstruction reconstruction(
            gMPI, mesh, topology, geometry, mode,
            configuration.reconstruction, nodeHalo);
        reconstruction.Build();
        std::vector<DNDS::index> runtimeDependencies = integrationDependencies;
        const std::vector<DNDS::index> reconstructionDependencies =
            reconstruction.CollectNodeDependencies();
        runtimeDependencies.insert(
            runtimeDependencies.end(),
            reconstructionDependencies.begin(),
            reconstructionDependencies.end());
        nodeHalo.Finalize(geometry, std::move(runtimeDependencies));
        geometry.RemapNodeIndices(
            [&](DNDS::index meshLocal)
            { return nodeHalo.MeshLocalToLocal(meshLocal); });
        reconstruction.RemapNodeIndices();

        NodeStatePair means, points;
        const auto allocateState = [&](NodeStatePair &field,
                                       const std::string &name)
        {
            field.InitPair(name, gMPI);
            field.father->Resize(mesh->NumNode(), 1, 1);
            field.son->Resize(nodeHalo.NumNodeGhost(), 1, 1);
            field.BorrowSetup(nodeHalo.Layout());
            field.trans.initPersistentPull();
            for (DNDS::index i = 0; i < field.Size(); i++)
                field[i].setZero();
        };
        allocateState(means, "NCFV.test.means");
        allocateState(points, "NCFV.test.points");
        NodeMatrixPair gradients, coefficients;
        const auto allocate = [&](NodeMatrixPair &field, const std::string &name, int rows)
        {
            field.InitPair(name, gMPI);
            field.father->Resize(mesh->NumNode(), rows, 1);
            field.son->Resize(nodeHalo.NumNodeGhost(), rows, 1);
            for (DNDS::index i = 0; i < nodeHalo.NumNodeProc(); i++)
                field[i].setZero();
            field.BorrowSetup(nodeHalo.Layout());
            field.trans.initPersistentPull();
        };
        allocate(gradients, "NCFV.test.gradients", 3);
        allocate(coefficients, "NCFV.test.coefficients", 9);
        const DNDS::real a = (5 + 3 * std::sqrt(5.0)) / 20;
        const DNDS::real b = (5 - std::sqrt(5.0)) / 20;
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            const auto &volume = retained.NodeVolume(i);
            Vector3 lower = Vector3::Constant(std::numeric_limits<DNDS::real>::max());
            Vector3 upper = -lower;
            DNDS::real integral = 0;
            for (const auto &tet : volume.microVolumes)
            {
                for (int p = 0; p < 4; p++)
                {
                    lower = lower.cwiseMin(tet.points[p]);
                    upper = upper.cwiseMax(tet.points[p]);
                    Vector3 x = Vector3::Zero();
                    for (int j = 0; j < 4; j++)
                        x += (p == j ? a : b) * tet.points[j];
                    integral += tet.measure * value(x) / 4;
                }
            }
            CHECK((geometry.ReferenceLengths(i) - 0.5 * (upper - lower)).norm() < 2e-13);
            CHECK((reconstruction.Operator(i).referenceLengths - geometry.ReferenceLengths(i)).norm() < 2e-13);
            CHECK(geometry.NodeVolume(i).microVolumes.empty());
            if (mode == IntegrationMode::EfficientDifferential)
                CHECK(geometry.NodeVolume(i).volumeQuadrature.empty());
            means[i](0) = integral / volume.moments.measure;
        }
        means.trans.startPersistentPull();
        means.trans.waitPersistentPull();
        reconstruction.ComputeCoefficients(means, gradients, coefficients);
        gradients.trans.startPersistentPull();
        gradients.trans.waitPersistentPull();
        coefficients.trans.startPersistentPull();
        coefficients.trans.waitPersistentPull();
        reconstruction.RecoverPointValues(means, gradients, coefficients, points);
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            CHECK(std::abs(points[i](0) - value(mesh->coords[i])) < 2e-10);
            if (mode == IntegrationMode::EfficientDifferential)
                CHECK((gradients[i].col(0) - gradient(mesh->coords[i])).norm() < 2e-9);
        }
        // Both owner and ghost evaluations use the owner's directional scales.
        for (DNDS::index i = 0; i < nodeHalo.NumNodeProc(); i++)
        {
            Vector3 expected = stretch / 8;
            const Vector3 coordinate = nodeHalo.Coordinate(i);
            for (int d = 0; d < 3; d++)
                if (std::abs(coordinate(d)) < 1e-12 ||
                    std::abs(coordinate(d) - stretch(d)) < 1e-12)
                    expected(d) *= 0.5;
            CHECK((nodeHalo.ReferenceLengths(i) - expected).norm() < 2e-13);
            if (mode == IntegrationMode::TraditionalQuadrature)
            {
                const Vector3 offset = stretch.cwiseProduct(Vector3{0.01, -0.02, 0.03});
                const Vector3 computed = Reconstruction::EvaluateBasisGradient(
                                             offset, reconstruction.ReferenceLengths(i), 3) *
                                         coefficients[i];
                CHECK((computed - gradient(coordinate + offset)).norm() < 2e-9);
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
    int globalResult = 0;
    MPI_Allreduce(&localResult, &globalResult, 1, MPI_INT, MPI_MAX, gMPI.comm);
    MPI_Finalize();
    return globalResult;
}
