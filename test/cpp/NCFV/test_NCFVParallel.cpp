/**
 * @file test_NCFVParallel.cpp
 * @brief MPI integration test for both standalone NCFV algorithms.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "NCFV/NCFVSolver.hpp"
#include "NCFV/NCFVPressureLimiter.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>

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

        [[nodiscard]] DNDS::real IntegralRelative(
            const RawMoments &moments,
            const Vector3 &anchor) const
        {
            return moments.measure * Value(anchor) +
                   Gradient(anchor).dot(moments.first) +
                   0.5 * (hessian.cwiseProduct(moments.second)).sum();
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
        configuration.algorithm.surfaceQuadratureOrder = 3;
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
            (ProjectRoot() / "cases/ncfv_euler/3D/generated_periodic_meshes/periodic_hex_iv10.cgns").string();
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

    void VerifyTraditionalSurfaceQuadrature(const Solver<3> &solver)
    {
        constexpr int surfaceOrder = 3;
        const Geom::Elem::Quadrature referenceRule(
            Geom::Elem::Element{Geom::Elem::Tri3}, surfaceOrder - 1);
        const int pointsPerTriangle = referenceRule.GetNumPoints();
        CHECK(pointsPerTriangle == 3);

        QuadraticField field;
        field.constant = 0.73;
        field.linear = Vector3{0.31, -0.47, 0.29};
        field.hessian << 0.8, -0.2, 0.13,
            -0.2, -0.5, 0.17,
            0.13, 0.17, 0.4;

        DNDS::real localMaximumError = 0;
        DNDS::index localInvalid = 0;
        DNDS::index localTriangleCount = 0;

        const auto verifyTriangle = [&](const std::vector<SurfaceQuadraturePoint> &quadrature,
                                        std::size_t offset,
                                        const std::vector<Vector3> &coordinates,
                                        DNDS::real measure,
                                        const Vector3 &vectorMeasure)
        {
            const RawMoments moments =
                DualGeometry::ExactSimplexMoments(coordinates, measure);
            const DNDS::real exactIntegral = field.Integral(moments);
            const Vector3 unitNormal = vectorMeasure / measure;
            DNDS::real numericalIntegral = 0;
            for (int iG = 0; iG < pointsPerTriangle; iG++)
            {
                const auto &point = quadrature[offset + static_cast<std::size_t>(iG)];
                localInvalid += !point.coordinate.allFinite() ||
                                !point.vectorWeight.allFinite() ||
                                !std::isfinite(point.weight) || point.weight <= 0;
                numericalIntegral += point.weight * field.Value(point.coordinate);
                localMaximumError = std::max(
                    localMaximumError,
                    (point.vectorWeight - point.weight * unitNormal).norm() /
                        std::max<DNDS::real>(point.weight, verySmallReal));
            }
            localMaximumError = std::max(
                localMaximumError,
                std::abs(numericalIntegral - exactIntegral) /
                    std::max<DNDS::real>(1.0, std::abs(exactIntegral)));
            localTriangleCount++;
        };

        for (const auto &surface : solver.Geometry().EdgeSurfaces())
        {
            const std::size_t expected =
                surface.microSurfaces.size() * static_cast<std::size_t>(pointsPerTriangle);
            localInvalid += surface.quadrature.size() != expected;
            if (surface.quadrature.size() != expected)
                continue;
            for (std::size_t iMicro = 0; iMicro < surface.microSurfaces.size(); iMicro++)
            {
                const auto &micro = surface.microSurfaces[iMicro];
                std::vector<Vector3> coordinates;
                coordinates.reserve(static_cast<std::size_t>(micro.nPoints));
                for (int point = 0; point < micro.nPoints; point++)
                    coordinates.push_back(micro.points[static_cast<std::size_t>(point)]);
                verifyTriangle(surface.quadrature,
                               iMicro * static_cast<std::size_t>(pointsPerTriangle),
                               coordinates, micro.measure, micro.vectorMeasure);
            }
        }

        for (const auto &volume : solver.Geometry().NodeVolumes())
            for (const auto &piece : volume.boundaryPieces)
            {
                localInvalid += piece.nPoints != 3 ||
                                piece.quadrature.size() !=
                                    static_cast<std::size_t>(pointsPerTriangle);
                if (piece.nPoints != 3 ||
                    piece.quadrature.size() != static_cast<std::size_t>(pointsPerTriangle))
                    continue;
                std::vector<Vector3> coordinates;
                coordinates.reserve(3);
                for (int point = 0; point < piece.nPoints; point++)
                    coordinates.push_back(
                        piece.points[static_cast<std::size_t>(point)].coordinate);
                verifyTriangle(piece.quadrature, 0, coordinates,
                               piece.measure, piece.vectorMeasure);
            }

        DNDS::real globalMaximumError = 0;
        DNDS::index localCounts[2]{localInvalid, localTriangleCount};
        DNDS::index globalCounts[2]{};
        MPI_Allreduce(&localMaximumError, &globalMaximumError, 1,
                      DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        MPI_Allreduce(localCounts, globalCounts, 2,
                      DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
        CHECK(globalCounts[0] == 0);
        CHECK(globalCounts[1] > 0);
        CHECK(globalMaximumError < 3e-13);
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
            const DNDS::real exactIntegral = state.IntegralRelative(
                volume.moments, mesh->coords[iNode]);
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

    void VerifyCompactQuadraticStencils(
        const Solver<3> &solver,
        const ReconstructionSettings &settings)
    {
        const int expected = static_cast<int>(std::ceil(
            settings.stencilSizeFactor *
            solver.ReconstructionData().BasisSize()));
        DNDS::index localInvalid = 0;
        for (DNDS::index iNode = 0;
             iNode < solver.Mesh()->NumNode(); iNode++)
        {
            const auto &op = solver.ReconstructionData().Operator(iNode);
            localInvalid +=
                static_cast<int>(op.stencil.size()) != expected ||
                op.numericalRank != solver.ReconstructionData().BasisSize() ||
                !std::isfinite(op.conditionNumber) ||
                op.conditionNumber > settings.maximumConditionNumber ||
                !std::isfinite(op.cubicErrorIndicator) ||
                op.directNeighborCount <= 0;
        }
        DNDS::index globalInvalid = 0;
        MPI_Allreduce(&localInvalid, &globalInvalid, 1,
                      DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
        CHECK(globalInvalid == 0);
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
            VerifyTraditionalSurfaceQuadrature(solver);
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

        VerifyCompactQuadraticStencils(
            solver, configuration.reconstruction);

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

TEST_CASE("NCFV fixed-size compact stencils preserve coordinate-plane fields")
{
    for (const auto mode : {IntegrationMode::EfficientDifferential,
                           IntegrationMode::TraditionalQuadrature})
    for (const auto method : {ReconstructionMethod::SVDLeastSquares,
                             ReconstructionMethod::LeastSquares})
    {
        auto configuration = MakeExactPeriodicConfiguration(mode);
        configuration.reconstruction.method = method;
        Solver<3> solver(gMPI, configuration);
        solver.Initialize();
        const auto &halo = solver.NodeCommunication();
        DNDS::real localMaximum[3]{};
        DNDS::index wrongSizes = 0;
        for (DNDS::index node = 0; node < solver.Mesh()->NumNode(); node++)
        {
            const auto &op = solver.ReconstructionData().Operator(node);
            wrongSizes += op.stencil.size() != 16;
            Eigen::MatrixXd matrix(op.stencil.size(), 9);
            std::vector<Vector3> directions;
            for (std::size_t j = 0; j < op.stencil.size(); j++)
            {
                matrix.row(j) = (Reconstruction::MeanBasis(
                                     halo.MomentsRelative(node, op.stencil[j]),
                                     Vector3::Zero(), op.referenceLengths, 3) -
                                 op.targetBasisMean).transpose();
                directions.push_back(halo.Displacement(node, op.stencil[j])
                                         .cwiseQuotient(op.referenceLengths).normalized());
            }
            // Includes z, xz, yz and z^2: genuine 3-D quadratic data must
            // retain their derivatives while arbitrary planar data cancel.
            localMaximum[0] = std::max(localMaximum[0],
                (op.inverseRows * matrix - Eigen::MatrixXd::Identity(op.inverseRows.rows(), 9)).cwiseAbs().maxCoeff());
            for (int normal = 0; normal < 3; normal++)
            {
                std::map<std::array<long long, 2>, DNDS::real> columnWeights;
                DNDS::real anchorWeight = 0;
                for (std::size_t j = 0; j < op.stencil.size(); j++)
                {
                    const Vector3 offset = halo.Displacement(node, op.stencil[j])
                                               .cwiseQuotient(op.referenceLengths);
                    const std::array<long long, 2> key{
                        std::llround(offset((normal + 1) % 3) * 1e9),
                        std::llround(offset((normal + 2) % 3) * 1e9)};
                    const DNDS::real weight = op.inverseRows(normal, j) / op.referenceLengths(normal);
                    columnWeights[key] += weight;
                    anchorWeight -= weight;
                }
                columnWeights[{0, 0}] += anchorWeight;
                for (const auto &[key, weight] : columnWeights)
                    localMaximum[1] = std::max(localMaximum[1], std::abs(weight));
            }
            for (int x = -1; x <= 1; x++)
                for (int y = -1; y <= 1; y++)
                    for (int z = -1; z <= 1; z++)
                        if (x || y || z)
                        {
                            const Vector3 probe = Vector3(x, y, z).normalized();
                            DNDS::real cosine = -1;
                            for (const auto &direction : directions)
                                cosine = std::max(cosine, probe.dot(direction));
                            localMaximum[2] = std::max(localMaximum[2], 1 - cosine);
                        }
        }
        DNDS::real maximum[3]{};
        DNDS::index globalWrongSizes = 0;
        MPI_Allreduce(localMaximum, maximum, 3, DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        MPI_Allreduce(&wrongSizes, &globalWrongSizes, 1, DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
        CHECK(globalWrongSizes == 0);
        CHECK(maximum[0] < 2e-10);
        CHECK(maximum[1] < 2e-10);
        CHECK(maximum[2] < .25);
    }
}

TEST_CASE("NCFV z-only periodic topology supports limited efficient reconstruction")
{
    auto configuration = MakeExactPeriodicConfiguration(IntegrationMode::EfficientDifferential);
    configuration.mesh.periodicLengths = {0, 0, 4};
    configuration.mesh.periodicBoundaryPairs = {"bc-4", "bc-4-1"};
    for (auto &zone : configuration.physics.boundaryZones)
        if (zone.name != "bc-4" && zone.name != "bc-4-1")
            zone.mode = BoundaryMode::SupersonicOutlet;
    configuration.reconstruction.enableLimiter = true;
    configuration.reconstruction.limitPointRecovery = true;
    configuration.physics.initialPrimitive = {1.0, 0, 0, 0.1, 1.0};
    configuration.physics.farFieldPrimitive = configuration.physics.initialPrimitive;
    configuration.Validate();
    Solver<3> solver(gMPI, configuration);
    solver.Initialize();
    CHECK(solver.Mesh()->isPeriodic);
    CHECK(solver.Mesh()->NumNodeGlobal() == 1210);
    CHECK(solver.Mesh()->NumCellGlobal() == 1000);
    CHECK(solver.EdgeTopology().NumEdgeGlobal() == 3410);
    CHECK(solver.Geometry().MaximumClosureError() < 2e-13);
    CHECK(solver.EvaluateResidual() < 2e-11);
    const Eigen::Vector<double, 5> initial = solver.StateField()[0];
    solver.Run();
    DNDS::real localChange = 0, globalChange = 0;
    for (DNDS::index i = 0; i < solver.Mesh()->NumNode(); i++)
        localChange = std::max(localChange, (solver.StateField()[i] - initial).norm());
    MPI_Allreduce(&localChange, &globalChange, 1, DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
    CHECK(globalChange < 2e-12);

    configuration.initialField.expressions = {{{
        "inRegion := 1;",
        "UPrim[0] := 1 + 0.01 * sin(2*pi*x[2]/4);",
        "0;"}}};
    Solver<3> transported(gMPI, configuration);
    transported.Initialize();
    const auto totals = [&]()
    {
        Eigen::Vector<double, 5> local = Eigen::Vector<double, 5>::Zero(), global;
        for (DNDS::index i = 0; i < transported.Mesh()->NumNode(); i++)
            local += transported.Geometry().NodeVolume(i).moments.measure * transported.StateField()[i];
        MPI_Allreduce(local.data(), global.data(), 5, DNDS_MPI_REAL, MPI_SUM, gMPI.comm);
        return global;
    };
    const auto initialTotals = totals();
    transported.Run();
    CHECK((totals() - initialTotals).norm() < 5e-11);
}

TEST_CASE("NCFV Roe SSPRK3 has third-order temporal self-convergence")
{
    for (const auto mode : {IntegrationMode::TraditionalQuadrature,
                            IntegrationMode::EfficientDifferential})
        for (const auto method : {ReconstructionMethod::LeastSquares,
                                  ReconstructionMethod::Variational})
        {
            std::vector<Eigen::MatrixXd> states;
            std::vector<DNDS::real> volumes;
            Eigen::MatrixXd coordinates;
            for (int refinement = 0; refinement < 3; refinement++)
            {
                auto configuration = MakeExactPeriodicConfiguration(mode);
                configuration.reconstruction.method = method;
                configuration.reconstruction.variationalWeight = 5;
                configuration.reconstruction.variationalIterations = 3;
                configuration.physics.initialPrimitive = {1, 1, 1, 0, 1};
                configuration.initialField.expressions = {{{
                    "inRegion := 1;",
                    "UPrim[0] := 1 + 0.1 * sin(2*pi*x[0]/10) * cos(2*pi*x[1]/10);",
                    "UPrim[1] := 1 + 0.05 * cos(2*pi*x[1]/10);",
                    "UPrim[4] := 1 + 0.05 * sin(2*pi*x[0]/10);",
                    "0;"}}};
                configuration.time.useCFLTimeStep = false;
                configuration.time.timeStep = 0.05 / (1 << refinement);
                configuration.time.maximumTimeStep = 1;
                configuration.time.endTime = 0.2;
                configuration.time.iterations = 4 * (1 << refinement);
                configuration.io.writeResolvedConfiguration = false;
                Solver<3> solver(gMPI, configuration);
                solver.Initialize();
                const auto count = solver.Mesh()->NumNode();
                if (refinement == 0)
                {
                    volumes.resize(count);
                    coordinates.resize(count, 3);
                    for (DNDS::index i = 0; i < count; i++)
                    {
                        volumes[i] = solver.Geometry().NodeVolume(i).moments.measure;
                        coordinates.row(i) = solver.Mesh()->coords[i].transpose();
                    }
                }
                REQUIRE(count == coordinates.rows());
                for (DNDS::index i = 0; i < count; i++)
                    REQUIRE((coordinates.row(i).transpose() - solver.Mesh()->coords[i]).norm() < 1e-13);
                solver.Run();
                CHECK(solver.SimulationTime() == doctest::Approx(0.2));
                Eigen::MatrixXd state(count, 5);
                for (DNDS::index i = 0; i < count; i++)
                    state.row(i) = solver.StateField()[i].transpose();
                states.push_back(std::move(state));
            }
            DNDS::real local[2]{}, global[2]{};
            for (DNDS::index i = 0; i < coordinates.rows(); i++)
                for (int pair = 0; pair < 2; pair++)
                    local[pair] += volumes[i] *
                        (states[pair].row(i) - states[pair + 1].row(i)).squaredNorm();
            MPI_Allreduce(local, global, 2, DNDS_MPI_REAL, MPI_SUM, gMPI.comm);
            REQUIRE(global[1] > 1e-24);
            const DNDS::real order = 0.5 * std::log2(global[0] / global[1]);
            if (gMPI.rank == 0)
                std::cout << "NCFV temporal order mode=" << static_cast<int>(mode)
                          << " method=" << static_cast<int>(method)
                          << " order=" << order << std::endl;
            CHECK(order > 2.8);
            CHECK(order < 3.3);
        }
}

TEST_CASE("NCFV directional normalization survives stretching, ghost exchange and geometry disposal")
{
    const Vector3 stretch{0.05, 2, 5};
    Configuration configuration = MakeConfiguration(IntegrationMode::EfficientDifferential);
    SUBCASE("compact stencil")
    {
    }
    SUBCASE("complete graph rings")
    {
        configuration.reconstruction.completeStencilRings = true;
    }
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
    struct ReconstructionCase
    {
        IntegrationMode mode;
        ReconstructionMethod method;
        DNDS::real weight;
    };

    for (const auto &[mode, method, weight] :
         std::array{
             ReconstructionCase{IntegrationMode::EfficientDifferential,
                                ReconstructionMethod::SVDLeastSquares, 1.0},
             ReconstructionCase{IntegrationMode::EfficientDifferential,
                                ReconstructionMethod::LeastSquares, 1.0},
             ReconstructionCase{IntegrationMode::EfficientDifferential,
                                ReconstructionMethod::Variational, 0.5},
             ReconstructionCase{IntegrationMode::TraditionalQuadrature,
                                ReconstructionMethod::SVDLeastSquares, 1.0},
             ReconstructionCase{IntegrationMode::TraditionalQuadrature,
                                ReconstructionMethod::LeastSquares, 1.0},
             ReconstructionCase{IntegrationMode::TraditionalQuadrature,
                                ReconstructionMethod::Variational, 0.5}})
    {
        configuration.algorithm.mode = mode;
        configuration.reconstruction.method = method;
        configuration.reconstruction.variationalWeight = weight;
        configuration.algorithm.retainMicroGeometry = true;
        DualGeometry retained(gMPI, mesh, topology, configuration.algorithm,
                              false);
        retained.Build();
        configuration.algorithm.retainMicroGeometry = false;
        DualGeometry geometry(gMPI, mesh, topology, configuration.algorithm,
                              false);
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

TEST_CASE("NCFV pressure limiting bounds the complete conservative segment")
{
    using State = Eigen::Vector<double, 5>;
    const auto pressure = [](const State &u)
    {
        return .4 * (u(4) - .5 * u.segment<3>(1).squaredNorm() / u(0));
    };
    State mean, candidate;
    mean << 1, -1, 0, 0, 3;
    candidate << 1, 1, 0, 0, 3;
    // Both endpoints have p=1, but the interior peak is p=1.2.
    const double upperCap = LimitIdealGasPressureFactor<3>(mean, candidate, 1.4, .9, 1.1, 1.0);
    CHECK(upperCap == doctest::Approx((1 - std::sqrt(.5)) / 2).epsilon(2e-8));
    for (int i = 0; i <= 100; i++)
    {
        const double theta = upperCap * i / 100;
        CHECK(pressure(mean + theta * (candidate - mean)) <= 1.1 + 2e-10);
        CHECK(pressure(mean + theta * (candidate - mean)) >= .9 - 2e-10);
    }
    CHECK(LimitIdealGasPressureFactor<3>(mean, candidate, 1.4, .9, 1.2, 1.0) == 1.0);
    CHECK(LimitIdealGasPressureFactor<3>(mean, candidate, 1.4, .9, 1.1, .5 * upperCap) == .5 * upperCap);

    mean << 1, 0, 0, 0, 2.5;
    candidate << 1, 0, 0, 0, .5;
    const double lowerCap = LimitIdealGasPressureFactor<3>(mean, candidate, 1.4, .6, 1.2, 1.0);
    CHECK(lowerCap == doctest::Approx(.5).epsilon(1e-8));
    for (int i = 0; i <= 100; i++)
        CHECK(pressure(mean + lowerCap * i / 100 * (candidate - mean)) >= .6 - 2e-10);

    // A constant-pressure contact must remain untouched, even with a density jump.
    mean << 1, .7, 0, 0, 2.5 + .245;
    candidate << 2, 1.4, 0, 0, 2.5 + .49;
    CHECK(LimitIdealGasPressureFactor<3>(mean, candidate, 1.4, 1, 1, 1.0) == 1.0);
    CHECK(LimitIdealGasPressureFactor<3>(mean, candidate, 1.4, 1, 1, 0.0) == 0.0);

    mean << 1, 0, 0, 0, 2.5;
    candidate << .05, .25, 0, 0, 1.125;
    const double variableDensityCap = LimitIdealGasPressureFactor<3>(mean, candidate, 1.4, .3, 1.1, 1.0);
    CHECK(variableDensityCap > 0);
    CHECK(variableDensityCap < 1);
    for (int i = 0; i <= 100; i++)
    {
        const double value = pressure(mean + variableDensityCap * i / 100 * (candidate - mean));
        CHECK(value >= .3 - 2e-10);
        CHECK(value <= 1.1 + 2e-10);
    }
}

TEST_CASE("NCFV pressure limiting constrains recovered points and full macro interfaces")
{
    auto configuration = MakeConfiguration(IntegrationMode::EfficientDifferential);
    configuration.reconstruction.enableLimiter = true;
    configuration.reconstruction.limitPointRecovery = true;
    configuration.physics.viscous.enabled = false;
    configuration.physics.boundaryMode = BoundaryMode::SupersonicOutlet;
    configuration.physics.boundaryZones.front().mode = BoundaryMode::SupersonicOutlet;
    Solver<3> solver(gMPI, configuration);
    solver.Initialize();
    const auto &mesh = solver.Mesh();
    const auto &halo = solver.NodeCommunication();
    NodeStatePair means, rhs;
    for (auto *field : {&means, &rhs})
    {
        field->InitPair(field == &means ? "NCFV.test.pressureMeans" : "NCFV.test.pressureRhs", gMPI);
        field->father->Resize(mesh->NumNode(), 5, 1);
        field->son->Resize(halo.NumNodeGhost(), 5, 1);
        field->BorrowSetup(halo.Layout());
        field->trans.initPersistentPull();
    }
    for (DNDS::index i = 0; i < mesh->NumNode(); i++)
    {
        const double angle = .7 * mesh->node2nodeOrig(i, 0);
        means[i] << 1, std::cos(angle), std::sin(angle), 0, 3;
    }
    const auto pressure = [](const Eigen::Vector<double, 5> &u)
    {
        return .4 * (u(4) - .5 * u.segment<3>(1).squaredNorm() / u(0));
    };
    double localUnboundedExcess = 0;
    for (bool enabled : {false, true})
    {
        configuration.reconstruction.limitPressure = enabled;
        SpatialOperator<3> spatial(
            gMPI, mesh, solver.EdgeTopology(), solver.Geometry(), solver.ReconstructionData(),
            solver.Boundaries(), configuration.algorithm.mode, configuration.reconstruction,
            configuration.physics, configuration.time, halo);
        spatial.Initialize();
        CHECK(std::isfinite(spatial.EvaluateRHS(means, rhs)));
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            CHECK(pressure(Eigen::Vector<double, 5>(means[i])) == doctest::Approx(1.0));
            const auto checkState = [&](const Eigen::Vector<double, 5> &state)
            {
                if (enabled)
                {
                    CHECK(pressure(state) <= 1 + 2e-9);
                    CHECK(pressure(state) >= 1 - 2e-9);
                }
                else
                    localUnboundedExcess = std::max(localUnboundedExcess, pressure(state) - 1);
            };
            checkState(Eigen::Vector<double, 5>(spatial.PointValues()[i]));
            for (const auto &incidence : solver.EdgeTopology().Node2Edge(i))
            {
                const auto &surface = solver.Geometry().EdgeSurface(incidence.edge);
                const int side = surface.nodes[0] == i ? 0 : 1;
                Eigen::Vector<double, 5> correction = Eigen::Vector<double, 5>::Zero();
                for (const auto &entry : surface.efficientStencil)
                    correction += spatial.Gradients()[entry.node].transpose() * entry.stateGradientWeights[side].head<3>();
                checkState(Eigen::Vector<double, 5>(spatial.PointValues()[i]) +
                           spatial.LimiterFactors()[i](0) * correction / surface.measure);
            }
        }
    }
    double globalUnboundedExcess = 0;
    MPI_Allreduce(&localUnboundedExcess, &globalUnboundedExcess, 1, DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
    CHECK(globalUnboundedExcess > 1e-3);
}

TEST_CASE("NCFV limited efficient anchors stay physical without changing conservative means")
{
    bool limitRecovery = false;
    SUBCASE("endpoint reference")
    {
    }
    SUBCASE("mean reference and point recovery")
    {
        limitRecovery = true;
    }
    Configuration configuration = MakeConfiguration(IntegrationMode::EfficientDifferential);
    configuration.reconstruction.enableLimiter = true;
    configuration.reconstruction.limitPointRecovery = limitRecovery;
    configuration.physics.viscous.enabled = false;
    configuration.physics.boundaryMode = BoundaryMode::SupersonicOutlet;
    configuration.physics.boundaryZones.front().mode = BoundaryMode::SupersonicOutlet;
    Solver<3> solver(gMPI, configuration);
    solver.Initialize();
    const auto &mesh = solver.Mesh();
    const auto &halo = solver.NodeCommunication();
    NodeStatePair means, rhs, rawPoints;
    const auto allocate = [&](NodeStatePair &field, const std::string &name)
    {
        field.InitPair(name, gMPI);
        field.father->Resize(mesh->NumNode(), 5, 1);
        field.son->Resize(halo.NumNodeGhost(), 5, 1);
        field.BorrowSetup(halo.Layout());
        field.trans.initPersistentPull();
        for (DNDS::index i = 0; i < field.Size(); i++)
            field[i].setZero();
    };
    allocate(means, "NCFV.test.positiveMeans");
    allocate(rhs, "NCFV.test.positiveRhs");
    allocate(rawPoints, "NCFV.test.rawPointRecovery");
    for (DNDS::index i = 0; i < mesh->NumNode(); i++)
    {
        means[i](0) = 1;
        // Positive high-contrast means produce negative point energies through
        // the neighboring-gradient recovery, before any interface correction.
        means[i](4) = mesh->node2nodeOrig(i, 0) % 5 == 0 ? 10.0 : 0.01;
    }
    SpatialOperator<3> spatial(
        gMPI, mesh, solver.EdgeTopology(), solver.Geometry(), solver.ReconstructionData(),
        solver.Boundaries(), configuration.algorithm.mode, configuration.reconstruction,
        configuration.physics, configuration.time, halo);
    spatial.Initialize();
    CHECK(std::isfinite(spatial.EvaluateRHS(means, rhs)));
    solver.ReconstructionData().RecoverPointValues(
        means, spatial.Gradients(), spatial.Coefficients(), rawPoints);
    DNDS::index localUnphysicalRawPoints = 0;
    for (DNDS::index i = 0; i < mesh->NumNode(); i++)
    {
        localUnphysicalRawPoints += rawPoints[i](4) <= 0;
        const auto point = spatial.PointValues()[i];
        CHECK(point.allFinite());
        CHECK(point(0) > 1e-12);
        const DNDS::real pressure = (configuration.physics.gamma - 1) *
                                    (point(4) - 0.5 * point.template segment<3>(1).squaredNorm() / point(0));
        CHECK(pressure > 1e-12);
        CHECK(rhs[i].allFinite());
        CHECK(means[i](0) == 1);
        CHECK(means[i].template segment<3>(1).squaredNorm() == 0);
        CHECK(means[i](4) == (mesh->node2nodeOrig(i, 0) % 5 == 0 ? 10.0 : 0.01));
        if (!limitRecovery && rawPoints[i](4) > 1e-12)
            CHECK((point - rawPoints[i]).squaredNorm() == 0);
        if (limitRecovery)
        {
            Eigen::Vector<double, 5> lo = means[i], hi = means[i];
            for (const auto &incidence : solver.EdgeTopology().Node2Edge(i))
            {
                const auto &surface = solver.Geometry().EdgeSurface(incidence.edge);
                const auto neighbor = surface.nodes[surface.nodes[0] == i ? 1 : 0];
                lo = lo.cwiseMin(Eigen::Vector<double, 5>(means[neighbor]));
                hi = hi.cwiseMax(Eigen::Vector<double, 5>(means[neighbor]));
            }
            CHECK((point.array() >= lo.array() - 1e-12).all());
            CHECK((point.array() <= hi.array() + 1e-12).all());
            const DNDS::real factor = spatial.LimiterFactors()[i](0);
            if (factor == 0)
                CHECK((point - means[i]).squaredNorm() == 0);
            for (const auto &incidence : solver.EdgeTopology().Node2Edge(i))
            {
                const auto &surface = solver.Geometry().EdgeSurface(incidence.edge);
                const int side = surface.nodes[0] == i ? 0 : 1;
                Eigen::Vector<double, 5> correction = Eigen::Vector<double, 5>::Zero();
                for (const auto &entry : surface.efficientStencil)
                    correction += spatial.Gradients()[entry.node].transpose() * entry.stateGradientWeights[side].head<3>();
                const Eigen::Vector<double, 5> limited = point + factor * correction / surface.measure;
                CHECK((limited.array() >= lo.array() - 1e-12).all());
                CHECK((limited.array() <= hi.array() + 1e-12).all());
                CHECK(limited(0) > 1e-12);
                CHECK((configuration.physics.gamma - 1) *
                          (limited(4) - .5 * limited.segment<3>(1).squaredNorm() / limited(0)) > 1e-12);
            }
        }
    }
    DNDS::index globalUnphysicalRawPoints = 0;
    MPI_Allreduce(&localUnphysicalRawPoints, &globalUnphysicalRawPoints, 1,
                  DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
    CHECK(globalUnphysicalRawPoints > 0);
}

TEST_CASE("NCFV binary Barth fallback preserves shared gradients and uses zeroth-order fluxes")
{
    using State = Eigen::Vector<double, 5>;
    bool periodic = false;
    SUBCASE("constant extrapolation boundary")
    {
    }
    SUBCASE("fully periodic conservation")
    {
        periodic = true;
    }
    auto configuration = periodic
                             ? MakeExactPeriodicConfiguration(IntegrationMode::EfficientDifferential)
                             : MakeConfiguration(IntegrationMode::EfficientDifferential);
    configuration.reconstruction.method = ReconstructionMethod::SVDLeastSquares;
    configuration.reconstruction.enableLimiter = true;
    configuration.reconstruction.limitPointRecovery = true;
    configuration.reconstruction.barthZeroOrderFallback = true;
    configuration.reconstruction.sharedGradientFaceSweep = false;
    configuration.reconstruction.limitFluxCorrection = false;
    configuration.physics.viscous.enabled = false;
    configuration.physics.boundaryMode = BoundaryMode::SupersonicOutlet;
    if (!periodic)
        configuration.physics.boundaryZones.front().mode = BoundaryMode::SupersonicOutlet;
    configuration.Validate();
    Solver<3> solver(gMPI, configuration);
    solver.Initialize();
    const auto &mesh = solver.Mesh();
    const auto &halo = solver.NodeCommunication();
    NodeStatePair means, rhs;
    for (auto *field : {&means, &rhs})
    {
        field->InitPair(field == &means ? "NCFV.test.binaryMeans" : "NCFV.test.binaryRhs", gMPI);
        field->father->Resize(mesh->NumNode(), 5, 1);
        field->son->Resize(halo.NumNodeGhost(), 5, 1);
        field->BorrowSetup(halo.Layout());
        field->trans.initPersistentPull();
    }
    NodeMatrixPair rawGradients, rawCoefficients;
    const auto allocateMatrix = [&](NodeMatrixPair &field, const std::string &name, int rows)
    {
        field.InitPair(name, gMPI);
        field.father->Resize(mesh->NumNode(), rows, 5);
        field.son->Resize(halo.NumNodeGhost(), rows, 5);
        field.BorrowSetup(halo.Layout());
        field.trans.initPersistentPull();
    };
    allocateMatrix(rawGradients, "NCFV.test.binaryRawGradients", 3);
    allocateMatrix(rawCoefficients, "NCFV.test.binaryRawCoefficients", 9);
    SpatialOperator<3> spatial(
        gMPI, mesh, solver.EdgeTopology(), solver.Geometry(), solver.ReconstructionData(),
        solver.Boundaries(), configuration.algorithm.mode, configuration.reconstruction,
        configuration.physics, configuration.time, halo);
    spatial.Initialize();
    const Vector3 velocity{.3, -.2, .1};
    Vector3 gridVelocity = Vector3::Zero();
    const auto state = [&](DNDS::real density, DNDS::real pressure) -> State
    {
        State result;
        result << density, density * velocity,
            pressure / (configuration.physics.gamma - 1) + .5 * density * velocity.squaredNorm();
        return result;
    };
    const auto physicalFlux = [&](const State &u, const Vector3 &area) -> State
    {
        const Vector3 momentum = u.segment<3>(1);
        const DNDS::real pressure = (configuration.physics.gamma - 1) *
                                    (u(4) - .5 * momentum.squaredNorm() / u(0));
        const DNDS::real normalVelocity = momentum.dot(area) / u(0);
        State flux;
        flux << momentum.dot(area), momentum * normalVelocity + pressure * area,
            (u(4) + pressure) * normalVelocity;
        return flux;
    };
    // The affine case checks smooth interior anchors without requiring a
    // discontinuous periodic affine extension or unlimited smooth extrema.
    for (int fieldKind = 0; fieldKind < 3; fieldKind++)
    {
        if (periodic && fieldKind == 1)
            continue;
        const auto expectedState = [&](DNDS::index node) -> State
        {
            if (fieldKind == 0)
                return state(1, 1);
            if (fieldKind == 1)
            {
                const auto &moments = solver.Geometry().NodeVolume(node).moments;
                const Vector3 centroid = mesh->coords[node] + moments.first / moments.measure;
                return state(1 + Vector3(.04, .03, .02).dot(centroid),
                             1 + Vector3(.01, -.02, .03).dot(centroid));
            }
            const auto original = mesh->node2nodeOrig(node, 0);
            return state(original % 5 == 0 ? 2 : 1, original % 7 == 0 ? 4 : .2);
        };
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
            means[node] = expectedState(node);
        CHECK(std::isfinite(spatial.EvaluateRHS(means, rhs)));
        solver.ReconstructionData().ComputeCoefficients(means, rawGradients, rawCoefficients);
        rawGradients.trans.startPersistentPull();
        rawGradients.trans.waitPersistentPull();
        const auto &factors = spatial.LimiterFactors();
        for (DNDS::index node = 0; node < halo.NumNodeProc(); node++)
        {
            CHECK((factors[node](0) == 0 || factors[node](0) == 1));
            CHECK((spatial.Gradients()[node] - rawGradients[node]).cwiseAbs().maxCoeff() < 1e-12);
            if (fieldKind == 0)
                CHECK(factors[node](0) == 1);
        }
        // Counts ensure that the discontinuous test actually exercises
        // both-sided zero-order Riemann fluxes and physical boundary fallback.
        std::array<DNDS::index, 5> localCounts{}, counts{};
        State localTotal = State::Zero(), total;
        for (DNDS::index node = 0; node < mesh->NumNode(); node++)
        {
            const State mean = means[node];
            const auto &volume = solver.Geometry().NodeVolume(node);
            const DNDS::real factor = factors[node](0);
            CHECK((mean - expectedState(node)).cwiseAbs().maxCoeff() == 0);
            CHECK(rhs[node].allFinite());
            localTotal += volume.moments.measure * rhs[node];
            localCounts[4] += rawGradients[node].cwiseAbs().maxCoeff() > 1e-12;
            if (fieldKind == 0)
                CHECK(rhs[node].cwiseAbs().maxCoeff() < 2e-10);
            if (fieldKind == 1 && volume.boundaryPieces.empty())
            {
                CHECK(factor == 1);
                localCounts[3]++;
            }
            if (factor != 0)
                continue;
            localCounts[0]++;
            CHECK((State(spatial.PointValues()[node]) - mean).cwiseAbs().maxCoeff() == 0);
            for (const auto &incidence : solver.EdgeTopology().Node2Edge(node))
            {
                const auto &surface = solver.Geometry().EdgeSurface(incidence.edge);
                const int side = surface.nodes[0] == node ? 0 : 1;
                State correction = State::Zero();
                for (const auto &entry : surface.efficientStencil)
                    correction += spatial.Gradients()[entry.node].transpose() *
                                  entry.stateGradientWeights[side].head<3>();
                const State faceMean = State(spatial.PointValues()[node]) +
                                       factor * correction / surface.measure;
                CHECK((faceMean - mean).cwiseAbs().maxCoeff() == 0);
            }
            if (!periodic && !volume.boundaryPieces.empty())
            {
                // Recover the boundary flux from the actual residual rather
                // than repeating the production boundary integration path.
                State actual = -volume.moments.measure * rhs[node];
                for (const auto &incidence : solver.EdgeTopology().Node2Edge(node))
                    actual -= incidence.outwardSign * State(spatial.EdgeFluxes()[incidence.edge]);
                State expected = State::Zero();
                for (const auto &piece : volume.boundaryPieces)
                    expected += physicalFlux(mean, piece.vectorMeasure);
                CHECK((actual - expected).cwiseAbs().maxCoeff() < 5e-11);
                localCounts[2]++;
            }
        }
        for (DNDS::index edge = 0; edge < solver.EdgeTopology().NumEdge(); edge++)
        {
            const auto &surface = solver.Geometry().EdgeSurface(edge);
            if (factors[surface.nodes[0]](0) != 0 || factors[surface.nodes[1]](0) != 0)
                continue;
            const State left = means[surface.nodes[0]], right = means[surface.nodes[1]];
            const Vector3 normal = surface.vectorMeasure.normalized();
            CHECK(std::abs(surface.vectorMeasure.norm() - surface.measure) < 1e-12);
            State flux = State::Zero();
            DNDS::real acousticMinus = 0, contact = 0, acousticPlus = 0;
            Euler::Gas::InviscidFlux_IdealGas_Dispatcher<3>(
                configuration.physics.riemannSolver, left, right, left, right,
                gridVelocity, normal, configuration.physics.gamma, configuration.physics.gamma,
                flux, 0.0, 1.0, 1.0, []() {}, acousticMinus, contact, acousticPlus);
            const State expected = surface.measure * flux;
            CHECK((State(spatial.EdgeFluxes()[edge]) - expected).cwiseAbs().maxCoeff() < 5e-11);
            localCounts[1]++;
        }
        MPI_Allreduce(localCounts.data(), counts.data(), 5, DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
        MPI_Allreduce(localTotal.data(), total.data(), 5, DNDS_MPI_REAL, MPI_SUM, gMPI.comm);
        if (periodic)
            CHECK(total.cwiseAbs().maxCoeff() < 1e-9);
        if (fieldKind == 1)
            CHECK(counts[3] > 0);
        if (fieldKind == 2)
        {
            CHECK(counts[0] > 0);
            CHECK(counts[1] > 0);
            CHECK(counts[4] > 0);
            if (!periodic)
                CHECK(counts[2] > 0);
        }
    }
}

TEST_CASE("NCFV conservative flux correction preserves entropy bounds and periodic conservation")
{
    auto configuration = MakeExactPeriodicConfiguration(IntegrationMode::EfficientDifferential);
    // Exercise the flux limiter independently of the reconstruction limiter.
    configuration.reconstruction.enableLimiter = false;
    configuration.reconstruction.limitFluxCorrection = true;
    SUBCASE("one-ring bar bounds")
    {
    }
    SUBCASE("edge-pair bar bounds")
    {
        configuration.reconstruction.fluxCorrectionPairBounds = true;
    }
    configuration.time.cfl = .5;
    Solver<3> solver(gMPI, configuration);
    solver.Initialize();
    const auto &mesh = solver.Mesh();
    const auto &halo = solver.NodeCommunication();
    NodeStatePair means, rhs;
    for (auto *field : {&means, &rhs})
    {
        field->InitPair(field == &means ? "NCFV.test.fluxMeans" : "NCFV.test.fluxRhs", gMPI);
        field->father->Resize(mesh->NumNode(), 5, 1);
        field->son->Resize(halo.NumNodeGhost(), 5, 1);
        field->BorrowSetup(halo.Layout());
        field->trans.initPersistentPull();
    }
    for (DNDS::index i = 0; i < mesh->NumNode(); i++)
    {
        const double density = mesh->node2nodeOrig(i, 0) % 5 == 0 ? 2.0 : 1.0;
        means[i] << density, .5 * density, 0, 0, 2.5 + .125 * density;
    }
    SpatialOperator<3> spatial(
        gMPI, mesh, solver.EdgeTopology(), solver.Geometry(), solver.ReconstructionData(),
        solver.Boundaries(), configuration.algorithm.mode, configuration.reconstruction,
        configuration.physics, configuration.time, halo);
    spatial.Initialize();
    CHECK(std::isfinite(spatial.EvaluateRHS(means, rhs)));
    Eigen::Vector<double, 5> localTotal = Eigen::Vector<double, 5>::Zero();
    double localLimited = 0, localChange = 0;
    for (DNDS::index i = 0; i < mesh->NumNode(); i++)
    {
        const double volume = solver.Geometry().NodeVolume(i).moments.measure;
        localTotal += volume * rhs[i];
        const Eigen::Vector<double, 5> next = means[i] + spatial.LocalTimeStep(i) * rhs[i];
        CHECK(next(0) >= 1 - 1e-11);
        CHECK(next(0) <= 2 + 1e-11);
        CHECK(next(3) == doctest::Approx(0).epsilon(1e-11));
        CHECK(.4 * (next(4) - .5 * next.segment<3>(1).squaredNorm() / next(0)) == doctest::Approx(1).epsilon(1e-11));
        localChange += (next - means[i]).norm();
    }
    for (DNDS::index edge = 0; edge < solver.EdgeTopology().NumEdge(); edge++)
    {
        const double factor = spatial.EdgeFluxCorrectionFactors()[edge](0);
        CHECK(factor >= 0);
        CHECK(factor <= 1);
        localLimited += factor < .99;
    }
    Eigen::Vector<double, 5> total = Eigen::Vector<double, 5>::Zero();
    double limited = 0, change = 0;
    MPI_Allreduce(localTotal.data(), total.data(), 5, DNDS_MPI_REAL, MPI_SUM, gMPI.comm);
    MPI_Allreduce(&localLimited, &limited, 1, DNDS_MPI_REAL, MPI_SUM, gMPI.comm);
    MPI_Allreduce(&localChange, &change, 1, DNDS_MPI_REAL, MPI_SUM, gMPI.comm);
    CHECK(total.cwiseAbs().maxCoeff() < 1e-10);
    CHECK(limited > 0);
    CHECK(change > 0);
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
