#include "NCFVDualGeometry.hpp"

#include "DNDS/Errors.hpp"
#include "Geom/Quadrature.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace DNDS::NCFV
{
    namespace
    {
        constexpr std::size_t MaxAffineSupportNodes = 8;

        template <class TEntry>
        TEntry &FindStencilNode(
            std::vector<TEntry> &stencil,
            index node)
        {
            const auto iterator = std::lower_bound(
                stencil.begin(), stencil.end(), node,
                [](const TEntry &entry, index value)
                { return entry.node < value; });
            DNDS_check_throw_info(
                iterator != stencil.end() && iterator->node == node,
                "NCFV efficient stencil was not initialized with every support node");
            return *iterator;
        }

        std::vector<index> CollectSupportNodes(
            const std::vector<AffinePoint> &points)
        {
            std::size_t capacity = 0;
            for (const AffinePoint &point : points)
                capacity += point.support.size();

            std::vector<index> nodes;
            nodes.reserve(capacity);
            for (const AffinePoint &point : points)
                for (const AffineCoefficient &coefficient : point.support)
                    nodes.push_back(coefficient.node);
            std::sort(nodes.begin(), nodes.end());
            nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
            return nodes;
        }

        template <class TEntry>
        void InitializeStencil(
            std::vector<TEntry> &stencil,
            const std::vector<index> &nodes)
        {
            DNDS_check_throw_info(stencil.empty(),
                                  "NCFV efficient stencil was initialized more than once");
            stencil.reserve(nodes.size());
            for (index node : nodes)
                stencil.emplace_back(TEntry{node});
        }

        template <class TCallback>
        void ForEachDifferentialWeight(
            const std::vector<AffinePoint> &points,
            index anchorNode,
            const Vector3 &anchorCoordinate,
            real measure,
            TCallback &&callback)
        {
            DNDS_check_throw_info(points.size() >= 2 && points.size() <= 4,
                                  "DifferentialWeights accepts line, triangle, or tetrahedron vertices");

            std::array<Vector3, 4> displacements;
            Vector3 displacementSum = Vector3::Zero();
            std::array<index, MaxAffineSupportNodes> supportNodes;
            std::size_t supportNodeCount = 0;
            const auto appendSupportNode = [&](index node)
            {
                for (std::size_t i = 0; i < supportNodeCount; i++)
                    if (supportNodes[i] == node)
                        return;
                DNDS_check_throw_info(
                    supportNodeCount < supportNodes.size(),
                    "NCFV affine simplex exceeds the supported O1 cell vertex count");
                supportNodes[supportNodeCount++] = node;
            };

            appendSupportNode(anchorNode);
            for (std::size_t i = 0; i < points.size(); i++)
            {
                displacements[i] = points[i].coordinate - anchorCoordinate;
                displacementSum += displacements[i];
                for (const AffineCoefficient &coefficient : points[i].support)
                    appendSupportNode(coefficient.node);
            }
            for (std::size_t i = 1; i < supportNodeCount; i++)
            {
                const index node = supportNodes[i];
                std::size_t insertion = i;
                while (insertion > 0 && supportNodes[insertion - 1] > node)
                {
                    supportNodes[insertion] = supportNodes[insertion - 1];
                    insertion--;
                }
                supportNodes[insertion] = node;
            }

            std::array<real, MaxAffineSupportNodes> coefficientSums{};
            std::array<Vector3, MaxAffineSupportNodes> displacementCoefficientSums;
            for (std::size_t i = 0; i < supportNodeCount; i++)
                displacementCoefficientSums[i].setZero();
            for (std::size_t iPoint = 0; iPoint < points.size(); iPoint++)
                for (const AffineCoefficient &coefficient : points[iPoint].support)
                {
                    const auto iterator = std::lower_bound(
                        supportNodes.begin(), supportNodes.begin() + supportNodeCount,
                        coefficient.node);
                    const std::size_t iNode = static_cast<std::size_t>(
                        iterator - supportNodes.begin());
                    DNDS_assert(iNode < supportNodeCount &&
                                supportNodes[iNode] == coefficient.node);
                    coefficientSums[iNode] += coefficient.coefficient;
                    displacementCoefficientSums[iNode] +=
                        displacements[iPoint] * coefficient.coefficient;
                }

            const real n = static_cast<real>(points.size());
            const real secondFactor = measure / (2.0 * n * (n + 1.0));
            for (std::size_t iNode = 0; iNode < supportNodeCount; iNode++)
            {
                const real delta = supportNodes[iNode] == anchorNode ? 1.0 : 0.0;
                const real coefficientSum = coefficientSums[iNode] - n * delta;
                const Vector3 displacementCoefficientSum =
                    displacementCoefficientSums[iNode] - delta * displacementSum;
                Vector3 value = measure / n * displacementSum * delta;
                value += secondFactor *
                         (displacementSum * coefficientSum +
                          displacementCoefficientSum);
                if (value.squaredNorm() > sqr(verySmallReal))
                    callback(supportNodes[iNode], value);
            }
        }

        template <class TEntry>
        void CheckStencilNodes(
            const std::vector<TEntry> &stencil,
            index processNodeCount)
        {
            index previous = UnInitIndex;
            for (const TEntry &entry : stencil)
            {
                DNDS_check_throw_info(
                    entry.node >= 0 && entry.node < processNodeCount,
                    "NCFV efficient stencil contains an invalid local/ghost node index");
                DNDS_check_throw_info(
                    previous == UnInitIndex || previous < entry.node,
                    "NCFV efficient stencil node indices are not sorted and unique");
                previous = entry.node;
            }
        }

        void MergePointRecoveryWeights(
            std::vector<EfficientPointRecoveryNode> &stencil,
            const std::vector<AffinePoint> &points,
            index anchorNode,
            const Vector3 &anchorCoordinate,
            real measure)
        {
            ForEachDifferentialWeight(
                points, anchorNode, anchorCoordinate, measure,
                [&](index node, const Vector3 &value)
                { FindStencilNode(stencil, node).gradientWeight += value; });
        }

        void MergeSurfaceDifferentialWeights(
            std::vector<EfficientSurfaceNode> &stencil,
            const std::vector<AffinePoint> &points,
            index anchorNode,
            const Vector3 &anchorCoordinate,
            real measure,
            const Vector3 &unitNormal,
            int side)
        {
            DNDS_assert(side == 0 || side == 1);
            ForEachDifferentialWeight(
                points, anchorNode, anchorCoordinate, measure,
                [&](index node, const Vector3 &value)
                {
                    EfficientSurfaceNode &entry = FindStencilNode(stencil, node);
                    entry.stateGradientWeights[static_cast<std::size_t>(side)] += value;
                    entry.fluxGradientWeights[static_cast<std::size_t>(side)] +=
                        value * unitNormal.transpose();
                });
        }

        void MergeSurfaceGradientWeights(
            std::vector<EfficientSurfaceNode> &stencil,
            const std::vector<AffinePoint> &points,
            real measure)
        {
            const real pointFactor = measure / static_cast<real>(points.size());
            for (const auto &point : points)
                for (const auto &coefficient : point.support)
                    FindStencilNode(stencil, coefficient.node).gradientWeight +=
                        pointFactor * coefficient.coefficient;
        }

        void MergeBoundaryWeights(
            std::vector<EfficientBoundaryNode> &stencil,
            const std::vector<AffinePoint> &points,
            real measure)
        {
            InitializeStencil(stencil, CollectSupportNodes(points));
            const real pointFactor = measure / static_cast<real>(points.size());
            for (const auto &point : points)
                for (const auto &coefficient : point.support)
                {
                    EfficientBoundaryNode &entry = FindStencilNode(stencil, coefficient.node);
                    const real weight = pointFactor * coefficient.coefficient;
                    entry.integrationWeight += weight;
                }
        }

        bool ContainsNode(const std::vector<index> &nodes, index target)
        {
            return std::find(nodes.begin(), nodes.end(), target) != nodes.end();
        }

        bool ContainsEdge(const std::vector<index> &nodes, index node0, index node1)
        {
            return ContainsNode(nodes, node0) && ContainsNode(nodes, node1);
        }

        Vector3 OrientedSurfaceVector(
            const std::vector<AffinePoint> &points,
            const Vector3 &orientationDirection)
        {
            Vector3 vectorMeasure = Vector3::Zero();
            if (points.size() == 2)
            {
                const Vector3 tangent = points[1].coordinate - points[0].coordinate;
                vectorMeasure << tangent.y(), -tangent.x(), 0.0;
            }
            else if (points.size() == 3)
            {
                vectorMeasure = 0.5 *
                                (points[1].coordinate - points[0].coordinate)
                                    .cross(points[2].coordinate - points[0].coordinate);
            }
            else
            {
                DNDS_check_throw_info(false, "A dual surface simplex must be a line or triangle");
            }
            if (vectorMeasure.dot(orientationDirection) < 0)
                vectorMeasure *= -1;
            return vectorMeasure;
        }

        std::vector<index> FaceVertexNodes(
            const ssp<Geom::UnstructuredMesh> &mesh,
            index iCell,
            int iLocalFace)
        {
            auto cell = mesh->GetCellElement(iCell);
            const auto face = cell.ObtainFace(iLocalFace);
            std::vector<index> faceNodes(static_cast<std::size_t>(face.GetNumNodes()));
            cell.ExtractFaceNodes(iLocalFace, mesh->cell2node[iCell], faceNodes);
            faceNodes.resize(static_cast<std::size_t>(face.GetNumVertices()));
            return faceNodes;
        }

        std::vector<index> FaceEdgeNodes(
            Geom::Elem::Element face,
            const std::vector<index> &faceNodes,
            int iLocalEdge)
        {
            const auto edge = face.ObtainFace(iLocalEdge);
            std::vector<index> edgeNodes(static_cast<std::size_t>(edge.GetNumNodes()));
            face.ExtractFaceNodes(iLocalEdge, faceNodes, edgeNodes);
            edgeNodes.resize(static_cast<std::size_t>(edge.GetNumVertices()));
            return edgeNodes;
        }

        void AccumulateMoments(RawMoments &destination, const RawMoments &source)
        {
            destination.measure += source.measure;
            destination.first += source.first;
            destination.second += source.second;
        }
    }

    std::pair<real, real> DualGeometry::SimplexMeasure(
        const std::vector<Vector3> &points,
        int dimension)
    {
        DNDS_check_throw_info(static_cast<int>(points.size()) == dimension + 1,
                              "Simplex point count does not match its dimension");
        if (dimension == 1)
        {
            const real length = (points[1] - points[0]).norm();
            return {length, length};
        }
        if (dimension == 2)
        {
            const Vector3 left = points[1] - points[0];
            const Vector3 right = points[2] - points[0];
            const real signedJacobian = left.x() * right.y() - left.y() * right.x();
            return {0.5 * left.cross(right).norm(), signedJacobian};
        }
        if (dimension == 3)
        {
            Eigen::Matrix3d jacobian;
            jacobian.col(0) = points[1] - points[0];
            jacobian.col(1) = points[2] - points[0];
            jacobian.col(2) = points[3] - points[0];
            const real signedJacobian = jacobian.determinant();
            return {std::abs(signedJacobian) / 6.0, signedJacobian};
        }
        DNDS_check_throw_info(false, "NCFV simplex dimension must be 1, 2, or 3");
        return {0, 0};
    }

    RawMoments DualGeometry::ExactSimplexMoments(
        const std::vector<Vector3> &points,
        real measure)
    {
        DNDS_check_throw_info(points.size() >= 2 && points.size() <= 4,
                              "ExactSimplexMoments accepts line, triangle, or tetrahedron vertices");
        const real n = static_cast<real>(points.size());
        Vector3 sum = Vector3::Zero();
        Matrix3 diagonalSum = Matrix3::Zero();
        for (const Vector3 &point : points)
        {
            sum += point;
            diagonalSum += point * point.transpose();
        }
        RawMoments moments;
        moments.measure = measure;
        moments.first = measure / n * sum;
        moments.second = measure / (n * (n + 1.0)) *
                         (sum * sum.transpose() + diagonalSum);
        return moments;
    }

    std::vector<SparseVectorWeight> DualGeometry::DifferentialWeights(
        const std::vector<AffinePoint> &points,
        index anchorNode,
        const Vector3 &anchorCoordinate,
        real measure)
    {
        std::vector<SparseVectorWeight> result;
        result.reserve(MaxAffineSupportNodes);
        ForEachDifferentialWeight(
            points, anchorNode, anchorCoordinate, measure,
            [&](index node, const Vector3 &value)
            { result.push_back({node, value}); });
        return result;
    }

    AffinePoint DualGeometry::MakeNodePoint(index iNode) const
    {
        AffinePoint point;
        point.coordinate = _mesh->coords[iNode];
        point.support.push_back({iNode, 1.0});
        return point;
    }

    Vector3 DualGeometry::CoordinateInFrame(
        index iNode,
        const Vector3 &frameAnchor) const
    {
        Vector3 displacement = _mesh->coords[iNode] - frameAnchor;
        for (int d = 0; d < _mesh->getDim(); d++)
            if (_periodicLengths(d) > 0)
                displacement(d) -= _periodicLengths(d) *
                                   std::round(displacement(d) /
                                              _periodicLengths(d));
        return frameAnchor + displacement;
    }

    AffinePoint DualGeometry::MakeAveragePoint(
        const std::vector<index> &nodes,
        const Vector3 *frameAnchor) const
    {
        DNDS_check_throw_info(!nodes.empty(), "Cannot average an empty primal-vertex set");
        AffinePoint point;
        const Vector3 anchor = frameAnchor
                                   ? *frameAnchor
                                   : Vector3(_mesh->coords[nodes.front()]);
        const real coefficient = 1.0 / static_cast<real>(nodes.size());
        point.support.reserve(nodes.size());
        for (index node : nodes)
        {
            point.coordinate += coefficient * CoordinateInFrame(node, anchor);
            point.support.push_back({node, coefficient});
        }
        std::sort(point.support.begin(), point.support.end(), [](const auto &left, const auto &right)
                  { return left.node < right.node; });
        return point;
    }

    AffinePoint DualGeometry::MakeCellPoint(
        index iCell,
        const Vector3 *frameAnchor) const
    {
        const auto element = _mesh->GetCellElement(iCell);
        std::vector<index> vertices(static_cast<std::size_t>(element.GetNumVertices()));
        for (int i = 0; i < element.GetNumVertices(); i++)
            vertices[static_cast<std::size_t>(i)] = _mesh->cell2node(iCell, i);
        return MakeAveragePoint(vertices, frameAnchor);
    }

    AffinePoint DualGeometry::MakeFacePoint(
        index iFace,
        const Vector3 *frameAnchor) const
    {
        const auto element = _topology.GetFaceElement(iFace);
        std::vector<index> vertices(static_cast<std::size_t>(element.GetNumVertices()));
        for (int i = 0; i < element.GetNumVertices(); i++)
            vertices[static_cast<std::size_t>(i)] = _topology.Face2Node()(iFace, i);
        return MakeAveragePoint(vertices, frameAnchor);
    }

    AffinePoint DualGeometry::MakeEdgePoint(
        index node0,
        index node1,
        const Vector3 *frameAnchor) const
    {
        return MakeAveragePoint({node0, node1}, frameAnchor);
    }

    void DualGeometry::BuildConstructionPoints()
    {
        _cellPoints.resize(static_cast<std::size_t>(_mesh->NumCellProc()));
        for (index iCell = 0; iCell < _mesh->NumCellProc(); iCell++)
            _cellPoints[static_cast<std::size_t>(iCell)] = MakeCellPoint(iCell).coordinate;

        _facePoints.resize(static_cast<std::size_t>(_topology.NumFaceProc()));
        for (index iFace = 0; iFace < _topology.NumFaceProc(); iFace++)
            _facePoints[static_cast<std::size_t>(iFace)] = MakeFacePoint(iFace).coordinate;

        _edgePoints.resize(static_cast<std::size_t>(_topology.NumEdgeProc()));
        for (index iEdge = 0; iEdge < _topology.NumEdgeProc(); iEdge++)
            _edgePoints[static_cast<std::size_t>(iEdge)] =
                MakeEdgePoint(_topology.Edge2Node()(iEdge, 0),
                              _topology.Edge2Node()(iEdge, 1))
                    .coordinate;
    }

    Vector3 DualGeometry::ReferenceLengthsFromBounds(
        const Vector3 &lower, const Vector3 &upper, int dimension)
    {
        DNDS_check_throw_info(dimension == 2 || dimension == 3,
                              "NCFV reference lengths require dimension 2 or 3");
        Vector3 lengths = Vector3::Ones();
        for (int d = 0; d < dimension; d++)
        {
            lengths(d) = 0.5 * (upper(d) - lower(d));
            DNDS_check_throw_info(std::isfinite(lengths(d)) && lengths(d) > verySmallReal,
                                  "NCFV dual volume has a degenerate reference extent");
        }
        return lengths;
    }

    void DualGeometry::AddVolumeSimplex(
        NodeControlVolume &controlVolume,
        index iCell,
        const std::vector<AffinePoint> &points) const
    {
        const int dimension = _mesh->getDim();
        const std::vector<Vector3> coordinates = [&]()
        {
            std::vector<Vector3> result;
            result.reserve(points.size());
            for (const auto &point : points)
                result.push_back(point.coordinate);
            return result;
        }();
        const auto [measure, signedJacobian] = SimplexMeasure(coordinates, dimension);
        DNDS_check_throw_info(measure > verySmallReal,
                              "NCFV generated a degenerate dual micro-volume");

        const Vector3 anchorCoordinate = points.front().coordinate;
        for (const Vector3 &point : coordinates)
        {
            const Vector3 offset = point - anchorCoordinate;
            controlVolume.lowerOffset = controlVolume.lowerOffset.cwiseMin(offset);
            controlVolume.upperOffset = controlVolume.upperOffset.cwiseMax(offset);
        }

        if (_settings.mode == IntegrationMode::EfficientDifferential)
        {
            std::vector<Vector3> localCoordinates;
            localCoordinates.reserve(coordinates.size());
            for (const Vector3 &coordinate : coordinates)
                localCoordinates.push_back(coordinate - anchorCoordinate);
            AccumulateMoments(
                controlVolume.moments,
                ExactSimplexMoments(localCoordinates, measure));
            const double weightStart = _settings.profileIntegrationInitialization
                                           ? MPI_Wtime()
                                           : 0.0;
            MergePointRecoveryWeights(
                controlVolume.pointRecoveryStencil,
                points, controlVolume.node,
                anchorCoordinate, measure);
            if (_settings.profileIntegrationInitialization)
            {
                _integrationInitializationTiming.volumeSeconds += MPI_Wtime() - weightStart;
                _integrationInitializationTiming.volumeCalls++;
            }
        }
        else
        {
            const auto simplexElement = dimension == 2
                                            ? Geom::Elem::Element{Geom::Elem::Tri3}
                                            : Geom::Elem::Element{Geom::Elem::Tet4};
            const real jacobian = dimension == 2 ? 2.0 * measure : 6.0 * measure;
            const auto appendQuadraturePoint = [&](const Geom::Elem::Quadrature &quadrature, int iG)
            {
                const auto [parametric, referenceWeight] = quadrature.GetQuadraturePointInfo(iG);
                Vector3 physical = points[0].coordinate;
                physical += parametric[0] * (points[1].coordinate - points[0].coordinate);
                physical += parametric[1] * (points[2].coordinate - points[0].coordinate);
                if (dimension == 3)
                    physical += parametric[2] * (points[3].coordinate - points[0].coordinate);
                const real physicalWeight = referenceWeight * jacobian;
                controlVolume.volumeQuadrature.push_back({physical, physicalWeight});
            };

            if (_settings.profileIntegrationInitialization)
            {
                const std::size_t firstPoint = controlVolume.volumeQuadrature.size();
                const double quadratureStart = MPI_Wtime();
                Geom::Elem::Quadrature quadrature(simplexElement, _settings.quadratureOrder);
                for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
                    appendQuadraturePoint(quadrature, iG);
                _integrationInitializationTiming.volumeSeconds += MPI_Wtime() - quadratureStart;
                _integrationInitializationTiming.volumeCalls++;
                for (std::size_t iPoint = firstPoint;
                     iPoint < controlVolume.volumeQuadrature.size(); iPoint++)
                {
                    const auto &point = controlVolume.volumeQuadrature[iPoint];
                    const Vector3 offset = point.coordinate - anchorCoordinate;
                    controlVolume.moments.measure += point.weight;
                    controlVolume.moments.first += point.weight * offset;
                    controlVolume.moments.second +=
                        point.weight * offset * offset.transpose();
                }
            }
            else
            {
                Geom::Elem::Quadrature quadrature(simplexElement, _settings.quadratureOrder);
                for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
                {
                    appendQuadraturePoint(quadrature, iG);
                    const auto &point = controlVolume.volumeQuadrature.back();
                    const Vector3 offset = point.coordinate - anchorCoordinate;
                    controlVolume.moments.measure += point.weight;
                    controlVolume.moments.first += point.weight * offset;
                    controlVolume.moments.second +=
                        point.weight * offset * offset.transpose();
                }
            }
        }

        if (_settings.retainMicroGeometry)
        {
            MicroVolume micro;
            micro.primalCell = iCell;
            micro.nPoints = dimension + 1;
            micro.signedJacobian = signedJacobian;
            micro.measure = measure;
            for (int i = 0; i < micro.nPoints; i++)
                micro.points[static_cast<std::size_t>(i)] = points[static_cast<std::size_t>(i)].coordinate;
            controlVolume.microVolumes.push_back(std::move(micro));
        }
    }

    void DualGeometry::AddEdgeSurfaceSimplex(
        EdgeControlSurface &surface,
        index iCell,
        index iFace,
        const std::vector<AffinePoint> &points) const
    {
        const Vector3 edgeDirection =
            surface.nodeCoordinates[1] - surface.nodeCoordinates[0];
        const Vector3 vectorMeasure = OrientedSurfaceVector(points, edgeDirection);
        const real measure = vectorMeasure.norm();
        DNDS_check_throw_info(measure > verySmallReal,
                              "NCFV generated a degenerate dual edge surface");
        const Vector3 unitNormal = vectorMeasure / measure;
        surface.vectorMeasure += vectorMeasure;
        surface.measure += measure;

        if (_settings.mode == IntegrationMode::EfficientDifferential)
        {
            const double weightStart = _settings.profileIntegrationInitialization
                                           ? MPI_Wtime()
                                           : 0.0;
            MergeSurfaceDifferentialWeights(
                surface.efficientStencil, points, surface.nodes[0],
                surface.nodeCoordinates[0], measure, unitNormal, 0);
            MergeSurfaceDifferentialWeights(
                surface.efficientStencil, points, surface.nodes[1],
                surface.nodeCoordinates[1], measure, unitNormal, 1);
            MergeSurfaceGradientWeights(
                surface.efficientStencil, points, measure);
            if (_settings.profileIntegrationInitialization)
            {
                _integrationInitializationTiming.internalSurfaceSeconds +=
                    MPI_Wtime() - weightStart;
                _integrationInitializationTiming.internalSurfaceCalls++;
            }
        }
        else
        {
            const double quadratureStart = _settings.profileIntegrationInitialization
                                               ? MPI_Wtime()
                                               : 0.0;
            if (_mesh->getDim() == 2)
            {
                Geom::Elem::Quadrature quadrature(
                    Geom::Elem::Element{Geom::Elem::Line2},
                    _settings.SurfaceQuadraturePolynomialDegree());
                for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
                {
                    const auto [parametric, referenceWeight] = quadrature.GetQuadraturePointInfo(iG);
                    const real left = 0.5 * (1.0 - parametric[0]);
                    const real right = 0.5 * (1.0 + parametric[0]);
                    const Vector3 physical = left * points[0].coordinate + right * points[1].coordinate;
                    const real physicalWeight = referenceWeight * measure / 2.0;
                    surface.quadrature.push_back(
                        {physical, unitNormal * physicalWeight, physicalWeight});
                }
            }
            else
            {
                Geom::Elem::Quadrature quadrature(
                    Geom::Elem::Element{Geom::Elem::Tri3},
                    _settings.SurfaceQuadraturePolynomialDegree());
                for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
                {
                    const auto [parametric, referenceWeight] = quadrature.GetQuadraturePointInfo(iG);
                    const Vector3 physical =
                        points[0].coordinate +
                        parametric[0] * (points[1].coordinate - points[0].coordinate) +
                        parametric[1] * (points[2].coordinate - points[0].coordinate);
                    const real physicalWeight = referenceWeight * 2.0 * measure;
                    surface.quadrature.push_back(
                        {physical, unitNormal * physicalWeight, physicalWeight});
                }
            }
            if (_settings.profileIntegrationInitialization)
            {
                _integrationInitializationTiming.internalSurfaceSeconds +=
                    MPI_Wtime() - quadratureStart;
                _integrationInitializationTiming.internalSurfaceCalls++;
            }
        }

        if (_settings.retainMicroGeometry)
        {
            MicroSurface micro;
            micro.primalCell = iCell;
            micro.primalFace = iFace;
            micro.nPoints = _mesh->getDim();
            micro.vectorMeasure = vectorMeasure;
            micro.measure = measure;
            for (int i = 0; i < micro.nPoints; i++)
                micro.points[static_cast<std::size_t>(i)] = points[static_cast<std::size_t>(i)].coordinate;
            surface.microSurfaces.push_back(std::move(micro));
        }
    }

    void DualGeometry::AddBoundarySimplex(
        NodeControlVolume &controlVolume,
        index iCell,
        index iFace,
        const std::vector<AffinePoint> &points) const
    {
        const Vector3 frameAnchor = points.front().coordinate;
        const Vector3 outwardDirection =
            MakeFacePoint(iFace, &frameAnchor).coordinate -
            MakeCellPoint(iCell, &frameAnchor).coordinate;
        const Vector3 vectorMeasure = OrientedSurfaceVector(points, outwardDirection);
        const real measure = vectorMeasure.norm();
        DNDS_check_throw_info(measure > verySmallReal,
                              "NCFV generated a degenerate dual boundary surface");
        const Vector3 unitNormal = vectorMeasure / measure;

        BoundaryPiece piece;
        piece.zone = _topology.FaceElemInfo()(iFace, 0).zone;
        piece.nPoints = _mesh->getDim();
        piece.vectorMeasure = vectorMeasure;
        piece.measure = measure;
        for (int i = 0; i < piece.nPoints; i++)
            piece.points[static_cast<std::size_t>(i)] = points[static_cast<std::size_t>(i)];

        if (_settings.mode == IntegrationMode::EfficientDifferential)
        {
            const double weightStart = _settings.profileIntegrationInitialization
                                           ? MPI_Wtime()
                                           : 0.0;
            MergeBoundaryWeights(piece.efficientStencil, points, measure);
            if (_settings.profileIntegrationInitialization)
            {
                _integrationInitializationTiming.boundarySurfaceSeconds +=
                    MPI_Wtime() - weightStart;
                _integrationInitializationTiming.boundarySurfaceCalls++;
            }
        }
        else
        {
            const double quadratureStart = _settings.profileIntegrationInitialization
                                               ? MPI_Wtime()
                                               : 0.0;
            if (_mesh->getDim() == 2)
            {
                Geom::Elem::Quadrature quadrature(
                    Geom::Elem::Element{Geom::Elem::Line2},
                    _settings.SurfaceQuadraturePolynomialDegree());
                for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
                {
                    const auto [parametric, referenceWeight] = quadrature.GetQuadraturePointInfo(iG);
                    const real left = 0.5 * (1.0 - parametric[0]);
                    const real right = 0.5 * (1.0 + parametric[0]);
                    const Vector3 physical = left * points[0].coordinate + right * points[1].coordinate;
                    const real physicalWeight = referenceWeight * measure / 2.0;
                    piece.quadrature.push_back(
                        {physical, unitNormal * physicalWeight, physicalWeight});
                }
            }
            else
            {
                Geom::Elem::Quadrature quadrature(
                    Geom::Elem::Element{Geom::Elem::Tri3},
                    _settings.SurfaceQuadraturePolynomialDegree());
                for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
                {
                    const auto [parametric, referenceWeight] = quadrature.GetQuadraturePointInfo(iG);
                    const Vector3 physical =
                        points[0].coordinate +
                        parametric[0] * (points[1].coordinate - points[0].coordinate) +
                        parametric[1] * (points[2].coordinate - points[0].coordinate);
                    const real physicalWeight = referenceWeight * 2.0 * measure;
                    piece.quadrature.push_back(
                        {physical, unitNormal * physicalWeight, physicalWeight});
                }
            }
            if (_settings.profileIntegrationInitialization)
            {
                _integrationInitializationTiming.boundarySurfaceSeconds +=
                    MPI_Wtime() - quadratureStart;
                _integrationInitializationTiming.boundarySurfaceCalls++;
            }
        }
        if (_settings.mode == IntegrationMode::EfficientDifferential)
        {
            DNDS_check_throw_info(!piece.efficientStencil.empty() &&
                                      piece.quadrature.empty(),
                                  "Efficient NCFV boundary stencil is empty or stores Gauss points");
            CheckStencilNodes(piece.efficientStencil, _mesh->NumNodeProc());
        }
        else
            DNDS_check_throw_info(piece.efficientStencil.empty() &&
                                      !piece.quadrature.empty(),
                                  "Traditional NCFV boundary integration storage is invalid");
        controlVolume.boundaryPieces.push_back(std::move(piece));
    }

    void DualGeometry::BuildOwnedEdgeSurfaces()
    {
        _edgeSurfaces.clear();
        _edgeSurfaces.resize(static_cast<std::size_t>(_topology.NumEdge()));
        _ghostEdgeSurfaces.clear();

        const auto buildSurface = [&](index iEdge, EdgeControlSurface &surface)
        {
            surface.edge = iEdge;
            surface.nodes = {_topology.Edge2Node()(iEdge, 0),
                             _topology.Edge2Node()(iEdge, 1)};
            const Vector3 frameAnchor = _mesh->coords[surface.nodes[0]];
            surface.nodeCoordinates[0] = frameAnchor;
            surface.nodeCoordinates[1] =
                CoordinateInFrame(surface.nodes[1], frameAnchor);
            const AffinePoint edgePoint =
                MakeEdgePoint(surface.nodes[0], surface.nodes[1], &frameAnchor);
            surface.edgePoint = edgePoint.coordinate;

            if (_settings.mode == IntegrationMode::EfficientDifferential)
            {
                const double layoutStart = _settings.profileIntegrationInitialization
                                               ? MPI_Wtime()
                                               : 0.0;
                std::vector<index> supportNodes;
                supportNodes.reserve(
                    static_cast<std::size_t>(_topology.Edge2Cell()[iEdge].size()) *
                    MaxAffineSupportNodes);
                for (index iCell : _topology.Edge2Cell()[iEdge])
                {
                    DNDS_check_throw_info(iCell >= 0,
                                          "NCFV edge geometry has an unresolved parent cell");
                    const auto cell = _mesh->GetCellElement(iCell);
                    for (int iNode = 0; iNode < cell.GetNumVertices(); iNode++)
                        supportNodes.push_back(_mesh->cell2node(iCell, iNode));
                }
                std::sort(supportNodes.begin(), supportNodes.end());
                supportNodes.erase(
                    std::unique(supportNodes.begin(), supportNodes.end()),
                    supportNodes.end());
                InitializeStencil(surface.efficientStencil, supportNodes);
                if (_settings.profileIntegrationInitialization)
                {
                    _integrationInitializationTiming.internalSurfaceSeconds +=
                        MPI_Wtime() - layoutStart;
                    _integrationInitializationTiming.internalSurfaceCalls++;
                }
            }

            for (index iCell : _topology.Edge2Cell()[iEdge])
            {
                DNDS_check_throw_info(iCell >= 0,
                                      "NCFV edge geometry has an unresolved parent cell");
                auto cell = _mesh->GetCellElement(iCell);
                const AffinePoint cellPoint = MakeCellPoint(iCell, &frameAnchor);
                int matches = 0;
                for (int iLocalFace = 0; iLocalFace < cell.GetNumFaces(); iLocalFace++)
                {
                    const std::vector<index> faceNodes =
                        FaceVertexNodes(_mesh, iCell, iLocalFace);
                    if (!ContainsEdge(faceNodes, surface.nodes[0], surface.nodes[1]))
                        continue;
                    const index iFace = _topology.CellFace(iCell, iLocalFace);
                    if (_mesh->getDim() == 2)
                        AddEdgeSurfaceSimplex(surface, iCell, iFace, {edgePoint, cellPoint});
                    else
                        AddEdgeSurfaceSimplex(
                            surface, iCell, iFace,
                            {edgePoint, MakeFacePoint(iFace, &frameAnchor), cellPoint});
                    matches++;
                }
                const int expectedMatches = _mesh->getDim() == 2 ? 1 : 2;
                DNDS_check_throw_info(matches == expectedMatches,
                                      "NCFV failed to enumerate every cell-face chain around an edge");
            }
            DNDS_check_throw_info(surface.measure > verySmallReal &&
                                      surface.vectorMeasure.norm() > verySmallReal,
                                  "NCFV generated an empty or folded primal-edge macro surface");
            if (_settings.mode == IntegrationMode::EfficientDifferential)
            {
                DNDS_check_throw_info(!surface.efficientStencil.empty() &&
                                          surface.quadrature.empty(),
                                      "Efficient NCFV macro-surface stencil is empty or stores Gauss points");
                CheckStencilNodes(surface.efficientStencil, _mesh->NumNodeProc());
            }
            else
                DNDS_check_throw_info(surface.efficientStencil.empty() &&
                                          !surface.quadrature.empty(),
                                      "Traditional NCFV surface integration storage is invalid");
        };

        for (index iEdge = 0; iEdge < _topology.NumEdge(); iEdge++)
            buildSurface(iEdge, _edgeSurfaces[static_cast<std::size_t>(iEdge)]);

        if (_settings.mode == IntegrationMode::EfficientDifferential &&
            _buildGhostEdgeSurfaces)
        {
            _ghostEdgeSurfaces.resize(
                static_cast<std::size_t>(_topology.NumEdgeGhost()));
            for (index iGhost = 0; iGhost < _topology.NumEdgeGhost(); iGhost++)
            {
                const index iEdge = _topology.NumEdge() + iGhost;
                buildSurface(
                    iEdge,
                    _ghostEdgeSurfaces[static_cast<std::size_t>(iGhost)]);
            }
        }
    }

    void DualGeometry::BuildOwnedNodeVolumes()
    {
        _nodeVolumes.clear();
        _nodeVolumes.resize(static_cast<std::size_t>(_mesh->NumNode()));
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            NodeControlVolume &controlVolume = _nodeVolumes[static_cast<std::size_t>(iNode)];
            controlVolume.node = iNode;
            const AffinePoint nodePoint = MakeNodePoint(iNode);
            const Vector3 frameAnchor = nodePoint.coordinate;

            std::vector<index> incidentCells(_mesh->node2cell[iNode].begin(),
                                             _mesh->node2cell[iNode].end());
            std::sort(incidentCells.begin(), incidentCells.end(), [this](index left, index right)
                      { return _mesh->CellIndexLocal2Global(left) < _mesh->CellIndexLocal2Global(right); });
            incidentCells.erase(std::unique(incidentCells.begin(), incidentCells.end()),
                                incidentCells.end());

            if (_settings.mode == IntegrationMode::EfficientDifferential)
            {
                const double layoutStart = _settings.profileIntegrationInitialization
                                               ? MPI_Wtime()
                                               : 0.0;
                std::vector<index> supportNodes;
                supportNodes.reserve(incidentCells.size() * MaxAffineSupportNodes);
                for (index iCell : incidentCells)
                {
                    DNDS_check_throw_info(iCell >= 0,
                                          "NCFV owned node has an unresolved incident cell");
                    const auto cell = _mesh->GetCellElement(iCell);
                    for (int iLocalNode = 0; iLocalNode < cell.GetNumVertices(); iLocalNode++)
                        supportNodes.push_back(_mesh->cell2node(iCell, iLocalNode));
                }
                std::sort(supportNodes.begin(), supportNodes.end());
                supportNodes.erase(
                    std::unique(supportNodes.begin(), supportNodes.end()),
                    supportNodes.end());
                InitializeStencil(controlVolume.pointRecoveryStencil, supportNodes);
                if (_settings.profileIntegrationInitialization)
                {
                    _integrationInitializationTiming.volumeSeconds +=
                        MPI_Wtime() - layoutStart;
                    _integrationInitializationTiming.volumeCalls++;
                }
            }

            for (index iCell : incidentCells)
            {
                DNDS_check_throw_info(iCell >= 0,
                                      "NCFV owned node has an unresolved incident cell");
                auto cell = _mesh->GetCellElement(iCell);
                const AffinePoint cellPoint = MakeCellPoint(iCell, &frameAnchor);
                for (int iLocalFace = 0; iLocalFace < cell.GetNumFaces(); iLocalFace++)
                {
                    const auto face = cell.ObtainFace(iLocalFace);
                    const std::vector<index> faceNodes =
                        FaceVertexNodes(_mesh, iCell, iLocalFace);
                    if (!ContainsNode(faceNodes, iNode))
                        continue;
                    const index iFace = _topology.CellFace(iCell, iLocalFace);
                    const bool isBoundary = _topology.FaceIsBoundary(iFace);

                    if (_mesh->getDim() == 2)
                    {
                        DNDS_check_throw_info(faceNodes.size() == 2,
                                              "A 2-D O1 primal edge must have two vertices");
                        const index neighbor = faceNodes[0] == iNode ? faceNodes[1] : faceNodes[0];
                        const AffinePoint edgePoint =
                            MakeEdgePoint(iNode, neighbor, &frameAnchor);
                        AddVolumeSimplex(controlVolume, iCell,
                                         {nodePoint, edgePoint, cellPoint});
                        if (isBoundary)
                            AddBoundarySimplex(controlVolume, iCell, iFace,
                                               {nodePoint, edgePoint});
                    }
                    else
                    {
                        const AffinePoint facePoint =
                            MakeFacePoint(iFace, &frameAnchor);
                        for (int iLocalEdge = 0; iLocalEdge < face.GetNumFaces(); iLocalEdge++)
                        {
                            const std::vector<index> edgeNodes =
                                FaceEdgeNodes(face, faceNodes, iLocalEdge);
                            if (!ContainsNode(edgeNodes, iNode))
                                continue;
                            DNDS_check_throw_info(edgeNodes.size() == 2,
                                                  "A 3-D O1 primal edge must have two vertices");
                            const index neighbor = edgeNodes[0] == iNode ? edgeNodes[1] : edgeNodes[0];
                            const AffinePoint edgePoint =
                                MakeEdgePoint(iNode, neighbor, &frameAnchor);
                            AddVolumeSimplex(controlVolume, iCell,
                                             {nodePoint, edgePoint, facePoint, cellPoint});
                            if (isBoundary)
                                AddBoundarySimplex(controlVolume, iCell, iFace,
                                                   {nodePoint, edgePoint, facePoint});
                        }
                    }
                }
            }

            DNDS_check_throw_info(controlVolume.moments.measure > verySmallReal,
                                  "NCFV generated a node with zero dual volume");
            controlVolume.lengthScale = std::pow(
                controlVolume.moments.measure,
                1.0 / static_cast<real>(_mesh->getDim()));
            controlVolume.referenceLengths = ReferenceLengthsFromBounds(
                controlVolume.lowerOffset, controlVolume.upperOffset, _mesh->getDim());
            if (_settings.mode == IntegrationMode::EfficientDifferential)
            {
                const double normalizationStart = _settings.profileIntegrationInitialization
                                                      ? MPI_Wtime()
                                                      : 0.0;
                controlVolume.pointRecoveryStencil.erase(
                    std::remove_if(
                        controlVolume.pointRecoveryStencil.begin(),
                        controlVolume.pointRecoveryStencil.end(),
                        [](const EfficientPointRecoveryNode &entry)
                        { return entry.gradientWeight.squaredNorm() == 0; }),
                    controlVolume.pointRecoveryStencil.end());
                for (auto &entry : controlVolume.pointRecoveryStencil)
                    entry.gradientWeight /= controlVolume.moments.measure;
                if (_settings.profileIntegrationInitialization)
                {
                    _integrationInitializationTiming.volumeNormalizationSeconds +=
                        MPI_Wtime() - normalizationStart;
                    _integrationInitializationTiming.volumeNormalizationCalls++;
                }
                DNDS_check_throw_info(!controlVolume.pointRecoveryStencil.empty(),
                                      "Efficient NCFV point-recovery stencil is empty");
                CheckStencilNodes(
                    controlVolume.pointRecoveryStencil, _mesh->NumNodeProc());
                DNDS_check_throw_info(controlVolume.volumeQuadrature.empty(),
                                      "Efficient NCFV unexpectedly stored volume Gauss points");
            }
            else
            {
                DNDS_check_throw_info(!controlVolume.volumeQuadrature.empty(),
                                      "Traditional NCFV did not store volume quadrature points");
                DNDS_check_throw_info(controlVolume.pointRecoveryStencil.empty(),
                                      "Traditional NCFV unexpectedly stored differential recovery weights");
            }
        }
    }

    void DualGeometry::BuildCommunicationFields()
    {
        _nodeMoments.InitPair("NCFV.nodeMoments", _mpi);
        _nodeMoments.father->Resize(_mesh->NumNode());
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const RawMoments &moments = _nodeVolumes[static_cast<std::size_t>(iNode)].moments;
            for (int i = 0; i < 10; i++)
                _nodeMoments(iNode, i) = 0;
            _nodeMoments(iNode, 0) = moments.measure;
            _nodeMoments(iNode, 1) = moments.first.x();
            _nodeMoments(iNode, 2) = moments.first.y();
            _nodeMoments(iNode, 3) = moments.first.z();
            _nodeMoments(iNode, 4) = moments.second(0, 0);
            _nodeMoments(iNode, 5) = moments.second(1, 1);
            _nodeMoments(iNode, 6) = moments.second(2, 2);
            _nodeMoments(iNode, 7) = moments.second(0, 1);
            _nodeMoments(iNode, 8) = moments.second(1, 2);
            _nodeMoments(iNode, 9) = moments.second(2, 0);
        }
        _nodeMoments.BorrowAndPull(_mesh->coords);

        _nodeReferenceLengths.InitPair("NCFV.nodeReferenceLengths", _mpi);
        _nodeReferenceLengths.father->Resize(_mesh->NumNode());
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            for (int d = 0; d < 3; d++)
                _nodeReferenceLengths(iNode, d) =
                    _nodeVolumes[static_cast<std::size_t>(iNode)].referenceLengths(d);
        _nodeReferenceLengths.BorrowAndPull(_mesh->coords);

        _edgeMetrics.InitPair("NCFV.edgeMetrics", _mpi);
        _edgeMetrics.father->Resize(_topology.NumEdge());
        for (index iEdge = 0; iEdge < _topology.NumEdge(); iEdge++)
        {
            const auto &surface = _edgeSurfaces[static_cast<std::size_t>(iEdge)];
            _edgeMetrics(iEdge, 0) = surface.vectorMeasure.x();
            _edgeMetrics(iEdge, 1) = surface.vectorMeasure.y();
            _edgeMetrics(iEdge, 2) = surface.vectorMeasure.z();
            _edgeMetrics(iEdge, 3) = surface.measure;
        }
        _edgeMetrics.BorrowAndPull(const_cast<Geom::tAdjPair &>(_topology.Edge2Node()));
    }

    void DualGeometry::CheckClosure()
    {
        real localMaximum = 0;
        index localFailureNode = UnInitIndex;
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            Vector3 closure = Vector3::Zero();
            real measureSum = 0;
            for (const NodeEdgeIncidence &incidence : _topology.Node2Edge(iNode))
            {
                Vector3 vectorMeasure;
                vectorMeasure << _edgeMetrics(incidence.edge, 0),
                    _edgeMetrics(incidence.edge, 1),
                    _edgeMetrics(incidence.edge, 2);
                closure += incidence.outwardSign * vectorMeasure;
                measureSum += _edgeMetrics(incidence.edge, 3);
            }
            for (const auto &piece : _nodeVolumes[static_cast<std::size_t>(iNode)].boundaryPieces)
            {
                closure += piece.vectorMeasure;
                measureSum += piece.measure;
            }
            const real relativeError = closure.norm() / std::max(measureSum, verySmallReal);
            if (relativeError > localMaximum)
            {
                localMaximum = relativeError;
                localFailureNode = iNode;
            }
        }
        MPI_Allreduce(&localMaximum, &_maximumClosureError, 1, DNDS_MPI_REAL, MPI_MAX, _mpi.comm);
        if (_settings.checkGeometryClosure)
            DNDS_check_throw_info(
                _maximumClosureError <= _settings.closureTolerance,
                fmt::format("NCFV dual-area closure failed: global max={}, local worst node={}",
                            _maximumClosureError, localFailureNode));
    }

    std::vector<index> DualGeometry::CollectNodeDependencies() const
    {
        DNDS_check_throw_info(
            !_nodeIndicesRemapped,
            "NCFV integration dependencies must be collected before halo remapping");
        std::vector<index> globals;
        const auto append = [&](index meshLocal)
        {
            DNDS_check_throw_info(
                meshLocal >= 0 && meshLocal < _mesh->NumNodeProc(),
                "NCFV integration data contains an invalid mesh-local node");
            globals.push_back(_mesh->NodeIndexLocal2Global(meshLocal));
        };
        const auto appendPoint = [&](const AffinePoint &point)
        {
            for (const auto &coefficient : point.support)
                append(coefficient.node);
        };
        for (const auto &volume : _nodeVolumes)
        {
            for (const auto &entry : volume.pointRecoveryStencil)
                append(entry.node);
            for (const auto &piece : volume.boundaryPieces)
            {
                for (const auto &entry : piece.efficientStencil)
                    append(entry.node);
                for (int iPoint = 0; iPoint < piece.nPoints; iPoint++)
                    appendPoint(piece.points[static_cast<std::size_t>(iPoint)]);
            }
        }
        const auto appendSurface = [&](const EdgeControlSurface &surface)
        {
            append(surface.nodes[0]);
            append(surface.nodes[1]);
            for (const auto &entry : surface.efficientStencil)
                append(entry.node);
        };
        for (const auto &surface : _edgeSurfaces)
            appendSurface(surface);
        for (const auto &surface : _ghostEdgeSurfaces)
            appendSurface(surface);
        std::sort(globals.begin(), globals.end());
        globals.erase(std::unique(globals.begin(), globals.end()), globals.end());
        return globals;
    }

    void DualGeometry::RemapNodeIndices(
        const std::function<index(index)> &meshLocalToHalo)
    {
        DNDS_check_throw_info(
            !_nodeIndicesRemapped,
            "NCFV integration node indices were already remapped");
        const auto remapPoint = [&](AffinePoint &point)
        {
            for (auto &coefficient : point.support)
                coefficient.node = meshLocalToHalo(coefficient.node);
            std::sort(point.support.begin(), point.support.end(),
                      [](const auto &left, const auto &right)
                      { return left.node < right.node; });
        };
        const auto remapStencil = [&](auto &stencil)
        {
            for (auto &entry : stencil)
                entry.node = meshLocalToHalo(entry.node);
            std::sort(stencil.begin(), stencil.end(),
                      [](const auto &left, const auto &right)
                      { return left.node < right.node; });
        };
        for (auto &volume : _nodeVolumes)
        {
            const index remappedNode = meshLocalToHalo(volume.node);
            DNDS_check_throw_info(
                remappedNode == volume.node,
                "NCFV exact node halo changed an owned-node local index");
            volume.node = remappedNode;
            remapStencil(volume.pointRecoveryStencil);
            for (auto &piece : volume.boundaryPieces)
            {
                remapStencil(piece.efficientStencil);
                for (int iPoint = 0; iPoint < piece.nPoints; iPoint++)
                    remapPoint(piece.points[static_cast<std::size_t>(iPoint)]);
            }
        }
        const auto remapSurface = [&](EdgeControlSurface &surface)
        {
            surface.nodes[0] = meshLocalToHalo(surface.nodes[0]);
            surface.nodes[1] = meshLocalToHalo(surface.nodes[1]);
            remapStencil(surface.efficientStencil);
        };
        for (auto &surface : _edgeSurfaces)
            remapSurface(surface);
        for (auto &surface : _ghostEdgeSurfaces)
            remapSurface(surface);
        _nodeIndicesRemapped = true;
    }

    void DualGeometry::Build()
    {
        _integrationInitializationTiming = {};
        BuildConstructionPoints();
        BuildOwnedEdgeSurfaces();
        BuildOwnedNodeVolumes();
        BuildCommunicationFields();
        CheckClosure();

        index localMicroVolumes = 0;
        index localVolumeQuadrature = 0;
        index localSurfaceQuadrature = 0;
        index localBoundaryQuadrature = 0;
        for (const auto &volume : _nodeVolumes)
        {
            localMicroVolumes += static_cast<index>(volume.microVolumes.size());
            localVolumeQuadrature += static_cast<index>(volume.volumeQuadrature.size());
            for (const auto &piece : volume.boundaryPieces)
                localBoundaryQuadrature += static_cast<index>(piece.quadrature.size());
        }
        for (const auto &surface : _edgeSurfaces)
            localSurfaceQuadrature += static_cast<index>(surface.quadrature.size());
        index globalMicroVolumes = 0;
        index globalVolumeQuadrature = 0;
        index globalSurfaceQuadrature = 0;
        index globalBoundaryQuadrature = 0;
        MPI_Allreduce(&localMicroVolumes, &globalMicroVolumes, 1, DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
        MPI_Allreduce(&localVolumeQuadrature, &globalVolumeQuadrature, 1, DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
        MPI_Allreduce(&localSurfaceQuadrature, &globalSurfaceQuadrature, 1, DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
        MPI_Allreduce(&localBoundaryQuadrature, &globalBoundaryQuadrature, 1, DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);

        if (_mpi.rank == 0)
            log() << "NCFV dual geometry: mode="
                  << (_settings.mode == IntegrationMode::EfficientDifferential
                          ? "EfficientDifferential"
                          : "TraditionalQuadrature")
                  << ", volume quadrature order=" << _settings.quadratureOrder
                  << ", surface quadrature order=" << _settings.surfaceQuadratureOrder
                  << ", retained micro-volumes=" << globalMicroVolumes
                  << ", stored volume quadrature=" << globalVolumeQuadrature
                  << ", stored internal-surface quadrature=" << globalSurfaceQuadrature
                  << ", stored boundary-surface quadrature=" << globalBoundaryQuadrature
                  << ", max closure=" << _maximumClosureError << std::endl;
    }
}
