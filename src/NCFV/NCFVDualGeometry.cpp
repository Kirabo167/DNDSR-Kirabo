#include "NCFVDualGeometry.hpp"

#include "DNDS/Errors.hpp"
#include "Geom/Quadrature.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace DNDS::NCFV
{
    namespace
    {
        void MergeVectorWeights(
            std::vector<SparseVectorWeight> &destination,
            const std::vector<SparseVectorWeight> &source,
            real scale = 1.0)
        {
            std::map<index, Vector3> merged;
            for (const auto &weight : destination)
            {
                auto [iterator, inserted] = merged.try_emplace(weight.node, Vector3::Zero());
                iterator->second += weight.value;
            }
            for (const auto &weight : source)
            {
                auto [iterator, inserted] = merged.try_emplace(weight.node, Vector3::Zero());
                iterator->second += scale * weight.value;
            }
            destination.clear();
            destination.reserve(merged.size());
            for (const auto &[node, value] : merged)
                if (value.squaredNorm() > sqr(verySmallReal))
                    destination.push_back({node, value});
        }

        void MergeFluxWeights(
            std::vector<SparseMatrixWeight> &destination,
            const std::vector<SparseVectorWeight> &source,
            const Vector3 &unitNormal)
        {
            std::map<index, Matrix3> merged;
            for (const auto &weight : destination)
            {
                auto [iterator, inserted] = merged.try_emplace(weight.node, Matrix3::Zero());
                iterator->second += weight.value;
            }
            for (const auto &weight : source)
            {
                auto [iterator, inserted] = merged.try_emplace(weight.node, Matrix3::Zero());
                iterator->second += weight.value * unitNormal.transpose();
            }
            destination.clear();
            destination.reserve(merged.size());
            for (const auto &[node, value] : merged)
                if (value.squaredNorm() > sqr(verySmallReal))
                    destination.push_back({node, value});
        }

        void MergeScalarWeights(
            std::vector<SparseScalarWeight> &destination,
            const std::vector<AffinePoint> &points,
            real measure)
        {
            std::map<index, real> merged;
            for (const auto &weight : destination)
                merged[weight.node] += weight.value;
            const real pointFactor = measure / static_cast<real>(points.size());
            for (const auto &point : points)
                for (const auto &coefficient : point.support)
                    merged[coefficient.node] += pointFactor * coefficient.coefficient;

            destination.clear();
            destination.reserve(merged.size());
            for (const auto &[node, value] : merged)
                if (std::abs(value) > verySmallReal)
                    destination.push_back({node, value});
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
        DNDS_check_throw_info(points.size() >= 2 && points.size() <= 4,
                              "DifferentialWeights accepts line, triangle, or tetrahedron vertices");
        const real n = static_cast<real>(points.size());
        std::vector<Vector3> displacement(points.size(), Vector3::Zero());
        Vector3 displacementSum = Vector3::Zero();
        std::set<index> supportNodes{anchorNode};
        for (std::size_t i = 0; i < points.size(); i++)
        {
            displacement[i] = points[i].coordinate - anchorCoordinate;
            displacementSum += displacement[i];
            for (const auto &coefficient : points[i].support)
                supportNodes.insert(coefficient.node);
        }

        const real secondFactor = measure / (2.0 * n * (n + 1.0));
        std::vector<SparseVectorWeight> result;
        result.reserve(supportNodes.size());
        for (index node : supportNodes)
        {
            const real delta = node == anchorNode ? 1.0 : 0.0;
            real coefficientSum = 0;
            Vector3 displacementCoefficientSum = Vector3::Zero();
            for (std::size_t i = 0; i < points.size(); i++)
            {
                real coefficient = 0;
                for (const auto &entry : points[i].support)
                    if (entry.node == node)
                        coefficient += entry.coefficient;
                coefficientSum += coefficient - delta;
                displacementCoefficientSum += displacement[i] * (coefficient - delta);
            }

            Vector3 value = measure / n * displacementSum * delta;
            value += secondFactor *
                     (displacementSum * coefficientSum + displacementCoefficientSum);
            if (value.squaredNorm() > sqr(verySmallReal))
                result.push_back({node, value});
        }
        return result;
    }

    AffinePoint DualGeometry::MakeNodePoint(index iNode) const
    {
        AffinePoint point;
        point.coordinate = _mesh->coords[iNode];
        point.support.push_back({iNode, 1.0});
        return point;
    }

    AffinePoint DualGeometry::MakeAveragePoint(const std::vector<index> &nodes) const
    {
        DNDS_check_throw_info(!nodes.empty(), "Cannot average an empty primal-vertex set");
        AffinePoint point;
        const real coefficient = 1.0 / static_cast<real>(nodes.size());
        point.support.reserve(nodes.size());
        for (index node : nodes)
        {
            point.coordinate += coefficient * _mesh->coords[node];
            point.support.push_back({node, coefficient});
        }
        std::sort(point.support.begin(), point.support.end(), [](const auto &left, const auto &right)
                  { return left.node < right.node; });
        return point;
    }

    AffinePoint DualGeometry::MakeCellPoint(index iCell) const
    {
        const auto element = _mesh->GetCellElement(iCell);
        std::vector<index> vertices(static_cast<std::size_t>(element.GetNumVertices()));
        for (int i = 0; i < element.GetNumVertices(); i++)
            vertices[static_cast<std::size_t>(i)] = _mesh->cell2node(iCell, i);
        return MakeAveragePoint(vertices);
    }

    AffinePoint DualGeometry::MakeFacePoint(index iFace) const
    {
        const auto element = _topology.GetFaceElement(iFace);
        std::vector<index> vertices(static_cast<std::size_t>(element.GetNumVertices()));
        for (int i = 0; i < element.GetNumVertices(); i++)
            vertices[static_cast<std::size_t>(i)] = _topology.Face2Node()(iFace, i);
        return MakeAveragePoint(vertices);
    }

    AffinePoint DualGeometry::MakeEdgePoint(index node0, index node1) const
    {
        return MakeAveragePoint({node0, node1});
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

        for (const Vector3 &point : coordinates)
        {
            const Vector3 offset = point - _mesh->coords[controlVolume.node];
            controlVolume.lowerOffset = controlVolume.lowerOffset.cwiseMin(offset);
            controlVolume.upperOffset = controlVolume.upperOffset.cwiseMax(offset);
        }

        if (_settings.mode == IntegrationMode::EfficientDifferential)
        {
            AccumulateMoments(controlVolume.moments, ExactSimplexMoments(coordinates, measure));
            MergeVectorWeights(
                controlVolume.pointRecoveryWeights,
                DifferentialWeights(points, controlVolume.node,
                                    _mesh->coords[controlVolume.node], measure));
        }
        else
        {
            const auto simplexElement = dimension == 2
                                            ? Geom::Elem::Element{Geom::Elem::Tri3}
                                            : Geom::Elem::Element{Geom::Elem::Tet4};
            Geom::Elem::Quadrature quadrature(simplexElement, _settings.quadratureOrder);
            const real jacobian = dimension == 2 ? 2.0 * measure : 6.0 * measure;
            for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
            {
                const auto [parametric, referenceWeight] = quadrature.GetQuadraturePointInfo(iG);
                Vector3 physical = points[0].coordinate;
                physical += parametric[0] * (points[1].coordinate - points[0].coordinate);
                physical += parametric[1] * (points[2].coordinate - points[0].coordinate);
                if (dimension == 3)
                    physical += parametric[2] * (points[3].coordinate - points[0].coordinate);
                const real physicalWeight = referenceWeight * jacobian;
                controlVolume.volumeQuadrature.push_back({physical, physicalWeight});
                controlVolume.moments.measure += physicalWeight;
                controlVolume.moments.first += physicalWeight * physical;
                controlVolume.moments.second += physicalWeight * physical * physical.transpose();
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
            _mesh->coords[surface.nodes[1]] - _mesh->coords[surface.nodes[0]];
        const Vector3 vectorMeasure = OrientedSurfaceVector(points, edgeDirection);
        const real measure = vectorMeasure.norm();
        DNDS_check_throw_info(measure > verySmallReal,
                              "NCFV generated a degenerate dual edge surface");
        const Vector3 unitNormal = vectorMeasure / measure;
        surface.vectorMeasure += vectorMeasure;
        surface.measure += measure;

        if (_settings.mode == IntegrationMode::EfficientDifferential)
        {
            const auto leftWeights = DifferentialWeights(
                points, surface.nodes[0], _mesh->coords[surface.nodes[0]], measure);
            const auto rightWeights = DifferentialWeights(
                points, surface.nodes[1], _mesh->coords[surface.nodes[1]], measure);
            MergeVectorWeights(surface.leftStateWeights, leftWeights);
            MergeVectorWeights(surface.rightStateWeights, rightWeights);
            MergeFluxWeights(surface.leftFluxWeights, leftWeights, unitNormal);
            MergeFluxWeights(surface.rightFluxWeights, rightWeights, unitNormal);
            MergeScalarWeights(surface.gradientIntegralWeights, points, measure);
        }
        else
        {
            if (_mesh->getDim() == 2)
            {
                Geom::Elem::Quadrature quadrature(
                    Geom::Elem::Element{Geom::Elem::Line2}, _settings.quadratureOrder);
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
                    Geom::Elem::Element{Geom::Elem::Tri3}, _settings.quadratureOrder);
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
        const Vector3 outwardDirection =
            _facePoints.at(static_cast<std::size_t>(iFace)) -
            _cellPoints.at(static_cast<std::size_t>(iCell));
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
            MergeScalarWeights(piece.gradientIntegralWeights, points, measure);
        else
        {
            if (_mesh->getDim() == 2)
            {
                Geom::Elem::Quadrature quadrature(
                    Geom::Elem::Element{Geom::Elem::Line2}, _settings.quadratureOrder);
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
                    Geom::Elem::Element{Geom::Elem::Tri3}, _settings.quadratureOrder);
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
        }
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
            surface.edgePoint = _edgePoints[static_cast<std::size_t>(iEdge)];
            const AffinePoint edgePoint = MakeEdgePoint(surface.nodes[0], surface.nodes[1]);

            for (index iCell : _topology.Edge2Cell()[iEdge])
            {
                DNDS_check_throw_info(iCell >= 0,
                                      "NCFV edge geometry has an unresolved parent cell");
                auto cell = _mesh->GetCellElement(iCell);
                const AffinePoint cellPoint = MakeCellPoint(iCell);
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
                            {edgePoint, MakeFacePoint(iFace), cellPoint});
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
                DNDS_check_throw_info(surface.quadrature.empty(),
                                      "Efficient NCFV unexpectedly stored Gauss points");
            else
                DNDS_check_throw_info(!surface.quadrature.empty(),
                                      "Traditional NCFV did not store surface quadrature points");
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

            std::vector<index> incidentCells(_mesh->node2cell[iNode].begin(),
                                             _mesh->node2cell[iNode].end());
            std::sort(incidentCells.begin(), incidentCells.end(), [this](index left, index right)
                      { return _mesh->CellIndexLocal2Global(left) < _mesh->CellIndexLocal2Global(right); });
            incidentCells.erase(std::unique(incidentCells.begin(), incidentCells.end()),
                                incidentCells.end());

            for (index iCell : incidentCells)
            {
                DNDS_check_throw_info(iCell >= 0,
                                      "NCFV owned node has an unresolved incident cell");
                auto cell = _mesh->GetCellElement(iCell);
                const AffinePoint cellPoint = MakeCellPoint(iCell);
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
                        const AffinePoint edgePoint = MakeEdgePoint(iNode, neighbor);
                        AddVolumeSimplex(controlVolume, iCell,
                                         {nodePoint, edgePoint, cellPoint});
                        if (isBoundary)
                            AddBoundarySimplex(controlVolume, iCell, iFace,
                                               {nodePoint, edgePoint});
                    }
                    else
                    {
                        const AffinePoint facePoint = MakeFacePoint(iFace);
                        for (int iLocalEdge = 0; iLocalEdge < face.GetNumFaces(); iLocalEdge++)
                        {
                            const std::vector<index> edgeNodes =
                                FaceEdgeNodes(face, faceNodes, iLocalEdge);
                            if (!ContainsNode(edgeNodes, iNode))
                                continue;
                            DNDS_check_throw_info(edgeNodes.size() == 2,
                                                  "A 3-D O1 primal edge must have two vertices");
                            const index neighbor = edgeNodes[0] == iNode ? edgeNodes[1] : edgeNodes[0];
                            const AffinePoint edgePoint = MakeEdgePoint(iNode, neighbor);
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
                for (auto &weight : controlVolume.pointRecoveryWeights)
                    weight.value /= controlVolume.moments.measure;
                DNDS_check_throw_info(controlVolume.volumeQuadrature.empty(),
                                      "Efficient NCFV unexpectedly stored volume Gauss points");
            }
            else
            {
                DNDS_check_throw_info(!controlVolume.volumeQuadrature.empty(),
                                      "Traditional NCFV did not store volume quadrature points");
                DNDS_check_throw_info(controlVolume.pointRecoveryWeights.empty(),
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

    void DualGeometry::Build()
    {
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
                  << ", retained micro-volumes=" << globalMicroVolumes
                  << ", stored volume quadrature=" << globalVolumeQuadrature
                  << ", stored internal-surface quadrature=" << globalSurfaceQuadrature
                  << ", stored boundary-surface quadrature=" << globalBoundaryQuadrature
                  << ", max closure=" << _maximumClosureError << std::endl;
    }
}
