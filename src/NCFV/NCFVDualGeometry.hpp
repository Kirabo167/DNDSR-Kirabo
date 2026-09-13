/**
 * @file NCFVDualGeometry.hpp
 * @brief Median-dual control volumes generated from primal vertex averages.
 */
#pragma once

#include "NCFVConfig.hpp"
#include "NCFVTopology.hpp"

#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"

#include <array>
#include <unordered_map>
#include <vector>

namespace DNDS::NCFV
{
    using Vector3 = Eigen::Vector3d;
    using Matrix3 = Eigen::Matrix3d;

    struct AffineCoefficient
    {
        index node = UnInitIndex;
        real coefficient = 0;
    };

    /** @brief A dual construction point and its interpolation from primal vertices. */
    struct AffinePoint
    {
        Vector3 coordinate = Vector3::Zero();
        std::vector<AffineCoefficient> support;
    };

    struct SparseVectorWeight
    {
        index node = UnInitIndex;
        Vector3 value = Vector3::Zero();
    };

    /** @brief Scalar affine-integration weight, normally carrying surface measure. */
    struct SparseScalarWeight
    {
        index node = UnInitIndex;
        real value = 0;
    };

    struct SparseMatrixWeight
    {
        index node = UnInitIndex;
        Matrix3 value = Matrix3::Zero();
    };

    struct RawMoments
    {
        real measure = 0;
        Vector3 first = Vector3::Zero();
        Matrix3 second = Matrix3::Zero();
    };

    struct VolumeQuadraturePoint
    {
        Vector3 coordinate = Vector3::Zero();
        real weight = 0;
    };

    struct SurfaceQuadraturePoint
    {
        Vector3 coordinate = Vector3::Zero();
        Vector3 vectorWeight = Vector3::Zero();
        real weight = 0;
    };

    struct MicroVolume
    {
        index primalCell = UnInitIndex;
        int nPoints = 0;
        std::array<Vector3, 4> points{};
        real signedJacobian = 0;
        real measure = 0;
    };

    struct MicroSurface
    {
        index primalCell = UnInitIndex;
        index primalFace = UnInitIndex;
        int nPoints = 0;
        std::array<Vector3, 3> points{};
        Vector3 vectorMeasure = Vector3::Zero();
        real measure = 0;
    };

    struct BoundaryPiece
    {
        Geom::t_index zone = Geom::INTERNAL_ZONE;
        int nPoints = 0;
        std::array<AffinePoint, 3> points{};
        Vector3 vectorMeasure = Vector3::Zero();
        real measure = 0;
        std::vector<SparseScalarWeight> gradientIntegralWeights;
        std::vector<SurfaceQuadraturePoint> quadrature;
    };

    struct NodeControlVolume
    {
        index node = UnInitIndex;
        RawMoments moments;
        // Extrema relative to the primal node, accumulated before optional
        // micro-geometry is discarded. The node belongs to every micro-volume.
        Vector3 lowerOffset = Vector3::Zero();
        Vector3 upperOffset = Vector3::Zero();
        Vector3 referenceLengths = Vector3::Ones();
        // Volume scale is retained for CFL/wall models, not polynomial bases.
        real lengthScale = 0;
        std::vector<SparseVectorWeight> pointRecoveryWeights;
        std::vector<MicroVolume> microVolumes;
        std::vector<VolumeQuadraturePoint> volumeQuadrature;
        std::vector<BoundaryPiece> boundaryPieces;
    };

    struct EdgeControlSurface
    {
        index edge = UnInitIndex;
        std::array<index, 2> nodes{UnInitIndex, UnInitIndex};
        Vector3 edgePoint = Vector3::Zero();
        Vector3 vectorMeasure = Vector3::Zero();
        real measure = 0;
        std::vector<SparseVectorWeight> leftStateWeights;
        std::vector<SparseVectorWeight> rightStateWeights;
        std::vector<SparseMatrixWeight> leftFluxWeights;
        std::vector<SparseMatrixWeight> rightFluxWeights;
        std::vector<SparseScalarWeight> gradientIntegralWeights;
        std::vector<MicroSurface> microSurfaces;
        std::vector<SurfaceQuadraturePoint> quadrature;
    };

    using NodeMomentPair = ArrayPair<ArrayEigenVector<10>>;
    using NodeReferenceLengthPair = ArrayPair<ArrayEigenVector<3>>;
    using EdgeMetricPair = ArrayPair<ArrayEigenVector<4>>;

    /**
     * @brief Builds and stores all geometry derived from the median-dual process.
     *
     * Edge, face, and cell construction points are arithmetic averages of O1
     * vertices.  They are intentionally not geometric centroids.
     */
    class DualGeometry
    {
        const MPIInfo &_mpi;
        ssp<Geom::UnstructuredMesh> _mesh;
        const Topology &_topology;
        AlgorithmSettings _settings;
        bool _buildGhostEdgeSurfaces = false;

        std::vector<Vector3> _cellPoints;
        std::vector<Vector3> _facePoints;
        std::vector<Vector3> _edgePoints;
        std::vector<NodeControlVolume> _nodeVolumes;
        std::vector<EdgeControlSurface> _edgeSurfaces;
        // Efficient limiting of an owned node also needs the exact macro-surface
        // mean on incident edges owned by another rank.  Keep those private
        // halo surfaces separate so EdgeSurfaces() retains its owned-only
        // iteration and accounting semantics.
        std::vector<EdgeControlSurface> _ghostEdgeSurfaces;
        NodeMomentPair _nodeMoments;
        NodeReferenceLengthPair _nodeReferenceLengths;
        EdgeMetricPair _edgeMetrics;
        real _maximumClosureError = 0;

        AffinePoint MakeNodePoint(index iNode) const;
        AffinePoint MakeAveragePoint(const std::vector<index> &nodes) const;
        AffinePoint MakeCellPoint(index iCell) const;
        AffinePoint MakeFacePoint(index iFace) const;
        AffinePoint MakeEdgePoint(index node0, index node1) const;

        void BuildConstructionPoints();
        void BuildOwnedEdgeSurfaces();
        void BuildOwnedNodeVolumes();
        void BuildCommunicationFields();
        void CheckClosure();

        void AddVolumeSimplex(
            NodeControlVolume &controlVolume,
            index iCell,
            const std::vector<AffinePoint> &points) const;
        void AddEdgeSurfaceSimplex(
            EdgeControlSurface &surface,
            index iCell,
            index iFace,
            const std::vector<AffinePoint> &points) const;
        void AddBoundarySimplex(
            NodeControlVolume &controlVolume,
            index iCell,
            index iFace,
            const std::vector<AffinePoint> &points) const;

    public:
        DualGeometry(
            const MPIInfo &mpi,
            const ssp<Geom::UnstructuredMesh> &mesh,
            const Topology &topology,
            const AlgorithmSettings &settings,
            bool buildGhostEdgeSurfaces = false)
            : _mpi(mpi), _mesh(mesh), _topology(topology), _settings(settings),
              _buildGhostEdgeSurfaces(buildGhostEdgeSurfaces)
        {
        }

        void Build();

        [[nodiscard]] const std::vector<Vector3> &CellConstructionPoints() const { return _cellPoints; }
        [[nodiscard]] const std::vector<Vector3> &FaceConstructionPoints() const { return _facePoints; }
        [[nodiscard]] const std::vector<Vector3> &EdgeConstructionPoints() const { return _edgePoints; }
        [[nodiscard]] const NodeControlVolume &NodeVolume(index iNode) const
        {
            return _nodeVolumes.at(static_cast<std::size_t>(iNode));
        }
        [[nodiscard]] const EdgeControlSurface &EdgeSurface(index iEdge) const
        {
            if (iEdge < _topology.NumEdge())
                return _edgeSurfaces.at(static_cast<std::size_t>(iEdge));
            return _ghostEdgeSurfaces.at(
                static_cast<std::size_t>(iEdge - _topology.NumEdge()));
        }
        [[nodiscard]] const std::vector<NodeControlVolume> &NodeVolumes() const { return _nodeVolumes; }
        [[nodiscard]] const std::vector<EdgeControlSurface> &EdgeSurfaces() const { return _edgeSurfaces; }
        [[nodiscard]] NodeMomentPair &NodeMoments() { return _nodeMoments; }
        [[nodiscard]] const NodeMomentPair &NodeMoments() const { return _nodeMoments; }
        [[nodiscard]] Vector3 ReferenceLengths(index iNode) const
        {
            return Vector3{_nodeReferenceLengths(iNode, 0),
                           _nodeReferenceLengths(iNode, 1),
                           _nodeReferenceLengths(iNode, 2)};
        }
        [[nodiscard]] EdgeMetricPair &EdgeMetrics() { return _edgeMetrics; }
        [[nodiscard]] const EdgeMetricPair &EdgeMetrics() const { return _edgeMetrics; }
        [[nodiscard]] real MaximumClosureError() const { return _maximumClosureError; }
        [[nodiscard]] IntegrationMode Mode() const { return _settings.mode; }

        /** @brief Exact raw moments of a line/triangle/tetrahedron simplex. */
        static RawMoments ExactSimplexMoments(const std::vector<Vector3> &points, real measure);

        /** @brief Geometric measure and raw signed Jacobian of a simplex. */
        static std::pair<real, real> SimplexMeasure(const std::vector<Vector3> &points, int dimension);

        /** @brief Thesis (3-34): half the dual-volume extent in each active direction. */
        static Vector3 ReferenceLengthsFromBounds(
            const Vector3 &lower, const Vector3 &upper, int dimension);

        /**
         * @brief Expand the all-differential simplex rule into primal-node weights.
         * @param points Intrinsic simplex vertices (2 for line, 3 for triangle, 4 for tetrahedron).
         * @param anchorNode Primal node about which the Taylor expansion is taken.
         * @param anchorCoordinate Coordinate of anchorNode.
         * @param measure Positive line/area/volume measure.
         */
        static std::vector<SparseVectorWeight> DifferentialWeights(
            const std::vector<AffinePoint> &points,
            index anchorNode,
            const Vector3 &anchorCoordinate,
            real measure);
    };
}
