/**
 * @file NCFVDualGeometry.hpp
 * @brief Median-dual control volumes generated from primal vertex averages.
 */
#pragma once

#include "NCFVConfig.hpp"
#include "NCFVTopology.hpp"

#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"

#include <array>
#include <functional>
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

    /**
     * @brief One node addressed by point-value recovery of a dual control volume.
     *
     * node is a process-local DNDS node index.  It may address either the
     * owned part or the ghost part of a synchronized node field.
     */
    struct EfficientPointRecoveryNode
    {
        index node = UnInitIndex;
        Vector3 gradientWeight = Vector3::Zero();
    };

    /**
     * @brief All efficient-integration weights associated with one macro-surface node.
     *
     * The outer EdgeControlSurface::efficientStencil is the unique, sorted
     * support-node set.  side 0/1 denotes the left/right primal-edge endpoint.
     * Every derivative weight below is an integral weight; division by surface
     * measure is performed only when a mean is requested.  The zero-order
     * entries are represented without duplication by EdgeControlSurface::nodes
     * together with measure/vectorMeasure because they are nonzero only at the
     * two anchor nodes and have the same geometric weight on both sides.
     */
    struct EfficientSurfaceNode
    {
        index node = UnInitIndex;
        std::array<Vector3, 2> stateGradientWeights{
            Vector3::Zero(), Vector3::Zero()};
        std::array<Matrix3, 2> fluxGradientWeights{
            Matrix3::Zero(), Matrix3::Zero()};
        real gradientWeight = 0;
    };

    /** @brief Compact support-node weights for one boundary micro-surface. */
    struct EfficientBoundaryNode
    {
        index node = UnInitIndex;
        // The affine value and affine-gradient surface rules have the same
        // nodal coefficient on a boundary simplex.
        real integrationWeight = 0;
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
        /** Physical quadrature-point coordinate, constructed and stored at initialization. */
        Vector3 coordinate = Vector3::Zero();
        /** Oriented physical surface weight n_g w_g, stored for flux integration. */
        Vector3 vectorWeight = Vector3::Zero();
        /** Positive physical surface weight w_g, stored for scalar/viscous integration. */
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

    /** Method-specific work performed while constructing integration data. */
    struct IntegrationInitializationTiming
    {
        double volumeSeconds = 0;
        double internalSurfaceSeconds = 0;
        double boundarySurfaceSeconds = 0;
        double volumeNormalizationSeconds = 0;
        index volumeCalls = 0;
        index internalSurfaceCalls = 0;
        index boundarySurfaceCalls = 0;
        index volumeNormalizationCalls = 0;

        [[nodiscard]] double TotalSeconds() const
        {
            return volumeSeconds + internalSurfaceSeconds +
                   boundarySurfaceSeconds + volumeNormalizationSeconds;
        }

        [[nodiscard]] index TimedBlockCalls() const
        {
            return volumeCalls + internalSurfaceCalls +
                   boundarySurfaceCalls + volumeNormalizationCalls;
        }
    };

    struct BoundaryPiece
    {
        Geom::t_index zone = Geom::INTERNAL_ZONE;
        int nPoints = 0;
        std::array<AffinePoint, 3> points{};
        Vector3 vectorMeasure = Vector3::Zero();
        real measure = 0;
        std::vector<EfficientBoundaryNode> efficientStencil;
        std::vector<SurfaceQuadraturePoint> quadrature;
    };

    struct NodeControlVolume
    {
        index node = UnInitIndex;
        /**
         * Moments about the primal node x_j:
         *   { |V_j|, integral(x-x_j)dV, integral((x-x_j)(x-x_j)^T)dV }.
         * Keeping these moments local avoids cancellation from forming an
         * absolute second moment and translating it afterwards.
         */
        RawMoments moments;
        // Extrema relative to the primal node, accumulated before optional
        // micro-geometry is discarded. The node belongs to every micro-volume.
        Vector3 lowerOffset = Vector3::Zero();
        Vector3 upperOffset = Vector3::Zero();
        Vector3 referenceLengths = Vector3::Ones();
        // Volume scale is retained for CFL/wall models, not polynomial bases.
        real lengthScale = 0;
        std::vector<EfficientPointRecoveryNode> pointRecoveryStencil;
        std::vector<MicroVolume> microVolumes;
        std::vector<VolumeQuadraturePoint> volumeQuadrature;
        std::vector<BoundaryPiece> boundaryPieces;
    };

    struct EdgeControlSurface
    {
        index edge = UnInitIndex;
        std::array<index, 2> nodes{UnInitIndex, UnInitIndex};
        /** Endpoint coordinates in this macro-surface's minimum-image frame. */
        std::array<Vector3, 2> nodeCoordinates{
            Vector3::Zero(), Vector3::Zero()};
        Vector3 edgePoint = Vector3::Zero();
        Vector3 vectorMeasure = Vector3::Zero();
        real measure = 0;
        std::vector<EfficientSurfaceNode> efficientStencil;
        std::vector<MicroSurface> microSurfaces;
        std::vector<SurfaceQuadraturePoint> quadrature;
    };

    /** Packed node-local moments: volume, c1_j, and the six symmetric C2_j entries. */
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
        Vector3 _periodicLengths = Vector3::Zero();
        bool _nodeIndicesRemapped = false;

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
        mutable IntegrationInitializationTiming _integrationInitializationTiming;

        AffinePoint MakeNodePoint(index iNode) const;
        AffinePoint MakeAveragePoint(
            const std::vector<index> &nodes,
            const Vector3 *frameAnchor = nullptr) const;
        AffinePoint MakeCellPoint(
            index iCell,
            const Vector3 *frameAnchor = nullptr) const;
        AffinePoint MakeFacePoint(
            index iFace,
            const Vector3 *frameAnchor = nullptr) const;
        AffinePoint MakeEdgePoint(
            index node0,
            index node1,
            const Vector3 *frameAnchor = nullptr) const;
        [[nodiscard]] Vector3 CoordinateInFrame(
            index iNode,
            const Vector3 &frameAnchor) const;

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
            bool buildGhostEdgeSurfaces = false,
            const MeshSettings *meshSettings = nullptr)
            : _mpi(mpi), _mesh(mesh), _topology(topology), _settings(settings),
              _buildGhostEdgeSurfaces(buildGhostEdgeSurfaces)
        {
            if (meshSettings)
            {
                DNDS_check_throw_info(
                    meshSettings->periodicLengths.size() == 3,
                    "NCFV dual geometry requires three periodic lengths");
                for (int d = 0; d < 3; d++)
                    _periodicLengths(d) =
                        meshSettings->periodicLengths[static_cast<std::size_t>(d)];
            }
        }

        void Build();

        /** Global node IDs referenced by all retained integration data. */
        [[nodiscard]] std::vector<index> CollectNodeDependencies() const;

        /** Convert mesh-local integration support IDs to the exact node halo once. */
        void RemapNodeIndices(const std::function<index(index)> &meshLocalToHalo);

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
        [[nodiscard]] const IntegrationInitializationTiming &IntegrationInitializationProfile() const
        {
            return _integrationInitializationTiming;
        }

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
