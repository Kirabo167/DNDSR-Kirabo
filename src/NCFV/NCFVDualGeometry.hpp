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

        /**
         * @brief Return sum of the four measured integration-initialization phase times in seconds.
         * @return Sum of the four measured integration-initialization phase times in seconds.
         */
        [[nodiscard]] double TotalSeconds() const
        {
            return volumeSeconds + internalSurfaceSeconds +
                   boundarySurfaceSeconds + volumeNormalizationSeconds;
        }

        /**
         * @brief Return total number of timed integration-initialization blocks.
         * @return Total number of timed integration-initialization blocks.
         */
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

        /**
         * @brief Represent one primal vertex as a coordinate with unit affine support.
         * @param iNode Mesh-local primal-node index.
         * @return Affine point whose only support node has coefficient one.
         */
        AffinePoint MakeNodePoint(index iNode) const;
        /**
         * @brief Average primal vertices and retain their equal affine interpolation weights.
         * @param nodes Nonempty list of mesh-local primal vertices to average.
         * @param frameAnchor Optional coordinate-frame anchor for minimum-image averaging; nullptr selects the
         *     first support vertex as anchor.
         * @return Affine construction point in a consistent periodic frame.
         */
        AffinePoint MakeAveragePoint(
            const std::vector<index> &nodes,
            const Vector3 *frameAnchor = nullptr) const;
        /**
         * @brief Build the arithmetic vertex average of one primal cell.
         * @param iCell Process-local primal-cell index.
         * @param frameAnchor Optional coordinate-frame anchor for minimum-image averaging; nullptr selects the
         *     first support vertex as anchor.
         * @return Affine construction point with primal-vertex support.
         */
        AffinePoint MakeCellPoint(
            index iCell,
            const Vector3 *frameAnchor = nullptr) const;
        /**
         * @brief Build the arithmetic vertex average of one primal face.
         * @param iFace Process-local primal-face index.
         * @param frameAnchor Optional coordinate-frame anchor for minimum-image averaging; nullptr selects the
         *     first support vertex as anchor.
         * @return Affine construction point with primal-vertex support.
         */
        AffinePoint MakeFacePoint(
            index iFace,
            const Vector3 *frameAnchor = nullptr) const;
        /**
         * @brief Build the arithmetic average of two primal-edge endpoints.
         * @param node0 First mesh-local endpoint index.
         * @param node1 Second mesh-local endpoint index.
         * @param frameAnchor Optional coordinate-frame anchor for minimum-image averaging; nullptr selects the
         *     first support vertex as anchor.
         * @return Affine midpoint in the selected periodic frame.
         */
        AffinePoint MakeEdgePoint(
            index node0,
            index node1,
            const Vector3 *frameAnchor = nullptr) const;
        /**
         * @brief Map a primal vertex into the nearest periodic image of an anchor.
         * @param iNode Mesh-local primal-node index.
         * @param frameAnchor Physical coordinate defining the desired periodic frame.
         * @return Three-dimensional coordinate in the anchor frame.
         */
        [[nodiscard]] Vector3 CoordinateInFrame(
            index iNode,
            const Vector3 &frameAnchor) const;

        /**
         * @brief Compute arithmetic primal-cell, face, and edge construction points.
         */
        void BuildConstructionPoints();
        /**
         * @brief Assemble dual macro-surfaces for owned edges and any requested ghost edges.
         */
        void BuildOwnedEdgeSurfaces();
        /**
         * @brief Assemble median-dual simplexes, local moments, scales, and boundary pieces for owned nodes.
         */
        void BuildOwnedNodeVolumes();
        /**
         * @brief Allocate and synchronize packed node moments, reference lengths, and edge metrics.
         */
        void BuildCommunicationFields();
        /**
         * @brief Reduce the dual-volume surface-vector imbalance and enforce the configured closure tolerance.
         */
        void CheckClosure();

        /**
         * @brief Accumulate one median-dual volume simplex and its selected integration data.
         * @param [in,out] controlVolume Owned-node dual volume receiving moments, recovery weights, or volume
         *     quadrature.
         * @param iCell Process-local parent primal-cell index retained in micro-geometry metadata.
         * @param points Affine simplex vertices: three for a 2-D triangle or four for a 3-D tetrahedron.
         */
        void AddVolumeSimplex(
            NodeControlVolume &controlVolume,
            index iCell,
            const std::vector<AffinePoint> &points) const;
        /**
         * @brief Accumulate one internal dual surface simplex with consistent outward orientation.
         * @param [in,out] surface Geometry receiving oriented measures and the selected integration
         *     weights/quadrature.
         * @param iCell Process-local parent primal-cell index.
         * @param iFace Process-local parent primal-face index; supplies boundary metadata when applicable.
         * @param points Affine surface vertices: two for a 2-D segment or three for a 3-D triangle.
         */
        void AddEdgeSurfaceSimplex(
            EdgeControlSurface &surface,
            index iCell,
            index iFace,
            const std::vector<AffinePoint> &points) const;
        /**
         * @brief Accumulate one external boundary simplex with consistent outward orientation.
         * @param [in,out] controlVolume Geometry receiving oriented measures and the selected integration
         *     weights/quadrature.
         * @param iCell Process-local parent primal-cell index.
         * @param iFace Process-local parent primal-face index; supplies boundary metadata when applicable.
         * @param points Affine surface vertices: two for a 2-D segment or three for a 3-D triangle.
         */
        void AddBoundarySimplex(
            NodeControlVolume &controlVolume,
            index iCell,
            index iFace,
            const std::vector<AffinePoint> &points) const;

    public:
        /**
         * @brief Bind the median-dual builder to primal topology and integration settings.
         * @param mpi MPI communicator metadata; collective operations must be entered by every participating rank.
         * @param mesh Distributed O1 primal mesh with owned/ghost connectivity prepared for NCFV.
         * @param topology Built NCFV edge/face topology; must outlive this object.
         * @param settings Integration mode, quadrature, micro-geometry retention, and closure checks.
         * @param buildGhostEdgeSurfaces Retain additional ghost-edge macro-surfaces needed by owned-node limiting
         *     when true.
         * @param meshSettings Optional three periodic lengths for minimum-image geometry; nullptr disables
         *     periodic wrapping.
         */
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

        /**
         * @brief Build median-dual volumes, macro-surfaces, moments, integration data, and communication fields.
         * @details Constructs the data for the selected integration mode and performs the enabled closure audit.
         * @note Collective; call after Topology::Build() and before runtime node-index remapping.
         */
        void Build();

        /**
         * @brief Global node IDs referenced by all retained integration data.
         * @return Sorted unique global node IDs for point recovery, owned/ghost macro-surfaces, and boundary
         *     integration.
         * @pre Node indices still use the original mesh-local layout.
         */
        [[nodiscard]] std::vector<index> CollectNodeDependencies() const;

        /**
         * @brief Convert mesh-local integration support IDs to the exact node halo once.
         * @param meshLocalToHalo Callback mapping a mesh-local support-node ID to its active runtime NCFV halo
         *     index.
         * @pre Build() has completed and remapping has not already been applied.
         */
        void RemapNodeIndices(const std::function<index(index)> &meshLocalToHalo);

        /**
         * @brief Return read-only arithmetic vertex averages of local primal cells.
         * @return Read-only arithmetic vertex averages of local primal cells.
         */
        [[nodiscard]] const std::vector<Vector3> &CellConstructionPoints() const { return _cellPoints; }
        /**
         * @brief Return read-only arithmetic vertex averages of local primal faces.
         * @return Read-only arithmetic vertex averages of local primal faces.
         */
        [[nodiscard]] const std::vector<Vector3> &FaceConstructionPoints() const { return _facePoints; }
        /**
         * @brief Return read-only arithmetic endpoint averages of local primal edges.
         * @return Read-only arithmetic endpoint averages of local primal edges.
         */
        [[nodiscard]] const std::vector<Vector3> &EdgeConstructionPoints() const { return _edgePoints; }
        /**
         * @brief Access one owned node's median-dual control volume.
         * @param iNode Owned primal-node index.
         * @return Read-only volume moments, scales, integration stencils, and boundary pieces.
         */
        [[nodiscard]] const NodeControlVolume &NodeVolume(index iNode) const
        {
            return _nodeVolumes.at(static_cast<std::size_t>(iNode));
        }
        /**
         * @brief Access an owned or explicitly retained ghost macro-surface.
         * @param iEdge Process-local primal-edge index; ghost access requires buildGhostEdgeSurfaces.
         * @return Read-only oriented dual control surface associated with the edge.
         */
        [[nodiscard]] const EdgeControlSurface &EdgeSurface(index iEdge) const
        {
            if (iEdge < _topology.NumEdge())
                return _edgeSurfaces.at(static_cast<std::size_t>(iEdge));
            return _ghostEdgeSurfaces.at(
                static_cast<std::size_t>(iEdge - _topology.NumEdge()));
        }
        /**
         * @brief Return read-only median-dual control-volume geometry for owned nodes.
         * @return Read-only median-dual control-volume geometry for owned nodes.
         */
        [[nodiscard]] const std::vector<NodeControlVolume> &NodeVolumes() const { return _nodeVolumes; }
        /**
         * @brief Return read-only macro-surfaces for owned edges; private ghost surfaces are excluded.
         * @return Read-only macro-surfaces for owned edges; private ghost surfaces are excluded.
         */
        [[nodiscard]] const std::vector<EdgeControlSurface> &EdgeSurfaces() const { return _edgeSurfaces; }
        /**
         * @brief Return owned/ghost packed node-centered moments: volume, first moment, and six symmetric
         *     second-moment entries.
         * @return Owned/ghost packed node-centered moments: volume, first moment, and six symmetric second-moment
         *     entries.
         */
        [[nodiscard]] NodeMomentPair &NodeMoments() { return _nodeMoments; }
        /**
         * @brief Return owned/ghost packed node-centered moments: volume, first moment, and six symmetric
         *     second-moment entries.
         * @return Owned/ghost packed node-centered moments: volume, first moment, and six symmetric second-moment
         *     entries.
         */
        [[nodiscard]] const NodeMomentPair &NodeMoments() const { return _nodeMoments; }
        /**
         * @brief Read the polynomial reference half-spans of a communicated node volume.
         * @param iNode Node index in the mesh-local geometry communication layout.
         * @return Three reference lengths; active components are positive.
         */
        [[nodiscard]] Vector3 ReferenceLengths(index iNode) const
        {
            return Vector3{_nodeReferenceLengths(iNode, 0),
                           _nodeReferenceLengths(iNode, 1),
                           _nodeReferenceLengths(iNode, 2)};
        }
        /**
         * @brief Return owned/ghost packed macro-surface metrics: oriented vector measure and positive scalar
         *     measure.
         * @return Owned/ghost packed macro-surface metrics: oriented vector measure and positive scalar measure.
         */
        [[nodiscard]] EdgeMetricPair &EdgeMetrics() { return _edgeMetrics; }
        /**
         * @brief Return owned/ghost packed macro-surface metrics: oriented vector measure and positive scalar
         *     measure.
         * @return Owned/ghost packed macro-surface metrics: oriented vector measure and positive scalar measure.
         */
        [[nodiscard]] const EdgeMetricPair &EdgeMetrics() const { return _edgeMetrics; }
        /**
         * @brief Return global maximum closure error measured by CheckClosure().
         * @return Global maximum closure error measured by CheckClosure().
         */
        [[nodiscard]] real MaximumClosureError() const { return _maximumClosureError; }
        /**
         * @brief Return configured traditional-quadrature or efficient-differential integration mode.
         * @return Configured traditional-quadrature or efficient-differential integration mode.
         */
        [[nodiscard]] IntegrationMode Mode() const { return _settings.mode; }
        /**
         * @brief Return read-only local integration-initialization timing and call counters.
         * @return Read-only local integration-initialization timing and call counters.
         */
        [[nodiscard]] const IntegrationInitializationTiming &IntegrationInitializationProfile() const
        {
            return _integrationInitializationTiming;
        }

        /**
         *  @brief Exact raw moments of a line/triangle/tetrahedron simplex.
         * @param points Two, three, or four simplex vertex coordinates in the desired moment frame.
         * @param measure Positive length, area, or volume supplied independently of vertex ordering.
         * @return Exact zeroth, first, and second raw integral moments about the coordinate origin.
         */
        static RawMoments ExactSimplexMoments(const std::vector<Vector3> &points, real measure);

        /**
         *  @brief Geometric measure and raw signed Jacobian of a simplex.
         * @param points Simplex vertices: dimension+1 points, with dimension 1, 2, or 3.
         * @param dimension Intrinsic simplex dimension: 1 for a line, 2 for a triangle, 3 for a tetrahedron.
         * @return Pair (positive geometric measure, raw signed Jacobian); orientation is meaningful for
         *     full-dimensional signed determinants.
         */
        static std::pair<real, real> SimplexMeasure(const std::vector<Vector3> &points, int dimension);

        /**
         *  @brief Thesis (3-34): half the dual-volume extent in each active direction.
         * @param lower Component-wise lower coordinate/offset bounds of the dual volume.
         * @param upper Corresponding component-wise upper bounds.
         * @param dimension Active spatial dimension, either 2 or 3; the third coordinate is ignored in 2-D.
         * @return Half-span in each active direction; inactive directions retain unit reference length.
         */
        static Vector3 ReferenceLengthsFromBounds(
            const Vector3 &lower, const Vector3 &upper, int dimension);

        /**
         * @brief Expand the all-differential simplex rule into primal-node weights.
         * @param points Intrinsic simplex vertices (2 for line, 3 for triangle, 4 for tetrahedron).
         * @param anchorNode Primal node about which the Taylor expansion is taken.
         * @param anchorCoordinate Coordinate of anchorNode.
         * @param measure Positive line/area/volume measure.
         * @return Sparse primal-node gradient weights whose contraction implements the anchor-centered
         *     differential simplex rule.
         */
        static std::vector<SparseVectorWeight> DifferentialWeights(
            const std::vector<AffinePoint> &points,
            index anchorNode,
            const Vector3 &anchorCoordinate,
            real measure);
    };
}
