/**
 * @file NCFVNodeHalo.hpp
 * @brief Dependency-driven node owner/ghost layout for the NCFV solver.
 */
#pragma once

#include "NCFVDualGeometry.hpp"

#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"

#include <vector>

namespace DNDS::NCFV
{
    /** Coordinate, centred moments, and reconstruction scales for one node. */
    using NodeHaloGeometryPair = ArrayPair<ArrayEigenVector<16>>;

    /**
     * @brief Builds the exact runtime node halo independently of the mesh halo.
     *
     * The mesh keeps the cell layers needed for dual-geometry assembly.
     * This object first exposes complete owner-side node adjacency rows, expands
     * them through sparse owner/ghost pulls during initialization, and finally
     * retains nodes referenced by reconstruction and integration.
     * Runtime state arrays borrow this final mapping.
     */
    class NodeHalo
    {
        const MPIInfo &_mpi;
        ssp<Geom::UnstructuredMesh> _mesh;
        Vector3 _periodicLengths = Vector3::Zero();

        Geom::tAdjPair _nodeAdjacency;
        NodeHaloGeometryPair _preliminaryGeometry;
        NodeHaloGeometryPair _geometry;
        std::vector<std::vector<std::vector<index>>> _ringsGlobal;
        bool _preliminaryReady = false;
        bool _finalized = false;
        index _preliminaryGhostCount = 0;

        /**
         * @brief Build complete owned-node graph-neighbor rows in global node IDs from primal cell stars.
         */
        void BuildOwnedAdjacency();
        /**
         * @brief Pull remote adjacency rows to expand owned-node graph rings for stencil selection.
         * @param maximumRings Positive maximum graph distance to retain from each owned node.
         */
        void ExpandRings(int maximumRings);
        /**
         * @brief Allocate an owner/ghost geometry pair for a specified global dependency set.
         * @param [out] pair Geometry field receiving coordinates, centered moments, reference lengths, and its
         *     pull mapping.
         * @param geometry Built owned median-dual geometry used to populate owner rows.
         * @param requiredGlobals Global node IDs whose owner data must be available locally; duplicates are
         *     removed.
         * @note Collective; pulls requested geometry from its owning ranks.
         */
        void BuildGeometryPair(
            NodeHaloGeometryPair &pair,
            const DualGeometry &geometry,
            std::vector<index> requiredGlobals) const;

        /**
         * @brief Return preliminary geometry layout before Finalize(), or the exact runtime layout afterward;
         *     requires initialization.
         * @return Preliminary geometry layout before Finalize(), or the exact runtime layout afterward; requires
         *     initialization.
         */
        [[nodiscard]] const NodeHaloGeometryPair &ActiveGeometry() const;
        /**
         * @brief Return preliminary geometry layout before Finalize(), or the exact runtime layout afterward;
         *     requires initialization.
         * @return Preliminary geometry layout before Finalize(), or the exact runtime layout afterward; requires
         *     initialization.
         */
        [[nodiscard]] NodeHaloGeometryPair &ActiveGeometry();

    public:
        /**
         * @brief Bind the dependency-driven node communication layout and its periodic lengths.
         * @param mpi MPI communicator metadata; collective operations must be entered by every participating rank.
         * @param mesh Distributed O1 primal mesh with owned/ghost connectivity prepared for NCFV.
         * @param settings Mesh settings containing exactly three periodic lengths; zero disables wrapping along an
         *     axis.
         * @note The referenced MPIInfo must outlive this object. BuildPreliminary() creates the first usable
         *     layout.
         */
        NodeHalo(
            const MPIInfo &mpi,
            const ssp<Geom::UnstructuredMesh> &mesh,
            const MeshSettings &settings);

        /**
         * @brief Build maximum-ring topology and temporary geometry used to select stencils.
         * @param geometry Built median-dual geometry in mesh-local numbering.
         * @param maximumRings Positive maximum number of graph-neighbor rings required by reconstruction.
         * @param integrationDependencies Global support-node IDs needed by retained integration data in addition
         *     to graph rings.
         * @note Collective; call before Reconstruction::Build().
         */
        void BuildPreliminary(
            const DualGeometry &geometry,
            int maximumRings,
            const std::vector<index> &integrationDependencies);

        /**
         * @brief Replace the temporary layout by the runtime dependency set.
         * @param geometry Built owner-side dual geometry from which packed runtime geometry is populated.
         * @param runtimeDependencies Global node IDs required by final stencils and integration; passed by value
         *     for sorting/deduplication.
         * @note Collective. Runtime fields must borrow the finalized layout, and retained support IDs must then be
         *     remapped.
         */
        void Finalize(
            const DualGeometry &geometry,
            std::vector<index> runtimeDependencies);

        /**
         * @brief Convert a node index between mesh/global numbering and the active NCFV halo.
         * @param globalNode Stable distributed primal-node ID.
         * @return Owned or ghost node index in the active halo; throws if the global ID is absent.
         */
        [[nodiscard]] index GlobalToLocal(index globalNode) const;
        /**
         * @brief Convert a node index between mesh/global numbering and the active NCFV halo.
         * @param meshLocalNode Owned or ghost index in the primal mesh node layout.
         * @return Matching active-halo node index, resolved through the stable global node ID.
         */
        [[nodiscard]] index MeshLocalToLocal(index meshLocalNode) const;
        /**
         * @brief Convert a node index between mesh/global numbering and the active NCFV halo.
         * @param localNode Node index in the active NCFV halo layout, covering owned and retained ghost nodes.
         * @return Stable global node ID from the active halo mapping.
         */
        [[nodiscard]] index LocalToGlobal(index localNode) const;
        /**
         * @brief Read coordinate data from the active owner/ghost geometry field.
         * @param localNode Node index in the active NCFV halo layout, covering owned and retained ghost nodes.
         * @return Three-dimensional stored coordinate of the requested node.
         */
        [[nodiscard]] Vector3 Coordinate(index localNode) const;
        /**
         * @brief Compute the shortest periodic displacement between two active-halo nodes.
         * @param fromLocal Anchor node index in the active halo.
         * @param toLocal Target node index in the same layout.
         * @return Coordinate(toLocal) - Coordinate(fromLocal), wrapped along active periodic axes.
         */
        [[nodiscard]] Vector3 Displacement(index fromLocal, index toLocal) const;
        /**
         * @brief Compute a minimum-image displacement using global node IDs.
         * @param fromGlobal Global anchor-node ID present in the active halo.
         * @param toGlobal Global target-node ID present in the active halo.
         * @return Wrapped displacement from the anchor to the target.
         */
        [[nodiscard]] Vector3 DisplacementGlobal(index fromGlobal, index toGlobal) const;
        /**
         * @brief Map a physical point to the nearest periodic image relative to an anchor node.
         * @param fromLocal Anchor-node index in the active halo.
         * @param point Physical target coordinate.
         * @return Wrapped point-minus-anchor displacement.
         */
        [[nodiscard]] Vector3 DisplacementToPoint(
            index fromLocal,
            const Vector3 &point) const;
        /**
         * @brief Translate communicated node-centered moments to a chosen anchor-node frame.
         * @param anchorLocal Anchor-node index in the active halo.
         * @param nodeLocal Index of the node whose dual volume supplies the moments.
         * @return Volume and first/second raw moments about the anchor, with minimum-image displacement accounted
         *     for.
         */
        [[nodiscard]] RawMoments MomentsRelative(
            index anchorLocal,
            index nodeLocal) const;
        /**
         * @brief Read volume data from the active owner/ghost geometry field.
         * @param localNode Node index in the active NCFV halo layout, covering owned and retained ghost nodes.
         * @return Positive measure of the node's median-dual control volume.
         */
        [[nodiscard]] real Volume(index localNode) const;
        /**
         * @brief Read referencelengths data from the active owner/ghost geometry field.
         * @param localNode Node index in the active NCFV halo layout, covering owned and retained ghost nodes.
         * @return Three polynomial reference half-spans communicated from the node owner.
         */
        [[nodiscard]] Vector3 ReferenceLengths(index localNode) const;

        /**
         * @brief Access the precomputed graph rings used for one owned node's stencil search.
         * @param ownedNode Owned primal-node index.
         * @return Read-only ring lists of global node IDs, starting with the first neighbor ring.
         */
        [[nodiscard]] const std::vector<std::vector<index>> &RingsGlobal(
            index ownedNode) const
        {
            return _ringsGlobal.at(static_cast<std::size_t>(ownedNode));
        }

        /**
         * @brief Return number of primal nodes owned by this rank.
         * @return Number of primal nodes owned by this rank.
         */
        [[nodiscard]] index NumNode() const { return _mesh->NumNode(); }
        /**
         * @brief Return number of ghost nodes in the active preliminary/runtime geometry layout.
         * @return Number of ghost nodes in the active preliminary/runtime geometry layout.
         */
        [[nodiscard]] index NumNodeGhost() const
        {
            return ActiveGeometry().son->Size();
        }
        /**
         * @brief Return total node count in the active layout, owned plus ghosts.
         * @return Total node count in the active layout, owned plus ghosts.
         */
        [[nodiscard]] index NumNodeProc() const
        {
            return ActiveGeometry().Size();
        }
        /**
         * @brief Return ghost-node count saved from the preliminary stencil-selection layout.
         * @return Ghost-node count saved from the preliminary stencil-selection layout.
         */
        [[nodiscard]] index PreliminaryGhostCount() const
        {
            return _preliminaryGhostCount;
        }
        /**
         * @brief Return true after the exact runtime dependency layout has replaced the preliminary layout.
         * @return True after the exact runtime dependency layout has replaced the preliminary layout.
         */
        [[nodiscard]] bool IsFinalized() const { return _finalized; }

        /**
         * @brief Return active node geometry pair whose owner/ghost communication mapping runtime fields borrow.
         * @return Active node geometry pair whose owner/ghost communication mapping runtime fields borrow.
         */
        [[nodiscard]] NodeHaloGeometryPair &Layout() { return ActiveGeometry(); }
        /**
         * @brief Return active node geometry pair whose owner/ghost communication mapping runtime fields borrow.
         * @return Active node geometry pair whose owner/ghost communication mapping runtime fields borrow.
         */
        [[nodiscard]] const NodeHaloGeometryPair &Layout() const
        {
            return ActiveGeometry();
        }
    };
}
