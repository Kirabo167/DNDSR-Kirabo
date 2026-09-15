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
     * The mesh keeps one point-complete cell layer for dual-geometry assembly.
     * This object first exposes complete owner-side node adjacency rows, expands
     * them through sparse owner/ghost pulls during initialization, and finally
     * retains only nodes referenced by accepted reconstruction and integration
     * stencils. Runtime state arrays borrow this final mapping.
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

        void BuildOwnedAdjacency();
        void ExpandRings(int maximumRings);
        void BuildGeometryPair(
            NodeHaloGeometryPair &pair,
            const DualGeometry &geometry,
            std::vector<index> requiredGlobals) const;

        [[nodiscard]] const NodeHaloGeometryPair &ActiveGeometry() const;
        [[nodiscard]] NodeHaloGeometryPair &ActiveGeometry();

    public:
        NodeHalo(
            const MPIInfo &mpi,
            const ssp<Geom::UnstructuredMesh> &mesh,
            const MeshSettings &settings);

        /** Build maximum-ring topology and temporary geometry used to select stencils. */
        void BuildPreliminary(
            const DualGeometry &geometry,
            int maximumRings,
            const std::vector<index> &integrationDependencies);

        /** Replace the temporary layout by the exact accepted runtime dependency set. */
        void Finalize(
            const DualGeometry &geometry,
            std::vector<index> runtimeDependencies);

        [[nodiscard]] index GlobalToLocal(index globalNode) const;
        [[nodiscard]] index MeshLocalToLocal(index meshLocalNode) const;
        [[nodiscard]] index LocalToGlobal(index localNode) const;
        [[nodiscard]] Vector3 Coordinate(index localNode) const;
        [[nodiscard]] Vector3 Displacement(index fromLocal, index toLocal) const;
        [[nodiscard]] Vector3 DisplacementGlobal(index fromGlobal, index toGlobal) const;
        [[nodiscard]] Vector3 DisplacementToPoint(
            index fromLocal,
            const Vector3 &point) const;
        [[nodiscard]] RawMoments MomentsRelative(
            index anchorLocal,
            index nodeLocal) const;
        [[nodiscard]] real Volume(index localNode) const;
        [[nodiscard]] Vector3 ReferenceLengths(index localNode) const;

        [[nodiscard]] const std::vector<std::vector<index>> &RingsGlobal(
            index ownedNode) const
        {
            return _ringsGlobal.at(static_cast<std::size_t>(ownedNode));
        }

        [[nodiscard]] index NumNode() const { return _mesh->NumNode(); }
        [[nodiscard]] index NumNodeGhost() const
        {
            return ActiveGeometry().son->Size();
        }
        [[nodiscard]] index NumNodeProc() const
        {
            return ActiveGeometry().Size();
        }
        [[nodiscard]] index PreliminaryGhostCount() const
        {
            return _preliminaryGhostCount;
        }
        [[nodiscard]] bool IsFinalized() const { return _finalized; }

        [[nodiscard]] NodeHaloGeometryPair &Layout() { return ActiveGeometry(); }
        [[nodiscard]] const NodeHaloGeometryPair &Layout() const
        {
            return ActiveGeometry();
        }
    };
}
