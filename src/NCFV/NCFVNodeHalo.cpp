#include "NCFVNodeHalo.hpp"

#include "DNDS/Errors.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <unordered_set>

namespace DNDS::NCFV
{
    namespace
    {
        constexpr int CoordinateOffset = 0;
        constexpr int MeasureOffset = 3;
        constexpr int FirstOffset = 4;
        constexpr int SecondOffset = 7;
        constexpr int ReferenceLengthOffset = 13;

        void SortUnique(std::vector<index> &values)
        {
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        }
    }

    NodeHalo::NodeHalo(
        const MPIInfo &mpi,
        const ssp<Geom::UnstructuredMesh> &mesh,
        const MeshSettings &settings)
        : _mpi(mpi), _mesh(mesh)
    {
        DNDS_check_throw_info(settings.periodicLengths.size() == 3,
                              "NCFV node halo requires three periodic lengths");
        for (int d = 0; d < 3; d++)
            _periodicLengths(d) = settings.periodicLengths[static_cast<std::size_t>(d)];
    }

    void NodeHalo::BuildOwnedAdjacency()
    {
        _nodeAdjacency.InitPair("NCFV.nodeAdjacency", _mpi);
        _nodeAdjacency.father->Resize(_mesh->NumNode());

        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            std::set<index> neighbors;
            for (index iCell : _mesh->node2cell[iNode])
            {
                DNDS_check_throw_info(
                    iCell >= 0 && iCell < _mesh->NumCellProc(),
                    "NCFV exact node halo found an unresolved owned-node cell star");
                auto cell = _mesh->GetCellElement(iCell);
                const int nEdges = _mesh->getDim() == 2
                                       ? cell.GetNumFaces()
                                       : cell.GetNumEdges();
                for (int iEdge = 0; iEdge < nEdges; iEdge++)
                {
                    const auto edge = _mesh->getDim() == 2
                                          ? cell.ObtainFace(iEdge)
                                          : cell.ObtainEdge(iEdge);
                    std::vector<index> nodes(
                        static_cast<std::size_t>(edge.GetNumNodes()));
                    if (_mesh->getDim() == 2)
                        cell.ExtractFaceNodes(iEdge, _mesh->cell2node[iCell], nodes);
                    else
                        cell.ExtractEdgeNodes(iEdge, _mesh->cell2node[iCell], nodes);
                    DNDS_check_throw_info(nodes.size() >= 2,
                                          "NCFV node graph encountered an invalid edge");
                    if (nodes[0] == iNode && nodes[1] != iNode)
                        neighbors.insert(_mesh->NodeIndexLocal2Global(nodes[1]));
                    if (nodes[1] == iNode && nodes[0] != iNode)
                        neighbors.insert(_mesh->NodeIndexLocal2Global(nodes[0]));
                }
            }
            DNDS_check_throw_info(
                !neighbors.empty(),
                "NCFV exact node halo produced an isolated owned node; periodic directions need at least two cells");
            _nodeAdjacency.father->ResizeRow(
                iNode, static_cast<rowsize>(neighbors.size()));
            rowsize iNeighbor = 0;
            for (index globalNeighbor : neighbors)
                _nodeAdjacency(iNode, iNeighbor++) = globalNeighbor;
        }
        _nodeAdjacency.father->Compress();
        _nodeAdjacency.TransAttach();
        _nodeAdjacency.trans.createFatherGlobalMapping();
    }

    void NodeHalo::ExpandRings(int maximumRings)
    {
        DNDS_check_throw_info(maximumRings >= 1,
                              "NCFV exact node halo requires at least one reconstruction ring");
        _ringsGlobal.assign(
            static_cast<std::size_t>(_mesh->NumNode()),
            std::vector<std::vector<index>>(
                static_cast<std::size_t>(maximumRings)));

        std::vector<std::unordered_set<index>> visited(
            static_cast<std::size_t>(_mesh->NumNode()));
        std::vector<std::vector<index>> frontier(
            static_cast<std::size_t>(_mesh->NumNode()));
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const index global = _mesh->NodeIndexLocal2Global(iNode);
            visited[static_cast<std::size_t>(iNode)].insert(global);
            frontier[static_cast<std::size_t>(iNode)].push_back(global);
        }

        std::vector<index> requestedGhostRows;
        for (int ring = 0; ring < maximumRings; ring++)
        {
            for (const auto &nodeFrontier : frontier)
                for (index global : nodeFrontier)
                {
                    MPI_int ownerRank = UnInitMPIInt;
                    index ownerLocal = UnInitIndex;
                    const bool found =
                        _nodeAdjacency.father->pLGlobalMapping->search(
                            global, ownerRank, ownerLocal);
                    DNDS_check_throw_info(
                        found,
                        "NCFV exact node halo could not resolve a frontier-node owner");
                    if (ownerRank != _mpi.rank)
                        requestedGhostRows.push_back(global);
                }
            SortUnique(requestedGhostRows);

            if (ring > 0)
                _nodeAdjacency.trans.clearMPITypes();
            _nodeAdjacency.trans.createGhostMapping(requestedGhostRows);
            _nodeAdjacency.trans.createMPITypes();
            _nodeAdjacency.trans.pullOnce();

            index localFrontierCount = 0;
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            {
                std::set<index> next;
                for (index currentGlobal : frontier[static_cast<std::size_t>(iNode)])
                {
                    MPI_int ownerRank = UnInitMPIInt;
                    index currentLocal = UnInitIndex;
                    const bool found =
                        _nodeAdjacency.trans.pLGhostMapping->search_indexAppend(
                            currentGlobal, ownerRank, currentLocal);
                    DNDS_check_throw_info(
                        found && currentLocal >= 0,
                        "NCFV exact node halo is missing a requested adjacency row");
                    for (index neighborGlobal : _nodeAdjacency[currentLocal])
                        if (visited[static_cast<std::size_t>(iNode)]
                                .insert(neighborGlobal)
                                .second)
                            next.insert(neighborGlobal);
                }
                auto &stored = _ringsGlobal[static_cast<std::size_t>(iNode)]
                                           [static_cast<std::size_t>(ring)];
                stored.assign(next.begin(), next.end());
                frontier[static_cast<std::size_t>(iNode)] = stored;
                localFrontierCount += static_cast<index>(stored.size());
            }

            index globalFrontierCount = 0;
            MPI_Allreduce(&localFrontierCount, &globalFrontierCount, 1,
                          DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
            if (globalFrontierCount == 0)
                break;
        }
    }

    void NodeHalo::BuildGeometryPair(
        NodeHaloGeometryPair &pair,
        const DualGeometry &geometry,
        std::vector<index> requiredGlobals) const
    {
        SortUnique(requiredGlobals);
        std::vector<index> ghostGlobals;
        ghostGlobals.reserve(requiredGlobals.size());
        for (index global : requiredGlobals)
        {
            MPI_int ownerRank = UnInitMPIInt;
            index ownerLocal = UnInitIndex;
            const bool found = _mesh->coords.father->pLGlobalMapping->search(
                global, ownerRank, ownerLocal);
            DNDS_check_throw_info(found,
                                  "NCFV exact node halo dependency has no owner");
            if (ownerRank != _mpi.rank)
                ghostGlobals.push_back(global);
        }

        pair.InitPair("NCFV.nodeHaloGeometry", _mpi);
        pair.father->Resize(_mesh->NumNode());
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const Vector3 coordinate = _mesh->coords[iNode];
            const RawMoments &moments = geometry.NodeVolume(iNode).moments;
            const Vector3 centredFirst =
                moments.first - moments.measure * coordinate;
            const Matrix3 centredSecond =
                moments.second - coordinate * moments.first.transpose() -
                moments.first * coordinate.transpose() +
                moments.measure * coordinate * coordinate.transpose();
            const Vector3 referenceLengths =
                geometry.NodeVolume(iNode).referenceLengths;

            pair(iNode, CoordinateOffset + 0) = coordinate.x();
            pair(iNode, CoordinateOffset + 1) = coordinate.y();
            pair(iNode, CoordinateOffset + 2) = coordinate.z();
            pair(iNode, MeasureOffset) = moments.measure;
            pair(iNode, FirstOffset + 0) = centredFirst.x();
            pair(iNode, FirstOffset + 1) = centredFirst.y();
            pair(iNode, FirstOffset + 2) = centredFirst.z();
            pair(iNode, SecondOffset + 0) = centredSecond(0, 0);
            pair(iNode, SecondOffset + 1) = centredSecond(1, 1);
            pair(iNode, SecondOffset + 2) = centredSecond(2, 2);
            pair(iNode, SecondOffset + 3) = centredSecond(0, 1);
            pair(iNode, SecondOffset + 4) = centredSecond(1, 2);
            pair(iNode, SecondOffset + 5) = centredSecond(2, 0);
            for (int d = 0; d < 3; d++)
                pair(iNode, ReferenceLengthOffset + d) = referenceLengths(d);
        }
        pair.TransAttach();
        pair.trans.createFatherGlobalMapping();
        pair.trans.createGhostMapping(ghostGlobals);
        pair.trans.createMPITypes();
        pair.trans.pullOnce();
    }

    void NodeHalo::BuildPreliminary(
        const DualGeometry &geometry,
        int maximumRings,
        const std::vector<index> &integrationDependencies)
    {
        BuildOwnedAdjacency();
        ExpandRings(maximumRings);

        std::vector<index> required = integrationDependencies;
        for (const auto &nodeRings : _ringsGlobal)
            for (const auto &ring : nodeRings)
                required.insert(required.end(), ring.begin(), ring.end());
        BuildGeometryPair(_preliminaryGeometry, geometry, std::move(required));
        _preliminaryGhostCount = _preliminaryGeometry.son->Size();
        _preliminaryReady = true;
    }

    void NodeHalo::Finalize(
        const DualGeometry &geometry,
        std::vector<index> runtimeDependencies)
    {
        DNDS_check_throw_info(_preliminaryReady,
                              "NCFV exact node halo must build its preliminary layout first");
        BuildGeometryPair(_geometry, geometry, std::move(runtimeDependencies));
        _finalized = true;
        _preliminaryGeometry = {};
        _nodeAdjacency = {};
        _ringsGlobal.clear();
        _ringsGlobal.shrink_to_fit();

        index localCounts[2]{_preliminaryGhostCount, _geometry.son->Size()};
        index globalCounts[2]{};
        MPI_Allreduce(localCounts, globalCounts, 2,
                      DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
        if (_mpi.rank == 0)
            log() << "NCFV exact node halo: preliminary ghosts="
                  << globalCounts[0] << ", retained runtime ghosts="
                  << globalCounts[1] << std::endl;
    }

    const NodeHaloGeometryPair &NodeHalo::ActiveGeometry() const
    {
        DNDS_check_throw_info(
            _finalized || _preliminaryReady,
            "NCFV exact node halo has not been initialized");
        return _finalized ? _geometry : _preliminaryGeometry;
    }

    NodeHaloGeometryPair &NodeHalo::ActiveGeometry()
    {
        return const_cast<NodeHaloGeometryPair &>(
            static_cast<const NodeHalo *>(this)->ActiveGeometry());
    }

    index NodeHalo::GlobalToLocal(index globalNode) const
    {
        MPI_int ownerRank = UnInitMPIInt;
        index localNode = UnInitIndex;
        const bool found = ActiveGeometry().trans.pLGhostMapping->search_indexAppend(
            globalNode, ownerRank, localNode);
        DNDS_check_throw_info(
            found && localNode >= 0,
            "NCFV exact node halo does not contain a requested global node");
        return localNode;
    }

    index NodeHalo::MeshLocalToLocal(index meshLocalNode) const
    {
        DNDS_check_throw_info(
            meshLocalNode >= 0 && meshLocalNode < _mesh->NumNodeProc(),
            "NCFV exact node halo received an invalid mesh-local node");
        return GlobalToLocal(_mesh->NodeIndexLocal2Global(meshLocalNode));
    }

    index NodeHalo::LocalToGlobal(index localNode) const
    {
        DNDS_check_throw_info(
            localNode >= 0 && localNode < ActiveGeometry().Size(),
            "NCFV exact node halo received an invalid local node");
        return ActiveGeometry().trans.pLGhostMapping->operator()(-1, localNode);
    }

    Vector3 NodeHalo::Coordinate(index localNode) const
    {
        const auto &data = ActiveGeometry();
        return Vector3{data(localNode, CoordinateOffset + 0),
                       data(localNode, CoordinateOffset + 1),
                       data(localNode, CoordinateOffset + 2)};
    }

    Vector3 NodeHalo::Displacement(index fromLocal, index toLocal) const
    {
        Vector3 displacement = Coordinate(toLocal) - Coordinate(fromLocal);
        for (int d = 0; d < _mesh->getDim(); d++)
            if (_periodicLengths(d) > 0)
                displacement(d) -= _periodicLengths(d) *
                                   std::round(displacement(d) /
                                              _periodicLengths(d));
        return displacement;
    }

    Vector3 NodeHalo::DisplacementGlobal(
        index fromGlobal,
        index toGlobal) const
    {
        return Displacement(GlobalToLocal(fromGlobal), GlobalToLocal(toGlobal));
    }

    Vector3 NodeHalo::DisplacementToPoint(
        index fromLocal,
        const Vector3 &point) const
    {
        Vector3 displacement = point - Coordinate(fromLocal);
        for (int d = 0; d < _mesh->getDim(); d++)
            if (_periodicLengths(d) > 0)
                displacement(d) -= _periodicLengths(d) *
                                   std::round(displacement(d) /
                                              _periodicLengths(d));
        return displacement;
    }

    RawMoments NodeHalo::MomentsRelative(
        index anchorLocal,
        index nodeLocal) const
    {
        const auto &data = ActiveGeometry();
        RawMoments moments;
        moments.measure = data(nodeLocal, MeasureOffset);
        Vector3 centredFirst{
            data(nodeLocal, FirstOffset + 0),
            data(nodeLocal, FirstOffset + 1),
            data(nodeLocal, FirstOffset + 2)};
        Matrix3 centredSecond = Matrix3::Zero();
        centredSecond(0, 0) = data(nodeLocal, SecondOffset + 0);
        centredSecond(1, 1) = data(nodeLocal, SecondOffset + 1);
        centredSecond(2, 2) = data(nodeLocal, SecondOffset + 2);
        centredSecond(0, 1) = centredSecond(1, 0) =
            data(nodeLocal, SecondOffset + 3);
        centredSecond(1, 2) = centredSecond(2, 1) =
            data(nodeLocal, SecondOffset + 4);
        centredSecond(2, 0) = centredSecond(0, 2) =
            data(nodeLocal, SecondOffset + 5);

        const Vector3 displacement = Displacement(anchorLocal, nodeLocal);
        moments.first = centredFirst + moments.measure * displacement;
        moments.second = centredSecond +
                         displacement * centredFirst.transpose() +
                         centredFirst * displacement.transpose() +
                         moments.measure * displacement * displacement.transpose();
        return moments;
    }

    real NodeHalo::Volume(index localNode) const
    {
        return ActiveGeometry()(localNode, MeasureOffset);
    }

    Vector3 NodeHalo::ReferenceLengths(index localNode) const
    {
        const auto &data = ActiveGeometry();
        return Vector3{data(localNode, ReferenceLengthOffset + 0),
                       data(localNode, ReferenceLengthOffset + 1),
                       data(localNode, ReferenceLengthOffset + 2)};
    }
}
