#include "NCFVTopology.hpp"

#include "DNDS/Errors.hpp"
#include "Geom/Mesh/MeshConnectivity.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <set>

namespace DNDS::NCFV
{
    namespace
    {
        struct EdgeKey
        {
            std::array<index, 2> nodes{UnInitIndex, UnInitIndex};
            index relativePbi = 0;

            bool operator<(const EdgeKey &other) const
            {
                if (nodes != other.nodes)
                    return nodes < other.nodes;
                return relativePbi < other.relativePbi;
            }
        };

        struct FaceKey
        {
            int nVertices = 0;
            std::array<index, 4> nodes{
                UnInitIndex, UnInitIndex, UnInitIndex, UnInitIndex};
            std::array<index, 4> normalizedPbi{0, 0, 0, 0};

            bool operator<(const FaceKey &other) const
            {
                if (nVertices != other.nVertices)
                    return nVertices < other.nVertices;
                if (nodes != other.nodes)
                    return nodes < other.nodes;
                return normalizedPbi < other.normalizedPbi;
            }
        };

        struct ExchangeResult
        {
            std::vector<index> values;
            std::vector<int> receiveOffsets;
        };

        EdgeKey CanonicalEdgeKey(
            index node0,
            index node1,
            Geom::NodePeriodicBits pbi0 = {},
            Geom::NodePeriodicBits pbi1 = {})
        {
            EdgeKey key;
            key.nodes = node0 < node1
                            ? std::array<index, 2>{node0, node1}
                            : std::array<index, 2>{node1, node0};
            key.relativePbi = static_cast<index>(pbi0 ^ pbi1);
            return key;
        }

        MPI_int EdgeDirectoryRank(const EdgeKey &key, MPI_int nRanks)
        {
            std::uint64_t value = static_cast<std::uint64_t>(key.nodes[0]);
            value ^= static_cast<std::uint64_t>(key.nodes[1]) +
                     UINT64_C(0x9e3779b97f4a7c15) + (value << 6U) + (value >> 2U);
            value ^= static_cast<std::uint64_t>(key.relativePbi) +
                     UINT64_C(0x9e3779b97f4a7c15) + (value << 6U) + (value >> 2U);
            value ^= value >> 30U;
            value *= UINT64_C(0xbf58476d1ce4e5b9);
            value ^= value >> 27U;
            value *= UINT64_C(0x94d049bb133111eb);
            value ^= value >> 31U;
            return static_cast<MPI_int>(value % static_cast<std::uint64_t>(nRanks));
        }

        FaceKey CanonicalFaceKey(
            const std::vector<index> &nodes,
            const std::vector<Geom::NodePeriodicBits> &pbi = {})
        {
            DNDS_check_throw_info(nodes.size() >= 2 && nodes.size() <= 4,
                                  "NCFV supports O1 line, triangle, and quadrilateral faces");
            DNDS_check_throw_info(
                pbi.empty() || pbi.size() == nodes.size(),
                "NCFV periodic face key has mismatched node/pbi lengths");

            FaceKey best;
            bool hasBest = false;
            // A sub-entity frame may be XOR-shifted uniformly between parent
            // cells.  Minimize over all three-bit shifts to obtain a frame-
            // invariant key, including meshes where node IDs repeat on a very
            // coarse periodic quotient.
            for (index shift = 0; shift < 8; shift++)
            {
                std::vector<std::pair<index, index>> pairs;
                pairs.reserve(nodes.size());
                for (std::size_t i = 0; i < nodes.size(); i++)
                {
                    const index bits = pbi.empty()
                                           ? 0
                                           : static_cast<index>(uint8_t(pbi[i]));
                    pairs.emplace_back(nodes[i], bits ^ shift);
                }
                std::sort(pairs.begin(), pairs.end());
                FaceKey candidate;
                candidate.nVertices = static_cast<int>(pairs.size());
                for (std::size_t i = 0; i < pairs.size(); i++)
                {
                    candidate.nodes[i] = pairs[i].first;
                    candidate.normalizedPbi[i] = pairs[i].second;
                }
                if (!hasBest || candidate < best)
                {
                    best = candidate;
                    hasBest = true;
                }
            }
            return best;
        }

        MPI_int FaceDirectoryRank(const FaceKey &key, MPI_int nRanks)
        {
            std::uint64_t value = static_cast<std::uint64_t>(key.nVertices);
            for (int i = 0; i < key.nVertices; i++)
            {
                value ^= static_cast<std::uint64_t>(key.nodes[static_cast<std::size_t>(i)]) +
                         UINT64_C(0x9e3779b97f4a7c15) + (value << 6U) + (value >> 2U);
                value ^= value >> 30U;
                value *= UINT64_C(0xbf58476d1ce4e5b9);
                value ^= value >> 27U;
                value ^= static_cast<std::uint64_t>(
                             key.normalizedPbi[static_cast<std::size_t>(i)]) +
                         UINT64_C(0x9e3779b97f4a7c15) + (value << 6U) +
                         (value >> 2U);
            }
            return static_cast<MPI_int>(value % static_cast<std::uint64_t>(nRanks));
        }

        void AppendFaceRecord(
            std::vector<index> &message,
            const FaceKey &key,
            index payload)
        {
            message.push_back(static_cast<index>(key.nVertices));
            message.insert(message.end(), key.nodes.begin(), key.nodes.end());
            message.insert(
                message.end(), key.normalizedPbi.begin(),
                key.normalizedPbi.end());
            message.push_back(payload);
        }

        FaceKey ReadFaceRecord(const std::vector<index> &message, std::size_t offset)
        {
            FaceKey key;
            key.nVertices = static_cast<int>(message[offset]);
            DNDS_check_throw_info(key.nVertices >= 2 && key.nVertices <= 4,
                                  "NCFV face-directory record has an invalid vertex count");
            for (int i = 0; i < 4; i++)
                key.nodes[static_cast<std::size_t>(i)] =
                    message[offset + static_cast<std::size_t>(i + 1)];
            for (int i = 0; i < 4; i++)
                key.normalizedPbi[static_cast<std::size_t>(i)] =
                    message[offset + static_cast<std::size_t>(i + 5)];
            return key;
        }

        ExchangeResult ExchangeByRank(
            const std::vector<std::vector<index>> &outgoing,
            const MPIInfo &mpi)
        {
            DNDS_check_throw_info(outgoing.size() == static_cast<std::size_t>(mpi.size),
                                  "NCFV edge-directory send table has the wrong rank count");
            std::vector<int> sendCounts(static_cast<std::size_t>(mpi.size), 0);
            std::vector<int> receiveCounts(static_cast<std::size_t>(mpi.size), 0);
            for (MPI_int rank = 0; rank < mpi.size; rank++)
            {
                DNDS_check_throw_info(
                    outgoing[static_cast<std::size_t>(rank)].size() <=
                        static_cast<std::size_t>(std::numeric_limits<int>::max()),
                    "NCFV edge-directory message exceeds MPI int count capacity");
                sendCounts[static_cast<std::size_t>(rank)] =
                    static_cast<int>(outgoing[static_cast<std::size_t>(rank)].size());
            }
            MPI_Alltoall(sendCounts.data(), 1, MPI_INT,
                         receiveCounts.data(), 1, MPI_INT, mpi.comm);

            std::vector<int> sendOffsets(static_cast<std::size_t>(mpi.size + 1), 0);
            std::vector<int> receiveOffsets(static_cast<std::size_t>(mpi.size + 1), 0);
            for (MPI_int rank = 0; rank < mpi.size; rank++)
            {
                sendOffsets[static_cast<std::size_t>(rank + 1)] =
                    sendOffsets[static_cast<std::size_t>(rank)] +
                    sendCounts[static_cast<std::size_t>(rank)];
                receiveOffsets[static_cast<std::size_t>(rank + 1)] =
                    receiveOffsets[static_cast<std::size_t>(rank)] +
                    receiveCounts[static_cast<std::size_t>(rank)];
            }

            std::vector<index> sendValues(
                static_cast<std::size_t>(sendOffsets.back()));
            for (MPI_int rank = 0; rank < mpi.size; rank++)
                std::copy(outgoing[static_cast<std::size_t>(rank)].begin(),
                          outgoing[static_cast<std::size_t>(rank)].end(),
                          sendValues.begin() + sendOffsets[static_cast<std::size_t>(rank)]);

            ExchangeResult result;
            result.values.resize(static_cast<std::size_t>(receiveOffsets.back()));
            result.receiveOffsets = receiveOffsets;
            MPI_Alltoallv(sendValues.data(), sendCounts.data(), sendOffsets.data(),
                          DNDS_MPI_INDEX, result.values.data(), receiveCounts.data(),
                          receiveOffsets.data(), DNDS_MPI_INDEX, mpi.comm);
            return result;
        }
    }

    Geom::SubEntityQueryPbi Topology::BuildEdgeQuery() const
    {
        Geom::SubEntityQueryPbi query;
        query.numSubEntities = [this](index iCell) -> int
        {
            auto element = _mesh->GetCellElement(iCell);
            return _mesh->getDim() == 2 ? element.GetNumFaces() : element.GetNumEdges();
        };
        query.describe = [this](index iCell, int iSub) -> Geom::SubEntityDesc
        {
            auto element = _mesh->GetCellElement(iCell);
            const auto edge = _mesh->getDim() == 2
                                  ? element.ObtainFace(iSub)
                                  : element.ObtainEdge(iSub);
            return {edge.GetNumVertices(), edge.GetNumNodes(), static_cast<Geom::t_index>(edge.type)};
        };
        query.extractNodes = [this](index iCell, int iSub,
                                    const std::function<index(int)> &parentNodes,
                                    index *out)
        {
            auto element = _mesh->GetCellElement(iCell);
            const auto edge = _mesh->getDim() == 2
                                  ? element.ObtainFace(iSub)
                                  : element.ObtainEdge(iSub);
            std::vector<index> cellNodes(static_cast<std::size_t>(element.GetNumNodes()));
            for (int i = 0; i < element.GetNumNodes(); i++)
                cellNodes[static_cast<std::size_t>(i)] = parentNodes(i);
            std::vector<index> edgeNodes(static_cast<std::size_t>(edge.GetNumNodes()));
            if (_mesh->getDim() == 2)
                element.ExtractFaceNodes(iSub, cellNodes, edgeNodes);
            else
                element.ExtractEdgeNodes(iSub, cellNodes, edgeNodes);
            for (int i = 0; i < edge.GetNumNodes(); i++)
                out[i] = edgeNodes[static_cast<std::size_t>(i)];
        };
        if (_mesh->isPeriodic)
        {
            query.matchExtra = [this](index iParent, int iSub,
                                      index /*candidateEntity*/,
                                      index candidateParent,
                                      int candidateSub) -> bool
            {
                auto parentA = _mesh->GetCellElement(iParent);
                const auto edgeA = _mesh->getDim() == 2
                                       ? parentA.ObtainFace(iSub)
                                       : parentA.ObtainEdge(iSub);
                const int nNodes = edgeA.GetNumNodes();
                std::vector<index> nodesA(static_cast<std::size_t>(nNodes));
                std::vector<Geom::NodePeriodicBits> pbiA(
                    static_cast<std::size_t>(nNodes));
                if (_mesh->getDim() == 2)
                {
                    parentA.ExtractFaceNodes(
                        iSub, _mesh->cell2node[iParent], nodesA);
                    parentA.ExtractFaceNodes(
                        iSub, _mesh->cell2nodePbi[iParent], pbiA);
                }
                else
                {
                    parentA.ExtractEdgeNodes(
                        iSub, _mesh->cell2node[iParent], nodesA);
                    parentA.ExtractEdgeNodes(
                        iSub, _mesh->cell2nodePbi[iParent], pbiA);
                }

                auto parentB = _mesh->GetCellElement(candidateParent);
                std::vector<index> nodesB(static_cast<std::size_t>(nNodes));
                std::vector<Geom::NodePeriodicBits> pbiB(
                    static_cast<std::size_t>(nNodes));
                if (_mesh->getDim() == 2)
                {
                    parentB.ExtractFaceNodes(
                        candidateSub, _mesh->cell2node[candidateParent], nodesB);
                    parentB.ExtractFaceNodes(
                        candidateSub, _mesh->cell2nodePbi[candidateParent], pbiB);
                }
                else
                {
                    parentB.ExtractEdgeNodes(
                        candidateSub, _mesh->cell2node[candidateParent], nodesB);
                    parentB.ExtractEdgeNodes(
                        candidateSub, _mesh->cell2nodePbi[candidateParent], pbiB);
                }
                using NodeAndPbi = std::pair<index, Geom::NodePeriodicBits>;
                const auto compare = [](const NodeAndPbi &left,
                                        const NodeAndPbi &right)
                {
                    return left.first == right.first
                               ? uint8_t(left.second) < uint8_t(right.second)
                               : left.first < right.first;
                };
                std::vector<NodeAndPbi> pairsA(static_cast<std::size_t>(nNodes));
                std::vector<NodeAndPbi> pairsB(static_cast<std::size_t>(nNodes));
                for (int i = 0; i < nNodes; i++)
                {
                    pairsA[static_cast<std::size_t>(i)] =
                        {nodesA[static_cast<std::size_t>(i)],
                         pbiA[static_cast<std::size_t>(i)]};
                    pairsB[static_cast<std::size_t>(i)] =
                        {nodesB[static_cast<std::size_t>(i)],
                         pbiB[static_cast<std::size_t>(i)]};
                }
                std::sort(pairsA.begin(), pairsA.end(), compare);
                std::sort(pairsB.begin(), pairsB.end(), compare);
                const auto frameXor = pairsA.front().second ^
                                      pairsB.front().second;
                for (int i = 1; i < nNodes; i++)
                    if ((pairsA[static_cast<std::size_t>(i)].second ^
                         pairsB[static_cast<std::size_t>(i)].second) != frameXor)
                        return false;
                return true;
            };
            query.extractPbi = [this](
                                       index iParent,
                                       int iSub,
                                       const std::function<Geom::NodePeriodicBits(int)> &parentPbi,
                                       Geom::NodePeriodicBits *out)
            {
                auto parent = _mesh->GetCellElement(iParent);
                const auto edge = _mesh->getDim() == 2
                                      ? parent.ObtainFace(iSub)
                                      : parent.ObtainEdge(iSub);
                std::vector<Geom::NodePeriodicBits> parentBits(
                    static_cast<std::size_t>(parent.GetNumNodes()));
                for (int i = 0; i < parent.GetNumNodes(); i++)
                    parentBits[static_cast<std::size_t>(i)] = parentPbi(i);
                std::vector<Geom::NodePeriodicBits> edgeBits(
                    static_cast<std::size_t>(edge.GetNumNodes()));
                if (_mesh->getDim() == 2)
                    parent.ExtractFaceNodes(iSub, parentBits, edgeBits);
                else
                    parent.ExtractEdgeNodes(iSub, parentBits, edgeBits);
                for (int i = 0; i < edge.GetNumNodes(); i++)
                    out[i] = edgeBits[static_cast<std::size_t>(i)];
            };
        }
        return query;
    }

    void Topology::Build()
    {
        DNDS_check_throw_info(_mesh->cell2node.isLocal(),
                              "NCFV requires local primal cell-to-node indices");
        DNDS_check_throw_info(
            !_mesh->isPeriodic || _mesh->cell2nodePbi.father,
            "NCFV periodic topology requires cell-to-node periodic bits");

        for (index iCell = 0; iCell < _mesh->NumCellProc(); iCell++)
        {
            const auto element = _mesh->GetCellElement(iCell);
            DNDS_check_throw_info(element.GetOrder() == 1,
                                  "NCFV dual construction currently requires O1 primal elements");
            DNDS_check_throw_info(element.GetDim() == _mesh->getDim(),
                                  "NCFV encountered a cell with the wrong topological dimension");
        }

        Geom::OwnershipResolverMulti ownership =
            [this](const std::vector<index> &parents,
                   const std::vector<MPI_int> &parentRanks,
                   index nLocalParents) -> Geom::OwnershipDecision
        {
            DNDS_check_throw_info(!parents.empty(), "A primal edge has no parent cell");
            MPI_int minimumRank = *std::min_element(parentRanks.begin(), parentRanks.end());
            bool hasLocalParent = false;
            for (index parent : parents)
                hasLocalParent = hasLocalParent || parent < nLocalParents;
            if (!hasLocalParent || minimumRank != _mpi.rank)
                return {false, {}};

            std::vector<MPI_int> peers;
            for (std::size_t i = 0; i < parents.size(); i++)
                if (parents[i] >= nLocalParents && parentRanks[i] != _mpi.rank)
                    peers.push_back(parentRanks[i]);
            std::sort(peers.begin(), peers.end());
            peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
            return {true, std::move(peers)};
        };

        auto result = Geom::MeshConnectivity::InterpolateGlobal(
            _mesh->cell2node,
            _mesh->isPeriodic ? _mesh->cell2nodePbi : Geom::tPbiPair{},
            *_mesh->cell2node.trans.pLGhostMapping,
            *_mesh->cell2node.father->pLGlobalMapping,
            *_mesh->coords.trans.pLGhostMapping,
            BuildEdgeQuery(),
            _mesh->NumCell(),
            _mesh->NumCellProc(),
            _mesh->NumNodeProc(),
            ownership,
            _mpi);

        _cell2edge.father = result.parent2entity.father;
        _cell2edge.son = result.parent2entity.son;

        _edge2node.father = result.entity2node.father;
        _edge2node.son = result.entity2node.son;
        if (_mesh->isPeriodic)
        {
            _edge2nodePbi.father = result.entity2nodePbi.father;
            _edge2nodePbi.son = result.entity2nodePbi.son;
        }
        _edge2node.TransAttach();
        _edge2node.trans.createFatherGlobalMapping();

        // InterpolateGlobal resolves every edge of an owned cell.  A node-centred
        // control volume additionally needs edges that touch an owned node but
        // whose incident cells are all ghosts on this rank.  Resolve those IDs
        // through a distributed endpoint-pair directory before building the
        // ordinary edge ghost mapping.
        std::vector<index> visibleEdgeGlobals = ResolveOwnedNodeEdgeGlobals();
        for (index iCell = 0; iCell < _cell2edge.Size(); iCell++)
            for (rowsize i = 0; i < _cell2edge.RowSize(iCell); i++)
            {
                const index globalEdge = _cell2edge(iCell, i);
                if (globalEdge == UnInitIndex)
                    continue;
                visibleEdgeGlobals.push_back(globalEdge);
            }
        std::sort(visibleEdgeGlobals.begin(), visibleEdgeGlobals.end());
        visibleEdgeGlobals.erase(
            std::unique(visibleEdgeGlobals.begin(), visibleEdgeGlobals.end()),
            visibleEdgeGlobals.end());

        std::vector<index> ghostEdgeGlobals;
        for (index globalEdge : visibleEdgeGlobals)
        {
            MPI_int ownerRank = UnInitMPIInt;
            index ownerLocal = UnInitIndex;
            const bool found = _edge2node.father->pLGlobalMapping->search(
                globalEdge, ownerRank, ownerLocal);
            DNDS_check_throw_info(found, "NCFV could not resolve an interpolated edge owner");
            if (ownerRank != _mpi.rank)
                ghostEdgeGlobals.push_back(globalEdge);
        }
        std::sort(ghostEdgeGlobals.begin(), ghostEdgeGlobals.end());
        ghostEdgeGlobals.erase(std::unique(ghostEdgeGlobals.begin(), ghostEdgeGlobals.end()),
                               ghostEdgeGlobals.end());
        _edge2node.trans.createGhostMapping(ghostEdgeGlobals);
        _edge2node.trans.createMPITypes();
        _edge2node.trans.pullOnce();

        if (_mesh->isPeriodic)
            _edge2nodePbi.BorrowAndPull(_edge2node);

        _edge2cell.father = result.entity2parent.father;
        _edge2cell.son = result.entity2parent.son;
        _edge2cell.BorrowAndPull(_edge2node);

        _edgeElemInfo.father = result.entityElemInfo.father;
        _edgeElemInfo.son = result.entityElemInfo.son;
        _edgeElemInfo.BorrowAndPull(_edge2node);

        BuildFaceHalo();
        CanonicalizeAndLocalize();
        BuildNodeIncidenceAndAudit();

        index globalEdges = _edge2node.father->globalSize();
        if (_mpi.rank == 0)
            log() << "NCFV primal-edge topology: global edges=" << globalEdges << std::endl;
    }

    std::vector<index> Topology::ResolveOwnedNodeEdgeGlobals() const
    {
        std::vector<std::vector<index>> advertisements(
            static_cast<std::size_t>(_mpi.size));
        for (index iEdge = 0; iEdge < _edge2node.father->Size(); iEdge++)
        {
            const EdgeKey key = CanonicalEdgeKey(
                _edge2node.father->operator()(iEdge, 0),
                _edge2node.father->operator()(iEdge, 1),
                _mesh->isPeriodic
                    ? _edge2nodePbi.father->operator()(iEdge, 0)
                    : Geom::NodePeriodicBits{},
                _mesh->isPeriodic
                    ? _edge2nodePbi.father->operator()(iEdge, 1)
                    : Geom::NodePeriodicBits{});
            const index globalEdge =
                _edge2node.father->pLGlobalMapping->operator()(_mpi.rank, iEdge);
            auto &message = advertisements[static_cast<std::size_t>(
                EdgeDirectoryRank(key, _mpi.size))];
            message.insert(
                message.end(),
                {key.nodes[0], key.nodes[1], key.relativePbi, globalEdge});
        }

        const ExchangeResult receivedAdvertisements =
            ExchangeByRank(advertisements, _mpi);
        DNDS_check_throw_info(receivedAdvertisements.values.size() % 4 == 0,
                              "NCFV edge-directory advertisement is malformed");
        std::map<EdgeKey, index> directory;
        for (std::size_t offset = 0; offset < receivedAdvertisements.values.size();
             offset += 4)
        {
            const EdgeKey key{
                {receivedAdvertisements.values[offset],
                 receivedAdvertisements.values[offset + 1]},
                receivedAdvertisements.values[offset + 2]};
            const index globalEdge = receivedAdvertisements.values[offset + 3];
            const auto [iterator, inserted] = directory.emplace(key, globalEdge);
            DNDS_check_throw_info(
                inserted || iterator->second == globalEdge,
                "NCFV global edge interpolation produced duplicate endpoint pairs");
        }

        std::set<EdgeKey> requiredSet;
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            for (index iCell : _mesh->node2cell[iNode])
            {
                DNDS_check_throw_info(iCell >= 0,
                                      "NCFV owned node has an unresolved incident cell");
                auto cell = _mesh->GetCellElement(iCell);
                const int nEdges = _mesh->getDim() == 2
                                       ? cell.GetNumFaces()
                                       : cell.GetNumEdges();
                for (int iLocalEdge = 0; iLocalEdge < nEdges; iLocalEdge++)
                {
                    const auto edge = _mesh->getDim() == 2
                                          ? cell.ObtainFace(iLocalEdge)
                                          : cell.ObtainEdge(iLocalEdge);
                    std::vector<index> nodes(static_cast<std::size_t>(edge.GetNumNodes()));
                    std::vector<Geom::NodePeriodicBits> pbi(
                        static_cast<std::size_t>(edge.GetNumNodes()));
                    if (_mesh->getDim() == 2)
                    {
                        cell.ExtractFaceNodes(iLocalEdge, _mesh->cell2node[iCell], nodes);
                        if (_mesh->isPeriodic)
                            cell.ExtractFaceNodes(
                                iLocalEdge, _mesh->cell2nodePbi[iCell], pbi);
                    }
                    else
                    {
                        cell.ExtractEdgeNodes(iLocalEdge, _mesh->cell2node[iCell], nodes);
                        if (_mesh->isPeriodic)
                            cell.ExtractEdgeNodes(
                                iLocalEdge, _mesh->cell2nodePbi[iCell], pbi);
                    }
                    DNDS_check_throw_info(nodes.size() >= 2,
                                          "NCFV encountered an invalid primal edge");
                    if (nodes[0] != iNode && nodes[1] != iNode)
                        continue;
                    requiredSet.insert(CanonicalEdgeKey(
                        _mesh->NodeIndexLocal2Global(nodes[0]),
                        _mesh->NodeIndexLocal2Global(nodes[1]),
                        pbi[0], pbi[1]));
                }
            }

        const std::vector<EdgeKey> required(requiredSet.begin(), requiredSet.end());
        std::vector<std::vector<index>> queries(static_cast<std::size_t>(_mpi.size));
        for (std::size_t slot = 0; slot < required.size(); slot++)
        {
            const EdgeKey &key = required[slot];
            auto &message = queries[static_cast<std::size_t>(
                EdgeDirectoryRank(key, _mpi.size))];
            message.insert(message.end(),
                           {key.nodes[0], key.nodes[1], key.relativePbi,
                            static_cast<index>(slot)});
        }

        const ExchangeResult receivedQueries = ExchangeByRank(queries, _mpi);
        std::vector<std::vector<index>> replies(static_cast<std::size_t>(_mpi.size));
        for (MPI_int sourceRank = 0; sourceRank < _mpi.size; sourceRank++)
        {
            const int begin = receivedQueries.receiveOffsets[static_cast<std::size_t>(sourceRank)];
            const int end = receivedQueries.receiveOffsets[static_cast<std::size_t>(sourceRank + 1)];
            DNDS_check_throw_info((end - begin) % 4 == 0,
                                  "NCFV edge-directory query is malformed");
            auto &reply = replies[static_cast<std::size_t>(sourceRank)];
            for (int offset = begin; offset < end; offset += 4)
            {
                const EdgeKey key{
                    {receivedQueries.values[static_cast<std::size_t>(offset)],
                     receivedQueries.values[static_cast<std::size_t>(offset + 1)]},
                    receivedQueries.values[static_cast<std::size_t>(offset + 2)]};
                const index slot =
                    receivedQueries.values[static_cast<std::size_t>(offset + 3)];
                const auto iterator = directory.find(key);
                DNDS_check_throw_info(
                    iterator != directory.end(),
                    fmt::format(
                        "NCFV could not resolve edge ({}, {}, pbi={}) in the distributed directory",
                        key.nodes[0], key.nodes[1], key.relativePbi));
                reply.push_back(slot);
                reply.push_back(iterator->second);
            }
        }

        const ExchangeResult receivedReplies = ExchangeByRank(replies, _mpi);
        DNDS_check_throw_info(receivedReplies.values.size() % 2 == 0,
                              "NCFV edge-directory reply is malformed");
        std::vector<index> resolved(required.size(), UnInitIndex);
        for (std::size_t offset = 0; offset < receivedReplies.values.size(); offset += 2)
        {
            const index slot = receivedReplies.values[offset];
            DNDS_check_throw_info(slot >= 0 &&
                                      slot < static_cast<index>(resolved.size()),
                                  "NCFV edge-directory reply contains an invalid slot");
            resolved[static_cast<std::size_t>(slot)] =
                receivedReplies.values[offset + 1];
        }
        DNDS_check_throw_info(
            std::find(resolved.begin(), resolved.end(), UnInitIndex) == resolved.end(),
            "NCFV edge-directory did not answer every owned-node edge query");
        return resolved;
    }

    void Topology::BuildFaceHalo()
    {
        constexpr std::size_t recordWidth = 10;

        // The mesh face halo is cell-centred: it contains every face needed by
        // owned cells.  A node-centred dual volume additionally needs faces of
        // fully ghost incident cells.  Resolve all faces in the available cell
        // halo by a distributed canonical-vertex-set directory, then keep that
        // enlarged halo private to NCFV.
        std::vector<std::vector<index>> advertisements(
            static_cast<std::size_t>(_mpi.size));
        for (index iFace = 0; iFace < _mesh->NumFace(); iFace++)
        {
            const auto face = _mesh->GetFaceElement(iFace);
            std::vector<index> globalNodes(
                static_cast<std::size_t>(face.GetNumVertices()));
            std::vector<Geom::NodePeriodicBits> pbi(
                static_cast<std::size_t>(face.GetNumVertices()));
            for (int iNode = 0; iNode < face.GetNumVertices(); iNode++)
            {
                globalNodes[static_cast<std::size_t>(iNode)] =
                    _mesh->NodeIndexLocal2Global(_mesh->face2node(iFace, iNode));
                if (_mesh->isPeriodic)
                    pbi[static_cast<std::size_t>(iNode)] =
                        _mesh->face2nodePbi(iFace, iNode);
            }
            const FaceKey key = CanonicalFaceKey(globalNodes, pbi);
            auto &message = advertisements[static_cast<std::size_t>(
                FaceDirectoryRank(key, _mpi.size))];
            AppendFaceRecord(message, key, _mesh->FaceIndexLocal2Global(iFace));
        }

        const ExchangeResult receivedAdvertisements =
            ExchangeByRank(advertisements, _mpi);
        DNDS_check_throw_info(
            receivedAdvertisements.values.size() % recordWidth == 0,
            "NCFV face-directory advertisement is malformed");
        std::map<FaceKey, index> directory;
        for (std::size_t offset = 0; offset < receivedAdvertisements.values.size();
             offset += recordWidth)
        {
            const FaceKey key = ReadFaceRecord(receivedAdvertisements.values, offset);
            const index globalFace = receivedAdvertisements.values[offset + 9];
            const auto [iterator, inserted] = directory.emplace(key, globalFace);
            DNDS_check_throw_info(
                inserted || iterator->second == globalFace,
                "NCFV mesh contains duplicate global faces with the same O1 vertices");
        }

        std::vector<std::vector<FaceKey>> cellFaceKeys(
            static_cast<std::size_t>(_mesh->NumCellProc()));
        std::set<FaceKey> requiredSet;
        for (index iCell = 0; iCell < _mesh->NumCellProc(); iCell++)
        {
            auto cell = _mesh->GetCellElement(iCell);
            auto &keys = cellFaceKeys[static_cast<std::size_t>(iCell)];
            keys.reserve(static_cast<std::size_t>(cell.GetNumFaces()));
            for (int iLocalFace = 0; iLocalFace < cell.GetNumFaces(); iLocalFace++)
            {
                const auto face = cell.ObtainFace(iLocalFace);
                std::vector<index> localNodes(static_cast<std::size_t>(face.GetNumNodes()));
                std::vector<Geom::NodePeriodicBits> facePbi(
                    static_cast<std::size_t>(face.GetNumNodes()));
                cell.ExtractFaceNodes(iLocalFace, _mesh->cell2node[iCell], localNodes);
                if (_mesh->isPeriodic)
                    cell.ExtractFaceNodes(
                        iLocalFace, _mesh->cell2nodePbi[iCell], facePbi);
                localNodes.resize(static_cast<std::size_t>(face.GetNumVertices()));
                facePbi.resize(static_cast<std::size_t>(face.GetNumVertices()));
                std::vector<index> globalNodes(localNodes.size());
                for (std::size_t iNode = 0; iNode < localNodes.size(); iNode++)
                    globalNodes[iNode] = _mesh->NodeIndexLocal2Global(localNodes[iNode]);
                FaceKey key = CanonicalFaceKey(globalNodes, facePbi);
                keys.push_back(key);
                requiredSet.insert(std::move(key));
            }
        }

        const std::vector<FaceKey> required(requiredSet.begin(), requiredSet.end());
        std::vector<std::vector<index>> queries(static_cast<std::size_t>(_mpi.size));
        for (std::size_t slot = 0; slot < required.size(); slot++)
        {
            const FaceKey &key = required[slot];
            auto &message = queries[static_cast<std::size_t>(
                FaceDirectoryRank(key, _mpi.size))];
            AppendFaceRecord(message, key, static_cast<index>(slot));
        }

        const ExchangeResult receivedQueries = ExchangeByRank(queries, _mpi);
        std::vector<std::vector<index>> replies(static_cast<std::size_t>(_mpi.size));
        for (MPI_int sourceRank = 0; sourceRank < _mpi.size; sourceRank++)
        {
            const int begin = receivedQueries.receiveOffsets[static_cast<std::size_t>(sourceRank)];
            const int end = receivedQueries.receiveOffsets[static_cast<std::size_t>(sourceRank + 1)];
            DNDS_check_throw_info((end - begin) % static_cast<int>(recordWidth) == 0,
                                  "NCFV face-directory query is malformed");
            auto &reply = replies[static_cast<std::size_t>(sourceRank)];
            for (int offset = begin; offset < end; offset += static_cast<int>(recordWidth))
            {
                const FaceKey key = ReadFaceRecord(
                    receivedQueries.values, static_cast<std::size_t>(offset));
                const index slot = receivedQueries.values[static_cast<std::size_t>(offset) + 9];
                const auto iterator = directory.find(key);
                DNDS_check_throw_info(
                    iterator != directory.end(),
                    "NCFV could not resolve a primal face in the distributed directory");
                reply.push_back(slot);
                reply.push_back(iterator->second);
            }
        }

        const ExchangeResult receivedReplies = ExchangeByRank(replies, _mpi);
        DNDS_check_throw_info(receivedReplies.values.size() % 2 == 0,
                              "NCFV face-directory reply is malformed");
        std::vector<index> resolved(required.size(), UnInitIndex);
        for (std::size_t offset = 0; offset < receivedReplies.values.size(); offset += 2)
        {
            const index slot = receivedReplies.values[offset];
            DNDS_check_throw_info(slot >= 0 && slot < static_cast<index>(resolved.size()),
                                  "NCFV face-directory reply contains an invalid slot");
            resolved[static_cast<std::size_t>(slot)] =
                receivedReplies.values[offset + 1];
        }
        DNDS_check_throw_info(
            std::find(resolved.begin(), resolved.end(), UnInitIndex) == resolved.end(),
            "NCFV face-directory did not answer every face query");

        std::map<FaceKey, index> requiredGlobals;
        std::vector<index> ghostGlobals;
        for (std::size_t slot = 0; slot < required.size(); slot++)
        {
            const index globalFace = resolved[slot];
            requiredGlobals.emplace(required[slot], globalFace);
            MPI_int ownerRank = UnInitMPIInt;
            index ownerLocal = UnInitIndex;
            const bool found = _mesh->face2node.father->pLGlobalMapping->search(
                globalFace, ownerRank, ownerLocal);
            DNDS_check_throw_info(found, "NCFV could not resolve a primal-face owner");
            if (ownerRank != _mpi.rank)
                ghostGlobals.push_back(globalFace);
        }
        std::sort(ghostGlobals.begin(), ghostGlobals.end());
        ghostGlobals.erase(std::unique(ghostGlobals.begin(), ghostGlobals.end()),
                           ghostGlobals.end());

        // Copy the owned face data to global node indices before pulling.  The
        // original mesh has already localized its face rows, so sharing those
        // arrays directly would send owner-local node numbers across ranks.
        _face2node.InitPair("NCFV.face2node", _mpi);
        _face2node.father->Resize(_mesh->NumFace());
        for (index iFace = 0; iFace < _mesh->NumFace(); iFace++)
        {
            _face2node.father->ResizeRow(iFace, _mesh->face2node.RowSize(iFace));
            for (rowsize iNode = 0; iNode < _mesh->face2node.RowSize(iFace); iNode++)
                _face2node(iFace, iNode) = _mesh->NodeIndexLocal2Global(
                    _mesh->face2node(iFace, iNode));
        }
        _face2node.father->Compress();
        _face2node.TransAttach();
        _face2node.trans.createFatherGlobalMapping();
        _face2node.trans.createGhostMapping(ghostGlobals);
        _face2node.trans.createMPITypes();
        _face2node.trans.pullOnce();

        _faceElemInfo.InitPair("NCFV.faceElemInfo", _mpi);
        _faceElemInfo.father->Resize(_mesh->NumFace());
        for (index iFace = 0; iFace < _mesh->NumFace(); iFace++)
            _faceElemInfo(iFace, 0) = _mesh->faceElemInfo(iFace, 0);
        _faceElemInfo.BorrowAndPull(_face2node);

        for (index iFace = 0; iFace < _face2node.Size(); iFace++)
            for (rowsize iNode = 0; iNode < _face2node.RowSize(iFace); iNode++)
            {
                const index localNode = _mesh->NodeIndexGlobal2Local(
                    _face2node(iFace, iNode));
                DNDS_check_throw_info(localNode >= 0,
                                      "NCFV primal-face node is absent from the node halo");
                _face2node(iFace, iNode) = localNode;
            }

        _cell2face.clear();
        _cell2face.resize(static_cast<std::size_t>(_mesh->NumCellProc()));
        for (index iCell = 0; iCell < _mesh->NumCellProc(); iCell++)
        {
            auto &faces = _cell2face[static_cast<std::size_t>(iCell)];
            const auto &keys = cellFaceKeys[static_cast<std::size_t>(iCell)];
            faces.resize(keys.size(), UnInitIndex);
            for (std::size_t iLocalFace = 0; iLocalFace < keys.size(); iLocalFace++)
            {
                const auto globalIterator = requiredGlobals.find(keys[iLocalFace]);
                DNDS_check_throw_info(globalIterator != requiredGlobals.end(),
                                      "NCFV lost a resolved primal face");
                MPI_int ownerRank = UnInitMPIInt;
                index localFace = UnInitIndex;
                const bool found = _face2node.trans.pLGhostMapping->search_indexAppend(
                    globalIterator->second, ownerRank, localFace);
                DNDS_check_throw_info(found && localFace >= 0,
                                      "NCFV primal face is absent from its private halo");
                faces[iLocalFace] = localFace;
            }
        }
    }

    void Topology::CanonicalizeAndLocalize()
    {
        for (index iEdge = 0; iEdge < _edge2node.Size(); iEdge++)
        {
            DNDS_check_throw_info(_edge2node.RowSize(iEdge) == 2,
                                  "NCFV O1 primal edges must have exactly two nodes");
            if (_edge2node(iEdge, 1) < _edge2node(iEdge, 0))
            {
                std::swap(_edge2node(iEdge, 0), _edge2node(iEdge, 1));
                if (_mesh->isPeriodic)
                    std::swap(
                        _edge2nodePbi(iEdge, 0),
                        _edge2nodePbi(iEdge, 1));
            }
            for (rowsize i = 0; i < _edge2node.RowSize(iEdge); i++)
            {
                const index localNode = _mesh->NodeIndexGlobal2Local(_edge2node(iEdge, i));
                DNDS_check_throw_info(localNode >= 0,
                                      "NCFV edge endpoint is absent from the node halo");
                _edge2node(iEdge, i) = localNode;
            }

            std::vector<index> parentGlobals;
            parentGlobals.reserve(static_cast<std::size_t>(_edge2cell.RowSize(iEdge)));
            for (rowsize i = 0; i < _edge2cell.RowSize(iEdge); i++)
                parentGlobals.push_back(_edge2cell(iEdge, i));
            std::sort(parentGlobals.begin(), parentGlobals.end());
            for (rowsize i = 0; i < _edge2cell.RowSize(iEdge); i++)
            {
                const index localCell = _mesh->CellIndexGlobal2Local(
                    parentGlobals[static_cast<std::size_t>(i)]);
                DNDS_check_throw_info(localCell >= 0,
                                      "NCFV edge parent is absent from the fixed point-complete cell halo");
                _edge2cell(iEdge, i) = localCell;
            }
        }

        for (index iCell = 0; iCell < _cell2edge.Size(); iCell++)
            for (rowsize i = 0; i < _cell2edge.RowSize(iCell); i++)
            {
                const index globalEdge = _cell2edge(iCell, i);
                if (globalEdge == UnInitIndex)
                    continue;
                MPI_int ownerRank = UnInitMPIInt;
                index localEdge = UnInitIndex;
                const bool found = _edge2node.trans.pLGhostMapping->search_indexAppend(
                    globalEdge, ownerRank, localEdge);
                if (found)
                    _cell2edge(iCell, i) = localEdge;
            }
    }

    void Topology::BuildNodeIncidenceAndAudit()
    {
        _node2edge.clear();
        _node2edge.resize(static_cast<std::size_t>(_mesh->NumNodeProc()));
        for (index iEdge = 0; iEdge < _edge2node.Size(); iEdge++)
        {
            const index node0 = _edge2node(iEdge, 0);
            const index node1 = _edge2node(iEdge, 1);
            _node2edge[static_cast<std::size_t>(node0)].push_back({iEdge, 1.0});
            _node2edge[static_cast<std::size_t>(node1)].push_back({iEdge, -1.0});
        }

        for (auto &incidences : _node2edge)
            std::sort(incidences.begin(), incidences.end(), [this](const auto &left, const auto &right)
                      { return EdgeIndexLocal2Global(left.edge) < EdgeIndexLocal2Global(right.edge); });

        // Every topological edge of every cell touching an owned node must be
        // represented in the edge halo.  This catches an insufficient ghost tree
        // before reconstruction or residual evaluation can silently omit a flux.
        std::vector<std::set<index>> actualNeighbors(static_cast<std::size_t>(_mesh->NumNode()));
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            for (const auto &incidence : _node2edge[static_cast<std::size_t>(iNode)])
            {
                const auto row = _edge2node[incidence.edge];
                actualNeighbors[static_cast<std::size_t>(iNode)].insert(
                    row[0] == iNode ? row[1] : row[0]);
            }

        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            std::set<index> expectedNeighbors;
            for (index iCell : _mesh->node2cell[iNode])
            {
                DNDS_check_throw_info(iCell >= 0, "Owned NCFV node has an unresolved incident cell");
                auto element = _mesh->GetCellElement(iCell);
                const int nEdges = _mesh->getDim() == 2
                                       ? element.GetNumFaces()
                                       : element.GetNumEdges();
                for (int iEdge = 0; iEdge < nEdges; iEdge++)
                {
                    const auto edge = _mesh->getDim() == 2
                                          ? element.ObtainFace(iEdge)
                                          : element.ObtainEdge(iEdge);
                    std::vector<index> edgeNodes(static_cast<std::size_t>(edge.GetNumNodes()));
                    if (_mesh->getDim() == 2)
                        element.ExtractFaceNodes(iEdge, _mesh->cell2node[iCell], edgeNodes);
                    else
                        element.ExtractEdgeNodes(iEdge, _mesh->cell2node[iCell], edgeNodes);
                    if (edgeNodes[0] == iNode)
                        expectedNeighbors.insert(edgeNodes[1]);
                    if (edgeNodes[1] == iNode)
                        expectedNeighbors.insert(edgeNodes[0]);
                }
            }
            DNDS_check_throw_info(
                expectedNeighbors == actualNeighbors[static_cast<std::size_t>(iNode)],
                "NCFV edge halo is incomplete for an owned node after point-complete halo construction");
        }
    }
}
