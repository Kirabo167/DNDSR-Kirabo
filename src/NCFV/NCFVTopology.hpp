/**
 * @file NCFVTopology.hpp
 * @brief Distributed primal-edge topology owned by the standalone NCFV module.
 */
#pragma once

#include "Geom/Mesh/Mesh.hpp"

#include <vector>

namespace DNDS::NCFV
{
    struct NodeEdgeIncidence
    {
        index edge = UnInitIndex;
        real outwardSign = 0;
    };

    /**
     * @brief Global primal edges and their cell/node incidence.
     *
     * The object deliberately lives outside UnstructuredMesh.  It uses the same
     * father/son and pull mapping conventions, but does not add fields to Geom.
     */
    class Topology
    {
        const MPIInfo &_mpi;
        ssp<Geom::UnstructuredMesh> _mesh;

        Geom::tAdjPair _cell2edge;
        Geom::tAdjPair _edge2node;
        Geom::tPbiPair _edge2nodePbi;
        Geom::tAdjPair _edge2cell;
        Geom::tElemInfoArrayPair _edgeElemInfo;
        Geom::tAdjPair _face2node;
        Geom::tElemInfoArrayPair _faceElemInfo;
        std::vector<std::vector<index>> _cell2face;
        std::vector<std::vector<NodeEdgeIncidence>> _node2edge;

        /**
         * @brief Build per-cell primal-edge queries with periodic endpoint information.
         * @return Sub-entity query used to match and number edges across MPI partitions.
         */
        Geom::SubEntityQueryPbi BuildEdgeQuery() const;
        /**
         * @brief Resolve the global edges needed to complete every owned-node edge star.
         * @return Sorted unique global edge IDs required by owned nodes.
         * @note Uses distributed edge-directory exchanges.
         */
        std::vector<index> ResolveOwnedNodeEdgeGlobals() const;
        /**
         * @brief Resolve primal-face ownership and fetch faces required by the local cell/edge geometry.
         */
        void BuildFaceHalo();
        /**
         * @brief Canonicalize endpoint ordering and convert edge/face incidence to process-local connectivity.
         */
        void CanonicalizeAndLocalize();
        /**
         * @brief Assemble signed owned-node edge incidence and check that all referenced cells and edges are
         *     available.
         */
        void BuildNodeIncidenceAndAudit();

    public:
        /**
         * @brief Bind an NCFV edge/face topology to a distributed primal mesh.
         * @param mpi MPI communicator metadata; collective operations must be entered by every participating rank.
         * @param mesh Distributed O1 primal mesh with owned/ghost connectivity prepared for NCFV.
         * @note The referenced MPIInfo must outlive this object; Build() performs topology construction.
         */
        Topology(const MPIInfo &mpi, const ssp<Geom::UnstructuredMesh> &mesh)
            : _mpi(mpi), _mesh(mesh)
        {
        }

        /**
         * @brief Construct globally unique primal edges, face halos, and owned-node edge incidence.
         * @details Resolves distributed ownership and periodic identities, converts connectivity to local indices,
         *     and audits complete owned-node stars.
         * @note Collective over the mesh communicator; the mesh connectivity must be ready.
         */
        void Build();

        /**
         * @brief Return number of edges owned by this rank.
         * @return Number of edges owned by this rank.
         */
        [[nodiscard]] index NumEdge() const { return _edge2node.father->Size(); }
        /**
         * @brief Return number of ghost edges retained by this rank.
         * @return Number of ghost edges retained by this rank.
         */
        [[nodiscard]] index NumEdgeGhost() const { return _edge2node.son->Size(); }
        /**
         * @brief Return total process-local edge count, owned plus ghost.
         * @return Total process-local edge count, owned plus ghost.
         */
        [[nodiscard]] index NumEdgeProc() const { return _edge2node.Size(); }
        /**
         * @brief Return global number of uniquely owned edges.
         * @return Global number of uniquely owned edges.
         */
        [[nodiscard]] index NumEdgeGlobal() const { return _edge2node.father->globalSize(); }
        /**
         * @brief Return number of primal faces owned by this rank.
         * @return Number of primal faces owned by this rank.
         */
        [[nodiscard]] index NumFace() const { return _face2node.father->Size(); }
        /**
         * @brief Return number of ghost primal faces retained by this rank.
         * @return Number of ghost primal faces retained by this rank.
         */
        [[nodiscard]] index NumFaceGhost() const { return _face2node.son->Size(); }
        /**
         * @brief Return total process-local primal-face count, owned plus ghost.
         * @return Total process-local primal-face count, owned plus ghost.
         */
        [[nodiscard]] index NumFaceProc() const { return _face2node.Size(); }

        /**
         * @brief Convert an owned or ghost edge index to its stable distributed ID.
         * @param iEdge Process-local edge index in [0,NumEdgeProc()).
         * @return Global edge ID in the topology ownership mapping.
         */
        [[nodiscard]] index EdgeIndexLocal2Global(index iEdge) const
        {
            return _edge2node.trans.pLGhostMapping->operator()(-1, iEdge);
        }

        /**
         * @brief Return read-only process-local primal-edge IDs for each local cell.
         * @return Read-only process-local primal-edge IDs for each local cell.
         */
        [[nodiscard]] const Geom::tAdjPair &Cell2Edge() const { return _cell2edge; }
        /**
         * @brief Return endpoint connectivity of owned/ghost edges, in canonical order after Build().
         * @return Endpoint connectivity of owned/ghost edges, in canonical order after Build().
         */
        [[nodiscard]] Geom::tAdjPair &Edge2Node() { return _edge2node; }
        /**
         * @brief Return endpoint connectivity of owned/ghost edges, in canonical order after Build().
         * @return Endpoint connectivity of owned/ghost edges, in canonical order after Build().
         */
        [[nodiscard]] const Geom::tAdjPair &Edge2Node() const { return _edge2node; }
        /**
         * @brief Return read-only periodic-image bits for each canonical edge endpoint.
         * @return Read-only periodic-image bits for each canonical edge endpoint.
         */
        [[nodiscard]] const Geom::tPbiPair &Edge2NodePbi() const
        {
            return _edge2nodePbi;
        }
        /**
         * @brief Return read-only adjacent primal-cell IDs for each local edge.
         * @return Read-only adjacent primal-cell IDs for each local edge.
         */
        [[nodiscard]] const Geom::tAdjPair &Edge2Cell() const { return _edge2cell; }
        /**
         * @brief Return read-only element metadata for owned/ghost primal edges.
         * @return Read-only element metadata for owned/ghost primal edges.
         */
        [[nodiscard]] const Geom::tElemInfoArrayPair &EdgeElemInfo() const { return _edgeElemInfo; }
        /**
         * @brief Return read-only vertex connectivity of owned/ghost primal faces.
         * @return Read-only vertex connectivity of owned/ghost primal faces.
         */
        [[nodiscard]] const Geom::tAdjPair &Face2Node() const { return _face2node; }
        /**
         * @brief Return read-only primal-face element types and boundary-zone IDs.
         * @return Read-only primal-face element types and boundary-zone IDs.
         */
        [[nodiscard]] const Geom::tElemInfoArrayPair &FaceElemInfo() const { return _faceElemInfo; }
        /**
         * @brief Find a process-local primal face from its cell-local face number.
         * @param iCell Process-local primal-cell index.
         * @param iLocalFace Zero-based face number in that cell's element topology.
         * @return Process-local primal-face index.
         */
        [[nodiscard]] index CellFace(index iCell, int iLocalFace) const
        {
            return _cell2face.at(static_cast<std::size_t>(iCell))
                .at(static_cast<std::size_t>(iLocalFace));
        }
        /**
         * @brief Return the reference-element descriptor of a primal face.
         * @param iFace Process-local primal-face index in [0,NumFaceProc()).
         * @return Element descriptor containing face shape and vertex ordering.
         */
        [[nodiscard]] Geom::Elem::Element GetFaceElement(index iFace) const
        {
            return Geom::Elem::Element{_faceElemInfo(iFace, 0).getElemType()};
        }
        /**
         * @brief Test whether a primal face belongs to an external physical boundary.
         * @param iFace Process-local primal-face index.
         * @return True when the face-zone ID is classified as an external boundary.
         */
        [[nodiscard]] bool FaceIsBoundary(index iFace) const
        {
            return Geom::FaceIDIsExternalBC(_faceElemInfo(iFace, 0).zone);
        }
        /**
         * @brief Return the signed edge incidences used to sum outward fluxes for an owned node.
         * @param iNode Owned primal-node index.
         * @return Read-only incidence list; each entry stores a local edge ID and its outward flux sign.
         */
        [[nodiscard]] const std::vector<NodeEdgeIncidence> &Node2Edge(index iNode) const
        {
            return _node2edge.at(static_cast<std::size_t>(iNode));
        }
    };
}
