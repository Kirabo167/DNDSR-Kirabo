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

        Geom::SubEntityQueryPbi BuildEdgeQuery() const;
        std::vector<index> ResolveOwnedNodeEdgeGlobals() const;
        void BuildFaceHalo();
        void CanonicalizeAndLocalize();
        void BuildNodeIncidenceAndAudit();

    public:
        Topology(const MPIInfo &mpi, const ssp<Geom::UnstructuredMesh> &mesh)
            : _mpi(mpi), _mesh(mesh)
        {
        }

        void Build();

        [[nodiscard]] index NumEdge() const { return _edge2node.father->Size(); }
        [[nodiscard]] index NumEdgeGhost() const { return _edge2node.son->Size(); }
        [[nodiscard]] index NumEdgeProc() const { return _edge2node.Size(); }
        [[nodiscard]] index NumEdgeGlobal() const { return _edge2node.father->globalSize(); }
        [[nodiscard]] index NumFace() const { return _face2node.father->Size(); }
        [[nodiscard]] index NumFaceGhost() const { return _face2node.son->Size(); }
        [[nodiscard]] index NumFaceProc() const { return _face2node.Size(); }

        [[nodiscard]] index EdgeIndexLocal2Global(index iEdge) const
        {
            return _edge2node.trans.pLGhostMapping->operator()(-1, iEdge);
        }

        [[nodiscard]] const Geom::tAdjPair &Cell2Edge() const { return _cell2edge; }
        [[nodiscard]] Geom::tAdjPair &Edge2Node() { return _edge2node; }
        [[nodiscard]] const Geom::tAdjPair &Edge2Node() const { return _edge2node; }
        [[nodiscard]] const Geom::tPbiPair &Edge2NodePbi() const
        {
            return _edge2nodePbi;
        }
        [[nodiscard]] const Geom::tAdjPair &Edge2Cell() const { return _edge2cell; }
        [[nodiscard]] const Geom::tElemInfoArrayPair &EdgeElemInfo() const { return _edgeElemInfo; }
        [[nodiscard]] const Geom::tAdjPair &Face2Node() const { return _face2node; }
        [[nodiscard]] const Geom::tElemInfoArrayPair &FaceElemInfo() const { return _faceElemInfo; }
        [[nodiscard]] index CellFace(index iCell, int iLocalFace) const
        {
            return _cell2face.at(static_cast<std::size_t>(iCell))
                .at(static_cast<std::size_t>(iLocalFace));
        }
        [[nodiscard]] Geom::Elem::Element GetFaceElement(index iFace) const
        {
            return Geom::Elem::Element{_faceElemInfo(iFace, 0).getElemType()};
        }
        [[nodiscard]] bool FaceIsBoundary(index iFace) const
        {
            return Geom::FaceIDIsExternalBC(_faceElemInfo(iFace, 0).zone);
        }
        [[nodiscard]] const std::vector<NodeEdgeIncidence> &Node2Edge(index iNode) const
        {
            return _node2edge.at(static_cast<std::size_t>(iNode));
        }
    };
}
