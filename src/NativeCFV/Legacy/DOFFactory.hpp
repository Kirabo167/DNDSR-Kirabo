#pragma once
/// @file DOFFactory.hpp
/// @brief Free-function DOF array builders extracted from FiniteVolume.
///
/// These build cell/face/node DOF arrays with ghost communication setup.
/// FiniteVolume::BuildUDof and BuildUGradD delegate to these.

#include "NativeCFV/Legacy/VRDefines.hpp"
#include "NativeGeom/Unstructured/UnstructuredMesh.hpp"
#include "NativeGeom/Compat/LegacyFieldBridge.hpp"

namespace DNDS::NativeCFV
{
    /// @brief Build a DOF array pair matching a mesh location (cell/face/node).
    ///
    /// Creates father+son arrays, optionally sets up ghost communication
    /// by borrowing the indexing from the mesh's corresponding adjacency.
    template <int nVarsFixed = 1>
    void BuildUDofOnMesh(
        tUDof<nVarsFixed> &u,
        const std::string &name,
        const MPIInfo &mpi,
        const ssp<NativeGeom::UnstructuredMesh> &mesh,
        int nVars,
        bool buildSon = true,
        bool buildTrans = true,
        NativeGeom::MeshLoc varloc = NativeGeom::MeshLoc::Cell)
    {
        u.InitPair(name, mpi);
        DNDS_assert(varloc);
        switch (varloc)
        {
        case NativeGeom::MeshLoc::Cell:
            u.father->Resize(mesh->NumCell(), nVars, 1);
            break;
        case NativeGeom::MeshLoc::Node:
            u.father->Resize(mesh->NumNode(), nVars, 1);
            break;
        case NativeGeom::MeshLoc::Face:
            u.father->Resize(mesh->NumFace(), nVars, 1);
            break;
        default:
            DNDS_assert(false);
        }
        if (buildSon)
            switch (varloc)
            {
            case NativeGeom::MeshLoc::Cell:
                u.son->Resize(mesh->NumCellGhost(), nVars, 1);
                break;
            case NativeGeom::MeshLoc::Node:
                u.son->Resize(mesh->NumNodeGhost(), nVars, 1);
                break;
            case NativeGeom::MeshLoc::Face:
                u.son->Resize(mesh->NumFaceGhost(), nVars, 1);
                break;
            default:
                DNDS_assert(false);
            }

        if (buildTrans)
        {
            DNDS_assert(buildSon);
            u.TransAttach();
            switch (varloc)
            {
            case NativeGeom::MeshLoc::Cell:
                NativeGeom::LegacyFieldBridge::BorrowIndexing(u.trans, mesh->Entities(NativeGeom::EntityKind::Cell));
                break;
            case NativeGeom::MeshLoc::Node:
                NativeGeom::LegacyFieldBridge::BorrowIndexing(u.trans, mesh->Entities(NativeGeom::EntityKind::Node));
                break;
            case NativeGeom::MeshLoc::Face:
                NativeGeom::LegacyFieldBridge::BorrowIndexing(u.trans, mesh->Entities(NativeGeom::EntityKind::Face));
                break;
            default:
                DNDS_assert(false);
            }
            u.trans.createMPITypes();
            u.trans.initPersistentPull();
            u.trans.initPersistentPush();
        }

        for (index iCell = 0; iCell < u.Size(); iCell++)
            u[iCell].setZero();
    }

    /// @brief Build a gradient DOF array pair matching a mesh location.
    template <int nVarsFixed, int dim>
    void BuildUGradDOnMesh(
        tUGrad<nVarsFixed, dim> &u,
        const std::string &name,
        const MPIInfo &mpi,
        const ssp<NativeGeom::UnstructuredMesh> &mesh,
        int nVars,
        bool buildSon = true,
        bool buildTrans = true,
        NativeGeom::MeshLoc varloc = NativeGeom::MeshLoc::Cell)
    {
        u.InitPair(name, mpi);
        switch (varloc)
        {
        case NativeGeom::MeshLoc::Cell:
            u.father->Resize(mesh->NumCell(), dim, nVars);
            break;
        case NativeGeom::MeshLoc::Node:
            u.father->Resize(mesh->NumNode(), dim, nVars);
            break;
        case NativeGeom::MeshLoc::Face:
            u.father->Resize(mesh->NumFace(), dim, nVars);
            break;
        default:
            DNDS_assert(false);
        }
        if (buildSon)
            switch (varloc)
            {
            case NativeGeom::MeshLoc::Cell:
                u.son->Resize(mesh->NumCellGhost(), dim, nVars);
                break;
            case NativeGeom::MeshLoc::Node:
                u.son->Resize(mesh->NumNodeGhost(), dim, nVars);
                break;
            case NativeGeom::MeshLoc::Face:
                u.son->Resize(mesh->NumFaceGhost(), dim, nVars);
                break;
            default:
                DNDS_assert(false);
            }
        if (buildTrans)
        {
            DNDS_assert(buildSon);
            u.TransAttach();
            switch (varloc)
            {
            case NativeGeom::MeshLoc::Cell:
                NativeGeom::LegacyFieldBridge::BorrowIndexing(u.trans, mesh->Entities(NativeGeom::EntityKind::Cell));
                break;
            case NativeGeom::MeshLoc::Node:
                NativeGeom::LegacyFieldBridge::BorrowIndexing(u.trans, mesh->Entities(NativeGeom::EntityKind::Node));
                break;
            case NativeGeom::MeshLoc::Face:
                NativeGeom::LegacyFieldBridge::BorrowIndexing(u.trans, mesh->Entities(NativeGeom::EntityKind::Face));
                break;
            default:
                DNDS_assert(false);
            }
            u.trans.createMPITypes();
            u.trans.initPersistentPull();
            u.trans.initPersistentPush();
        }

        for (index iCell = 0; iCell < u.Size(); iCell++)
            u[iCell].setZero();
    }
}
