#pragma once

#include "NativeGeom/Access/MeshView.hpp"

namespace DNDS::NativeCFV::Detail
{
    /// Map a shared face point into its selected parent's coordinate frame.
    inline NativeGeom::tPoint FacePointInCell(const NativeGeom::UnstructuredMeshView &mesh, index iFace, int side, const NativeGeom::tPoint &point)
    {
        const auto zone = mesh.GetFaceZone(iFace);
        if (side == 1 && NativeGeom::FaceIDIsPeriodicMain(zone))
            return mesh.Periodic().TransCoord(point, zone);
        if (side == 1 && NativeGeom::FaceIDIsPeriodicDonor(zone))
            return mesh.Periodic().TransCoordBack(point, zone);
        return point;
    }
    inline NativeGeom::tPoint OtherPointInCell(const NativeGeom::UnstructuredMeshView &mesh, index iFace, int side, const NativeGeom::tPoint &point)
    {
        const auto zone = mesh.GetFaceZone(iFace);
        if ((side == 1 && NativeGeom::FaceIDIsPeriodicMain(zone)) || (side == 0 && NativeGeom::FaceIDIsPeriodicDonor(zone)))
            return mesh.Periodic().TransCoord(point, zone);
        if ((side == 0 && NativeGeom::FaceIDIsPeriodicMain(zone)) || (side == 1 && NativeGeom::FaceIDIsPeriodicDonor(zone)))
            return mesh.Periodic().TransCoordBack(point, zone);
        return point;
    }
    /// Orthogonal map from the other parent to this parent's frame.
    inline NativeGeom::tGPoint OtherFrameInCell(const NativeGeom::UnstructuredMeshView &mesh, index iFace, int side)
    {
        const auto zone = mesh.GetFaceZone(iFace);
        if (!NativeGeom::FaceIDIsPeriodic(zone))
            return NativeGeom::tGPoint::Identity();
        const int ordinal = NativeGeom::FaceIDIsPeriodicMain(zone) ? -zone : -zone - 3;
        const NativeGeom::tGPoint rotation = mesh.Periodic().rotation.at(ordinal).map();
        if ((side == 1 && NativeGeom::FaceIDIsPeriodicMain(zone)) || (side == 0 && NativeGeom::FaceIDIsPeriodicDonor(zone)))
            return rotation;
        return rotation.transpose();
    }
    inline void ValidateVectorBlocks(index components, const std::vector<index> &starts)
    {
        index end = 0;
        for (auto first : starts)
        {
            DNDS_check_throw_info(first >= end && first <= components && components - first >= 3,
                                  "periodic vector triples must be sorted, disjoint and within the component range");
            end = first + 3;
        }
    }
}
