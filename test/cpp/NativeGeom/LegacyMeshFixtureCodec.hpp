#pragma once
// TEST ONLY: one-way historical primary wire decoding into Runtime ownership.
// Derived groups may be reconstructed natively; no legacy mesh exists here.
#include "NativeGeom/Transition/MeshTransition.hpp"
#include "NativeGeom/Compat/NativeMeshLegacyArchive.hpp"
#include "NativeGeom/Unstructured/NativeMeshPreparation.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
namespace DNDS::NativeGeom::TestFixtureCodec
{
    inline std::shared_ptr<UnstructuredMesh> DecodePrimaryReference(UnstructuredMesh &source)
    {
        auto serializer = std::make_shared<Serializer::SerializerJSON>();
        serializer->CreatePath("/");
        WriteCurrentPrimaryMeshSnapshot(source, serializer, "fixture", 1, CurrentMeshCheckpointPolicy::DropAuxiliary);
        auto state = NativeMeshLegacyArchive::Read(source.mpi, serializer, "fixture");
        const auto groups = source.CurrentNativeRequirements();
        if (groups.faces || groups.edges)
            state = NativeMeshPreparation::PrepareTopology(*state, source.mpi, groups.faces, groups.edges);
        return UnstructuredMesh::FromNative(source.mpi, std::move(state));
    }
}
