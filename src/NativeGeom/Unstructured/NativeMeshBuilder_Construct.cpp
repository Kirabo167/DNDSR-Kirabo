#include "NativeGeom/Unstructured/NativeMeshPreparation.hpp"
#include "NativeGeom/Connectivity/NativeMeshTopology.hpp"
#include "NativeGeom/Unstructured/UnstructuredMesh.hpp"
#include "NativeGeom/Transition/FieldTransfer.hpp"
#include "NativeGeom/IO/Metis.hpp"
#include "NativeGeom/Transition/NativeMeshPartitionGraph.hpp"
#include "NativeGeom/Unstructured/ReaderPartitionGraph.hpp"
#include "NativeGeom/Connectivity/TopologySubEntityQuery.hpp"
#include "NativeGeom/Connectivity/ConnectivityInterpolateRows.hpp"
#include "NativeGeom/Unstructured/NativeMeshBuilder.hpp"
#include "NativeGeom/Transition/NativeMeshAssembler.hpp"
#include <map>
#include <cmath>
#include <limits>
#include <set>

namespace DNDS::NativeGeom
{
    std::shared_ptr<const UnstructuredMeshState> ConstructNativeMesh(NativeMeshInput input, NativeMeshBuildOptions options)
    {
        const auto mpi = input.mpi;
        auto owned = NativeMeshBuilder::Build(std::move(input), options);
        GhostSpec recipe;
        owned.LocalStage(mpi, [&]
                         { recipe = DefaultNativeMeshGhostSpec(owned, options.cellLayers, options.completeNodeStar); });
        return NativeMeshAssembler::Build(mpi, std::move(owned), recipe);
    }
}
