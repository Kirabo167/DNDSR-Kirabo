#include "NativeGeom/Connectivity/NativeMeshTopology.hpp"
#include "NativeGeom/Unstructured/NativeMeshPreparation.hpp"
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
    std::shared_ptr<const UnstructuredMeshState> PrepareNativeMeshTopology(
        const UnstructuredMeshState &source, const MPIInfo &mpi,
        bool faces, bool edges, int ghostLayers, const std::map<EntityKind, index> &generatedIdStarts)
    {
        const auto communication = source.PreparationCommunication();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !communication || communication->ChecksCollectivePreflight());
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { DNDS_check_throw_info(ghostLayers > 0, "topology finalization requires positive default cell layers"); });
        auto owned = NativeMeshTopology::Build(NativeMeshAssembler::CaptureOwned(source, mpi), mpi, faces, edges, generatedIdStarts);
        GhostSpec recipe;
        owned.LocalStage(mpi, [&]
                         { recipe = ExtendNativeMeshGhostSpecForTopology(owned, faces, edges, ghostLayers); });
        owned.ghostSpec = std::move(recipe);
        auto result = AssembleNativeMesh(mpi, std::move(owned), ghostLayers);
        if (!result->Requirements().faces)
            return result;
        std::vector<index> partStarts;
        for (int part = 0; part < source.NLocalParts(); ++part)
            partStarts.push_back(source.LocalPartStart(part));
        partStarts.push_back(source.NumOwned(EntityKind::Cell));
        return result->WithLocalPartitions(std::move(partStarts));
    }
}
