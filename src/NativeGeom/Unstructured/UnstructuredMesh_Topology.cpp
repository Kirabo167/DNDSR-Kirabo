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
    void UnstructuredMesh::PrepareNativeTopology(bool faces, bool edges, int ghostLayers)
    {
        const auto communication = PreparationCommunication();
        const bool checks = !communication || communication->ChecksCollectivePreflight();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, checks);
        CheckCollectiveMutationAvailable(checks);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { DNDS_check_throw_info(_nativeState, "native topology preparation requires a native mesh"); });
        const auto stamp = versions.Stamp();
        auto result = NativeMeshPreparation::PrepareTopology(*_nativeState, mpi, faces, edges, ghostLayers);
        std::shared_ptr<UnstructuredMeshState> prepared;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            stamp.Validate();
            prepared = std::shared_ptr<UnstructuredMeshState>(new UnstructuredMeshState(*result)); });
        MeshMutationToken mutation(this, versions, Runtime::Change::Topology | Runtime::Change::Layout | Runtime::Change::Distribution | Runtime::Change::Allocation);
        mutation.Commit();
        prepared->BindPreparedEpoch(versions);
        _nativeState = std::move(prepared);
        _nativeSmoothing.reset();
        node2bndNode.clear();
        node2parentNode.clear();
        cell2parentCell.clear();
    }
}
