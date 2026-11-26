#pragma once

// Included after NCFVSolver.hpp in the case-local 3-D solver.
#include <numeric>

namespace DNDS::NCFV
{
    class PeriodicLayer
    {
        using State = Eigen::Vector<real, 5>;
        using Field = std::vector<State>;
        MPIInfo _mpi;
        int _copies;
        MPI_Datatype _stateType;
        std::vector<int> _sendCounts, _recvCounts, _sendOffsets, _recvOffsets;
        std::vector<index> _sendNode, _recvPhysical;
        std::vector<int> _imageSlot;
        std::vector<real> _weight;
        Field _send, _recv;

        static std::vector<int> Offsets(const std::vector<int> &counts)
        {
            std::vector<int> offsets(counts.size() + 1, 0);
            std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);
            return offsets;
        }

    public:
        std::vector<index> physicalIds, masterNodes;
        Field means;

        PeriodicLayer(MPIInfo mpi, Geom::UnstructuredMesh &mesh, const DualGeometry &geometry,
                      const NodeStatePair &initial, int side, int copies, real depth)
            : _mpi(mpi), _copies(copies), _sendCounts(mpi.size), _recvCounts(mpi.size)
        {
            static_assert(sizeof(State) == 5 * sizeof(real));
            const index count = index(side) * side;
            DNDS_check_throw_info(mesh.NumNodeGlobal() == count * copies, "Incorrect image workspace size");
            std::vector<int> owner(count, -1);
            std::vector<index> ownerLocal(count, -1);
            for (index i = 0; i < mesh.NumNode(); ++i)
            {
                const index id = mesh.node2nodeOrig(i, 0);
                DNDS_check_throw_info(id >= 0 && id < count * copies, "Unexpected periodic representative ID");
                if (id >= count)
                    continue;
                owner[id] = mpi.rank;
                ownerLocal[id] = masterNodes.size();
                masterNodes.push_back(i);
                physicalIds.push_back(id);
                means.emplace_back(initial[i]);
            }
            MPI_Allreduce(MPI_IN_PLACE, owner.data(), count, MPI_INT, MPI_MAX, mpi.comm);
            for (index i = 0; i < mesh.NumNode(); ++i)
            {
                const auto rank = owner[mesh.node2nodeOrig(i, 0) % count];
                DNDS_check_throw_info(rank >= 0, "A periodic column has no physical owner");
                ++_sendCounts[rank];
            }
            MPI_Alltoall(_sendCounts.data(), 1, MPI_INT, _recvCounts.data(), 1, MPI_INT, mpi.comm);
            _sendOffsets = Offsets(_sendCounts);
            _recvOffsets = Offsets(_recvCounts);
            _sendNode.resize(_sendOffsets.back());
            _recvPhysical.resize(_recvOffsets.back());
            auto cursor = _sendOffsets;
            std::vector<index> sendIds(_sendNode.size()), recvIds(_recvPhysical.size());
            std::vector<real> sendVolume(_sendNode.size());
            _weight.resize(_recvPhysical.size());
            for (index i = 0; i < mesh.NumNode(); ++i)
            {
                const index id = mesh.node2nodeOrig(i, 0);
                const int slot = cursor[owner[id % count]]++;
                _sendNode[slot] = i;
                sendIds[slot] = id;
                sendVolume[slot] = geometry.NodeVolume(i).moments.measure;
            }
            MPI_Alltoallv(sendIds.data(), _sendCounts.data(), _sendOffsets.data(), DNDS_MPI_INDEX,
                          recvIds.data(), _recvCounts.data(), _recvOffsets.data(), DNDS_MPI_INDEX, mpi.comm);
            MPI_Alltoallv(sendVolume.data(), _sendCounts.data(), _sendOffsets.data(), DNDS_MPI_REAL,
                          _weight.data(), _recvCounts.data(), _recvOffsets.data(), DNDS_MPI_REAL, mpi.comm);
            _imageSlot.assign(means.size() * copies, -1);
            for (int k = 0; k < int(recvIds.size()); ++k)
            {
                const index id = recvIds[k], p = ownerLocal[id % count];
                DNDS_check_throw_info(p >= 0, "Periodic image routed to incorrect rank");
                _recvPhysical[k] = p;
                auto &slot = _imageSlot[p * copies + id / count];
                DNDS_check_throw_info(slot == -1, "Duplicate periodic image");
                slot = k;
            }
            real maxVolumeError = 0;
            for (std::size_t p = 0; p < means.size(); ++p)
            {
                const index x = physicalIds[p] % side, y = physicalIds[p] / side;
                const real expected = depth / ((side - 1.) * (side - 1.)) *
                    ((x == 0 || x == side - 1) ? .5 : 1.) * ((y == 0 || y == side - 1) ? .5 : 1.);
                real total = 0;
                for (int z = 0; z < copies; ++z)
                {
                    int k = _imageSlot[p * copies + z];
                    DNDS_check_throw_info(k >= 0, "Missing periodic image");
                    maxVolumeError = std::max(maxVolumeError, std::abs(_weight[k] / expected - 1));
                    total += _weight[k];
                }
                for (int z = 0; z < copies; ++z)
                    _weight[_imageSlot[p * copies + z]] /= total;
            }
            MPI_Allreduce(MPI_IN_PLACE, &maxVolumeError, 1, DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(maxVolumeError < 1e-10, "Periodic image dual-volume check failed");
            MPI_Type_contiguous(5, DNDS_MPI_REAL, &_stateType);
            MPI_Type_commit(&_stateType);
            _send.resize(sendIds.size());
            _recv.resize(recvIds.size());
            if (mpi.rank == 0)
                log() << "Periodic single layer: physical nodes=" << count << ", physical cells="
                      << (side-1)*(side-1) << ", physical Lz=" << depth << ", image periods=" << copies
                      << ", maximum relative dual volume error=" << maxVolumeError << std::endl;
        }

        ~PeriodicLayer() { MPI_Type_free(&_stateType); }

        void Lift(const Field &physical, NodeStatePair &workspace)
        {
            for (std::size_t k = 0; k < _recv.size(); ++k)
                _recv[k] = physical[_recvPhysical[k]];
            MPI_Alltoallv(_recv.data(), _recvCounts.data(), _recvOffsets.data(), _stateType,
                          _send.data(), _sendCounts.data(), _sendOffsets.data(), _stateType, _mpi.comm);
            for (std::size_t k = 0; k < _send.size(); ++k)
                workspace[_sendNode[k]] = _send[k];
        }

        void FoldResidual(const NodeStatePair &imageRhs, Field &physicalRhs)
        {
            for (std::size_t k = 0; k < _send.size(); ++k)
                _send[k] = imageRhs[_sendNode[k]];
            MPI_Alltoallv(_send.data(), _sendCounts.data(), _sendOffsets.data(), _stateType,
                          _recv.data(), _recvCounts.data(), _recvOffsets.data(), _stateType, _mpi.comm);
            for (std::size_t p = 0; p < means.size(); ++p)
            {
                State sum = State::Zero();
                for (int z = 0; z < _copies; ++z)
                {
                    const int slot = _imageSlot[p * _copies + z];
                    sum += _weight[slot] * _recv[slot];
                }
                physicalRhs[p] = sum;
            }
        }
    };
}
