#pragma once
#include "ResidentNCFV.hpp"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#if defined(OPEN_MPI)
#    include <mpi-ext.h>
#endif

namespace DNDS::GPU::ResidentDetail
{
    using I = DeviceIndex;
    struct Range
    {
        explicit Range(const char *name) { nvtxRangePushA(name); }
        ~Range() { nvtxRangePop(); }
        Range(const Range &) = delete;
    };
    inline void Check(cudaError_t status, const char *where)
    {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
    }
    inline void MPIcheck(int status)
    {
        if (status != MPI_SUCCESS)
            throw std::runtime_error("Resident GPU MPI operation failed");
    }
    template <class T>
    struct Buffer
    {
        T *data = nullptr;
        std::size_t count = 0;
        Buffer() = default;
        Buffer(const Buffer &) = delete;
        ~Buffer()
        {
            if (data)
                cudaFree(data);
        }
        void Resize(std::size_t size)
        {
            if (size == count)
                return;
            if (data)
                Check(cudaFree(data), "free resident buffer");
            data = nullptr;
            count = size;
            if (size)
                Check(cudaMalloc(reinterpret_cast<void **>(&data), size * sizeof(T)), "allocate resident buffer");
        }
        void Upload(const std::vector<T> &values)
        {
            Resize(values.size());
            if (count)
                Check(cudaMemcpy(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice), "upload static resident data");
        }
    };
    struct BufferHost
    {
        double *data = nullptr;
        BufferHost() = default;
        BufferHost(const BufferHost &) = delete;
        ~BufferHost()
        {
            if (data)
                cudaFreeHost(data);
        }
        void Resize(std::size_t size)
        {
            if (data)
                Check(cudaFreeHost(data), "free pinned halo packet");
            data = nullptr;
            if (size)
                Check(cudaMallocHost(reinterpret_cast<void **>(&data), size * sizeof(double)), "allocate pinned halo packet");
        }
    };
    __device__ inline I Thread() { return I(blockIdx.x) * blockDim.x + threadIdx.x; }
    __device__ inline void Minimum(double *address, double value)
    {
        auto *bits = reinterpret_cast<unsigned long long *>(address);
        auto old = *bits;
        while (value < __longlong_as_double(old))
        {
            const auto assumed = old;
            old = atomicCAS(bits, assumed, __double_as_longlong(value));
            if (old == assumed)
                break;
        }
    }
    __device__ inline void Maximum(double *address, double value)
    {
        auto *bits = reinterpret_cast<unsigned long long *>(address);
        auto old = *bits;
        while (value > __longlong_as_double(old))
        {
            const auto assumed = old;
            old = atomicCAS(bits, assumed, __double_as_longlong(value));
            if (old == assumed)
                break;
        }
    }
    static __global__ void Gather(const double *field, const I *indices, double *buffer, I count, int stride)
    {
        const I i = Thread();
        if (i < count * stride)
            buffer[i] = field[indices[i / stride] * stride + i % stride];
        __threadfence_system();
    }
    static __global__ void Scatter(double *field, const I *indices, const double *buffer, I count, int stride)
    {
        const I i = Thread();
        if (i < count * stride)
            field[indices[i / stride] * stride + i % stride] = buffer[i];
    }
    static __global__ void ScatterFactors(double *field, const I *indices, const double *values, I count, bool minimum)
    {
        const I i = Thread();
        if (i >= count)
            return;
        if (minimum)
            Minimum(field + indices[i], values[i]);
        else if (values[i] != 1)
        {
            auto *bits = reinterpret_cast<unsigned long long *>(field + indices[i]);
            auto old = *bits;
            for (;;)
            {
                const auto assumed = old;
                old = atomicCAS(bits, assumed, __double_as_longlong(__longlong_as_double(assumed) * values[i]));
                if (old == assumed)
                    break;
            }
        }
    }
    static __global__ void Ones(double *values, I count)
    {
        const I i = Thread();
        if (i < count)
            values[i] = 1;
    }
    struct Halo
    {
        MPI_Comm comm;
        bool active = false;
        std::vector<int> sendCounts, recvCounts, sendDisps, recvDisps;
        Buffer<I> sendIndices, recvIndices;
        Buffer<double> sendBuffer, recvBuffer;
        BufferHost hostSend, hostRecv;
        bool cudaAware = false;
        std::uint64_t bytes = 0, deviceBytes = 0;
        Halo(const std::vector<I> &globals, I owned, MPI_Comm communicator) : comm(communicator)
        {
            int ranks = 0;
            MPI_Comm_size(comm, &ranks);
            if (owned > std::numeric_limits<int>::max())
                throw std::runtime_error("GPU halo local count overflow");
            int local = int(owned);
            std::vector<int> sizes(ranks), offsets(ranks);
            MPIcheck(MPI_Allgather(&local, 1, MPI_INT, sizes.data(), 1, MPI_INT, comm));
            for (int rank = 1; rank < ranks; ++rank)
                offsets[rank] = offsets[rank - 1] + sizes[rank - 1];
            std::vector<I> all(offsets.back() + sizes.back());
            MPIcheck(MPI_Allgatherv(globals.data(), local, MPI_INT64_T, all.data(), sizes.data(), offsets.data(), MPI_INT64_T, comm));
            std::unordered_map<I, std::pair<int, I>> owners;
            for (int rank = 0; rank < ranks; ++rank)
                for (int i = 0; i < sizes[rank]; ++i)
                    owners.emplace(all[offsets[rank] + i], std::make_pair(rank, I(i)));
            std::vector<std::vector<I>> requests(ranks), destinations(ranks);
            for (I i = owned; i < I(globals.size()); ++i)
            {
                const auto owner = owners.at(globals[i]);
                requests[owner.first].push_back(owner.second);
                destinations[owner.first].push_back(i);
            }
            recvCounts.resize(ranks);
            sendCounts.resize(ranks);
            recvDisps.resize(ranks);
            sendDisps.resize(ranks);
            for (int rank = 0; rank < ranks; ++rank)
                recvCounts[rank] = int(requests[rank].size());
            MPIcheck(MPI_Alltoall(recvCounts.data(), 1, MPI_INT, sendCounts.data(), 1, MPI_INT, comm));
            for (int rank = 1; rank < ranks; ++rank)
            {
                recvDisps[rank] = recvDisps[rank - 1] + recvCounts[rank - 1];
                sendDisps[rank] = sendDisps[rank - 1] + sendCounts[rank - 1];
            }
            std::vector<I> request, receive;
            for (int rank = 0; rank < ranks; ++rank)
            {
                request.insert(request.end(), requests[rank].begin(), requests[rank].end());
                receive.insert(receive.end(), destinations[rank].begin(), destinations[rank].end());
            }
            std::vector<I> send(sendDisps.back() + sendCounts.back());
            MPIcheck(MPI_Alltoallv(request.data(), recvCounts.data(), recvDisps.data(), MPI_INT64_T,
                                   send.data(), sendCounts.data(), sendDisps.data(), MPI_INT64_T, comm));
            sendIndices.Upload(send);
            recvIndices.Upload(receive);
            const int localActive = !send.empty() || !receive.empty();
            int globalActive = 0;
            MPIcheck(MPI_Allreduce(&localActive, &globalActive, 1, MPI_INT, MPI_MAX, comm));
            active = globalActive != 0;
            if (!active)
                return;
#if defined(OPEN_MPI) && defined(MPIX_CUDA_AWARE_SUPPORT)
            cudaAware = MPIX_Query_cuda_support() != 0;
#endif
            // Allocate once, including the largest ACM cubic coefficient packet.
            // Reverse factor exchange uses the opposite adjacency, hence max(ns,nr).
            constexpr std::size_t packetAlignment = 2 * 1024 * 1024;
            const auto requiredBytes = std::max<std::size_t>(1, std::max(send.size(), receive.size()) * 76) * sizeof(double);
            const auto capacity = ((requiredBytes + packetAlignment - 1) / packetAlignment) * packetAlignment / sizeof(double);
            sendBuffer.Resize(capacity);
            recvBuffer.Resize(capacity);
            if (!cudaAware)
            {
                hostSend.Resize(capacity);
                hostRecv.Resize(capacity);
            }
            Check(cudaDeviceSynchronize(), "finish GPU halo setup");
        }
        const char *BackendName() const
        {
            if (!active)
                return "none";
            return cudaAware ? "CUDA-aware MPI" : "MPI staged GPU packets";
        }
        void Exchange(int stride, bool reverse)
        {
            const auto &sc = reverse ? recvCounts : sendCounts;
            const auto &rc = reverse ? sendCounts : recvCounts;
            const auto &sd = reverse ? recvDisps : sendDisps;
            const auto &rd = reverse ? sendDisps : recvDisps;
            const I sendSize = I(sd.back() + sc.back()) * stride;
            const I receiveSize = I(rd.back() + rc.back()) * stride;
            Check(cudaDeviceSynchronize(), "complete device halo packing");
            if (!cudaAware && sendSize)
                Check(cudaMemcpy(hostSend.data, sendBuffer.data, sendSize * sizeof(double), cudaMemcpyDeviceToHost), "stage packed MPI send");
            auto countsOut = sc, countsIn = rc, offsetsOut = sd, offsetsIn = rd;
            for (std::size_t i = 0; i < sc.size(); ++i)
            {
                countsOut[i] *= stride;
                countsIn[i] *= stride;
                offsetsOut[i] *= stride;
                offsetsIn[i] *= stride;
            }
            MPIcheck(MPI_Alltoallv(cudaAware ? sendBuffer.data : hostSend.data, countsOut.data(), offsetsOut.data(), MPI_DOUBLE,
                                   cudaAware ? recvBuffer.data : hostRecv.data, countsIn.data(), offsetsIn.data(), MPI_DOUBLE, comm));
            if (!cudaAware && receiveSize)
                Check(cudaMemcpy(recvBuffer.data, hostRecv.data, receiveSize * sizeof(double), cudaMemcpyHostToDevice), "stage packed MPI receive");
            if (!cudaAware)
                bytes += (sendSize + receiveSize) * sizeof(double);
        }
        void Pull(double *field, int stride)
        {
            Range range("MPI halo transport");
            const I ns = sendIndices.count, nr = recvIndices.count;
            if (!active)
                return;
            if (stride > 76)
                throw std::runtime_error("Resident GPU halo packet capacity exceeded");
            if (ns)
                Gather<<<(ns * stride + 127) / 128, 128>>>(field, sendIndices.data, sendBuffer.data, ns, stride);
            Exchange(stride, false);
            if (nr)
                Scatter<<<(nr * stride + 127) / 128, 128>>>(field, recvIndices.data, recvBuffer.data, nr, stride);
            deviceBytes += (ns + nr) * stride * sizeof(double);
        }
        void PushFactors(double *ownerValues, const double *localFactors, bool minimum)
        {
            Range range("MPI halo transport");
            const I ns = sendIndices.count, nr = recvIndices.count;
            if (!active)
                return;
            if (nr)
                Gather<<<(nr + 127) / 128, 128>>>(localFactors, recvIndices.data, sendBuffer.data, nr, 1);
            Exchange(1, true);
            if (ns)
                ScatterFactors<<<(ns + 127) / 128, 128>>>(ownerValues, sendIndices.data, recvBuffer.data, ns, minimum);
            deviceBytes += (ns + nr) * sizeof(double);
        }
    };
}
