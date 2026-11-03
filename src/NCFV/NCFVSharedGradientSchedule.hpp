#pragma once

#include "DNDS/MPI.hpp"
#include "DNDS/Errors.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace DNDS::NCFV
{
    /**
     * @brief Distributed coloring of complete face supports, with cached sparse factor pushes.
     * @details A color contains at most one face using any physical node, including remote
     * and periodic dependencies. Coloring is an initialization operation: local greedy
     * proposals are arbitrated by node owners and only faces winning at every support node
     * are committed. No mesh-wide graph is gathered. Runtime messages use cached peer/row
     * lists on a private communicator; there are no runtime all-to-all count exchanges.
     */
    class SharedGradientSchedule
    {
        struct Peer
        {
            MPI_int rank = UnInitMPIInt;
            std::vector<index> rows;
            std::vector<real> values;
        };

        struct Group
        {
            std::vector<std::size_t> faces;
            std::vector<Peer> sends;
            std::vector<Peer> receives;
        };

        const MPIInfo &_mpi;
        MPI_Comm _comm = MPI_COMM_NULL;
        std::vector<int> _colors;
        std::vector<Group> _groups;
        int _rounds = 0;
        real _initializationSeconds = 0;

        /** @brief Initialization-only variable-length integer exchange, indexed by peer rank. */
        std::vector<std::vector<index>> Exchange(const std::vector<std::vector<index>> &out) const
        {
            std::vector<int> sendCounts(_mpi.size), receiveCounts(_mpi.size);
            std::vector<int> sendOffsets(_mpi.size + 1), receiveOffsets(_mpi.size + 1);
            for (MPI_int rank = 0; rank < _mpi.size; rank++)
            {
                DNDS_check_throw_info(out[rank].size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                                      "NCFV coloring message exceeds MPI count capacity");
                sendCounts[rank] = static_cast<int>(out[rank].size());
            }
            MPI_Alltoall(sendCounts.data(), 1, MPI_INT, receiveCounts.data(), 1, MPI_INT, _mpi.comm);
            for (MPI_int rank = 0; rank < _mpi.size; rank++)
            {
                DNDS_check_throw_info(sendCounts[rank] <= std::numeric_limits<int>::max() - sendOffsets[rank] &&
                                          receiveCounts[rank] <= std::numeric_limits<int>::max() - receiveOffsets[rank],
                                      "NCFV coloring exchange exceeds MPI displacement capacity");
                sendOffsets[rank + 1] = sendOffsets[rank] + sendCounts[rank];
                receiveOffsets[rank + 1] = receiveOffsets[rank] + receiveCounts[rank];
            }
            std::vector<index> send(sendOffsets.back()), receive(receiveOffsets.back());
            for (MPI_int rank = 0; rank < _mpi.size; rank++)
                std::copy(out[rank].begin(), out[rank].end(), send.begin() + sendOffsets[rank]);
            MPI_Alltoallv(send.data(), sendCounts.data(), sendOffsets.data(), DNDS_MPI_INDEX,
                          receive.data(), receiveCounts.data(), receiveOffsets.data(), DNDS_MPI_INDEX, _mpi.comm);
            std::vector<std::vector<index>> in(_mpi.size);
            for (MPI_int rank = 0; rank < _mpi.size; rank++)
                in[rank].assign(receive.begin() + receiveOffsets[rank], receive.begin() + receiveOffsets[rank + 1]);
            return in;
        }

        /** @brief Bijective mixing avoids a spatially ordered chain of priority conflicts. */
        static std::uint64_t Priority(index face)
        {
            auto value = static_cast<std::uint64_t>(face) + UINT64_C(0x9e3779b97f4a7c15);
            value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
            value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
            return value ^ (value >> 31);
        }

    public:
        explicit SharedGradientSchedule(const MPIInfo &mpi) : _mpi(mpi) {}
        SharedGradientSchedule(const SharedGradientSchedule &) = delete;
        SharedGradientSchedule &operator=(const SharedGradientSchedule &) = delete;

        ~SharedGradientSchedule()
        {
            if (_comm != MPI_COMM_NULL)
            {
                int finalized = 0;
                MPI_Finalized(&finalized);
                if (!finalized)
                    MPI_Comm_free(&_comm);
            }
        }

        /**
         * @brief Color local owned faces using complete supports in one extended node layout.
         * @param owned Number of authoritative node rows; ghosts immediately follow these rows.
         * @param owners Owner rank for each appended ghost.
         * @param ownerRows Authoritative local row on that ghost's owner.
         * @param supports Unique extended node rows read or modified by each face.
         * @note Collective. Periodic aliases must already have been collapsed by the node halo.
         */
        void Initialize(index owned, const std::vector<MPI_int> &owners,
                        const std::vector<index> &ownerRows,
                        const std::vector<std::vector<index>> &supports)
        {
            DNDS_check_throw_info(_comm == MPI_COMM_NULL && owners.size() == ownerRows.size(),
                                  "NCFV shared-gradient schedule initialized twice or with inconsistent ghosts");
            const real started = MPI_Wtime();
            MPI_Comm_dup(_mpi.comm, &_comm);
            const index size = owned + static_cast<index>(owners.size());
            std::vector<std::vector<std::pair<MPI_int, index>>> subscribers(owned);
            std::vector<std::vector<index>> out(_mpi.size);
            for (std::size_t ghost = 0; ghost < owners.size(); ghost++)
            {
                out[owners[ghost]].push_back(ownerRows[ghost]);
                out[owners[ghost]].push_back(owned + static_cast<index>(ghost));
            }
            auto in = Exchange(out);
            for (MPI_int rank = 0; rank < _mpi.size; rank++)
                for (std::size_t i = 0; i < in[rank].size(); i += 2)
                {
                    const index row = in[rank][i];
                    DNDS_check_throw_info(row >= 0 && row < owned, "NCFV coloring has an invalid owner row");
                    subscribers[row].emplace_back(rank, in[rank][i + 1]);
                }
            index count = static_cast<index>(supports.size()), offset = 0;
            MPI_Exscan(&count, &offset, 1, DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
            if (_mpi.rank == 0)
                offset = 0;
            _colors.assign(supports.size(), -1);
            std::vector<std::size_t> order(supports.size());
            for (std::size_t face = 0; face < supports.size(); face++)
            {
                order[face] = face;
                for (index row : supports[face])
                    DNDS_check_throw_info(row >= 0 && row < size, "NCFV coloring support row is out of range");
                DNDS_check_throw_info(std::is_sorted(supports[face].begin(), supports[face].end()) &&
                                          std::adjacent_find(supports[face].begin(), supports[face].end()) == supports[face].end(),
                                      "NCFV coloring supports must be sorted and unique");
            }
            std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right)
                      { return Priority(offset + static_cast<index>(left)) < Priority(offset + static_cast<index>(right)); });
            std::vector<std::set<int>> used(size);
            index remaining = 0;
            MPI_Allreduce(&count, &remaining, 1, DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
            while (remaining)
            {
                _rounds++;
                auto unavailable = used;
                std::vector<int> proposed(supports.size(), -1);
                std::vector<std::map<int, index>> winners(owned);
                const auto claim = [&](index row, int color, index face)
                {
                    auto inserted = winners[row].emplace(color, face);
                    if (!inserted.second && Priority(face) < Priority(inserted.first->second))
                        inserted.first->second = face;
                };
                out.assign(_mpi.size, {});
                for (std::size_t face : order)
                    if (_colors[face] < 0)
                    {
                        std::set<int> forbidden;
                        for (index row : supports[face])
                            forbidden.insert(unavailable[row].begin(), unavailable[row].end());
                        int color = 0;
                        while (forbidden.count(color))
                            color++;
                        proposed[face] = color;
                        for (index row : supports[face])
                        {
                            unavailable[row].insert(color);
                            if (row < owned)
                                claim(row, color, offset + static_cast<index>(face));
                            else
                            {
                                auto &message = out[owners[row - owned]];
                                message.insert(message.end(), {ownerRows[row - owned], color,
                                                               offset + static_cast<index>(face), static_cast<index>(face)});
                            }
                        }
                    }
                in = Exchange(out);
                for (const auto &message : in)
                    for (std::size_t i = 0; i < message.size(); i += 4)
                        claim(message[i], static_cast<int>(message[i + 1]), message[i + 2]);
                out.assign(_mpi.size, {});
                for (MPI_int rank = 0; rank < _mpi.size; rank++)
                    for (std::size_t i = 0; i < in[rank].size(); i += 4)
                        if (winners[in[rank][i]].at(static_cast<int>(in[rank][i + 1])) != in[rank][i + 2])
                            out[rank].push_back(in[rank][i + 3]);
                const auto rejected = Exchange(out);
                std::vector<bool> accepted(supports.size(), true);
                for (const auto &message : rejected)
                    for (index face : message)
                        accepted[face] = false;
                for (std::size_t face : order)
                    if (proposed[face] >= 0)
                        for (index row : supports[face])
                            if (row < owned && winners[row].at(proposed[face]) != offset + static_cast<index>(face))
                                accepted[face] = false;
                std::vector<std::pair<index, int>> updates;
                const auto commit = [&](index row, int color)
                {
                    DNDS_check_throw_info(used[row].insert(color).second,
                                          "NCFV coloring conflict: two same-color faces share a full-support node");
                    updates.emplace_back(row, color);
                };
                index progress[2]{};
                out.assign(_mpi.size, {});
                for (std::size_t face : order)
                    if (proposed[face] >= 0)
                    {
                        if (!accepted[face])
                        {
                            progress[0]++;
                            continue;
                        }
                        _colors[face] = proposed[face];
                        progress[1]++;
                        for (index row : supports[face])
                            if (row < owned)
                                commit(row, proposed[face]);
                            else
                            {
                                auto &message = out[owners[row - owned]];
                                message.insert(message.end(), {ownerRows[row - owned], proposed[face]});
                            }
                    }
                in = Exchange(out);
                for (const auto &message : in)
                    for (std::size_t i = 0; i < message.size(); i += 2)
                        commit(message[i], static_cast<int>(message[i + 1]));
                out.assign(_mpi.size, {});
                for (const auto &[row, color] : updates)
                    for (const auto &[rank, remoteRow] : subscribers[row])
                    {
                        out[rank].push_back(remoteRow);
                        out[rank].push_back(color);
                    }
                in = Exchange(out);
                for (const auto &message : in)
                    for (std::size_t i = 0; i < message.size(); i += 2)
                        DNDS_check_throw_info(used[message[i]].insert(static_cast<int>(message[i + 1])).second,
                                              "NCFV coloring received a duplicate committed node/color");
                index globalProgress[2]{};
                MPI_Allreduce(progress, globalProgress, 2, DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
                DNDS_check_throw_info(globalProgress[1] > 0, "NCFV distributed coloring made no progress");
                remaining = globalProgress[0];
            }
            int localColors = 0, colorCount = 0;
            for (int color : _colors)
                localColors = std::max(localColors, color + 1);
            MPI_Allreduce(&localColors, &colorCount, 1, MPI_INT, MPI_MAX, _mpi.comm);
            std::vector<int> localUsed(colorCount), globalUsed(colorCount), compressed(colorCount);
            for (int color : _colors)
                localUsed[color] = 1;
            MPI_Allreduce(localUsed.data(), globalUsed.data(), colorCount, MPI_INT, MPI_MAX, _mpi.comm);
            int denseCount = 0;
            for (int color = 0; color < colorCount; color++)
                if (globalUsed[color])
                    compressed[color] = denseCount++;
            _groups.resize(denseCount);
            for (std::size_t face = 0; face < _colors.size(); face++)
            {
                _colors[face] = compressed[_colors[face]];
                _groups[_colors[face]].faces.push_back(face);
            }
            for (auto &group : _groups)
            {
                std::vector<std::vector<index>> ghosts(_mpi.size);
                for (std::size_t face : group.faces)
                    for (index row : supports[face])
                        if (row >= owned)
                            ghosts[owners[row - owned]].push_back(row - owned);
                out.assign(_mpi.size, {});
                for (MPI_int rank = 0; rank < _mpi.size; rank++)
                    if (!ghosts[rank].empty())
                    {
                        // Each node occurs once in the entire global color, not once per rank.
                        DNDS_check_throw_info(std::set<index>(ghosts[rank].begin(), ghosts[rank].end()).size() == ghosts[rank].size(),
                                              "NCFV color contains duplicate local support nodes");
                        for (index ghost : ghosts[rank])
                            out[rank].push_back(ownerRows[ghost]);
                        group.sends.push_back({rank, ghosts[rank], std::vector<real>(ghosts[rank].size())});
                    }
                in = Exchange(out);
                for (MPI_int rank = 0; rank < _mpi.size; rank++)
                    if (!in[rank].empty())
                        group.receives.push_back({rank, in[rank], std::vector<real>(in[rank].size())});
            }
            _initializationSeconds = MPI_Wtime() - started;
        }

        /** @brief Cache a single concurrent ghost-to-owner reduction, without face coloring. */
        void InitializeReduction(const std::vector<MPI_int> &owners, const std::vector<index> &ownerRows)
        {
            DNDS_check_throw_info(_comm == MPI_COMM_NULL && owners.size() == ownerRows.size(),
                                  "NCFV node reduction initialized twice or with inconsistent ghosts");
            MPI_Comm_dup(_mpi.comm, &_comm);
            _groups.resize(1);
            std::vector<std::vector<index>> rows(_mpi.size), ghosts(_mpi.size);
            for (std::size_t ghost = 0; ghost < owners.size(); ghost++)
            {
                rows[owners[ghost]].push_back(ownerRows[ghost]);
                ghosts[owners[ghost]].push_back(static_cast<index>(ghost));
            }
            const auto incoming = Exchange(rows);
            for (MPI_int rank = 0; rank < _mpi.size; rank++)
            {
                if (!ghosts[rank].empty())
                    _groups[0].sends.push_back({rank, ghosts[rank], std::vector<real>(ghosts[rank].size())});
                if (!incoming[rank].empty())
                    _groups[0].receives.push_back({rank, incoming[rank], std::vector<real>(incoming[rank].size())});
            }
        }

        /** @brief Apply one group's remote factors using the supplied owner reduction. */
        template <class ApplyFactor>
        void ExchangeFactors(int color, const std::vector<real> &ghostFactors, ApplyFactor apply)
        {
            auto &group = _groups.at(color);
            std::vector<MPI_Request> requests;
            requests.reserve(group.sends.size() + group.receives.size());
            for (auto &peer : group.receives)
            {
                requests.push_back(MPI_REQUEST_NULL);
                MPI_Irecv(peer.values.data(), static_cast<int>(peer.values.size()), DNDS_MPI_REAL,
                          peer.rank, 0, _comm, &requests.back());
            }
            for (auto &peer : group.sends)
            {
                for (std::size_t i = 0; i < peer.rows.size(); i++)
                    peer.values[i] = ghostFactors[peer.rows[i]];
                requests.push_back(MPI_REQUEST_NULL);
                MPI_Isend(peer.values.data(), static_cast<int>(peer.values.size()), DNDS_MPI_REAL,
                          peer.rank, 0, _comm, &requests.back());
            }
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            for (const auto &peer : group.receives)
                for (std::size_t i = 0; i < peer.rows.size(); i++)
                    apply(peer.rows[i], peer.values[i]);
        }

        [[nodiscard]] int ColorCount() const { return static_cast<int>(_groups.size()); }
        [[nodiscard]] int ColoringRounds() const { return _rounds; }
        [[nodiscard]] real InitializationSeconds() const { return _initializationSeconds; }
        [[nodiscard]] int FaceColor(std::size_t face) const { return _colors.at(face); }
        [[nodiscard]] const std::vector<std::size_t> &Faces(int color) const { return _groups.at(color).faces; }
    };
}
