#pragma once

#include "NCFVPressureLimiter.hpp"
#include "NCFVReconstruction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace DNDS::NCFV
{
    /**
     * One face pass in rank order, then canonical face order within each rank.
     *
     * Each event reads the current shared gradients and immediately multiplies
     * the union of both face/recovery supports by one common factor. A rank
     * block is executed on its owner only. Sparse remote factor products are
     * applied by node owners before the next block starts. Consequently this
     * is a serial face ordering, not a simultaneous product of stale factors.
     * Changing the partition or rank count can change that ordering and result.
     * One pass does not guarantee that later events preserve earlier bounds.
     */
    template <int dimension>
    class SharedGradientFaceSweep
    {
        static constexpr int NVars = dimension + 2;
        using State = Eigen::Vector<real, NVars>;

        /** @brief One source-gradient contribution to a face-side mean-minus-conservative-mean trace. */
        struct Entry
        {
            index node = UnInitIndex; ///< Row in this sweep's extended dependency halo, not the solver halo.
            Vector3 weight = Vector3::Zero(); ///< Combined normalized interface/recovery weight of this source.
        };

        /**
         * @brief One owned interface event in the prescribed sequential sweep order.
         * @details Face supports combine both sides' point-recovery and interface-integration
         * dependencies. A face factor is applied once per unique support node, then remains
         * visible to later face events through the shared accumulated node scales.
         */
        struct Face
        {
            std::array<index, 2> anchors{}; ///< Left/right anchor rows in the sweep's extended halo.
            std::array<std::vector<Entry>, 2> entries; ///< Folded gradient weights for each anchor's trace.
            std::vector<index> support; ///< Unique extended-halo rows receiving this event's common factor.
            std::array<index, 5> key{}; ///< Original endpoint IDs plus three quantized periodic offsets for sorting.
        };

    public:
        /**
         * @brief Counters and timings from the last completed shared-gradient face pass.
         * @details After Apply(), face counts and minimumFaceFactor have been reduced over
         * MPI. Timings remain local wall-clock seconds; rankBlocks is the communicator size.
         */
        struct Diagnostics
        {
            index updatedFaces = 0; ///< Global number of face events with a factor smaller than one.
            index zeroFactorFaces = 0; ///< Global number of events that set their common face factor to zero.
            real minimumFaceFactor = 1; ///< Smallest event factor over all ranks; one if no event limits.
            int rankBlocks = 0; ///< Number of sequential MPI rank blocks processed by the pass.
            real inputHaloSeconds = 0; ///< Pulling raw gradients and fixed anchor bounds before the sweep.
            real localSweepSeconds = 0; ///< Local time in ordered face loops, without factor communication.
            real blockCommunicationSeconds = 0; ///< Exchanging remote factor products and refreshing node scales.
            real totalSeconds = 0; ///< Inclusive local wall time for Apply(), including final exchanges/reductions.
        };

    private:
        const MPIInfo &_mpi;
        ssp<Geom::UnstructuredMesh> _mesh;
        const Topology &_topology;
        const DualGeometry &_geometry;
        const NodeHalo &_nodeHalo;
        NodeMatrixPair _rawGradients; ///< Frozen dimension-by-NVars source gradients in the extended halo.
        NodeStatePair _scales; ///< One cumulative multiplicative gradient factor per extended-halo node.
        // Columns: mean, lower bound, upper bound. Final row holds pressure.
        NodeMatrixPair _anchorData; ///< (NVars+1)-by-3 mean/lower/upper matrix; final row stores pressure bounds.
        std::vector<Face> _faces;
        std::vector<MPI_int> _ghostOwnerRanks;
        std::vector<index> _ghostOwnerLocals;
        Diagnostics _lastDiagnostics;
        bool _initialized = false;

        /**
         * @brief Sort a node dependency list and remove duplicates in place.
         * @param [in,out] values Global node IDs to canonicalize into ascending unique order.
         */
        static void SortUnique(std::vector<index> &values)
        {
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        }

        /**
         * @brief Resolve a global dependency ID in a particular owner/ghost field layout.
         * @param pair Array pair with an initialized local/ghost mapping.
         * @param global Global node ID that must be present in pair.
         * @return Matching owned or appended ghost row index.
         * @tparam Pair Array-pair type exposing a transformer mapping.
         * @throws std::runtime_error If the dependency is absent.
         */
        template <class Pair>
        static index FindLocal(const Pair &pair, index global)
        {
            MPI_int ownerRank = UnInitMPIInt;
            index local = UnInitIndex;
            const bool found = pair.trans.pLGhostMapping->search_indexAppend(
                global, ownerRank, local);
            DNDS_check_throw_info(found && local >= 0,
                                  "NCFV shared-gradient sweep is missing a node dependency");
            return local;
        }

        /**
         * @brief Recover ideal-gas pressure for a conservative limiter state.
         * @param state Conservative state [rho,rho*u...,rho*E] with positive density.
         * @param gamma Ratio of specific heats, strictly greater than one.
         * @return Pressure after subtracting kinetic energy from total energy density.
         */
        real Pressure(const State &state, real gamma) const
        {
            return (gamma - 1) *
                   (state(NVars - 1) - .5 * state.template segment<dimension>(1).squaredNorm() / state(0));
        }

        /**
         * @brief Check finiteness and the strict density/pressure floors used by this sweep.
         * @param state Candidate conservative state.
         * @param gamma Ideal-gas ratio of specific heats.
         * @return True when all entries are finite and density/pressure both exceed 1e-12.
         */
        bool IsPhysical(const State &state, real gamma) const
        {
            return state.allFinite() && state(0) > 1e-12 && Pressure(state, gamma) > 1e-12;
        }

        /**
         * @brief Restrict one mean-to-face increment using cached local bounds and physical-state constraints.
         * @param anchor Anchor row in this sweep's extended dependency layout.
         * @param increment Conservative face-mean increment relative to the cached anchor mean.
         * @param gamma Ideal-gas ratio of specific heats, greater than one.
         * @param limitPressure Additionally enforce cached pressure extrema when true; positivity is always
         *     enforced.
         * @return Common scalar factor in [0,1] allowed by all conservative components and enabled pressure
         *     constraints.
         */
        real TraceFactor(index anchor, const State &increment,
                         real gamma, bool limitPressure) const
        {
            const auto data = _anchorData[anchor];
            const State mean = data.template block<NVars, 1>(0, 0);
            const State candidate = mean + increment;
            DNDS_check_throw_info(IsPhysical(mean, gamma),
                                  "NCFV shared-gradient sweep received a nonphysical mean");
            DNDS_check_throw_info(increment.allFinite(),
                                  "NCFV shared-gradient sweep received a nonfinite face increment");
            constexpr real tolerance = 1e-10;
            real factor = 1;
            for (int variable = 0; variable < NVars; variable++)
            {
                if (candidate(variable) > data(variable, 2) + tolerance &&
                    increment(variable) > tolerance)
                    factor = std::min(factor,
                                      (data(variable, 2) - mean(variable)) / increment(variable));
                if (candidate(variable) < data(variable, 1) - tolerance &&
                    increment(variable) < -tolerance)
                    factor = std::min(factor,
                                      (data(variable, 1) - mean(variable)) / increment(variable));
            }
            factor = std::clamp(factor, real(0), real(1));
            // Positivity is always active, independently of pressure extrema.
            if (!IsPhysical(mean + factor * increment, gamma))
            {
                real lower = 0, upper = factor;
                for (int iteration = 0; iteration < 48; iteration++)
                {
                    const real middle = .5 * (lower + upper);
                    if (IsPhysical(mean + middle * increment, gamma))
                        lower = middle;
                    else
                        upper = middle;
                }
                factor = .95 * lower;
            }
            if (limitPressure)
                factor = LimitIdealGasPressureFactor<dimension>(
                    mean, candidate, gamma, data(NVars, 1), data(NVars, 2), factor);
            return factor;
        }

        /**
         * @brief Contract current shared-gradient scales with precombined recovery/interface weights.
         * @param entries Sparse support entries in the sweep's extended node layout; each stores a vector gradient
         *     weight.
         * @return Conservative face-mean correction using the most recently accumulated shared node scales.
         */
        State TraceIncrement(const std::vector<Entry> &entries) const
        {
            State increment = State::Zero();
            for (const auto &entry : entries)
                increment.noalias() += _scales[entry.node](0) *
                                       (_rawGradients[entry.node].transpose() * entry.weight.template head<dimension>());
            return increment;
        }

        /**
         * @brief Send one active rank's accumulated ghost-node factors to their owners and multiply owner scales.
         * @param activeRank Rank whose face block has just completed; every rank participates in the exchange.
         * @param remoteFactors One multiplicative factor per extended ghost node on the active rank; entries equal
         *     to one send no update.
         * @note Collective; transfers sparse owner-local node IDs and matching factor products.
         */
        void ExchangeRemoteFactors(MPI_int activeRank, const std::vector<real> &remoteFactors)
        {
            std::vector<int> sendCounts(static_cast<std::size_t>(_mpi.size), 0);
            std::vector<int> receiveCounts(static_cast<std::size_t>(_mpi.size), 0);
            if (_mpi.rank == activeRank)
                for (std::size_t ghost = 0; ghost < remoteFactors.size(); ghost++)
                    if (remoteFactors[ghost] < 1)
                        sendCounts[static_cast<std::size_t>(_ghostOwnerRanks[ghost])]++;
            MPI_Alltoall(sendCounts.data(), 1, MPI_INT,
                         receiveCounts.data(), 1, MPI_INT, _mpi.comm);
            std::vector<int> sendOffsets(static_cast<std::size_t>(_mpi.size + 1), 0);
            std::vector<int> receiveOffsets(static_cast<std::size_t>(_mpi.size + 1), 0);
            for (MPI_int rank = 0; rank < _mpi.size; rank++)
            {
                const auto i = static_cast<std::size_t>(rank);
                DNDS_check_throw_info(sendCounts[i] <= std::numeric_limits<int>::max() - sendOffsets[i] &&
                                          receiveCounts[i] <= std::numeric_limits<int>::max() - receiveOffsets[i],
                                      "NCFV shared-gradient factor exchange exceeds MPI count capacity");
                sendOffsets[i + 1] = sendOffsets[i] + sendCounts[i];
                receiveOffsets[i + 1] = receiveOffsets[i] + receiveCounts[i];
            }
            std::vector<index> sendNodes(static_cast<std::size_t>(sendOffsets.back()));
            std::vector<real> sendFactors(static_cast<std::size_t>(sendOffsets.back()));
            std::vector<int> cursors = sendOffsets;
            if (_mpi.rank == activeRank)
                for (std::size_t ghost = 0; ghost < remoteFactors.size(); ghost++)
                    if (remoteFactors[ghost] < 1)
                    {
                        const int position = cursors[static_cast<std::size_t>(_ghostOwnerRanks[ghost])]++;
                        sendNodes[static_cast<std::size_t>(position)] = _ghostOwnerLocals[ghost];
                        sendFactors[static_cast<std::size_t>(position)] = remoteFactors[ghost];
                    }
            std::vector<index> receiveNodes(static_cast<std::size_t>(receiveOffsets.back()));
            std::vector<real> receiveFactors(static_cast<std::size_t>(receiveOffsets.back()));
            MPI_Alltoallv(sendNodes.data(), sendCounts.data(), sendOffsets.data(), DNDS_MPI_INDEX,
                          receiveNodes.data(), receiveCounts.data(), receiveOffsets.data(), DNDS_MPI_INDEX,
                          _mpi.comm);
            MPI_Alltoallv(sendFactors.data(), sendCounts.data(), sendOffsets.data(), DNDS_MPI_REAL,
                          receiveFactors.data(), receiveCounts.data(), receiveOffsets.data(), DNDS_MPI_REAL,
                          _mpi.comm);
            for (std::size_t i = 0; i < receiveNodes.size(); i++)
            {
                DNDS_check_throw_info(receiveNodes[i] >= 0 && receiveNodes[i] < _mesh->NumNode(),
                                      "NCFV shared-gradient factor exchange has an invalid owner row");
                _scales[receiveNodes[i]](0) *= receiveFactors[i];
            }
        }

    public:
        /**
         * @brief Bind a sequential shared-gradient face limiter to the finalized NCFV data.
         * @param mpi MPI communicator metadata; the referenced object must outlive the sweep.
         * @param mesh Distributed primal mesh providing owned-node identity and ownership mappings.
         * @param topology Built primal-edge topology and complete owned-node incidence.
         * @param geometry Efficient macro-surface and point-recovery weights with finalized halo support IDs.
         * @param nodeHalo Finalized NCFV node communication layout; the referenced dependencies must outlive this
         *     object.
         */
        SharedGradientFaceSweep(const MPIInfo &mpi,
                                const ssp<Geom::UnstructuredMesh> &mesh,
                                const Topology &topology,
                                const DualGeometry &geometry,
                                const NodeHalo &nodeHalo)
            : _mpi(mpi), _mesh(mesh), _topology(topology), _geometry(geometry), _nodeHalo(nodeHalo)
        {
        }

        /**
         * @brief Build an extended support halo and canonical face order for the shared-gradient sweep.
         * @details Pulls complete point-recovery rows for remote anchors, allocates persistent
         *     gradient/factor/bound exchanges, and sorts owned faces by stable original-node keys.
         * @pre NodeHalo::Finalize() and geometry node-index remapping have completed; initialize this sweep only
         *     once.
         * @note Collective over the solver communicator.
         */
        void Initialize()
        {
            DNDS_check_throw_info(!_initialized && _nodeHalo.IsFinalized(),
                                  "NCFV shared-gradient sweep must initialize once after node remapping");
            // Fetch complete recovery rows for remote face anchors. These can
            // reference nodes outside the existing reconstruction/flux halo.
            Geom::tAdjPair recoveryNodes;
            ArrayPair<ParArray<real, NonUniformSize>> recoveryWeights;
            Geom::tAdj1Pair originalIDs;
            recoveryNodes.InitPair("NCFV.sharedSweep.recoveryNodes", _mpi);
            recoveryWeights.InitPair("NCFV.sharedSweep.recoveryWeights", _mpi);
            originalIDs.InitPair("NCFV.sharedSweep.originalIDs", _mpi);
            recoveryNodes.father->Resize(_mesh->NumNode());
            recoveryWeights.father->Resize(_mesh->NumNode());
            originalIDs.father->Resize(_mesh->NumNode());
            for (index node = 0; node < _mesh->NumNode(); node++)
            {
                const auto &recovery = _geometry.NodeVolume(node).pointRecoveryStencil;
                DNDS_check_throw_info(recovery.size() <= static_cast<std::size_t>(std::numeric_limits<rowsize>::max() / 3),
                                      "NCFV point-recovery stencil exceeds row capacity");
                recoveryNodes.father->ResizeRow(node, static_cast<rowsize>(recovery.size()));
                recoveryWeights.father->ResizeRow(node, static_cast<rowsize>(3 * recovery.size()));
                for (std::size_t entry = 0; entry < recovery.size(); entry++)
                {
                    recoveryNodes(node, static_cast<rowsize>(entry)) = _nodeHalo.LocalToGlobal(recovery[entry].node);
                    for (int d = 0; d < 3; d++)
                        recoveryWeights(node, static_cast<rowsize>(3 * entry + d)) = recovery[entry].gradientWeight(d);
                }
                originalIDs(node, 0) = _mesh->node2nodeOrig(node, 0);
            }
            recoveryNodes.father->Compress();
            recoveryWeights.father->Compress();
            std::vector<index> ghostAnchors;
            for (index edge = 0; edge < _topology.NumEdge(); edge++)
                for (index anchor : _geometry.EdgeSurface(edge).nodes)
                    if (anchor >= _mesh->NumNode())
                        ghostAnchors.push_back(_nodeHalo.LocalToGlobal(anchor));
            SortUnique(ghostAnchors);
            recoveryNodes.TransAttach();
            recoveryNodes.trans.createFatherGlobalMapping();
            recoveryNodes.trans.createGhostMapping(ghostAnchors);
            recoveryNodes.trans.createMPITypes();
            recoveryNodes.trans.pullOnce();
            recoveryWeights.BorrowAndPull(recoveryNodes);
            originalIDs.BorrowAndPull(recoveryNodes);

            std::vector<index> dependencies;
            for (index edge = 0; edge < _topology.NumEdge(); edge++)
            {
                const auto &surface = _geometry.EdgeSurface(edge);
                for (index anchor : surface.nodes)
                {
                    const index global = _nodeHalo.LocalToGlobal(anchor);
                    dependencies.push_back(global);
                    const index recoveryLocal = FindLocal(recoveryNodes, global);
                    for (index dependency : recoveryNodes[recoveryLocal])
                        dependencies.push_back(dependency);
                }
                for (const auto &entry : surface.efficientStencil)
                    dependencies.push_back(_nodeHalo.LocalToGlobal(entry.node));
            }
            SortUnique(dependencies);
            std::vector<index> ghostDependencies;
            for (index global : dependencies)
            {
                MPI_int ownerRank = UnInitMPIInt;
                index ownerLocal = UnInitIndex;
                const bool found = _mesh->coords.father->pLGlobalMapping->search(global, ownerRank, ownerLocal);
                DNDS_check_throw_info(found, "NCFV shared-gradient dependency has no owner");
                if (ownerRank != _mpi.rank)
                    ghostDependencies.push_back(global);
            }
            _rawGradients.InitPair("NCFV.sharedSweep.rawGradients", _mpi);
            _rawGradients.father->Resize(_mesh->NumNode(), dimension, NVars);
            _rawGradients.son->Resize(static_cast<index>(ghostDependencies.size()), dimension, NVars);
            _rawGradients.TransAttach();
            _rawGradients.trans.createFatherGlobalMapping();
            _rawGradients.trans.createGhostMapping(ghostDependencies);
            _rawGradients.trans.createMPITypes();
            _rawGradients.trans.initPersistentPull();
            _scales.InitPair("NCFV.sharedSweep.scales", _mpi);
            _scales.father->Resize(_mesh->NumNode(), 1, 1);
            _scales.son->Resize(_rawGradients.son->Size(), 1, 1);
            _scales.BorrowSetup(_rawGradients);
            _scales.trans.initPersistentPull();
            for (index node = 0; node < _scales.Size(); node++)
                _scales[node](0) = 1;
            _anchorData.InitPair("NCFV.sharedSweep.anchorData", _mpi);
            _anchorData.father->Resize(_mesh->NumNode(), NVars + 1, 3);
            _anchorData.son->Resize(_rawGradients.son->Size(), NVars + 1, 3);
            _anchorData.BorrowSetup(_rawGradients);
            _anchorData.trans.initPersistentPull();
            _ghostOwnerRanks.resize(static_cast<std::size_t>(_rawGradients.son->Size()));
            _ghostOwnerLocals.resize(_ghostOwnerRanks.size());
            for (index ghost = 0; ghost < _rawGradients.son->Size(); ghost++)
            {
                const index global = _rawGradients.trans.pLGhostMapping->operator()(-1, _mesh->NumNode() + ghost);
                const bool found = _rawGradients.father->pLGlobalMapping->search(
                    global, _ghostOwnerRanks[static_cast<std::size_t>(ghost)],
                    _ghostOwnerLocals[static_cast<std::size_t>(ghost)]);
                DNDS_check_throw_info(found, "NCFV shared-gradient ghost has no owner");
            }

            _faces.reserve(static_cast<std::size_t>(_topology.NumEdge()));
            for (index edge = 0; edge < _topology.NumEdge(); edge++)
            {
                const auto &surface = _geometry.EdgeSurface(edge);
                DNDS_check_throw_info(surface.measure > verySmallReal,
                                      "NCFV shared-gradient face has zero measure");
                Face face;
                std::set<index> support;
                std::array<index, 2> originals{};
                for (int side = 0; side < 2; side++)
                {
                    const index anchorGlobal = _nodeHalo.LocalToGlobal(surface.nodes[side]);
                    face.anchors[side] = FindLocal(_rawGradients, anchorGlobal);
                    const index recoveryLocal = FindLocal(recoveryNodes, anchorGlobal);
                    originals[side] = originalIDs(recoveryLocal, 0);
                    std::map<index, Vector3> folded;
                    const auto recoveryRow = recoveryNodes[recoveryLocal];
                    for (rowsize entry = 0; entry < recoveryRow.size(); entry++)
                    {
                        const index local = FindLocal(_rawGradients, recoveryRow[entry]);
                        Vector3 weight;
                        for (int d = 0; d < 3; d++)
                            weight(d) = -recoveryWeights(recoveryLocal, 3 * entry + d);
                        auto inserted = folded.emplace(local, Vector3::Zero());
                        inserted.first->second += weight;
                        if ((weight.array() != 0).any())
                            support.insert(local);
                    }
                    for (const auto &entry : surface.efficientStencil)
                    {
                        const index local = FindLocal(_rawGradients, _nodeHalo.LocalToGlobal(entry.node));
                        const Vector3 weight = entry.stateGradientWeights[side] / surface.measure;
                        auto inserted = folded.emplace(local, Vector3::Zero());
                        inserted.first->second += weight;
                        if ((weight.array() != 0).any())
                            support.insert(local);
                    }
                    for (const auto &[node, weight] : folded)
                        face.entries[side].push_back({node, weight});
                }
                face.support.assign(support.begin(), support.end());
                const Vector3 displacement = (originals[0] < originals[1] ? 1 : -1) *
                                             _nodeHalo.Displacement(surface.nodes[0], surface.nodes[1]);
                face.key = {std::min(originals[0], originals[1]), std::max(originals[0], originals[1]),
                            std::llround(displacement(0) / 1e-10), std::llround(displacement(1) / 1e-10),
                            std::llround(displacement(2) / 1e-10)};
                _faces.push_back(std::move(face));
            }
            std::sort(_faces.begin(), _faces.end(),
                      [](const Face &left, const Face &right) { return left.key < right.key; });
            index localFaces = static_cast<index>(_faces.size());
            index globalFaces = 0;
            MPI_Allreduce(&localFaces, &globalFaces, 1, DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
            if (_mpi.rank == 0)
                log() << "NCFV shared-gradient face sweep: one pass, MPI rank order, "
                      << _mpi.size << " rank blocks, " << globalFaces
                      << " global faces; no final bound recheck" << std::endl;
            _initialized = true;
        }

        /**
         * @brief Apply one face pass in MPI rank order, updating shared gradient factors immediately after each
         *     face.
         * @details Each face limits the union of its two recovery/interface supports with one common factor.
         *     Remote owner factors are applied between rank blocks.
         * @param means Physical conservative dual-volume means with first-ring ghost values already synchronized.
         * @param [in,out] gradients Runtime physical conservative-gradient field; owned entries are scaled and
         *     final ghost gradients are pulled.
         * @param gamma Ideal-gas ratio of specific heats, strictly greater than one.
         * @param limitPressure Enforce pressure extrema in addition to conservative bounds and positivity when
         *     true.
         * @pre Initialize() has completed.
         * @note Collective. One pass has no final bound recheck; changing rank count or partition may change face
         *     order and the result.
         */
        void Apply(const NodeStatePair &means, NodeMatrixPair &gradients,
                   real gamma, bool limitPressure = false)
        {
            DNDS_check_throw_info(_initialized && gamma > 1,
                                  "NCFV shared-gradient sweep was not initialized or has invalid gamma");
            const real started = MPI_Wtime();
            _lastDiagnostics = {};
            _lastDiagnostics.rankBlocks = _mpi.size;
            for (index node = 0; node < _mesh->NumNode(); node++)
            {
                _rawGradients[node] = gradients[node];
                const State mean = means[node];
                auto data = _anchorData[node];
                for (int column = 0; column < 3; column++)
                {
                    data.template block<NVars, 1>(0, column) = mean;
                    data(NVars, column) = Pressure(mean, gamma);
                }
                for (const auto &incidence : _topology.Node2Edge(node))
                {
                    const auto &surface = _geometry.EdgeSurface(incidence.edge);
                    const index neighbor = surface.nodes[surface.nodes[0] == node ? 1 : 0];
                    const State neighborMean = means[neighbor];
                    data.template block<NVars, 1>(0, 1) =
                        data.template block<NVars, 1>(0, 1).cwiseMin(neighborMean);
                    data.template block<NVars, 1>(0, 2) =
                        data.template block<NVars, 1>(0, 2).cwiseMax(neighborMean);
                    const real neighborPressure = Pressure(neighborMean, gamma);
                    data(NVars, 1) = std::min(data(NVars, 1), neighborPressure);
                    data(NVars, 2) = std::max(data(NVars, 2), neighborPressure);
                }
            }
            for (index node = 0; node < _scales.Size(); node++)
                _scales[node](0) = 1;
            real phaseStart = MPI_Wtime();
            _rawGradients.trans.startPersistentPull();
            _rawGradients.trans.waitPersistentPull();
            _anchorData.trans.startPersistentPull();
            _anchorData.trans.waitPersistentPull();
            _lastDiagnostics.inputHaloSeconds = MPI_Wtime() - phaseStart;
            std::vector<real> remoteFactors(static_cast<std::size_t>(_scales.son->Size()), 1);
            for (MPI_int activeRank = 0; activeRank < _mpi.size; activeRank++)
            {
                phaseStart = MPI_Wtime();
                if (_mpi.rank == activeRank)
                    for (const auto &face : _faces)
                    {
                        const State leftIncrement = TraceIncrement(face.entries[0]);
                        const State rightIncrement = TraceIncrement(face.entries[1]);
                        const real factor = std::min(
                            TraceFactor(face.anchors[0], leftIncrement, gamma, limitPressure),
                            TraceFactor(face.anchors[1], rightIncrement, gamma, limitPressure));
                        if (factor >= 1)
                            continue;
                        _lastDiagnostics.updatedFaces++;
                        _lastDiagnostics.zeroFactorFaces += factor == 0;
                        _lastDiagnostics.minimumFaceFactor = std::min(_lastDiagnostics.minimumFaceFactor, factor);
                        for (index node : face.support)
                        {
                            _scales[node](0) *= factor;
                            if (node >= _mesh->NumNode())
                                remoteFactors[static_cast<std::size_t>(node - _mesh->NumNode())] *= factor;
                        }
                    }
                _lastDiagnostics.localSweepSeconds += MPI_Wtime() - phaseStart;
                phaseStart = MPI_Wtime();
                ExchangeRemoteFactors(activeRank, remoteFactors);
                if (activeRank + 1 < _mpi.size)
                {
                    _scales.trans.startPersistentPull();
                    _scales.trans.waitPersistentPull();
                }
                _lastDiagnostics.blockCommunicationSeconds += MPI_Wtime() - phaseStart;
            }
            for (index node = 0; node < _mesh->NumNode(); node++)
                gradients[node] = _scales[node](0) * _rawGradients[node];
            gradients.trans.startPersistentPull();
            gradients.trans.waitPersistentPull();
            index localCounts[2]{_lastDiagnostics.updatedFaces, _lastDiagnostics.zeroFactorFaces};
            index globalCounts[2]{};
            MPI_Allreduce(localCounts, globalCounts, 2, DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
            _lastDiagnostics.updatedFaces = globalCounts[0];
            _lastDiagnostics.zeroFactorFaces = globalCounts[1];
            MPI_Allreduce(MPI_IN_PLACE, &_lastDiagnostics.minimumFaceFactor, 1,
                          DNDS_MPI_REAL, MPI_MIN, _mpi.comm);
            _lastDiagnostics.totalSeconds = MPI_Wtime() - started;
        }

        /**
         * @brief Return statistics from the most recent shared-gradient face pass.
         * @return Read-only diagnostics: globally reduced face counts/minimum factor plus local timings in
         *     seconds.
         */
        [[nodiscard]] const Diagnostics &LastDiagnostics() const { return _lastDiagnostics; }
        /**
         * @brief Return the accumulated shared-gradient multiplier of one owned node.
         * @param localNode Owned node index in [0,mesh.NumNode()).
         * @return Scalar multiplier accumulated during the latest initialized sweep.
         */
        [[nodiscard]] real OwnedFactor(index localNode) const
        {
            DNDS_check_throw_info(_initialized && localNode >= 0 && localNode < _mesh->NumNode(),
                                  "NCFV shared-gradient factor query requires an owned node");
            return _scales[localNode](0);
        }
    };
}
