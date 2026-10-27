/**
 * @file NCFVSolver.hpp
 * @brief Standalone MPI NCFV (Node Center Finite Volume Method) solver assembly.
 */
#pragma once

#include "NCFVSpatial.hpp"

#include "CFV/DOFFactory.hpp"
#include "Geom/Mesh/Mesh_Helpers.hpp"

#include <memory>
#include <unordered_map>

namespace DNDS::NCFV
{
    /** Per-rank wall times for one completed production SSPRK3 step. */
    struct StepPhaseTiming
    {
        RhsPhaseTiming rhs;
        real baseStateCopySeconds = 0;
        real stageUpdateSeconds = 0;
        real totalSeconds = 0;
    };

    template <int dimension>
    class Solver
    {
        static_assert(dimension == 2 || dimension == 3);

        using State = Eigen::Vector<real, dimension + 2>;

        MPIInfo _mpi;
        Configuration _configuration;
        ssp<Geom::UnstructuredMesh> _mesh;
        ssp<Geom::UnstructuredMeshSerialRW> _reader;
        std::unordered_map<std::string, Geom::t_index> _boundaryNameToID;
        std::unique_ptr<Topology> _topology;
        std::unique_ptr<DualGeometry> _geometry;
        std::unique_ptr<NodeHalo> _nodeHalo;
        std::unique_ptr<Reconstruction> _reconstruction;
        std::unique_ptr<BoundaryRegistry> _boundaries;
        std::unique_ptr<SpatialOperator<dimension>> _spatial;

        NodeStatePair _state;
        NodeStatePair _baseState;
        NodeStatePair _stageState;
        NodeStatePair _rhs;
        index _currentIteration = 0;
        real _simulationTime = 0;
        index _lastOutputIteration = UnInitIndex;
        index _lastRestartIteration = UnInitIndex;
        index _lastStepRiemannSolverCalls = 0;
        StepPhaseTiming _lastStepTiming;

        /**
         * @brief Read, partition, and prepare the primal mesh and periodic connectivity.
         * @details Builds the mesh/reader and retains the CGNS boundary-name mapping.
         * @note Collective over the solver communicator.
         */
        void ReadMesh();
        /**
         * @brief Broadcast the reader's CGNS boundary name/ID mapping to all solver ranks.
         * @note Collective over the solver communicator.
         */
        void BroadcastBoundaryNameMap();
        /**
         * @brief Allocate state, RK stage/history, and RHS fields using the finalized node-halo mapping.
         * @details Initializes persistent state pulls for the runtime dependency layout.
         * @note Collective over the solver communicator.
         */
        void AllocateFields();
        /**
         * @brief Populate a fresh conservative solution from uniform, regional, expression, file, or vortex data.
         * @note Collective over the solver communicator.
         */
        void InitializeFreshField();
        /**
         * @brief Initialize conservative dual-volume means of the analytic vortex with the selected integration
         *     rule.
         * @note Collective over the solver communicator.
         */
        void InitializeVortex();
        /**
         * @brief Write analytic-vortex error diagnostics for the current solution.
         * @param initial True to label initialization diagnostics; false for diagnostics at the current evolved
         *     time.
         */
        void WriteVortexDiagnostics(bool initial);
        /**
         * @brief Apply configured box, plane, and expression initializers to owned-node state entries.
         * @details Later configured assignments may replace values selected by earlier initializers.
         * @note Collective over the solver communicator.
         */
        void ApplyInitialRegions();
        /**
         * @brief Read configured global-node primitive data and convert matching owned values to conservative
         *     storage.
         * @details Honors the configured file index base and completeness requirement.
         * @note Collective over the solver communicator.
         */
        void ReadInitialNodeFile();
        /**
         * @brief Pull current owner values into the finalized node ghosts and wait for completion.
         * @param [in,out] state Allocated conservative node field borrowing the finalized halo mapping.
         */
        void SynchronizeState(NodeStatePair &state);
        /**
         * @brief Read the rank-local restart state and restore iteration/time metadata.
         * @param baseName Restart file base path; the reader applies the restart naming convention.
         * @pre Mesh, runtime node halo, and state fields are initialized with compatible ownership.
         */
        void ReadRestart(const std::string &baseName);
        /**
         * @brief Serialize conservative dual means and restart metadata for the current distributed solution.
         * @param baseName Restart file base path; parent directories and rank-specific names follow the writer
         *     convention.
         */
        void WriteRestart(const std::string &baseName);
        /**
         * @brief Write conservative node-volume data and derived primitive fields in the configured VTK format.
         * @param baseName Snapshot file base path used by the mesh VTK writer and output series.
         */
        void WriteOutput(const std::string &baseName);
        /**
         * @brief Write the fully resolved case configuration when enabled.
         * @details Uses the configured output location and rank-zero file writing.
         */
        void WriteResolvedConfiguration() const;
        /**
         * @brief Append a fixed-width iteration stamp to an output or restart prefix.
         * @param prefix Directory/file prefix without the iteration suffix.
         * @param iteration Nonnegative iteration number to format.
         * @return Prefix followed by the iteration stamp used for snapshot naming.
         */
        [[nodiscard]] std::string StampedName(
            const std::string &prefix,
            index iteration) const;
        /**
         * @brief Convert ideal-gas primitive data to the NCFV conservative state layout.
         * @param primitive dimension+2 entries [rho,u...,p] with positive density and pressure.
         * @return State [rho,rho*u...,rho*E], with rho*E = p/(gamma-1) + rho*|u|^2/2.
         */
        State PrimitiveToConservative(const std::vector<real> &primitive) const;
        /**
         * @brief Recover density, velocity, and pressure from ideal-gas conservative data.
         * @param conservative Conservative state [rho,rho*u...,rho*E] with positive density.
         * @return Primitive state [rho,u...,p] in the same dimension+2-entry vector.
         */
        State ConservativeToPrimitive(const State &conservative) const;
        /**
         * @brief Test ideal-gas density, pressure, and component finiteness.
         * @param state Conservative state [rho,rho*u...,rho*E].
         * @return True for a finite state with admissible positive density and pressure.
         */
        bool StateIsPhysical(const State &state) const;
        /**
         * @brief Count inadmissible owned states globally for step acceptance.
         * @param state Distributed conservative field; ghost entries do not contribute to the count.
         * @return MPI sum of owned states that fail StateIsPhysical().
         */
        index CountNonphysicalOwnedStates(const NodeStatePair &state) const;
        /**
         * @brief Reject a distributed state if any owned node is nonphysical.
         * @param state Conservative field to check on owned nodes.
         * @param where Context label included in the failure message.
         * @throws std::runtime_error If the globally reduced nonphysical-state count is nonzero.
         */
        void CheckOwnedState(const NodeStatePair &state, const std::string &where) const;

    public:
        /**
         * @brief Construct the dimension-specific NCFV driver and check its runtime dimension.
         * @param mpi MPI communicator metadata; collective operations must be entered by every participating rank.
         * @param configuration Complete case settings copied into the driver; dimension must match the template
         *     argument.
         */
        Solver(const MPIInfo &mpi, const Configuration &configuration)
            : _mpi(mpi), _configuration(configuration)
        {
            DNDS_check_throw_info(_configuration.dimension == dimension,
                                  "NCFV compile-time and configured dimensions differ");
        }

        /**
         * @brief Build the complete NCFV mesh, topology, dual geometry, exact node halo, reconstruction, and
         *     initial state.
         * @details Loads restart data when requested, applies strong boundaries, and performs configured initial
         *     output/diagnostics.
         * @note Collective over the solver communicator.
         */
        void Initialize();
        /**
         * @brief Advance the conservative node-volume means with SSPRK3 and configured time-step controls.
         * @details Reports convergence, retries rejected nonphysical steps when enabled, and writes
         *     scheduled/final VTK and restart data.
         * @note Collective over the solver communicator.
         */
        void Run();
        /**
         * @brief Evaluate and store the current spatial RHS and update CFL time-step diagnostics.
         * @return Volume-weighted global RMS conservative residual from the spatial operator.
         * @pre Initialize() has completed; all ranks participate.
         */
        real EvaluateResidual() { return _spatial->EvaluateRHS(_state, _rhs); }
        /**
         * @brief Enable intrusive internal-edge flux timing after Initialize().
         * @param enabled True to collect intrusive internal-edge kernel timings; false to retain ordinary phase
         *     timing only.
         */
        void EnableDetailedFluxTiming(bool enabled)
        {
            DNDS_check_throw_info(_spatial != nullptr,
                                  "Initialize solver before enabling detailed flux timing");
            _spatial->EnableDetailedFluxTiming(enabled);
        }
        /**
         * @brief Return shared distributed primal mesh; initialized by Initialize().
         * @return Shared distributed primal mesh; initialized by Initialize().
         */
        [[nodiscard]] const ssp<Geom::UnstructuredMesh> &Mesh() const { return _mesh; }
        /**
         * @brief Return read-only built primal-edge/face topology.
         * @return Read-only built primal-edge/face topology.
         */
        [[nodiscard]] const Topology &EdgeTopology() const { return *_topology; }
        /**
         * @brief Return read-only built median-dual integration geometry.
         * @return Read-only built median-dual integration geometry.
         */
        [[nodiscard]] const DualGeometry &Geometry() const { return *_geometry; }
        /**
         * @brief Return read-only finalized NCFV node-halo layout; Initialize() must have completed.
         * @return Read-only finalized NCFV node-halo layout; Initialize() must have completed.
         */
        [[nodiscard]] const NodeHalo &NodeCommunication() const
        {
            DNDS_check_throw_info(_nodeHalo != nullptr,
                                  "Initialize solver before accessing node communication");
            return *_nodeHalo;
        }
        /**
         * @brief Return read-only precomputed runtime reconstruction operators.
         * @return Read-only precomputed runtime reconstruction operators.
         */
        [[nodiscard]] const Reconstruction &ReconstructionData() const { return *_reconstruction; }
        /**
         * @brief Return read-only current conservative dual-volume means in the exact runtime halo layout.
         * @return Read-only current conservative dual-volume means in the exact runtime halo layout.
         */
        [[nodiscard]] const NodeStatePair &StateField() const { return _state; }
        /**
         * @brief Return read-only most recently evaluated spatial RHS, with authoritative entries on owned nodes.
         * @return Read-only most recently evaluated spatial RHS, with authoritative entries on owned nodes.
         */
        [[nodiscard]] const NodeStatePair &ResidualField() const { return _rhs; }
        /**
         * @brief Return read-only boundary-zone registry built from mesh names and case settings.
         * @return Read-only boundary-zone registry built from mesh names and case settings.
         */
        [[nodiscard]] const BoundaryRegistry &Boundaries() const { return *_boundaries; }
        /**
         * @brief Return current completed iteration index, including any restored restart offset.
         * @return Current completed iteration index, including any restored restart offset.
         */
        [[nodiscard]] index CurrentIteration() const { return _currentIteration; }
        /**
         * @brief Return current simulation time restored or advanced by the driver.
         * @return Current simulation time restored or advanced by the driver.
         */
        [[nodiscard]] real SimulationTime() const { return _simulationTime; }
        /**
         * @brief Local selected-Riemann-solver calls in the most recently completed SSPRK3 step.
         * @return Local selected-Riemann-solver call count in the latest completed SSPRK3 step.
         */
        [[nodiscard]] index LastStepRiemannSolverCalls() const
        {
            return _lastStepRiemannSolverCalls;
        }
        /**
         * @brief Local phase times in the most recently completed SSPRK3 step.
         * @return Read-only per-rank phase timings for the latest completed SSPRK3 step, in seconds.
         */
        [[nodiscard]] const StepPhaseTiming &LastStepTiming() const
        {
            return _lastStepTiming;
        }
        /**
         * @brief Return read-only per-rank timings for the latest spatial RHS evaluation, in seconds.
         * @return Read-only per-rank timings for the latest spatial RHS evaluation, in seconds.
         */
        [[nodiscard]] const RhsPhaseTiming &LastRhsTiming() const
        {
            return _spatial->LastRhsTiming();
        }
        /**
         * @brief Return the spatial operator's latest cached step for an owned node.
         * @param iNode Owned primal-node index.
         * @return Positive cached local or globally uniform time interval.
         */
        [[nodiscard]] real LocalTimeStep(index iNode) const
        {
            return _spatial->LocalTimeStep(iNode);
        }
    };

    extern template class Solver<2>;
    extern template class Solver<3>;
}
