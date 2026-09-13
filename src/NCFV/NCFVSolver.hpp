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
        std::unique_ptr<PeriodicNodes> _periodic;
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

        void ReadMesh();
        void BroadcastBoundaryNameMap();
        void AllocateFields();
        void InitializeFreshField();
        void InitializeVortex();
        void WriteVortexDiagnostics(bool initial);
        void ApplyInitialRegions();
        void ReadInitialNodeFile();
        void SynchronizeState(NodeStatePair &state);
        void ReadRestart(const std::string &baseName);
        void WriteRestart(const std::string &baseName);
        void WriteOutput(const std::string &baseName);
        void WriteResolvedConfiguration() const;
        [[nodiscard]] std::string StampedName(
            const std::string &prefix,
            index iteration) const;
        State PrimitiveToConservative(const std::vector<real> &primitive) const;
        State ConservativeToPrimitive(const State &conservative) const;
        bool StateIsPhysical(const State &state) const;
        void CheckOwnedState(const NodeStatePair &state, const std::string &where) const;

    public:
        Solver(const MPIInfo &mpi, const Configuration &configuration)
            : _mpi(mpi), _configuration(configuration)
        {
            DNDS_check_throw_info(_configuration.dimension == dimension,
                                  "NCFV compile-time and configured dimensions differ");
        }

        void Initialize();
        void Run();
        real EvaluateResidual() { return _spatial->EvaluateRHS(_state, _rhs); }
        /** Enable intrusive internal-edge flux timing after Initialize(). */
        void EnableDetailedFluxTiming(bool enabled)
        {
            DNDS_check_throw_info(_spatial != nullptr,
                                  "Initialize solver before enabling detailed flux timing");
            _spatial->EnableDetailedFluxTiming(enabled);
        }
        /** Select the precomputed or legacy on-demand physical-flux gradients. */
        void UsePrecomputedPhysicalFluxGradients(bool enabled)
        {
            DNDS_check_throw_info(
                _spatial != nullptr,
                "Initialize solver before selecting physical-flux gradients");
            _spatial->UsePrecomputedPhysicalFluxGradients(enabled);
        }

        [[nodiscard]] const ssp<Geom::UnstructuredMesh> &Mesh() const { return _mesh; }
        [[nodiscard]] const Topology &EdgeTopology() const { return *_topology; }
        [[nodiscard]] const DualGeometry &Geometry() const { return *_geometry; }
        [[nodiscard]] const Reconstruction &ReconstructionData() const { return *_reconstruction; }
        [[nodiscard]] const NodeStatePair &StateField() const { return _state; }
        [[nodiscard]] const NodeStatePair &ResidualField() const { return _rhs; }
        [[nodiscard]] const BoundaryRegistry &Boundaries() const { return *_boundaries; }
        [[nodiscard]] index CurrentIteration() const { return _currentIteration; }
        [[nodiscard]] real SimulationTime() const { return _simulationTime; }
        /** Local selected-Riemann-solver calls in the most recently completed SSPRK3 step. */
        [[nodiscard]] index LastStepRiemannSolverCalls() const
        {
            return _lastStepRiemannSolverCalls;
        }
        /** Local phase times in the most recently completed SSPRK3 step. */
        [[nodiscard]] const StepPhaseTiming &LastStepTiming() const
        {
            return _lastStepTiming;
        }
        [[nodiscard]] const RhsPhaseTiming &LastRhsTiming() const
        {
            return _spatial->LastRhsTiming();
        }
        [[nodiscard]] real LocalTimeStep(index iNode) const
        {
            return _spatial->LocalTimeStep(iNode);
        }
    };

    extern template class Solver<2>;
    extern template class Solver<3>;
}
