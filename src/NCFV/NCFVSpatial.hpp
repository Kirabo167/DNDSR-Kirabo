/**
 * @file NCFVSpatial.hpp
 * @brief Euler/Navier--Stokes residuals for both third-order NCFV modes.
 */
#pragma once

#include "NCFVBoundary.hpp"
#include "NCFVReconstruction.hpp"

#include "Euler/Gas.hpp"



namespace DNDS::NCFV
{
    /** Fine-grained wall times for one side of the efficient physical-flux integral. */
    struct EfficientPhysicalFluxIntegralTiming
    {
        real zeroOrderFluxSeconds = 0;
        real precomputedGradientIntegralSeconds = 0;
        real finalAssemblySeconds = 0;

        EfficientPhysicalFluxIntegralTiming &operator+=(
            const EfficientPhysicalFluxIntegralTiming &other)
        {
            zeroOrderFluxSeconds += other.zeroOrderFluxSeconds;
            precomputedGradientIntegralSeconds +=
                other.precomputedGradientIntegralSeconds;
            finalAssemblySeconds += other.finalAssemblySeconds;
            return *this;
        }
    };

    /** Per-rank wall times for one complete spatial residual evaluation. */
    struct RhsPhaseTiming
    {
        real reconstructionSeconds = 0;
        real meanHaloSeconds = 0;
        real coefficientComputeSeconds = 0;
        real coefficientHaloSeconds = 0;
        real pointRecoverySeconds = 0;
        real limiterSeconds = 0;
        real pointValueHaloSeconds = 0;
        real physicalFluxGradientComputeSeconds = 0;
        real edgeFluxSeconds = 0;
        // Timings accumulated over all owned internal-edge evaluations in an RHS.
        // In efficient mode, the state-preparation entries measure the two
        // EfficientSurfaceMean calls.  In traditional mode they instead measure
        // the two quadrature-point trace reconstructions, since no face mean is
        // formed by that formulation.
        real edgeLeftStatePreparationSeconds = 0;
        real edgeRightStatePreparationSeconds = 0;
        // In efficient mode these are the left/right derivative-based physical
        // flux integrals.  In traditional mode the combined entry is the
        // quadrature numerical-flux integration; the side entries remain zero.
        real edgeLeftPhysicalFluxIntegralSeconds = 0;
        real edgeRightPhysicalFluxIntegralSeconds = 0;
        real edgeNumericalFluxAndAssemblySeconds = 0;
        EfficientPhysicalFluxIntegralTiming edgeLeftPhysicalFluxDetail;
        EfficientPhysicalFluxIntegralTiming edgeRightPhysicalFluxDetail;
        real edgeRiemannFluxSeconds = 0;
        real edgeCentralFluxSeconds = 0;
        real edgeDissipationAssemblySeconds = 0;
        real edgeInviscidFinalAssemblySeconds = 0;
        real edgeViscousFluxSeconds = 0;
        real edgeHaloSeconds = 0;
        real localTimeStepSeconds = 0;
        real residualAssemblySeconds = 0;
        real residualNormSeconds = 0;
        real totalSeconds = 0;

        RhsPhaseTiming &operator+=(const RhsPhaseTiming &other)
        {
            reconstructionSeconds += other.reconstructionSeconds;
            meanHaloSeconds += other.meanHaloSeconds;
            coefficientComputeSeconds += other.coefficientComputeSeconds;
            coefficientHaloSeconds += other.coefficientHaloSeconds;
            pointRecoverySeconds += other.pointRecoverySeconds;
            limiterSeconds += other.limiterSeconds;
            pointValueHaloSeconds += other.pointValueHaloSeconds;
            physicalFluxGradientComputeSeconds +=
                other.physicalFluxGradientComputeSeconds;
            edgeFluxSeconds += other.edgeFluxSeconds;
            edgeLeftStatePreparationSeconds += other.edgeLeftStatePreparationSeconds;
            edgeRightStatePreparationSeconds += other.edgeRightStatePreparationSeconds;
            edgeLeftPhysicalFluxIntegralSeconds += other.edgeLeftPhysicalFluxIntegralSeconds;
            edgeRightPhysicalFluxIntegralSeconds += other.edgeRightPhysicalFluxIntegralSeconds;
            edgeNumericalFluxAndAssemblySeconds += other.edgeNumericalFluxAndAssemblySeconds;
            edgeLeftPhysicalFluxDetail += other.edgeLeftPhysicalFluxDetail;
            edgeRightPhysicalFluxDetail += other.edgeRightPhysicalFluxDetail;
            edgeRiemannFluxSeconds += other.edgeRiemannFluxSeconds;
            edgeCentralFluxSeconds += other.edgeCentralFluxSeconds;
            edgeDissipationAssemblySeconds += other.edgeDissipationAssemblySeconds;
            edgeInviscidFinalAssemblySeconds += other.edgeInviscidFinalAssemblySeconds;
            edgeViscousFluxSeconds += other.edgeViscousFluxSeconds;
            edgeHaloSeconds += other.edgeHaloSeconds;
            localTimeStepSeconds += other.localTimeStepSeconds;
            residualAssemblySeconds += other.residualAssemblySeconds;
            residualNormSeconds += other.residualNormSeconds;
            totalSeconds += other.totalSeconds;
            return *this;
        }
    };

    template <int dimension>
    class SpatialOperator
    {
        static_assert(dimension == 2 || dimension == 3);

        using State = Eigen::Vector<real, dimension + 2>;
        using StateGradient = Eigen::Matrix<real, dimension, dimension + 2>;
        using PhysicalFluxGradient =
            Eigen::Matrix<real, dimension + 2, dimension * dimension>;
        using SpatialVector = Eigen::Vector<real, dimension>;

        /** Column packing for the cached vector derivative d(F_j)/d(x_i). */
        static constexpr int FluxGradientColumn(
            int derivativeDirection,
            int fluxDirection)
        {
            return derivativeDirection * dimension + fluxDirection;
        }

        const MPIInfo &_mpi;
        ssp<Geom::UnstructuredMesh> _mesh;
        const Topology &_topology;
        const DualGeometry &_geometry;
        const Reconstruction &_reconstruction;
        const BoundaryRegistry &_boundaries;
        IntegrationMode _mode;
        ReconstructionSettings _reconstructionSettings;
        PhysicsSettings _physics;
        TimeSettings _time;
        const NodeHalo &_nodeHalo;
        real _maximumStep = veryLargeReal;
        bool _detailedFluxTiming = false;

        NodeMatrixPair _stateGradients;
        NodeMatrixPair _physicalFluxGradients;
        NodeMatrixPair _coefficients;
        NodeStatePair _pointValues;
        // Thesis (3-96): one anchor coefficient multiplies the complete
        // differential-weight correction for that side of every macro surface.
        NodeStatePair _limiterFactors;
        NodeStatePair _edgeFlux;
        NodeStatePair _edgeSpectralRadius;
        NodeStatePair _localTimeSteps;
        State _farField = State::Zero();
        real _lastMinimumTimeStep = 0;
        real _lastMaximumTimeStep = 0;
        RhsPhaseTiming _lastRhsTiming;
        // Per-rank counter. EvaluateRHS currently evaluates its edge and
        // boundary loops serially, so no atomic synchronization is needed.
        mutable index _riemannSolverCallCount = 0;

        State PrimitiveToConservative(const std::vector<real> &primitive) const;
        State ConservativeToPrimitive(const State &state) const;
        real Pressure(const State &state) const;
        real Temperature(const State &state) const;
        real MolecularViscosity(const State &state) const;
        bool IsPhysical(const State &state) const;
        State PreservePhysical(const State &candidate, const State &anchor) const;

        State PhysicalFlux(const State &state, const SpatialVector &normal) const;
        State NumericalFlux(
            const State &left,
            const State &right,
            const SpatialVector &unitNormal) const;
        State BoundaryExterior(
            const State &inside,
            const SpatialVector &unitNormal,
            const BoundaryZoneSettings &boundary) const;
        State BoundaryNumericalFlux(
            const State &inside,
            const SpatialVector &unitNormal,
            const BoundaryZoneSettings &boundary) const;

        State EvaluateTraditionalState(index anchorNode, const Vector3 &point) const;
        StateGradient EvaluateTraditionalGradient(index anchorNode, const Vector3 &point) const;
        StateGradient EfficientSurfaceGradient(
            int side,
            const EdgeControlSurface &surface) const;
        StateGradient EfficientBoundarySurfaceGradient(
            index anchorNode,
            const BoundaryPiece &piece) const;
        State EfficientIntegratedPhysicalFlux(
            int side,
            const EdgeControlSurface &surface,
            EfficientPhysicalFluxIntegralTiming &timing) const;
        State EfficientSurfaceMean(
            int side,
            const EdgeControlSurface &surface) const;
        State InternalViscousFlux(
            const State &left,
            const State &right,
            const StateGradient &leftGradient,
            const StateGradient &rightGradient,
            const SpatialVector &unitNormal,
            real characteristicDistance) const;
        State BoundaryViscousFlux(
            const State &inside,
            const StateGradient &insideGradient,
            const SpatialVector &unitNormal,
            const BoundaryZoneSettings &boundary,
            real lengthScale) const;
        real SurfaceSpectralRadius(
            const State &state,
            const SpatialVector &unitNormal,
            real measure,
            real lengthScale) const;
        State EvaluateOwnedEdgeFlux(index iEdge);
        real EvaluateOwnedEdgeSpectralRadius(index iEdge) const;
        State EvaluateOwnedBoundaryFlux(index iNode) const;
        real EvaluateOwnedBoundarySpectralRadius(index iNode) const;

        void AllocateNodeMatrix(
            NodeMatrixPair &field,
            const std::string &name,
            int rows,
            int columns = dimension + 2);
        void AllocateLocalNodeMatrix(
            NodeMatrixPair &field,
            const std::string &name,
            int rows,
            int columns);
        void AllocateNodeField(
            NodeStatePair &field,
            const std::string &name,
            int rows);
        void AllocateLocalNodeField(
            NodeStatePair &field,
            const std::string &name,
            int rows);
        void AllocateEdgeField(NodeStatePair &field, const std::string &name, int rows);
        void Reconstruct(NodeStatePair &means);
        void ComputePhysicalFluxGradients();
        void UpdateLocalTimeSteps();

    public:
        SpatialOperator(
            const MPIInfo &mpi,
            const ssp<Geom::UnstructuredMesh> &mesh,
            const Topology &topology,
            const DualGeometry &geometry,
            const Reconstruction &reconstruction,
            const BoundaryRegistry &boundaries,
            IntegrationMode mode,
            const ReconstructionSettings &reconstructionSettings,
            const PhysicsSettings &physics,
            const TimeSettings &time,
            const NodeHalo &nodeHalo)
            : _mpi(mpi), _mesh(mesh), _topology(topology), _geometry(geometry),
              _reconstruction(reconstruction), _boundaries(boundaries), _mode(mode),
              _reconstructionSettings(reconstructionSettings), _physics(physics),
              _time(time), _nodeHalo(nodeHalo)
        {
        }

        void Initialize();
        void SetMaximumStep(real step) { _maximumStep = step; }
        /** Enable intrusive per-kernel interface-flux timing for diagnostics. */
        void EnableDetailedFluxTiming(bool enabled) { _detailedFluxTiming = enabled; }

        /**
         * @brief Evaluate d(dual mean)/dt; performs all node and edge halo pulls.
         * @return Volume-weighted global RMS residual, or zero when
         *         computeResidualNorm is false.
         */
        real EvaluateRHS(
            NodeStatePair &means,
            NodeStatePair &rhs,
            bool updateTimeSteps = true,
            bool computeResidualNorm = true);

        /** @brief Apply configured strong inlet/no-slip conditions to owned nodes. */
        void ApplyStrongBoundaryConditions(NodeStatePair &state) const;

        [[nodiscard]] real LocalTimeStep(index iNode) const
        {
            return _localTimeSteps[iNode](0, 0);
        }
        [[nodiscard]] real LastMinimumTimeStep() const { return _lastMinimumTimeStep; }
        [[nodiscard]] real LastMaximumTimeStep() const { return _lastMaximumTimeStep; }

        /** Reset the local count of calls to the selected inviscid Riemann solver. */
        void ResetRiemannSolverCallCount() { _riemannSolverCallCount = 0; }

        /** Return the local count of NumericalFlux dispatcher calls since the last reset. */
        [[nodiscard]] index RiemannSolverCallCount() const
        {
            return _riemannSolverCallCount;
        }

        /** Local phase times for the most recent EvaluateRHS() call. */
        [[nodiscard]] const RhsPhaseTiming &LastRhsTiming() const
        {
            return _lastRhsTiming;
        }

        [[nodiscard]] const NodeMatrixPair &Gradients() const { return _stateGradients; }
        [[nodiscard]] const NodeMatrixPair &PhysicalFluxGradients() const
        {
            return _physicalFluxGradients;
        }
        [[nodiscard]] const NodeMatrixPair &Coefficients() const { return _coefficients; }
        [[nodiscard]] const NodeStatePair &PointValues() const { return _pointValues; }
        [[nodiscard]] const NodeStatePair &LimiterFactors() const { return _limiterFactors; }
        [[nodiscard]] const NodeStatePair &LocalTimeSteps() const { return _localTimeSteps; }
    };

    extern template class SpatialOperator<2>;
    extern template class SpatialOperator<3>;
}
