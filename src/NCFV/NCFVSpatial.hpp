/**
 * @file NCFVSpatial.hpp
 * @brief Euler/Navier--Stokes residuals for both third-order NCFV modes.
 */
#pragma once

#include "NCFVBoundary.hpp"
#include "NCFVReconstruction.hpp"
#include "NCFVSharedGradientLimiter.hpp"

#include "Euler/Gas.hpp"



namespace DNDS::NCFV
{
    /**
     * @brief Fine-grained local wall times for one side of an efficient physical-flux integral.
     * @details Values are seconds and are accumulated only by enabled detailed flux timing.
     * They describe subphases of a larger interface timer, not additional disjoint work.
     */
    struct EfficientPhysicalFluxIntegralTiming
    {
        real zeroOrderFluxSeconds = 0; ///< Evaluating the physical flux at the side's anchor point state.
        real precomputedGradientIntegralSeconds = 0; ///< Contracting cached flux derivatives with integral weights.
        real finalAssemblySeconds = 0; ///< Combining zero-order and limited differential contributions.

        /**
         * @brief Accumulate corresponding phase timings without MPI reduction.
         * @param other Timing record whose per-phase seconds are added to this record.
         * @return Reference to this updated timing record.
         */
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

    /**
     * @brief Local wall-clock phase accounting for one complete spatial RHS evaluation.
     * @details All values are seconds on this rank, with no automatic MPI reduction.
     * Reconstruction and edge timers include their nested subphase/detail timers: summing
     * every member would double-count work. Mode-specific or disabled detail fields stay zero.
     */
    struct RhsPhaseTiming
    {
        real reconstructionSeconds = 0; ///< Inclusive reconstruction, limiting, and associated halo exchanges.
        real meanHaloSeconds = 0; ///< Pulling conservative dual means required by reconstruction.
        real coefficientComputeSeconds = 0; ///< Computing gradient/polynomial coefficient fields.
        real coefficientHaloSeconds = 0; ///< Pulling reconstructed gradient/coefficient dependencies.
        real pointRecoverySeconds = 0; ///< Recovering primal-node point states from dual means.
        real limiterSeconds = 0; ///< Constructing/applying selected reconstruction and physical-state limits.
        real pointValueHaloSeconds = 0; ///< Synchronizing recovered or limited point-state anchors.
        real physicalFluxGradientComputeSeconds = 0; ///< Computing cached Cartesian Euler-flux derivatives.
        real edgeFluxSeconds = 0; ///< Inclusive owned-edge flux and edge spectral-radius evaluation loop.
        // Timings accumulated over all owned internal-edge evaluations in an RHS.
        // In efficient mode, the state-preparation entries measure the two
        // EfficientSurfaceMean calls.  In traditional mode they instead measure
        // the two quadrature-point trace reconstructions, since no face mean is
        // formed by that formulation.
        real edgeLeftStatePreparationSeconds = 0; ///< Left surface mean or left quadrature trace preparation.
        real edgeRightStatePreparationSeconds = 0; ///< Right surface mean or right quadrature trace preparation.
        // In efficient mode these are the left/right derivative-based physical
        // flux integrals.  In traditional mode the combined entry is the
        // quadrature numerical-flux integration; the side entries remain zero.
        real edgeLeftPhysicalFluxIntegralSeconds = 0; ///< Efficient left-side physical-flux integration.
        real edgeRightPhysicalFluxIntegralSeconds = 0; ///< Efficient right-side physical-flux integration.
        real edgeNumericalFluxAndAssemblySeconds = 0; ///< Combined interface numerical-flux assembly time.
        EfficientPhysicalFluxIntegralTiming edgeLeftPhysicalFluxDetail; ///< Nested efficient left-side timers.
        EfficientPhysicalFluxIntegralTiming edgeRightPhysicalFluxDetail; ///< Nested efficient right-side timers.
        real edgeRiemannFluxSeconds = 0; ///< Selected Riemann flux evaluated at interface means/traces.
        real edgeCentralFluxSeconds = 0; ///< Centered physical-flux contribution used to isolate dissipation.
        real edgeDissipationAssemblySeconds = 0; ///< Forming the numerical-minus-central correction.
        real edgeInviscidFinalAssemblySeconds = 0; ///< Combining inviscid integral terms.
        real edgeViscousFluxSeconds = 0; ///< Internal viscous flux evaluation and integration.
        real edgeHaloSeconds = 0; ///< Pulling edge-owner fluxes and spectral radii onto incident-node ranks.
        real localTimeStepSeconds = 0; ///< Computing/clamping steps and reducing their global extrema.
        real residualAssemblySeconds = 0; ///< Summing boundary and signed edge fluxes, then dividing by volume.
        real residualNormSeconds = 0; ///< Forming/reducing the optional volume-weighted global RMS norm.
        real totalSeconds = 0; ///< Inclusive EvaluateRHS() wall time; not a sum of all stored subphase times.

        /**
         * @brief Accumulate corresponding phase timings without MPI reduction.
         * @param other Timing record whose per-phase seconds are added to this record.
         * @return Reference to this updated timing record.
         */
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

    /**
     * @brief Owner/ghost work arrays and spatial RHS evaluation for node-centered ideal-gas flow.
     * @tparam dimension Two or three active coordinates; conservative states have dimension+2 entries.
     * @details Node arrays use the finalized NodeHalo mapping and store [rho,rho*u...,rho*E].
     * Edge arrays instead borrow Topology ownership so a single integrated flux serves both
     * incident control volumes. Reconstructed point values are separate from evolved dual means.
     */
    template <int dimension>
    class SpatialOperator
    {
        static_assert(dimension == 2 || dimension == 3);

        using State = Eigen::Vector<real, dimension + 2>; ///< Density, momentum components, total energy density.
        using StateGradient = Eigen::Matrix<real, dimension, dimension + 2>; ///< (direction, conserved component).
        /// Conserved-component rows; column i*dimension+j stores d(F_j)/d(x_i).
        using PhysicalFluxGradient =
            Eigen::Matrix<real, dimension + 2, dimension * dimension>;
        using SpatialVector = Eigen::Vector<real, dimension>;

        /**
         * @brief Column packing for the cached vector derivative d(F_j)/d(x_i).
         * @param derivativeDirection Zero-based derivative coordinate i in d(F_j)/d(x_i).
         * @param fluxDirection Zero-based Cartesian physical-flux direction j.
         * @return derivativeDirection*dimension + fluxDirection; conserved components occupy matrix rows.
         */
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
        State _interiorInviscidCorrectionScales = State::Ones();
        bool _scaleInteriorInviscidCorrection = false;
        int _singleZeroInviscidCorrectionComponent = -1;

        NodeMatrixPair _stateGradients;
        NodeMatrixPair _physicalFluxGradients;
        NodeMatrixPair _coefficients;
        NodeStatePair _pointValues;
        // Thesis (3-96): one anchor coefficient multiplies the complete
        // differential-weight correction for that side of every macro surface.
        NodeStatePair _limiterFactors;
        NodeStatePair _edgeFlux;
        NodeMatrixPair _fluxCorrectionBounds;
        NodeStatePair _edgeFluxCorrectionFactors;
        NodeStatePair _edgeSpectralRadius;
        NodeStatePair _localTimeSteps;
        std::unique_ptr<SharedGradientFaceSweep<dimension>> _sharedGradientFaceSweep;
        State _farField = State::Zero();
        real _lastMinimumTimeStep = 0;
        real _lastMaximumTimeStep = 0;
        RhsPhaseTiming _lastRhsTiming;
        // Per-rank counter. EvaluateRHS currently evaluates its edge and
        // boundary loops serially, so no atomic synchronization is needed.
        mutable index _riemannSolverCallCount = 0;

        /**
         * @brief Convert ideal-gas primitive data to the NCFV conservative state layout.
         * @param primitive dimension+2 entries [rho,u...,p] with positive density and pressure.
         * @return State [rho,rho*u...,rho*E], with rho*E = p/(gamma-1) + rho*|u|^2/2.
         */
        State PrimitiveToConservative(const std::vector<real> &primitive) const;
        /**
         * @brief Recover density, velocity, and pressure from ideal-gas conservative data.
         * @param state Conservative state [rho,rho*u...,rho*E] with positive density.
         * @return Primitive state [rho,u...,p] in the same dimension+2-entry vector.
         */
        State ConservativeToPrimitive(const State &state) const;
        /**
         * @brief Recover ideal-gas pressure from conservative energy and momentum.
         * @param state Conservative state [rho,rho*u...,rho*E] with positive density.
         * @return Pressure (gamma-1)*(rho*E - |rho*u|^2/(2*rho)).
         */
        real Pressure(const State &state) const;
        /**
         * @brief Recover ideal-gas temperature from the current conservative state.
         * @param state Conservative state with positive density and pressure.
         * @return Temperature p/(rho*R) using the configured gas constant.
         */
        real Temperature(const State &state) const;
        /**
         * @brief Evaluate the selected constant or Sutherland molecular viscosity law.
         * @param state Conservative state supplying temperature when the viscosity law needs it.
         * @return Dynamic molecular viscosity in the case's consistent units.
         */
        real MolecularViscosity(const State &state) const;
        /**
         * @brief Check component finiteness and admissible ideal-gas density and pressure.
         * @param state Candidate conservative state.
         * @return True if the state passes all physical-state checks.
         */
        bool IsPhysical(const State &state) const;
        /**
         * @brief Blend an inadmissible reconstructed state toward its admissible anchor.
         * @param candidate Reconstructed conservative state to test.
         * @param anchor Physical conservative anchor, normally a mean or recovered node point state.
         * @return Candidate if physical; otherwise an admissible anchor-to-candidate state found by bisection.
         */
        State PreservePhysical(const State &candidate, const State &anchor) const;

        /**
         * @brief Evaluate the Euler flux projected onto a supplied direction.
         * @param state Physical conservative state [rho,rho*u...,rho*E].
         * @param normal Projection direction in active coordinates; flux scales with its length.
         * @return dimension+2 conservative flux entries projected onto normal, without separate surface
         *     integration.
         */
        State PhysicalFlux(const State &state, const SpatialVector &normal) const;
        /**
         * @brief Evaluate one conservative component of the projected Euler flux.
         * @param state Physical conservative state.
         * @param normal Flux-projection direction in active coordinates.
         * @param component Conserved-component index in [0,dimension+2).
         * @return Selected component of PhysicalFlux(state,normal).
         */
        real PhysicalFluxComponent(
            const State &state,
            const SpatialVector &normal,
            int component) const;
        /**
         * @brief Dispatch the configured inviscid Riemann solver and increment its local call counter.
         * @param left Physical conservative trace on the left side.
         * @param right Physical conservative trace on the right side.
         * @param unitNormal Unit normal oriented from the left trace toward the right trace.
         * @return Oriented numerical conservative flux per unit interface measure.
         */
        State NumericalFlux(
            const State &left,
            const State &right,
            const SpatialVector &unitNormal) const;
        /**
         * @brief Construct an exterior conservative trace for one normalized boundary condition.
         * @param inside Admissible conservative trace reconstructed from the interior.
         * @param unitNormal Outward unit normal in the active spatial dimension.
         * @param boundary Normalized per-zone primitive/wall/pressure boundary settings.
         * @return Exterior state paired with the interior trace by the boundary Riemann solve.
         */
        State BoundaryExterior(
            const State &inside,
            const SpatialVector &unitNormal,
            const BoundaryZoneSettings &boundary) const;
        /**
         * @brief Evaluate the inviscid flux for a physical boundary, including any wall specialization.
         * @param inside Admissible conservative trace reconstructed from the interior.
         * @param unitNormal Outward unit normal in the active spatial dimension.
         * @param boundary Normalized per-zone primitive/wall/pressure boundary settings.
         * @return Outward conservative numerical flux per unit boundary measure.
         */
        State BoundaryNumericalFlux(
            const State &inside,
            const SpatialVector &unitNormal,
            const BoundaryZoneSettings &boundary) const;

        /**
         * @brief Evaluate the traditional quadratic reconstruction at a physical integration point.
         * @param anchorNode Anchor node index in the finalized NCFV halo layout.
         * @param point Physical quadrature coordinate mapped relative to the anchor.
         * @return Conservative point trace, with physical-state fallback to the anchor mean when needed.
         */
        State EvaluateTraditionalState(index anchorNode, const Vector3 &point) const;
        /**
         * @brief Evaluate the traditional quadratic reconstruction at a physical integration point.
         * @param anchorNode Anchor node index in the finalized NCFV halo layout.
         * @param point Physical quadrature coordinate mapped relative to the anchor.
         * @return Physical conservative-gradient matrix at point.
         */
        StateGradient EvaluateTraditionalGradient(index anchorNode, const Vector3 &point) const;
        /**
         * @brief Evaluate the efficient differential surface gradient.
         * @param side Zero for the first (left) primal-edge endpoint, one for the second (right) endpoint.
         * @param surface Macro-surface geometry with finalized halo support IDs and precomputed integral weights.
         * @return Mean conservative gradient on the macro-surface, without an oriented normal contraction.
         */
        StateGradient EfficientSurfaceGradient(
            int side,
            const EdgeControlSurface &surface) const;
        /**
         * @brief Average conservative gradients on a boundary micro-surface using precomputed support weights.
         * @param anchorNode Anchor node in the finalized halo layout.
         * @param piece Boundary simplex with normalized integral stencil weights.
         * @return Physical conservative-gradient matrix averaged over the boundary piece.
         */
        StateGradient EfficientBoundarySurfaceGradient(
            index anchorNode,
            const BoundaryPiece &piece) const;
        /**
         * @brief Evaluate the efficient differential physical-flux integral.
         * @param side Zero for the first (left) primal-edge endpoint, one for the second (right) endpoint.
         * @param surface Macro-surface geometry with finalized halo support IDs and precomputed integral weights.
         * @param [out] timing Per-kernel timing record populated when detailed flux timing is enabled.
         * @return Oriented physical-flux integral for the selected side, including differential correction
         *     weights.
         */
        State EfficientIntegratedPhysicalFlux(
            int side,
            const EdgeControlSurface &surface,
            EfficientPhysicalFluxIntegralTiming &timing) const;
        /**
         * @brief Evaluate the efficient differential surface state mean.
         * @param side Zero for the first (left) primal-edge endpoint, one for the second (right) endpoint.
         * @param surface Macro-surface geometry with finalized halo support IDs and precomputed integral weights.
         * @return Conservative macro-surface mean obtained from the anchor point value and differential
         *     correction.
         */
        State EfficientSurfaceMean(
            int side,
            const EdgeControlSurface &surface) const;
        /**
         * @brief Evaluate Newtonian stress and heat flux using corrected two-sided interface gradients.
         * @param left Physical conservative state on the left.
         * @param right Physical conservative state on the right.
         * @param leftGradient Physical conservative-state derivatives, with derivative directions as rows and
         *     conserved components as columns. Left interface reconstruction.
         * @param rightGradient Physical conservative-state derivatives, with derivative directions as rows and
         *     conserved components as columns. Right interface reconstruction.
         * @param unitNormal Left-to-right unit interface normal.
         * @param characteristicDistance Positive normal correction distance used to enforce the state jump.
         * @return Conservative viscous flux per unit measure, with zero mass flux.
         */
        State InternalViscousFlux(
            const State &left,
            const State &right,
            const StateGradient &leftGradient,
            const StateGradient &rightGradient,
            const SpatialVector &unitNormal,
            real characteristicDistance) const;
        /**
         * @brief Apply viscous boundary constraints to the stress/heat-flux interface evaluation.
         * @param inside Physical interior conservative trace.
         * @param insideGradient Physical conservative-state derivatives, with derivative directions as rows and
         *     conserved components as columns.
         * @param unitNormal Outward unit boundary normal.
         * @param boundary Wall/thermal or other normalized boundary settings.
         * @param lengthScale Positive local dual-volume scale for one-sided gradient corrections.
         * @return Outward viscous flux per unit measure, with boundary velocity/thermal constraints applied.
         */
        State BoundaryViscousFlux(
            const State &inside,
            const StateGradient &insideGradient,
            const SpatialVector &unitNormal,
            const BoundaryZoneSettings &boundary,
            real lengthScale) const;
        /**
         * @brief Estimate a surface-integrated inviscid plus optional viscous CFL contribution.
         * @param state Physical conservative state supplying flow, sound speed, and viscosity.
         * @param unitNormal Unit surface-normal direction.
         * @param measure Positive scalar surface length/area.
         * @param lengthScale Positive local volume scale used by the viscous spectral estimate.
         * @return Integrated spectral radius, including the configured viscous diffusivity factor when enabled.
         */
        real SurfaceSpectralRadius(
            const State &state,
            const SpatialVector &unitNormal,
            real measure,
            real lengthScale) const;
        /**
         * @brief Compute the local Lax-Friedrichs wave-speed bound for two admissible means.
         * @param left Left conservative mean.
         * @param right Right conservative mean.
         * @param normal Unit left-to-right normal.
         * @return Maximum of |u dot normal| + sound speed over the two states.
         */
        real LowOrderSpeed(const State &left, const State &right,
                           const SpatialVector &normal) const;
        /**
         * @brief Integrate the local Lax-Friedrichs baseline used for conservative flux correction.
         * @param left Left conservative mean.
         * @param right Right conservative mean.
         * @param normal Unit left-to-right normal.
         * @param measure Positive macro-surface measure.
         * @return Surface-integrated low-order flux with a shared two-state wave-speed bound.
         */
        State LowOrderFlux(const State &left, const State &right,
                           const SpatialVector &normal, real measure) const;
        /**
         * @brief Build and synchronize node bounds from neighboring means and LLF bar states.
         * @param means Conservative dual-volume means [rho,rho*u...,rho*E] in the NCFV node-halo layout. Ghost
         *     entries must already be synchronized.
         */
        void PrepareFluxCorrectionBounds(const NodeStatePair &means);
        /**
         * @brief Blend a high-order edge flux toward LLF using one factor shared by both endpoints.
         * @param iEdge Owned primal-edge index.
         * @param means Conservative dual-volume means [rho,rho*u...,rho*E] in the NCFV node-halo layout.
         * @param highOrder Oriented surface-integrated candidate flux.
         * @return Conservative limited edge integral; stores the accepted edge correction factor.
         * @pre PrepareFluxCorrectionBounds() has populated current node bounds.
         */
        State LimitEdgeFluxCorrection(index iEdge, const NodeStatePair &means,
                                      const State &highOrder);
        /**
         * @brief Evaluate the edge-interface contribution.
         * @param iEdge Owned primal-edge index.
         * @param means Conservative dual-volume means [rho,rho*u...,rho*E] in the NCFV node-halo layout.
         * @return Oriented inviscid-minus-viscous macro-surface flux integral, optionally flux-limited.
         */
        State EvaluateOwnedEdgeFlux(index iEdge, const NodeStatePair &means);
        /**
         * @brief Evaluate the edge-interface contribution.
         * @param iEdge Owned primal-edge index.
         * @param means Conservative dual-volume means [rho,rho*u...,rho*E] in the NCFV node-halo layout.
         * @return Positive integrated CFL spectral-radius contribution for this edge.
         */
        real EvaluateOwnedEdgeSpectralRadius(index iEdge, const NodeStatePair &means) const;
        /**
         * @brief Evaluate the owned-node physical-boundary contribution.
         * @param iNode Owned primal-node index.
         * @param means Conservative dual-volume means [rho,rho*u...,rho*E] in the NCFV node-halo layout.
         * @return Sum of outward inviscid-minus-viscous boundary-piece flux integrals for this node.
         */
        State EvaluateOwnedBoundaryFlux(index iNode, const NodeStatePair &means) const;
        /**
         * @brief Evaluate the owned-node physical-boundary contribution.
         * @param iNode Owned primal-node index.
         * @param means Conservative dual-volume means [rho,rho*u...,rho*E] in the NCFV node-halo layout.
         * @return Sum of boundary-piece spectral-radius contributions for this node.
         */
        real EvaluateOwnedBoundarySpectralRadius(index iNode, const NodeStatePair &means) const;

        /**
         * @brief Allocate owned/ghost node matrices with persistent pulls.
         * @details Borrows the finalized node-halo mapping.
         * @param [out] field Distributed array pair to initialize, resize, and clear to zero.
         * @param name Diagnostic array name passed to InitPair().
         * @param rows Number of scalar entries per vector, or rows per matrix.
         * @param columns Number of columns per matrix; defaults to the conservative state size where specified.
         */
        void AllocateNodeMatrix(
            NodeMatrixPair &field,
            const std::string &name,
            int rows,
            int columns = dimension + 2);
        /**
         * @brief Allocate owned/ghost local-cache matrices without an MPI transformer.
         * @param [out] field Distributed array pair to initialize, resize, and clear to zero.
         * @param name Diagnostic array name passed to InitPair().
         * @param rows Number of scalar entries per vector, or rows per matrix.
         * @param columns Number of columns per matrix; defaults to the conservative state size where specified.
         */
        void AllocateLocalNodeMatrix(
            NodeMatrixPair &field,
            const std::string &name,
            int rows,
            int columns);
        /**
         * @brief Allocate owned/ghost node vectors with persistent pulls.
         * @details Borrows the finalized node-halo mapping.
         * @param [out] field Distributed array pair to initialize, resize, and clear to zero.
         * @param name Diagnostic array name passed to InitPair().
         * @param rows Number of scalar entries per vector, or rows per matrix.
         */
        void AllocateNodeField(
            NodeStatePair &field,
            const std::string &name,
            int rows);
        /**
         * @brief Allocate owned-only node vectors without ghost storage.
         * @param [out] field Distributed array pair to initialize, resize, and clear to zero.
         * @param name Diagnostic array name passed to InitPair().
         * @param rows Number of scalar entries per vector, or rows per matrix.
         */
        void AllocateLocalNodeField(
            NodeStatePair &field,
            const std::string &name,
            int rows);
        /**
         * @brief Allocate owned/ghost edge vectors with persistent pulls.
         * @details Borrows the topology edge owner/ghost mapping.
         * @param [out] field Distributed array pair to initialize, resize, and clear to zero.
         * @param name Diagnostic array name passed to InitPair().
         * @param rows Number of scalar entries per vector, or rows per matrix.
         */
        void AllocateEdgeField(NodeStatePair &field, const std::string &name, int rows);
        /**
         * @brief Prepare synchronized reconstruction gradients, point states, and limiter data.
         * @details Computes coefficients, recovers point values, applies configured limiting, and refreshes
         *     required runtime ghost fields.
         * @param [in,out] means Conservative dual-volume means [rho,rho*u...,rho*E] in the NCFV node-halo layout.
         *     Ghost values are pulled before stencil evaluation.
         */
        void Reconstruct(NodeStatePair &means);
        /**
         * @brief Cache Cartesian Euler-flux derivatives from synchronized point states and conservative gradients.
         * @details Recomputes owned and ghost cache rows locally; columns use
         *     FluxGradientColumn(derivativeDirection,fluxDirection).
         */
        void ComputePhysicalFluxGradients();
        /**
         * @brief Combine edge and boundary spectral radii into fixed/CFL steps and global extrema.
         * @details Applies configured step limits and the additional maximum-step cap; global stepping replaces
         *     local intervals by their MPI minimum.
         * @param means Conservative dual-volume means [rho,rho*u...,rho*E] in the NCFV node-halo layout. Used for
         *     the current boundary spectral contributions.
         */
        void UpdateLocalTimeSteps(const NodeStatePair &means);

    public:
        /**
         * @brief Bind the spatial RHS evaluator to initialized geometry, reconstruction, and boundary data.
         * @param mpi MPI communicator metadata; collective operations must be entered by every participating rank.
         * @param mesh Distributed O1 primal mesh with owned/ghost connectivity prepared for NCFV.
         * @param topology Built primal-edge incidence and edge communication layout.
         * @param geometry Median-dual volumes and oriented macro-surface integration data.
         * @param reconstruction Finalized node reconstruction operators.
         * @param boundaries Normalized zone settings for physical boundary evaluation.
         * @param mode Traditional quadrature or efficient differential flux integration.
         * @param reconstructionSettings Reconstruction/limiter controls, including optional flux correction.
         * @param physics Ideal-gas, Riemann-solver, viscous, and boundary defaults.
         * @param time Fixed/CFL step controls and admissible step bounds.
         * @param nodeHalo Finalized exact node dependency layout borrowed by runtime fields.
         * @note Referenced objects and MPI metadata must outlive this evaluator. Call Initialize() before
         *     EvaluateRHS().
         */
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

        /**
         * @brief Allocate and clear reconstruction, flux, limiter, and time-step work fields with their required
         *     halo mappings.
         */
        void Initialize();
        /**
         * @brief Set an additional time-step cap for subsequent CFL updates.
         * @param step Positive maximum allowed interval, typically the remaining time before a physical end time.
         */
        void SetMaximumStep(real step) { _maximumStep = step; }
        /**
         * @brief Enable intrusive per-kernel interface-flux timing for diagnostics.
         * @param enabled True to record extra internal flux-kernel timings; false to disable this intrusive
         *     instrumentation.
         */
        void EnableDetailedFluxTiming(bool enabled) { _detailedFluxTiming = enabled; }
        /**
         * @brief Scale the interior numerical-minus-central flux by conserved component.
         * @param scales dimension+2 finite factors in [0,1] multiplying the interior numerical-minus-central flux
         *     correction by conserved component.
         * @note Diagnostic control changes interior dissipation; all-one scales select the normal production flux.
         */
        void SetInteriorInviscidCorrectionScalesForDiagnostics(const State &scales)
        {
            DNDS_check_throw_info(scales.allFinite() &&
                                      (scales.array() >= 0).all() &&
                                      (scales.array() <= 1).all(),
                                  "Invalid interior inviscid correction scales");
            _interiorInviscidCorrectionScales = scales;
            _scaleInteriorInviscidCorrection =
                (scales.array() != 1).any();
            _singleZeroInviscidCorrectionComponent = -1;
            for (int component = 0; component < dimension + 2; component++)
            {
                if (scales(component) == 0 &&
                    _singleZeroInviscidCorrectionComponent == -1)
                    _singleZeroInviscidCorrectionComponent = component;
                else if (scales(component) != 1)
                {
                    _singleZeroInviscidCorrectionComponent = -1;
                    break;
                }
            }
        }

        /**
         * @brief Evaluate d(dual mean)/dt; performs all node and edge halo pulls.
         * @return Volume-weighted global RMS residual, or zero when
         *         computeResidualNorm is false.
         * @param [in,out] means Conservative dual-volume means [rho,rho*u...,rho*E] in the NCFV node-halo layout.
         *     Owned values are retained; required ghost values are pulled.
         * @param [out] rhs Allocated derivative field -sum(outward integrated flux)/dual volume, filled on owned
         *     nodes.
         * @param updateTimeSteps Refresh cached fixed/CFL steps when true; false reuses earlier-stage steps.
         * @param computeResidualNorm Reduce and return the global volume-weighted RMS when true; false skips that
         *     reduction and returns zero.
         * @note Collective; refreshes reconstruction, node/edge halos, phase timings, and flux diagnostics.
         */
        real EvaluateRHS(
            NodeStatePair &means,
            NodeStatePair &rhs,
            bool updateTimeSteps = true,
            bool computeResidualNorm = true);

        /**
         *  @brief Apply configured strong inlet/no-slip conditions to owned nodes.
         * @param [in,out] state Conservative node field whose owned inlet/no-slip states are constrained in place;
         *     ghosts are not pulled.
         */
        void ApplyStrongBoundaryConditions(NodeStatePair &state) const;

        /**
         * @brief Read one cached owned-node time step.
         * @param iNode Owned node index; a step update must already have been evaluated.
         * @return Cached local pseudo-time or globally uniform physical interval.
         */
        [[nodiscard]] real LocalTimeStep(index iNode) const
        {
            return _localTimeSteps[iNode](0, 0);
        }
        /**
         * @brief Return global minimum of the most recently updated owned-node time steps.
         * @return Global minimum of the most recently updated owned-node time steps.
         */
        [[nodiscard]] real LastMinimumTimeStep() const { return _lastMinimumTimeStep; }
        /**
         * @brief Return global maximum of the most recently updated owned-node time steps.
         * @return Global maximum of the most recently updated owned-node time steps.
         */
        [[nodiscard]] real LastMaximumTimeStep() const { return _lastMaximumTimeStep; }

        /**
         * @brief Reset the local count of calls to the selected inviscid Riemann solver.
         * @details Clears only this rank's cumulative dispatcher counter; no MPI reduction or flux evaluation is
         *     performed.
         */
        void ResetRiemannSolverCallCount() { _riemannSolverCallCount = 0; }

        /**
         * @brief Return the local count of NumericalFlux dispatcher calls since the last reset.
         * @return Local numerical-flux dispatcher count since the last reset.
         */
        [[nodiscard]] index RiemannSolverCallCount() const
        {
            return _riemannSolverCallCount;
        }

        /**
         * @brief Local phase times for the most recent EvaluateRHS() call.
         * @return Read-only local phase times for the latest EvaluateRHS() call, in seconds.
         */
        [[nodiscard]] const RhsPhaseTiming &LastRhsTiming() const
        {
            return _lastRhsTiming;
        }

        /**
         * @brief Return read-only physical conservative gradients, with derivative directions as rows.
         * @return Read-only physical conservative gradients, with derivative directions as rows.
         */
        [[nodiscard]] const NodeMatrixPair &Gradients() const { return _stateGradients; }
        /**
         * @brief Return read-only cached flux derivatives; conserved rows and FluxGradientColumn(i,j) columns.
         * @return Read-only cached flux derivatives; conserved rows and FluxGradientColumn(i,j) columns.
         */
        [[nodiscard]] const NodeMatrixPair &PhysicalFluxGradients() const
        {
            return _physicalFluxGradients;
        }
        /**
         * @brief Return read-only polynomial reconstruction coefficients stored by mode and conserved component.
         * @return Read-only polynomial reconstruction coefficients stored by mode and conserved component.
         */
        [[nodiscard]] const NodeMatrixPair &Coefficients() const { return _coefficients; }
        /**
         * @brief Return read-only recovered conservative primal-node point states.
         * @return Read-only recovered conservative primal-node point states.
         */
        [[nodiscard]] const NodeStatePair &PointValues() const { return _pointValues; }
        /**
         * @brief Return read-only oriented integrated owned/ghost edge fluxes used to assemble node residuals.
         * @return Read-only oriented integrated owned/ghost edge fluxes used to assemble node residuals.
         */
        [[nodiscard]] const NodeStatePair &EdgeFluxes() const { return _edgeFlux; }
        /**
         * @brief Return read-only scalar edge correction factors from the optional conservative flux limiter.
         * @return Read-only scalar edge correction factors from the optional conservative flux limiter.
         */
        [[nodiscard]] const NodeStatePair &EdgeFluxCorrectionFactors() const { return _edgeFluxCorrectionFactors; }
        /**
         * @brief Return read-only node lower/upper conservative bounds plus pressure-floor rows for flux
         *     correction.
         * @return Read-only node lower/upper conservative bounds plus pressure-floor rows for flux correction.
         */
        [[nodiscard]] const NodeMatrixPair &FluxCorrectionBounds() const { return _fluxCorrectionBounds; }
        /**
         * @brief Return read-only common scalar reconstruction limiter factors per node.
         * @return Read-only common scalar reconstruction limiter factors per node.
         */
        [[nodiscard]] const NodeStatePair &LimiterFactors() const { return _limiterFactors; }
        /**
         * @brief Cumulative shared-gradient factor; side factors remain one in this mode.
         * @param ownedNode Owned primal-node index.
         * @return Latest shared-gradient multiplier, or one when the sweep is not enabled.
         */
        [[nodiscard]] real SharedGradientFactor(index ownedNode) const
        {
            return _sharedGradientFaceSweep
                       ? _sharedGradientFaceSweep->OwnedFactor(ownedNode)
                       : 1.0;
        }
        /**
         * @brief Read diagnostics from the optional shared-gradient face sweep.
         * @return Pointer to read-only most recent sweep diagnostics, or nullptr when the sweep is disabled.
         */
        [[nodiscard]] const typename SharedGradientFaceSweep<dimension>::Diagnostics *
        SharedGradientSweepDiagnostics() const
        {
            return _sharedGradientFaceSweep ? &_sharedGradientFaceSweep->LastDiagnostics() : nullptr;
        }
        /**
         * @brief Return read-only cached scalar owned-node time-step field.
         * @return Read-only cached scalar owned-node time-step field.
         */
        [[nodiscard]] const NodeStatePair &LocalTimeSteps() const { return _localTimeSteps; }
    };

    extern template class SpatialOperator<2>;
    extern template class SpatialOperator<3>;
}
