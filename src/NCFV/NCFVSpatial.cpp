#include "NCFVSpatial.hpp"
#include "NCFVPressureLimiter.hpp"
#include "NCFVTurbulence.hxx"

#include "DNDS/Errors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace DNDS::NCFV
{
    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::PrimitiveToConservative(
        const std::vector<real> &primitive) const
    {
        DNDS_check_throw_info(primitive.size() == static_cast<std::size_t>(StateSize),
                              "NCFV primitive-state size mismatch");
        return Layout::ToConservative(Eigen::Map<const State>(primitive.data()), _physics.gamma,
                                      _physics.turbulence.viscosityReference);
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::ConservativeToPrimitive(const State &state) const
    {
        return Layout::ToPrimitive(state, _physics.gamma, _physics.turbulence.viscosityReference);
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::Pressure(const State &state) const
    {
        return Layout::Pressure(state, _physics.gamma);
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::Temperature(const State &state) const
    {
        return Pressure(state) /
               (state(0) * _physics.viscous.gasConstant);
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::MolecularViscosity(const State &state) const
    {
        const auto &settings = _physics.viscous;
        switch (settings.model)
        {
        case ViscosityModel::Constant:
            return settings.dynamicViscosity;
        case ViscosityModel::Sutherland:
        {
            const real temperature = Temperature(state);
            DNDS_check_throw_info(temperature > 0,
                                  "NCFV Sutherland law received a non-positive temperature");
            const real ratio = temperature / settings.referenceTemperature;
            const real denominator = temperature + settings.sutherlandConstant;
            DNDS_check_throw_info(std::abs(denominator) > verySmallReal,
                                  "NCFV Sutherland denominator is zero");
            return settings.dynamicViscosity * ratio * std::sqrt(ratio) *
                   (settings.referenceTemperature + settings.sutherlandConstant) /
                   denominator;
        }
        case ViscosityModel::DensityProportional:
            return settings.dynamicViscosity * state(0);
        }
        DNDS_check_throw_info(false, "NCFV viscosity model is invalid");
        return 0;
    }

    template <int dimension, int nTurbulence>
    bool SpatialOperator<dimension, nTurbulence>::IsPhysical(const State &state) const
    {
        return Layout::IsPhysical(state, _physics.gamma);
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::PreservePhysical(
        const State &candidate,
        const State &anchor) const
    {
        if (IsPhysical(candidate))
            return candidate;
        DNDS_check_throw_info(
            IsPhysical(anchor),
            fmt::format("NCFV encountered a nonphysical anchor: rho={}, E={}, p={}; candidate rho={}, E={}, p={}",
                        anchor(0), anchor(dimension + 1),
                        anchor(0) != 0 ? Pressure(anchor) : -veryLargeReal,
                        candidate(0), candidate(dimension + 1),
                        candidate(0) != 0 ? Pressure(candidate) : -veryLargeReal));
        if (!candidate.allFinite())
            return anchor;
        real lower = 0;
        real upper = 1;
        for (int iteration = 0; iteration < 40; iteration++)
        {
            const real middle = 0.5 * (lower + upper);
            if (IsPhysical(anchor + middle * (candidate - anchor)))
                lower = middle;
            else
                upper = middle;
        }
        return anchor + (0.95 * lower) * (candidate - anchor);
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::PhysicalFlux(
        const State &state,
        const SpatialVector &normal) const
    {
        State flux = State::Zero();
        const SpatialVector momentum = state.template segment<dimension>(1);
        const SpatialVector velocity = momentum / state(0);
        const real pressure = Pressure(state) + TurbulentPressure(state);
        const real normalVelocity = velocity.dot(normal);
        flux(0) = state(0) * normalVelocity;
        flux.template segment<dimension>(1) = momentum * normalVelocity + pressure * normal;
        flux(dimension + 1) = (state(dimension + 1) + pressure) * normalVelocity;
        if constexpr (nTurbulence > 0)
            flux.template tail<nTurbulence>() = state.template tail<nTurbulence>() * normalVelocity;
        return flux;
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::PhysicalFluxComponent(
        const State &state,
        const SpatialVector &normal,
        int component) const
    {
        const auto momentum = state.template segment<dimension>(1);
        const real normalMomentum = momentum.dot(normal);
        if (component == 0)
            return normalMomentum;
        const real normalVelocity = normalMomentum / state(0);
        if (component >= FlowSize)
            return state(component) * normalVelocity;
        const real pressure = Pressure(state) + TurbulentPressure(state);
        if (component <= dimension)
            return state(component) * normalVelocity +
                   pressure * normal(component - 1);
        return (state(dimension + 1) + pressure) * normalVelocity;
    }

    template <int dimension, int nTurbulence>
    void SpatialOperator<dimension, nTurbulence>::ComputePhysicalFluxGradients()
    {
        DNDS_assert(_mode == IntegrationMode::EfficientDifferential);
        DNDS_assert(_physicalFluxGradients.Size() == _stateGradients.Size());
        DNDS_assert(_physicalFluxGradients.Size() == _pointValues.Size());

        const real gammaMinusOne = _physics.gamma - 1.0;
        for (index iNode = 0; iNode < _physicalFluxGradients.Size(); iNode++)
        {
            const State state = _pointValues[iNode];
            const StateGradient stateGradient = _stateGradients[iNode];
            PhysicalFluxGradient physicalFluxGradient;

            const real density = state(0);
            const real inverseDensity = 1.0 / density;
            const SpatialVector momentum = state.template segment<dimension>(1);
            const SpatialVector velocity = momentum * inverseDensity;
            const real velocitySquared = velocity.squaredNorm();
            const real pressure = gammaMinusOne *
                                  (state(dimension + 1) -
                                   0.5 * density * velocitySquared) + TurbulentPressure(state);
            const real totalEnthalpyDensity = state(dimension + 1) + pressure;

            for (int derivativeDirection = 0;
                 derivativeDirection < dimension; derivativeDirection++)
            {
                const State stateDerivative =
                    stateGradient.row(derivativeDirection).transpose();
                const real densityDerivative = stateDerivative(0);
                const SpatialVector momentumDerivative =
                    stateDerivative.template segment<dimension>(1);
                const real totalEnergyDensityDerivative =
                    stateDerivative(dimension + 1);
                const SpatialVector velocityDerivative =
                    (momentumDerivative - densityDerivative * velocity) *
                    inverseDensity;
                const real pressureDerivative = gammaMinusOne *
                                                (totalEnergyDensityDerivative -
                                                 momentumDerivative.dot(velocity) +
                                                 0.5 * densityDerivative *
                                                     velocitySquared) + TurbulentPressure(stateDerivative);

                for (int fluxDirection = 0;
                     fluxDirection < dimension; fluxDirection++)
                {
                    const int column = FluxGradientColumn(
                        derivativeDirection, fluxDirection);
                    physicalFluxGradient(0, column) =
                        momentumDerivative(fluxDirection);
                    physicalFluxGradient(dimension + 1, column) =
                        (totalEnergyDensityDerivative + pressureDerivative) *
                            velocity(fluxDirection) +
                        totalEnthalpyDensity * velocityDerivative(fluxDirection);
                    if constexpr (nTurbulence > 0)
                        physicalFluxGradient.col(column).template tail<nTurbulence>() =
                            stateDerivative.template tail<nTurbulence>() * velocity(fluxDirection) +
                            state.template tail<nTurbulence>() * velocityDerivative(fluxDirection);
                }

                // d(m_i m_j / rho + p delta_ij) is symmetric in i and j.
                for (int fluxDirection = 0;
                     fluxDirection < dimension; fluxDirection++)
                    for (int momentumDirection = 0;
                         momentumDirection <= fluxDirection; momentumDirection++)
                    {
                        real derivative =
                            momentumDerivative(momentumDirection) *
                                velocity(fluxDirection) +
                            momentumDerivative(fluxDirection) *
                                velocity(momentumDirection) -
                            densityDerivative * velocity(momentumDirection) *
                                velocity(fluxDirection);
                        if (momentumDirection == fluxDirection)
                            derivative += pressureDerivative;
                        physicalFluxGradient(
                            1 + momentumDirection,
                            FluxGradientColumn(
                                derivativeDirection, fluxDirection)) =
                            derivative;
                        physicalFluxGradient(
                            1 + fluxDirection,
                            FluxGradientColumn(
                                derivativeDirection, momentumDirection)) =
                            derivative;
                    }
            }
            _physicalFluxGradients[iNode] = physicalFluxGradient;
        }
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::NumericalFlux(
        const State &left,
        const State &right,
        const SpatialVector &unitNormal) const
    {
        ++_riemannSolverCallCount;
        FlowState gasFlux = FlowState::Zero();
        const FlowState gasLeft = left.template head<FlowSize>();
        const FlowState gasRight = right.template head<FlowSize>();
        SpatialVector gridVelocity = SpatialVector::Zero();
        real acousticMinus = 0;
        real contact = 0;
        real acousticPlus = 0;
        Euler::Gas::InviscidFlux_IdealGas_Dispatcher<dimension>(
            _physics.riemannSolver,
            gasLeft, gasRight, gasLeft, gasRight,
            gridVelocity, unitNormal, _physics.gamma, _physics.gamma, gasFlux,
            0.0, 1.0, 1.0,
            []() {}, acousticMinus, contact, acousticPlus);
        State flux;
        flux.template head<FlowSize>() = gasFlux;
        if constexpr (nTurbulence > 0)
        {
            const State &upwind = flux[0] >= 0 ? left : right;
            flux.template tail<nTurbulence>() = (flux[0] / upwind[0]) * upwind.template tail<nTurbulence>();
        }
        if constexpr (nTurbulence == 2)
        {
            const State mean = .5 * (left + right);
            const real stress = TurbulentPressure(mean);
            flux.template segment<dimension>(1) += stress * unitNormal;
            flux[dimension + 1] += stress * mean.template segment<dimension>(1).dot(unitNormal) / mean[0];
        }
        return flux;
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::LowOrderBoundaryExterior(
        const State &inside, const SpatialVector &unitNormal, const BoundaryZoneSettings &boundary) const
    {
        if (BoundaryRegistry::IsWall(boundary.mode))
        {
            // The LLF update also uses this same ghost in its bar-state bounds and CFL.
            const SpatialVector momentum = inside.template segment<dimension>(1);
            State outside = inside;
            if (boundary.mode == BoundaryMode::SlipWall || boundary.mode == BoundaryMode::Symmetry)
            {
                outside.template segment<dimension>(1) = momentum - 2 * momentum.dot(unitNormal) * unitNormal;
                return outside;
            }
            const SpatialVector reflected = 2 * inside[0] * Eigen::Map<const SpatialVector>(boundary.wallVelocity.data()) - momentum;
            outside.template segment<dimension>(1) = reflected;
            outside[dimension + 1] += .5 * (reflected.squaredNorm() - momentum.squaredNorm()) / inside[0];
            return outside;
        }
        return OpenBoundaryTrace(inside, State::Zero(), unitNormal, boundary, 0, 0).first;
    }

    template <int dimension, int nTurbulence>
    auto SpatialOperator<dimension, nTurbulence>::OpenBoundaryTrace(
        const State &inside, const State &increment, const SpatialVector &normal,
        const BoundaryZoneSettings &boundary, real mmsExponential, real phaseIncrement) const
        -> std::pair<State, State>
    {
        const FlowState flowInside = inside.template head<FlowSize>();
        const FlowState flowIncrement = increment.template head<FlowSize>();
        std::pair<FlowState, FlowState> gasTrace;
        switch (boundary.mode)
        {
        case BoundaryMode::FarField:
        case BoundaryMode::CharacteristicFarField:
        case BoundaryMode::SupersonicInlet:
        case BoundaryMode::SupersonicOutlet:
            if constexpr (dimension == 3 && nTurbulence == 0)
                if (_physics.exponentialMMS)
                    return ExponentialMMS::CharacteristicTrace(inside, increment, normal, mmsExponential, phaseIncrement);
            gasTrace = ReservoirCharacteristicTrace<dimension>(flowInside, flowIncrement, normal, _physics.gamma, boundary.primitive);
            break;
        case BoundaryMode::PressureOutlet:
            gasTrace = PressureOutletCharacteristicTrace<dimension>(flowInside, flowIncrement, normal, _physics.gamma, boundary);
            break;
        case BoundaryMode::Extrapolation:
            return {inside, increment};
        default:
            DNDS_check_throw_info(false, "NCFV open trace requires an inlet, outlet, extrapolation or far-field boundary");
            return {inside, increment};
        }
        if constexpr (nTurbulence == 0)
            return gasTrace;
        else
        {
            State trace, change;
            trace.template head<FlowSize>() = gasTrace.first;
            change.template head<FlowSize>() = gasTrace.second;
            // The turbulence specific variables follow the convected gas family.
            if (inside.template segment<dimension>(1).dot(normal) < 0)
            {
                const Eigen::Map<const Eigen::Vector<real, nTurbulence>> specific(boundary.primitive.data() + FlowSize);
                trace.template tail<nTurbulence>() = trace[0] * InverseTurbulenceScale() * specific;
                change.template tail<nTurbulence>() = change[0] * InverseTurbulenceScale() * specific;
            }
            else
            {
                const Eigen::Vector<real, nTurbulence> specific = inside.template tail<nTurbulence>() / inside[0];
                const Eigen::Vector<real, nTurbulence> variation =
                    (increment.template tail<nTurbulence>() - increment[0] * specific) / inside[0];
                trace.template tail<nTurbulence>() = trace[0] * specific;
                change.template tail<nTurbulence>() = change[0] * specific + trace[0] * variation;
            }
            return {trace, change};
        }
    }

    template <int dimension, int nTurbulence>
    template <bool derivative>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::ReconstructedBoundaryFlux(
        const State &inside, const State &increment, const SpatialVector &normal,
        const BoundaryZoneSettings &boundary, real mmsExponential, real phaseIncrement, real wallSecondary) const
    {
        if (BoundaryRegistry::IsWall(boundary.mode))
        {
            const bool noSlip = BoundaryRegistry::IsNoSlipWall(boundary.mode);
            real wallNormalSpeed = 0;
            if (noSlip)
                wallNormalSpeed = normal.dot(Eigen::Map<const SpatialVector>(boundary.wallVelocity.data()));
            CharacteristicBoundary::Scalar density;
            const auto [gasFlux, gasChange] = WallCharacteristicFlux<dimension>(
                inside.template head<FlowSize>(), increment.template head<FlowSize>(), normal, _physics.gamma,
                wallNormalSpeed, nTurbulence == 2 && !noSlip ? &density : nullptr);
            State flux = State::Zero();
            flux.template head<FlowSize>() = derivative ? gasChange : gasFlux;
            if constexpr (nTurbulence == 2)
            {
                if (noSlip)
                    flux[FlowSize + 1] = flux[0] * wallSecondary;
                else
                {
                    const real k = inside[FlowSize] / inside[0];
                    const real dk = (increment[FlowSize] - k * increment[0]) / inside[0];
                    const real rhoK = derivative ? density.increment * k + density.value * dk : density.value * k;
                    flux.template segment<dimension>(1) += (2.0 / 3.0) * rhoK * normal;
                }
            }
            return flux;
        }
        const auto [trace, change] = OpenBoundaryTrace(inside, increment, normal, boundary, mmsExponential, phaseIncrement);
        if constexpr (derivative)
            return PhysicalFluxIncrement(trace, change, normal);
        else
            return PhysicalFlux(trace, normal);
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::BoundaryPointFlux(const State &inside, const SpatialVector &normal,
                                                 const BoundaryZoneSettings &boundary, real mmsExponential,
                                                 real wallSecondary) const
    {
        if constexpr (dimension == 3 && nTurbulence == 0)
            if (boundary.mode == BoundaryMode::MMSExactExterior)
                return NumericalFlux(inside, ExponentialMMS::State(ExponentialMMS::PowersOf(mmsExponential)), normal);
        return ReconstructedBoundaryFlux<false>(inside, State::Zero(), normal, boundary, mmsExponential, 0, wallSecondary);
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::PhysicalFluxIncrement(const State &state, const State &increment,
                                                     const SpatialVector &normal) const
    {
        const SpatialVector velocity = state.template segment<dimension>(1) / state[0];
        const SpatialVector gridVelocity = SpatialVector::Zero();
        SpatialVector velocityIncrement;
        real pressureIncrement;
        Euler::Gas::IdealGasUIncrement<dimension>(state, increment, velocity, _physics.gamma, velocityIncrement, pressureIncrement);
        State fluxIncrement;
        Euler::Gas::GasInviscidFluxFacialIncrement<dimension>(state, increment, normal, velocity, velocityIncrement,
                                                              gridVelocity, pressureIncrement + TurbulentPressure(increment),
                                                              Pressure(state) + TurbulentPressure(state), fluxIncrement);
        return fluxIncrement;
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::BoundaryMacroInviscidFlux(index node, const BoundaryMacroFace &face,
                                                                     const State &mean, real mmsTimeFactor) const
    {
        const SpatialVector normal = face.normal.template head<dimension>();
        const auto &boundary = _boundaries.Get(face.zone);
        if (boundary.mode == BoundaryMode::Periodic)
            return State::Zero();
        const auto exponential = [&](index support)
        { return _physics.exponentialMMS ? _mmsNodalExponential[support] * mmsTimeFactor : 0; };
        const State anchor = _pointValues[node];

        if constexpr (dimension == 3 && nTurbulence == 0)
            if (boundary.mode == BoundaryMode::MMSExactExterior)
            {
                // Reproduce the two-state P2/Roe control used in the prism study.
                const State exact = ExponentialMMS::State(ExponentialMMS::PowersOf(exponential(node)));
                State leftIntegral = face.measure * anchor, rightIntegral = face.measure * exact;
                State fluxSum = face.measure * (PhysicalFlux(anchor, normal) + PhysicalFlux(exact, normal));
                for (const auto &entry : face.weights)
                {
                    const State left = _pointValues[entry.node];
                    const State dl = _stateGradients[entry.node].transpose() * entry.value;
                    const auto basis = ExponentialMMS::PowersOf(exponential(entry.node));
                    const State right = ExponentialMMS::State(basis);
                    const State dr = ExponentialMMS::direction.dot(entry.value) *
                                     ExponentialMMS::State(ExponentialMMS::PhaseDerivative(basis));
                    leftIntegral += dl;
                    rightIntegral += dr;
                    fluxSum += PhysicalFluxIncrement(left, dl, normal) + PhysicalFluxIncrement(right, dr, normal);
                }
                const State leftMean = PreservePhysical(leftIntegral / face.measure, anchor);
                const State rightMean = PreservePhysical(rightIntegral / face.measure, exact);
                return .5 * fluxSum + face.measure * (NumericalFlux(leftMean, rightMean, normal) -
                                                       .5 * (PhysicalFlux(leftMean, normal) + PhysicalFlux(rightMean, normal)));
            }

        const real wallSecondary = WallSecondaryPrimitive(mean, boundary, BoundaryNormalDistance(node, normal));
        State integral = face.measure * ReconstructedBoundaryFlux<false>(
            anchor, State::Zero(), normal, boundary, exponential(node), 0, wallSecondary);
        for (const auto &entry : face.weights)
        {
            const State increment = _stateGradients[entry.node].transpose() * entry.value.template head<dimension>();
            const real phaseIncrement = _physics.exponentialMMS ? ExponentialMMS::direction.dot(entry.value) : 0;
            integral += ReconstructedBoundaryFlux<true>(
                State(_pointValues[entry.node]), increment, normal, boundary, exponential(entry.node), phaseIncrement, wallSecondary);
        }
        return integral;
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::EvaluateTraditionalState(
        index anchorNode,
        const Vector3 &point) const
    {
        const real volume = _nodeHalo.Volume(anchorNode);
        DNDS_check_throw_info(volume > verySmallReal,
                              "NCFV traditional reconstruction has a zero-volume anchor");
        const Vector3 referenceLengths = _reconstruction.ReferenceLengths(anchorNode);
        State state = _pointValues[anchorNode];
        state += _coefficients[anchorNode].transpose() *
                 Reconstruction::EvaluateBasis(
                     _nodeHalo.DisplacementToPoint(anchorNode, point),
                     referenceLengths, dimension);
        return PreservePhysical(state, State(_pointValues[anchorNode]));
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::StateGradient
    SpatialOperator<dimension, nTurbulence>::EvaluateTraditionalGradient(
        index anchorNode,
        const Vector3 &point) const
    {
        const real volume = _nodeHalo.Volume(anchorNode);
        DNDS_check_throw_info(volume > verySmallReal,
                              "NCFV traditional gradient has a zero-volume anchor");
        const Vector3 referenceLengths = _reconstruction.ReferenceLengths(anchorNode);
        return Reconstruction::EvaluateBasisGradient(
                   _nodeHalo.DisplacementToPoint(anchorNode, point),
                   referenceLengths, dimension) *
               _coefficients[anchorNode];
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::StateGradient
    SpatialOperator<dimension, nTurbulence>::EfficientSurfaceGradient(
        int side,
        const EdgeControlSurface &surface) const
    {
        DNDS_check_throw_info(side == 0 || side == 1,
                              "NCFV efficient surface side must be zero or one");
        DNDS_check_throw_info(surface.measure > verySmallReal,
                              "NCFV efficient surface has zero measure");
        StateGradient gradient = StateGradient::Zero();
        for (const EfficientSurfaceNode &entry : surface.efficientStencil)
            gradient += entry.gradientWeight *
                        StateGradient(_stateGradients[entry.node]);
        const index anchorNode =
            surface.nodes[static_cast<std::size_t>(side)];
        return _limiterFactors[anchorNode](0, 0) *
               gradient / surface.measure;
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::StateGradient
    SpatialOperator<dimension, nTurbulence>::EfficientBoundarySurfaceGradient(
        index anchorNode,
        const BoundaryPiece &piece) const
    {
        DNDS_check_throw_info(piece.measure > verySmallReal,
                              "NCFV efficient boundary surface has zero measure");
        StateGradient gradient = StateGradient::Zero();
        for (const EfficientBoundaryNode &entry : piece.efficientStencil)
            gradient += entry.integrationWeight *
                        StateGradient(_stateGradients[entry.node]);
        return _limiterFactors[anchorNode](0, 0) *
               gradient / piece.measure;
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::EfficientIntegratedPhysicalFlux(
        int side,
        const EdgeControlSurface &surface,
        EfficientPhysicalFluxIntegralTiming &timing) const
    {
        DNDS_check_throw_info(side == 0 || side == 1,
                              "NCFV efficient surface side must be zero or one");
        const std::size_t sideIndex = static_cast<std::size_t>(side);
        const index anchorNode = surface.nodes[sideIndex];
        double phaseStart = 0;
        if (_detailedFluxTiming)
            phaseStart = MPI_Wtime();
        State integral = PhysicalFlux(
            State(_pointValues[anchorNode]),
            surface.vectorMeasure.template head<dimension>());
        if (_detailedFluxTiming)
        {
            timing.zeroOrderFluxSeconds += MPI_Wtime() - phaseStart;
            phaseStart = MPI_Wtime();
        }
        if (_reconstructionSettings.barthZeroOrderFallback &&
            _limiterFactors[anchorNode](0, 0) == 0)
            return integral;
        State correction = State::Zero();
        for (const EfficientSurfaceNode &entry : surface.efficientStencil)
        {
            for (int derivativeDirection = 0; derivativeDirection < dimension;
                 derivativeDirection++)
                for (int fluxDirection = 0; fluxDirection < dimension; fluxDirection++)
                    correction +=
                        entry.fluxGradientWeights[sideIndex](
                            derivativeDirection, fluxDirection) *
                        _physicalFluxGradients[entry.node]
                            .col(FluxGradientColumn(
                                derivativeDirection, fluxDirection));
        }
        if (_detailedFluxTiming)
        {
            timing.precomputedGradientIntegralSeconds +=
                MPI_Wtime() - phaseStart;
            phaseStart = MPI_Wtime();
        }
        const State result = integral +
                             _limiterFactors[anchorNode](0, 0) * correction;
        if (_detailedFluxTiming)
            timing.finalAssemblySeconds += MPI_Wtime() - phaseStart;
        return result;
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::EfficientSurfaceMean(
        int side,
        const EdgeControlSurface &surface) const
    {
        DNDS_check_throw_info(side == 0 || side == 1,
                              "NCFV efficient surface side must be zero or one");
        DNDS_check_throw_info(surface.measure > verySmallReal,
                              "NCFV efficient surface has zero measure");
        const std::size_t sideIndex = static_cast<std::size_t>(side);
        const index anchorNode = surface.nodes[sideIndex];
        if (_reconstructionSettings.barthZeroOrderFallback &&
            _limiterFactors[anchorNode](0, 0) == 0)
            return State(_pointValues[anchorNode]);
        State integral = surface.measure * State(_pointValues[anchorNode]);
        State correction = State::Zero();
        for (const EfficientSurfaceNode &entry : surface.efficientStencil)
        {
            correction += _stateGradients[entry.node].transpose() *
                          entry.stateGradientWeights[sideIndex]
                              .template head<dimension>();
        }
        // One coefficient for the complete side reconstruction is required by
        // thesis (3-96); scaling each contributing nodal gradient by its own
        // coefficient would define a different interface mean.
        integral += _limiterFactors[anchorNode](0, 0) * correction;
        const State mean = integral / surface.measure;
        return PreservePhysical(mean, State(_pointValues[anchorNode]));
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::InternalViscousFlux(
        const State &left,
        const State &right,
        const StateGradient &leftGradient,
        const StateGradient &rightGradient,
        const SpatialVector &unitNormal,
        real characteristicDistance, real wallDistance) const
    {
        if (!_physics.viscous.enabled)
            return State::Zero();

        // Thesis (4-153)/(4-156) uses the arithmetic mean of the primitive
        // variables q, not the primitive state obtained from an arithmetic
        // mean of conservative variables.
        const State facePrimitive =
            0.5 * (ConservativeToPrimitive(left) +
                   ConservativeToPrimitive(right));
        const State faceState = Layout::ToConservative(facePrimitive, _physics.gamma, _physics.turbulence.viscosityReference);
        const real distance = std::max(characteristicDistance, verySmallReal);
        // Conservative dGRP gradient, thesis (4-154)/(4-170).  The 1/2 in
        // both the averaged gradients and jump correction is essential.
        StateGradient conservativeGradient = 0.5 * (leftGradient + rightGradient);
        conservativeGradient +=
            unitNormal * ((right - left).transpose() / (2.0 * distance));
        StateGradient primitiveGradient;
        Euler::Gas::GradientCons2Prim_IdealGas<dimension>(
            faceState, conservativeGradient, primitiveGradient, _physics.gamma,
            Eigen::Vector<real, 0>{});

        return ViscousTraceFlux(faceState, primitiveGradient, unitNormal, false, wallDistance);
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::BoundaryViscousFlux(
        const State &inside,
        const StateGradient &insideGradient,
        const SpatialVector &unitNormal,
        const BoundaryZoneSettings &boundary,
        real lengthScale, real normalDistance, real wallDistance, real wallSecondary) const
    {
        if (!_physics.viscous.enabled ||
            boundary.mode == BoundaryMode::SlipWall ||
            boundary.mode == BoundaryMode::Symmetry)
            return State::Zero();

        StateGradient primitiveGradient;
        Euler::Gas::GradientCons2Prim_IdealGas<dimension>(
            inside, insideGradient, primitiveGradient, _physics.gamma,
            Eigen::Vector<real, 0>{});
        const bool noSlip = BoundaryRegistry::IsNoSlipWall(boundary.mode);
        const bool adiabatic = boundary.mode == BoundaryMode::NoSlipAdiabaticWall;
        if constexpr (nTurbulence == 0)
            if (!noSlip)
                return ViscousTraceFlux(inside, primitiveGradient, unitNormal, adiabatic, wallDistance);
        const State insidePrimitive = ConservativeToPrimitive(inside);
        State trace = insidePrimitive;
        if (noSlip)
        {
            trace.template segment<dimension>(1) = Eigen::Map<const SpatialVector>(boundary.wallVelocity.data());
            if (boundary.mode == BoundaryMode::NoSlipIsothermalWall)
                trace(dimension + 1) = trace[0] * _physics.viscous.gasConstant * boundary.wallTemperature;
            const real correctionDistance = std::max(0.5 * lengthScale, verySmallReal);
            primitiveGradient.template leftCols<FlowSize>() += unitNormal *
                ((trace - insidePrimitive).template head<FlowSize>().transpose() / correctionDistance);
        }
        if constexpr (nTurbulence > 0)
        {
            const bool inflow = !noSlip && boundary.mode != BoundaryMode::Extrapolation &&
                                inside.template segment<dimension>(1).dot(unitNormal) < 0;
            auto scalarGradient = primitiveGradient.template rightCols<nTurbulence>();
            if (noSlip || inflow)
            {
                if (noSlip)
                {
                    trace[FlowSize] = 0;
                    if constexpr (nTurbulence == 2)
                        trace[FlowSize + 1] = wallSecondary;
                }
                else
                    trace.template tail<nTurbulence>() =
                        Eigen::Map<const Eigen::Vector<real, nTurbulence>>(boundary.primitive.data() + FlowSize);
                scalarGradient += unitNormal * (InverseTurbulenceScale() / normalDistance) *
                                  (trace - insidePrimitive).template tail<nTurbulence>().transpose();
            }
            else if (boundary.mode != BoundaryMode::Extrapolation)
            {
                // Homogeneous normal diffusion for outgoing characteristics.
                const Eigen::RowVector<real, nTurbulence> normalDerivative = unitNormal.transpose() * scalarGradient;
                scalarGradient -= unitNormal * normalDerivative;
            }
        }
        return ViscousTraceFlux(Layout::ToConservative(trace, _physics.gamma, _physics.turbulence.viscosityReference),
                                primitiveGradient, unitNormal, adiabatic, wallDistance, noSlip);
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::SurfaceSpectralRadius(
        const State &state,
        real kinematicDiffusivity,
        const SpatialVector &unitNormal,
        real measure,
        real lengthScale) const
    {
        const SpatialVector velocity =
            state.template segment<dimension>(1) / state(0);
        const real acousticSpeed =
            std::sqrt(_physics.gamma * (Pressure(state) + TurbulentPressure(state)) / state(0));
        real spectralRadius =
            (std::abs(velocity.dot(unitNormal)) + acousticSpeed) * measure;
        if (_physics.viscous.enabled)
        {
            spectralRadius += _physics.viscous.spectralRadiusFactor *
                              kinematicDiffusivity * measure /
                              std::max(lengthScale, verySmallReal);
        }
        return spectralRadius;
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::LowOrderSpeed(
        const State &left, const State &right, const SpatialVector &normal) const
    {
        const auto speed = [&](const State &state)
        {
            return std::abs(state.template segment<dimension>(1).dot(normal) / state(0)) +
                   std::sqrt(_physics.gamma * Pressure(state) / state(0));
        };
        return std::max(speed(left), speed(right));
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::LowOrderFlux(
        const State &left, const State &right, const SpatialVector &normal, real measure) const
    {
        return .5 * measure * (PhysicalFlux(left, normal) + PhysicalFlux(right, normal) -
                                LowOrderSpeed(left, right, normal) * (right - left));
    }

    template <int dimension, int nTurbulence>
    void SpatialOperator<dimension, nTurbulence>::PrepareFluxCorrectionBounds(const NodeStatePair &means)
    {
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const State self = means[iNode];
            State minimum = self, maximum = self;
            real pressureMinimum = Pressure(self);
            real entropyMinimum = pressureMinimum / std::pow(self(0), _physics.gamma);
            const auto include = [&](const State &state)
            {
                DNDS_check_throw_info(IsPhysical(state), "NCFV LLF bar state must be physical");
                minimum = minimum.cwiseMin(state);
                maximum = maximum.cwiseMax(state);
                pressureMinimum = std::min(pressureMinimum, Pressure(state));
                entropyMinimum = std::min(entropyMinimum, Pressure(state) / std::pow(state(0), _physics.gamma));
            };
            for (const auto &incidence : _topology.Node2Edge(iNode))
            {
                const auto &surface = _geometry.EdgeSurface(incidence.edge);
                const bool left = surface.nodes[0] == iNode;
                const index neighbor = surface.nodes[left ? 1 : 0];
                const SpatialVector normal = (left ? 1.0 : -1.0) *
                                             surface.vectorMeasure.template head<dimension>() / surface.measure;
                const State other = means[neighbor];
                const real speed = LowOrderSpeed(self, other, normal);
                include(other);
                include(.5 * (self + other) -
                        .5 / speed * (PhysicalFlux(other, normal) - PhysicalFlux(self, normal)));
            }
            for (const auto &piece : _geometry.NodeVolume(iNode).boundaryPieces)
            {
                const auto &boundary = _boundaries.Get(piece.zone);
                if (boundary.mode == BoundaryMode::Periodic)
                    continue;
                const SpatialVector normal = piece.vectorMeasure.template head<dimension>() / piece.measure;
                const State other = LowOrderBoundaryExterior(self, normal, boundary);
                const real speed = LowOrderSpeed(self, other, normal);
                include(other);
                include(.5 * (self + other) -
                        .5 / speed * (PhysicalFlux(other, normal) - PhysicalFlux(self, normal)));
            }
            _fluxCorrectionBounds[iNode].template topRows<StateSize>().col(0) = minimum;
            _fluxCorrectionBounds[iNode].template topRows<StateSize>().col(1) = maximum;
            _fluxCorrectionBounds[iNode](StateSize, 0) = pressureMinimum;
            _fluxCorrectionBounds[iNode](StateSize, 1) = pressureMinimum;
            _fluxCorrectionBounds[iNode](StateSize + 1, 0) = entropyMinimum;
            _fluxCorrectionBounds[iNode](StateSize + 1, 1) = entropyMinimum;
        }
        _fluxCorrectionBounds.trans.startPersistentPull();
        _fluxCorrectionBounds.trans.waitPersistentPull();
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::LimitEdgeFluxCorrection(
        index iEdge, const NodeStatePair &means, const State &highOrder)
    {
        const auto &surface = _geometry.EdgeSurface(iEdge);
        const State left = means[surface.nodes[0]], right = means[surface.nodes[1]];
        const SpatialVector normal = surface.vectorMeasure.template head<dimension>() / surface.measure;
        const real speed = LowOrderSpeed(left, right, normal);
        const real dissipation = speed * surface.measure;
        const State lowOrder = LowOrderFlux(left, right, normal, surface.measure);
        const State correction = highOrder - lowOrder;
        const State bar = .5 * (left + right) -
                          .5 / speed * (PhysicalFlux(right, normal) - PhysicalFlux(left, normal));
        real factor = _reconstructionSettings.maximumFluxCorrection;
        for (int side = 0; side < 2; side++)
        {
            const index node = surface.nodes[side];
            const State increment = (side == 0 ? -1.0 : 1.0) * correction / dissipation;
            Eigen::Matrix<real, StateSize + 2, 2> bounds = _fluxCorrectionBounds[node];
            if (_reconstructionSettings.fluxCorrectionPairBounds)
            {
                bounds.template topRows<StateSize>().col(0) = left.cwiseMin(right).cwiseMin(bar);
                bounds.template topRows<StateSize>().col(1) = left.cwiseMax(right).cwiseMax(bar);
                bounds(StateSize, 0) = std::min({Pressure(left), Pressure(right), Pressure(bar)});
                bounds(StateSize + 1, 0) = std::min({Pressure(left) / std::pow(left(0), _physics.gamma),
                                                   Pressure(right) / std::pow(right(0), _physics.gamma),
                                                   Pressure(bar) / std::pow(bar(0), _physics.gamma)});
            }
            for (int component = 0; component < StateSize; component++)
            {
                const real padding = 64 * std::numeric_limits<real>::epsilon() *
                                     std::max({1.0, std::abs(bounds(component, 0)), std::abs(bounds(component, 1))});
                if (increment(component) > 0)
                    factor = std::min(factor, (bounds(component, 1) + padding - bar(component)) / increment(component));
                else if (increment(component) < 0)
                    factor = std::min(factor, (bounds(component, 0) - padding - bar(component)) / increment(component));
            }
            factor = std::clamp(factor, 0.0, _reconstructionSettings.maximumFluxCorrection);
            // Pressure is concave along a conservative segment. Its lower
            // superlevel set is convex; an upper pressure box is not.
            const real pressureMinimum = bounds(StateSize, 0);
            const real pressurePadding = 64 * std::numeric_limits<real>::epsilon() *
                                         std::max(1.0, std::abs(pressureMinimum));
            const auto admissible = [&](real theta)
            {
                const State candidate = bar + theta * increment;
                return IsPhysical(candidate) && Pressure(candidate) >= pressureMinimum - pressurePadding &&
                       (!_reconstructionSettings.fluxCorrectionEntropy ||
                        Pressure(candidate) >= bounds(StateSize + 1, 0) * std::pow(candidate(0), _physics.gamma) - pressurePadding);
            };
            if (!admissible(factor))
            {
                real lower = 0, upper = factor;
                for (int iteration = 0; iteration < 48; iteration++)
                {
                    const real middle = .5 * (lower + upper);
                    if (admissible(middle))
                        lower = middle;
                    else
                        upper = middle;
                }
                factor = lower;
            }
        }
        if (_reconstructionSettings.fluxCorrectionBinary && factor < _reconstructionSettings.maximumFluxCorrection)
            factor = 0;
        _edgeFluxCorrectionFactors[iEdge](0) = factor;
        // A single owner-computed flux is shared by both incident volumes.
        return lowOrder + factor * correction;
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::EvaluateOwnedEdgeFlux(index iEdge, const NodeStatePair &means)
    {
        const auto &surface = _geometry.EdgeSurface(iEdge);
        const index node0 = surface.nodes[0];
        const index node1 = surface.nodes[1];
        const SpatialVector unitNormal =
            surface.vectorMeasure.template head<dimension>().normalized();
        const real volume0 = _nodeHalo.Volume(node0);
        const real volume1 = _nodeHalo.Volume(node1);
        // Thesis (4-155), not the projected primal-edge length.
        const real characteristicDistance =
            std::min(volume0, volume1) / surface.measure;
        real wallDistance = 1;
        if constexpr (nTurbulence > 0)
            wallDistance = .5 * (_turbulenceGeometry[node0](0) + _turbulenceGeometry[node1](0));

        if (_mode == IntegrationMode::TraditionalQuadrature)
        {
            State integral = State::Zero();
            for (const auto &point : surface.quadrature)
            {
                const SpatialVector pointNormal =
                    point.vectorWeight.template head<dimension>() / point.weight;
                double phaseStart = 0;
                if (_detailedFluxTiming)
                    phaseStart = MPI_Wtime();
                const State left = EvaluateTraditionalState(node0, point.coordinate);
                if (_detailedFluxTiming)
                {
                    _lastRhsTiming.edgeLeftStatePreparationSeconds +=
                        MPI_Wtime() - phaseStart;
                    phaseStart = MPI_Wtime();
                }
                const State right = EvaluateTraditionalState(node1, point.coordinate);
                if (_detailedFluxTiming)
                {
                    _lastRhsTiming.edgeRightStatePreparationSeconds +=
                        MPI_Wtime() - phaseStart;
                    phaseStart = MPI_Wtime();
                }
                State flux = NumericalFlux(left, right, pointNormal);
                if (_singleZeroInviscidCorrectionComponent >= 0)
                {
                    const int component = _singleZeroInviscidCorrectionComponent;
                    flux(component) = 0.5 *
                        (PhysicalFluxComponent(left, pointNormal, component) +
                         PhysicalFluxComponent(right, pointNormal, component));
                }
                else if (_scaleInteriorInviscidCorrection)
                {
                    const State central = 0.5 *
                        (PhysicalFlux(left, pointNormal) +
                         PhysicalFlux(right, pointNormal));
                    flux = central + _interiorInviscidCorrectionScales.cwiseProduct(
                                         flux - central);
                }
                if (_physics.viscous.enabled)
                    flux -= InternalViscousFlux(
                        left, right,
                        EvaluateTraditionalGradient(node0, point.coordinate),
                        EvaluateTraditionalGradient(node1, point.coordinate),
                        pointNormal, characteristicDistance, wallDistance);
                integral += point.weight * flux;
                if (_detailedFluxTiming)
                    _lastRhsTiming.edgeNumericalFluxAndAssemblySeconds +=
                        MPI_Wtime() - phaseStart;
            }
            return integral;
        }

        double phaseStart = 0;
        if (_detailedFluxTiming)
            phaseStart = MPI_Wtime();
        const State leftIntegral = EfficientIntegratedPhysicalFlux(
            0, surface,
            _lastRhsTiming.edgeLeftPhysicalFluxDetail);
        if (_detailedFluxTiming)
        {
            _lastRhsTiming.edgeLeftPhysicalFluxIntegralSeconds +=
                MPI_Wtime() - phaseStart;
            phaseStart = MPI_Wtime();
        }
        const State rightIntegral = EfficientIntegratedPhysicalFlux(
            1, surface,
            _lastRhsTiming.edgeRightPhysicalFluxDetail);
        if (_detailedFluxTiming)
        {
            _lastRhsTiming.edgeRightPhysicalFluxIntegralSeconds +=
                MPI_Wtime() - phaseStart;
            phaseStart = MPI_Wtime();
        }
        const State leftMean = EfficientSurfaceMean(0, surface);
        if (_detailedFluxTiming)
        {
            _lastRhsTiming.edgeLeftStatePreparationSeconds +=
                MPI_Wtime() - phaseStart;
            phaseStart = MPI_Wtime();
        }
        const State rightMean = EfficientSurfaceMean(1, surface);
        if (_detailedFluxTiming)
        {
            _lastRhsTiming.edgeRightStatePreparationSeconds +=
                MPI_Wtime() - phaseStart;
            phaseStart = MPI_Wtime();
        }
        const double numericalBlockStart = phaseStart;
        const State numerical = NumericalFlux(leftMean, rightMean, unitNormal);
        if (_detailedFluxTiming)
        {
            _lastRhsTiming.edgeRiemannFluxSeconds += MPI_Wtime() - phaseStart;
            phaseStart = MPI_Wtime();
        }
        const State centralMean = 0.5 *
                                  (PhysicalFlux(leftMean, unitNormal) +
                                   PhysicalFlux(rightMean, unitNormal));
        if (_detailedFluxTiming)
        {
            _lastRhsTiming.edgeCentralFluxSeconds += MPI_Wtime() - phaseStart;
            phaseStart = MPI_Wtime();
        }
        State dissipativeCorrection =
            surface.measure * (numerical - centralMean);
        if (_scaleInteriorInviscidCorrection)
            dissipativeCorrection = _interiorInviscidCorrectionScales.cwiseProduct(
                dissipativeCorrection);
        if (_detailedFluxTiming)
        {
            _lastRhsTiming.edgeDissipationAssemblySeconds += MPI_Wtime() - phaseStart;
            phaseStart = MPI_Wtime();
        }
        State integral = 0.5 * (leftIntegral + rightIntegral) +
                         dissipativeCorrection;
        if (_detailedFluxTiming)
        {
            _lastRhsTiming.edgeInviscidFinalAssemblySeconds +=
                MPI_Wtime() - phaseStart;
            _lastRhsTiming.edgeNumericalFluxAndAssemblySeconds +=
                MPI_Wtime() - numericalBlockStart;
        }
        if (_physics.viscous.enabled)
        {
            if (_detailedFluxTiming)
                phaseStart = MPI_Wtime();
            const StateGradient leftSurfaceGradient = EfficientSurfaceGradient(
                0, surface);
            const StateGradient rightSurfaceGradient = EfficientSurfaceGradient(
                1, surface);
            integral -= surface.measure *
                        InternalViscousFlux(
                            leftMean, rightMean,
                            leftSurfaceGradient, rightSurfaceGradient,
                            unitNormal, characteristicDistance, wallDistance);
            if (_detailedFluxTiming)
                _lastRhsTiming.edgeViscousFluxSeconds +=
                    MPI_Wtime() - phaseStart;
        }
        if (_reconstructionSettings.limitFluxCorrection)
            integral = LimitEdgeFluxCorrection(iEdge, means, integral);
        return integral;
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::EvaluateOwnedEdgeSpectralRadius(index iEdge, const NodeStatePair &means) const
    {
        const auto &surface = _geometry.EdgeSurface(iEdge);
        if (_reconstructionSettings.limitFluxCorrection)
            return surface.measure * LowOrderSpeed(
                       State(means[surface.nodes[0]]), State(means[surface.nodes[1]]),
                       surface.vectorMeasure.template head<dimension>() / surface.measure);
        const State mean = PreservePhysical(
            0.5 * (State(_pointValues[surface.nodes[0]]) +
                   State(_pointValues[surface.nodes[1]])),
            State(_pointValues[surface.nodes[0]]));
        const SpatialVector normal =
            surface.vectorMeasure.template head<dimension>().normalized();
        const real length0 = std::pow(
            _nodeHalo.Volume(surface.nodes[0]),
            1.0 / static_cast<real>(dimension));
        const real length1 = std::pow(
            _nodeHalo.Volume(surface.nodes[1]),
            1.0 / static_cast<real>(dimension));
        real length = std::min(length0, length1), wallDistance = 1;
        StateGradient gradient = StateGradient::Zero();
        if constexpr (nTurbulence > 0)
        {
            length = std::min(length, std::min(_nodeHalo.Volume(surface.nodes[0]), _nodeHalo.Volume(surface.nodes[1])) / surface.measure);
            wallDistance = .5 * (_turbulenceGeometry[surface.nodes[0]](0) + _turbulenceGeometry[surface.nodes[1]](0));
            gradient = .5 * (NodalGradient(surface.nodes[0]) + NodalGradient(surface.nodes[1]));
        }
        return SurfaceSpectralRadius(mean, MaximumDiffusivity(mean, gradient, wallDistance), normal, surface.measure, length);
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::EvaluateOwnedBoundaryFlux(index iNode, const NodeStatePair &means, real mmsTimeFactor) const
    {
        State total = State::Zero();
        const auto &controlVolume = _geometry.NodeVolume(iNode);
        if (controlVolume.boundaryPieces.empty())
            return total;
        real wallDistance = 1;
        if constexpr (nTurbulence > 0)
            wallDistance = _turbulenceGeometry[iNode](0);
        const auto nodalFlux = [&](const State &inside, const SpatialVector &normal,
                                   const BoundaryZoneSettings &boundary, index node, real wallSecondary)
        {
            const real exponential = _physics.exponentialMMS ? _mmsNodalExponential[node] * mmsTimeFactor : 0;
            return BoundaryPointFlux(inside, normal, boundary, exponential, wallSecondary);
        };
        if (_reconstructionSettings.limitFluxCorrection)
        {
            const State inside = means[iNode];
            for (const auto &piece : controlVolume.boundaryPieces)
            {
                const auto &boundary = _boundaries.Get(piece.zone);
                if (boundary.mode == BoundaryMode::Periodic)
                    continue;
                const SpatialVector normal = piece.vectorMeasure.template head<dimension>() / piece.measure;
                total += LowOrderFlux(inside, LowOrderBoundaryExterior(inside, normal, boundary), normal, piece.measure);
            }
            return total;
        }
        if (_mode == IntegrationMode::TraditionalQuadrature)
        {
            for (const auto &piece : controlVolume.boundaryPieces)
            {
                const auto &boundary = _boundaries.Get(piece.zone);
                if (boundary.mode == BoundaryMode::Periodic)
                    continue;
                const SpatialVector pieceNormal = piece.vectorMeasure.template head<dimension>() / piece.measure;
                const real normalDistance = BoundaryNormalDistance(iNode, pieceNormal);
                const real wallSecondary = WallSecondaryPrimitive(State(means[iNode]), boundary, normalDistance);
                for (const auto &point : piece.quadrature)
                {
                    const SpatialVector unitNormal =
                        point.vectorWeight.template head<dimension>() / point.weight;
                    const State inside = EvaluateTraditionalState(iNode, point.coordinate);
                    const real exponential = _physics.exponentialMMS
                        ? std::exp(ExponentialMMS::direction.dot(point.coordinate)) * mmsTimeFactor : 0;
                    State flux = BoundaryPointFlux(inside, unitNormal, boundary, exponential, wallSecondary);
                    if (_physics.viscous.enabled)
                        flux -= BoundaryViscousFlux(
                            inside,
                            EvaluateTraditionalGradient(iNode, point.coordinate),
                            unitNormal, boundary, controlVolume.lengthScale, normalDistance, wallDistance, wallSecondary);
                    total += point.weight * flux;
                }
            }
            return total;
        }

        const bool macroFaceBoundary = _geometry.UsesMacroFaceBoundary();
        if (macroFaceBoundary)
        {
            for (const auto &face : _boundaryMacroFaces.at(iNode))
                total += BoundaryMacroInviscidFlux(iNode, face, State(means[iNode]), mmsTimeFactor);
            if (!_physics.viscous.enabled)
                return total;
        }

        // No Gaussian points: the initialization-built stencil already merges
        // every construction-point support into one sorted local/ghost node set.
        State boundaryMinimum = means[iNode], boundaryMaximum = boundaryMinimum;
        if (_reconstructionSettings.barthZeroOrderFallback)
            for (const auto &incidence : _topology.Node2Edge(iNode))
            {
                const auto &surface = _geometry.EdgeSurface(incidence.edge);
                const index neighbor = surface.nodes[surface.nodes[0] == iNode ? 1 : 0];
                boundaryMinimum = boundaryMinimum.cwiseMin(State(means[neighbor]));
                boundaryMaximum = boundaryMaximum.cwiseMax(State(means[neighbor]));
            }
        for (const auto &piece : controlVolume.boundaryPieces)
        {
            const auto &boundary = _boundaries.Get(piece.zone);
            if (boundary.mode == BoundaryMode::Periodic)
                continue;
            if (macroFaceBoundary && (boundary.mode == BoundaryMode::SlipWall || boundary.mode == BoundaryMode::Symmetry))
                continue; // These pieces have zero viscous flux.
            const SpatialVector unitNormal =
                piece.vectorMeasure.template head<dimension>() / piece.measure;
            const real normalDistance = BoundaryNormalDistance(iNode, unitNormal);
            const real wallSecondary = WallSecondaryPrimitive(State(means[iNode]), boundary, normalDistance);
            if (_reconstructionSettings.barthZeroOrderFallback &&
                _limiterFactors[iNode](0, 0) == 0)
            {
                total += piece.measure * nodalFlux(State(means[iNode]), unitNormal, boundary, iNode, wallSecondary);
                continue;
            }
            State insideIntegral = State::Zero();
            State inviscidIntegral = State::Zero();
            for (const EfficientBoundaryNode &entry : piece.efficientStencil)
            {
                const State nodalState = _pointValues[entry.node];
                insideIntegral += entry.integrationWeight * nodalState;
                // Thesis section 4.3.2 interpolates already evaluated nodal
                // numerical fluxes, so the same precomputed value weight is
                // applied after evaluating the numerical flux.
                if (!macroFaceBoundary)
                    inviscidIntegral += entry.integrationWeight *
                                        nodalFlux(nodalState, unitNormal, boundary, entry.node, wallSecondary);
            }
            const State candidate = insideIntegral / piece.measure;
            if (_reconstructionSettings.barthZeroOrderFallback &&
                (!IsPhysical(candidate) ||
                 (candidate.array() < boundaryMinimum.array() - 1e-10).any() ||
                 (candidate.array() > boundaryMaximum.array() + 1e-10).any()))
            {
                // Neighboring recovered points may have changed after the
                // initial Barth test. Check the actual boundary mean once
                // more and use a constant side when it is outside the bounds.
                total += piece.measure * nodalFlux(State(means[iNode]), unitNormal, boundary, iNode, wallSecondary);
                continue;
            }
            const State insideMean = PreservePhysical(candidate, State(_pointValues[iNode]));
            total += inviscidIntegral;
            if (_physics.viscous.enabled)
            {
                // Wall stresses and heat flux use each original piece's interior trace and normal.
                total -= piece.measure * BoundaryViscousFlux(
                                             insideMean,
                                             EfficientBoundarySurfaceGradient(
                                                 iNode, piece),
                                             unitNormal, boundary,
                                             controlVolume.lengthScale, normalDistance, wallDistance, wallSecondary);
            }
        }
        return total;
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::EvaluateOwnedBoundarySpectralRadius(
        index iNode, const NodeStatePair &means) const
    {
        real total = 0;
        const auto &controlVolume = _geometry.NodeVolume(iNode);
        const State state = _reconstructionSettings.limitFluxCorrection ? State(means[iNode]) : State(_pointValues[iNode]);
        real wallDistance = 1;
        if constexpr (nTurbulence > 0)
            wallDistance = _turbulenceGeometry[iNode](0);
        const StateGradient gradient = NodalGradient(iNode);
        const real diffusivity = MaximumDiffusivity(state, gradient, wallDistance);
        for (const auto &piece : controlVolume.boundaryPieces)
        {
            const auto &boundary = _boundaries.Get(piece.zone);
            if (boundary.mode == BoundaryMode::Periodic)
                continue;
            const SpatialVector normal =
                piece.vectorMeasure.template head<dimension>() / piece.measure;
            if (_reconstructionSettings.limitFluxCorrection)
            {
                total += piece.measure * LowOrderSpeed(state, LowOrderBoundaryExterior(state, normal, boundary), normal);
                continue;
            }
            real length = controlVolume.lengthScale, faceDiffusivity = diffusivity;
            if constexpr (nTurbulence > 0)
            {
                length = std::min(length, BoundaryNormalDistance(iNode, normal));
                if (!BoundaryRegistry::IsWall(boundary.mode) && boundary.mode != BoundaryMode::Extrapolation &&
                    state.template segment<dimension>(1).dot(normal) < 0)
                {
                    State incoming = state;
                    incoming.template tail<nTurbulence>() = state[0] * InverseTurbulenceScale() *
                        Eigen::Map<const Eigen::Vector<real, nTurbulence>>(boundary.primitive.data() + FlowSize);
                    faceDiffusivity = std::max(faceDiffusivity, MaximumDiffusivity(incoming, gradient, wallDistance));
                }
            }
            total += SurfaceSpectralRadius(state, faceDiffusivity, normal, piece.measure, length);
        }
        return total;
    }

    template <int dimension, int nTurbulence>
    void SpatialOperator<dimension, nTurbulence>::AllocateNodeMatrix(
        NodeMatrixPair &field,
        const std::string &name,
        int rows,
        int columns)
    {
        field.InitPair(name, _mpi);
        field.father->Resize(_mesh->NumNode(), rows, columns);
        field.son->Resize(
            _nodeHalo.NumNodeGhost(),
            rows, columns);
        field.BorrowSetup(_nodeHalo.Layout());
        field.trans.initPersistentPull();
        for (index iNode = 0; iNode < field.Size(); iNode++)
            field[iNode].setZero();
    }

    template <int dimension, int nTurbulence>
    void SpatialOperator<dimension, nTurbulence>::AllocateLocalNodeMatrix(
        NodeMatrixPair &field,
        const std::string &name,
        int rows,
        int columns)
    {
        // This cache is recomputed for owned and ghost nodes from already
        // synchronized inputs, so it deliberately has no MPI transformer.
        field.InitPair(name, _mpi);
        field.father->Resize(_mesh->NumNode(), rows, columns);
        field.son->Resize(
            _nodeHalo.NumNodeGhost(),
            rows, columns);
        for (index iNode = 0; iNode < field.Size(); iNode++)
            field[iNode].setZero();
    }

    template <int dimension, int nTurbulence>
    void SpatialOperator<dimension, nTurbulence>::AllocateNodeField(
        NodeStatePair &field,
        const std::string &name,
        int rows)
    {
        field.InitPair(name, _mpi);
        field.father->Resize(_mesh->NumNode(), rows, 1);
        field.son->Resize(
            _nodeHalo.NumNodeGhost(),
            rows, 1);
        field.BorrowSetup(_nodeHalo.Layout());
        field.trans.initPersistentPull();
        for (index iNode = 0; iNode < field.Size(); iNode++)
            field[iNode].setZero();
    }

    template <int dimension, int nTurbulence>
    void SpatialOperator<dimension, nTurbulence>::AllocateLocalNodeField(
        NodeStatePair &field,
        const std::string &name,
        int rows)
    {
        field.InitPair(name, _mpi);
        field.father->Resize(_mesh->NumNode(), rows, 1);
        field.son->Resize(0, rows, 1);
        for (index iNode = 0; iNode < field.Size(); iNode++)
            field[iNode].setZero();
    }

    template <int dimension, int nTurbulence>
    void SpatialOperator<dimension, nTurbulence>::AllocateEdgeField(
        NodeStatePair &field,
        const std::string &name,
        int rows)
    {
        field.InitPair(name, _mpi);
        field.father->Resize(_topology.NumEdge(), rows, 1);
        field.son->Resize(_topology.NumEdgeGhost(), rows, 1);
        field.BorrowSetup(const_cast<Geom::tAdjPair &>(_topology.Edge2Node()));
        field.trans.initPersistentPull();
        for (index iEdge = 0; iEdge < field.Size(); iEdge++)
            field[iEdge].setZero();
    }

    template <int dimension, int nTurbulence>
    void SpatialOperator<dimension, nTurbulence>::Initialize()
    {
        DNDS_check_throw_info(_turbulence.ComponentNumber() == nTurbulence,
                              "NCFV configured turbulence model does not match the spatial state layout");
        if (_geometry.UsesMacroFaceBoundary())
        {
            _boundaryMacroFaces.clear();
            for (const auto &volume : _geometry.NodeVolumes())
                if (!volume.boundaryPieces.empty())
                    _boundaryMacroFaces.emplace(volume.node, volume.BuildBoundaryMacroFaces());
        }
        if constexpr (dimension == 3 && nTurbulence == 0)
            if (_physics.exponentialMMS)
            {
                const auto spatialBasis = [](const Vector3 &x)
                { return ExponentialMMS::Powers(ExponentialMMS::direction.dot(x)); };
                if (_mode == IntegrationMode::EfficientDifferential)
                {
                    _mmsNodalExponential.resize(_mesh->NumNode() + _nodeHalo.NumNodeGhost());
                    for (index node = 0; node < static_cast<index>(_mmsNodalExponential.size()); ++node)
                        _mmsNodalExponential[node] = std::exp(ExponentialMMS::direction.dot(_nodeHalo.Coordinate(node)));
                }
                _mmsBasisMeans.resize(_mesh->NumNode());
                for (const auto &volume : _geometry.NodeVolumes())
                    _mmsBasisMeans[volume.node] = volume.VolumeMean<3>(
                        _nodeHalo.Coordinate(volume.node), spatialBasis, [&](index node) -> Eigen::Matrix<real, 3, 5>
                        { return ExponentialMMS::direction * ExponentialMMS::PhaseDerivative(
                                  ExponentialMMS::PowersOf(_mmsNodalExponential[node])).transpose(); });
            }
        AllocateNodeMatrix(_stateGradients, "NCFV.stateGradients", dimension);
        if (_mode == IntegrationMode::EfficientDifferential)
        {
            if (_reconstructionSettings.method == ReconstructionMethod::Variational || nTurbulence > 0)
                AllocateNodeMatrix(_coefficients, "NCFV.coefficients",
                                   Reconstruction::QuadraticBasisSize(dimension));
            AllocateLocalNodeMatrix(
                _physicalFluxGradients, "NCFV.physicalFluxGradients",
                StateSize, dimension * dimension);
        }
        else
            AllocateNodeMatrix(_coefficients, "NCFV.coefficients",
                               Reconstruction::QuadraticBasisSize(dimension));
        AllocateNodeField(
            _pointValues, "NCFV.pointValues", StateSize);
        AllocateNodeField(
            _limiterFactors, "NCFV.limiterFactors", 1);
        AllocateLocalNodeField(
            _localTimeSteps, "NCFV.localTimeSteps", 1);

        AllocateEdgeField(_edgeFlux, "NCFV.edgeFlux", StateSize);
        AllocateEdgeField(_edgeSpectralRadius, "NCFV.edgeSpectralRadius", 1);
        if (_reconstructionSettings.limitFluxCorrection)
        {
            AllocateNodeMatrix(_fluxCorrectionBounds, "NCFV.fluxCorrectionBounds", StateSize + 2, 2);
            AllocateEdgeField(_edgeFluxCorrectionFactors, "NCFV.edgeFluxCorrectionFactors", 1);
        }

        BuildTurbulenceGeometry();
        for (index iNode = 0; iNode < _limiterFactors.Size(); iNode++)
            _limiterFactors[iNode](0, 0) = 1.0;
        for (index iNode = 0; iNode < _localTimeSteps.Size(); iNode++)
            _localTimeSteps[iNode](0, 0) = _time.timeStep;
        if (_reconstructionSettings.sharedGradientFaceSweep)
        {
            _sharedGradientFaceSweep =
                std::make_unique<SharedGradientFaceSweep<dimension, nTurbulence>>(
                    _mpi, _mesh, _topology, _geometry, _nodeHalo);
            _sharedGradientFaceSweep->Initialize(_reconstructionSettings.sharedGradientFaceSweepParallel,
                                                 _reconstructionSettings.sharedGradientBudget);
        }
    }

    template <int dimension, int nTurbulence>
    void SpatialOperator<dimension, nTurbulence>::Reconstruct(NodeStatePair &means)
    {
        double phaseStart = MPI_Wtime();
        means.trans.startPersistentPull();
        means.trans.waitPersistentPull();
        _lastRhsTiming.meanHaloSeconds += MPI_Wtime() - phaseStart;

        phaseStart = MPI_Wtime();
        _reconstruction.ComputeCoefficients(means, _stateGradients, _coefficients);
        _lastRhsTiming.coefficientComputeSeconds += MPI_Wtime() - phaseStart;

        phaseStart = MPI_Wtime();
        if (_mode == IntegrationMode::EfficientDifferential)
        {
            _stateGradients.trans.startPersistentPull();
            _stateGradients.trans.waitPersistentPull();
        }
        else
        {
            _coefficients.trans.startPersistentPull();
            _coefficients.trans.waitPersistentPull();
        }
        _lastRhsTiming.coefficientHaloSeconds += MPI_Wtime() - phaseStart;

        phaseStart = MPI_Wtime();
        _reconstruction.RecoverPointValues(
            means, _stateGradients, _coefficients, _pointValues);
        _lastRhsTiming.pointRecoverySeconds += MPI_Wtime() - phaseStart;
        for (index iNode = 0; iNode < _limiterFactors.Size(); iNode++)
            _limiterFactors[iNode](0, 0) = 1.0;
        if (_reconstructionSettings.sharedGradientFaceSweep)
        {
            phaseStart = MPI_Wtime();
            _sharedGradientFaceSweep->Apply(
                means, _stateGradients, _physics.gamma,
                _reconstructionSettings.limitPressure);
            _reconstruction.RecoverPointValues(
                means, _stateGradients, _coefficients, _pointValues);
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
                _pointValues[iNode] = PreservePhysical(
                    State(_pointValues[iNode]), State(means[iNode]));
            _pointValues.trans.startPersistentPull();
            _pointValues.trans.waitPersistentPull();
            _lastRhsTiming.limiterSeconds += MPI_Wtime() - phaseStart;
            // Gradients now contain all interface factors. The subsequent
            // chain-rule flux gradients use these final gradients and points;
            // keep side factors at one to avoid limiting them a second time.
            return;
        }
        bool pointValuesSynchronized = false;
        if (_reconstructionSettings.limitFluxCorrection && !_reconstructionSettings.enableLimiter)
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
                // The high flux must exist before it can be limited. Retain
                // only physical-state protection when Barth bounds are off.
                _pointValues[iNode] = PreservePhysical(State(_pointValues[iNode]), State(means[iNode]));
        if (_reconstructionSettings.enableLimiter)
        {
            phaseStart = MPI_Wtime();
            if (_mode == IntegrationMode::EfficientDifferential)
            {
                // Point recovery combines neighboring gradients. Barth factors
                // limit the interface correction, so they cannot repair a
                // nonphysical point anchor. Limit that recovery toward the
                // conservative mean before sharing anchors or evaluating fluxes.
                // Physical point values and the evolved means remain unchanged.
                if (!_reconstructionSettings.limitPointRecovery)
                    for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
                        _pointValues[iNode] = PreservePhysical(
                            State(_pointValues[iNode]), State(means[iNode]));
                // Pairwise bounds in thesis (3-97) use both endpoint point
                // values, including a ghost endpoint on partition edges.
                const double haloStart = MPI_Wtime();
                _pointValues.trans.startPersistentPull();
                _pointValues.trans.waitPersistentPull();
                _lastRhsTiming.pointValueHaloSeconds +=
                    MPI_Wtime() - haloStart;
                pointValuesSynchronized = true;
            }
            auto factors = _reconstruction.ComputeLimiterFactors(
                means, _pointValues, _stateGradients, _coefficients);
            if (_mode == IntegrationMode::EfficientDifferential)
            {
                if (_reconstructionSettings.limitPointRecovery)
                {
                    for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
                    {
                        const State mean = means[iNode];
                        const State rawPoint = _pointValues[iNode];
                        real &factor = factors[static_cast<std::size_t>(iNode)];
                        real pressureMinimum = Pressure(mean), pressureMaximum = pressureMinimum;
                        if (_reconstructionSettings.limitPressure)
                            for (const auto &incidence : _topology.Node2Edge(iNode))
                            {
                                const auto &surface = _geometry.EdgeSurface(incidence.edge);
                                const index neighbor = surface.nodes[surface.nodes[0] == iNode ? 1 : 0];
                                const real neighborPressure = Pressure(State(means[neighbor]));
                                pressureMinimum = std::min(pressureMinimum, neighborPressure);
                                pressureMaximum = std::max(pressureMaximum, neighborPressure);
                            }
                        const auto limitPhysical = [&](const State &candidate)
                        {
                            if (_reconstructionSettings.barthZeroOrderFallback)
                            {
                                DNDS_check_throw_info(IsPhysical(mean),
                                                      "NCFV binary Barth fallback requires physical conservative means");
                                if (factor == 0)
                                    return;
                                if (!IsPhysical(candidate))
                                {
                                    factor = 0;
                                    return;
                                }
                                if (_reconstructionSettings.limitPressure)
                                    factor = LimitIdealGasPressureFactor<dimension>(
                                        mean.template head<FlowSize>(), candidate.template head<FlowSize>(),
                                        _physics.gamma, pressureMinimum, pressureMaximum, factor);
                                return;
                            }
                            DNDS_check_throw_info(candidate.allFinite() && IsPhysical(mean),
                                                  "NCFV mean-anchored limiter requires finite reconstruction and physical means");
                            if (!IsPhysical(mean + factor * (candidate - mean)))
                            {
                                real lower = 0, upper = factor;
                                for (int iteration = 0; iteration < 40; iteration++)
                                {
                                    const real middle = .5 * (lower + upper);
                                    if (IsPhysical(mean + middle * (candidate - mean)))
                                        lower = middle;
                                    else
                                        upper = middle;
                                }
                                factor = .95 * lower;
                            }
                            if (_reconstructionSettings.limitPressure)
                                factor = LimitIdealGasPressureFactor<dimension>(
                                    mean.template head<FlowSize>(), candidate.template head<FlowSize>(),
                                    _physics.gamma, pressureMinimum, pressureMaximum, factor);
                        };
                        limitPhysical(rawPoint);
                        for (const auto &incidence : _topology.Node2Edge(iNode))
                        {
                            const auto &surface = _geometry.EdgeSurface(incidence.edge);
                            const int side = surface.nodes[0] == iNode ? 0 : 1;
                            State correction = State::Zero();
                            for (const auto &entry : surface.efficientStencil)
                                correction += _stateGradients[entry.node].transpose() *
                                              entry.stateGradientWeights[side].template head<dimension>();
                            limitPhysical(rawPoint + correction / surface.measure);
                        }
                        // One factor scales the entire mean-to-point/surface
                        // increment. Shared source gradients remain unscaled.
                        if (_reconstructionSettings.barthZeroOrderFallback)
                        {
                            factor = factor < 1 ? 0 : 1;
                            _pointValues[iNode] = factor == 0 ? mean : rawPoint;
                        }
                        else
                            _pointValues[iNode] = mean + factor * (rawPoint - mean);
                    }
                    _pointValues.trans.startPersistentPull();
                    _pointValues.trans.waitPersistentPull();
                }
                for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
                    _limiterFactors[iNode](0, 0) =
                        factors[static_cast<std::size_t>(iNode)];
                _limiterFactors.trans.startPersistentPull();
                _limiterFactors.trans.waitPersistentPull();
            }
            else
            {
                _reconstruction.ApplyLimiter(
                    factors, _stateGradients, _coefficients);
                _coefficients.trans.startPersistentPull();
                _coefficients.trans.waitPersistentPull();
                _reconstruction.RecoverPointValues(
                    means, _stateGradients, _coefficients, _pointValues);
            }
            _lastRhsTiming.limiterSeconds += MPI_Wtime() - phaseStart;
        }
        if (!pointValuesSynchronized)
        {
            if constexpr (nTurbulence > 0)
                for (index iNode = 0; iNode < _mesh->NumNode(); ++iNode)
                    _pointValues[iNode] = PreservePhysical(State(_pointValues[iNode]), State(means[iNode]));
            phaseStart = MPI_Wtime();
            _pointValues.trans.startPersistentPull();
            _pointValues.trans.waitPersistentPull();
            _lastRhsTiming.pointValueHaloSeconds += MPI_Wtime() - phaseStart;
        }
    }

    template <int dimension, int nTurbulence>
    void SpatialOperator<dimension, nTurbulence>::UpdateLocalTimeSteps(const NodeStatePair &means)
    {
        std::vector<real> candidate(static_cast<std::size_t>(_mesh->NumNode()),
                                    _time.timeStep);
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            if (!_time.useCFLTimeStep)
                continue;
            real spectralRadius = EvaluateOwnedBoundarySpectralRadius(iNode, means);
            for (const auto &incidence : _topology.Node2Edge(iNode))
                spectralRadius += _edgeSpectralRadius[incidence.edge](0, 0);
            const real volume = _geometry.NodeVolume(iNode).moments.measure;
            if constexpr (nTurbulence > 0)
                spectralRadius += volume * _sourceDestructionRates[iNode];
            candidate[static_cast<std::size_t>(iNode)] = std::clamp(
                _time.cfl * volume / std::max(spectralRadius, verySmallReal),
                _time.minimumTimeStep, _time.maximumTimeStep);
        }

        for (real &step : candidate)
            step = std::min(step, _maximumStep);

        if (_time.useCFLTimeStep && !_time.useLocalTimeStep)
        {
            real localMinimum = candidate.empty()
                                    ? std::numeric_limits<real>::max()
                                    : *std::min_element(candidate.begin(), candidate.end());
            real globalMinimum = 0;
            MPI_Allreduce(&localMinimum, &globalMinimum, 1,
                          DNDS_MPI_REAL, MPI_MIN, _mpi.comm);
            std::fill(candidate.begin(), candidate.end(), globalMinimum);
        }

        real localMinimum = std::numeric_limits<real>::max();
        real localMaximum = 0;
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const real value = candidate[static_cast<std::size_t>(iNode)];
            _localTimeSteps[iNode](0, 0) = value;
            localMinimum = std::min(localMinimum, value);
            localMaximum = std::max(localMaximum, value);
        }
        if (_time.useCFLTimeStep && _time.useLocalTimeStep)
        {
            MPI_Allreduce(&localMinimum, &_lastMinimumTimeStep, 1,
                          DNDS_MPI_REAL, MPI_MIN, _mpi.comm);
            MPI_Allreduce(&localMaximum, &_lastMaximumTimeStep, 1,
                          DNDS_MPI_REAL, MPI_MAX, _mpi.comm);
        }
        else
        {
            _lastMinimumTimeStep = localMinimum;
            _lastMaximumTimeStep = localMaximum;
        }
    }

    template <int dimension, int nTurbulence>
    void SpatialOperator<dimension, nTurbulence>::ApplyStrongBoundaryConditions(
        NodeStatePair &state) const
    {
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const BoundaryZoneSettings *inlet = nullptr, *wall = nullptr;
            Geom::t_index inletZone = std::numeric_limits<Geom::t_index>::max();
            Geom::t_index wallZone = inletZone;
            bool allInletFacesIncoming = true;
            for (const auto &piece : _geometry.NodeVolume(iNode).boundaryPieces)
            {
                const auto &candidate = _boundaries.Get(piece.zone);
                if (!candidate.strongState ||
                    !BoundaryRegistry::IsStrongType(candidate.mode))
                    continue;
                if (candidate.mode == BoundaryMode::SupersonicInlet)
                {
                    // Only prescribe the entire updated dual mean when every strong inlet
                    // normal at this node has five (four in 2-D) strictly incoming waves.
                    const State mean = state[iNode];
                    if (allInletFacesIncoming &&
                        StateLayout<dimension, 0>::IsPhysical(mean.template head<FlowSize>(), _physics.gamma))
                    {
                        const real normalMomentum = mean.template segment<dimension>(1).dot(
                            piece.vectorMeasure.template head<dimension>()) / piece.measure;
                        allInletFacesIncoming = normalMomentum < -std::sqrt(_physics.gamma * mean[0] * Pressure(mean));
                    }
                    else
                        allInletFacesIncoming = false;
                    if (piece.zone < inletZone)
                    {
                        inlet = &candidate;
                        inletZone = piece.zone;
                    }
                }
                else if (piece.zone < wallZone)
                {
                    wall = &candidate;
                    wallZone = piece.zone;
                }
            }
            if (inlet && allInletFacesIncoming)
            {
                state[iNode] = PrimitiveToConservative(inlet->primitive);
                continue;
            }
            if (!wall)
                continue;
            const State inside = state[iNode];
            State primitive = ConservativeToPrimitive(inside);
            for (int i = 0; i < dimension; i++)
                primitive(1 + i) =
                    wall->wallVelocity[static_cast<std::size_t>(i)];
            if (wall->mode == BoundaryMode::NoSlipIsothermalWall)
                primitive(dimension + 1) =
                    primitive(0) * _physics.viscous.gasConstant *
                    wall->wallTemperature;
            State imposed = Layout::ToConservative(primitive, _physics.gamma, _physics.turbulence.viscosityReference);
            // The wall fixes the scalar trace through its flux. Its dual mean
            // remains a transported volume average, including at a wall node.
            if constexpr (nTurbulence > 0)
                imposed.template tail<nTurbulence>() = inside.template tail<nTurbulence>();
            state[iNode] = imposed;
        }
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::EvaluateRHS(
        NodeStatePair &means,
        NodeStatePair &rhs,
        bool updateTimeSteps,
        bool computeResidualNorm,
        real physicalTime)
    {
        _lastRhsTiming = {};
        const real mmsTimeFactor = _physics.exponentialMMS ? std::exp(ExponentialMMS::frequency * physicalTime) : 1;
        const double totalStart = MPI_Wtime();
        double phaseStart = MPI_Wtime();
        Reconstruct(means);
        _lastRhsTiming.reconstructionSeconds = MPI_Wtime() - phaseStart;

        if (_mode == IntegrationMode::EfficientDifferential)
        {
            phaseStart = MPI_Wtime();
            ComputePhysicalFluxGradients();
            _lastRhsTiming.physicalFluxGradientComputeSeconds =
                MPI_Wtime() - phaseStart;
        }
        if (_reconstructionSettings.limitFluxCorrection)
            PrepareFluxCorrectionBounds(means);

        const bool prepareCFL = updateTimeSteps && _time.useCFLTimeStep;
        const bool prepareSpeeds = prepareCFL || _reconstructionSettings.limitFluxCorrection;
        phaseStart = MPI_Wtime();
        for (index iEdge = 0; iEdge < _topology.NumEdge(); iEdge++)
        {
            _edgeFlux[iEdge] = EvaluateOwnedEdgeFlux(iEdge, means);
            if (prepareSpeeds)
                _edgeSpectralRadius[iEdge](0, 0) = EvaluateOwnedEdgeSpectralRadius(iEdge, means);
        }
        _lastRhsTiming.edgeFluxSeconds = MPI_Wtime() - phaseStart;

        phaseStart = MPI_Wtime();
        _edgeFlux.trans.startPersistentPull();
        if (prepareSpeeds)
            _edgeSpectralRadius.trans.startPersistentPull();
        _edgeFlux.trans.waitPersistentPull();
        if (prepareSpeeds)
            _edgeSpectralRadius.trans.waitPersistentPull();
        _lastRhsTiming.edgeHaloSeconds = MPI_Wtime() - phaseStart;

        phaseStart = MPI_Wtime();
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            State residual = EvaluateOwnedBoundaryFlux(iNode, means, mmsTimeFactor);
            for (const auto &incidence : _topology.Node2Edge(iNode))
                residual += incidence.outwardSign * State(_edgeFlux[incidence.edge]);
            const real volume = _geometry.NodeVolume(iNode).moments.measure;
            rhs[iNode] = -residual / volume;
            if constexpr (nTurbulence > 0)
                rhs[iNode] += TurbulenceSourceMean(iNode, State(means[iNode]), prepareCFL);
        }
        if constexpr (dimension == 3 && nTurbulence == 0)
            if (_physics.exponentialMMS)
            {
                const ExponentialMMS::Coefficients sourceAtTime = ExponentialMMS::sourceCoefficients.transpose() *
                    ExponentialMMS::PowersOf(mmsTimeFactor).asDiagonal();
                for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
                    rhs[iNode].noalias() += sourceAtTime * _mmsBasisMeans[iNode];
            }
        _lastRhsTiming.residualAssemblySeconds = MPI_Wtime() - phaseStart;

        if (updateTimeSteps)
        {
            phaseStart = MPI_Wtime();
            UpdateLocalTimeSteps(means);
            _lastRhsTiming.localTimeStepSeconds = MPI_Wtime() - phaseStart;
        }

        if (_reconstructionSettings.limitFluxCorrection)
        {
            // SSPRK stages reuse the first stage's step. Check the actual
            // convex-combination condition again with every stage's speeds.
            real localMaximum = 0, globalMaximum = 0;
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            {
                real spectralRadius = EvaluateOwnedBoundarySpectralRadius(iNode, means);
                for (const auto &incidence : _topology.Node2Edge(iNode))
                    spectralRadius += _edgeSpectralRadius[incidence.edge](0);
                localMaximum = std::max(localMaximum, _localTimeSteps[iNode](0) * spectralRadius /
                                                         _geometry.NodeVolume(iNode).moments.measure);
            }
            MPI_Allreduce(&localMaximum, &globalMaximum, 1, DNDS_MPI_REAL, MPI_MAX, _mpi.comm);
            DNDS_check_throw_info(globalMaximum <= 1 + 1e-12,
                                  "NCFV flux correction stage violates the LLF convex CFL condition; reduce time.cfl or minimumTimeStep");
        }

        real residualNorm = 0;
        if (computeResidualNorm)
        {
            phaseStart = MPI_Wtime();
            real localTotals[2]{0, 0};
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            {
                const real volume = _geometry.NodeVolume(iNode).moments.measure;
                localTotals[0] += volume * rhs[iNode].squaredNorm();
                localTotals[1] += volume;
            }

            real globalTotals[2]{};
            MPI_Allreduce(localTotals, globalTotals, 2,
                          DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
            residualNorm = std::sqrt(
                globalTotals[0] /
                std::max(globalTotals[1], verySmallReal));
            _lastRhsTiming.residualNormSeconds = MPI_Wtime() - phaseStart;
        }
        _lastRhsTiming.totalSeconds = MPI_Wtime() - totalStart;
        return residualNorm;
    }

    template class SpatialOperator<DNDS_NCFV_INSTANTIATION_DIM, DNDS_NCFV_INSTANTIATION_RANS>;
}
