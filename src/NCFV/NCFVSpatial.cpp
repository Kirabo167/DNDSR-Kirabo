#include "NCFVSpatial.hpp"

#include "DNDS/Errors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace DNDS::NCFV
{
    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::PrimitiveToConservative(
        const std::vector<real> &primitive) const
    {
        DNDS_check_throw_info(primitive.size() == static_cast<std::size_t>(dimension + 2),
                              "NCFV primitive-state size mismatch");
        State state = State::Zero();
        const real density = primitive[0];
        SpatialVector velocity;
        for (int i = 0; i < dimension; i++)
            velocity(i) = primitive[static_cast<std::size_t>(i + 1)];
        const real pressure = primitive[static_cast<std::size_t>(dimension + 1)];
        state(0) = density;
        state.template segment<dimension>(1) = density * velocity;
        state(dimension + 1) = pressure / (_physics.gamma - 1.0) +
                               0.5 * density * velocity.squaredNorm();
        return state;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::ConservativeToPrimitive(const State &state) const
    {
        State primitive = State::Zero();
        primitive(0) = state(0);
        primitive.template segment<dimension>(1) =
            state.template segment<dimension>(1) / state(0);
        primitive(dimension + 1) = Pressure(state);
        return primitive;
    }

    template <int dimension>
    real SpatialOperator<dimension>::Pressure(const State &state) const
    {
        return (_physics.gamma - 1.0) *
               (state(dimension + 1) -
                0.5 * state.template segment<dimension>(1).squaredNorm() / state(0));
    }

    template <int dimension>
    real SpatialOperator<dimension>::Temperature(const State &state) const
    {
        return Pressure(state) /
               (state(0) * _physics.viscous.gasConstant);
    }

    template <int dimension>
    real SpatialOperator<dimension>::MolecularViscosity(const State &state) const
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

    template <int dimension>
    bool SpatialOperator<dimension>::IsPhysical(const State &state) const
    {
        return state.allFinite() && state(0) > 1e-12 && Pressure(state) > 1e-12;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::PreservePhysical(
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

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::PhysicalFlux(
        const State &state,
        const SpatialVector &normal) const
    {
        State flux = State::Zero();
        const SpatialVector momentum = state.template segment<dimension>(1);
        const SpatialVector velocity = momentum / state(0);
        const real pressure = Pressure(state);
        const real normalVelocity = velocity.dot(normal);
        flux(0) = state(0) * normalVelocity;
        flux.template segment<dimension>(1) = momentum * normalVelocity + pressure * normal;
        flux(dimension + 1) = (state(dimension + 1) + pressure) * normalVelocity;
        return flux;
    }

    template <int dimension>
    void SpatialOperator<dimension>::ComputePhysicalFluxGradients()
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
                                   0.5 * density * velocitySquared);
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
                                                     velocitySquared);

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

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::NumericalFlux(
        const State &left,
        const State &right,
        const SpatialVector &unitNormal) const
    {
        ++_riemannSolverCallCount;
        State flux = State::Zero();
        SpatialVector gridVelocity = SpatialVector::Zero();
        real acousticMinus = 0;
        real contact = 0;
        real acousticPlus = 0;
        Euler::Gas::InviscidFlux_IdealGas_Dispatcher<dimension>(
            _physics.riemannSolver,
            left, right, left, right,
            gridVelocity, unitNormal, _physics.gamma, _physics.gamma, flux,
            0.0, 1.0, 1.0,
            []() {}, acousticMinus, contact, acousticPlus);
        return flux;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::BoundaryExterior(
        const State &inside,
        const SpatialVector &unitNormal,
        const BoundaryZoneSettings &boundary) const
    {
        switch (boundary.mode)
        {
        case BoundaryMode::FarField:
        case BoundaryMode::SupersonicInlet:
            return PrimitiveToConservative(boundary.primitive);
        case BoundaryMode::SupersonicOutlet:
        case BoundaryMode::Periodic:
            return inside;
        case BoundaryMode::PressureOutlet:
        {
            State primitive = ConservativeToPrimitive(inside);
            primitive(dimension + 1) = boundary.staticPressure;
            return PrimitiveToConservative(
                std::vector<real>(primitive.data(), primitive.data() + primitive.size()));
        }
        case BoundaryMode::SlipWall:
        case BoundaryMode::Symmetry:
        {
            State outside = inside;
            SpatialVector momentum = inside.template segment<dimension>(1);
            momentum -= 2.0 * momentum.dot(unitNormal) * unitNormal;
            outside.template segment<dimension>(1) = momentum;
            return outside;
        }
        case BoundaryMode::NoSlipAdiabaticWall:
        case BoundaryMode::NoSlipIsothermalWall:
        {
            State primitive = ConservativeToPrimitive(inside);
            for (int i = 0; i < dimension; i++)
                primitive(1 + i) =
                    2.0 * boundary.wallVelocity[static_cast<std::size_t>(i)] -
                    primitive(1 + i);
            return PrimitiveToConservative(
                std::vector<real>(primitive.data(), primitive.data() + primitive.size()));
        }
        }
        DNDS_check_throw_info(false, "NCFV boundary mode is invalid");
        return inside;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::BoundaryNumericalFlux(
        const State &inside,
        const SpatialVector &unitNormal,
        const BoundaryZoneSettings &boundary) const
    {
        return NumericalFlux(
            inside, BoundaryExterior(inside, unitNormal, boundary), unitNormal);
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::EvaluateTraditionalState(
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

    template <int dimension>
    typename SpatialOperator<dimension>::StateGradient
    SpatialOperator<dimension>::EvaluateTraditionalGradient(
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

    template <int dimension>
    typename SpatialOperator<dimension>::StateGradient
    SpatialOperator<dimension>::EfficientSurfaceGradient(
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

    template <int dimension>
    typename SpatialOperator<dimension>::StateGradient
    SpatialOperator<dimension>::EfficientBoundarySurfaceGradient(
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

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::EfficientIntegratedPhysicalFlux(
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

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::EfficientSurfaceMean(
        int side,
        const EdgeControlSurface &surface) const
    {
        DNDS_check_throw_info(side == 0 || side == 1,
                              "NCFV efficient surface side must be zero or one");
        DNDS_check_throw_info(surface.measure > verySmallReal,
                              "NCFV efficient surface has zero measure");
        const std::size_t sideIndex = static_cast<std::size_t>(side);
        const index anchorNode = surface.nodes[sideIndex];
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

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::InternalViscousFlux(
        const State &left,
        const State &right,
        const StateGradient &leftGradient,
        const StateGradient &rightGradient,
        const SpatialVector &unitNormal,
        real characteristicDistance) const
    {
        State flux = State::Zero();
        if (!_physics.viscous.enabled)
            return flux;

        // Thesis (4-153)/(4-156) uses the arithmetic mean of the primitive
        // variables q, not the primitive state obtained from an arithmetic
        // mean of conservative variables.
        const State facePrimitive =
            0.5 * (ConservativeToPrimitive(left) +
                   ConservativeToPrimitive(right));
        const State faceState = PrimitiveToConservative(
            std::vector<real>(facePrimitive.data(),
                              facePrimitive.data() + facePrimitive.size()));
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

        const real viscosity = MolecularViscosity(faceState);
        const real cp = _physics.viscous.gasConstant * _physics.gamma /
                        (_physics.gamma - 1.0);
        const real conductivity =
            viscosity * cp / _physics.viscous.prandtlNumber;
        Euler::Gas::ViscousFlux_IdealGas<dimension>(
            faceState, primitiveGradient, unitNormal, false,
            _physics.gamma, _physics.gamma, viscosity, 0.0, false,
            conductivity, cp, flux);
        return flux;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::BoundaryViscousFlux(
        const State &inside,
        const StateGradient &insideGradient,
        const SpatialVector &unitNormal,
        const BoundaryZoneSettings &boundary,
        real lengthScale) const
    {
        State flux = State::Zero();
        if (!_physics.viscous.enabled ||
            boundary.mode == BoundaryMode::SlipWall ||
            boundary.mode == BoundaryMode::Symmetry)
            return flux;

        StateGradient primitiveGradient;
        Euler::Gas::GradientCons2Prim_IdealGas<dimension>(
            inside, insideGradient, primitiveGradient, _physics.gamma,
            Eigen::Vector<real, 0>{});
        State fluxState = inside;
        bool adiabatic = false;

        if (boundary.mode == BoundaryMode::NoSlipAdiabaticWall ||
            boundary.mode == BoundaryMode::NoSlipIsothermalWall)
        {
            State insidePrimitive = ConservativeToPrimitive(inside);
            State wallPrimitive = insidePrimitive;
            for (int i = 0; i < dimension; i++)
                wallPrimitive(1 + i) =
                    boundary.wallVelocity[static_cast<std::size_t>(i)];
            if (boundary.mode == BoundaryMode::NoSlipIsothermalWall)
                wallPrimitive(dimension + 1) =
                    wallPrimitive(0) * _physics.viscous.gasConstant *
                    boundary.wallTemperature;
            adiabatic = boundary.mode == BoundaryMode::NoSlipAdiabaticWall;

            const real wallDistance = std::max(0.5 * lengthScale, verySmallReal);
            const Eigen::RowVector<real, dimension + 2> desiredNormalDerivative =
                (wallPrimitive - insidePrimitive).transpose() / wallDistance;
            primitiveGradient += unitNormal * desiredNormalDerivative;
            fluxState = PrimitiveToConservative(
                std::vector<real>(wallPrimitive.data(),
                                  wallPrimitive.data() + wallPrimitive.size()));
        }

        const real viscosity = MolecularViscosity(fluxState);
        const real cp = _physics.viscous.gasConstant * _physics.gamma /
                        (_physics.gamma - 1.0);
        const real conductivity =
            viscosity * cp / _physics.viscous.prandtlNumber;
        Euler::Gas::ViscousFlux_IdealGas<dimension>(
            fluxState, primitiveGradient, unitNormal, adiabatic,
            _physics.gamma, _physics.gamma, viscosity, 0.0, false,
            conductivity, cp, flux);
        return flux;
    }

    template <int dimension>
    real SpatialOperator<dimension>::SurfaceSpectralRadius(
        const State &state,
        const SpatialVector &unitNormal,
        real measure,
        real lengthScale) const
    {
        const SpatialVector velocity =
            state.template segment<dimension>(1) / state(0);
        const real acousticSpeed =
            std::sqrt(_physics.gamma * Pressure(state) / state(0));
        real spectralRadius =
            (std::abs(velocity.dot(unitNormal)) + acousticSpeed) * measure;
        if (_physics.viscous.enabled)
        {
            const real kinematicViscosity = MolecularViscosity(state) / state(0);
            const real largestDiffusivity = std::max(
                (4.0 / 3.0) * kinematicViscosity,
                kinematicViscosity / _physics.viscous.prandtlNumber);
            spectralRadius += _physics.viscous.spectralRadiusFactor *
                              largestDiffusivity * measure /
                              std::max(lengthScale, verySmallReal);
        }
        return spectralRadius;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::EvaluateOwnedEdgeFlux(index iEdge)
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
                if (_physics.viscous.enabled)
                    flux -= InternalViscousFlux(
                        left, right,
                        EvaluateTraditionalGradient(node0, point.coordinate),
                        EvaluateTraditionalGradient(node1, point.coordinate),
                        pointNormal, characteristicDistance);
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
        const State dissipativeCorrection =
            surface.measure * (numerical - centralMean);
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
                            unitNormal, characteristicDistance);
            if (_detailedFluxTiming)
                _lastRhsTiming.edgeViscousFluxSeconds +=
                    MPI_Wtime() - phaseStart;
        }
        return integral;
    }

    template <int dimension>
    real SpatialOperator<dimension>::EvaluateOwnedEdgeSpectralRadius(index iEdge) const
    {
        const auto &surface = _geometry.EdgeSurface(iEdge);
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
        return SurfaceSpectralRadius(
            mean, normal, surface.measure, std::min(length0, length1));
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::EvaluateOwnedBoundaryFlux(index iNode) const
    {
        State total = State::Zero();
        const auto &controlVolume = _geometry.NodeVolume(iNode);
        if (_mode == IntegrationMode::TraditionalQuadrature)
        {
            for (const auto &piece : controlVolume.boundaryPieces)
            {
                const auto &boundary = _boundaries.Get(piece.zone);
                if (boundary.mode == BoundaryMode::Periodic)
                    continue;
                for (const auto &point : piece.quadrature)
                {
                    const SpatialVector unitNormal =
                        point.vectorWeight.template head<dimension>() / point.weight;
                    const State inside = EvaluateTraditionalState(iNode, point.coordinate);
                    State flux = BoundaryNumericalFlux(inside, unitNormal, boundary);
                    if (_physics.viscous.enabled)
                        flux -= BoundaryViscousFlux(
                            inside,
                            EvaluateTraditionalGradient(iNode, point.coordinate),
                            unitNormal, boundary, controlVolume.lengthScale);
                    total += point.weight * flux;
                }
            }
            return total;
        }

        // No Gaussian points: the initialization-built stencil already merges
        // every construction-point support into one sorted local/ghost node set.
        for (const auto &piece : controlVolume.boundaryPieces)
        {
            const auto &boundary = _boundaries.Get(piece.zone);
            if (boundary.mode == BoundaryMode::Periodic)
                continue;
            const SpatialVector unitNormal =
                piece.vectorMeasure.template head<dimension>() / piece.measure;
            State insideIntegral = State::Zero();
            State inviscidIntegral = State::Zero();
            for (const EfficientBoundaryNode &entry : piece.efficientStencil)
            {
                const State nodalState = _pointValues[entry.node];
                insideIntegral += entry.integrationWeight * nodalState;
                // Thesis section 4.3.2 interpolates already evaluated nodal
                // numerical fluxes, so the same precomputed value weight is
                // applied after BoundaryNumericalFlux().
                inviscidIntegral += entry.integrationWeight *
                                    BoundaryNumericalFlux(
                                        nodalState, unitNormal, boundary);
            }
            const State insideMean = PreservePhysical(
                insideIntegral / piece.measure, State(_pointValues[iNode]));
            total += inviscidIntegral;
            if (_physics.viscous.enabled)
                total -= piece.measure * BoundaryViscousFlux(
                                             insideMean,
                                             EfficientBoundarySurfaceGradient(
                                                 iNode, piece),
                                             unitNormal, boundary,
                                             controlVolume.lengthScale);
        }
        return total;
    }

    template <int dimension>
    real SpatialOperator<dimension>::EvaluateOwnedBoundarySpectralRadius(
        index iNode) const
    {
        real total = 0;
        const auto &controlVolume = _geometry.NodeVolume(iNode);
        const State state = _pointValues[iNode];
        for (const auto &piece : controlVolume.boundaryPieces)
        {
            if (_boundaries.Get(piece.zone).mode == BoundaryMode::Periodic)
                continue;
            const SpatialVector normal =
                piece.vectorMeasure.template head<dimension>() / piece.measure;
            total += SurfaceSpectralRadius(
                state, normal, piece.measure, controlVolume.lengthScale);
        }
        return total;
    }

    template <int dimension>
    void SpatialOperator<dimension>::AllocateNodeMatrix(
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

    template <int dimension>
    void SpatialOperator<dimension>::AllocateLocalNodeMatrix(
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

    template <int dimension>
    void SpatialOperator<dimension>::AllocateNodeField(
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

    template <int dimension>
    void SpatialOperator<dimension>::AllocateLocalNodeField(
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

    template <int dimension>
    void SpatialOperator<dimension>::AllocateEdgeField(
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

    template <int dimension>
    void SpatialOperator<dimension>::Initialize()
    {
        AllocateNodeMatrix(_stateGradients, "NCFV.stateGradients", dimension);
        if (_mode == IntegrationMode::EfficientDifferential)
            AllocateLocalNodeMatrix(
                _physicalFluxGradients, "NCFV.physicalFluxGradients",
                dimension + 2, dimension * dimension);
        else
            AllocateNodeMatrix(_coefficients, "NCFV.coefficients",
                               Reconstruction::QuadraticBasisSize(dimension));
        AllocateNodeField(
            _pointValues, "NCFV.pointValues", dimension + 2);
        AllocateNodeField(
            _limiterFactors, "NCFV.limiterFactors", 1);
        AllocateLocalNodeField(
            _localTimeSteps, "NCFV.localTimeSteps", 1);

        AllocateEdgeField(_edgeFlux, "NCFV.edgeFlux", dimension + 2);
        AllocateEdgeField(_edgeSpectralRadius, "NCFV.edgeSpectralRadius", 1);

        _farField = PrimitiveToConservative(_physics.farFieldPrimitive);
        for (index iNode = 0; iNode < _limiterFactors.Size(); iNode++)
            _limiterFactors[iNode](0, 0) = 1.0;
        for (index iNode = 0; iNode < _localTimeSteps.Size(); iNode++)
            _localTimeSteps[iNode](0, 0) = _time.timeStep;
    }

    template <int dimension>
    void SpatialOperator<dimension>::Reconstruct(NodeStatePair &means)
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
        bool pointValuesSynchronized = false;
        if (_reconstructionSettings.enableLimiter)
        {
            phaseStart = MPI_Wtime();
            if (_mode == IntegrationMode::EfficientDifferential)
            {
                // Pairwise bounds in thesis (3-97) use both endpoint point
                // values, including a ghost endpoint on partition edges.
                const double haloStart = MPI_Wtime();
                _pointValues.trans.startPersistentPull();
                _pointValues.trans.waitPersistentPull();
                _lastRhsTiming.pointValueHaloSeconds +=
                    MPI_Wtime() - haloStart;
                pointValuesSynchronized = true;
            }
            const auto factors = _reconstruction.ComputeLimiterFactors(
                means, _pointValues, _stateGradients, _coefficients);
            if (_mode == IntegrationMode::EfficientDifferential)
            {
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
            phaseStart = MPI_Wtime();
            _pointValues.trans.startPersistentPull();
            _pointValues.trans.waitPersistentPull();
            _lastRhsTiming.pointValueHaloSeconds += MPI_Wtime() - phaseStart;
        }
    }

    template <int dimension>
    void SpatialOperator<dimension>::UpdateLocalTimeSteps()
    {
        std::vector<real> candidate(static_cast<std::size_t>(_mesh->NumNode()),
                                    _time.timeStep);
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            if (!_time.useCFLTimeStep)
                continue;
            real spectralRadius = EvaluateOwnedBoundarySpectralRadius(iNode);
            for (const auto &incidence : _topology.Node2Edge(iNode))
                spectralRadius += _edgeSpectralRadius[incidence.edge](0, 0);
            const real volume = _geometry.NodeVolume(iNode).moments.measure;
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

    template <int dimension>
    void SpatialOperator<dimension>::ApplyStrongBoundaryConditions(
        NodeStatePair &state) const
    {
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const BoundaryZoneSettings *selected = nullptr;
            int selectedPriority = -1;
            Geom::t_index selectedZone = std::numeric_limits<Geom::t_index>::max();
            for (const auto &piece : _geometry.NodeVolume(iNode).boundaryPieces)
            {
                const auto &candidate = _boundaries.Get(piece.zone);
                if (!candidate.strongState ||
                    !BoundaryRegistry::IsStrongType(candidate.mode))
                    continue;
                const int priority = candidate.mode == BoundaryMode::SupersonicInlet
                                         ? 2
                                         : 1;
                if (priority > selectedPriority ||
                    (priority == selectedPriority && piece.zone < selectedZone))
                {
                    selected = &candidate;
                    selectedPriority = priority;
                    selectedZone = piece.zone;
                }
            }
            if (!selected)
                continue;

            if (selected->mode == BoundaryMode::SupersonicInlet)
            {
                state[iNode] = PrimitiveToConservative(selected->primitive);
                continue;
            }

            State primitive = ConservativeToPrimitive(State(state[iNode]));
            for (int i = 0; i < dimension; i++)
                primitive(1 + i) =
                    selected->wallVelocity[static_cast<std::size_t>(i)];
            if (selected->mode == BoundaryMode::NoSlipIsothermalWall)
                primitive(dimension + 1) =
                    primitive(0) * _physics.viscous.gasConstant *
                    selected->wallTemperature;
            state[iNode] = PrimitiveToConservative(
                std::vector<real>(primitive.data(),
                                  primitive.data() + primitive.size()));
        }
    }

    template <int dimension>
    real SpatialOperator<dimension>::EvaluateRHS(
        NodeStatePair &means,
        NodeStatePair &rhs,
        bool updateTimeSteps,
        bool computeResidualNorm)
    {
        _lastRhsTiming = {};
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

        phaseStart = MPI_Wtime();
        for (index iEdge = 0; iEdge < _topology.NumEdge(); iEdge++)
        {
            _edgeFlux[iEdge] = EvaluateOwnedEdgeFlux(iEdge);
            _edgeSpectralRadius[iEdge](0, 0) =
                EvaluateOwnedEdgeSpectralRadius(iEdge);
        }
        _lastRhsTiming.edgeFluxSeconds = MPI_Wtime() - phaseStart;

        phaseStart = MPI_Wtime();
        _edgeFlux.trans.startPersistentPull();
        _edgeSpectralRadius.trans.startPersistentPull();
        _edgeFlux.trans.waitPersistentPull();
        _edgeSpectralRadius.trans.waitPersistentPull();
        _lastRhsTiming.edgeHaloSeconds = MPI_Wtime() - phaseStart;

        if (updateTimeSteps)
        {
            phaseStart = MPI_Wtime();
            UpdateLocalTimeSteps();
            _lastRhsTiming.localTimeStepSeconds = MPI_Wtime() - phaseStart;
        }

        phaseStart = MPI_Wtime();
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            State residual = EvaluateOwnedBoundaryFlux(iNode);
            for (const auto &incidence : _topology.Node2Edge(iNode))
                residual += incidence.outwardSign * State(_edgeFlux[incidence.edge]);
            const real volume = _geometry.NodeVolume(iNode).moments.measure;
            rhs[iNode] = -residual / volume;
        }
        _lastRhsTiming.residualAssemblySeconds = MPI_Wtime() - phaseStart;

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

    template class SpatialOperator<2>;
    template class SpatialOperator<3>;
}
