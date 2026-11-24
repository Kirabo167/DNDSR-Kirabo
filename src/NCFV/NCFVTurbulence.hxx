#pragma once

#include "Physics/RANS/WallTarget.hpp"

namespace DNDS::NCFV
{
    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::TurbulentPressure(const State &state) const
    {
        if constexpr (nTurbulence == 2)
            return (2.0 / 3.0) * state[FlowSize];
        return 0;
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::InverseTurbulenceScale() const
    {
        if constexpr (nTurbulence == 1)
            return 1 / _physics.turbulence.viscosityReference;
        return 1;
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::BoundaryNormalDistance(index node, const SpatialVector &normal) const
    {
        if constexpr (nTurbulence == 0)
            return 1;
        const auto &volume = _geometry.NodeVolume(node);
        const real projected = -normal.dot(volume.moments.first.template head<dimension>()) / volume.moments.measure;
        return projected > 0 ? projected : 0.5 * volume.lengthScale;
    }

    template <int dimension, int nTurbulence>
    void SpatialOperator<dimension, nTurbulence>::BuildTurbulenceGeometry()
    {
        if constexpr (nTurbulence > 0)
        {
            Geom::UnstructuredMesh::WallDistOptions wallOptions;
            wallOptions.searchPeriodicTranslations = true;
            _mesh->BuildNodeWallDist([&](Geom::t_index zone)
                                     { return BoundaryRegistry::IsNoSlipWall(_boundaries.Get(zone).mode); }, wallOptions);
            AllocateNodeField(_turbulenceGeometry, "NCFV.turbulenceGeometry", 3);
            const bool hybridSA = nTurbulence == 1 && _physics.turbulence.desScale <= 100;
            const bool useCellSpan = hybridSA || _turbulence.Kind() == Physics::RANS::ModelKind::KOmegaSST;
            const real farWallDistance = std::pow(veryLargeReal, .25);
            for (const auto &volume : _geometry.NodeVolumes())
            {
                const index node = volume.node;
                Vector3 direction = _mesh->nodeWallDist[node];
                real distance = direction.stableNorm();
                if (distance > 0 && distance < farWallDistance)
                    direction /= distance;
                else
                    direction.setZero();
                for (const auto &piece : volume.boundaryPieces)
                    if (BoundaryRegistry::IsNoSlipWall(_boundaries.Get(piece.zone).mode))
                    {
                        const SpatialVector normal = piece.vectorMeasure.template head<dimension>() / piece.measure;
                        const real centroidDistance = BoundaryNormalDistance(node, normal);
                        if (distance == 0 || centroidDistance < distance)
                        {
                            distance = centroidDistance;
                            direction.setZero();
                            direction.template head<dimension>() = -normal;
                        }
                    }
                if (distance == 0)
                    distance = 0.5 * volume.lengthScale;

                real hMax = 0, hNormal = 0;
                if (useCellSpan)
                    for (index cell : _mesh->node2cell[node])
                    {
                        Geom::tSmallCoords coordinates;
                        _mesh->GetCoordsOnCell(cell, coordinates);
                        for (Eigen::Index i = 0; i < coordinates.cols(); ++i)
                            for (Eigen::Index j = 0; j < i; ++j)
                            {
                                const Vector3 delta = coordinates.col(i) - coordinates.col(j);
                                hMax = std::max(hMax, delta.norm());
                                if (hybridSA)
                                    hNormal = std::max(hNormal, std::abs(delta.dot(direction)));
                            }
                    }
                _turbulenceGeometry[node] << distance, std::max(hMax, volume.lengthScale), hNormal;
            }
            _turbulenceGeometry.trans.startPersistentPull();
            _turbulenceGeometry.trans.waitPersistentPull();
            if (_time.useCFLTimeStep)
                _sourceDestructionRates.resize(static_cast<std::size_t>(_mesh->NumNode()));
        }
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::SourceWallDistance(index node, const Vector3 &offset) const
    {
        // Extend the nearest wall plane to an interior quadrature point. At
        // wall vertices use the incident no-slip faces and take the nearest.
        const real effectiveDistance = _turbulenceGeometry[node](0);
        const Vector3 wallVector = _mesh->nodeWallDist[node];
        const real distance = wallVector.stableNorm();
        if (distance > 0)
        {
            const real sample = distance + wallVector.dot(offset) / distance;
            return sample > 0 ? sample : effectiveDistance;
        }

        real sample = veryLargeReal;
        for (const auto &piece : _geometry.NodeVolume(node).boundaryPieces)
            if (BoundaryRegistry::IsNoSlipWall(_boundaries.Get(piece.zone).mode))
            {
                const real projected = -piece.vectorMeasure.dot(offset) / piece.measure;
                if (projected > 0)
                    sample = std::min(sample, projected);
            }
        return sample < veryLargeReal ? sample : effectiveDistance;
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::WallSecondaryPrimitive(
        const State &mean, const BoundaryZoneSettings &boundary, real distance) const
    {
        if constexpr (nTurbulence == 2)
            if (BoundaryRegistry::IsNoSlipWall(boundary.mode))
            {
                State reference = mean;
                if (boundary.mode == BoundaryMode::NoSlipIsothermalWall)
                    reference[dimension + 1] = mean[0] * _physics.viscous.gasConstant * boundary.wallTemperature /
                                                   (_physics.gamma - 1) +
                                               0.5 * mean.template segment<dimension>(1).squaredNorm() / mean[0];
                const auto point = TurbulencePoint(reference, StateGradient::Zero(), distance, MolecularViscosity(reference));
                return Physics::RANS::PrepareRANSWallTarget(_turbulence.Kind(), point, reference[0])[1] / reference[0];
            }
        return 0;
    }

    template <int dimension, int nTurbulence>
    Physics::RANS::Point SpatialOperator<dimension, nTurbulence>::TurbulencePoint(
        const State &state, const StateGradient &gradient, real wallDistance, real viscosity) const
    {
        Physics::RANS::Point point;
        point.state.template head<dimension + 1>() = state.template head<dimension + 1>();
        point.state[4] = state[dimension + 1];
        point.gradient.template topLeftCorner<dimension, dimension + 1>() = gradient.template leftCols<dimension + 1>();
        point.gradient.col(4).template head<dimension>() = gradient.col(dimension + 1);
        if constexpr (nTurbulence > 0)
        {
            point.state.template segment<nTurbulence>(5) = state.template tail<nTurbulence>();
            point.gradient.template block<dimension, nTurbulence>(0, 5) = gradient.template rightCols<nTurbulence>();
        }
        point.molecularViscosity = viscosity;
        point.wallDistance = wallDistance;
        point.gamma = _physics.gamma;
        point.cp = _physics.viscous.gasConstant * _physics.gamma / (_physics.gamma - 1);
        return point;
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::ViscousTraceFlux(
        const State &state, const StateGradient &primitiveGradient, const SpatialVector &normal,
        bool adiabatic, real wallDistance, bool noSlip) const
    {
        const real molecular = MolecularViscosity(state);
        const real cp = _physics.viscous.gasConstant * _physics.gamma / (_physics.gamma - 1);
        Physics::RANS::TurbulentTransport transport;
        State flux = State::Zero();
        if constexpr (nTurbulence > 0)
        {
            if (noSlip)
            {
                // The wall targets k=0 or nuTilde=0 give molecular diffusion.
                const real sigma = nTurbulence == 1 ? Physics::RANS::SA::sigma : 1;
                flux.template tail<nTurbulence>() = (molecular / sigma) *
                    primitiveGradient.template rightCols<nTurbulence>().transpose() * normal;
            }
            else
            {
                // Rebase primitive derivatives onto the face state before
                // passing conservative gradients to the common closure.
                const SpatialVector velocity = state.template segment<dimension>(1) / state[0];
                StateGradient gradient = primitiveGradient;
                gradient.template middleCols<dimension>(1) = state[0] * primitiveGradient.template middleCols<dimension>(1) +
                                                               primitiveGradient.col(0) * velocity.transpose();
                gradient.col(dimension + 1) = primitiveGradient.col(dimension + 1) / (_physics.gamma - 1) +
                                              0.5 * velocity.squaredNorm() * primitiveGradient.col(0) +
                                              state[0] * primitiveGradient.template middleCols<dimension>(1) * velocity;
                gradient.template rightCols<nTurbulence>() = state[0] * primitiveGradient.template rightCols<nTurbulence>() +
                    primitiveGradient.col(0) * (state.template tail<nTurbulence>().transpose() / state[0]);
                const auto point = TurbulencePoint(state, gradient, wallDistance, molecular);
                const auto properties = _turbulence.TryTransport(point);
                DNDS_check_throw_info(properties, "NCFV RANS transport evaluation failed at a viscous trace");
                transport = properties.value;
                Vector3 normal3 = Vector3::Zero();
                normal3.template head<dimension>() = normal;
                const auto diffusion = _turbulence.TryDiffusionFlux(point, normal3, transport.viscosity);
                DNDS_check_throw_info(diffusion, "NCFV RANS diffusion evaluation failed at a viscous trace");
                for (int equation = 0; equation < nTurbulence; ++equation)
                    flux[FlowSize + equation] = diffusion.value[equation];
            }
        }
        const real viscosity = molecular + transport.viscosity;
        Euler::Gas::ViscousFlux_IdealGas<dimension>(
            state, primitiveGradient, normal, adiabatic, _physics.gamma, _physics.gamma,
            viscosity, transport.viscosity / viscosity, _physics.turbulence.useQCR,
            molecular * cp / _physics.viscous.prandtlNumber + transport.conductivity, cp, flux);
        return flux;
    }

    template <int dimension, int nTurbulence>
    real SpatialOperator<dimension, nTurbulence>::MaximumDiffusivity(
        const State &state, const StateGradient &gradient, real wallDistance) const
    {
        if (!_physics.viscous.enabled)
            return 0;
        const real molecular = MolecularViscosity(state);
        const real thermal = molecular / _physics.viscous.prandtlNumber;
        if constexpr (nTurbulence == 0)
        {
            const real nu = molecular / state[0];
            return std::max((4.0 / 3.0) * nu, nu / _physics.viscous.prandtlNumber);
        }
        const auto point = TurbulencePoint(state, gradient, wallDistance, molecular);
        const auto transport = _turbulence.TryTransport(point);
        DNDS_check_throw_info(transport, "NCFV RANS transport evaluation failed in the CFL estimate");
        const real eddy = transport.value.viscosity;
        real largest = std::max((4.0 / 3.0) * (molecular + eddy), thermal + eddy / _physics.turbulence.turbulentPrandtl);
        if constexpr (nTurbulence == 1)
            largest = std::max(largest, (molecular + std::abs(state[FlowSize]) * _physics.turbulence.viscosityReference) /
                                            Physics::RANS::SA::sigma);
        return largest / state[0];
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::StateGradient
    SpatialOperator<dimension, nTurbulence>::NodalGradient(index node) const
    {
        if constexpr (nTurbulence == 0)
            return StateGradient::Zero();
        if (_mode == IntegrationMode::TraditionalQuadrature)
            return EvaluateTraditionalGradient(node, _nodeHalo.Coordinate(node));
        return _stateGradients[node];
    }

    template <int dimension, int nTurbulence>
    typename SpatialOperator<dimension, nTurbulence>::State
    SpatialOperator<dimension, nTurbulence>::TurbulenceSourceMean(index node, const State &mean, bool prepareTimeStep)
    {
        State result = State::Zero();
        if constexpr (nTurbulence > 0)
        {
            const auto &volume = _geometry.NodeVolume(node);
            const auto &op = _reconstruction.Operator(node);
            const auto coefficients = _coefficients[node];
            const real factor = _sharedGradientFaceSweep ? _sharedGradientFaceSweep->OwnedFactor(node) : _limiterFactors[node](0);
            real maxDestruction = 0;
            for (const auto &sample : volume.volumeQuadrature)
            {
                const Vector3 offset = _nodeHalo.DisplacementToPoint(node, sample.coordinate);
                const Eigen::VectorXd basis = Reconstruction::EvaluateBasis(offset, op.referenceLengths, dimension) - op.targetBasisMean;
                const State state = PreservePhysical(mean + factor * (coefficients.transpose() * basis), mean);
                const StateGradient gradient = factor * Reconstruction::EvaluateBasisGradient(offset, op.referenceLengths, dimension) * coefficients;
                auto point = TurbulencePoint(state, gradient, SourceWallDistance(node, offset), MolecularViscosity(state));
                point.cellLength = _turbulenceGeometry[node](1);
                point.wallNormalCellLength = _turbulenceGeometry[node](2);
                const auto source = _turbulence.TrySource(point);
                DNDS_check_throw_info(source, "NCFV RANS volume source evaluation failed");
                for (int equation = 0; equation < nTurbulence; ++equation)
                    result[FlowSize + equation] += sample.weight * source.value[equation];
                if (prepareTimeStep)
                {
                    const auto destruction = _turbulence.TrySource(point, 1);
                    DNDS_check_throw_info(destruction, "NCFV RANS source destruction estimate failed");
                    maxDestruction = std::max({maxDestruction, destruction.value[0], destruction.value[1]});
                }
            }
            result /= volume.moments.measure;
            if (prepareTimeStep)
            {
                if constexpr (nTurbulence == 2)
                    for (int equation = FlowSize; equation < StateSize; ++equation)
                        maxDestruction = std::max(maxDestruction, -result[equation] / mean[equation]);
                _sourceDestructionRates[node] = maxDestruction;
            }
        }
        return result;
    }
}
