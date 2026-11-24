#pragma once

#include "NCFVConfig.hpp"

#include <cmath>

namespace DNDS::NCFV
{
    namespace CharacteristicBoundary
    {
        /** A scalar and its derivative along a fixed, possibly area-weighted displacement. */
        struct Scalar
        {
            real value = 0;
            real increment = 0;
        };

        /** Primitive values and directional derivatives of a reconstructed conservative trace. */
        template <int dimension>
        struct Interior
        {
            using Vector = Eigen::Vector<real, dimension>;
            using State = Eigen::Vector<real, dimension + 2>;
            Scalar density, pressure, sound, vn;
            Vector velocity, dVelocity;

            Interior(const State &state, const State &increment, const Vector &normal, real gamma)
            {
                density = {state[0], increment[0]};
                DNDS_check_throw_info(density.value > 0, "NCFV characteristic boundary requires positive interior density");
                velocity = state.template segment<dimension>(1) / density.value;
                pressure.value = (gamma - 1) * (state[dimension + 1] - .5 * density.value * velocity.squaredNorm());
                DNDS_check_throw_info(pressure.value > 0, "NCFV characteristic boundary requires positive interior pressure");
                Euler::Gas::IdealGasUIncrement<dimension>(state, increment, velocity, gamma, dVelocity, pressure.increment);
                sound.value = std::sqrt(gamma * pressure.value / density.value);
                sound.increment = .5 * sound.value * (pressure.increment / pressure.value - density.increment / density.value);
                vn = {velocity.dot(normal), dVelocity.dot(normal)};
            }
        };

        template <int dimension>
        auto ConservativeTrace(Scalar rho, Scalar p, const Eigen::Vector<real, dimension> &u,
                               const Eigen::Vector<real, dimension> &du, real gamma)
        {
            using State = Eigen::Vector<real, dimension + 2>;
            State state, increment;
            state << rho.value, rho.value * u, p.value / (gamma - 1) + .5 * rho.value * u.squaredNorm();
            increment << rho.increment, rho.increment * u + rho.value * du,
                p.increment / (gamma - 1) + .5 * rho.increment * u.squaredNorm() + rho.value * u.dot(du);
            return std::make_pair(state, increment);
        }

        template <int dimension>
        struct Incoming
        {
            using Vector = Eigen::Vector<real, dimension>;
            Scalar minus, plus, entropy; // J-, J+, and K = p/rho^gamma.
            Vector tangentialVelocity = Vector::Zero();
            Vector tangentialIncrement = Vector::Zero();
        };

        inline Scalar Entropy(Scalar density, Scalar pressure, real gamma)
        {
            const real k = pressure.value / std::pow(density.value, gamma);
            return {k, k * (pressure.increment / pressure.value - gamma * density.increment / density.value)};
        }

        /** Project prescribed data onto the incoming families of the interior trace.
         * This is called only when the minus acoustic family enters the domain. */
        template <int dimension, class TangentialData>
        Incoming<dimension> Prescribed(Scalar density, Scalar pressure, Scalar normalVelocity,
                                       real gamma, bool needPlus, bool needConvected, TangentialData tangentialData)
        {
            Incoming<dimension> data;
            const real a = std::sqrt(gamma * pressure.value / density.value);
            const real acoustic = 2 * a / (gamma - 1);
            const real dAcoustic = .5 * acoustic * (pressure.increment / pressure.value - density.increment / density.value);
            data.minus = {normalVelocity.value - acoustic, normalVelocity.increment - dAcoustic};
            if (needPlus)
                data.plus = {normalVelocity.value + acoustic, normalVelocity.increment + dAcoustic};
            if (needConvected)
            {
                data.entropy = Entropy(density, pressure, gamma);
                tangentialData(data.tangentialVelocity, data.tangentialIncrement);
            }
            return data;
        }

        /** Recover a conservative boundary trace from J+/-, K, and tangential velocity.
         * Speeds are evaluated on the interior side with an outward unit normal. Zero-speed
         * families use the interior value. The derivative stays within the selected regime. */
        template <int dimension, class IncomingData>
        auto Trace(const Eigen::Vector<real, dimension + 2> &inside,
                   const Eigen::Vector<real, dimension + 2> &insideIncrement,
                   const Eigen::Vector<real, dimension> &normal, real gamma, IncomingData incomingData)
            -> std::pair<Eigen::Vector<real, dimension + 2>, Eigen::Vector<real, dimension + 2>>
        {
            using Vector = Eigen::Vector<real, dimension>;
            const real gm1 = gamma - 1;
            const Interior<dimension> left(inside, insideIncrement, normal, gamma);
            if (left.vn.value >= left.sound.value)
                return {inside, insideIncrement};

            const bool needPlus = left.vn.value + left.sound.value < 0;
            const bool needConvected = left.vn.value < 0;
            auto waves = incomingData(needPlus, needConvected);
            if (!needPlus)
            {
                waves.plus = {left.vn.value + 2 * left.sound.value / gm1,
                              left.vn.increment + 2 * left.sound.increment / gm1};
            }
            if (!needConvected)
            {
                waves.entropy = Entropy(left.density, left.pressure, gamma);
                waves.tangentialVelocity = left.velocity - left.vn.value * normal;
                waves.tangentialIncrement = left.dVelocity - left.vn.increment * normal;
            }

            const real boundaryVn = .5 * (waves.plus.value + waves.minus.value);
            const real dBoundaryVn = .5 * (waves.plus.increment + waves.minus.increment);
            const real a = .25 * gm1 * (waves.plus.value - waves.minus.value);
            const real da = .25 * gm1 * (waves.plus.increment - waves.minus.increment);
            DNDS_check_throw_info(a > 0, "NCFV incoming and outgoing characteristic data give a nonpositive boundary sound speed");
            const real rho = std::pow(a * a / (gamma * waves.entropy.value), 1 / gm1);
            const real drho = rho / gm1 * (2 * da / a - waves.entropy.increment / waves.entropy.value);
            const real p = rho * a * a / gamma;
            const real dp = p * (drho / rho + 2 * da / a);
            const Vector u = waves.tangentialVelocity + boundaryVn * normal;
            const Vector du = waves.tangentialIncrement + dBoundaryVn * normal;
            return ConservativeTrace<dimension>({rho, drho}, {p, dp}, u, du, gamma);
        }
    }

    /** Characteristic boundary with spatially constant reservoir primitive data [rho,u...,p]. */
    template <int dimension>
    auto ReservoirCharacteristicTrace(const Eigen::Vector<real, dimension + 2> &inside,
                                      const Eigen::Vector<real, dimension + 2> &insideIncrement,
                                      const Eigen::Vector<real, dimension> &normal, real gamma,
                                      const std::vector<real> &primitive)
    {
        using Vector = Eigen::Vector<real, dimension>;
        return CharacteristicBoundary::Trace<dimension>(inside, insideIncrement, normal, gamma,
            [&](bool needPlus, bool needConvected)
            {
                const Eigen::Map<const Vector> velocity(primitive.data() + 1);
                const real vn = velocity.dot(normal);
                return CharacteristicBoundary::Prescribed<dimension>({primitive.front(), 0}, {primitive.back(), 0},
                    {vn, 0}, gamma, needPlus, needConvected, [&](Vector &tangent, Vector &)
                    { tangent = velocity - vn * normal; });
            });
    }

    template <int dimension>
    auto ReservoirCharacteristicTrace(const Eigen::Vector<real, dimension + 2> &inside,
                                      const Eigen::Vector<real, dimension> &normal, real gamma,
                                      const std::vector<real> &primitive)
    {
        return ReservoirCharacteristicTrace<dimension>(inside, Eigen::Vector<real, dimension + 2>::Zero(),
                                                       normal, gamma, primitive).first;
    }

    /** Eulerian normal flux of the exact one-sided wall Riemann state and its directional derivative.
     * The relative normal Mach number determines the shock, expansion, or detached vacuum at the wall.
     * The normal and prescribed normal velocity are held fixed when differentiating. The NCFV mesh
     * is stationary, so a nonzero prescribed normal velocity contributes Eulerian transport. */
    template <int dimension>
    auto WallCharacteristicFlux(const Eigen::Vector<real, dimension + 2> &inside,
                                const Eigen::Vector<real, dimension + 2> &insideIncrement,
                                const Eigen::Vector<real, dimension> &normal, real gamma, real wallNormalSpeed)
        -> std::pair<Eigen::Vector<real, dimension + 2>, Eigen::Vector<real, dimension + 2>>
    {
        using Vector = Eigen::Vector<real, dimension>;
        using State = Eigen::Vector<real, dimension + 2>;
        using CharacteristicBoundary::Scalar;
        const CharacteristicBoundary::Interior<dimension> left(inside, insideIncrement, normal, gamma);
        const real gm1 = gamma - 1;
        const real mach = (left.vn.value - wallNormalSpeed) / left.sound.value;
        const real dMach = (left.vn.increment - mach * left.sound.increment) / left.sound.value;
        Scalar pressureRatio, densityRatio;
        if (mach > 0)
        {
            // Rankine--Hugoniot wall pressure and the corresponding density jump.
            const real z = .25 * (gamma + 1) * mach;
            const real root = std::hypot(1., z);
            const real waveSpeed = root + z;
            pressureRatio.value = 1 + gamma * mach * waveSpeed;
            pressureRatio.increment = gamma * waveSpeed * (waveSpeed / root) * dMach;
            const real beta = gm1 / (gamma + 1);
            const real denominator = 1 + beta * pressureRatio.value;
            densityRatio.value = (pressureRatio.value + beta) / denominator;
            densityRatio.increment = (1 - beta * beta) * pressureRatio.increment / (denominator * denominator);
        }
        else
        {
            // The outgoing J+ is continuous across the wall expansion fan.
            const real soundRatio = 1 + .5 * gm1 * mach;
            if (soundRatio <= 0)
                return {State::Zero(), State::Zero()};
            densityRatio.value = std::pow(soundRatio, 2 / gm1);
            densityRatio.increment = densityRatio.value * dMach / soundRatio;
            pressureRatio.value = densityRatio.value * soundRatio * soundRatio;
            pressureRatio.increment = gamma * pressureRatio.value * dMach / soundRatio;
        }

        const real p = left.pressure.value * pressureRatio.value;
        const real dp = left.pressure.increment * pressureRatio.value + left.pressure.value * pressureRatio.increment;
        State flux = State::Zero(), differential = State::Zero();
        flux.template segment<dimension>(1) = p * normal;
        differential.template segment<dimension>(1) = dp * normal;
        if (wallNormalSpeed != 0)
        {
            const real rho = left.density.value * densityRatio.value;
            const real drho = left.density.increment * densityRatio.value + left.density.value * densityRatio.increment;
            const Vector u = left.velocity + (wallNormalSpeed - left.vn.value) * normal;
            const Vector du = left.dVelocity - left.vn.increment * normal;
            flux[0] = wallNormalSpeed * rho;
            flux.template segment<dimension>(1) += flux[0] * u;
            flux[dimension + 1] = wallNormalSpeed * (gamma / gm1 * p + .5 * rho * u.squaredNorm());
            differential[0] = wallNormalSpeed * drho;
            differential.template segment<dimension>(1) += differential[0] * u + flux[0] * du;
            differential[dimension + 1] = wallNormalSpeed *
                (gamma / gm1 * dp + .5 * drho * u.squaredNorm() + rho * u.dot(du));
        }
        return {flux, differential};
    }

    /** Characteristic static-pressure outlet. Subsonic flow preserves the interior J+.
     * Outgoing entropy and tangential velocity come from the interior, while a backflow takes
     * those quantities from the zone reservoir. Supersonic outflow passes through unchanged;
     * supersonic inflow takes reservoir density and velocity together with the configured static pressure.
     * The normalized boundary registry supplies the complete reservoir primitive vector for backflow. */
    template <int dimension>
    auto PressureOutletCharacteristicTrace(const Eigen::Vector<real, dimension + 2> &inside,
                                           const Eigen::Vector<real, dimension + 2> &insideIncrement,
                                           const Eigen::Vector<real, dimension> &normal, real gamma,
                                           const BoundaryZoneSettings &boundary)
        -> std::pair<Eigen::Vector<real, dimension + 2>, Eigen::Vector<real, dimension + 2>>
    {
        using Vector = Eigen::Vector<real, dimension>;
        using CharacteristicBoundary::Scalar;
        const CharacteristicBoundary::Interior<dimension> left(inside, insideIncrement, normal, gamma);
        if (left.vn.value >= left.sound.value)
            return {inside, insideIncrement};

        const Scalar pressure{boundary.staticPressure, 0};
        Scalar entropy;
        Vector tangent, dTangent;
        if (left.vn.value < 0)
        {
            DNDS_check_throw_info(boundary.primitive.size() == dimension + 2,
                                  "NCFV pressure outlet backflow requires reservoir primitive data");
            const Eigen::Map<const Vector> reservoirVelocity(boundary.primitive.data() + 1);
            const Scalar reservoirDensity{boundary.primitive.front(), 0};
            if (left.vn.value + left.sound.value < 0)
                return CharacteristicBoundary::ConservativeTrace<dimension>(
                    reservoirDensity, pressure, reservoirVelocity, Vector::Zero(), gamma);
            entropy = CharacteristicBoundary::Entropy(reservoirDensity, {boundary.primitive.back(), 0}, gamma);
            tangent = reservoirVelocity - reservoirVelocity.dot(normal) * normal;
            dTangent.setZero();
        }
        else
        {
            entropy = CharacteristicBoundary::Entropy(left.density, left.pressure, gamma);
            tangent = left.velocity - left.vn.value * normal;
            dTangent = left.dVelocity - left.vn.increment * normal;
        }

        Scalar density;
        density.value = std::pow(pressure.value / entropy.value, 1 / gamma);
        density.increment = -density.value / gamma * (entropy.increment / entropy.value);
        const real sound = std::sqrt(gamma * pressure.value / density.value);
        const real dSound = -.5 * sound * (density.increment / density.value);
        const real vn = left.vn.value + 2 / (gamma - 1) * (left.sound.value - sound);
        const real dvn = left.vn.increment + 2 / (gamma - 1) * (left.sound.increment - dSound);
        return CharacteristicBoundary::ConservativeTrace<dimension>(
            density, pressure, tangent + vn * normal, dTangent + dvn * normal, gamma);
    }
}
