#pragma once

#include "NCFVConfig.hpp"
#include "NCFVCharacteristicBoundary.hpp"

#include <cmath>

namespace DNDS::NCFV
{
    /** @brief Smooth forced Euler solution of arXiv:2305.01466, Sec. 6.1.3 (gamma=1.4).
     * The primitive field is [1+.1e,.2+.1e,.1+.2e,.3+.3e,1.3+1.2e],
     * with e=exp(direction dot x + .65t). Rows of the two matrices are the
     * coefficients of e^k in U and in S = U_t + div F(U), respectively. */
    struct ExponentialMMS
    {
        using Basis = Eigen::Vector<real, 5>;
        using Coefficients = Eigen::Matrix<real, 5, 5>;
        inline static const Eigen::Vector3d direction{.75, .35, .65};
        static constexpr real frequency = .65;
        inline static const Coefficients stateCoefficients = (Coefficients() <<
            1, .2, .1, .3, 3.32,
            .1, .12, .21, .33, 3.137,
            0, .01, .02, .03, .083,
            0, 0, 0, 0, .007,
            0, 0, 0, 0, 0).finished();
        inline static const Coefficients sourceCoefficients = (Coefficients() <<
            0, 0, 0, 0, 0,
            .443, 1.0916, .6703, 1.2219, 5.25791,
            .068, .1022, .184, .2862, 3.12014,
            0, .0102, .0204, .0306, .10629,
            0, 0, 0, 0, .00952).finished();

        static Basis PowersOf(real e)
        {
            const real e2 = e * e;
            return {1, e, e2, e2 * e, e2 * e2};
        }

        static Basis Powers(real phase) { return PowersOf(std::exp(phase)); }

        static Basis PhaseDerivative(const Basis &basis)
        {
            return {0, basis[1], 2 * basis[2], 3 * basis[3], 4 * basis[4]};
        }

        static Basis State(const Basis &basis) { return stateCoefficients.transpose() * basis; }

        /** Supply the incoming characteristic data for the manufactured open boundaries.
         * exponential is exp(direction.dot(x) + frequency*t). phaseIncrement is
         * direction.dot(displacement) at fixed stage time. The outgoing amplitudes
         * and their directional increments come from the reconstructed interior. */
        static std::pair<Basis, Basis> CharacteristicTrace(const Basis &inside, const Basis &insideIncrement,
                                                           const Eigen::Vector3d &normal, real exponential,
                                                           real phaseIncrement)
        {
            constexpr real gamma = 1.4;
            return CharacteristicBoundary::Trace<3>(inside, insideIncrement, normal, gamma,
                [&](bool needPlus, bool needConvected)
                {
                    const Eigen::Vector3d base{.2, .1, .3}, slope{.1, .2, .3};
                    const real normalBase = base.dot(normal), normalSlope = slope.dot(normal);
                    const real dExp = exponential * phaseIncrement;
                    return CharacteristicBoundary::Prescribed<3>(
                        {1 + .1 * exponential, .1 * dExp}, {1.3 + 1.2 * exponential, 1.2 * dExp},
                        {normalBase + normalSlope * exponential, normalSlope * dExp}, gamma, needPlus, needConvected,
                        [&](Eigen::Vector3d &tangent, Eigen::Vector3d &dTangent)
                        {
                            const Eigen::Vector3d tangentSlope = slope - normalSlope * normal;
                            tangent = base - normalBase * normal + exponential * tangentSlope;
                            dTangent = dExp * tangentSlope;
                        });
                });
        }

        static Basis CharacteristicTrace(const Basis &inside, const Eigen::Vector3d &normal, real exponential)
        {
            return CharacteristicTrace(inside, Basis::Zero(), normal, exponential, 0).first;
        }
    };

    /**
     * @brief Thesis (3-71), (3-72): exact Euler vortex and its conservative gradient.
     * The primitive background is nondimensional, rho=p=T=1, w=0.
     * @param configuration Vortex strength/center, periodic lengths, gamma, and nondimensional background
     *     primitive state.
     * @param point Physical evaluation coordinate; x/y offsets are mapped to the nearest periodic image.
     * @param time Physical time used to advect the vortex center with the background velocity.
     * @return Pair (state, gradient), with dimension+2 conservative entries and a dimension by dimension+2
     *     derivative matrix.
     * @tparam dimension Spatial dimension, 2 or 3; the 3-D solution is uniform along z.
     * @pre The analytic background has rho=p=T=1 and zero spanwise velocity.
     * @throws std::runtime_error If the vortex produces nonpositive temperature.
     */
    template <int dimension>
    auto IsentropicVortex(const Configuration &configuration,
                         const Eigen::Vector3d &point, real time)
    {
        using State = Eigen::Vector<real, dimension + 2>;
        using Gradient = Eigen::Matrix<real, dimension, dimension + 2>;
        const auto &initial = configuration.initialField;
        const auto &background = configuration.physics.initialPrimitive;
        const real gamma = configuration.physics.gamma;
        Eigen::Vector2d delta;
        for (int d = 0; d < 2; d++)
        {
            delta(d) = point(d) - initial.vortexCenter[d] - time * background[1 + d];
            const real length = configuration.mesh.periodicLengths[d];
            if (length > 0)
                delta(d) -= length * std::round(delta(d) / length);
        }
        const real a = initial.vortexStrength / (2.0 * pi) *
                       std::exp(0.5 * (1.0 - delta.squaredNorm()));
        const real temperature = 1.0 - (gamma - 1.0) / (2.0 * gamma) * a * a;
        DNDS_check_throw_info(temperature > 0, "NCFV vortex has nonpositive temperature");
        const real rho = std::pow(temperature, 1.0 / (gamma - 1.0));
        const real pressure = std::pow(rho, gamma);
        Eigen::Vector<real, dimension> velocity = Eigen::Vector<real, dimension>::Zero();
        velocity(0) = background[1] - a * delta(1);
        velocity(1) = background[2] + a * delta(0);
        State state;
        state(0) = rho;
        state.template segment<dimension>(1) = rho * velocity;
        state(dimension + 1) = pressure / (gamma - 1.0) + 0.5 * rho * velocity.squaredNorm();
        Gradient gradient = Gradient::Zero();
        for (int d = 0; d < 2; d++)
        {
            const real da = -a * delta(d);
            const real dT = -(gamma - 1.0) / gamma * a * da;
            const real drho = rho * dT / ((gamma - 1.0) * temperature);
            const real dp = gamma * pressure / rho * drho;
            Eigen::Vector<real, dimension> dv = Eigen::Vector<real, dimension>::Zero();
            dv(0) = -da * delta(1) - (d == 1 ? a : 0.0);
            dv(1) = da * delta(0) + (d == 0 ? a : 0.0);
            gradient(d, 0) = drho;
            gradient.row(d).template segment<dimension>(1) = (drho * velocity + rho * dv).transpose();
            gradient(d, dimension + 1) = dp / (gamma - 1.0) +
                0.5 * drho * velocity.squaredNorm() + rho * velocity.dot(dv);
        }
        return std::make_pair(state, gradient);
    }
}
