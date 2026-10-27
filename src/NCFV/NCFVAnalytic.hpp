#pragma once

#include "NCFVConfig.hpp"

#include <cmath>

namespace DNDS::NCFV
{
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
