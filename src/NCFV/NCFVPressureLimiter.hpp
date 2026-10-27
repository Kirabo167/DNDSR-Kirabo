#pragma once

#include "DNDS/Defines.hpp"

#include <algorithm>
#include <cmath>

namespace DNDS::NCFV
{
    /**
     * @brief Restrict a conservative mean-to-candidate segment to local pressure bounds.
     * The returned factor bounds the entire prefix [0,factor], so later common
     * factor reductions remain safe even though a pressure upper sublevel set
     * is not convex. The caller must first ensure positive segment densities.
     * @param mean Admissible anchor state [rho,rho*u...,rho*E], with pressure inside the requested interval.
     * @param candidate Candidate conservative state defining mean + theta*(candidate-mean).
     * @param gamma Ideal-gas ratio of specific heats, strictly greater than one.
     * @param pressureMinimum Positive lower pressure bound.
     * @param pressureMaximum Upper pressure bound, at least pressureMinimum.
     * @param factor Initial interpolation-factor cap in [0,1], already restricted to positive segment densities.
     * @return Factor in [0,factor] whose entire interpolation prefix satisfies the pressure bounds within roundoff
     *     tolerance.
     * @tparam dimension Number of momentum components, either 2 or 3.
     * @pre Density is positive throughout the supplied interpolation prefix.
     */
    template <int dimension>
    real LimitIdealGasPressureFactor(
        const Eigen::Vector<real, dimension + 2> &mean,
        const Eigen::Vector<real, dimension + 2> &candidate,
        real gamma, real pressureMinimum, real pressureMaximum, real factor)
    {
        DNDS_assert(gamma > 1 && factor >= 0 && factor <= 1);
        DNDS_assert(pressureMinimum > 0 && pressureMaximum >= pressureMinimum);
        if (factor == 0)
            return 0;
        const Eigen::Vector<real, dimension + 2> increment = candidate - mean;
        const auto momentum = mean.template segment<dimension>(1);
        const auto momentumIncrement = increment.template segment<dimension>(1);
        const real gammaMinusOne = gamma - 1;
        const auto pressure = [&](real theta)
        {
            const real density = mean(0) + theta * increment(0);
            DNDS_assert(density > 0);
            return gammaMinusOne *
                   (mean(dimension + 1) + theta * increment(dimension + 1) -
                    .5 * (momentum + theta * momentumIncrement).squaredNorm() / density);
        };
        const real tolerance = 1e-10 * std::max({real(1), std::abs(pressureMinimum),
                                                std::abs(pressureMaximum)});
        const real lowerPressure = pressureMinimum - tolerance;
        const real upperPressure = pressureMaximum + tolerance;
        DNDS_assert(pressure(0) >= lowerPressure && pressure(0) <= upperPressure);

        // Q(theta)=rho(theta)*(E(theta)-pUpper/(gamma-1))-|m(theta)|^2/2.
        // A positive Q at any point signals an upper-bound violation, including
        // an interior pressure hump with two admissible endpoint states.
        const real quadratic = increment(0) * increment(dimension + 1) -
                               .5 * momentumIncrement.squaredNorm();
        const real linear = mean(0) * increment(dimension + 1) +
                            increment(0) * (mean(dimension + 1) - upperPressure / gammaMinusOne) -
                            momentum.dot(momentumIncrement);
        real upperProbe = factor;
        if (quadratic < 0 && linear > 0 && linear < -2 * quadratic * factor)
            upperProbe = -.5 * linear / quadratic;
        if (pressure(upperProbe) > upperPressure)
        {
            real lower = 0, upper = upperProbe;
            for (int iteration = 0; iteration < 48; iteration++)
            {
                const real middle = .5 * (lower + upper);
                if (pressure(middle) <= upperPressure)
                    lower = middle;
                else
                    upper = middle;
            }
            factor = lower;
        }

        // Pressure is concave along a positive-density conservative segment.
        // Its lower-bound superlevel set is therefore a connected prefix here.
        if (pressure(factor) < lowerPressure)
        {
            real lower = 0, upper = factor;
            for (int iteration = 0; iteration < 48; iteration++)
            {
                const real middle = .5 * (lower + upper);
                if (pressure(middle) >= lowerPressure)
                    lower = middle;
                else
                    upper = middle;
            }
            factor = lower;
        }
        return factor;
    }
}
