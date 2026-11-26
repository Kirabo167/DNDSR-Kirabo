#pragma once

#include "DNDS/Defines.hpp"
#include <algorithm>
#include <cmath>

namespace DNDS::NCFV
{
    /** Rotation-invariant, one-ring sensor for the cell-local Euler P1 fallback. */
    template <int dimension>
    struct PlanarCompression
    {
        using Vector = Eigen::Vector<real, dimension>;
        using State = Eigen::Vector<real, dimension + 2>;
        using Gradient = Eigen::Matrix<real, dimension, dimension + 2>;

        static real Ramp(real value, real start, real end)
        {
            return std::clamp((value - start) / (end - start), real(0), real(1));
        }

        static Vector PressureGradient(const State &state, const Gradient &gradient, real gamma)
        {
            const Vector velocity = state.template segment<dimension>(1) / state(0);
            return (gamma - 1) * (gradient.col(dimension + 1) -
                                  gradient.template middleCols<dimension>(1) * velocity +
                                  (0.5 * velocity.squaredNorm()) * gradient.col(0));
        }

        template <class Neighbors, class Pressures, class Offset, class NeighborGradient>
        static real Weight(index node, const State &state, const Gradient &gradient, real gamma,
                           const Neighbors &neighbors, const Pressures &pressure,
                           const Offset &offset, const NeighborGradient &neighborPressureGradient)
        {
            real low = pressure[node], high = low;
            for (index other : neighbors)
            {
                low = std::min(low, pressure[other]);
                high = std::max(high, pressure[other]);
            }
            const real jump = Ramp((high - low) / high, 0.02, 0.10);
            if (jump == 0)
                return 0;

            const Vector gradP = PressureGradient(state, gradient, gamma);
            const real magnitude = gradP.norm();
            if (!(magnitude > 0))
                return 0;
            const Vector normal = gradP / magnitude;
            const real normalVelocity = normal.dot(state.template segment<dimension>(1)) / state(0);
            const real compression = -normal.dot(gradient.template middleCols<dimension>(1) * normal -
                                                  gradient.col(0) * normalVelocity) / state(0);
            if (!(compression > 0))
                return 0;

            real normalLength = 0;
            for (index other : neighbors)
                normalLength = std::max(normalLength, std::abs(normal.dot(offset(other))));
            const real soundSpeed = std::sqrt(gamma * pressure[node] / state(0));
            const real strength = Ramp(normalLength * compression / soundSpeed, 0.01, 0.06);
            if (strength == 0)
                return 0;

            // A second shock normal or a reversal of pressure gradient reduces coherence.
            real aligned = gradP.squaredNorm(), total = aligned;
            for (index other : neighbors)
            {
                const Vector otherGrad = neighborPressureGradient(other);
                const real projection = std::max(real(0), normal.dot(otherGrad));
                aligned += projection * projection;
                total += otherGrad.squaredNorm();
            }
            return jump * strength * Ramp(aligned / total, 0.90, 0.98);
        }
    };
}
