#pragma once

#include "Physics/RANS/Point.hpp"

namespace DNDS::Physics::RANS
{
    /// Conservative no-slip RANS target. Use molecular viscosity at the wall
    /// temperature for an isothermal wall and at the interior mean otherwise.
    /// The caller supplies positive wall distance, mean density and trace density.
    DNDS_DEVICE_CALLABLE inline std::array<real, 2> PrepareRANSWallTarget(
        ModelKind kind, const Point &mean, real interiorDensity) noexcept
    {
        std::array<real, 2> values{};
        if (kind == ModelKind::KOmegaWilcox || kind == ModelKind::KOmegaSST)
            values[1] = mean.molecularViscosity / sqr(mean.wallDistance) * kWallOmegaCoeff;
        else if (kind == ModelKind::RealizableKE)
            values[1] = 2 * mean.molecularViscosity / mean.state[0] *
                        (mean.state[5] / mean.state[0]) / sqr(mean.wallDistance) * interiorDensity;
        return values;
    }
}
