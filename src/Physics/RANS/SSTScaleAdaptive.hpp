#pragma once

#include "DNDS/Defines.hpp"
#include <algorithm>
#include <cmath>

namespace DNDS::Physics::RANS
{
    struct SASKinematics
    {
        real velocityLaplacianMagnitude = 0;
        real filterWidth = 0; ///< A positive width selects SAS in the SST source kernel.
    };

    struct SASSource
    {
        real omegaProduction = 0; ///< Source per unit mass.
        real negativeOmegaDerivative = 0; ///< Frozen k, gradients, velocity and blending.
    };

    /// Menter and Egorov (2010), doi:10.1007/s10494-010-9264-5.
    /// The gradient arguments are |grad(k)/k|^2 and |grad(omega)/omega|^2.
    DNDS_DEVICE_CALLABLE inline SASSource EvaluateSASSource(
        real k, real omega, real strainMagnitude, real kGradientRatioSquared,
        real omegaGradientRatioSquared, real betaStar, real sigmaOmega, SASKinematics sas) noexcept
    {
        constexpr real kappa = 0.41;
        constexpr real zeta2 = 3.51;
        constexpr real Cs = 0.11;
        constexpr real C = 2.0;
        constexpr real sigmaPhi = 2.0 / 3.0;
        // SST-1994 gives beta/betaStar - gamma = sigmaOmega*kappa^2/sqrt(betaStar).
        const real equilibriumCoefficient = sigmaOmega * kappa * kappa / std::sqrt(betaStar);
        const real minimumLength = Cs * std::sqrt(kappa * zeta2 / equilibriumCoefficient) * sas.filterWidth;
        const real modelLength = std::sqrt(k) / (std::sqrt(std::sqrt(betaStar)) * omega);
        // Evaluate S/L_vK without dividing by a vanishing velocity Laplacian.
        const real scaleRate = std::min(sas.velocityLaplacianMagnitude / kappa, strainMagnitude / minimumLength);
        const real rate = modelLength * scaleRate;
        const real production = zeta2 * kappa * rate * rate;
        const real damping = (2 * C / sigmaPhi) * k * std::max(kGradientRatioSquared, omegaGradientRatioSquared);
        if (production <= damping)
            return {};

        const real source = production - damping;
        // With frozen gradients, production and the omega damping branch both scale as omega^-2.
        const real differentiatedTerm = omegaGradientRatioSquared > kGradientRatioSquared ? source : production;
        return {source, 2 * differentiatedTerm / omega};
    }
}
