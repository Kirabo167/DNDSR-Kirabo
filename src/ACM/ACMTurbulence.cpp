/**
 * @file ACMTurbulence.cpp
 * @brief Constant-density turbulence-closure kernels used by the ACM transport module.
 *
 * @details The formulas mirror the model families available in the Euler solver while
 * operating on primitive constant-density variables.  They are implemented locally so
 * the ACM library neither includes nor links against Euler.  The isotropic `2 rho k/3`
 * stress is absorbed into the ACM pressure; `mu_t` therefore augments only the deviatoric
 * velocity-gradient stress in the four-variable flow equations.
 *
 * @author Runzhi Ma
 * @date 2026-09-02
 * @note Modifier: Runzhi Ma.
 */

#include "ACMTurbulence.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace DNDS::ACM
{
    namespace
    {
        constexpr real betaStar = 0.09;
        constexpr real minimumClosureValue = 1e-30;

        /**
         * @brief Return a positive finite closure denominator.
         * @param value Candidate denominator.
         * @return A denominator no smaller than `1e-30`; positive infinity is retained.
         * @note Modifier: Runzhi Ma.
         */
        real PositiveDenominator(real value)
        {
            if (std::isnan(value) || value < minimumClosureValue)
                return minimumClosureValue;
            return value;
        }

        /**
         * @brief Evaluate the SA `f_v1` damping function without cubing a large `chi`.
         * @param chi Non-negative modified-to-molecular viscosity ratio.
         * @return A finite value in `[0,1]`, including the asymptotic value for `chi=+inf`.
         * @note Modifier: Runzhi Ma.
         */
        real SAViscosityDamping(real chi)
        {
            constexpr real cv1 = 7.1;
            if (!(chi > 0))
                return 0;
            if (!std::isfinite(chi) || chi >= cv1 * 1e6)
                return 1;
            const real ratio = chi / cv1;
            const real ratio3 = ratio * ratio * ratio;
            return ratio3 / (1.0 + ratio3);
        }

        /**
         * @brief Evaluate the SA `f_v2` function using a reciprocal form stable at large `chi`.
         * @param chi Non-negative modified-to-molecular viscosity ratio.
         * @param fv1 Previously evaluated `f_v1` damping value.
         * @return Finite SA `f_v2` value.
         * @note Modifier: Runzhi Ma.
         */
        real SASecondaryDamping(real chi, real fv1)
        {
            if (!(chi > 0))
                return 1;
            const real ratio = 1.0 / (1.0 / chi + fv1);
            return 1.0 - ratio;
        }

        /**
         * @brief Build the deviatoric strain tensor for the constant-density velocity field.
         * @param velocityGradient Tensor with entries `du_i/dx_j`.
         * @return Deviatoric strain tensor `S_ij-div(u) delta_ij/3`.
         * @note Modifier: Runzhi Ma.
         */
        VelocityGradient DeviatoricStrain(const VelocityGradient &velocityGradient)
        {
            VelocityGradient strain = 0.5 * (velocityGradient + velocityGradient.transpose());
            strain.diagonal().array() -= velocityGradient.trace() / 3.0;
            return strain;
        }

        /**
         * @brief Compute the strain-rate magnitude used by the two-equation closures.
         * @param velocityGradient Tensor with entries `du_i/dx_j`.
         * @return `sqrt(2 S_ij S_ij)`.
         * @note Modifier: Runzhi Ma.
         */
        real StrainMagnitude(const VelocityGradient &velocityGradient)
        {
            return std::sqrt(2.0 * DeviatoricStrain(velocityGradient).squaredNorm());
        }

        /**
         * @brief Compute the full strain-rate magnitude used by the Wilcox stress limiter.
         * @param velocityGradient Tensor with entries `du_i/dx_j`.
         * @return `sqrt(2 S_ij S_ij)` before removal of the isotropic strain.
         * @note Modifier: Runzhi Ma.
         */
        real FullStrainMagnitude(const VelocityGradient &velocityGradient)
        {
            const VelocityGradient strain =
                0.5 * (velocityGradient + velocityGradient.transpose());
            return std::sqrt(2.0 * strain.squaredNorm());
        }

        /**
         * @brief Compute the vorticity-tensor magnitude used by SA and SST.
         * @param velocityGradient Tensor with entries `du_i/dx_j`.
         * @return `sqrt(2 Omega_ij Omega_ij)`.
         * @note Modifier: Runzhi Ma.
         */
        real VorticityMagnitude(const VelocityGradient &velocityGradient)
        {
            const VelocityGradient rotation =
                0.5 * (velocityGradient - velocityGradient.transpose());
            return std::sqrt(2.0 * rotation.squaredNorm());
        }

        /** @brief SST blending-function pair evaluated from primitive k-omega data. */
        struct SSTBlending
        {
            real f1 = 0; ///< Near-wall blending function.
            real f2 = 0; ///< Eddy-viscosity shielding function.
        };

        /**
         * @brief Evaluate robust Menter SST `F1` and `F2` blending functions.
         * @param turbulence Primitive `[k,omega]` state.
         * @param turbulenceGradient Primitive gradient of `[k,omega]`.
         * @param wallDistance Positive wall distance.
         * @param rho0 Constant density.
         * @param molecularViscosity Molecular dynamic viscosity.
         * @return Bounded blending factors in `[0,1]`.
         * @note Modifier: Runzhi Ma.
         */
        SSTBlending EvaluateSSTBlending(
            const TurbulenceState &turbulence,
            const TurbulenceGradient &turbulenceGradient,
            real wallDistance,
            real rho0,
            real molecularViscosity)
        {
            constexpr real sigmaOmega2 = 0.856;
            const real k = PositiveDenominator(turbulence(0));
            const real omega = PositiveDenominator(turbulence(1));
            const real d = PositiveDenominator(wallDistance);
            const real nu = molecularViscosity / PositiveDenominator(rho0);
            const real cross = turbulenceGradient.col(0).dot(turbulenceGradient.col(1));
            const real cdKw = std::max(
                2.0 * rho0 * sigmaOmega2 * cross / omega,
                1e-10);

            const real nearWall = std::sqrt(k) / (betaStar * omega * d);
            const real viscousWall = 500.0 * nu / (d * d * omega);
            const real crossWall = 4.0 * rho0 * sigmaOmega2 * k / (cdKw * d * d);
            const real argument1 = std::clamp(
                std::min(std::max(nearWall, viscousWall), crossWall),
                0.0,
                10.0);
            const real argument2 = std::clamp(
                std::max(2.0 * nearWall, viscousWall),
                0.0,
                10.0);
            return {
                std::tanh(std::pow(argument1, 4)),
                std::tanh(argument2 * argument2)};
        }

        /**
         * @brief Limit a computed eddy viscosity to the configured molecular-viscosity ratio.
         * @param value Candidate dynamic eddy viscosity.
         * @param molecularViscosity Molecular dynamic viscosity.
         * @param settings Turbulence settings containing the ratio cap.
         * @return Finite non-negative capped viscosity.
         * @note Modifier: Runzhi Ma.
         */
        real LimitEddyViscosity(
            real value,
            real molecularViscosity,
            const TurbulenceSettings &settings)
        {
            DNDS_check_throw_info(!std::isnan(value),
                                  "ACM turbulence closure produced NaN eddy viscosity");
            const real ratioCap = settings.maximumEddyViscosityRatio;
            const real finiteCap = ratioCap > std::numeric_limits<real>::max() /
                                                   molecularViscosity
                                       ? std::numeric_limits<real>::max()
                                       : ratioCap * molecularViscosity;
            if (value <= 0)
                return 0;
            if (!std::isfinite(value))
                return finiteCap;
            return std::min(value, finiteCap);
        }
    }

    /** @copydoc TurbulenceSettings::Validate */
    void TurbulenceSettings::Validate() const
    {
        const auto finitePair = [](const std::array<real, 2> &values)
        {
            return std::all_of(values.begin(), values.end(), [](real value)
                               { return std::isfinite(value); });
        };
        DNDS_check_throw_info(
            finitePair(initialValue) && finitePair(farFieldValue) &&
                finitePair(minimumValue) && finitePair(maximumValue),
            "ACM turbulence values must be finite");
        DNDS_check_throw_info(minimumValue[0] > 0 && minimumValue[1] > 0,
                              "ACM turbulence positivity floors must be positive");
        DNDS_check_throw_info(
            maximumValue[0] >= minimumValue[0] &&
                maximumValue[1] >= minimumValue[1],
            "ACM turbulence ceilings must not be below their positivity floors");
        DNDS_check_throw_info(
            std::isfinite(maximumEddyViscosityRatio) && maximumEddyViscosityRatio >= 0,
            "ACM maximum eddy-viscosity ratio must be finite and non-negative");
        DNDS_check_throw_info(
            std::isfinite(wallOmegaCoefficient) && wallOmegaCoefficient >= 0,
            "ACM wall-omega coefficient must be finite and non-negative");
        DNDS_check_throw_info(transportSubsteps > 0,
                              "ACM turbulence transportSubsteps must be positive");
        DNDS_check_throw_info(
            std::isfinite(transportTimeScale) && transportTimeScale > 0 &&
                transportTimeScale <= 1,
            "ACM turbulence transportTimeScale must lie in (0,1]");
        DNDS_check_throw_info(wallDistanceMethod >= 0 && wallDistanceMethod <= 1,
                              "ACM wall-distance method must be 0 or 1");
        DNDS_check_throw_info(
            wallDistanceExecution >= 0 && wallDistanceSubdivide >= 0 && wallDistanceVerbose >= 0,
            "ACM wall-distance integer controls must be non-negative");
        DNDS_check_throw_info(
            std::isfinite(minimumWallDistance) && minimumWallDistance > 0,
            "ACM minimum wall distance must be finite and positive");
    }

    /** @copydoc TurbulenceSettings::InitialState */
    TurbulenceState TurbulenceSettings::InitialState() const
    {
        return ClampTurbulenceState(
            Eigen::Map<const TurbulenceState>(initialValue.data()),
            *this);
    }

    /** @copydoc TurbulenceSettings::FarFieldState */
    TurbulenceState TurbulenceSettings::FarFieldState() const
    {
        return ClampTurbulenceState(
            Eigen::Map<const TurbulenceState>(farFieldValue.data()),
            *this);
    }

    /** @copydoc TurbulenceVariableCount */
    int TurbulenceVariableCount(TurbulenceModel model)
    {
        switch (model)
        {
        case TurbulenceModel::Laminar:
            return 0;
        case TurbulenceModel::SpalartAllmaras:
            return 1;
        case TurbulenceModel::KOmegaWilcox:
        case TurbulenceModel::KOmegaSST:
        case TurbulenceModel::RealizableKEpsilon:
            return 2;
        }
        DNDS_check_throw_info(false, "Unknown ACM turbulence model");
        return 0;
    }

    /** @copydoc TurbulenceModelName */
    const char *TurbulenceModelName(TurbulenceModel model)
    {
        switch (model)
        {
        case TurbulenceModel::Laminar:
            return "Laminar";
        case TurbulenceModel::SpalartAllmaras:
            return "SpalartAllmaras";
        case TurbulenceModel::KOmegaWilcox:
            return "KOmegaWilcox";
        case TurbulenceModel::KOmegaSST:
            return "KOmegaSST";
        case TurbulenceModel::RealizableKEpsilon:
            return "RealizableKEpsilon";
        }
        DNDS_check_throw_info(false, "Unknown ACM turbulence model");
        return "Unknown";
    }

    const char *TurbulenceVariableName(TurbulenceModel model, int variable)
    {
        DNDS_check_throw_info(variable >= 0 && variable < TurbulenceVariableCount(model),
                              "Invalid ACM turbulence variable index");
        if (model == TurbulenceModel::SpalartAllmaras)
            return "TurbulenceNuTilde";
        if (variable == 0)
            return "TurbulenceK";
        return model == TurbulenceModel::RealizableKEpsilon
                   ? "TurbulenceEpsilon"
                   : "TurbulenceOmega";
    }

    /** @copydoc ClampTurbulenceState */
    TurbulenceState ClampTurbulenceState(
        const TurbulenceState &state,
        const TurbulenceSettings &settings)
    {
        TurbulenceState result = TurbulenceState::Zero();
        const int nActive = TurbulenceVariableCount(settings.model);
        for (int variable = 0; variable < nActive; variable++)
        {
            const std::size_t component = static_cast<std::size_t>(variable);
            real candidate = state(variable);
            if (std::isnan(candidate) || candidate == -std::numeric_limits<real>::infinity())
                candidate = settings.minimumValue[component];
            else if (candidate == std::numeric_limits<real>::infinity())
                candidate = settings.maximumValue[component];
            result(variable) = std::clamp(
                candidate,
                settings.minimumValue[component],
                settings.maximumValue[component]);
        }
        return result;
    }

    /** @copydoc TurbulentDynamicViscosity */
    real TurbulentDynamicViscosity(
        const TurbulenceState &turbulenceInput,
        const VelocityGradient &velocityGradient,
        const TurbulenceGradient &turbulenceGradient,
        real wallDistance,
        real rho0,
        real molecularViscosity,
        const TurbulenceSettings &settings)
    {
        DNDS_check_throw_info(
            velocityGradient.allFinite() && turbulenceGradient.allFinite() &&
                std::isfinite(wallDistance) && wallDistance > 0 &&
                std::isfinite(rho0) && rho0 > 0 &&
                std::isfinite(molecularViscosity) && molecularViscosity > 0,
            "Invalid input to ACM turbulent-viscosity closure");
        const TurbulenceState turbulence =
            ClampTurbulenceState(turbulenceInput, settings);

        if (settings.model == TurbulenceModel::Laminar)
            return 0;
        if (settings.model == TurbulenceModel::SpalartAllmaras)
        {
            const real nuTilde = turbulence(0);
            const real chi = rho0 * nuTilde / molecularViscosity;
            const real fv1 = SAViscosityDamping(chi);
            return LimitEddyViscosity(
                rho0 * nuTilde * std::max(fv1, 0.0),
                molecularViscosity,
                settings);
        }
        if (settings.model == TurbulenceModel::KOmegaWilcox)
        {
            constexpr real cLimit = 7.0 / 8.0;
            const real k = turbulence(0);
            const real omegaLimited = std::max(
                turbulence(1),
                cLimit * FullStrainMagnitude(velocityGradient) / std::sqrt(betaStar));
            return LimitEddyViscosity(
                rho0 * k / PositiveDenominator(omegaLimited),
                molecularViscosity,
                settings);
        }
        if (settings.model == TurbulenceModel::KOmegaSST)
        {
            constexpr real a1 = 0.31;
            const SSTBlending blending = EvaluateSSTBlending(
                turbulence,
                turbulenceGradient,
                wallDistance,
                rho0,
                molecularViscosity);
            const real denominator = std::max(
                a1 * turbulence(1),
                VorticityMagnitude(velocityGradient) * blending.f2);
            return LimitEddyViscosity(
                rho0 * a1 * turbulence(0) / PositiveDenominator(denominator),
                molecularViscosity,
                settings);
        }

        constexpr real cMu = 0.09;
        const real k = turbulence(0);
        const real epsilon = turbulence(1);
        const real turbulentReynolds =
            rho0 * k * k / PositiveDenominator(molecularViscosity * epsilon);
        const real sqrtReynolds = std::sqrt(std::max(turbulentReynolds, 0.0));
        const real numerator = -std::expm1(-0.01 * turbulentReynolds);
        const real denominator = std::max(-std::expm1(-sqrtReynolds), 1e-14);
        const real fMu = numerator / denominator *
                         std::max(1.0, std::sqrt(2.0 / PositiveDenominator(turbulentReynolds)));
        real eddyViscosity = cMu * fMu * rho0 * k * k / PositiveDenominator(epsilon);
        eddyViscosity = std::min(
            eddyViscosity,
            (2.0 / 3.0) * rho0 * k /
                PositiveDenominator(StrainMagnitude(velocityGradient)));
        return LimitEddyViscosity(eddyViscosity, molecularViscosity, settings);
    }

    /** @copydoc TurbulenceDiffusiveFlux */
    TurbulenceState TurbulenceDiffusiveFlux(
        const TurbulenceState &turbulenceInput,
        const TurbulenceGradient &turbulenceGradient,
        const Vector3 &unitNormalInput,
        real wallDistance,
        real rho0,
        real molecularViscosity,
        real turbulentViscosity,
        const TurbulenceSettings &settings)
    {
        DNDS_check_throw_info(
            turbulenceGradient.allFinite() && unitNormalInput.allFinite() &&
                unitNormalInput.norm() > 0 && std::isfinite(turbulentViscosity) &&
                turbulentViscosity >= 0,
            "Invalid input to ACM turbulence diffusion closure");
        const Vector3 unitNormal = unitNormalInput.normalized();
        const TurbulenceState turbulence =
            ClampTurbulenceState(turbulenceInput, settings);
        TurbulenceState flux = TurbulenceState::Zero();
        if (settings.model == TurbulenceModel::Laminar)
            return flux;

        const Eigen::RowVector<real, 2> normalDerivative =
            unitNormal.transpose() * turbulenceGradient;
        if (settings.model == TurbulenceModel::SpalartAllmaras)
        {
            constexpr real sigma = 2.0 / 3.0;
            flux(0) = normalDerivative(0) *
                      (molecularViscosity / rho0 + turbulence(0)) / sigma;
        }
        else if (settings.model == TurbulenceModel::KOmegaWilcox)
        {
            constexpr real sigmaK = 0.5;
            constexpr real sigmaOmega = 0.5;
            flux(0) = normalDerivative(0) *
                      (molecularViscosity + sigmaK * turbulentViscosity) / rho0;
            flux(1) = normalDerivative(1) *
                      (molecularViscosity + sigmaOmega * turbulentViscosity) / rho0;
        }
        else if (settings.model == TurbulenceModel::KOmegaSST)
        {
            constexpr real sigmaK1 = 0.85;
            constexpr real sigmaK2 = 1.0;
            constexpr real sigmaOmega1 = 0.5;
            constexpr real sigmaOmega2 = 0.856;
            const SSTBlending blending = EvaluateSSTBlending(
                turbulence,
                turbulenceGradient,
                wallDistance,
                rho0,
                molecularViscosity);
            const real sigmaK = sigmaK1 * blending.f1 + sigmaK2 * (1.0 - blending.f1);
            const real sigmaOmega =
                sigmaOmega1 * blending.f1 + sigmaOmega2 * (1.0 - blending.f1);
            flux(0) = normalDerivative(0) *
                      (molecularViscosity + sigmaK * turbulentViscosity) / rho0;
            flux(1) = normalDerivative(1) *
                      (molecularViscosity + sigmaOmega * turbulentViscosity) / rho0;
        }
        else
        {
            constexpr real sigmaK = 1.0;
            constexpr real sigmaEpsilon = 1.3;
            flux(0) = normalDerivative(0) *
                      (molecularViscosity + turbulentViscosity / sigmaK) / rho0;
            flux(1) = normalDerivative(1) *
                      (molecularViscosity + turbulentViscosity / sigmaEpsilon) / rho0;
        }
        DNDS_check_throw_info(flux.allFinite(),
                              "ACM turbulence diffusion closure produced non-finite flux");
        return flux;
    }

    /** @copydoc TurbulenceSource */
    TurbulenceState TurbulenceSource(
        const TurbulenceState &turbulenceInput,
        const VelocityGradient &velocityGradient,
        const TurbulenceGradient &turbulenceGradient,
        real wallDistance,
        real rho0,
        real molecularViscosity,
        const TurbulenceSettings &settings)
    {
        TurbulenceState source = TurbulenceState::Zero();
        if (settings.model == TurbulenceModel::Laminar || !settings.enableSourceTerms)
            return source;
        DNDS_check_throw_info(
            velocityGradient.allFinite() && turbulenceGradient.allFinite() &&
                std::isfinite(wallDistance) && wallDistance > 0 &&
                std::isfinite(rho0) && rho0 > 0 &&
                std::isfinite(molecularViscosity) && molecularViscosity > 0,
            "Invalid input to ACM turbulence source closure");

        const TurbulenceState turbulence =
            ClampTurbulenceState(turbulenceInput, settings);
        const real d = std::max(wallDistance, settings.minimumWallDistance);
        const real nu = molecularViscosity / rho0;
        const real strainMagnitude = StrainMagnitude(velocityGradient);
        const real productionInvariant = strainMagnitude * strainMagnitude;

        if (settings.model == TurbulenceModel::SpalartAllmaras)
        {
            constexpr real cb1 = 0.1355;
            constexpr real cb2 = 0.622;
            constexpr real sigma = 2.0 / 3.0;
            constexpr real cw2 = 0.3;
            constexpr real cw3 = 2.0;
            constexpr real kappa = 0.41;
            constexpr real ct3 = 1.2;
            constexpr real ct4 = 0.5;
            constexpr real rLimit = 10.0;
            constexpr real cw1 = cb1 / (kappa * kappa) + (1.0 + cb2) / sigma;

            const real nuTilde = turbulence(0);
            const real chi = nuTilde / PositiveDenominator(nu);
            const real fv1 = SAViscosityDamping(chi);
            const real fv2 = SASecondaryDamping(chi, fv1);
            const real ft2 = chi > 50.0
                                 ? 0.0
                                 : ct3 * std::exp(-ct4 * chi * chi);
            const real modifiedVorticity = std::max(
                VorticityMagnitude(velocityGradient) +
                    nuTilde * fv2 / (kappa * kappa * d * d),
                minimumClosureValue);
            const real r = std::clamp(
                nuTilde /
                    PositiveDenominator(modifiedVorticity * kappa * kappa * d * d),
                0.0,
                rLimit);
            const real r6 = std::pow(r, 6);
            const real g = r + cw2 * (r6 - r);
            const real fw = g * std::pow(
                                    (1.0 + std::pow(cw3, 6)) /
                                        PositiveDenominator(std::pow(g, 6) + std::pow(cw3, 6)),
                                    1.0 / 6.0);
            const real production = cb1 * (1.0 - ft2) * modifiedVorticity * nuTilde;
            const real destruction =
                (cw1 * fw - cb1 * ft2 / (kappa * kappa)) *
                (nuTilde / d) * (nuTilde / d);
            const real gradientSource =
                cb2 / sigma * turbulenceGradient.col(0).squaredNorm();
            source(0) = production - destruction + gradientSource;
        }
        else
        {
            const real k = turbulence(0);
            const real secondVariable = turbulence(1);
            const real turbulentViscosity = TurbulentDynamicViscosity(
                turbulence,
                velocityGradient,
                turbulenceGradient,
                d,
                rho0,
                molecularViscosity,
                settings);
            const real rawProductionK =
                turbulentViscosity * productionInvariant / rho0 -
                (2.0 / 3.0) * k * velocityGradient.trace();
            const real productionK = std::max(rawProductionK, 0.0);

            if (settings.model == TurbulenceModel::KOmegaWilcox)
            {
                constexpr real alpha = 13.0 / 25.0;
                constexpr real beta0 = 0.0708;
                const VelocityGradient strainTwice =
                    velocityGradient + velocityGradient.transpose();
                const VelocityGradient rotationTwice =
                    velocityGradient.transpose() - velocityGradient;
                const real chiOmega = std::abs(
                    ((rotationTwice * rotationTwice).array() * strainTwice.array()).sum() *
                    0.125 /
                    PositiveDenominator(std::pow(betaStar * secondVariable, 3)));
                const real fBeta = !std::isfinite(chiOmega) || chiOmega > 1e100
                                       ? 0.85
                                       : (1.0 + 85.0 * chiOmega) /
                                             (1.0 + 100.0 * chiOmega);
                const real beta = beta0 * fBeta;
                const real limitedProduction = std::min(
                    productionK,
                    20.0 * betaStar * k * secondVariable);
                const real cross = turbulenceGradient.col(0).dot(turbulenceGradient.col(1));
                const real sigmaD = cross > 0 ? 0.125 : 0.0;
                source(0) = limitedProduction - betaStar * k * secondVariable;
                source(1) = alpha * secondVariable / PositiveDenominator(k) * limitedProduction -
                            beta * secondVariable * secondVariable +
                            sigmaD * cross / PositiveDenominator(secondVariable);
            }
            else if (settings.model == TurbulenceModel::KOmegaSST)
            {
                constexpr real sigmaOmega2 = 0.856;
                constexpr real beta1 = 0.075;
                constexpr real beta2 = 0.0828;
                constexpr real sigmaOmega1 = 0.5;
                constexpr real kappa = 0.41;
                const real gamma1 = beta1 / betaStar -
                                    sigmaOmega1 * kappa * kappa / std::sqrt(betaStar);
                const real gamma2 = beta2 / betaStar -
                                    sigmaOmega2 * kappa * kappa / std::sqrt(betaStar);
                const SSTBlending blending = EvaluateSSTBlending(
                    turbulence,
                    turbulenceGradient,
                    d,
                    rho0,
                    molecularViscosity);
                const real beta = beta1 * blending.f1 + beta2 * (1.0 - blending.f1);
                const real gamma = gamma1 * blending.f1 + gamma2 * (1.0 - blending.f1);
                const real limitedProduction = std::min(
                    productionK,
                    20.0 * betaStar * k * secondVariable);
                const real cross = turbulenceGradient.col(0).dot(turbulenceGradient.col(1));
                source(0) = limitedProduction - betaStar * k * secondVariable;
                source(1) = gamma * productionInvariant - beta * secondVariable * secondVariable +
                            2.0 * (1.0 - blending.f1) * sigmaOmega2 * cross /
                                PositiveDenominator(secondVariable);
            }
            else
            {
                constexpr real cMu = 0.09;
                constexpr real cEpsilon1 = 1.44;
                constexpr real cEpsilon2 = 1.92;
                constexpr real extraDissipationCoefficient = 0.3;
                constexpr real productionLimiterCoefficient = 1.065;
                const real turbulentReynolds =
                    rho0 * k * k /
                    PositiveDenominator(molecularViscosity * secondVariable);
                const real turbulentTime =
                    k / PositiveDenominator(secondVariable) *
                    std::max(1.0, 1.0 / PositiveDenominator(std::sqrt(turbulentReynolds / 2.0)));

                // Euler transports rho*k and rho*epsilon.  Dividing its realizable-model
                // source by the constant rho0 gives the primitive source used here.  Keep
                // the intended epsilon entry (Euler's current source helper contains an
                // obvious k/epsilon index typo at this point) while retaining its Shih
                // production cap and E correction.  Modifier: Runzhi Ma.
                real limitedProduction = rawProductionK;
                if (turbulentViscosity > minimumClosureValue)
                    limitedProduction = std::min(
                        limitedProduction,
                        productionLimiterCoefficient * cMu * rho0 * k * k /
                            turbulentViscosity);

                const real epsilon = secondVariable;
                const Vector3 turbulentTimeGradient =
                    turbulenceGradient.col(0) / PositiveDenominator(epsilon) -
                    k * turbulenceGradient.col(1) /
                        PositiveDenominator(epsilon * epsilon);
                const real psi = std::max(
                    turbulenceGradient.col(0).dot(turbulentTimeGradient),
                    0.0);
                const real lengthScale = std::max(
                    std::sqrt(k),
                    std::pow(
                        molecularViscosity * epsilon / rho0,
                        0.25));
                const real extraDissipation =
                    extraDissipationCoefficient *
                    std::sqrt(epsilon) * std::sqrt(turbulentTime) *
                    psi * lengthScale;

                source(0) = limitedProduction - epsilon;
                source(1) =
                    (cEpsilon1 * limitedProduction -
                     cEpsilon2 * epsilon + extraDissipation) /
                    PositiveDenominator(turbulentTime);
            }
        }

        DNDS_check_throw_info(source.allFinite(),
                              "ACM turbulence source closure produced non-finite values");
        return source;
    }

    /** @copydoc GenerateTurbulenceBoundaryState */
    TurbulenceState GenerateTurbulenceBoundaryState(
        BoundaryType boundaryType,
        const TurbulenceState &interior,
        real wallDistance,
        real rho0,
        real molecularViscosity,
        const TurbulenceSettings &settings)
    {
        DNDS_check_throw_info(
            interior.allFinite() && std::isfinite(wallDistance) && wallDistance > 0 &&
                std::isfinite(rho0) && rho0 > 0 &&
                std::isfinite(molecularViscosity) && molecularViscosity > 0,
            "Invalid input to ACM turbulence boundary state");
        if (settings.model == TurbulenceModel::Laminar)
            return TurbulenceState::Zero();

        if (boundaryType == BoundaryType::BCWall ||
            boundaryType == BoundaryType::BCWallIsothermal)
        {
            TurbulenceState faceValue = TurbulenceState::Zero();
            if (settings.model == TurbulenceModel::KOmegaWilcox ||
                settings.model == TurbulenceModel::KOmegaSST)
            {
                const real d = std::max(wallDistance, settings.minimumWallDistance);
                faceValue(1) = settings.wallOmegaCoefficient *
                               molecularViscosity / rho0 / (d * d);
                faceValue(1) = std::min(faceValue(1), settings.maximumValue[1]);
            }
            else if (settings.model == TurbulenceModel::RealizableKEpsilon)
            {
                const real d = std::max(wallDistance, settings.minimumWallDistance);
                const real k = std::max(interior(0), settings.minimumValue[0]);
                faceValue(1) = std::clamp(
                    2.0 * molecularViscosity / rho0 * k / (d * d),
                    settings.minimumValue[1],
                    settings.maximumValue[1]);
            }
            return 2.0 * faceValue - interior;
        }

        if (boundaryType == BoundaryType::BCFar ||
            boundaryType == BoundaryType::BCIn ||
            boundaryType == BoundaryType::BCInPsTs ||
            boundaryType == BoundaryType::BCSpecial)
            return 2.0 * settings.FarFieldState() - interior;

        // Outflow, pressure outflow, inviscid wall, and symmetry use zero normal gradient.
        return interior;
    }
}
