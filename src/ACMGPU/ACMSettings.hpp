/**
 * @file ACMSettings.hpp
 * @brief Runtime settings and validation rules for the constant-density ACM kernels.
 *
 * @details The settings use the DNDS configuration registry so a complete case JSON,
 * command-line overrides, and JSON-schema workflows remain reusable by the ACM module.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACM.hpp"
#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Errors.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace DNDS::ACMGPU
{
    /// Physical parameters and numerical options shared by all constant-density ACM face kernels.
    struct Settings
    {
        real rho0 = 1.0;                                              ///< Positive constant density used by momentum fluxes.
        real beta2 = 1.0;                                             ///< Positive artificial-compressibility parameter, beta squared.
        real alpha = 0.0;                                             ///< Arbitrary finite Turkel coupling parameter read from JSON.
        real dynamicViscosity = 0.0;                                  ///< Non-negative laminar dynamic viscosity.
        RiemannSolverType riemannSolverType = RiemannSolverType::Roe; ///< Selected inviscid flux.
        PressureStorage pressureStorage = PressureStorage::PhysicalP; ///< Pressure representation in `State`.
        real entropyFixRatio = 0.05;                                  ///< Relative Harten entropy-fix width.
        real pressureReference = 0.0;                                 ///< Gauge-pressure reference value.
        std::array<real, 4> farFieldValue{0, 0, 0, 0};                ///< Far-field `[u,v,w,p]`.
        bool enableViscousFlux = false;                               ///< Enables the laminar viscous kernel.

        DNDS_DECLARE_CONFIG(Settings)
        {
            DNDS_FIELD(rho0, "Constant density", DNDS::Config::range(0.0));
            DNDS_FIELD(beta2, "Artificial-compressibility beta squared", DNDS::Config::range(0.0));
            DNDS_FIELD(alpha, "Turkel alpha used by Gamma, characteristics, fluxes, and boundaries");
            DNDS_FIELD(dynamicViscosity, "Dynamic viscosity", DNDS::Config::range(0.0));
            DNDS_FIELD(
                riemannSolverType,
                "ACM Riemann solver",
                DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(RiemannSolverType)));
            DNDS_FIELD(
                pressureStorage,
                "Pressure storage policy",
                DNDS::Config::enum_values({"PhysicalP"}));
            DNDS_FIELD(entropyFixRatio, "Entropy-fix ratio", DNDS::Config::range(0.0));
            DNDS_FIELD(pressureReference, "Pressure gauge reference");
            DNDS_FIELD(farFieldValue, "Far-field [u,v,w,p]");
            DNDS_FIELD(enableViscousFlux, "Enable the laminar viscous flux");
            config.post_read([](T &settings)
                             { settings.Validate(); });
        }

        /**
         * @brief Validate all physical and numerical options before a kernel is evaluated.
         * @throws std::runtime_error If a value is non-finite, outside its admissible range, or not
         * supported by this initial implementation.
         */
        void Validate() const
        {
            DNDS_check_throw_info(std::isfinite(rho0) && rho0 > 0, "ACM rho0 must be finite and positive");
            DNDS_check_throw_info(std::isfinite(beta2) && beta2 > 0, "ACM beta2 must be finite and positive");
            DNDS_check_throw_info(std::isfinite(alpha), "ACM alpha must be finite");
            DNDS_check_throw_info(std::isfinite(dynamicViscosity) && dynamicViscosity >= 0,
                                  "ACM dynamicViscosity must be finite and non-negative");
            DNDS_check_throw_info(std::isfinite(entropyFixRatio) && entropyFixRatio >= 0,
                                  "ACM entropyFixRatio must be finite and non-negative");
            DNDS_check_throw_info(std::isfinite(pressureReference), "ACM pressureReference must be finite");
            DNDS_check_throw_info(
                std::all_of(farFieldValue.begin(), farFieldValue.end(), [](real value)
                            { return std::isfinite(value); }),
                "ACM farFieldValue contains a non-finite value");
            DNDS_check_throw_info(pressureStorage == PressureStorage::PhysicalP,
                                  "the initial ACM implementation only supports PhysicalP storage");
        }
    };
}
