/**
 * @file ACMTurbulence.hpp
 * @brief Runtime-selectable RANS closures for the constant-density ACM solver.
 *
 * @details The turbulence variables are intentionally kept outside the four-component
 * ACM flow state.  The first transported entry is SA nuTilde or turbulent kinetic
 * energy k; the second entry is omega or epsilon for a two-equation closure.  This
 * separation preserves the existing ACM 4x4 characteristic and implicit operators.
 * The formula organization follows the Euler RANS implementation, but this module has
 * no source or link dependency on Euler.
 *
 * @author Runzhi Ma
 * @date 2026-09-02
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACM.hpp"
#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Errors.hpp"

#include <array>

namespace DNDS::ACM
{
    /** @brief Turbulence closures available to the ACM solver at runtime. */
    enum class TurbulenceModel
    {
        Laminar,
        SpalartAllmaras,
        KOmegaWilcox,
        KOmegaSST,
        RealizableKEpsilon,
    };

    DNDS_DEFINE_ENUM_JSON(
        TurbulenceModel,
        {
            {TurbulenceModel::Laminar, "Laminar"},
            {TurbulenceModel::SpalartAllmaras, "SpalartAllmaras"},
            {TurbulenceModel::KOmegaWilcox, "KOmegaWilcox"},
            {TurbulenceModel::KOmegaSST, "KOmegaSST"},
            {TurbulenceModel::RealizableKEpsilon, "RealizableKEpsilon"},
        })

    using TurbulenceState = Eigen::Vector<real, 2>;            ///< `[nuTilde,unused]`, `[k,omega]`, or `[k,epsilon]`.
    using TurbulenceGradient = Eigen::Matrix<real, 3, 2>;      ///< Spatial rows and turbulence-variable columns.
    using VelocityGradient = Eigen::Matrix<real, 3, 3>;        ///< `du_i/dx_j` in global coordinates.

    /**
     * @brief Configuration shared by all ACM turbulence closures and their segregated transport.
     * @note Modifier: Runzhi Ma.
     */
    struct TurbulenceSettings
    {
        TurbulenceModel model = TurbulenceModel::Laminar;      ///< Runtime closure selection.
        std::array<real, 2> initialValue{1e-6, 1.0};            ///< Initial primitive turbulence variables.
        std::array<real, 2> farFieldValue{1e-6, 1.0};           ///< Inflow/far-field primitive turbulence variables.
        std::array<real, 2> minimumValue{1e-12, 1e-10};         ///< Positivity floors applied after every stage.
        std::array<real, 2> maximumValue{1e6, 1e12};            ///< Finite safety ceilings for explicit transport stages.
        real maximumEddyViscosityRatio = 1e5;                  ///< Upper bound `mu_t/mu`.
        real wallOmegaCoefficient = 800.0;                     ///< Wall value `omega=C*nu/d^2`, matching Euler's coefficient.
        bool enableSourceTerms = true;                         ///< Enable production/destruction source terms.
        bool secondOrderReconstruction = true;                 ///< Use limited Green--Gauss face reconstruction.
        int transportSubsteps = 4;                             ///< SSPRK3 turbulence substeps per flow step.
        real transportTimeScale = 0.25;                        ///< Turbulence step relative to the flow pseudo-time step.
        int wallDistanceMethod = 1;                            ///< Geom wall-distance method: 0 brute force, 1 AABB tree.
        int wallDistanceExecution = 0;                         ///< MPI wall-distance execution mode.
        int wallDistanceSubdivide = 0;                         ///< Curved/quad wall subdivision level.
        real minimumWallDistance = 1e-10;                      ///< Positive wall-distance clamp.
        int wallDistanceVerbose = 0;                           ///< Geom wall-distance diagnostic level.

        DNDS_DECLARE_CONFIG(TurbulenceSettings)
        {
            DNDS_FIELD(model, "ACM turbulence closure",
                       DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(TurbulenceModel)));
            DNDS_FIELD(initialValue, "Initial turbulence variables [q1,q2]");
            DNDS_FIELD(farFieldValue, "Far-field turbulence variables [q1,q2]");
            DNDS_FIELD(minimumValue, "Positive turbulence-variable floors [q1,q2]");
            DNDS_FIELD(maximumValue, "Finite turbulence-variable ceilings [q1,q2]");
            DNDS_FIELD(maximumEddyViscosityRatio, "Maximum turbulent-to-molecular viscosity ratio",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(wallOmegaCoefficient, "Wall omega coefficient in omega=C*nu/d^2",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(enableSourceTerms, "Enable turbulence production and destruction sources");
            DNDS_FIELD(secondOrderReconstruction, "Use limited second-order turbulence reconstruction");
            DNDS_FIELD(transportSubsteps, "Turbulence SSPRK3 substeps per ACM flow step",
                       DNDS::Config::range(1));
            DNDS_FIELD(transportTimeScale, "Turbulence-to-flow pseudo-time scale",
                       DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(wallDistanceMethod, "Geom wall-distance method",
                       DNDS::Config::range(0, 1));
            DNDS_FIELD(wallDistanceExecution, "MPI wall-distance execution mode",
                       DNDS::Config::range(0));
            DNDS_FIELD(wallDistanceSubdivide, "Wall-face subdivision level",
                       DNDS::Config::range(0));
            DNDS_FIELD(minimumWallDistance, "Minimum wall distance",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(wallDistanceVerbose, "Wall-distance diagnostic level",
                       DNDS::Config::range(0));
            config.post_read([](T &settings)
                             { settings.Validate(); });
        }

        /**
         * @brief Validate model-independent transport and positivity controls.
         * @throws std::runtime_error If a value is non-finite or outside its admissible range.
         */
        void Validate() const;

        /**
         * @brief Convert the configured initial values to a two-entry Eigen state.
         * @return Clamped initial turbulence state.
         */
        TurbulenceState InitialState() const;

        /**
         * @brief Convert the configured far-field values to a two-entry Eigen state.
         * @return Clamped far-field turbulence state.
         */
        TurbulenceState FarFieldState() const;
    };

    /**
     * @brief Return the number of active transport equations for a closure.
     * @param model Runtime turbulence model.
     * @return Zero for laminar, one for SA, and two for all two-equation models.
     */
    int TurbulenceVariableCount(TurbulenceModel model);

    /**
     * @brief Return the stable JSON/logging name of a turbulence closure.
     * @param model Runtime turbulence model.
     * @return Null-terminated model name matching the JSON enumeration spelling.
     * @note Modifier: Runzhi Ma.
     */
    const char *TurbulenceModelName(TurbulenceModel model);

    /** @brief Return the VTK-HDF field name for an active turbulence variable. */
    const char *TurbulenceVariableName(TurbulenceModel model, int variable);

    /**
     * @brief Apply configured finite bounds and clear entries unused by the selected model.
     * @param state Candidate primitive turbulence state.
     * @param settings Validated turbulence configuration.
     * @return Finite model-admissible state.
     */
    TurbulenceState ClampTurbulenceState(
        const TurbulenceState &state,
        const TurbulenceSettings &settings);

    /**
     * @brief Evaluate turbulent dynamic viscosity for one flow/turbulence state.
     * @param turbulence Primitive turbulence variables.
     * @param velocityGradient Velocity-gradient tensor `du_i/dx_j`.
     * @param turbulenceGradient Gradient of the primitive turbulence variables.
     * @param wallDistance Positive wall distance.
     * @param rho0 Positive constant density.
     * @param molecularViscosity Positive molecular dynamic viscosity.
     * @param settings Closure selection and viscosity-ratio limit.
     * @return Non-negative turbulent dynamic viscosity `mu_t`.
     */
    real TurbulentDynamicViscosity(
        const TurbulenceState &turbulence,
        const VelocityGradient &velocityGradient,
        const TurbulenceGradient &turbulenceGradient,
        real wallDistance,
        real rho0,
        real molecularViscosity,
        const TurbulenceSettings &settings);

    /**
     * @brief Evaluate diffusive turbulence flux projected onto a face normal.
     * @param turbulence Primitive face turbulence variables.
     * @param turbulenceGradient Corrected face gradient.
     * @param unitNormal Unit face normal.
     * @param wallDistance Positive face wall distance.
     * @param rho0 Positive constant density.
     * @param molecularViscosity Molecular dynamic viscosity.
     * @param turbulentViscosity Turbulent dynamic viscosity from TurbulentDynamicViscosity().
     * @param settings Runtime closure selection.
     * @return Primitive-equation diffusion flux; unused entries are zero.
     */
    TurbulenceState TurbulenceDiffusiveFlux(
        const TurbulenceState &turbulence,
        const TurbulenceGradient &turbulenceGradient,
        const Vector3 &unitNormal,
        real wallDistance,
        real rho0,
        real molecularViscosity,
        real turbulentViscosity,
        const TurbulenceSettings &settings);

    /**
     * @brief Evaluate turbulence production/destruction source rates.
     * @param turbulence Primitive cell turbulence variables.
     * @param velocityGradient Cell velocity-gradient tensor.
     * @param turbulenceGradient Cell turbulence gradients.
     * @param wallDistance Positive cell wall distance.
     * @param rho0 Positive constant density.
     * @param molecularViscosity Molecular dynamic viscosity.
     * @param settings Runtime closure and source switch.
     * @return Primitive-variable source rate; unused entries are zero.
     */
    TurbulenceState TurbulenceSource(
        const TurbulenceState &turbulence,
        const VelocityGradient &velocityGradient,
        const TurbulenceGradient &turbulenceGradient,
        real wallDistance,
        real rho0,
        real molecularViscosity,
        const TurbulenceSettings &settings);

    /**
     * @brief Construct a turbulence ghost state using ACM boundary semantics.
     * @param boundaryType ACM boundary family at the face.
     * @param interior Interior turbulence state at a face or cell center.
     * @param wallDistance Positive owner-cell wall distance.
     * @param rho0 Positive constant density.
     * @param molecularViscosity Molecular dynamic viscosity.
     * @param settings Runtime closure and far-field values.
     * @return Ghost state imposing wall, inflow/far-field, or extrapolation data.
     */
    TurbulenceState GenerateTurbulenceBoundaryState(
        BoundaryType boundaryType,
        const TurbulenceState &interior,
        real wallDistance,
        real rho0,
        real molecularViscosity,
        const TurbulenceSettings &settings);
}
