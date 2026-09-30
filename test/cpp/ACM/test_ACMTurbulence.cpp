/**
 * @file test_ACMTurbulence.cpp
 * @brief Unit tests for modular constant-density ACM turbulence closures.
 *
 * @author Runzhi Ma
 * @date 2026-09-02
 * @note Modifier: Runzhi Ma.
 */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "ACM/ACMTurbulence.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>

using namespace DNDS;
using namespace DNDS::ACM;

/// @test Verify runtime model selection and active segregated-state sizes.
TEST_CASE("ACM turbulence model configuration is runtime selectable")
{
    CHECK(TurbulenceVariableCount(TurbulenceModel::Laminar) == 0);
    CHECK(TurbulenceVariableCount(TurbulenceModel::SpalartAllmaras) == 1);
    CHECK(TurbulenceVariableCount(TurbulenceModel::KOmegaWilcox) == 2);
    CHECK(TurbulenceVariableCount(TurbulenceModel::KOmegaSST) == 2);
    CHECK(TurbulenceVariableCount(TurbulenceModel::RealizableKEpsilon) == 2);
    CHECK(std::string(TurbulenceModelName(TurbulenceModel::Laminar)) == "Laminar");
    CHECK(std::string(TurbulenceModelName(TurbulenceModel::KOmegaSST)) == "KOmegaSST");
    CHECK(std::string(TurbulenceVariableName(TurbulenceModel::SpalartAllmaras, 0)) ==
          "TurbulenceNuTilde");
    CHECK(std::string(TurbulenceVariableName(TurbulenceModel::KOmegaSST, 0)) ==
          "TurbulenceK");
    CHECK(std::string(TurbulenceVariableName(TurbulenceModel::KOmegaSST, 1)) ==
          "TurbulenceOmega");
    CHECK(std::string(TurbulenceVariableName(TurbulenceModel::RealizableKEpsilon, 1)) ==
          "TurbulenceEpsilon");

    TurbulenceSettings settings;
    settings.model = TurbulenceModel::KOmegaSST;
    settings.initialValue = {0.2, 3.0};
    settings.Validate();
    const nlohmann::ordered_json serialized = settings;
    const TurbulenceSettings roundTrip = serialized.get<TurbulenceSettings>();
    CHECK(roundTrip.model == TurbulenceModel::KOmegaSST);
    CHECK(roundTrip.InitialState()(0) == doctest::Approx(0.2));
    CHECK(roundTrip.InitialState()(1) == doctest::Approx(3.0));
}

/// @test Verify SST viscosity, production, destruction, and near-wall diffusion
/// against their homogeneous-shear formulas rather than only finite-value checks.
TEST_CASE("ACM SST near-wall homogeneous shear follows the model equations")
{
    TurbulenceSettings settings;
    settings.model = TurbulenceModel::KOmegaSST;
    TurbulenceState state;
    state << 0.2, 4.0;
    VelocityGradient velocityGradient = VelocityGradient::Zero();
    velocityGradient(0, 1) = 2.0;
    const TurbulenceGradient zeroGradient = TurbulenceGradient::Zero();
    constexpr real wallDistance = 0.01;
    constexpr real rho0 = 1.0;
    constexpr real molecularViscosity = 0.001;

    // |Omega|=2, S^2=4, and F1=F2=1 at this distance.
    const real turbulentViscosity = TurbulentDynamicViscosity(
        state, velocityGradient, zeroGradient, wallDistance, rho0,
        molecularViscosity, settings);
    CHECK(turbulentViscosity == doctest::Approx(0.31 * 0.2 / 2.0));

    const TurbulenceState source = TurbulenceSource(
        state, velocityGradient, zeroGradient, wallDistance, rho0,
        molecularViscosity, settings);
    constexpr real betaStar = 0.09;
    constexpr real beta1 = 0.075;
    constexpr real sigmaOmega1 = 0.5;
    constexpr real kappa = 0.41;
    const real gamma1 = beta1 / betaStar -
                        sigmaOmega1 * kappa * kappa / std::sqrt(betaStar);
    CHECK(source(0) == doctest::Approx(4.0 * turbulentViscosity -
                                      betaStar * state(0) * state(1)));
    CHECK(source(1) == doctest::Approx(4.0 * gamma1 -
                                      beta1 * state(1) * state(1)));

    TurbulenceGradient turbulenceGradient = TurbulenceGradient::Zero();
    turbulenceGradient(0, 0) = 1.0;
    turbulenceGradient(0, 1) = 2.0;
    const TurbulenceState flux = TurbulenceDiffusiveFlux(
        state, turbulenceGradient, Vector3::UnitX(), wallDistance,
        rho0, molecularViscosity, turbulentViscosity, settings);
    CHECK(flux(0) == doctest::Approx(molecularViscosity + 0.85 * turbulentViscosity));
    CHECK(flux(1) == doctest::Approx(2.0 * (molecularViscosity +
                                           0.5 * turbulentViscosity)));
}

/// @test Verify the realizable k-epsilon E correction responds to turbulent-time gradients.
TEST_CASE("ACM realizable k-epsilon includes the gradient source correction")
{
    TurbulenceSettings settings;
    settings.model = TurbulenceModel::RealizableKEpsilon;
    TurbulenceState state;
    state << 0.4, 0.08;
    VelocityGradient velocityGradient = VelocityGradient::Zero();
    velocityGradient(0, 1) = 2.0;

    const TurbulenceState sourceWithoutGradient = TurbulenceSource(
        state,
        velocityGradient,
        TurbulenceGradient::Zero(),
        0.1,
        1.2,
        1.8e-5,
        settings);
    TurbulenceGradient turbulenceGradient = TurbulenceGradient::Zero();
    turbulenceGradient(0, 0) = 0.1;
    const TurbulenceState sourceWithGradient = TurbulenceSource(
        state,
        velocityGradient,
        turbulenceGradient,
        0.1,
        1.2,
        1.8e-5,
        settings);

    CHECK(sourceWithoutGradient.allFinite());
    CHECK(sourceWithGradient.allFinite());
    CHECK(sourceWithGradient(0) == doctest::Approx(sourceWithoutGradient(0)));
    CHECK(sourceWithGradient(1) > sourceWithoutGradient(1));
}

/// @test Check positivity floors and clearing of model-inactive state entries.
TEST_CASE("ACM turbulence state clamp is model aware")
{
    TurbulenceSettings settings;
    settings.model = TurbulenceModel::SpalartAllmaras;
    settings.minimumValue = {1e-8, 1e-7};
    settings.maximumValue = {2.0, 3.0};
    TurbulenceState candidate;
    candidate << -1.0, 9.0;
    const TurbulenceState clamped = ClampTurbulenceState(candidate, settings);
    CHECK(clamped(0) == doctest::Approx(1e-8));
    CHECK(clamped(1) == doctest::Approx(0.0));

    candidate << std::numeric_limits<real>::infinity(), 9.0;
    const TurbulenceState upperClamped =
        ClampTurbulenceState(candidate, settings);
    CHECK(upperClamped(0) == doctest::Approx(2.0));
    CHECK(upperClamped(1) == doctest::Approx(0.0));
}

/// @test Verify the SA damping function remains finite in its large-chi asymptote.
TEST_CASE("ACM SA closure saturates safely for an extreme viscosity ratio")
{
    TurbulenceSettings settings;
    settings.model = TurbulenceModel::SpalartAllmaras;
    settings.maximumValue = {1e100, 1e100};
    settings.maximumEddyViscosityRatio = 1e5;
    TurbulenceState state;
    state << 1e100, 0.0;
    const real eddyViscosity = TurbulentDynamicViscosity(
        state,
        VelocityGradient::Zero(),
        TurbulenceGradient::Zero(),
        0.1,
        1.0,
        1e-300,
        settings);
    CHECK(std::isfinite(eddyViscosity));
    CHECK(eddyViscosity == doctest::Approx(1e-295));
}

/// @test Exercise every closure's viscosity, diffusion, and source kernel on finite data.
TEST_CASE("ACM turbulence closure kernels remain finite and non-negative")
{
    VelocityGradient velocityGradient = VelocityGradient::Zero();
    velocityGradient << 0.2, 1.1, -0.1,
        -0.4, -0.1, 0.3,
        0.2, -0.5, -0.1;
    TurbulenceGradient turbulenceGradient = TurbulenceGradient::Zero();
    turbulenceGradient.col(0) << 0.01, -0.02, 0.03;
    turbulenceGradient.col(1) << -0.04, 0.02, 0.01;
    const Vector3 normal = Vector3(1.0, 2.0, -1.0).normalized();
    constexpr real wallDistance = 0.2;
    constexpr real rho0 = 1.2;
    constexpr real molecularViscosity = 1.8e-5;

    for (const TurbulenceModel model : {
             TurbulenceModel::Laminar,
             TurbulenceModel::SpalartAllmaras,
             TurbulenceModel::KOmegaWilcox,
             TurbulenceModel::KOmegaSST,
             TurbulenceModel::RealizableKEpsilon})
    {
        CAPTURE(static_cast<int>(model));
        TurbulenceSettings settings;
        settings.model = model;
        settings.maximumEddyViscosityRatio = 1e5;
        TurbulenceState state;
        state << (model == TurbulenceModel::SpalartAllmaras ? 2e-5 : 0.4),
            (model == TurbulenceModel::RealizableKEpsilon ? 0.08 : 5.0);
        state = ClampTurbulenceState(state, settings);
        const real eddyViscosity = TurbulentDynamicViscosity(
            state,
            velocityGradient,
            turbulenceGradient,
            wallDistance,
            rho0,
            molecularViscosity,
            settings);
        CHECK(std::isfinite(eddyViscosity));
        CHECK(eddyViscosity >= 0.0);
        CHECK(eddyViscosity <=
              settings.maximumEddyViscosityRatio * molecularViscosity);
        if (model == TurbulenceModel::Laminar)
            CHECK(eddyViscosity == doctest::Approx(0.0));
        else
            CHECK(eddyViscosity > 0.0);

        const TurbulenceState diffusion = TurbulenceDiffusiveFlux(
            state,
            turbulenceGradient,
            normal,
            wallDistance,
            rho0,
            molecularViscosity,
            eddyViscosity,
            settings);
        const TurbulenceState source = TurbulenceSource(
            state,
            velocityGradient,
            turbulenceGradient,
            wallDistance,
            rho0,
            molecularViscosity,
            settings);
        CHECK(diffusion.allFinite());
        CHECK(source.allFinite());
    }
}

/// @test Verify wall and far-field ghost states impose face values by reflection.
TEST_CASE("ACM turbulence boundary states follow selected closure")
{
    constexpr real rho0 = 1.0;
    constexpr real molecularViscosity = 1e-5;
    constexpr real wallDistance = 0.01;

    TurbulenceSettings sa;
    sa.model = TurbulenceModel::SpalartAllmaras;
    TurbulenceState saInterior;
    saInterior << 3e-5, 0.0;
    const TurbulenceState saWall = GenerateTurbulenceBoundaryState(
        BoundaryType::BCWall,
        saInterior,
        wallDistance,
        rho0,
        molecularViscosity,
        sa);
    CHECK(0.5 * (saInterior(0) + saWall(0)) == doctest::Approx(0.0));

    TurbulenceSettings sst;
    sst.model = TurbulenceModel::KOmegaSST;
    sst.farFieldValue = {0.2, 4.0};
    TurbulenceState interior;
    interior << 0.1, 2.0;
    const TurbulenceState wall = GenerateTurbulenceBoundaryState(
        BoundaryType::BCWall,
        interior,
        wallDistance,
        rho0,
        molecularViscosity,
        sst);
    CHECK(0.5 * (interior(0) + wall(0)) == doctest::Approx(0.0));
    CHECK(0.5 * (interior(1) + wall(1)) ==
          doctest::Approx(
              sst.wallOmegaCoefficient * molecularViscosity /
              rho0 / (wallDistance * wallDistance)));

    TurbulenceSettings realizable;
    realizable.model = TurbulenceModel::RealizableKEpsilon;
    TurbulenceState realizableInterior;
    realizableInterior << 0.2, 0.05;
    const TurbulenceState realizableWall = GenerateTurbulenceBoundaryState(
        BoundaryType::BCWall,
        realizableInterior,
        wallDistance,
        rho0,
        molecularViscosity,
        realizable);
    CHECK(0.5 * (realizableInterior(0) + realizableWall(0)) ==
          doctest::Approx(0.0));
    CHECK(0.5 * (realizableInterior(1) + realizableWall(1)) ==
          doctest::Approx(
              2.0 * molecularViscosity / rho0 * realizableInterior(0) /
              (wallDistance * wallDistance)));

    const TurbulenceState far = GenerateTurbulenceBoundaryState(
        BoundaryType::BCFar,
        interior,
        wallDistance,
        rho0,
        molecularViscosity,
        sst);
    CHECK(0.5 * (interior(0) + far(0)) == doctest::Approx(0.2));
    CHECK(0.5 * (interior(1) + far(1)) == doctest::Approx(4.0));
}
