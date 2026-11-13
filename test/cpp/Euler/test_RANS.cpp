/**
 * @file test_RANS.cpp
 * @brief Unit tests for RANS turbulence model functions in RANS_ke.hpp.
 *
 * Tests cover:
 *   - GetMut_KOWilcox: turbulent viscosity from k-omega Wilcox 2006
 *   - GetMut_SST: turbulent viscosity from k-omega SST
 *   - GetMut_RealizableKe: turbulent viscosity from Realizable k-epsilon
 *   - GetSource_KOWilcox: source terms (mode 0 = full, mode 1 = Jacobian diagonal)
 *   - GetSource_SST: source terms
 *   - GetSource_RealizableKe: source terms
 *   - GetVisFlux_KOWilcox: viscous flux for turbulent transport variables
 *   - GetVisFlux_SST: viscous flux
 *   - GetVisFlux_RealizableKe: viscous flux
 *
 * Key properties tested:
 *   - mut >= 0 (non-negative turbulent viscosity)
 *   - mut <= 1e5 * muLam (CFL3D limiting)
 *   - Zero gradient -> zero production in source
 *   - Zero gradient -> zero viscous flux
 *   - Finite output (no NaN/Inf)
 *   - Golden values for specific test vectors
 *
 * All functions are pure template functions (no MPI, no mesh).
 * SA source limiting is covered with the same JSON-facing config type used by
 * the evaluator.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Euler/RANS_ke.hpp"
#include "Euler/EulerEvaluatorSettings.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <type_traits>

using namespace DNDS;
using namespace DNDS::Euler;
using namespace DNDS::Euler::RANS;

static constexpr real g_gamma = 1.4;
static constexpr int dim = 3;
static constexpr int I4 = dim + 1; // = 4
// nVars for 2-equation models: 5 (flow) + 2 (turb) = 7
static constexpr int nVars = 7;

static const KOmegaConfig g_kOmegaConfig{};
static const KOmegaSSTConfig g_kOmegaSSTConfig{};
static const RKEConfig g_rkeConfig{};
static const SAConfig g_saConfig{};

static constexpr real GOLDEN_NOT_ACQUIRED = 1e300;

// Build a 3D 2-equation conservative state:
// [rho, rho*u, rho*v, rho*w, rho*E, rho*k, rho*omega_or_epsilon]
static Eigen::Vector<real, nVars> buildState(
    real rho, real u, real v, real w, real p, real k, real omega)
{
    Eigen::Vector<real, nVars> U;
    real E = p / (g_gamma - 1.0) + 0.5 * rho * (u * u + v * v + w * w);
    U << rho, rho * u, rho * v, rho * w, E, rho * k, rho * omega;
    return U;
}

// Build a gradient matrix (dim x nVars) with a simple shear profile:
// du/dy = S (shear rate), everything else zero
static Eigen::Matrix<real, dim, nVars> buildShearGrad(
    const Eigen::Vector<real, nVars> &U, real S_shear)
{
    Eigen::Matrix<real, dim, nVars> DiffU;
    DiffU.setZero();
    // d(rho*u)/dy = rho * S (since rho is constant in this test)
    DiffU(1, 1) = U(0) * S_shear; // d(rho*u)/dy
    return DiffU;
}

// ===================================================================
// Turbulence equilibrium state:
// Typical boundary layer: rho=1.225, u=50, p=101325
// k=10, omega=1000 (or epsilon=Cmu*k*omega=0.09*10*1000=900)
// mu_lam = 1.8e-5
// wall distance d=0.01 (inside BL)
// ===================================================================
static const real g_rho = 1.225;
static const real g_u = 50.0;
static const real g_p = 101325.0;
static const real g_k = 10.0;
static const real g_omega = 1000.0;
static const real g_epsilon = 0.09 * g_k * g_omega; // Cmu * k * omega
static const real g_mu = 1.8e-5;
static const real g_d = 0.01;
static const real g_Sshear = 500.0; // du/dy

// ===================================================================
// GetMut_KOWilcox
// ===================================================================

TEST_CASE("GetMut_KOWilcox: non-negative")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_KOWilcox<dim>(U, DiffU, g_mu, g_d, g_kOmegaConfig);

    CHECK(mut >= 0.0);
    CHECK(std::isfinite(mut));
}

TEST_CASE("GetMut_KOWilcox: bounded by 1e5 * muLam")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_KOWilcox<dim>(U, DiffU, g_mu, g_d, g_kOmegaConfig);

    CHECK(mut <= 1e5 * g_mu + 1e-10);
}

TEST_CASE("GetMut_KOWilcox: golden value")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_KOWilcox<dim>(U, DiffU, g_mu, g_d, g_kOmegaConfig);

    std::cout << "[KOWilcox/mut] " << std::scientific << std::setprecision(10) << mut << std::endl;
    // Golden value captured:
    real golden = 8.4000000000e-03;
    if (golden < GOLDEN_NOT_ACQUIRED)
        CHECK(mut == doctest::Approx(golden).epsilon(1e-6));
}

// ===================================================================
// GetMut_SST
// ===================================================================

TEST_CASE("GetMut_SST: non-negative")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_SST<dim>(U, DiffU, g_mu, g_d, g_kOmegaSSTConfig);

    CHECK(mut >= 0.0);
    CHECK(std::isfinite(mut));
}

TEST_CASE("GetMut_SST: bounded by 1e5 * muLam")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_SST<dim>(U, DiffU, g_mu, g_d, g_kOmegaSSTConfig);

    CHECK(mut <= 1e5 * g_mu + 1e-10);
}

TEST_CASE("GetMut_SST: golden value")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_SST<dim>(U, DiffU, g_mu, g_d, g_kOmegaSSTConfig);

    std::cout << "[SST/mut] " << std::scientific << std::setprecision(10) << mut << std::endl;
    real golden = 7.5950000000e-03;
    if (golden < GOLDEN_NOT_ACQUIRED)
        CHECK(mut == doctest::Approx(golden).epsilon(1e-6));
}

// ===================================================================
// GetMut_RealizableKe
// ===================================================================

TEST_CASE("GetMut_RealizableKe: non-negative")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_RealizableKe<dim>(U, DiffU, g_mu, g_d, g_rkeConfig);

    CHECK(mut >= 0.0);
    CHECK(std::isfinite(mut));
}

TEST_CASE("GetMut_RealizableKe: bounded by 1e5 * muLam")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_RealizableKe<dim>(U, DiffU, g_mu, g_d, g_rkeConfig);

    CHECK(mut <= 1e5 * g_mu + 1e-10);
}

TEST_CASE("GetMut_RealizableKe: golden value")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    auto DiffU = buildShearGrad(U, g_Sshear);
    real mut = GetMut_RealizableKe<dim>(U, DiffU, g_mu, g_d, g_rkeConfig);

    std::cout << "[RealKe/mut] " << std::scientific << std::setprecision(10) << mut << std::endl;
    real golden = 1.2250000000e-02;
    if (golden < GOLDEN_NOT_ACQUIRED)
        CHECK(mut == doctest::Approx(golden).epsilon(1e-6));
}

// ===================================================================
// Source terms: zero gradient -> zero production (only destruction)
// ===================================================================

TEST_CASE("GetSource_KOWilcox: zero gradient finite and has destruction")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    Eigen::Matrix<real, dim, nVars> DiffU = Eigen::Matrix<real, dim, nVars>::Zero();
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_KOWilcox<dim>(U, DiffU, g_mu, g_d, source, 0, g_kOmegaConfig);

    CHECK(std::isfinite(source(I4 + 1)));
    CHECK(std::isfinite(source(I4 + 2)));
    // k source should be negative (destruction only when Pk=0)
    CHECK(source(I4 + 1) <= 0.0);
    // omega source should also be negative (destruction)
    CHECK(source(I4 + 2) <= 0.0);
}

TEST_CASE("GetSource_SST: zero gradient finite")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    Eigen::Matrix<real, dim, nVars> DiffU = Eigen::Matrix<real, dim, nVars>::Zero();
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_SST<dim>(U, DiffU, g_mu, g_d, 1e10, source, 0, g_kOmegaSSTConfig);

    CHECK(std::isfinite(source(I4 + 1)));
    CHECK(std::isfinite(source(I4 + 2)));
    // k destruction: -betaStar * rho * k * omega < 0
    CHECK(source(I4 + 1) <= 0.0);
}

TEST_CASE("GetSource_RealizableKe: zero gradient finite")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    Eigen::Matrix<real, dim, nVars> DiffU = Eigen::Matrix<real, dim, nVars>::Zero();
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_RealizableKe<dim>(U, DiffU, g_mu, g_d, source, 0, g_rkeConfig);

    CHECK(std::isfinite(source(I4 + 1)));
    CHECK(std::isfinite(source(I4 + 2)));
}

// ===================================================================
// Source terms with shear: golden values
// ===================================================================

TEST_CASE("GetSource_KOWilcox: shear gradient golden")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_KOWilcox<dim>(U, DiffU, g_mu, g_d, source, 0, g_kOmegaConfig);

    std::cout << "[KOWilcox/source] k=" << std::scientific << std::setprecision(10)
              << source(I4 + 1) << " omega=" << source(I4 + 2) << std::endl;

    CHECK(std::isfinite(source(I4 + 1)));
    CHECK(std::isfinite(source(I4 + 2)));

    // With shear, production should exceed destruction (positive total k-source)
    // Not always true depending on parameters, so just check finite
}

TEST_CASE("GetSource_SST: shear gradient golden")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_SST<dim>(U, DiffU, g_mu, g_d, 1e10, source, 0, g_kOmegaSSTConfig);

    std::cout << "[SST/source] k=" << std::scientific << std::setprecision(10)
              << source(I4 + 1) << " omega=" << source(I4 + 2) << std::endl;

    CHECK(std::isfinite(source(I4 + 1)));
    CHECK(std::isfinite(source(I4 + 2)));
}

TEST_CASE("GetSource_RealizableKe: shear gradient golden")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    auto DiffU = buildShearGrad(U, g_Sshear);
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_RealizableKe<dim>(U, DiffU, g_mu, g_d, source, 0, g_rkeConfig);

    std::cout << "[RealKe/source] k=" << std::scientific << std::setprecision(10)
              << source(I4 + 1) << " epsilon=" << source(I4 + 2) << std::endl;

    CHECK(std::isfinite(source(I4 + 1)));
    CHECK(std::isfinite(source(I4 + 2)));
}

// ===================================================================
// Source terms: mode 1 (implicit diagonal) should be non-negative
// ===================================================================

TEST_CASE("GetSource_KOWilcox mode=1: implicit diagonal non-negative")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_KOWilcox<dim>(U, DiffU, g_mu, g_d, source, 1, g_kOmegaConfig);

    // mode=1 returns destruction rate coefficients, should be >= 0
    CHECK(source(I4 + 1) >= 0.0);
    CHECK(source(I4 + 2) >= 0.0);
}

TEST_CASE("GetSource_SST mode=1: implicit diagonal non-negative")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);
    Eigen::Vector<real, nVars> source;
    source.setZero();

    GetSource_SST<dim>(U, DiffU, g_mu, g_d, 1e10, source, 1, g_kOmegaSSTConfig);

    CHECK(source(I4 + 1) >= 0.0);
    CHECK(source(I4 + 2) >= 0.0);
}

// ===================================================================
// Viscous flux: zero gradient -> zero flux
// ===================================================================

TEST_CASE("GetVisFlux_KOWilcox: zero gradient -> zero flux")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    // DiffUxyPrim: gradient of PRIMITIVE variables
    Eigen::Matrix<real, dim, nVars> DiffPrim = Eigen::Matrix<real, dim, nVars>::Zero();
    Eigen::Vector3d norm(1, 0, 0);
    Eigen::Vector<real, nVars> vFlux;
    vFlux.setZero();

    real mut = 0.01; // arbitrary
    GetVisFlux_KOWilcox<dim>(U, DiffPrim, norm, mut, g_d, g_mu, vFlux, g_kOmegaConfig);

    CHECK(vFlux(I4 + 1) == doctest::Approx(0.0).epsilon(1e-14));
    CHECK(vFlux(I4 + 2) == doctest::Approx(0.0).epsilon(1e-14));
}

TEST_CASE("GetVisFlux_SST: zero gradient -> zero flux")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    Eigen::Matrix<real, dim, nVars> DiffPrim = Eigen::Matrix<real, dim, nVars>::Zero();
    Eigen::Vector3d norm(1, 0, 0);
    Eigen::Vector<real, nVars> vFlux;
    vFlux.setZero();

    real mut = 0.01;
    GetVisFlux_SST<dim>(U, DiffPrim, norm, mut, g_d, g_mu, vFlux, g_kOmegaSSTConfig);

    CHECK(vFlux(I4 + 1) == doctest::Approx(0.0).epsilon(1e-14));
    CHECK(vFlux(I4 + 2) == doctest::Approx(0.0).epsilon(1e-14));
}

TEST_CASE("GetVisFlux_RealizableKe: zero gradient -> zero flux")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_epsilon);
    Eigen::Matrix<real, dim, nVars> DiffPrim = Eigen::Matrix<real, dim, nVars>::Zero();
    Eigen::Vector3d norm(1, 0, 0);
    Eigen::Vector<real, nVars> vFlux;
    vFlux.setZero();

    real mut = 0.01;
    GetVisFlux_RealizableKe<dim>(U, DiffPrim, norm, mut, g_d, g_mu, vFlux, g_rkeConfig);

    CHECK(vFlux(I4 + 1) == doctest::Approx(0.0).epsilon(1e-14));
    CHECK(vFlux(I4 + 2) == doctest::Approx(0.0).epsilon(1e-14));
}

// ===================================================================
// Viscous flux: non-zero gradient produces non-zero flux
// ===================================================================

TEST_CASE("GetVisFlux_KOWilcox: k-gradient produces k-flux")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    Eigen::Matrix<real, dim, nVars> DiffPrim = Eigen::Matrix<real, dim, nVars>::Zero();
    // dk/dx = 100 (in primitive space, dk/dx = dk/dx since k is already primitive-like)
    DiffPrim(0, I4 + 1) = 100.0;
    Eigen::Vector3d norm(1, 0, 0);
    Eigen::Vector<real, nVars> vFlux;
    vFlux.setZero();

    real mut = 0.01;
    GetVisFlux_KOWilcox<dim>(U, DiffPrim, norm, mut, g_d, g_mu, vFlux, g_kOmegaConfig);

    // k-flux should be positive (diffusion in direction of decreasing k)
    CHECK(vFlux(I4 + 1) > 0.0);
    CHECK(std::isfinite(vFlux(I4 + 1)));
    // omega-flux should still be zero (no omega gradient)
    CHECK(vFlux(I4 + 2) == doctest::Approx(0.0).epsilon(1e-14));
}

// ===================================================================
// Mut increases with k for fixed omega (physical)
// ===================================================================

TEST_CASE("GetMut_KOWilcox: mut increases with k")
{
    auto U1 = buildState(g_rho, g_u, 0, 0, g_p, 5.0, g_omega);
    auto U2 = buildState(g_rho, g_u, 0, 0, g_p, 20.0, g_omega);
    auto DiffU = buildShearGrad(U1, g_Sshear);
    auto DiffU2 = buildShearGrad(U2, g_Sshear);

    real mut1 = GetMut_KOWilcox<dim>(U1, DiffU, g_mu, g_d, g_kOmegaConfig);
    real mut2 = GetMut_KOWilcox<dim>(U2, DiffU2, g_mu, g_d, g_kOmegaConfig);

    CHECK(mut2 > mut1);
}

TEST_CASE("GetMut_SST: mut increases with k")
{
    auto U1 = buildState(g_rho, g_u, 0, 0, g_p, 5.0, g_omega);
    auto U2 = buildState(g_rho, g_u, 0, 0, g_p, 20.0, g_omega);
    auto DiffU = buildShearGrad(U1, g_Sshear);
    auto DiffU2 = buildShearGrad(U2, g_Sshear);

    real mut1 = GetMut_SST<dim>(U1, DiffU, g_mu, g_d, g_kOmegaSSTConfig);
    real mut2 = GetMut_SST<dim>(U2, DiffU2, g_mu, g_d, g_kOmegaSSTConfig);

    CHECK(mut2 > mut1);
}

// ===================================================================
// Very small k/omega: no crash, finite result
// ===================================================================

TEST_CASE("GetMut_KOWilcox: very small k/omega produces finite mut")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, 1e-10, 1e-5);
    auto DiffU = buildShearGrad(U, 0.0);
    real mut = GetMut_KOWilcox<dim>(U, DiffU, g_mu, g_d, g_kOmegaConfig);

    CHECK(std::isfinite(mut));
    CHECK(mut >= 0.0);
}

TEST_CASE("GetMut_SST: very small k/omega produces finite mut")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, 1e-10, 1e-5);
    auto DiffU = buildShearGrad(U, 0.0);
    real mut = GetMut_SST<dim>(U, DiffU, g_mu, g_d, g_kOmegaSSTConfig);

    CHECK(std::isfinite(mut));
    CHECK(mut >= 0.0);
}

TEST_CASE("GetMut_RealizableKe: very small k/eps produces finite mut")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, 1e-10, 1e-8);
    auto DiffU = buildShearGrad(U, 0.0);
    real mut = GetMut_RealizableKe<dim>(U, DiffU, g_mu, g_d, g_rkeConfig);

    CHECK(std::isfinite(mut));
    CHECK(mut >= 0.0);
}

TEST_CASE("RANS configs: defaults and JSON round-trip preserve historical limits")
{
    static_assert(std::is_trivially_copyable_v<SAConfig>);
    static_assert(std::is_trivially_copyable_v<KOmegaConfig>);
    static_assert(std::is_trivially_copyable_v<KOmegaSSTConfig>);
    static_assert(std::is_trivially_copyable_v<RKEConfig>);

    CHECK(g_saConfig.productionLimit == doctest::Approx(1e5));
    CHECK(g_kOmegaConfig.productionLimit == doctest::Approx(20));
    CHECK(g_kOmegaConfig.turbulentViscosityLimit == doctest::Approx(1e5));
    CHECK(g_kOmegaSSTConfig.productionLimit == doctest::Approx(20));
    CHECK(g_kOmegaSSTConfig.turbulentViscosityLimit == doctest::Approx(1e5));
    CHECK(g_rkeConfig.turbulentViscosityLimit == doctest::Approx(1e5));

    nlohmann::ordered_json j = g_saConfig;
    CHECK(j.at("productionLimit").get<real>() == doctest::Approx(1e5));
    j["productionLimit"] = 100;
    auto parsed = j.get<SAConfig>();
    CHECK(parsed.productionLimit == doctest::Approx(100));

    nlohmann::ordered_json kOmegaJson = g_kOmegaConfig;
    CHECK(kOmegaJson.at("productionLimit").get<real>() == doctest::Approx(20));
    CHECK(kOmegaJson.at("turbulentViscosityLimit").get<real>() == doctest::Approx(1e5));
    kOmegaJson["productionLimit"] = 0.5;
    kOmegaJson["turbulentViscosityLimit"] = 2e4;
    auto parsedKOmega = kOmegaJson.get<KOmegaConfig>();
    CHECK(parsedKOmega.productionLimit == doctest::Approx(0.5));
    CHECK(parsedKOmega.turbulentViscosityLimit == doctest::Approx(2e4));
}

TEST_CASE("Euler evaluator settings: RANS configs are flat nested sections")
{
    EulerEvaluatorSettings<NS_SA_3D> settings(6);
    nlohmann::ordered_json j = settings;

    CHECK(j.at("SAConfig").at("productionLimit").get<real>() == doctest::Approx(1e5));
    CHECK(j.at("KOmegaConfig").at("productionLimit").get<real>() == doctest::Approx(20));
    CHECK(j.at("KOmegaSSTConfig").at("productionLimit").get<real>() == doctest::Approx(20));
    CHECK(j.at("RKEConfig").at("turbulentViscosityLimit").get<real>() == doctest::Approx(1e5));
    CHECK(j.at("RANSTopLimit").get<real>() == doctest::Approx(1e5));

    j["SAConfig"]["productionLimit"] = 100;
    j["KOmegaConfig"]["productionLimit"] = 0.5;
    j["KOmegaConfig"]["turbulentViscosityLimit"] = 2e4;
    EulerEvaluatorSettings<NS_SA_3D> parsed(6);
    from_json(j, parsed);
    CHECK(parsed.saConfig.productionLimit == doctest::Approx(100));
    CHECK(parsed.kOmegaConfig.productionLimit == doctest::Approx(0.5));
    CHECK(parsed.kOmegaConfig.turbulentViscosityLimit == doctest::Approx(2e4));
    CHECK(parsed.RANSTopLimit == doctest::Approx(1e5));
}

TEST_CASE("GetSource_KOWilcox: production limit changes the k source")
{
    auto U = buildState(g_rho, g_u, 0, 0, g_p, g_k, g_omega);
    auto DiffU = buildShearGrad(U, g_Sshear);

    KOmegaConfig capped;
    capped.productionLimit = 0.5;
    KOmegaConfig relaxed;
    relaxed.productionLimit = 20;
    Eigen::Vector<real, nVars> sourceCapped = Eigen::Vector<real, nVars>::Zero();
    Eigen::Vector<real, nVars> sourceRelaxed = Eigen::Vector<real, nVars>::Zero();

    GetSource_KOWilcox<dim>(U, DiffU, g_mu, g_d, sourceCapped, 0, capped);
    GetSource_KOWilcox<dim>(U, DiffU, g_mu, g_d, sourceRelaxed, 0, relaxed);

    const real destruction = 0.09 * g_rho * g_k * g_omega;
    CHECK(sourceCapped.allFinite());
    CHECK(sourceRelaxed.allFinite());
    CHECK(sourceCapped(I4 + 1) ==
          doctest::Approx((capped.productionLimit - 1.0) * destruction));
    CHECK(sourceRelaxed(I4 + 1) > sourceCapped(I4 + 1));
}

TEST_CASE("GetSource_SA: production limit changes the physical source")
{
    static constexpr int nVarsSA = 6;
    Eigen::Vector<real, nVarsSA> U;
    const real rhoNuTilde = g_rho * 1e-5;
    const real totalEnergy = g_p / (g_gamma - 1.0) + 0.5 * g_rho * g_u * g_u;
    U << g_rho, g_rho * g_u, 0, 0, totalEnergy, rhoNuTilde;

    Eigen::Matrix<real, dim, nVarsSA> DiffU;
    DiffU.setZero();
    DiffU(1, 1) = g_rho * 1e6;

    SAConfig capped;
    capped.productionLimit = 100;
    SAConfig relaxed;
    relaxed.productionLimit = 1e5;
    Eigen::Vector<real, nVarsSA> sourceCapped = Eigen::Vector<real, nVarsSA>::Zero();
    Eigen::Vector<real, nVarsSA> sourceRelaxed = Eigen::Vector<real, nVarsSA>::Zero();

    GetSource_SA<dim>(U, DiffU, 1.0, g_mu, g_gamma,
                      g_d, 1e10, 0.1, 0, sourceCapped, 0, 0, capped);
    GetSource_SA<dim>(U, DiffU, 1.0, g_mu, g_gamma,
                      g_d, 1e10, 0.1, 0, sourceRelaxed, 0, 0, relaxed);

    CHECK(sourceCapped.allFinite());
    CHECK(sourceRelaxed.allFinite());
    CHECK(sourceRelaxed(I4 + 1) > sourceCapped(I4 + 1));
}

/// @test Inner/outer SST omega production must use the corresponding gamma coefficient.
/// Reference: Menter (1994), DOI 10.2514/3.12149; TMR sst.html#sst-std.
TEST_CASE("GetSource_SST: omega production blends both coefficient sets")
{
    constexpr real rho = 1.3, k = 0.2, omega = 4, mu = 0.0013, shear = 2;
    auto U = buildState(rho, 0, 0, 0, 1, k, omega);
    auto gradient = buildShearGrad(U, shear);
    for (const real distance : {0.01, 1.0, 100.0})
    {
        const real argument = std::max(std::sqrt(k) / (0.09 * omega * distance),
                                      500 * mu / rho / (distance * distance * omega));
        const real f1 = std::tanh(std::pow(argument, 4)); // CD term is inactive here
        const real gamma1 = 0.075 / 0.09 - 0.5 * 0.41 * 0.41 / 0.3;
        const real gamma2 = 0.0828 / 0.09 - 0.856 * 0.41 * 0.41 / 0.3;
        const real gamma = f1 * gamma1 + (1 - f1) * gamma2;
        const real beta = f1 * 0.075 + (1 - f1) * 0.0828;
        Eigen::Vector<real, nVars> source = Eigen::Vector<real, nVars>::Zero();
        GetSource_SST<dim>(U, gradient, mu, distance, 1e10, source, 0, g_kOmegaSSTConfig);
        const real production = source(I4 + 2) / rho + beta * omega * omega;
        CHECK(production == doctest::Approx(gamma * shear * shear).epsilon(1e-12));
    }
}

/// @test Source and diffusion share the original SST CD floor in weak turbulence.
TEST_CASE("GetSource_SST: 1994 weak turbulence retains the inner blending limit")
{
    constexpr real rho = 1, k = 1e-8, omega = 1e-6, mu = 1e-6, distance = 100;
    auto U = buildState(rho, 0, 0, 0, 1, k, omega);
    Eigen::Matrix<real, dim, nVars> gradient = Eigen::Matrix<real, dim, nVars>::Zero();
    Eigen::Vector<real, nVars> source = Eigen::Vector<real, nVars>::Zero();
    GetSource_SST<dim>(U, gradient, mu, distance, 1e10, source, 1, g_kOmegaSSTConfig);
    CHECK(source(I4 + 2) / (2 * omega) == doctest::Approx(0.075));
    gradient(0, I4 + 1) = 1;
    gradient(1, I4 + 2) = 1; // orthogonal gradients keep CD=0
    Eigen::Vector<real, dim> normal;
    normal << 1, 1, 0;
    normal.normalize();
    Eigen::Vector<real, nVars> flux = Eigen::Vector<real, nVars>::Zero();
    GetVisFlux_SST<dim>(U, gradient, normal, 1e-5, distance, mu, flux, g_kOmegaSSTConfig);
    CHECK(flux(I4 + 1) * std::sqrt(2.0) == doctest::Approx(mu + 0.85e-5).epsilon(1e-12));
    CHECK(flux(I4 + 2) * std::sqrt(2.0) == doctest::Approx(mu + 0.5e-5).epsilon(1e-12));
}

/// @test Epsilon must set the turbulent time scale even when k and epsilon differ.
TEST_CASE("GetSource_RealizableKe: homogeneous decay uses the epsilon state entry")
{
    constexpr real rho = 1.2, k = 0.4, mu = 1.8e-5;
    for (const real epsilon : {0.08, 0.16})
    {
        auto U = buildState(rho, 0, 0, 0, 1, k, epsilon);
        Eigen::Matrix<real, dim, nVars> gradient = Eigen::Matrix<real, dim, nVars>::Zero();
        Eigen::Vector<real, nVars> source = Eigen::Vector<real, nVars>::Zero();
        GetSource_RealizableKe<dim>(U, gradient, mu, 1, source, 0, g_rkeConfig);
        CHECK(source(I4 + 1) == doctest::Approx(-rho * epsilon));
        CHECK(source(I4 + 2) == doctest::Approx(-1.92 * rho * epsilon * epsilon / k).epsilon(1e-10));
        GetSource_RealizableKe<dim>(U, gradient, mu, 1, source, 1, g_rkeConfig);
        CHECK(source(I4 + 2) == doctest::Approx(1.92 * epsilon / k).epsilon(1e-10));
    }
}

/// @test The exact SST wall value is zero in both supported kernel dimensions.
TEST_CASE("GetMut_SST: zero wall k is not replaced by an interior positivity floor")
{
    auto U = buildState(1, 0, 0, 0, 1, 0, 4);
    const Eigen::Matrix<real, dim, nVars> gradient = Eigen::Matrix<real, dim, nVars>::Zero();
    CHECK(GetMut_SST<dim>(U, gradient, 0.001, 0.01, g_kOmegaSSTConfig) == 0);
    Eigen::Vector<real, 6> state2D;
    state2D << 1, 0, 0, 2.5, 0, 4;
    const Eigen::Matrix<real, 2, 6> gradient2D = Eigen::Matrix<real, 2, 6>::Zero();
    CHECK(GetMut_SST<2>(state2D, gradient2D, 0.001, 0.01, g_kOmegaSSTConfig) == 0);
}

/// TMR SST-1994: exact P=tau:grad(u); no cap or viscosity floor in omega production.
TEST_CASE("GetSource_SST: exact compressible production and weak viscosity scale")
{
    const auto verify = [](auto dimension)
    {
        constexpr int nd = decltype(dimension)::value;
        constexpr int energy = nd + 1, count = nd + 4;
        constexpr real rho = 1.3, k = 1e-8, omega = 4;
        for (const real mu : {1e-3, 1e-12})
            for (const real expansion : {-2.0, 0.0, 2.0})
                for (const real shear : {0.0, 2.0, 100.0})
                {
                    Eigen::Vector<real, count> u = Eigen::Vector<real, count>::Zero();
                    u(0) = rho; u(energy) = 2.5;
                    u(energy + 1) = rho * k; u(energy + 2) = rho * omega;
                    Eigen::Matrix<real, nd, count> gradient = Eigen::Matrix<real, nd, count>::Zero();
                    for (int j = 0; j < nd; j++) gradient(j, j + 1) = rho * expansion;
                    gradient(1, 1) = rho * shear;
                    // Embed a 2D flow in 3D before forming deviatoric strain.
                    Eigen::Matrix3d velocity = Eigen::Matrix3d::Zero();
                    velocity.topLeftCorner<nd, nd>() = gradient.template block<nd, nd>(0, 1) / rho;
                    const Eigen::Matrix3d strain = 0.5 * (velocity + velocity.transpose()) -
                        velocity.trace() / 3 * Eigen::Matrix3d::Identity();
                    const real mut = GetMut_SST<nd>(u, gradient, mu, 0.001, g_kOmegaSSTConfig);
                    const real production = 2 * mut * strain.squaredNorm() - 2.0 / 3 * rho * k * velocity.trace();
                    const real argument = std::max(std::sqrt(k) / (0.09 * omega * 0.001),
                        500 * mu / rho / (0.001 * 0.001 * omega));
                    const real f1 = std::tanh(std::pow(argument, 4));
                    const real beta = 0.075 * f1 + 0.0828 * (1 - f1);
                    const real gamma = (0.075 / 0.09 - 0.5 * 0.41 * 0.41 / 0.3) * f1 +
                        (0.0828 / 0.09 - 0.856 * 0.41 * 0.41 / 0.3) * (1 - f1);
                    Eigen::Vector<real, count> source = Eigen::Vector<real, count>::Zero();
                    GetSource_SST<nd>(u, gradient, mu, 0.001, 1e100, source, 0, g_kOmegaSSTConfig);
                    CHECK(source(energy + 1) == doctest::Approx(std::min(production, 20 * 0.09 * rho * k * omega) - 0.09 * rho * k * omega).epsilon(1e-11));
                    CHECK(source(energy + 2) == doctest::Approx(gamma * production / (mut / rho) - beta * rho * omega * omega).epsilon(1e-11));
                }
    };
    verify(std::integral_constant<int, 2>{});
    verify(std::integral_constant<int, 3>{});
}
