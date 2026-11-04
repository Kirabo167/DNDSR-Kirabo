/**
 * @file test_ACMCore.cpp
 * @brief Serial unit tests for constant-density ACM state, flux, boundary, and configuration kernels.
 *
 * @author Runzhi Ma
 * @date 2026-09-01
 * @note Modifier: Runzhi Ma.
 */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "ACM/ACMBC.hpp"
#include "ACM/ACMConfig.hpp"
#include "ACM/ACMFlux.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using namespace DNDS;
using namespace DNDS::ACM;

namespace
{
    /**
     * @brief Compare two four-component ACM vectors with doctest diagnostics.
     * @param actual Vector produced by the kernel under test.
     * @param expected Reference vector.
     * @param tolerance Relative epsilon used by `doctest::Approx` for each component.
     */
    void CheckVectorNear(const State &actual, const State &expected, real tolerance = 1e-11)
    {
        for (int i = 0; i < 4; i++)
        {
            CAPTURE(i);
            CAPTURE(actual(i));
            CAPTURE(expected(i));
            CHECK(actual(i) == doctest::Approx(expected(i)).epsilon(tolerance));
        }
    }
}

/// @test Verify orthonormal local-basis construction and lossless state rotation.
TEST_CASE("ACM local basis and state rotation preserve pressure")
{
    const Vector3 normal = Vector3(1, 2, -3).normalized();
    const Matrix3 basis = BuildLocalBasis(normal);
    CHECK((basis.transpose() * basis - Matrix3::Identity()).norm() < 1e-13);
    CHECK((basis.col(0) - normal).norm() < 1e-13);

    State state;
    state << 2.0, -1.0, 0.5, 7.0;
    const State local = ToLocalState(state, basis);
    const State roundTrip = FromLocalFlux(local, basis);
    CheckVectorNear(roundTrip, state);
    CHECK(local(3) == doctest::Approx(7.0));
}

/// @test Compare the analytic physical-flux Jacobian against centered finite differences.
TEST_CASE("ACM physical flux Jacobian matches finite differences")
{
    State state;
    state << 1.3, -0.4, 0.7, 2.2;
    constexpr real rho0 = 1.7;
    const Matrix4 analytic = PhysicalFluxJacobianLocal(state, rho0);
    Matrix4 finiteDifference;
    const real epsilon = 1e-7;
    for (int column = 0; column < 4; column++)
    {
        State perturbation = State::Zero();
        perturbation(column) = epsilon;
        finiteDifference.col(column) =
            (PhysicalFluxLocal(state + perturbation, rho0) -
             PhysicalFluxLocal(state - perturbation, rho0)) /
            (2 * epsilon);
    }

    CHECK((analytic - finiteDifference).norm() < 1e-8);
    CHECK(analytic(0, 0) == doctest::Approx(2 * state(0)));
    CHECK(analytic(1, 0) == doctest::Approx(state(1)));
    CHECK(analytic(2, 0) == doctest::Approx(state(2)));
    CHECK(analytic(3, 0) == doctest::Approx(1.0));
}

/// @test Check that Gamma inversion and the preconditioned characteristic speeds are consistent.
TEST_CASE("ACM Gamma and preconditioned eigensystem are consistent")
{
    State mean;
    mean << 0.8, -0.3, 0.2, 1.0;
    constexpr real rho0 = 1.25;
    constexpr real beta2 = 3.4;
    for (const real alpha : {-1.2, -0.4, 0.0, 0.5, 1.3})
    {
        CAPTURE(alpha);
        const Matrix4 gamma = GammaLocal(mean, beta2, alpha);
        const Matrix4 gammaInv = GammaInvLocal(mean, beta2, alpha);
        CHECK((gammaInv * gamma - Matrix4::Identity()).norm() < 1e-13);

        const Matrix4 preconditioned =
            PreconditionedJacobianLocal(mean, rho0, beta2, alpha);
        CHECK((preconditioned - gammaInv * PhysicalFluxJacobianLocal(mean, rho0)).norm() < 1e-13);
        CHECK(preconditioned(0, 0) == doctest::Approx((1 - alpha) * mean(0)));
        CHECK(preconditioned(1, 0) == doctest::Approx(-alpha * mean(1)));
        CHECK(preconditioned(2, 0) == doctest::Approx(-alpha * mean(2)));
        CHECK(preconditioned(3, 0) == doctest::Approx(beta2));

        Eigen::EigenSolver<Matrix4> solver(preconditioned, false);
        std::vector<real> numeric;
        for (int i = 0; i < 4; i++)
        {
            CHECK(std::abs(solver.eigenvalues()(i).imag()) < 1e-12);
            numeric.push_back(solver.eigenvalues()(i).real());
        }
        std::sort(numeric.begin(), numeric.end());
        const auto expectedValues = ComputeEigenvalues(mean(0), rho0, beta2, alpha);
        std::array<real, 4> expected{
            expectedValues.lambdaMinus,
            expectedValues.lambdaTangential,
            expectedValues.lambdaTangential,
            expectedValues.lambdaPlus};
        std::sort(expected.begin(), expected.end());
        for (int i = 0; i < 4; i++)
            CHECK(numeric[static_cast<std::size_t>(i)] ==
                  doctest::Approx(expected[static_cast<std::size_t>(i)]).epsilon(1e-11));
    }
}

/// @test Ensure the algebraically correct characteristic root is evaluated without squaring overflow.
TEST_CASE("ACM characteristic speeds remain finite for a representable large velocity")
{
    constexpr real normalVelocity = 1e200;
    constexpr real rho0 = 1.0;
    constexpr real beta2 = 4.0;
    constexpr real alpha = 0.5;
    const Eigenvalues eigenvalues =
        ComputeEigenvalues(normalVelocity, rho0, beta2, alpha);
    CHECK(std::isfinite(eigenvalues.lambdaMinus));
    CHECK(std::isfinite(eigenvalues.lambdaTangential));
    CHECK(std::isfinite(eigenvalues.lambdaPlus));
    CHECK(eigenvalues.lambdaTangential == doctest::Approx(normalVelocity));
    CHECK(eigenvalues.lambdaPlus == doctest::Approx(0.5 * normalVelocity));
}

/// @test Verify general-alpha ACM characteristic transforms used by WBAP/CWBAP.
TEST_CASE("ACM general-alpha characteristic transforms diagonalize the normal operator")
{
    Settings settings;
    settings.rho0 = 1.7;
    settings.beta2 = 3.2;
    State mean;
    mean << 0.8, -0.3, 0.5, 1.1;
    const Vector3 normal = Vector3(1.0, -2.0, 0.7).normalized();
    const Matrix3 basis = BuildLocalBasis(normal);
    const State meanLocal = ToLocalState(mean, basis);
    Matrix4 globalToLocal = Matrix4::Identity();
    globalToLocal.block<3, 3>(0, 0) = basis.transpose();

    for (const real alpha : {-1.0, -0.4, 0.0, 0.5, 1.0})
    {
        CAPTURE(alpha);
        settings.alpha = alpha;
        const Matrix4 right = RightEigenvectorsGlobal(mean, normal, settings);
        const Matrix4 left = LeftEigenvectorsGlobal(mean, normal, settings);
        CHECK((left * right - Matrix4::Identity()).norm() < 1e-10);

        const Matrix4 operatorGlobal =
            globalToLocal.inverse() *
            PreconditionedJacobianLocal(meanLocal, settings.rho0, settings.beta2, alpha) *
            globalToLocal;
        const Matrix4 diagonalized = left * operatorGlobal * right;
        Matrix4 offDiagonal = diagonalized;
        offDiagonal.diagonal().setZero();
        CHECK(offDiagonal.norm() < 1e-9);
        const Eigenvalues eigenvalues = ComputeEigenvalues(
            meanLocal(0), settings.rho0, settings.beta2, alpha);
        CHECK(diagonalized(0, 0) == doctest::Approx(eigenvalues.lambdaMinus).epsilon(1e-10));
        CHECK(diagonalized(1, 1) == doctest::Approx(eigenvalues.lambdaTangential).epsilon(1e-10));
        CHECK(diagonalized(2, 2) == doctest::Approx(eigenvalues.lambdaTangential).epsilon(1e-10));
        CHECK(diagonalized(3, 3) == doctest::Approx(eigenvalues.lambdaPlus).epsilon(1e-10));
    }
}

/// @test Check equal-state consistency and normal-reversal symmetry for both Riemann solvers.
TEST_CASE("ACM Riemann solvers are consistent and normal-symmetric")
{
    Settings settings;
    settings.rho0 = 1.3;
    settings.beta2 = 2.1;
    settings.entropyFixRatio = 0.03;

    State left;
    left << 0.8, -0.2, 0.4, 1.5;
    State right;
    right << -0.1, 0.5, -0.3, 0.7;
    const Vector3 normal = Vector3(0.3, -0.4, 0.5).normalized();
    const Matrix3 basis = BuildLocalBasis(normal);
    const State exact = FromLocalFlux(PhysicalFluxLocal(ToLocalState(left, basis), settings.rho0), basis);

    for (const auto type : {RiemannSolverType::Rusanov, RiemannSolverType::Roe})
    {
        const auto consistent = InviscidFlux(type, left, left, normal, settings);
        CheckVectorNear(consistent.flux, exact, 1e-10);

        const auto forward = InviscidFlux(type, left, right, normal, settings);
        const auto reverse = InviscidFlux(type, right, left, -normal, settings);
        CheckVectorNear(forward.flux, -reverse.flux, 1e-10);
        CHECK(forward.flux.allFinite());
        CHECK(forward.eigenvalues.SpectralRadius() > 0);
    }
}

/// @test Compare general-alpha Roe dissipation against an explicit characteristic-matrix reference.
TEST_CASE("ACM general-alpha Roe dissipation matches matrix reference")
{
    Settings settings;
    settings.rho0 = 1.1;
    settings.beta2 = 2.7;
    settings.entropyFixRatio = 0.0;
    State left;
    left << 0.9, -0.4, 0.3, 1.2;
    State right;
    right << -0.2, 0.5, -0.1, 0.6;

    for (const real alpha : {-1.0, -0.3, 0.0, 0.6, 1.0})
    {
        CAPTURE(alpha);
        settings.alpha = alpha;
        Eigenvalues eigenvalues;
        const State actual = RoeDissipationLocal(left, right, settings, eigenvalues);
        const State mean = 0.5 * (left + right);
        const Matrix4 rightEigenvectors = RightEigenvectorsGlobal(
            mean, Vector3::UnitX(), settings);
        const Matrix4 leftEigenvectors = LeftEigenvectorsGlobal(
            mean, Vector3::UnitX(), settings);
        Matrix4 lambdaAbs = Matrix4::Zero();
        lambdaAbs.diagonal() << std::abs(eigenvalues.lambdaMinus),
            std::abs(eigenvalues.lambdaTangential),
            std::abs(eigenvalues.lambdaTangential),
            std::abs(eigenvalues.lambdaPlus);
        const State expected = GammaLocal(mean, settings.beta2, alpha) *
                               rightEigenvectors * lambdaAbs * leftEigenvectors *
                               (right - left);
        CheckVectorNear(actual, expected, 1e-9);
    }
}

TEST_CASE("ACM fast Roe projection matches characteristic reference with entropy smoothing")
{
    for (const DNDS::real alpha : {-1.0, 0.0, 0.5, 1.0})
        for (int sample = 0; sample < 80; sample++)
        {
            Settings settings;
            settings.alpha = alpha;
            settings.rho0 = 1.1;
            settings.beta2 = 2.7;
            settings.entropyFixRatio = 0.05;
            const DNDS::real x = sample * 0.31;
            State mean, jump;
            mean << 2.0 * std::sin(x), 0.4 * std::cos(x), 0.3 * std::sin(2 * x), 0.2;
            jump << 0.13 * std::cos(x), 0.2, -0.1, 0.09;
            Matrix4 left, right;
            if (!TryCharacteristicMatricesGlobal(mean, Vector3::UnitX(), settings, left, right))
                continue;
            Eigenvalues eigenvalues;
            const State actual = RoeDissipationLocal(mean - 0.5 * jump, mean + 0.5 * jump, settings, eigenvalues);
            const DNDS::real delta = settings.entropyFixRatio *
                std::max(eigenvalues.SpectralRadius(), std::sqrt(settings.beta2 / settings.rho0));
            Matrix4 absolute = Matrix4::Zero();
            absolute.diagonal() << EntropyFixedAbs(eigenvalues.lambdaMinus, delta),
                EntropyFixedAbs(eigenvalues.lambdaTangential, delta),
                EntropyFixedAbs(eigenvalues.lambdaTangential, delta),
                EntropyFixedAbs(eigenvalues.lambdaPlus, delta);
            const State expected = GammaLocal(mean, settings.beta2, alpha) * right * absolute * left * jump;
            CheckVectorNear(actual, expected, 1e-9);
        }
}

/// @test Verify the exact matrix-absolute/Jordan limit at a defective positive collision.
TEST_CASE("ACM Roe collision fallback matches the confluent Jordan limit")
{
    Settings settings;
    settings.rho0 = 1.0;
    settings.beta2 = 1.0;
    settings.alpha = 1.0;
    settings.entropyFixRatio = 0.0;
    State left;
    left << 1.2, 0.4, -0.3, 0.8;
    State right;
    right << 0.8, -0.2, 0.5, 1.0;

    Eigenvalues eigenvalues;
    const State dissipation = RoeDissipationLocal(left, right, settings, eigenvalues);
    CHECK(dissipation.allFinite());
    CHECK(eigenvalues.lambdaPlus == doctest::Approx(1.0));
    CHECK(eigenvalues.lambdaTangential == doctest::Approx(1.0));

    Matrix4 leftEigenvectors;
    Matrix4 rightEigenvectors;
    CHECK_FALSE(TryCharacteristicMatricesGlobal(
        0.5 * (left + right),
        Vector3::UnitX(),
        settings,
        leftEigenvectors,
        rightEigenvectors));

    const State mean = 0.5 * (left + right);
    const State increment = right - left;
    const Matrix4 preconditioned = PreconditionedJacobianLocal(
        mean, settings.rho0, settings.beta2, settings.alpha);
    const Matrix4 shifted = preconditioned - Matrix4::Identity();
    // For eigenvalues {-1,1,1,1} with a size-two Jordan block at +1,
    // |B| = I + (B-I) + 1/2 (B-I)^2. Modifier: Runzhi Ma.
    const Matrix4 exactAbsolute =
        Matrix4::Identity() + shifted + 0.5 * shifted * shifted;
    const State expected = GammaLocal(mean, settings.beta2, settings.alpha) *
                           exactAbsolute * increment;
    CHECK((dissipation - expected).norm() < 1e-12);

    // The former equal-speed cluster fallback reduced this case to Gamma*deltaU and omitted
    // the derivative action on the generalized eigenvector.
    const State formerCluster =
        GammaLocal(mean, settings.beta2, settings.alpha) * increment;
    CHECK((dissipation - formerCluster).norm() > 1e-3);

    BoundaryCondition condition;
    condition.type = BoundaryType::BCFar;
    condition.value = {0.1, -0.1, 0.2, 0.3};
    const State ghost = GenerateBoundaryState(
        condition, 0.5 * (left + right), Vector3::UnitX(), settings);
    CHECK(ghost.allFinite());
}

/// @test Check that non-colliding Roe decompositions converge to the Jordan-limit value.
TEST_CASE("ACM Roe dissipation is continuous through a characteristic collision")
{
    Settings settings;
    settings.rho0 = 1.0;
    settings.beta2 = 1.0;
    settings.alpha = 1.0;
    settings.entropyFixRatio = 0.0;

    State mean;
    mean << 1.0, 0.1, 0.1, 0.9;
    State increment;
    increment << -0.4, -0.6, 0.8, 0.2;
    Eigenvalues collisionEigenvalues;
    const State collisionValue = RoeDissipationLocal(
        mean - 0.5 * increment,
        mean + 0.5 * increment,
        settings,
        collisionEigenvalues);

    for (const real offset : {-1e-5, 1e-5})
    {
        CAPTURE(offset);
        State nearbyMean = mean;
        nearbyMean(0) += offset;
        Eigenvalues nearbyEigenvalues;
        const State nearbyValue = RoeDissipationLocal(
            nearbyMean - 0.5 * increment,
            nearbyMean + 0.5 * increment,
            settings,
            nearbyEigenvalues);
        CHECK(nearbyValue.allFinite());
        CHECK((nearbyValue - collisionValue).norm() < 1e-4);
    }
}

/// @test Verify linearly exact viscous face gradients on orthogonal and skew center connections.
TEST_CASE("ACM corrected face gradient exactly reproduces linear fields")
{
    Eigen::Matrix<real, 3, 4> exactGradient;
    exactGradient << 0.3, -0.2, 0.4, 0.1,
        -0.5, 0.7, -0.1, 0.2,
        0.6, 0.8, -0.9, -0.3;
    State left;
    left << 0.4, -0.7, 0.2, 1.3;
    const Vector3 normal = Vector3(0.8, 0.5, -0.3).normalized();

    const std::array<Vector3, 2> displacements{
        Vector3(2.3 * normal),
        Vector3(1.2, -0.4, 0.7)};
    for (const Vector3 &displacement : displacements)
    {
        CAPTURE(displacement.transpose());
        const State right = left + exactGradient.transpose() * displacement;
        const Eigen::Matrix<real, 3, 4> corrected = CorrectedFaceGradient(
            exactGradient,
            exactGradient,
            left,
            right,
            displacement,
            normal);
        CHECK((corrected - exactGradient).norm() < 1e-12);
    }

    const Vector3 skewDisplacement(1.2, -0.4, 0.7);
    const State right = left + exactGradient.transpose() * skewDisplacement;
    const Eigen::Matrix<real, 3, 4> correctedFromJump = CorrectedFaceGradient(
        Eigen::Matrix<real, 3, 4>::Zero(),
        Eigen::Matrix<real, 3, 4>::Zero(),
        left,
        right,
        skewDisplacement,
        normal);
    CHECK((skewDisplacement.transpose() * correctedFromJump -
           (right - left).transpose())
              .norm() < 1e-12);
}

/// @test Verify the far-field boundary imports exactly the incoming general-alpha modes.
TEST_CASE("ACM general-alpha far field uses the common characteristic basis")
{
    Settings settings;
    settings.rho0 = 1.3;
    settings.beta2 = 2.4;
    settings.alpha = 0.7;
    State interior;
    interior << 0.6, -0.3, 0.4, 0.8;
    State farField;
    farField << -0.2, 0.5, -0.1, 1.2;
    const Vector3 normal = Vector3(0.4, -0.7, 0.2).normalized();

    BoundaryCondition condition;
    condition.type = BoundaryType::BCFar;
    Eigen::Map<State>(condition.value.data()) = farField;
    const State actual = GenerateBoundaryState(condition, interior, normal, settings);

    const Matrix4 right = RightEigenvectorsGlobal(interior, normal, settings);
    const Matrix4 left = LeftEigenvectorsGlobal(interior, normal, settings);
    const State amplitudes = left * (farField - interior);
    const real qn = interior.head<3>().dot(normal);
    const Eigenvalues eigenvalues = ComputeEigenvalues(
        qn, settings.rho0, settings.beta2, settings.alpha);
    const std::array<real, 4> waveSpeeds{
        eigenvalues.lambdaMinus,
        eigenvalues.lambdaTangential,
        eigenvalues.lambdaTangential,
        eigenvalues.lambdaPlus};
    State expected = interior;
    for (int wave = 0; wave < 4; wave++)
        if (waveSpeeds[static_cast<std::size_t>(wave)] < 0)
            expected += amplitudes(wave) * right.col(wave);
    CheckVectorNear(actual, expected, 1e-9);
}

/// @test Check that ghost states impose no-slip, slip, and pressure-outlet face values.
TEST_CASE("ACM boundary ghost states impose face values")
{
    State interior;
    interior << 2.0, -1.0, 0.5, 3.0;
    State prescribed;
    prescribed << 0.2, 0.3, -0.4, 1.1;
    const Vector3 normal = Vector3::UnitX();

    const State noSlip = GenerateBoundaryState(BoundaryType::NoSlipWall, interior, prescribed, normal);
    CheckVectorNear(0.5 * (interior + noSlip).head<4>(),
                    (State() << prescribed(0), prescribed(1), prescribed(2), interior(3)).finished());

    const State slip = GenerateBoundaryState(BoundaryType::SlipWall, interior, State::Zero(), normal);
    CHECK((0.5 * (interior.head<3>() + slip.head<3>())).dot(normal) == doctest::Approx(0.0));
    CHECK(slip(1) == doctest::Approx(interior(1)));
    CHECK(slip(2) == doctest::Approx(interior(2)));

    const State outlet = GenerateBoundaryState(BoundaryType::PressureOutlet, interior, prescribed, normal);
    CHECK(0.5 * (interior(3) + outlet(3)) == doctest::Approx(prescribed(3)));
}

/// @test Exercise every Euler-compatible ACM boundary family with four-variable semantics.
TEST_CASE("ACM implements every Euler boundary family without Euler state assumptions")
{
    Settings settings;
    settings.rho0 = 1.0;
    settings.beta2 = 1.0;
    settings.alpha = 0.0;
    State interior;
    interior << 0.25, -0.5, 0.75, 2.0;
    State prescribed;
    prescribed << -0.2, 0.1, -0.3, 1.25;
    const Vector3 normal = Vector3::UnitX();

    const auto evaluate = [&](BoundaryType type, int specialOption = 0)
    {
        BoundaryCondition condition;
        condition.type = type;
        Eigen::Map<State>(condition.value.data()) = prescribed;
        condition.specialOption = specialOption;
        return GenerateBoundaryState(condition, interior, normal, settings);
    };

    CHECK(evaluate(BoundaryType::BCFar).allFinite());
    CheckVectorNear(
        0.5 * (interior + evaluate(BoundaryType::BCWall)),
        (State() << prescribed(0), prescribed(1), prescribed(2), interior(3)).finished());
    CheckVectorNear(evaluate(BoundaryType::BCWallIsothermal), evaluate(BoundaryType::BCWall));
    CHECK((0.5 * (interior.head<3>() + evaluate(BoundaryType::BCWallInvis).head<3>()) -
           prescribed.head<3>())
              .dot(normal) == doctest::Approx(0.0));
    CheckVectorNear(evaluate(BoundaryType::BCOut), interior);
    CHECK(0.5 * (interior(3) + evaluate(BoundaryType::BCOutP)(3)) ==
          doctest::Approx(prescribed(3)));
    CheckVectorNear(evaluate(BoundaryType::BCIn), prescribed);
    CHECK((0.5 * (interior + evaluate(BoundaryType::BCInPsTs))).head<3>().isApprox(
        prescribed.head<3>()));
    CHECK(evaluate(BoundaryType::BCInPsTs)(3) == doctest::Approx(interior(3)));
    CHECK((0.5 * (interior.head<3>() + evaluate(BoundaryType::BCSym).head<3>())).dot(normal) ==
          doctest::Approx(0.0));
    CheckVectorNear(evaluate(BoundaryType::BCSpecial), prescribed);
}

/// @test Verify reserved Euler zone IDs and custom CGNS names select independent ACM conditions.
TEST_CASE("ACM boundary handler maps reserved and custom CGNS zones")
{
    State defaultValue;
    defaultValue << 1.0, 0.0, 0.0, 0.5;
    BoundaryCondition outlet;
    outlet.type = BoundaryType::BCOutP;
    outlet.name = "OUTLET";
    outlet.value = {0.0, 0.0, 0.0, 1.2};

    BoundaryHandler handler(BoundaryType::BCFar, defaultValue, {outlet});
    CHECK(handler.GetTypeFromID(Geom::BC_ID_DEFAULT_WALL) == BoundaryType::BCWall);
    CHECK(handler.GetTypeFromID(Geom::BC_ID_DEFAULT_WALL_INVIS) == BoundaryType::BCWallInvis);
    CHECK(handler.GetTypeFromID(Geom::BC_ID_DEFAULT_FAR) == BoundaryType::BCFar);
    const Geom::t_index outletID = handler.GetIDFromName("OUTLET");
    CHECK(handler.GetTypeFromID(outletID) == BoundaryType::BCOutP);
    CHECK(handler.GetValueFromID(outletID)(3) == doctest::Approx(1.2));
    const Geom::t_index appendedID = handler.GetIDFromName("UNMAPPED_ZONE");
    CHECK(handler.GetTypeFromID(appendedID) == BoundaryType::BCFar);
}

/// @test Verify round-trip serialization through the existing DNDS JSON configuration registry.
TEST_CASE("ACM settings use DNDS JSON registration")
{
    Settings settings;
    settings.rho0 = 2.0;
    settings.beta2 = 4.0;
    settings.alpha = 0.6;
    settings.riemannSolverType = RiemannSolverType::Rusanov;
    nlohmann::ordered_json json = settings;
    const Settings restored = json.get<Settings>();
    CHECK(restored.rho0 == doctest::Approx(2.0));
    CHECK(restored.beta2 == doctest::Approx(4.0));
    CHECK(restored.alpha == doctest::Approx(0.6));
    CHECK(restored.riemannSolverType == RiemannSolverType::Rusanov);
    CHECK(json["pressureStorage"] == "PhysicalP");

    nlohmann::ordered_json boundaryJson = {
        {"type", "BCOutP"},
        {"name", "OUTLET"},
        {"value", {0.0, 0.0, 0.0, 1.25}},
        {"frameOption", 0},
        {"anchorOption", 0},
        {"integrationOption", 0},
        {"specialOption", 0},
        {"rectifyOption", 0},
        {"valueExtra", nlohmann::ordered_json::array()},
    };
    const BoundaryCondition boundary = boundaryJson.get<BoundaryCondition>();
    CHECK(boundary.type == BoundaryType::BCOutP);
    CHECK(boundary.name == "OUTLET");
    CHECK(boundary.ValueState()(3) == doctest::Approx(1.25));
}
