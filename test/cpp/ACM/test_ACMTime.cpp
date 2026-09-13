/**
 * @file test_ACMTime.cpp
 * @brief Unit tests for ACM pseudo-time and BDF2 dual-time stepping kernels.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "ACM/ACMBDF2.hpp"
#include "ACM/ACMConfig.hpp"
#include "ACM/ACMTime.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

using namespace DNDS;
using namespace DNDS::ACM;

namespace
{
    /**
     * @brief Construct the raw residual `R=-rate*U` used by analytic time-integration tests.
     * @param rate Positive linear decay rate.
     * @return Residual callback operating independently on every rank-local state.
     */
    ResidualEvaluator LinearDecayResidual(real rate)
    {
        return [rate](const StateField &states, StateField &residual)
        {
            residual.resize(states.size());
            for (std::size_t i = 0; i < states.size(); i++)
                residual[i] = -rate * states[i];
        };
    }

    /**
     * @brief Construct exact diagonal blocks `dR/dU=-rate*I` for the linear decay residual.
     * @param rate Positive linear decay rate.
     * @return Block-diagonal Jacobian callback.
     */
    DiagonalJacobianEvaluator LinearDecayJacobian(real rate)
    {
        return [rate](const StateField &states, MatrixField &jacobian)
        {
            jacobian.assign(states.size(), -rate * Matrix4::Identity());
        };
    }
}

/// @test Inexact steady steps obey their target and tighten near the steady solution.
TEST_CASE("ACM steady inner accuracy follows the current outer residual")
{
    Settings settings;
    settings.beta2 = 1;
    TimeMarchSettings strict;
    strict.integrator = TimeIntegratorType::ImplicitEulerBlockJacobi;
    strict.implicitRelaxation = 0.7;
    strict.maxImplicitIterations = 40;
    strict.implicitTolerance = 1e-10;
    TimeMarchSettings inexact = strict;
    inexact.steadyRelativeTolerance = 0.01;
    const ScalarField dt(1, 0.1);
    StateField exactState(1, State::Zero());
    exactState[0](3) = 1;
    StateField inexactState = exactState;
    const auto exact = AdvanceImplicitEulerBlockJacobi(
        exactState, dt, settings, strict, LinearDecayResidual(1), LinearDecayJacobian(1));
    const auto loose = AdvanceImplicitEulerBlockJacobi(
        inexactState, dt, settings, inexact, LinearDecayResidual(1), LinearDecayJacobian(1));
    CHECK(exact.converged);
    CHECK(loose.converged);
    CHECK(loose.iterations < exact.iterations);
    CHECK(loose.finalDefectNorm <= loose.defectTolerance);
    CHECK(std::abs(inexactState[0](3) - 1.0 / 1.1) <= 2 * loose.defectTolerance / 11);
    CHECK(loose.defectTolerance == doctest::Approx(0.005));

    inexactState[0].setZero();
    inexactState[0](3) = 1e-9;
    const auto nearSteady = AdvanceImplicitEulerBlockJacobi(
        inexactState, dt, settings, inexact, LinearDecayResidual(1), LinearDecayJacobian(1));
    CHECK(nearSteady.converged);
    CHECK(nearSteady.defectTolerance == doctest::Approx(strict.implicitTolerance));

    inexact.integrator = TimeIntegratorType::BDF2DualTimeGMRES;
    CHECK_THROWS_AS(inexact.Validate(), std::runtime_error);
}

/// @test Adaptive CFL reacts to nonlinear progress and respects safety bounds.
TEST_CASE("ACM adaptive steady CFL grows only on successful residual reduction")
{
    TimeMarchSettings settings;
    settings.integrator = TimeIntegratorType::ImplicitEulerGMRES;
    settings.useCFLTimeStep = true;
    settings.steadyAdaptiveCFL = true;
    settings.Validate();
    CHECK(settings.NextSteadyCFL(1, 100, 25, true) == doctest::Approx(1.5));
    CHECK(settings.NextSteadyCFL(1, 100, 100, true) == doctest::Approx(1));
    CHECK(settings.NextSteadyCFL(1, 100, 200, true) == doctest::Approx(0.5));
    CHECK(settings.NextSteadyCFL(1, 100, 25, false) == doctest::Approx(0.5));
    CHECK(settings.NextSteadyCFL(0.1, 100, 200, false) == doctest::Approx(0.1));
    CHECK(settings.NextSteadyCFL(20, 100, 0, true) == doctest::Approx(20));
    CHECK(settings.NextSteadyCFL(1, 0, 0, true) == doctest::Approx(1));
    settings.useCFLTimeStep = false;
    CHECK_THROWS_AS(settings.Validate(), std::runtime_error);
}

/// @test Verify inverse-Gamma application independently of a time-integration scheme.
TEST_CASE("ACM inverse Gamma converts raw residual to pseudo-time derivative")
{
    Settings settings;
    settings.beta2 = 2.5;
    StateField states(1);
    states[0] << 0.7, -0.2, 0.4, 1.1;
    StateField residual(1);
    residual[0] << 0.3, -0.8, 0.1, 0.6;
    StateField derivative;

    ApplyGammaInverseToResidual(states, residual, derivative, settings);

    REQUIRE(derivative.size() == 1);
    const State reconstructed = GammaLocal(states[0], settings.beta2, settings.alpha) * derivative[0];
    CHECK((reconstructed - residual[0]).norm() < 1e-13);
}

/// @test Check the exact SSPRK3 amplification polynomial for a linear decay equation.
TEST_CASE("ACM explicit SSPRK3 has the expected linear amplification")
{
    Settings settings;
    settings.beta2 = 1.0;
    StateField states(1, State::Zero());
    states[0](3) = 2.0;
    constexpr real rate = 1.7;
    constexpr real dt = 0.08;
    const ScalarField pseudoTimeStep(1, dt);

    const TimeStepReport report = AdvanceExplicitSSPRK3(
        states,
        pseudoTimeStep,
        settings,
        LinearDecayResidual(rate));

    const real z = rate * dt;
    const real amplification = 1 - z + 0.5 * z * z - z * z * z / 6;
    CHECK(states[0](3) == doctest::Approx(2.0 * amplification).epsilon(1e-13));
    CHECK(states[0].head<3>().norm() == doctest::Approx(0.0));
    CHECK(report.iterations == 3);
    CHECK(report.converged);
}

/// @test Check backward Euler against its exact result for a block-diagonal linear residual.
TEST_CASE("ACM implicit backward Euler block Jacobi matches linear reference")
{
    Settings settings;
    settings.beta2 = 1.0;
    TimeMarchSettings timeSettings;
    timeSettings.integrator = TimeIntegratorType::ImplicitEulerBlockJacobi;
    timeSettings.maxImplicitIterations = 4;
    timeSettings.implicitTolerance = 1e-13;
    StateField states(1, State::Zero());
    states[0](3) = 2.0;
    constexpr real rate = 1.7;
    constexpr real dt = 0.08;
    const ScalarField pseudoTimeStep(1, dt);

    const TimeStepReport report = AdvanceImplicitEulerBlockJacobi(
        states,
        pseudoTimeStep,
        settings,
        timeSettings,
        LinearDecayResidual(rate),
        LinearDecayJacobian(rate));

    CHECK(states[0](3) == doctest::Approx(2.0 / (1 + rate * dt)).epsilon(1e-13));
    CHECK(states[0].head<3>().norm() == doctest::Approx(0.0));
    CHECK(report.converged);
    CHECK(report.iterations <= 2);
    CHECK(report.finalDefectNorm <= timeSettings.implicitTolerance);
}

/// @test Verify the nonlinear Gamma mass matrix is present on both sides of backward Euler.
TEST_CASE("ACM implicit backward Euler converges with a nonlinear Gamma mass matrix")
{
    Settings settings;
    settings.beta2 = 2.5;
    TimeMarchSettings timeSettings;
    timeSettings.integrator = TimeIntegratorType::ImplicitEulerBlockJacobi;
    timeSettings.maxImplicitIterations = 20;
    timeSettings.implicitTolerance = 1e-12;
    StateField states(1);
    states[0] << 0.8, -0.4, 0.5, 1.2;
    const State initial = states[0];
    State target;
    target << 0.2, -0.1, 0.3, 0.4;
    constexpr real rate = 1.3;
    constexpr real dt = 0.06;
    const ScalarField pseudoTimeStep(1, dt);
    const ResidualEvaluator residualEvaluator = [&](const StateField &input, StateField &residual)
    {
        residual.resize(input.size());
        for (std::size_t i = 0; i < input.size(); i++)
            residual[i] = GammaLocal(input[i], settings.beta2, settings.alpha) *
                          (-rate * (input[i] - target));
    };
    const DiagonalJacobianEvaluator jacobianEvaluator = [&](const StateField &input, MatrixField &jacobian)
    {
        jacobian.resize(input.size());
        constexpr real epsilon = 1e-7;
        for (std::size_t i = 0; i < input.size(); i++)
        {
            for (int iVariable = 0; iVariable < 4; iVariable++)
            {
                State plus = input[i];
                State minus = input[i];
                plus(iVariable) += epsilon;
                minus(iVariable) -= epsilon;
                const State residualPlus = GammaLocal(plus, settings.beta2, settings.alpha) *
                                           (-rate * (plus - target));
                const State residualMinus = GammaLocal(minus, settings.beta2, settings.alpha) *
                                            (-rate * (minus - target));
                jacobian[i].col(iVariable) = (residualPlus - residualMinus) / (2 * epsilon);
            }
        }
    };

    const TimeStepReport report = AdvanceImplicitEulerBlockJacobi(
        states,
        pseudoTimeStep,
        settings,
        timeSettings,
        residualEvaluator,
        jacobianEvaluator);

    const State expected = (initial + rate * dt * target) / (1 + rate * dt);
    CHECK((states[0] - expected).norm() < 1e-11);
    CHECK(report.converged);
    CHECK(report.finalDefectNorm <= timeSettings.implicitTolerance);
}

/// @test Compare the exact pseudo-time product Jacobian with a centered finite difference.
TEST_CASE("ACM pseudo-time product Jacobian includes the nonlinear Gamma derivative")
{
    Settings settings;
    settings.alpha = 0.37;
    settings.beta2 = 2.4;
    State previous;
    previous << 0.6, -0.2, 0.4, 0.3;
    State state;
    state << 0.9, 0.1, -0.5, 1.1;
    constexpr real pseudoTimeStep = 0.17;

    const auto pseudoTimeProduct = [&](const State &value) -> State
    {
        return GammaLocal(value, settings.beta2, settings.alpha) *
               (value - previous) / pseudoTimeStep;
    };
    Matrix4 finiteDifference;
    for (int variable = 0; variable < 4; variable++)
    {
        const real epsilon = 1e-7 * std::max(real(1), std::abs(state(variable)));
        State plus = state;
        State minus = state;
        plus(variable) += epsilon;
        minus(variable) -= epsilon;
        finiteDifference.col(variable) =
            (pseudoTimeProduct(plus) - pseudoTimeProduct(minus)) / (2 * epsilon);
    }

    const Matrix4 analytic =
        PseudoTimeProductJacobian(state, previous, pseudoTimeStep, settings);
    CHECK((analytic - finiteDifference).norm() <
          2e-8 * std::max(real(1), analytic.norm()));

    Matrix4 expectedExtra = Matrix4::Zero();
    const real extraDiagonal =
        (settings.alpha + 1) * (state(3) - previous(3)) /
        (settings.beta2 * pseudoTimeStep);
    expectedExtra.diagonal().head<3>().setConstant(extraDiagonal);
    const Matrix4 extra = analytic -
                          GammaLocal(state, settings.beta2, settings.alpha) /
                              pseudoTimeStep;
    CHECK((extra - expectedExtra).norm() < 1e-13);

    State equalPressure = state;
    equalPressure(3) = previous(3);
    CHECK((PseudoTimeProductJacobian(
               equalPressure, previous, pseudoTimeStep, settings) -
           GammaLocal(equalPressure, settings.beta2, settings.alpha) /
               pseudoTimeStep)
              .norm() < 1e-13);
    CHECK_THROWS(PseudoTimeProductJacobian(state, previous, 0, settings));
}

/// @test Ensure every nonlinear backward-Euler iteration uses the product-rule Jacobian.
TEST_CASE("ACM implicit backward Euler wires the exact Gamma product Jacobian")
{
    Settings settings;
    settings.alpha = 0.37;
    settings.beta2 = 2.4;
    TimeMarchSettings timeSettings;
    timeSettings.integrator = TimeIntegratorType::ImplicitEulerBlockJacobi;
    timeSettings.maxImplicitIterations = 2;
    timeSettings.implicitTolerance = 0;
    timeSettings.implicitRelaxation = 1;
    constexpr real pseudoTimeStep = 0.15;
    const ScalarField pseudoTimeSteps(1, pseudoTimeStep);

    StateField states(1);
    states[0] << 0.6, -0.2, 0.4, 0.3;
    const State initial = states[0];
    State constantResidual;
    constantResidual << 0.25, -0.15, 0.1, 0.8;
    const ResidualEvaluator residualEvaluator =
        [constantResidual](const StateField &input, StateField &residual)
    {
        residual.assign(input.size(), constantResidual);
    };
    const DiagonalJacobianEvaluator zeroJacobian =
        [](const StateField &input, MatrixField &jacobian)
    {
        jacobian.assign(input.size(), Matrix4::Zero());
    };

    State reference = initial;
    for (int iteration = 0; iteration < 2; iteration++)
    {
        const State defect = constantResidual -
                             GammaLocal(reference, settings.beta2, settings.alpha) *
                                 (reference - initial) / pseudoTimeStep;
        Matrix4 exactJacobian =
            GammaLocal(reference, settings.beta2, settings.alpha) /
            pseudoTimeStep;
        const real extraDiagonal =
            (settings.alpha + 1) * (reference(3) - initial(3)) /
            (settings.beta2 * pseudoTimeStep);
        exactJacobian.diagonal().head<3>().array() += extraDiagonal;
        reference += exactJacobian.partialPivLu().solve(defect);
    }

    const TimeStepReport report = AdvanceImplicitEulerBlockJacobi(
        states,
        pseudoTimeSteps,
        settings,
        timeSettings,
        residualEvaluator,
        zeroJacobian);

    CHECK(report.iterations == 2);
    CHECK((states[0] - reference).norm() < 1e-12);
}

/// @test Ensure invalid local time steps are rejected before a state can be modified.
TEST_CASE("ACM time integrators reject non-positive pseudo-time steps")
{
    Settings settings;
    StateField states(1, State::Zero());
    const ScalarField pseudoTimeStep(1, 0.0);
    CHECK_THROWS(AdvanceExplicitSSPRK3(
        states,
        pseudoTimeStep,
        settings,
        LinearDecayResidual(1.0)));
}

/// @test Verify backward-Euler startup and constant-step BDF2 coefficients.
TEST_CASE("ACM BDF2 history starts with backward Euler and then becomes second order")
{
    StateField initial(1, State::Zero());
    initial[0] << 1.0, 2.0, 3.0, 4.0;
    BDF2History history;
    history.Initialize(initial);

    const BDF2Coefficients startup = history.Coefficients();
    CHECK(startup.current == doctest::Approx(1.0));
    CHECK(startup.previous == doctest::Approx(-1.0));
    CHECK(startup.previousPrevious == doctest::Approx(0.0));
    CHECK(startup.order == 1);

    StateField completed = initial;
    completed[0](0) = 1.5;
    history.Commit(completed);
    const BDF2Coefficients secondOrder = history.Coefficients();
    CHECK(secondOrder.current == doctest::Approx(1.5));
    CHECK(secondOrder.previous == doctest::Approx(-2.0));
    CHECK(secondOrder.previousPrevious == doctest::Approx(0.5));
    CHECK(secondOrder.order == 2);
    CHECK(history.Previous()[0](0) == doctest::Approx(1.5));
    CHECK(history.PreviousPrevious()[0](0) == doctest::Approx(1.0));
}

/// @test Check the singular physical-time mass matrix and conventional BDF2 derivative.
TEST_CASE("ACM BDF2 differentiates velocity but not artificial-compressibility pressure")
{
    State current;
    State previous;
    State previousPrevious;
    current << 4.0, 8.0, -2.0, 100.0;
    previous << 3.0, 4.0, -1.0, -50.0;
    previousPrevious << 1.0, 2.0, 1.0, 25.0;
    constexpr real dt = 0.25;

    const State derivative = EvaluateBDF2PhysicalDerivative(
        current,
        previous,
        previousPrevious,
        GetBDF2Coefficients(1),
        dt);
    const State expected =
        PhysicalTimeMassMatrix() * (1.5 * current - 2.0 * previous + 0.5 * previousPrevious) / dt;
    CHECK((derivative - expected).norm() < 1e-14);
    CHECK(derivative(3) == doctest::Approx(0.0));
    CHECK(PhysicalTimeMassMatrix()(3, 3) == doctest::Approx(0.0));
}

/// @test Verify the BDF physical term is added consistently to defect and Jacobian diagonal.
TEST_CASE("ACM BDF2 physical defect and implicit diagonal use matching coefficients")
{
    StateField current(1, State::Zero());
    StateField previous(1, State::Zero());
    StateField previousPrevious(1, State::Zero());
    StateField spatialResidual(1, State::Zero());
    current[0] << 2.0, -1.0, 0.5, 7.0;
    previous[0] << 1.5, -0.7, 0.4, 3.0;
    previousPrevious[0] << 1.0, -0.2, 0.1, -2.0;
    spatialResidual[0] << 0.4, -0.3, 0.2, 0.1;
    constexpr real dt = 0.2;
    const BDF2Coefficients coefficients = GetBDF2Coefficients(1);

    StateField defect;
    FormBDF2PhysicalDefect(
        current,
        previous,
        previousPrevious,
        spatialResidual,
        coefficients,
        dt,
        defect);
    const State expectedDefect = spatialResidual[0] - EvaluateBDF2PhysicalDerivative(
                                                          current[0],
                                                          previous[0],
                                                          previousPrevious[0],
                                                          coefficients,
                                                          dt);
    CHECK((defect[0] - expectedDefect).norm() < 1e-14);

    MatrixField diagonal(1, 2.0 * Matrix4::Identity());
    AddBDF2PhysicalDiagonal(diagonal, coefficients, dt);
    const Matrix4 expectedDiagonal =
        2.0 * Matrix4::Identity() + coefficients.current * PhysicalTimeMassMatrix() / dt;
    CHECK((diagonal[0] - expectedDiagonal).norm() < 1e-14);
    CHECK(diagonal[0](3, 3) == doctest::Approx(2.0));
}

/// @test Ensure both BDF2 linear solvers are selectable through JSON enum conversion.
TEST_CASE("ACM BDF2 integrators round trip through JSON names")
{
    for (const TimeIntegratorType integrator : {
             TimeIntegratorType::BDF2DualTimeLUSGS,
             TimeIntegratorType::BDF2DualTimeGMRES})
    {
        const nlohmann::ordered_json encoded = integrator;
        const TimeIntegratorType decoded = encoded.get<TimeIntegratorType>();
        CHECK(decoded == integrator);
        CHECK(IsBDF2DualTimeIntegrator(decoded));
    }
    CHECK(BDF2UsesLUSGS(TimeIntegratorType::BDF2DualTimeLUSGS));
    CHECK_FALSE(BDF2UsesLUSGS(TimeIntegratorType::BDF2DualTimeGMRES));
    CHECK_THROWS(
        nlohmann::ordered_json("BDF2DualTimeTypo").get<TimeIntegratorType>());
}

/// @test Reject invalid physical steps and incompatible completed-state histories.
TEST_CASE("ACM BDF2 validates its physical step and history dimensions")
{
    TimeMarchSettings settings;
    settings.physicalTimeStep = 0;
    CHECK_THROWS(settings.Validate());

    State state = State::Zero();
    CHECK_THROWS(EvaluateBDF2PhysicalDerivative(
        state,
        state,
        state,
        GetBDF2Coefficients(0),
        0));

    BDF2History history;
    CHECK_THROWS(history.Coefficients());
    CHECK_THROWS(history.Initialize({}));

    StateField initial(2, State::Zero());
    history.Initialize(initial);
    CHECK_THROWS(history.Commit(StateField(1, State::Zero())));
    CHECK(history.CompletedPhysicalSteps() == 0);
}

/// @test Preserve legacy JSON loading without writing a new field back to a case file.
TEST_CASE("ACM legacy configuration receives an in-memory BDF2 physical-step default")
{
    const std::filesystem::path temporaryConfiguration =
        std::filesystem::temp_directory_path() / "dndsr_acm_legacy_bdf2_config.json";
    nlohmann::ordered_json legacyConfiguration = KernelConfiguration{};
    legacyConfiguration.at("timeMarchSettings").erase("physicalTimeStep");
    for (const char *key : {"steadyRelativeTolerance", "steadyAdaptiveCFL", "steadyCFLMin", "steadyCFLMax",
                            "steadyCFLGrowth", "steadyCFLReduction"})
        legacyConfiguration.at("timeMarchSettings").erase(key);
    for (const char *key : {"variationalTolerance", "variationalMaxIterations", "variationalCheckInterval", "variationalRelaxation"})
        legacyConfiguration.at("reconstructionSettings").erase(key);
    {
        std::ofstream output(temporaryConfiguration);
        REQUIRE(output.good());
        output << legacyConfiguration;
    }

    const LoadedConfiguration loaded = LoadConfiguration(
        temporaryConfiguration.string(),
        {},
        {});
    std::filesystem::remove(temporaryConfiguration);

    CHECK(loaded.configuration.timeMarchSettings.physicalTimeStep == doctest::Approx(0.01));
    CHECK(loaded.configuration.timeMarchSettings.steadyRelativeTolerance == 0);
    CHECK_FALSE(loaded.configuration.timeMarchSettings.steadyAdaptiveCFL);
    CHECK(loaded.configuration.reconstructionSettings.variationalTolerance == 0);
    REQUIRE(loaded.resolvedJson.contains("timeMarchSettings"));
    CHECK(loaded.resolvedJson.at("timeMarchSettings").contains("physicalTimeStep"));
}

/// @test Prevent BDF2 from silently treating segregated turbulence as physically second order.
TEST_CASE("ACM BDF2 configuration rejects turbulence without physical-time history")
{
    KernelConfiguration configuration;
    configuration.timeMarchSettings.integrator = TimeIntegratorType::BDF2DualTimeLUSGS;
    configuration.turbulenceSettings.model = TurbulenceModel::KOmegaSST;
    CHECK_THROWS(configuration.Validate());
}
