/**
 * @file test_core.cpp
 * @brief Independent algebra, flux, boundary, viscosity and time regression tests.
 * @author Runzhi Ma
 * @date 2026-09-03
 */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "ACMVariable/ACMConfig.hpp"
#include "ACMVariable/ACMBDF2.hpp"
#include <random>

using namespace DNDS::ACMVariable;
using DNDS::real;

TEST_CASE("Variable ACM nonlinear and linear tolerances accept either threshold")
{
    TimeMarchSettings settings;
    settings.implicitTolerance = 1e-8;
    settings.implicitRelativeTolerance = 1e-4;
    CHECK(settings.ImplicitTarget(0) == doctest::Approx(1e-8));
    CHECK(settings.ImplicitTarget(100) == doctest::Approx(1e-2));
    CHECK_NOTHROW(settings.Validate());
    const nlohmann::ordered_json encoded = settings;
    const auto decoded = encoded.get<TimeMarchSettings>();
    CHECK(decoded.implicitRelativeTolerance == doctest::Approx(1e-4));
    CHECK(decoded.linearAbsoluteTolerance == doctest::Approx(1e-10));
    CHECK_THROWS(settings.ImplicitTarget(-1));
    settings.implicitRelativeTolerance = 1;
    CHECK_THROWS(settings.Validate());
    settings.implicitRelativeTolerance = 1e-4;
    settings.linearAbsoluteTolerance = settings.linearRelativeTolerance = 0;
    CHECK_THROWS(settings.Validate());
}

/** @brief Construct conservative test state. @param rho Density.
 * @param u Velocity. @param p Physical/gauge pressure. @return [rho,rho*u,p]. */
static State MakeState(real rho, const Vector3 &u, real p)
{
    State result;
    result << rho, rho * u(0), rho * u(1), rho * u(2), p;
    return result;
}

TEST_CASE("Variable ACM exact physical Jacobian and Roe secant in arbitrary frames")
{
    std::mt19937 generator(93003);
    std::uniform_real_distribution<real> random(-1, 1);
    for (int sample = 0; sample < 160; ++sample)
    {
        const State left = MakeState(std::exp(2 * random(generator)),
            Vector3(random(generator), random(generator), random(generator)), random(generator));
        const State right = MakeState(std::exp(2 * random(generator)),
            Vector3(random(generator), random(generator), random(generator)), random(generator));
        const Vector3 normal = Vector3(random(generator), random(generator), random(generator)).normalized();
        const Matrix3 basis = BuildLocalBasis(normal);
        CHECK((basis.transpose() * basis - Matrix3::Identity()).norm() < 1e-12);
        const State l = ToLocalState(left, basis), r = ToLocalState(right, basis);
        const Matrix5 A = PhysicalFluxJacobianLocal(l);
        Matrix5 fd;
        for (int k = 0; k < 5; ++k)
        {
            State delta = State::Zero();
            delta(k) = 1e-6 * (k == 0 ? l(0) : std::max(real(1), std::abs(l(k))));
            fd.col(k) = (PhysicalFluxLocal(l + delta) - PhysicalFluxLocal(l - delta)) / (2 * delta(k));
        }
        CHECK((fd - A).norm() < 2e-8 * std::max(real(1), A.norm()));
        const State mean = RoeAverage(l, r);
        CHECK((PhysicalFluxJacobianLocal(mean) * (r - l) -
               (PhysicalFluxLocal(r) - PhysicalFluxLocal(l))).norm() < 2e-12);
        Settings settings;
        settings.alpha = random(generator);
        settings.beta2 = 2.3;
        Matrix5 L, R;
        REQUIRE(TryCharacteristicMatricesGlobal(left, normal, settings, L, R));
        CHECK((L * R - Matrix5::Identity()).norm() < 2e-10);
        Matrix5 rotation = Matrix5::Identity();
        rotation.block<3,3>(1,1) = basis;
        const Matrix5 B = rotation * PreconditionedJacobianLocal(l, settings.beta2, settings.alpha) * rotation.transpose();
        const auto e = ComputeEigenvalues(Velocity(left).dot(normal), left(0), settings.beta2, settings.alpha);
        State lambdas;
        lambdas << e.lambdaMinus, e.lambdaTangential, e.lambdaTangential, e.lambdaTangential, e.lambdaPlus;
        CHECK((B * R - R * lambdas.asDiagonal()).norm() < 2e-9 * std::max(real(1), R.norm()));
        CHECK((GammaLocal(l, settings.beta2, settings.alpha) *
               GammaInvLocal(l, settings.beta2, settings.alpha) - Matrix5::Identity()).norm() < 1e-12);
        const State flux = InviscidFlux(RiemannSolverType::Roe, left, right, normal, settings).flux;
        CHECK((flux + InviscidFlux(RiemannSolverType::Roe, right, left, -normal, settings).flux).norm() < 2e-10);
        CHECK((InviscidFlux(RiemannSolverType::Roe, left, left, normal, settings).flux -
               FromLocalFlux(PhysicalFluxLocal(l), basis)).norm() < 1e-12);
    }
}

TEST_CASE("Variable ACM defective collision retains Jordan derivative")
{
    Settings settings;
    settings.alpha = 1;
    settings.beta2 = 2;
    settings.entropyFixRatio = 0;
    for (real q : {-1.0, 1.0})
    {
        State state = MakeState(2, Vector3(q, 0.3, -0.4), 0);
        Matrix5 L, R;
        CHECK_FALSE(TryCharacteristicMatricesGlobal(state, Vector3::UnitX(), settings, L, R));
        const Matrix5 B = PreconditionedJacobianLocal(state, settings.beta2, settings.alpha);
        const Matrix5 fB = AbsolutePreconditionedJacobianLocal(state, settings);
        CHECK((fB * fB - B * B).norm() < 1e-11);
        CHECK((fB * B - B * fB).norm() < 1e-11);
        // The pure contact eigenvector and a generalized vector form the collision chain.
        State contact = MakeState(1, Vector3(q, 0.3, -0.4), 0);
        CHECK(((B - q * Matrix5::Identity()) * contact).norm() < 1e-12);
        for (real eps : {-1e-6, 1e-6})
        {
            State nearby = state;
            nearby(1) += eps;
            CHECK((AbsolutePreconditionedJacobianLocal(nearby, settings) - fB).norm() < 1e-4);
        }
    }
}

TEST_CASE("Variable ACM density chain rule and linearly exact viscous face gradient")
{
    const State state = MakeState(2.1, Vector3(0.5, -0.2, 0.8), 3);
    Eigen::Matrix<real,3,5> primitive = Eigen::Matrix<real,3,5>::Random();
    Eigen::Matrix<real,3,5> conservative = primitive;
    conservative.block<3,3>(0,1) =
        state(0) * primitive.block<3,3>(0,1) + primitive.col(0) * Velocity(state).transpose();
    CHECK((PrimitiveGradient(state, conservative) - primitive).norm() < 1e-12);
    const State a = PrimitiveState(state);
    const Vector3 d(0.8, 0.4, -0.3), n = Vector3::UnitX();
    const State b = a + primitive.transpose() * d;
    CHECK((CorrectedFaceGradient(primitive, primitive, a, b, d, n) - primitive).norm() < 1e-12);
    const State flux = ViscousFlux(primitive, n, 0.07);
    CHECK(flux(0) == 0);
    CHECK(flux(4) == 0);
    CHECK(flux.segment<3>(1).allFinite());
}

TEST_CASE("Variable ACM walls reflect velocity not momentum with imposed density")
{
    const State state = MakeState(2.5, Vector3(1, 2, 3), 0.7);
    BoundaryCondition bc;
    bc.type = BoundaryType::BCWall;
    bc.value = {7, 0, 0, 0, 0};
    const State ghost = GenerateBoundaryState(bc, state, Vector3::UnitX(), Settings{});
    CHECK(ghost(0) == state(0));
    CHECK((Velocity(ghost) + Velocity(state)).norm() < 1e-12);
    CHECK(ghost(4) == state(4));
    bc.type = BoundaryType::BCOutP;
    bc.value[4] = 2;
    CHECK(GenerateBoundaryState(bc, state, Vector3::UnitX(), Settings{})(4) == doctest::Approx(3.3));
}

TEST_CASE("Variable ACM BDF acts on density and momentum but never pressure")
{
    const State u = MakeState(2, Vector3(1,2,3), 8);
    const State v = MakeState(1, Vector3(3,2,1), 2);
    const State w = MakeState(3, Vector3(2,1,2), 6);
    const auto c = GetBDF2Coefficients(1);
    const State derivative = EvaluateBDF2PhysicalDerivative(u,v,w,c,0.2);
    CHECK((derivative.head<4>() - ((1.5*u-2*v+0.5*w)/0.2).head<4>()).norm() < 1e-12);
    CHECK(derivative(4) == 0);
    StateField states{u};
    const auto zero = [](const StateField &in, StateField &out) { out.assign(in.size(), State::Zero()); };
    const auto jac = [](const StateField &in, MatrixField &out) { out.assign(in.size(), Matrix5::Zero()); };
    const ScalarField steps{0.001};
    CHECK(AdvanceExplicitSSPRK3(states,steps,Settings{},zero).converged);
    CHECK((states[0]-u).norm() < 1e-12);
    CHECK(AdvanceImplicitEulerBlockJacobi(states,steps,Settings{},TimeMarchSettings{},zero,jac).converged);
}

TEST_CASE("Variable ACM pseudo-time product rule and endpoint Rusanov bound")
{
    Settings settings;
    settings.alpha = 0.37;
    settings.beta2 = 2.4;
    const State previous = MakeState(1.2, Vector3(0.2,-0.4,0.1), -0.3);
    const State state = MakeState(2.1, Vector3(-0.7,0.3,0.8), 0.9);
    const real dt = 0.17;
    const Matrix5 analytic = PseudoTimeProductJacobian(state, previous, dt, settings);
    Matrix5 numerical;
    for (int k = 0; k < 5; ++k)
    {
        State d = State::Zero(); d(k) = 1e-7;
        const State plus = GammaLocal(state+d, settings.beta2, settings.alpha) * (state+d-previous) / dt;
        const State minus = GammaLocal(state-d, settings.beta2, settings.alpha) * (state-d-previous) / dt;
        numerical.col(k) = (plus-minus)/(2*d(k));
    }
    CHECK((analytic-numerical).norm() < 2e-8 * analytic.norm());

    const State left = MakeState(0.04, Vector3(-0.2,0,0), 0);
    const State right = MakeState(4.0, Vector3(0.1,0,0), 0);
    Eigenvalues reported;
    const State dissipation = RusanovDissipationLocal(left,right,settings,reported);
    const real endpoint = std::max(
        ComputeEigenvalues(Velocity(left)(0),left(0),settings.beta2,settings.alpha).SpectralRadius(),
        ComputeEigenvalues(Velocity(right)(0),right(0),settings.beta2,settings.alpha).SpectralRadius());
    const State gammaJump = ApplyGammaLocal(RoeAverage(left,right),right-left,settings.beta2,settings.alpha);
    CHECK((dissipation-endpoint*gammaJump).norm() < 1e-12);
}

TEST_CASE("Standard SA conservative rewrite retains density product rule")
{
    const Vector3 densityGradient(0.7,-0.2,0.5);
    const Vector3 saGradient(-0.4,0.3,0.1);
    CHECK(SADensityDiffusionCorrection(2.0,0.02,0.3,densityGradient,saGradient) ==
          doctest::Approx(-(0.01+0.3)/(2.0/3.0)*densityGradient.dot(saGradient)));
}

TEST_CASE("Every variable-density turbulence mode has finite closure and wall data")
{
    const std::array<TurbulenceModel,5> models{
        TurbulenceModel::Laminar, TurbulenceModel::SpalartAllmaras,
        TurbulenceModel::KOmegaWilcox, TurbulenceModel::KOmegaSST,
        TurbulenceModel::RealizableKEpsilon};
    VelocityGradient velocityGradient = VelocityGradient::Zero();
    velocityGradient(0,1) = 0.7;
    velocityGradient(1,0) = -0.2;
    TurbulenceGradient turbulenceGradient = TurbulenceGradient::Zero();
    turbulenceGradient.col(0) = Vector3(0.03,-0.01,0.02);
    turbulenceGradient.col(1) = Vector3(-0.04,0.02,0.01);
    for (const TurbulenceModel model : models)
    {
        TurbulenceSettings settings;
        settings.model = model;
        const TurbulenceState state = model == TurbulenceModel::SpalartAllmaras
            ? TurbulenceState(0.002,0) : TurbulenceState(0.2,3.0);
        const real eddy = TurbulentDynamicViscosity(
            state,velocityGradient,turbulenceGradient,0.04,1.7,0.01,settings);
        CHECK(std::isfinite(eddy)); CHECK(eddy >= 0);
        CHECK(TurbulenceSource(state,velocityGradient,turbulenceGradient,
                               0.04,1.7,0.01,settings).allFinite());
        CHECK(TurbulenceDiffusiveFlux(state,turbulenceGradient,Vector3::UnitX(),
                                      0.04,1.7,0.01,eddy,settings).allFinite());
        CHECK(GenerateTurbulenceBoundaryState(BoundaryType::BCWall,state,0.04,
                                              1.7,0.01,settings).allFinite());
    }
}
