/**
 * @file test_ACMParallel.cpp
 * @brief MPI unit tests for ACM face-buffer evaluation and global checksum reduction.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "ACM/ACMParallel.hpp"
#include "ACM/ACMSolver.hpp"
#include "ACM/ACMTime.hpp"

#include <algorithm>
#include <filesystem>
#include <vector>

/**
 * @brief Initialize MPI around the doctest runner used by the parallel ACM test executable.
 * @param argc Number of command-line arguments.
 * @param argv Command-line arguments forwarded to doctest.
 * @return Doctest process result after all registered tests have run.
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}

using namespace DNDS;
using namespace DNDS::ACM;

using ACMConvergence2D = std::integral_constant<ACMModel, ACMModel::ConstantDensity2D>;
using ACMConvergence3D = std::integral_constant<ACMModel, ACMModel::ConstantDensity3D>;

/// @test Degree-zero VR must retain the first-order residual on nonuniform MPI flow fields.
TEST_CASE_TEMPLATE("ACM Wilcox degree-zero reconstruction matches first order", TModel, ACMConvergence2D, ACMConvergence3D)
{
    MPIInfo mpi;
    mpi.setWorld();
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    KernelConfiguration cfg;
    cfg.meshSettings.meshFile = (root / (TModel::value == ACMModel::ConstantDensity2D
                                           ? "data/mesh/ACMVariable_verify2D.cgns"
                                           : "data/mesh/ACMVariable_verify3D.cgns")).string();
    cfg.acmSettings.enableViscousFlux = true;
    cfg.acmSettings.dynamicViscosity = 0.001;
    cfg.turbulenceSettings.model = TurbulenceModel::KOmegaWilcox;
    cfg.turbulenceSettings.initialValue = {0.001, 10.0};
    cfg.reconstructionSettings.type = ReconstructionType::Variational;
    cfg.reconstructionSettings.limiterType = LimiterType::CWBAP;
    cfg.vfvSettings.maxOrder = 0;
    using TSolver = ACMSolver<TModel::value>;
    TSolver constant(mpi, cfg);
    constant.ReadMeshAndInitialize();
    cfg.reconstructionSettings.type = ReconstructionType::FirstOrder;
    cfg.reconstructionSettings.limiterType = LimiterType::LocalExtrema;
    cfg.vfvSettings.maxOrder = 1;
    TSolver firstOrder(mpi, cfg);
    firstOrder.ReadMeshAndInitialize();
    for (TSolver *solver : {&constant, &firstOrder})
        for (DNDS::index i = 0; i < solver->GetMesh()->NumCell(); i++)
        {
            const auto x = solver->GetReconstruction()->GetCellBary(i);
            solver->GetState()[i] << 0.4 + 0.08 * std::sin(x(0) + x(1)),
                -0.2 + 0.08 * std::cos(2 * x(0)), 0.0, 0.1 + 0.08 * std::sin(x(1));
        }
    typename TSolver::TDof constantRhs, referenceRhs;
    constant.GetReconstruction()->BuildUDof(constantRhs, 4);
    firstOrder.GetReconstruction()->BuildUDof(referenceRhs, 4);
    constant.GetEvaluator()->EvaluateRHS(constantRhs, constant.GetState());
    firstOrder.GetEvaluator()->EvaluateRHS(referenceRhs, firstOrder.GetState());
    constantRhs.addTo(referenceRhs, -1);
    CHECK(constantRhs.norm2() < 1e-13);
}

/// @test Check warm-start independence, fixed-state repeatability and collective cap failure in 2D/3D.
TEST_CASE_TEMPLATE("ACM converged reconstruction defines a repeatable residual", TModel, ACMConvergence2D, ACMConvergence3D)
{
    MPIInfo mpi;
    mpi.setWorld();
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    KernelConfiguration cfg;
    cfg.meshSettings.meshFile = (root / (TModel::value == ACMModel::ConstantDensity2D
                                           ? "data/mesh/ACMVariable_verify2D.cgns"
                                           : "data/mesh/ACMVariable_verify3D.cgns")).string();
    cfg.initialState = {0.4, -0.2, 0.0, 0.1};
    std::copy_n(cfg.initialState.begin(), cfg.acmSettings.farFieldValue.size(), cfg.acmSettings.farFieldValue.begin());
    std::copy_n(cfg.initialState.begin(), cfg.boundaryValue.size(), cfg.boundaryValue.begin());
    cfg.reconstructionSettings.type = ReconstructionType::Variational;
    cfg.reconstructionSettings.variationalTolerance = 1e-13;
    cfg.reconstructionSettings.variationalMaxIterations = 10000;
    cfg.vfvSettings.maxOrder = 2;
    cfg.Validate();
    using TSolver = ACMSolver<TModel::value>;
    TSolver warm(mpi, cfg);
    warm.ReadMeshAndInitialize();
    typename TSolver::TDof rhs, repeated, reference;
    for (auto *field : {&rhs, &repeated, &reference})
        warm.GetReconstruction()->BuildUDof(*field, 4);
    const auto setState = [](TSolver &solver, DNDS::real amplitude)
    {
        for (DNDS::index i = 0; i < solver.GetMesh()->NumCell(); i++)
        {
            const auto x = solver.GetReconstruction()->GetCellBary(i);
            solver.GetState()[i] << 0.4 + amplitude * std::sin(x(0) + x(1)),
                -0.2 + amplitude * std::cos(2 * x(0)), 0.0,
                0.1 + amplitude * std::sin(x(1));
        }
    };
    setState(warm, 0.03);
    warm.GetEvaluator()->EvaluateRHS(rhs, warm.GetState());
    setState(warm, 0.08);
    warm.GetEvaluator()->EvaluateRHS(rhs, warm.GetState());
    CHECK(warm.GetEvaluator()->GetReconstructionReport().converged);
    warm.GetEvaluator()->EvaluateRHS(repeated, warm.GetState());
    repeated.addTo(rhs, -1);
    CHECK(repeated.norm2() < 1e-13);

    // A fresh reconstruction of the same flow must agree despite the different history.
    TSolver cold(mpi, cfg);
    cold.ReadMeshAndInitialize();
    setState(cold, 0.08);
    cold.GetEvaluator()->EvaluateRHS(reference, cold.GetState());
    reference.addTo(rhs, -1);
    CHECK(reference.norm2() < 1e-8);

    cfg.reconstructionSettings.variationalIterations = 1;
    cfg.reconstructionSettings.variationalMaxIterations = 1;
    cfg.reconstructionSettings.variationalTolerance = 1e-30;
    TSolver capped(mpi, cfg);
    capped.ReadMeshAndInitialize();
    setState(capped, 0.08);
    CHECK_THROWS_WITH_AS(capped.GetEvaluator()->EvaluateRHS(reference, capped.GetState()),
                         doctest::Contains("reconstruction did not converge"), std::runtime_error);
}

/// @test Fixed BDF reconstruction must depend only on the current state, including after a linearization call.
TEST_CASE_TEMPLATE("ACM BDF2 fixed reconstruction is independent of RHS history", TModel, ACMConvergence2D, ACMConvergence3D)
{
    MPIInfo mpi;
    mpi.setWorld();
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    KernelConfiguration cfg;
    cfg.meshSettings.meshFile = (root / (TModel::value == ACMModel::ConstantDensity2D
                                           ? "data/mesh/ACMVariable_verify2D.cgns"
                                           : "data/mesh/ACMVariable_verify3D.cgns")).string();
    cfg.initialState = {0.4, -0.2, 0, 0.1};
    std::copy_n(cfg.initialState.begin(), cfg.boundaryValue.size(), cfg.boundaryValue.begin());
    cfg.reconstructionSettings.type = ReconstructionType::Variational;
    cfg.reconstructionSettings.variationalIterations = 3;
    cfg.reconstructionSettings.resetVariationalCoefficients = false;
    cfg.timeMarchSettings.integrator = TimeIntegratorType::BDF2DualTimeLUSGS;
    cfg.vfvSettings.maxOrder = 2;
    using TSolver = ACMSolver<TModel::value>;
    TSolver solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    typename TSolver::TDof rhs, repeated;
    solver.GetReconstruction()->BuildUDof(rhs, 4);
    solver.GetReconstruction()->BuildUDof(repeated, 4);
    for (DNDS::index i = 0; i < solver.GetMesh()->NumCell(); i++)
    {
        const auto x = solver.GetReconstruction()->GetCellBary(i);
        solver.GetState()[i] << 0.4 + 0.08 * std::sin(x(0) + x(1)),
            -0.2 + 0.08 * std::cos(2 * x(0)), 0, 0.1 + 0.08 * std::sin(x(1));
    }
    solver.GetEvaluator()->EvaluateRHS(rhs, solver.GetState());
    solver.GetEvaluator()->EvaluateRHS(repeated, solver.GetState());
    repeated.addTo(rhs, -1);
    CHECK(repeated.norm2() < 1e-13);
    MatrixField diagonal;
    typename TSolver::TEvaluator::FaceJacobianField faces;
    solver.GetEvaluator()->AssembleImplicitLinearization(
        solver.GetState(), ScalarField(static_cast<std::size_t>(solver.GetMesh()->NumCell()), 0.01), diagonal, faces);
    solver.GetEvaluator()->EvaluateRHS(repeated, solver.GetState());
    repeated.addTo(rhs, -1);
    CHECK(repeated.norm2() < 1e-13);
}

/// @test Physical RANS norms, BE fallback counts and rejected steps must agree on every MPI rank.
TEST_CASE_TEMPLATE("ACM RANS physical defect statistics and rejection are MPI global", TModel, ACMConvergence2D, ACMConvergence3D)
{
    MPIInfo mpi;
    mpi.setWorld();
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    KernelConfiguration cfg;
    cfg.meshSettings.meshFile = (root / (TModel::value == ACMModel::ConstantDensity2D
                                           ? "data/mesh/ACMVariable_verify2D.cgns"
                                           : "data/mesh/ACMVariable_verify3D.cgns")).string();
    cfg.initialState = {0, 0, 0, 0.3};
    cfg.defaultBoundaryType = BoundaryType::BCSym;
    cfg.reconstructionSettings.type = ReconstructionType::FirstOrder;
    cfg.acmSettings.enableViscousFlux = true;
    cfg.acmSettings.dynamicViscosity = 0.001;
    cfg.turbulenceSettings.model = TurbulenceModel::KOmegaSST;
    cfg.turbulenceSettings.initialValue = {0.2, 4};
    cfg.turbulenceSettings.secondOrderReconstruction = false;
    cfg.timeMarchSettings.integrator = TimeIntegratorType::BDF2DualTimeLUSGS;
    cfg.timeMarchSettings.nSteps = 1;
    cfg.timeMarchSettings.maxImplicitIterations = 1;
    cfg.timeMarchSettings.implicitTolerance = 1e-12;
    using TSolver = ACMSolver<TModel::value>;
    TSolver solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    auto boundary = std::make_shared<BoundaryHandler>(cfg.defaultBoundaryType, cfg.InitialState(), cfg.boundaryConditions);
    typename TSolver::TTurbulence transport(solver.GetMesh(), solver.GetReconstruction(),
                                             cfg.acmSettings, cfg.turbulenceSettings, boundary);
    transport.Initialize();
    transport.InitializePhysicalHistory();
    TurbulenceState newer;
    newer << 1e-4, 3.9;
    transport.GetState().setConstant(newer);
    transport.CommitPhysicalStep();
    constexpr DNDS::real dt = 0.05;
    const DNDS::real norm = transport.EvaluatePhysicalBDF2DefectNorm(
        solver.GetState(), GetBDF2Coefficients(1), dt, 2 * dt);
    const DNDS::real kDefect = 0.09 * newer(0) * newer(1);
    const DNDS::real omegaDefect = 0.0828 * newer(1) * newer(1) +
                                  (1.5 * newer(1) - 2 * newer(1) + 0.5 * 4) / dt;
    CHECK(transport.GetPhysicalEquationDefectNorms()[0] == doctest::Approx(kDefect));
    CHECK(transport.GetPhysicalEquationDefectNorms()[1] == doctest::Approx(omegaDefect));
    CHECK(norm == doctest::Approx(std::hypot(kDefect, omegaDefect) / std::sqrt(2.0)));
    CHECK(transport.GetPhysicalBackwardEulerCounts()[0] == solver.GetMesh()->NumCellGlobal());
    CHECK(transport.GetPhysicalBackwardEulerCounts()[1] == 0);
    CHECK_THROWS_WITH_AS(solver.Run(), doctest::Contains("histories were not committed"), std::runtime_error);
}

/// @test Verify face-index independence and rank-count scaling of the MPI all-reduced checksum.
TEST_CASE("ACM face buffer and MPI reduction are rank-count invariant")
{
    MPIInfo mpi;
    mpi.setWorld();
    Settings settings;
    settings.riemannSolverType = RiemannSolverType::Roe;
    settings.rho0 = 1.0;
    settings.beta2 = 2.0;

    FaceInput face;
    face.left << 1.0, 0.2, -0.1, 0.4;
    face.right << 0.3, -0.2, 0.5, 0.9;
    face.unitNormal = Vector3(1.0, 2.0, -1.0).normalized();
    constexpr int nFaces = 32;
    std::vector<FaceInput> faces(nFaces, face);
    std::vector<FluxResult> faceFluxBuffer;
    EvaluateFaceFluxes(faces, faceFluxBuffer, settings);

    REQUIRE(faceFluxBuffer.size() == faces.size());
    const FluxResult reference = InviscidFlux(
        settings.riemannSolverType, face.left, face.right, face.unitNormal, settings);
    for (const auto &result : faceFluxBuffer)
        CHECK((result.flux - reference.flux).norm() < 1e-12);

    const State localSum = LocalFluxSum(faceFluxBuffer);
    const State globalSum = GlobalFluxSum(localSum, mpi);
    const State expected = reference.flux * static_cast<real>(nFaces * mpi.size);
    for (int i = 0; i < 4; i++)
        CHECK(globalSum(i) == doctest::Approx(expected(i)).epsilon(1e-11));
}

/// @test Exercise global time-step diagnostics and zero-residual preservation on every MPI rank.
TEST_CASE("ACM explicit and implicit time stepping preserve a distributed steady state")
{
    MPIInfo mpi;
    mpi.setWorld();
    Settings settings;
    StateField states(8);
    for (std::size_t i = 0; i < states.size(); i++)
        states[i] << 0.2 * static_cast<real>(mpi.rank + 1), -0.1, 0.3, 1.0;
    const StateField reference = states;
    const ScalarField pseudoTimeStep(states.size(), 0.05);
    const ResidualEvaluator zeroResidual = [](const StateField &input, StateField &residual)
    {
        residual.assign(input.size(), State::Zero());
    };
    const DiagonalJacobianEvaluator zeroJacobian = [](const StateField &input, MatrixField &jacobian)
    {
        jacobian.assign(input.size(), Matrix4::Zero());
    };

    const auto explicitReport = AdvanceExplicitSSPRK3(
        states,
        pseudoTimeStep,
        settings,
        zeroResidual,
        &mpi);
    CHECK(explicitReport.converged);
    CHECK(explicitReport.initialDefectNorm == doctest::Approx(0.0));

    TimeMarchSettings timeSettings;
    timeSettings.integrator = TimeIntegratorType::ImplicitEulerBlockJacobi;
    const auto implicitReport = AdvanceImplicitEulerBlockJacobi(
        states,
        pseudoTimeStep,
        settings,
        timeSettings,
        zeroResidual,
        zeroJacobian,
        &mpi);
    CHECK(implicitReport.converged);
    CHECK(implicitReport.iterations == 1);
    for (std::size_t i = 0; i < states.size(); i++)
        CHECK((states[i] - reference[i]).norm() < 1e-14);
}

/// @test Verify every additional LU-SGS sweep is a true `b-A*x` residual correction.
TEST_CASE("ACM LU-SGS multiple sweeps apply residual correction")
{
    MPIInfo mpi;
    mpi.setWorld();
    const std::filesystem::path root = std::filesystem::path(__FILE__).parent_path()
                                               .parent_path().parent_path().parent_path();

    KernelConfiguration configuration;
    configuration.meshSettings.meshFile =
        (root / "data/mesh/ACMVariable_verify2D.cgns").string();
    configuration.initialState = {0.4, -0.2, 0.1, 0.3};
    std::copy_n(configuration.initialState.begin(), configuration.boundaryValue.size(), configuration.boundaryValue.begin());
    std::copy_n(configuration.initialState.begin(), configuration.acmSettings.farFieldValue.size(), configuration.acmSettings.farFieldValue.begin());
    configuration.acmSettings.beta2 = 2.0;
    configuration.acmSettings.entropyFixRatio = 0;
    configuration.reconstructionSettings.type = ReconstructionType::FirstOrder;
    configuration.reconstructionSettings.enableLimiter = false;
    configuration.Validate();

    ACMSolver<ACMModel::ConstantDensity2D> solver(mpi, configuration);
    solver.ReadMeshAndInitialize();
    const auto &vfv = solver.GetReconstruction();
    const auto &evaluator = solver.GetEvaluator();
    auto &state = solver.GetState();
    REQUIRE(vfv != nullptr);
    REQUIRE(evaluator != nullptr);

    using TDof = ACMSolver<ACMModel::ConstantDensity2D>::TDof;
    TDof rhs;
    TDof oneSweep;
    TDof twoSweeps;
    TDof firstResidual;
    TDof secondResidual;
    TDof operatorProduct;
    TDof correction;
    TDof expected;
    TDof difference;
    for (TDof *field : {&rhs, &oneSweep, &twoSweeps, &firstResidual,
                        &secondResidual, &operatorProduct, &correction,
                        &expected, &difference})
        vfv->BuildUDof(*field, 4);

    for (DNDS::index iCell = 0; iCell < solver.GetMesh()->NumCell(); iCell++)
    {
        const auto barycenter = vfv->GetCellBary(iCell);
        rhs[iCell] << 1.0 + 0.2 * barycenter(0),
            -0.3 + 0.1 * barycenter(1),
            0.2 + 0.05 * barycenter(0),
            0.5 - 0.1 * barycenter(1);
    }

    const ScalarField pseudoTimeStep(
        static_cast<std::size_t>(solver.GetMesh()->NumCell()), 0.01);
    MatrixField diagonal;
    ACMEvaluator<2>::FaceJacobianField faceJacobians;
    evaluator->AssembleImplicitLinearization(
        state, pseudoTimeStep, diagonal, faceJacobians, 0);

    evaluator->SolveLUSGS(rhs, diagonal, faceJacobians, oneSweep, 1);
    evaluator->ApplyImplicitLinearization(
        oneSweep, diagonal, faceJacobians, operatorProduct);
    firstResidual = rhs;
    firstResidual.addTo(operatorProduct, -1);

    evaluator->SolveLUSGS(
        firstResidual, diagonal, faceJacobians, correction, 1);
    expected = oneSweep;
    expected.addTo(correction, 1);

    evaluator->SolveLUSGS(rhs, diagonal, faceJacobians, twoSweeps, 2);
    difference = twoSweeps;
    difference.addTo(expected, -1);

    evaluator->ApplyImplicitLinearization(
        twoSweeps, diagonal, faceJacobians, operatorProduct);
    secondResidual = rhs;
    secondResidual.addTo(operatorProduct, -1);

    const real rhsNorm = rhs.norm2();
    const real firstResidualNorm = firstResidual.norm2();
    const real secondResidualNorm = secondResidual.norm2();
    const real correctionNorm = correction.norm2();
    const real expectedNorm = expected.norm2();
    CHECK(difference.norm2() < 1e-11 * std::max(real(1), expectedNorm));
    CHECK(correctionNorm > 1e-10 * std::max(real(1), oneSweep.norm2()));
    CHECK(firstResidualNorm < rhsNorm);
    CHECK(secondResidualNorm < firstResidualNorm);
}

using ACMUnifiedSA2D = std::integral_constant<ACMModel, ACMModel::ConstantDensitySA2D>;
using ACMUnifiedSA3D = std::integral_constant<ACMModel, ACMModel::ConstantDensitySA3D>;
using ACMUnified2EQ2D = std::integral_constant<ACMModel, ACMModel::ConstantDensity2EQ2D>;
using ACMUnified2EQ3D = std::integral_constant<ACMModel, ACMModel::ConstantDensity2EQ3D>;

TEST_CASE_TEMPLATE("ACM unified conservative transport uses the same face mass flux across MPI partitions",
                   TModel, ACMUnifiedSA2D, ACMUnifiedSA3D, ACMUnified2EQ2D, ACMUnified2EQ3D)
{
    MPIInfo mpi;
    mpi.setWorld();
    constexpr auto model = TModel::value;
    using Traits = ModelTraits<model>;
    using TSolver = ACMSolver<model>;
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    KernelConfiguration cfg;
    cfg.solver = ModelSelection<model>();
    cfg.meshSettings.meshFile = (root / (Traits::gDim == 2 ? "data/mesh/ACMVariable_verify2D.cgns"
                                                          : "data/mesh/ACMVariable_verify3D.cgns")).string();
    cfg.defaultBoundaryType = BoundaryType::BCOut;
    cfg.acmSettings.rho0 = 2.7;
    cfg.acmSettings.enableViscousFlux = true;
    cfg.acmSettings.dynamicViscosity = 0.001;
    cfg.turbulenceSettings.model = Traits::nExtraVars == 1 ? TurbulenceModel::SpalartAllmaras
                                                         : TurbulenceModel::KOmegaSST;
    cfg.turbulenceSettings.enableSourceTerms = false;
    cfg.turbulenceSettings.farFieldValue = {0.002, 3.2};
    cfg.turbulenceSettings.transportSubsteps = 1;
    cfg.initialState = {0.3, -0.2, 0, 0.1, 2.7 * 0.002};
    if constexpr (Traits::nExtraVars == 2)
        cfg.initialState.push_back(2.7 * 3.2);
    cfg.reconstructionSettings.type = ReconstructionType::Variational;
    cfg.reconstructionSettings.variationalIterations = 12;
    cfg.reconstructionSettings.resetVariationalCoefficients = true;
    cfg.reconstructionSettings.enableLimiter = false;
    cfg.vfvSettings.maxOrder = 2;
    cfg.vfvSettings.intOrder = 4;
    TSolver solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); cell++)
    {
        const auto center = solver.GetReconstruction()->GetCellBary(cell);
        solver.GetState()[cell](0) = 0.3 + 0.1 * std::sin(2 * 3.141592653589793 * center(0));
        solver.GetState()[cell](3) = 0.1 + 0.01 * std::cos(2 * 3.141592653589793 * center(1));
    }
    typename TSolver::TDof rhs;
    solver.GetReconstruction()->BuildUDof(rhs, Traits::nVarsFixed);
    solver.EvaluateRHS(rhs);
    DNDS::real localError = 0, localSignal = 0;
    for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); cell++)
    {
        localSignal = std::max(localSignal, std::abs(rhs[cell](3)));
        for (int variable = 4; variable < Traits::nVarsFixed; variable++)
            localError = std::max(localError, std::abs(rhs[cell](variable) - cfg.initialState[variable] * rhs[cell](3)));
    }
    DNDS::real error = 0, signal = 0;
    MPI::Allreduce(&localError, &error, 1, DNDS_MPI_REAL, MPI_MAX, mpi.comm);
    MPI::Allreduce(&localSignal, &signal, 1, DNDS_MPI_REAL, MPI_MAX, mpi.comm);
    CHECK(signal > 1e-4);
    CHECK(error < 1e-9 * std::max(DNDS::real(1), signal));
}
