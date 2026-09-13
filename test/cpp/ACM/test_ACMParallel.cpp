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
    cfg.acmSettings.farFieldValue = cfg.initialState;
    cfg.boundaryValue = cfg.initialState;
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
    configuration.boundaryValue = configuration.initialState;
    configuration.acmSettings.farFieldValue = configuration.initialState;
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
