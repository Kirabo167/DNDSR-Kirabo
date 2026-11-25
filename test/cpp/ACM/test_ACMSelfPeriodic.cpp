/**
 * @file test_ACMSelfPeriodic.cpp
 * @brief Regression tests for periodic ACM implicit operators and primitive RANS physical time.
 */
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "ACM/ACMSolver.hpp"
#include "ACMVariable/ACMSolver.hpp"
#include "Euler/EulerEvaluator.hpp"

#include <cgnslib.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

using namespace DNDS;

namespace
{
    void CheckCGNS(int status)
    {
        if (status != CG_OK)
            throw std::runtime_error(cg_get_error());
    }

    std::filesystem::path MakeTempMeshPath()
    {
        static int uniquenessAnchor;
        auto nonce = static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        nonce ^= static_cast<unsigned long long>(
            reinterpret_cast<std::uintptr_t>(&uniquenessAnchor));
        return std::filesystem::temp_directory_path() /
               ("dndsr_acm_self_periodic_" + std::to_string(nonce) + ".cgns");
    }

    struct TempMeshFile
    {
        std::filesystem::path path = MakeTempMeshPath();

        ~TempMeshFile()
        {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    };

    void WriteSingleCellQuad(const std::filesystem::path &path, bool periodic = true, bool skew = false)
    {
        int file = -1;
        CheckCGNS(cg_open(path.c_str(), CG_MODE_WRITE, &file));
        try
        {
            int base = 0;
            int zone = 0;
            int coordinate = 0;
            int section = 0;
            int boundary = 0;
            CheckCGNS(cg_base_write(file, "Base", 2, 2, &base));

            cgsize_t zoneSize[3] = {4, 1, 0};
            CheckCGNS(cg_zone_write(
                file, base, "Zone", zoneSize, Unstructured, &zone));

            const std::array<double, 4> x{0, 1, skew ? 0.37 : 0, skew ? 1.37 : 1};
            const std::array<double, 4> y{0, 0, skew ? 0.01 : 1, skew ? 0.01 : 1};
            CheckCGNS(cg_coord_write(
                file, base, zone, RealDouble, "CoordinateX", x.data(), &coordinate));
            CheckCGNS(cg_coord_write(
                file, base, zone, RealDouble, "CoordinateY", y.data(), &coordinate));

            const std::array<cgsize_t, 4> cell{1, 2, 4, 3};
            CheckCGNS(cg_section_write(
                file, base, zone, "Cells", QUAD_4, 1, 1, 0, cell.data(), &section));

            const std::array<const char *, 4> names = periodic
                                                        ? std::array<const char *, 4>{"PERIODIC_1", "PERIODIC_1_DONOR",
                                                                                     "PERIODIC_2", "PERIODIC_2_DONOR"}
                                                        : std::array<const char *, 4>{"Left", "Right", "Bottom", "Top"};
            const std::array<std::array<cgsize_t, 2>, 4> edges{{
                {{1, 3}}, {{2, 4}}, {{1, 2}}, {{3, 4}}
            }};
            for (std::size_t i = 0; i < names.size(); i++)
            {
                const cgsize_t element = static_cast<cgsize_t>(2 + i);
                CheckCGNS(cg_section_write(
                    file, base, zone, names[i], BAR_2,
                    element, element, 0, edges[i].data(), &section));
                cgsize_t range[2] = {element, element};
                CheckCGNS(cg_boco_write(
                    file, base, zone, names[i], BCTypeNull,
                    PointRange, 2, range, &boundary));
                CheckCGNS(cg_boco_gridlocation_write(
                    file, base, zone, boundary, EdgeCenter));
            }

            CheckCGNS(cg_close(file));
            file = -1;
        }
        catch (...)
        {
            if (file >= 0)
                cg_close(file);
            throw;
        }
    }

    template <int nVars, class TSolver>
    void VerifySelfPeriodicImplicitOperator(TSolver &solver)
    {
        using TDof = typename TSolver::TDof;
        using TEvaluator = typename TSolver::TEvaluator;
        using Matrix = Eigen::Matrix<real, nVars, nVars>;

        solver.ReadMeshAndInitialize();
        const auto &mesh = solver.GetMesh();
        const auto &vfv = solver.GetReconstruction();
        const auto &evaluator = solver.GetEvaluator();
        REQUIRE(mesh->NumCell() == 1);

        int selfIncidences = 0;
        const auto cellFaces = mesh->cell2face[0];
        for (rowsize ic2f = 0; ic2f < cellFaces.size(); ic2f++)
            if (mesh->CellFaceOther(0, cellFaces[ic2f], ic2f) == 0)
                selfIncidences++;
        REQUIRE(selfIncidences == 4);

        TDof rhs;
        TDof blockSolution;
        TDof lusgsSolution;
        TDof operatorProduct;
        for (TDof *field : {&rhs, &blockSolution, &lusgsSolution, &operatorProduct})
            vfv->BuildUDof(*field, nVars);

        for (int i = 0; i < nVars; i++)
            rhs[0](i) = 0.25 + 0.1 * static_cast<real>(i);

        std::vector<Matrix> residualDiagonal;
        evaluator->EvaluateDiagonalJacobian(
            solver.GetState(), residualDiagonal, 0);
        REQUIRE(residualDiagonal.size() == 1);
        CHECK(residualDiagonal[0].norm() < 1e-10);

        const ACM::ScalarField pseudoTimeStep(1, 0.02);
        std::vector<Matrix> diagonal;
        typename TEvaluator::FaceJacobianField faceJacobians;
        evaluator->AssembleImplicitLinearization(
            solver.GetState(), pseudoTimeStep, diagonal, faceJacobians, 0);

        evaluator->ApplyBlockJacobi(rhs, diagonal, blockSolution);
        evaluator->ApplyImplicitLinearization(
            blockSolution, diagonal, faceJacobians, operatorProduct);
        CHECK((operatorProduct[0] - rhs[0]).norm() < 1e-10);

        evaluator->SolveLUSGS(
            rhs, diagonal, faceJacobians, lusgsSolution, 1);
        evaluator->ApplyImplicitLinearization(
            lusgsSolution, diagonal, faceJacobians, operatorProduct);
        CHECK((operatorProduct[0] - rhs[0]).norm() < 1e-10);
    }

    ACM::KernelConfiguration MakeRANSPhysicalCase(
        const std::filesystem::path &mesh, ACM::TurbulenceModel model,
        ACM::TimeIntegratorType integrator, DNDS::real dt, int steps, bool secondOrder = false)
    {
        ACM::KernelConfiguration cfg;
        cfg.meshSettings.meshFile = mesh.string();
        cfg.defaultBoundaryType = ACM::BoundaryType::BCSym;
        cfg.initialState = {0, 0, 0, 0.3};
        cfg.acmSettings.enableViscousFlux = true;
        cfg.acmSettings.dynamicViscosity = 0.001;
        cfg.reconstructionSettings.type = ACM::ReconstructionType::GreenGauss;
        cfg.turbulenceSettings.model = model;
        cfg.turbulenceSettings.initialValue = {0.4, 0.08};
        if (model == ACM::TurbulenceModel::SpalartAllmaras)
        {
            cfg.initialState[0] = 1;
            cfg.turbulenceSettings.initialValue = {0.2, 100.0};
            cfg.turbulenceSettings.farFieldValue = {0.1, 100.0};
            cfg.turbulenceSettings.enableSourceTerms = false;
            ACM::BoundaryCondition inlet, outlet;
            inlet.name = "Left";
            inlet.type = ACM::BoundaryType::BCIn;
            std::copy_n(cfg.initialState.begin(), inlet.value.size(), inlet.value.begin());
            outlet.name = "Right";
            outlet.type = ACM::BoundaryType::BCOut;
            cfg.boundaryConditions = {inlet, outlet};
        }
        std::copy_n(cfg.initialState.begin(), cfg.acmSettings.farFieldValue.size(), cfg.acmSettings.farFieldValue.begin());
        cfg.turbulenceSettings.secondOrderReconstruction = secondOrder;
        cfg.turbulenceSettings.transportTimeScale = 1;
        cfg.timeMarchSettings.integrator = integrator;
        cfg.timeMarchSettings.nSteps = steps;
        cfg.timeMarchSettings.pseudoTimeStep = 0.01;
        cfg.timeMarchSettings.physicalTimeStep = dt;
        cfg.timeMarchSettings.maxImplicitIterations = 500;
        cfg.timeMarchSettings.implicitTolerance = 1e-12;
        cfg.timeMarchSettings.linearAbsoluteTolerance = 1e-14;
        cfg.Validate();
        return cfg;
    }

    ACM::TurbulenceState ExactRANSDiscreteDecay(ACM::TurbulenceModel model, DNDS::real dt, int steps)
    {
        ACM::TurbulenceState previous, older;
        previous << 0.4, 0.08;
        if (model == ACM::TurbulenceModel::SpalartAllmaras)
            previous << 0.2, 0;
        older = previous;
        for (int step = 0; step < steps; step++)
        {
            const DNDS::real a0 = step == 0 ? 1.0 : 1.5;
            const DNDS::real a1 = step == 0 ? -1.0 : -2.0;
            const DNDS::real a2 = step == 0 ? 0.0 : 0.5;
            const ACM::TurbulenceState historyRhs = -a1 * previous - a2 * older;
            ACM::TurbulenceState next = ACM::TurbulenceState::Zero();
            if (model == ACM::TurbulenceModel::SpalartAllmaras)
            {
                // Convection uses the prescribed face value; only diffusion reflects.
                // On a unit cell dq/dt=(1+2*D_in)*(q_in-q), D_in=(nu+q_in)/(2/3).
                constexpr DNDS::real inlet = 0.1;
                constexpr DNDS::real rate = 1 + 2 * (0.001 + inlet) / (2.0 / 3.0);
                next(0) = (historyRhs(0) + dt * rate * inlet) / (a0 + dt * rate);
            }
            else
            {
                // Zero-strain high-Re decay: dk/dt=-epsilon,
                // d(epsilon)/dt=-1.92*epsilon^2/k. Eliminate k to solve the positive root.
                const DNDS::real a = a0 * dt * (1.92 - 1);
                const DNDS::real b = a0 * historyRhs(0) + dt * historyRhs(1);
                const DNDS::real c = historyRhs(0) * historyRhs(1);
                next(1) = 2 * c / (b + std::sqrt(b * b + 4 * a * c));
                next(0) = (historyRhs(0) - dt * next(1)) / a0;
            }
            older = previous;
            previous = next;
        }
        return previous;
    }
}

TEST_CASE("ACM implicit solvers fold self-periodic coupling into the diagonal")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);

    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path);

    ACM::KernelConfiguration configuration;
    configuration.meshSettings.meshFile = meshFile.path.string();
    configuration.meshSettings.periodicTranslation1 = {1, 0, 0};
    configuration.meshSettings.periodicTranslation2 = {0, 1, 0};
    configuration.initialState = {0.4, -0.2, 0.1, 0.3};
    std::copy_n(configuration.initialState.begin(), configuration.boundaryValue.size(), configuration.boundaryValue.begin());
    std::copy_n(configuration.initialState.begin(), configuration.acmSettings.farFieldValue.size(), configuration.acmSettings.farFieldValue.begin());
    configuration.reconstructionSettings.type = ACM::ReconstructionType::FirstOrder;
    configuration.reconstructionSettings.enableLimiter = false;
    configuration.Validate();

    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, configuration);
    VerifySelfPeriodicImplicitOperator<4>(solver);
}

TEST_CASE("ACMVariable implicit solvers fold self-periodic coupling into the diagonal")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);

    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path);

    ACMVariable::KernelConfiguration configuration;
    configuration.meshSettings.meshFile = meshFile.path.string();
    configuration.meshSettings.periodicTranslation1 = {1, 0, 0};
    configuration.meshSettings.periodicTranslation2 = {0, 1, 0};
    configuration.initialState = {1.1, 0.2, -0.1, 0.05, 0.3};
    std::copy_n(configuration.initialState.begin(), configuration.boundaryValue.size(), configuration.boundaryValue.begin());
    std::copy_n(configuration.initialState.begin(), configuration.acmSettings.farFieldValue.size(), configuration.acmSettings.farFieldValue.begin());
    configuration.reconstructionSettings.type =
        ACMVariable::ReconstructionType::FirstOrder;
    configuration.reconstructionSettings.enableLimiter = false;
    configuration.Validate();

    ACMVariable::ACMSolver<ACMVariable::ACMModel::VariableDensity2D> solver(
        mpi, configuration);
    VerifySelfPeriodicImplicitOperator<5>(solver);
}

/// @test A floor-bound k never freezes omega; physical residuals are not projected to hide the blocked equation.
TEST_CASE("ACM SST omega advances when k is at its floor")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path, false);
    auto cfg = MakeRANSPhysicalCase(meshFile.path, ACM::TurbulenceModel::KOmegaSST,
                                    ACM::TimeIntegratorType::BDF2DualTimeLUSGS, 0.05, 0);
    cfg.turbulenceSettings.initialValue = {cfg.turbulenceSettings.minimumValue[0], 4};
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    auto boundary = std::make_shared<ACM::BoundaryHandler>(cfg.defaultBoundaryType, cfg.InitialState(), cfg.boundaryConditions);
    ACM::ACMTurbulenceTransport<2> transport(solver.GetMesh(), solver.GetReconstruction(),
                                             cfg.acmSettings, cfg.turbulenceSettings, boundary);
    for (const bool physical : {false, true})
    {
        transport.Initialize();
        if (physical)
        {
            transport.InitializePhysicalHistory();
            transport.AdvancePhysicalBDF2(solver.GetState(), {0.01}, ACM::GetBDF2Coefficients(0), 0.05, 0.05);
            CHECK(transport.GetPhysicalEquationDefectNorms()[0] > 0);
        }
        else
            transport.Advance(solver.GetState(), {0.01});
        CHECK(transport.GetState()[0](0) >= cfg.turbulenceSettings.minimumValue[0]);
        CHECK(transport.GetState()[0](0) < 1.01 * cfg.turbulenceSettings.minimumValue[0]);
        CHECK(transport.GetState()[0](1) < 3.999);
        CHECK(transport.GetState()[0](1) > 0);
    }
}

/// @test Local BE uses the newer k history while omega continues to solve its independent BDF2 root.
TEST_CASE("ACM SST positive physical marching solves a mixed BE and BDF2 step")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path, false);
    auto cfg = MakeRANSPhysicalCase(meshFile.path, ACM::TurbulenceModel::KOmegaSST,
                                    ACM::TimeIntegratorType::BDF2DualTimeLUSGS, 0.05, 0);
    cfg.turbulenceSettings.initialValue = {0.2, 4};
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    auto boundary = std::make_shared<ACM::BoundaryHandler>(cfg.defaultBoundaryType, cfg.InitialState(), cfg.boundaryConditions);
    ACM::ACMTurbulenceTransport<2> transport(solver.GetMesh(), solver.GetReconstruction(),
                                             cfg.acmSettings, cfg.turbulenceSettings, boundary);
    transport.Initialize();
    transport.InitializePhysicalHistory();
    transport.GetState()[0] << 1e-4, 3.9;
    transport.CommitPhysicalStep();
    constexpr DNDS::real dt = 0.05;
    const auto coefficients = ACM::GetBDF2Coefficients(1);
    transport.EvaluatePhysicalBDF2DefectNorm(solver.GetState(), coefficients, dt, 2 * dt);
    CHECK(transport.GetPhysicalBackwardEulerCounts()[0] == 1);
    CHECK(transport.GetPhysicalBackwardEulerCounts()[1] == 0);
    CHECK(transport.GetPhysicalEquationDefectNorms()[0] == doctest::Approx(0.09 * 1e-4 * 3.9));
    DNDS::real defect = 1;
    for (int iteration = 0; iteration < 500 && defect > 1e-11; iteration++)
        defect = transport.AdvancePhysicalBDF2(solver.GetState(), {0.01}, coefficients, dt, 2 * dt);
    CHECK(defect < 1e-11);
    constexpr DNDS::real historyOmega = 2 * 3.9 - 0.5 * 4;
    // This mesh has no wall: F1=0 and SST uses its far-field beta2=0.0828.
    const DNDS::real exactOmega = 2 * historyOmega /
                                 (1.5 + std::sqrt(1.5 * 1.5 + 4 * dt * 0.0828 * historyOmega));
    const DNDS::real exactK = 1e-4 / (1 + dt * 0.09 * exactOmega);
    CHECK(std::abs(transport.GetState()[0](1) - exactOmega) < 1e-10);
    CHECK(std::abs(transport.GetState()[0](0) - exactK) < 1e-12);
    CHECK(transport.GetPhysicalBackwardEulerCounts()[0] == 1);
}

/// @test Smooth SST decay retains second-order time accuracy when its positive histories need no fallback.
TEST_CASE("ACM SST BDF2 has second-order physical accuracy away from bounds")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path, false);
    constexpr DNDS::real endTime = 0.32;
    constexpr DNDS::real beta = 0.0828;
    constexpr DNDS::real betaStar = 0.09;
    const DNDS::real decay = 1 + beta * 4 * endTime;
    ACM::TurbulenceState exact;
    exact << 0.2 * std::pow(decay, -betaStar / beta), 4 / decay;
    for (const auto integrator : {ACM::TimeIntegratorType::BDF2DualTimeLUSGS,
                                 ACM::TimeIntegratorType::BDF2DualTimeGMRES})
    {
        std::array<DNDS::real, 3> errors{};
        for (std::size_t level = 0; level < errors.size(); level++)
        {
            const DNDS::real dt = 0.08 / static_cast<DNDS::real>(1 << level);
            auto cfg = MakeRANSPhysicalCase(meshFile.path, ACM::TurbulenceModel::KOmegaSST,
                                            integrator, dt, static_cast<int>(std::lround(endTime / dt)));
            cfg.turbulenceSettings.initialValue = {0.2, 4};
            ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
            solver.ReadMeshAndInitialize();
            solver.Run(); // Strict defaults require every physical equation to converge.
            errors[level] = (ACM::TurbulenceState((*solver.GetTurbulenceState())[0]) - exact).norm();
        }
        CHECK(errors[0] / errors[1] > 3.5);
        CHECK(errors[1] / errors[2] > 3.5);
        std::cout << "ACM RANS BDF2 refinement: model=KOmegaSST integrator=" << static_cast<int>(integrator)
                  << " errors=" << errors[0] << ',' << errors[1] << ',' << errors[2]
                  << " ratios=" << errors[0] / errors[1] << ',' << errors[1] / errors[2] << std::endl;
    }
}

/// @test Native BDF update logging measures the applied damped increment against the updated state.
TEST_CASE("ACM laminar physical marching reports actual relative updates")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path, false);
    auto cfg = MakeRANSPhysicalCase(meshFile.path, ACM::TurbulenceModel::Laminar,
                                    ACM::TimeIntegratorType::BDF2DualTimeLUSGS, 0.05, 1);
    cfg.defaultBoundaryType = ACM::BoundaryType::BCWall;
    cfg.initialState = {1, 0.2, 0, 0.3};
    cfg.acmSettings.dynamicViscosity = 0.1;
    cfg.timeMarchSettings.dualTimeConvergenceCriterion = "RelativeUpdate";
    cfg.timeMarchSettings.dualTimeUpdateMinimumIterations = 1;
    cfg.timeMarchSettings.dualTimeUpdateRelativeTolerance = 0;
    cfg.timeMarchSettings.dualTimeRequireConvergence = false;
    cfg.timeMarchSettings.maxImplicitIterations = 1;
    cfg.timeMarchSettings.implicitRelaxation = 0.2;
    cfg.outputSettings.interval = 0;
    cfg.Validate();
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    const ACM::State before = solver.GetState()[0];
    std::ostringstream captured;
    {
        struct Capture
        {
            std::streambuf *previous;
            explicit Capture(std::ostringstream &target) : previous(std::cout.rdbuf(target.rdbuf())) {}
            ~Capture() { std::cout.rdbuf(previous); }
        } capture(captured);
        solver.Run();
    }
    const ACM::State after = solver.GetState()[0];
    const DNDS::real applied = (after - before).norm();
    REQUIRE(applied > 1e-10);
    std::smatch match;
    const std::string output = captured.str();
    REQUIRE(std::regex_search(output, match, std::regex(
        "ACM dual-time inner=1 updateNorm=([^ ]+) stateNorm=([^ ]+) updateRatio=([^ ]+)")));
    CHECK(std::stod(match[1]) == doctest::Approx(applied).epsilon(1e-3));
    CHECK(std::stod(match[2]) == doctest::Approx(after.norm()).epsilon(1e-3));
    CHECK(std::stod(match[3]) == doctest::Approx(applied / after.norm()).epsilon(1e-3));
    CHECK(output.find("convergenceCriterion=RelativeUpdate") != std::string::npos);
}

/// @test Even a stationary physical solution performs the requested minimum before update-based exit.
TEST_CASE("ACM laminar physical marching enforces the update minimum iterations")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path, false);
    auto cfg = MakeRANSPhysicalCase(meshFile.path, ACM::TurbulenceModel::Laminar,
                                    ACM::TimeIntegratorType::BDF2DualTimeLUSGS, 0.05, 2);
    cfg.timeMarchSettings.dualTimeConvergenceCriterion = "RelativeUpdate";
    cfg.timeMarchSettings.dualTimeUpdateMinimumIterations = 2;
    cfg.timeMarchSettings.maxImplicitIterations = 5;
    cfg.outputSettings.interval = 0;
    cfg.Validate();
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    std::ostringstream captured;
    {
        struct Capture
        {
            std::streambuf *previous;
            explicit Capture(std::ostringstream &target) : previous(std::cout.rdbuf(target.rdbuf())) {}
            ~Capture() { std::cout.rdbuf(previous); }
        } capture(captured);
        solver.Run();
    }
    const std::string output = captured.str();
    const std::regex accepted("ACM physical step[^\\n]*inner=2 converged=1[^\\n]*convergenceCriterion=RelativeUpdate");
    CHECK(std::distance(std::sregex_iterator(output.begin(), output.end(), accepted), std::sregex_iterator()) == 2);
}

/// @test Unconverged production steps produce no accepted output; diagnostic advancement needs an explicit option.
TEST_CASE("ACM physical marching rejects an unconverged step by default")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path, false);
    struct TempOutput
    {
        std::filesystem::path path;
        ~TempOutput() { std::error_code error; std::filesystem::remove_all(path, error); }
    } output{meshFile.path.string() + ".outputs"};
    auto cfg = MakeRANSPhysicalCase(meshFile.path, ACM::TurbulenceModel::KOmegaSST,
                                    ACM::TimeIntegratorType::BDF2DualTimeLUSGS, 0.05, 2);
    cfg.timeMarchSettings.maxImplicitIterations = 1;
    cfg.outputSettings.interval = 1;
    cfg.outputSettings.writeInitial = false;
    cfg.outputSettings.directory = output.path.string();
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> rejected(mpi, cfg);
    rejected.ReadMeshAndInitialize();
    CHECK_THROWS_WITH_AS(rejected.Run(), doctest::Contains("histories were not committed"), std::runtime_error);
    CHECK_FALSE(std::filesystem::exists(output.path));
    cfg.timeMarchSettings.dualTimeRequireConvergence = false;
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> diagnostic(mpi, cfg);
    diagnostic.ReadMeshAndInitialize();
    CHECK_NOTHROW(diagnostic.Run());
    CHECK(std::filesystem::exists(output.path / "acm_00000001.vtkhdf"));
    CHECK(std::filesystem::exists(output.path / "acm_00000002.vtkhdf"));
}

/// @test Native SST physical marching recomputes the relative reference every step and stops above the absolute floor.
TEST_CASE("ACM SST physical marching uses each step's own relative RMS reference")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path, false);
    for (const auto integrator : {ACM::TimeIntegratorType::BDF2DualTimeLUSGS,
                                 ACM::TimeIntegratorType::BDF2DualTimeGMRES})
    {
        auto cfg = MakeRANSPhysicalCase(meshFile.path, ACM::TurbulenceModel::KOmegaSST, integrator, 0.05, 3);
        cfg.turbulenceSettings.initialValue = {0.2, 4};
        cfg.timeMarchSettings.implicitTolerance = 1e-8;
        cfg.timeMarchSettings.dualTimeConvergenceCriterion = "Relative";
        cfg.timeMarchSettings.dualTimeFlowRelativeTolerance = 1e-3;
        cfg.timeMarchSettings.dualTimeTurbulenceRelativeTolerance = 1e-3;
        ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        std::ostringstream captured;
        {
            struct Capture
            {
                std::streambuf *previous;
                explicit Capture(std::ostringstream &target) : previous(std::cout.rdbuf(target.rdbuf())) {}
                ~Capture() { std::cout.rdbuf(previous); }
            } capture(captured);
            solver.Run();
        }
        std::istringstream lines(captured.str());
        std::vector<DNDS::real> initialResiduals;
        for (std::string line; std::getline(lines, line);)
        {
            if (line.find("ACM physical step") == std::string::npos)
                continue;
            CAPTURE(line);
            CHECK(line.find("converged=1") != std::string::npos);
            const auto value = [&](const char *name)
            {
                std::smatch match;
                if (!std::regex_search(line, match, std::regex(std::string(name) + "=(\\S+)")))
                    throw std::runtime_error(std::string("Missing physical-step log field: ") + name);
                return std::stod(match[1].str());
            };
            const DNDS::real initial = value("turbInitial");
            const DNDS::real final = value("turbResidual");
            const DNDS::real target = value("turbTarget");
            const DNDS::real expected = std::max(1e-8, initial * 1e-3);
            CHECK(std::abs(target - expected) <= 0.005 * expected); // Printed values have three decimal digits.
            CHECK(final <= 1.005 * target);
            CHECK(value("turbRatio") <= 1.005e-3);
            CHECK(final > 1e-8); // The relative criterion must actually be able to stop the solve.
            CHECK(value("flowTarget") == 0); // Relative mode keeps a zero initial flow target at zero.
            CHECK(value("flowRatio") == 0);
            initialResiduals.push_back(initial);
        }
        REQUIRE(initialResiduals.size() == 3);
        CHECK(std::abs(initialResiduals[1] - initialResiduals[0]) > 0.01 * initialResiduals[0]);
        std::cout << "ACM SST grouped BDF2 references, integrator=" << static_cast<int>(integrator) << '\n'
                  << captured.str();
    }
}

/// @test A loose absolute threshold cannot stop a step that explicitly selects relative convergence.
TEST_CASE("ACM SST physical marching explicitly selects absolute or relative convergence")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path, false);
    for (const auto integrator : {ACM::TimeIntegratorType::BDF2DualTimeLUSGS,
                                 ACM::TimeIntegratorType::BDF2DualTimeGMRES})
        for (const std::string criterion : {"Absolute", "Relative"})
        {
            CAPTURE(static_cast<int>(integrator));
            CAPTURE(criterion);
            auto cfg = MakeRANSPhysicalCase(meshFile.path, ACM::TurbulenceModel::KOmegaSST, integrator, 0.05, 1);
            cfg.turbulenceSettings.initialValue = {0.2, 4};
            cfg.timeMarchSettings.implicitTolerance = 1;
            cfg.timeMarchSettings.dualTimeFlowRelativeTolerance = 1e-3;
            cfg.timeMarchSettings.dualTimeTurbulenceRelativeTolerance = 1e-3;
            cfg.timeMarchSettings.dualTimeConvergenceCriterion = criterion;
            ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
            solver.ReadMeshAndInitialize();
            std::ostringstream captured;
            {
                struct Capture
                {
                    std::streambuf *previous;
                    explicit Capture(std::ostringstream &target) : previous(std::cout.rdbuf(target.rdbuf())) {}
                    ~Capture() { std::cout.rdbuf(previous); }
                } capture(captured);
                CHECK_NOTHROW(solver.Run());
            }
            const std::string output = captured.str();
            CAPTURE(output);
            CHECK(output.find("converged=1") != std::string::npos);
            CHECK(output.find("convergenceCriterion=" + criterion) != std::string::npos);
            std::smatch match;
            REQUIRE(std::regex_search(output, match, std::regex("turbRatio=(\\S+)")));
            const DNDS::real ratio = std::stod(match[1].str());
            REQUIRE(std::regex_search(output, match, std::regex("inner=(\\S+)")));
            const int iterations = std::stoi(match[1].str());
            if (criterion == "Absolute")
            {
                CHECK(iterations == 0);
                CHECK(ratio == doctest::Approx(1));
            }
            else
            {
                CHECK(iterations > 0);
                CHECK(ratio <= 1.005e-3); // The absolute threshold is met initially, but is not selected.
            }
        }
}

/// @test Recover the exact nonlinear BE/BDF2 decay roots, including both completed history levels.
TEST_CASE("ACM Wilcox physical marching follows homogeneous k-omega decay")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path, false);

    constexpr DNDS::real dt = 0.05;
    constexpr DNDS::real betaStar = 0.09;
    constexpr DNDS::real beta = 0.0708;
    for (const auto integrator : {ACM::TimeIntegratorType::BDF2DualTimeLUSGS,
                                 ACM::TimeIntegratorType::BDF2DualTimeGMRES})
        for (const bool secondOrder : {false, true})
        {
            CAPTURE(static_cast<int>(integrator));
            CAPTURE(secondOrder);
            ACM::KernelConfiguration cfg;
            cfg.meshSettings.meshFile = meshFile.path.string();
            cfg.defaultBoundaryType = ACM::BoundaryType::BCSym;
            cfg.initialState = {0, 0, 0, 0.3};
            std::copy_n(cfg.initialState.begin(), cfg.acmSettings.farFieldValue.size(), cfg.acmSettings.farFieldValue.begin());
            cfg.acmSettings.enableViscousFlux = true;
            cfg.acmSettings.dynamicViscosity = 0.001;
            cfg.reconstructionSettings.type = ACM::ReconstructionType::FirstOrder;
            if (secondOrder)
            {
                cfg.reconstructionSettings.type = ACM::ReconstructionType::Variational;
                cfg.reconstructionSettings.limiterType = ACM::LimiterType::CWBAP;
                cfg.vfvSettings.maxOrder = 0;
            }
            cfg.turbulenceSettings.model = ACM::TurbulenceModel::KOmegaWilcox;
            cfg.turbulenceSettings.initialValue = {0.2, 4.0};
            cfg.turbulenceSettings.secondOrderReconstruction = secondOrder;
            cfg.turbulenceSettings.transportTimeScale = 1;
            cfg.timeMarchSettings.integrator = integrator;
            cfg.timeMarchSettings.nSteps = 3;
            cfg.timeMarchSettings.pseudoTimeStep = 0.01;
            cfg.timeMarchSettings.physicalTimeStep = dt;
            cfg.timeMarchSettings.maxImplicitIterations = 500;
            cfg.timeMarchSettings.implicitTolerance = 1e-11;
            cfg.Validate();

            // With uniform stationary flow and symmetry faces, spatial flux divergences vanish.
            // Each physical level therefore solves the two nonlinear decay equations.
            ACM::TurbulenceState previous, older;
            previous << 0.2, 4.0;
            older = previous;
            for (int step = 0; step < 3; step++)
            {
                const DNDS::real a0 = step == 0 ? 1.0 : 1.5;
                const DNDS::real a1 = step == 0 ? -1.0 : -2.0;
                const DNDS::real a2 = step == 0 ? 0.0 : 0.5;
                const DNDS::real omegaRhs = -a1 * previous(1) - a2 * older(1);
                ACM::TurbulenceState next;
                next(1) = 2 * omegaRhs /
                          (a0 + std::sqrt(a0 * a0 + 4 * dt * beta * omegaRhs));
                next(0) = (-a1 * previous(0) - a2 * older(0)) /
                          (a0 + dt * betaStar * next(1));
                older = previous;
                previous = next;
            }

            ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
            solver.ReadMeshAndInitialize();
            solver.Run();
            REQUIRE(solver.GetTurbulenceState() != nullptr);
            const ACM::TurbulenceState actual = (*solver.GetTurbulenceState())[0];
            CHECK((actual - previous).norm() < 1e-10);
            CHECK((solver.GetState()[0] - cfg.InitialState()).norm() < 1e-10);
        }
}

/// @test SA's inactive slot remains zero while both new models reach the independently derived time roots.
TEST_CASE("ACM SA and realizable k-epsilon physical marching follows the discrete equations")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path, false);
    for (const auto model : {ACM::TurbulenceModel::SpalartAllmaras, ACM::TurbulenceModel::RealizableKEpsilon})
        for (const auto integrator : {ACM::TimeIntegratorType::BDF2DualTimeLUSGS,
                                     ACM::TimeIntegratorType::BDF2DualTimeGMRES})
            for (const bool secondOrder : {false, true})
            {
                CAPTURE(static_cast<int>(model));
                CAPTURE(static_cast<int>(integrator));
                CAPTURE(secondOrder);
                const auto cfg = MakeRANSPhysicalCase(meshFile.path, model, integrator, 0.05, 3, secondOrder);
                ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
                solver.ReadMeshAndInitialize();
                solver.Run();
                REQUIRE(solver.GetTurbulenceState() != nullptr);
                const ACM::TurbulenceState actual = (*solver.GetTurbulenceState())[0];
                const auto expected = ExactRANSDiscreteDecay(model, 0.05, 3);
                CHECK((actual - expected).norm() < 1e-10);
                CHECK((solver.GetState()[0] - cfg.InitialState()).norm() < 1e-10);
                CHECK((actual.head(ACM::TurbulenceVariableCount(model)).array() > 0).all());
                if (model == ACM::TurbulenceModel::SpalartAllmaras)
                    CHECK(actual(1) == 0);
            }
}

/// @test Halving physical dt recovers second-order convergence against continuous decay solutions.
TEST_CASE("ACM SA and realizable k-epsilon BDF2 has second-order physical accuracy")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path, false);
    constexpr DNDS::real endTime = 0.32;
    for (const auto model : {ACM::TurbulenceModel::SpalartAllmaras, ACM::TurbulenceModel::RealizableKEpsilon})
        for (const auto integrator : {ACM::TimeIntegratorType::BDF2DualTimeLUSGS,
                                     ACM::TimeIntegratorType::BDF2DualTimeGMRES})
        {
            CAPTURE(static_cast<int>(model));
            CAPTURE(static_cast<int>(integrator));
            ACM::TurbulenceState exact = ACM::TurbulenceState::Zero();
            if (model == ACM::TurbulenceModel::SpalartAllmaras)
            {
                constexpr DNDS::real rate = 1 + 2 * (0.001 + 0.1) / (2.0 / 3.0);
                exact(0) = 0.1 + 0.1 * std::exp(-rate * endTime);
            }
            else
            {
                const DNDS::real factor = 1 + (1.92 - 1) * (0.08 / 0.4) * endTime;
                exact(0) = 0.4 * std::pow(factor, -1 / (1.92 - 1));
                exact(1) = 0.08 * std::pow(factor, -1.92 / (1.92 - 1));
            }
            std::array<DNDS::real, 3> errors;
            for (int level = 0; level < 3; level++)
            {
                const int steps = 4 << level;
                const auto cfg = MakeRANSPhysicalCase(meshFile.path, model, integrator, endTime / steps, steps);
                ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
                solver.ReadMeshAndInitialize();
                solver.Run();
                errors[level] = (ACM::TurbulenceState((*solver.GetTurbulenceState())[0]) - exact).norm();
                CHECK(errors[level] > 1e-12);
            }
            CHECK(errors[0] / errors[1] > 3);
            CHECK(errors[1] / errors[2] > 3);
            std::cout << "ACM RANS BDF2 refinement: model=" << ACM::TurbulenceModelName(model)
                      << " integrator=" << static_cast<int>(integrator)
                      << " errors=" << errors[0] << "," << errors[1] << "," << errors[2]
                      << " ratios=" << errors[0] / errors[1] << "," << errors[1] / errors[2] << std::endl;
        }
}

namespace
{
    template <ACM::ACMModel model>
    ACM::KernelConfiguration MakeUnifiedCase(const std::filesystem::path &mesh)
    {
        ACM::KernelConfiguration cfg;
        cfg.solver = ACM::ModelSelection<model>();
        cfg.meshSettings.meshFile = mesh.string();
        cfg.meshSettings.periodicTranslation1 = {1, 0, 0};
        cfg.meshSettings.periodicTranslation2 = {0, 1, 0};
        cfg.acmSettings.rho0 = 3;
        cfg.acmSettings.dynamicViscosity = 0.001;
        cfg.acmSettings.enableViscousFlux = true;
        cfg.initialState = {0, 0, 0, 0.3, 1.2};
        cfg.turbulenceSettings.model = ACM::ModelTraits<model>::nExtraVars == 1
            ? ACM::TurbulenceModel::SpalartAllmaras : ACM::TurbulenceModel::KOmegaWilcox;
        if constexpr (ACM::ModelTraits<model>::nExtraVars == 2)
            cfg.initialState.push_back(0.24);
        cfg.turbulenceSettings.transportSubsteps = 1;
        cfg.turbulenceSettings.transportTimeScale = 1;
        cfg.turbulenceSettings.enableSourceTerms = false;
        cfg.reconstructionSettings.type = ACM::ReconstructionType::FirstOrder;
        cfg.reconstructionSettings.enableLimiter = false;
        cfg.timeMarchSettings.pseudoTimeStep = 0.02;
        cfg.timeMarchSettings.physicalTimeStep = 0.01;
        cfg.timeMarchSettings.maxImplicitIterations = 500;
        cfg.timeMarchSettings.implicitTolerance = 1e-12;
        cfg.timeMarchSettings.linearAbsoluteTolerance = 1e-14;
        cfg.timeMarchSettings.nSteps = 3;
        return cfg;
    }

    /** @brief Multi-cell periodic box with distinct neighbour centers for viscous closures. */
    void WritePeriodicQuadGrid(const std::filesystem::path &path, int n = 2, bool triangles = false, bool warped = false)
    {
        int file = -1, base = 0, zone = 0, coordinate = 0, section = 0, boundary = 0;
        CheckCGNS(cg_open(path.c_str(), CG_MODE_WRITE, &file));
        try
        {
            CheckCGNS(cg_base_write(file, "Base", 2, 2, &base));
            const int nCell = (triangles ? 2 : 1) * n * n;
            cgsize_t size[3] = {static_cast<cgsize_t>((n + 1) * (n + 1)), static_cast<cgsize_t>(nCell), 0};
            CheckCGNS(cg_zone_write(file, base, "Zone", size, Unstructured, &zone));
            std::vector<double> x((n + 1) * (n + 1)), y(x.size());
            for (int j = 0; j <= n; j++)
                for (int i = 0; i <= n; i++)
                {
                    const double perturbation = warped ? 0.15 / n * std::sin(2 * std::acos(-1.0) * i / n) * std::sin(2 * std::acos(-1.0) * j / n) : 0;
                    x[(n + 1) * j + i] = double(i) / n + perturbation;
                    y[(n + 1) * j + i] = double(j) / n - 0.7 * perturbation;
                }
            CheckCGNS(cg_coord_write(file, base, zone, RealDouble, "CoordinateX", x.data(), &coordinate));
            CheckCGNS(cg_coord_write(file, base, zone, RealDouble, "CoordinateY", y.data(), &coordinate));
            std::vector<cgsize_t> cells((triangles ? 3 : 4) * nCell);
            int offset = 0;
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                {
                    const cgsize_t first = (n + 1) * j + i + 1;
                    cells[offset++] = first;
                    cells[offset++] = first + 1;
                    cells[offset++] = first + n + 2;
                    if (triangles)
                    {
                        cells[offset++] = first;
                        cells[offset++] = first + n + 2;
                    }
                    cells[offset++] = first + n + 1;
                }
            CheckCGNS(cg_section_write(file, base, zone, "Cells", triangles ? TRI_3 : QUAD_4, 1, nCell, 0, cells.data(), &section));
            const std::array<const char *, 4> names = {"PERIODIC_1", "PERIODIC_1_DONOR", "PERIODIC_2", "PERIODIC_2_DONOR"};
            std::array<std::vector<cgsize_t>, 4> edges;
            for (int k = 0; k < n; ++k)
            {
                edges[0].insert(edges[0].end(), {cgsize_t((n + 1) * k + 1), cgsize_t((n + 1) * (k + 1) + 1)});
                edges[1].insert(edges[1].end(), {cgsize_t((n + 1) * k + n + 1), cgsize_t((n + 1) * (k + 1) + n + 1)});
                edges[2].insert(edges[2].end(), {cgsize_t(k + 1), cgsize_t(k + 2)});
                edges[3].insert(edges[3].end(), {cgsize_t(n * (n + 1) + k + 1), cgsize_t(n * (n + 1) + k + 2)});
            }
            for (int i = 0; i < 4; i++)
            {
                const cgsize_t start = nCell + 1 + n * i;
                CheckCGNS(cg_section_write(file, base, zone, names[i], BAR_2, start, start + n - 1, 0, edges[i].data(), &section));
                cgsize_t range[2] = {start, start + n - 1};
                CheckCGNS(cg_boco_write(file, base, zone, names[i], BCTypeNull, PointRange, 2, range, &boundary));
                CheckCGNS(cg_boco_gridlocation_write(file, base, zone, boundary, EdgeCenter));
            }
            CheckCGNS(cg_close(file));
        }
        catch (...)
        {
            cg_close(file);
            throw;
        }
    }
}

using UnifiedSA = std::integral_constant<ACM::ACMModel, ACM::ACMModel::ConstantDensitySA2D>;
using Unified2EQ = std::integral_constant<ACM::ACMModel, ACM::ACMModel::ConstantDensity2EQ2D>;

TEST_CASE("ACM SST-SAS integrates the omega source and its implicit diagonal on a periodic mesh")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path, 4);
    using Solver = ACM::ACMSolver<Unified2EQ::value>;
    using Block = Eigen::Matrix<DNDS::real, 2, 2>;
    std::vector<Solver::TState> referenceRhs;
    std::vector<ACM::TurbulenceState> referenceSource;
    std::vector<Block> referenceDiagonal;
    for (const auto model : {ACM::TurbulenceModel::KOmegaSST, ACM::TurbulenceModel::KOmegaSSTSAS})
    {
        auto cfg = MakeUnifiedCase<Unified2EQ::value>(mesh.path);
        cfg.turbulenceSettings.model = model;
        cfg.turbulenceSettings.enableSourceTerms = true;
        cfg.turbulenceSettings.variationalReconstruction = true;
        cfg.vfvSettings.maxOrder = 2;
        cfg.vfvSettings.intOrder = 4;
        Solver solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        const auto metrics = solver.GetReconstruction();
        Solver::TFlowDof flow;
        Solver::TDof rhs;
        metrics->BuildUDof(flow, 4);
        metrics->BuildUDof(rhs, 6);
        const auto transport = solver.GetTurbulenceTransport();
        const auto cells = solver.GetMesh()->NumCell();
        for (DNDS::index cell = 0; cell < cells; ++cell)
        {
            const auto x = metrics->GetCellBary(cell);
            solver.GetState()[cell](0) = 0.1 * std::sin(2 * std::acos(-1.0) * x(1));
            flow[cell] = solver.GetState()[cell].template head<4>();
        }
        solver.EvaluateRHS(rhs);
        std::vector<Block> diagonal;
        transport->EvaluateImplicitDiagonal(flow, diagonal, 0);
        if (model == ACM::TurbulenceModel::KOmegaSST)
        {
            referenceDiagonal = diagonal;
            for (DNDS::index cell = 0; cell < cells; ++cell)
            {
                referenceRhs.emplace_back(rhs[cell]);
                referenceSource.emplace_back(transport->CellSource(cell));
            }
            continue;
        }

        DNDS::real sourceIncrease = 0;
        for (DNDS::index cell = 0; cell < cells; ++cell)
        {
            const Solver::TState rhsChange = rhs[cell] - referenceRhs[cell];
            const ACM::TurbulenceState sourceChange = transport->CellSource(cell) - referenceSource[cell];
            CHECK(rhsChange.template head<5>().norm() < 1e-12);
            CHECK(rhsChange(5) == doctest::Approx(cfg.acmSettings.rho0 * sourceChange(1)));
            CHECK(sourceChange(1) >= 0);
            sourceIncrease += sourceChange(1);

            const Block diagonalChange = diagonal[cell] - referenceDiagonal[cell];
            const DNDS::real omega = cfg.initialState[5] / cfg.acmSettings.rho0;
            CHECK(diagonalChange(1, 1) == doctest::Approx(2 * sourceChange(1) / omega));
            CHECK(diagonalChange.row(0).norm() < 1e-11);
            CHECK(std::abs(diagonalChange(1, 0)) < 1e-11);
        }
        CHECK(sourceIncrease > 1);
    }
}

TEST_CASE("ACM turbulence face Jacobian matches periodic transport finite differences")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    for (auto closure : {ACM::TurbulenceModel::KOmegaWilcox, ACM::TurbulenceModel::KOmegaSST})
    {
        auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
        cfg.solver = ACM::ModelSelection<ACM::ACMModel::ConstantDensity2D>();
        cfg.initialState = {0.3, -0.1, 0, 0.3};
        cfg.turbulenceSettings.model = closure;
        cfg.turbulenceSettings.initialValue = {0.4, 0.08};
        cfg.turbulenceSettings.secondOrderReconstruction = false;
        // Freeze molecular diffusion without a state-dependent eddy-viscosity derivative.
        cfg.turbulenceSettings.maximumEddyViscosityRatio = 1e-30;
        ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        auto transport = solver.GetTurbulenceTransport();
        auto &state = transport->GetState();
        auto &flow = solver.GetState();
        auto metrics = solver.GetReconstruction();
        const auto grid = solver.GetMesh();
        using Field = ACM::ACMTurbulenceTransport<2>::TTurbulenceDof;
        Field minusRhs, plusRhs;
        metrics->BuildUDof(minusRhs, 2);
        metrics->BuildUDof(plusRhs, 2);
        std::vector<ACM::TurbulenceState> base(grid->NumCell()), direction(grid->NumCell());
        for (DNDS::index cell = 0; cell < grid->NumCell(); cell++)
        {
            base[cell] << 0.4 + 0.02 * cell, 0.08 + 0.01 * cell;
            direction[cell] << std::sin(1.0 + cell), std::cos(0.3 + cell);
            state[cell] = base[cell];
        }
        std::vector<Eigen::Matrix<DNDS::real, 2, 2>> diagonal;
        ACM::ACMTurbulenceTransport<2>::FaceJacobianField faces;
        transport->EvaluateImplicitDiagonal(flow, diagonal, 0, &faces);
        std::vector<ACM::TurbulenceState> product(grid->NumCell());
        for (DNDS::index cell = 0; cell < grid->NumCell(); cell++)
            product[cell] = diagonal[cell] * direction[cell];
        for (DNDS::index face = 0; face < grid->NumFaceProc(); face++)
        {
            const auto cells = grid->face2cell[face];
            REQUIRE(cells[1] != UnInitIndex);
            if (cells[0] == cells[1])
                continue;
            product[cells[0]] += faces[face][1] * direction[cells[1]] / metrics->GetCellVol(cells[0]);
            product[cells[1]] -= faces[face][0] * direction[cells[0]] / metrics->GetCellVol(cells[1]);
        }
        constexpr DNDS::real epsilon = 1e-6;
        for (DNDS::index cell = 0; cell < grid->NumCell(); cell++)
            state[cell] = base[cell] + epsilon * direction[cell];
        transport->EvaluateRHS(plusRhs, flow);
        for (DNDS::index cell = 0; cell < grid->NumCell(); cell++)
            state[cell] = base[cell] - epsilon * direction[cell];
        transport->EvaluateRHS(minusRhs, flow);
        for (DNDS::index cell = 0; cell < grid->NumCell(); cell++)
        {
            const ACM::TurbulenceState finiteDifference = -(plusRhs[cell] - minusRhs[cell]) / (2 * epsilon);
            CHECK((product[cell] - finiteDifference).norm() < 1e-8);
        }
    }
}

TEST_CASE_TEMPLATE("ACM unified RANS stores density-weighted variables and shares every integration stage",
                   TModel, UnifiedSA, Unified2EQ)
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    for (const auto integrator : {ACM::TimeIntegratorType::ExplicitSSPRK3,
                                 ACM::TimeIntegratorType::ImplicitEulerBlockJacobi,
                                 ACM::TimeIntegratorType::ImplicitEulerLUSGS,
                                 ACM::TimeIntegratorType::ImplicitEulerGMRES,
                                 ACM::TimeIntegratorType::BDF2DualTimeLUSGS,
                                 ACM::TimeIntegratorType::BDF2DualTimeGMRES})
    {
        auto cfg = MakeUnifiedCase<TModel::value>(mesh.path);
        cfg.timeMarchSettings.integrator = integrator;
        ACM::ACMSolver<TModel::value> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        CHECK(solver.GetState()[0].size() == ACM::ModelTraits<TModel::value>::nVarsFixed);
        typename ACM::ACMSolver<TModel::value>::TState initial = solver.GetState()[0];
        CHECK(initial(4) == doctest::Approx(1.2));
        const auto primitive = ACM::ModelConservativeToPrimitive<TModel::value>(initial, cfg.acmSettings.rho0);
        CHECK(primitive(4) == doctest::Approx(0.4));
        CHECK((ACM::ModelPrimitiveToConservative<TModel::value>(primitive, cfg.acmSettings.rho0) - initial).norm() < 1e-14);
        solver.Run();
        CHECK((solver.GetState()[0] - initial).norm() < 1e-10);
        if (ACM::IsBDF2DualTimeIntegrator(integrator))
        {
            CHECK(solver.CompletedPhysicalSteps() == 3);
            CHECK(solver.SimulationTime() == doctest::Approx(0.03));
        }
    }
}

TEST_CASE("ACM turbulence advection cross Jacobian follows shared velocity pressure flux")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
    cfg.solver = ACM::ModelSelection<ACM::ACMModel::ConstantDensity2D>();
    cfg.initialState = {0.3, -0.1, 0, 0.3};
    cfg.turbulenceSettings.initialValue = {0.4, 0.08};
    cfg.turbulenceSettings.secondOrderReconstruction = false;
    cfg.turbulenceSettings.maximumEddyViscosityRatio = 1e-30;
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    auto &flow = solver.GetState();
    const auto evaluator = solver.GetEvaluator();
    const auto transport = solver.GetTurbulenceTransport();
    transport->SetFaceVolumeFlux([&](DNDS::index face, int point, DNDS::real time)
    {
        return evaluator->FaceVolumeFlux(flow, face, point, time);
    });
    using Field = ACM::ACMTurbulenceTransport<2>::TTurbulenceDof;
    Field plus, minus;
    solver.GetReconstruction()->BuildUDof(plus, 2);
    solver.GetReconstruction()->BuildUDof(minus, 2);
    const auto grid = solver.GetMesh();
    std::vector<ACM::State> base(grid->NumCell()), direction(grid->NumCell());
    for (DNDS::index cell = 0; cell < grid->NumCell(); cell++)
    {
        base[cell] = flow[cell];
        direction[cell] << std::sin(1.0 + cell), std::cos(0.2 + cell), 0, std::sin(0.4 + 2 * cell);
    }
    ACM::MatrixField diagonal;
    ACM::ACMEvaluator<2>::FaceJacobianField faces;
    evaluator->AssembleImplicitLinearization(flow, ACM::ScalarField(grid->NumCell(), 0.02), diagonal, faces);
    std::vector<ACM::TurbulenceState> product(grid->NumCell(), ACM::TurbulenceState::Zero());
    for (DNDS::index face = 0; face < grid->NumFaceProc(); face++)
    {
        const auto cells = grid->face2cell[face];
        const DNDS::real flux = (faces[face].left.row(3) * direction[cells[0]] +
                                 faces[face].right.row(3) * direction[cells[1]])(0);
        const ACM::TurbulenceState crossFlux = transport->FrozenFaceAdvectionValue(flow, face) * flux;
        product[cells[0]] += crossFlux / solver.GetReconstruction()->GetCellVol(cells[0]);
        product[cells[1]] -= crossFlux / solver.GetReconstruction()->GetCellVol(cells[1]);
    }
    constexpr DNDS::real epsilon = 1e-6;
    for (DNDS::index cell = 0; cell < grid->NumCell(); cell++)
        flow[cell] = base[cell] + epsilon * direction[cell];
    transport->EvaluateRHS(plus, flow);
    for (DNDS::index cell = 0; cell < grid->NumCell(); cell++)
        flow[cell] = base[cell] - epsilon * direction[cell];
    transport->EvaluateRHS(minus, flow);
    for (DNDS::index cell = 0; cell < grid->NumCell(); cell++)
        CHECK((product[cell] + (plus[cell] - minus[cell]) / (2 * epsilon)).norm() < 1e-7);
}

TEST_CASE("ACM full Newton Krylov preserves the converged reconstructed physical solution")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    using Solver = ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D>;
    std::vector<Solver::TState> reference;
    for (bool newton : {false, true})
    {
        auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
        cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::BDF2DualTimeGMRES;
        cfg.timeMarchSettings.newtonKrylov = newton;
        cfg.timeMarchSettings.nSteps = 1;
        cfg.timeMarchSettings.implicitTolerance = 1e-9;
        cfg.reconstructionSettings.type = ACM::ReconstructionType::Variational;
        cfg.reconstructionSettings.variationalTolerance = 1e-12;
        cfg.reconstructionSettings.variationalMaxIterations = 2000;
        cfg.reconstructionSettings.enableLimiter = false;
        cfg.vfvSettings.maxOrder = 2;
        cfg.turbulenceSettings.maximumEddyViscosityRatio = 1e-30;
        Solver solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); ++cell)
        {
            solver.GetState()[cell](0) += 0.01 * std::sin(1.0 + cell);
            solver.GetState()[cell](1) += 0.01 * std::cos(1.0 + cell);
            solver.GetState()[cell](4) += 0.01 * cell;
        }
        solver.Run();
        REQUIRE(solver.CompletedPhysicalSteps() == 1);
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); ++cell)
            if (!newton) reference.push_back(solver.GetState()[cell]);
            else CHECK((solver.GetState()[cell] - reference[cell]).norm() < 1e-8);
    }
}

TEST_CASE("ACM SA-IDDES advances the real wall-bounded transport operator")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WriteSingleCellQuad(mesh.path, false);
    auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensitySA2D>(mesh.path);
    cfg.turbulenceSettings.model = ACM::TurbulenceModel::SpalartAllmarasIDDES;
    cfg.turbulenceSettings.enableSourceTerms = true;
    cfg.initialState[4] = cfg.acmSettings.rho0 * 1e-3;
    cfg.defaultBoundaryType = ACM::BoundaryType::BCOut;
    ACM::BoundaryCondition wall;
    wall.name = "Bottom";
    wall.type = ACM::BoundaryType::BCWall;
    cfg.boundaryConditions = {wall};
    cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::ExplicitSSPRK3;
    cfg.timeMarchSettings.nSteps = 1;
    cfg.timeMarchSettings.pseudoTimeStep = 1e-4;
    ACM::ACMSolver<ACM::ACMModel::ConstantDensitySA2D> solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    solver.Run();
    CHECK(solver.GetState()[0].allFinite());
    CHECK(solver.GetState()[0](4) > 0);
    CHECK(solver.GetState()[0](4) < cfg.initialState[4]);
}

TEST_CASE("ACM laminar Newton Krylov ESDIRK3 has third-order shear decay")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path, 4);
    using Solver = ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D>;
    const auto error = [&](int count)
    {
        auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
        cfg.solver = ACM::ModelSelection<ACM::ACMModel::ConstantDensity2D>();
        cfg.initialState.resize(4);
        cfg.turbulenceSettings.model = ACM::TurbulenceModel::Laminar;
        cfg.acmSettings.dynamicViscosity = cfg.acmSettings.rho0 * 0.1;
        cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::ESDIRK3DualTimeGMRES;
        cfg.timeMarchSettings.newtonKrylov = true;
        cfg.timeMarchSettings.nSteps = count;
        cfg.timeMarchSettings.physicalTimeStep = 0.2 / count;
        cfg.timeMarchSettings.pseudoTimeStep = 1;
        cfg.timeMarchSettings.maxImplicitIterations = 100;
        cfg.timeMarchSettings.implicitTolerance = 1e-11;
        cfg.timeMarchSettings.linearAbsoluteTolerance = 1e-13;
        cfg.timeMarchSettings.linearRelativeTolerance = 1e-8;
        cfg.reconstructionSettings.type = ACM::ReconstructionType::Variational;
        cfg.reconstructionSettings.variationalTolerance = 1e-12;
        cfg.reconstructionSettings.variationalMaxIterations = 2000;
        cfg.reconstructionSettings.enableLimiter = false;
        cfg.vfvSettings.maxOrder = 3;
        cfg.vfvSettings.intOrder = 6;
        Solver solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        const auto vfv = solver.GetReconstruction();
        std::vector<DNDS::real> initial(solver.GetMesh()->NumCell());
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); ++cell)
        {
            const auto quad = vfv->GetCellQuad(cell);
            for (int point = 0; point < quad.GetNumPoints(); ++point)
                initial[cell] += quad.GetWeight(point) * vfv->GetCellJacobiDet(cell, point) / vfv->GetCellVol(cell) *
                    std::sin(2 * std::acos(-1.0) * vfv->GetCellQuadraturePPhys(cell, point)(1));
            solver.GetState()[cell](0) = initial[cell];
        }
        Solver::TDof rhs;
        vfv->BuildUDof(rhs, 4);
        solver.GetEvaluator()->EvaluateRHS(rhs, solver.GetState());
        DNDS::real numerator = 0, denominator = 0;
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); ++cell)
        {
            numerator += initial[cell] * rhs[cell](0);
            denominator += initial[cell] * initial[cell];
        }
        const DNDS::real lambda = numerator / denominator;
        REQUIRE(lambda < 0);
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); ++cell)
            CHECK(std::abs(rhs[cell](0) - lambda * initial[cell]) < 1e-9);
        solver.Run();
        DNDS::real result = 0;
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); ++cell)
            result += vfv->GetCellVol(cell) * std::pow(solver.GetState()[cell](0) - initial[cell] * std::exp(lambda * 0.2), 2);
        std::cout << "ACM shear temporal steps=" << count << " L2=" << std::sqrt(result) << std::endl;
        return std::sqrt(result);
    };
    const auto coarse = error(4), medium = error(8), fine = error(16);
    CHECK(coarse / medium > 7);
    CHECK(medium / fine > 7);
    CHECK(medium / fine < 9);
}

TEST_CASE("ACM high-order RANS advection diffusion converges on warped triangular grids")
{
    MPIInfo mpi;
    mpi.setWorld();
    using Solver = ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D>;
    const DNDS::real wave = 2 * std::acos(-1.0);
    std::vector<DNDS::real> errors;
    std::vector<DNDS::real> flowErrors;
    for (int n : {6, 12, 24})
    {
        TempMeshFile mesh;
        WritePeriodicQuadGrid(mesh.path, n, true, true);
        auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
        cfg.initialState[0] = 0.7;
        cfg.initialState[1] = 0.3;
        cfg.vfvSettings.maxOrder = 3;
        cfg.vfvSettings.intOrder = 6;
        cfg.reconstructionSettings.type = ACM::ReconstructionType::Variational;
        cfg.reconstructionSettings.enableLimiter = false;
        cfg.reconstructionSettings.variationalTolerance = 1e-11;
        cfg.reconstructionSettings.variationalMaxIterations = 2000;
        cfg.turbulenceSettings.model = ACM::TurbulenceModel::KOmegaSST;
        cfg.turbulenceSettings.variationalReconstruction = true;
        cfg.turbulenceSettings.variationalTolerance = 1e-11;
        cfg.turbulenceSettings.variationalMaxIterations = 2000;
        cfg.turbulenceSettings.enableSourceTerms = false;
        cfg.turbulenceSettings.includeIsotropicStress = false;
        cfg.turbulenceSettings.boundedSSTDiffusion = false;
        cfg.turbulenceSettings.maximumEddyViscosityRatio = 1e-30;
        cfg.acmSettings.dynamicViscosity = cfg.acmSettings.rho0 * 0.01;
        Solver solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        auto vfv = solver.GetReconstruction();
        std::vector<DNDS::real> exact(solver.GetMesh()->NumCell());
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); ++cell)
        {
            const auto quad = vfv->GetCellQuad(cell);
            DNDS::real average = 0;
            for (int point = 0; point < quad.GetNumPoints(); ++point)
            {
                const auto x = vfv->GetCellQuadraturePPhys(cell, point);
                const DNDS::real f = std::sin(wave * x(0)) * std::cos(wave * x(1));
                const DNDS::real fx = wave * std::cos(wave * x(0)) * std::cos(wave * x(1));
                const DNDS::real fy = -wave * std::sin(wave * x(0)) * std::sin(wave * x(1));
                const DNDS::real weight = quad.GetWeight(point) * vfv->GetCellJacobiDet(cell, point) / vfv->GetCellVol(cell);
                average += weight * f;
                exact[cell] += weight * 0.1 * (-0.7 * fx - 0.3 * fy - 0.02 * wave * wave * f);
            }
            solver.GetState()[cell](4) = cfg.acmSettings.rho0 * (1 + 0.1 * average);
            solver.GetState()[cell](5) = cfg.acmSettings.rho0 * (2 + 0.2 * average);
        }
        Solver::TDof rhs;
        vfv->BuildUDof(rhs, 6);
        solver.EvaluateRHS(rhs);
        DNDS::real error = 0, conservation = 0;
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); ++cell)
        {
            const DNDS::real residual = rhs[cell](4) / cfg.acmSettings.rho0;
            error += vfv->GetCellVol(cell) * std::pow(residual - exact[cell], 2);
            conservation += vfv->GetCellVol(cell) * residual;
            CHECK(rhs[cell](5) / cfg.acmSettings.rho0 == doctest::Approx(2 * residual).epsilon(1e-8).scale(1));
        }
        CHECK(std::abs(conservation) < 1e-10);
        errors.push_back(std::sqrt(error));
        std::cout << "ACM RANS spatial n=" << n << " L2=" << errors.back() << std::endl;
        // A divergence-free shear with spatially varying pressure independently
        // checks momentum convection, pressure, viscous flux and continuity.
        std::vector<ACM::State> flowExact(solver.GetMesh()->NumCell(), ACM::State::Zero());
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); ++cell)
        {
            ACM::State mean = ACM::State::Zero();
            const auto quad = vfv->GetCellQuad(cell);
            for (int point = 0; point < quad.GetNumPoints(); ++point)
            {
                const auto x = vfv->GetCellQuadraturePPhys(cell, point);
                const DNDS::real sx = std::sin(wave * x(0)), cx = std::cos(wave * x(0));
                const DNDS::real sy = std::sin(wave * x(1)), cy = std::cos(wave * x(1));
                const DNDS::real weight = quad.GetWeight(point) * vfv->GetCellJacobiDet(cell, point) / vfv->GetCellVol(cell);
                mean += weight * ACM::State(0.7 + 0.1 * sy, 0.3, 0, 0.2 * cx * cy);
                flowExact[cell](0) += weight * (-0.03 * wave * cy + 0.2 * wave / cfg.acmSettings.rho0 * sx * cy -
                                               0.001 * wave * wave * sy);
                flowExact[cell](1) += weight * 0.2 * wave / cfg.acmSettings.rho0 * cx * sy;
            }
            solver.GetState()[cell].template head<4>() = mean;
        }
        solver.EvaluateRHS(rhs);
        DNDS::real flowError = 0;
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); ++cell)
            flowError += vfv->GetCellVol(cell) *
                (ACM::State(rhs[cell].template head<4>()) - flowExact[cell]).squaredNorm();
        flowErrors.push_back(std::sqrt(flowError));
        std::cout << "ACM flow spatial n=" << n << " L2=" << flowErrors.back() << std::endl;
    }
    REQUIRE(errors.size() == 3);
    CHECK(std::log2(errors[1] / errors[2]) > 2.8);
    CHECK(std::log2(flowErrors[1] / flowErrors[2]) > 2.8);
}

TEST_CASE("ACM unified ESDIRK3 achieves third-order Wilcox physical decay")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    const auto error = [&](int count)
    {
        auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
        cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::ESDIRK3DualTimeGMRES;
        cfg.timeMarchSettings.nSteps = count;
        cfg.timeMarchSettings.physicalTimeStep = 2.0 / count;
        cfg.timeMarchSettings.pseudoTimeStep = 10;
        cfg.timeMarchSettings.maxImplicitIterations = 100;
        cfg.timeMarchSettings.implicitTolerance = 1e-12;
        cfg.timeMarchSettings.linearAbsoluteTolerance = 1e-14;
        cfg.timeMarchSettings.linearRelativeTolerance = 1e-10;
        cfg.turbulenceSettings.enableSourceTerms = true;
        cfg.initialState[5] = cfg.acmSettings.rho0 * 0.8;
        ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        solver.Run();
        const DNDS::real factor = 1 + 0.0708 * 0.8 * 2;
        const DNDS::real exactK = 0.4 * std::pow(factor, -0.09 / 0.0708);
        const DNDS::real exactOmega = 0.8 / factor;
        CHECK(solver.CompletedPhysicalSteps() == count);
        CHECK(solver.SimulationTime() == doctest::Approx(2));
        return std::hypot(solver.GetState()[0](4) / cfg.acmSettings.rho0 - exactK,
                          solver.GetState()[0](5) / cfg.acmSettings.rho0 - exactOmega);
    };
    const auto coarse = error(4), medium = error(8), fine = error(16);
    CHECK(coarse / medium > 7);
    CHECK(medium / fine > 7);
    CHECK(medium / fine < 9);
}

TEST_CASE("ACM unified two-equation BDF2 reproduces conservative Wilcox decay with both linear solvers")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    for (const auto &[integrator, restarts] : {std::pair{ACM::TimeIntegratorType::BDF2DualTimeLUSGS, 3},
                                             std::pair{ACM::TimeIntegratorType::BDF2DualTimeGMRES, 3},
                                             std::pair{ACM::TimeIntegratorType::BDF2DualTimeGMRES, 0}})
    {
        CAPTURE(static_cast<int>(integrator));
        CAPTURE(restarts);
        auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
        cfg.timeMarchSettings.integrator = integrator;
        cfg.timeMarchSettings.gmresRestarts = restarts;
        cfg.turbulenceSettings.enableSourceTerms = true;
        cfg.timeMarchSettings.nSteps = 4;
        ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        solver.Run();
        ACM::TurbulenceState expected, older;
        expected << 0.4, 0.08;
        older = expected;
        for (int step = 0; step < 4; step++)
        {
            const auto coefficients = ACM::GetBDF2Coefficients(step);
            const ACM::TurbulenceState history =
                -coefficients.previous * expected - coefficients.previousPrevious * older;
            ACM::TurbulenceState next;
            next(1) = 2 * history(1) / (coefficients.current +
                std::sqrt(coefficients.current * coefficients.current + 4 * 0.0708 * 0.01 * history(1)));
            next(0) = history(0) / (coefficients.current + 0.09 * 0.01 * next(1));
            older = expected;
            expected = next;
        }
        CHECK(solver.GetState()[0](4) / cfg.acmSettings.rho0 == doctest::Approx(expected(0)).epsilon(1e-10));
        CHECK(solver.GetState()[0](5) / cfg.acmSettings.rho0 == doctest::Approx(expected(1)).epsilon(1e-10));
    }
}

TEST_CASE("ACM field restart reinitializes BE BDF2 history after a physical step change")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    struct TempOutput
    {
        std::filesystem::path path;
        ~TempOutput() { std::error_code error; std::filesystem::remove_all(path, error); }
    } output{mesh.path.string() + ".restart"};
    auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
    cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::BDF2DualTimeLUSGS;
    cfg.turbulenceSettings.enableSourceTerms = true;
    cfg.timeMarchSettings.nSteps = 1;
    cfg.timeMarchSettings.physicalTimeStep = 0.001;
    cfg.outputSettings.interval = 1;
    cfg.outputSettings.writeInitial = false;
    cfg.outputSettings.directory = output.path.string();
    cfg.outputSettings.prefix = "restart";
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D> first(mpi, cfg);
    first.ReadMeshAndInitialize();
    first.Run();
    const auto initial = first.GetState()[0].eval();
    cfg.restartSettings.flowFile = (output.path / "restart_00000001.vtkhdf").string();
    cfg.restartSettings.completedSteps = 1;
    cfg.restartSettings.writerRanks = 1;
    cfg.restartSettings.requireTurbulence = true;
    CHECK_THROWS_AS(cfg.Validate(), std::runtime_error);
    cfg.restartSettings.initializeBDF2History = true;
    cfg.restartSettings.physicalTime = 0.001;
    cfg.timeMarchSettings.nSteps = 2;
    cfg.timeMarchSettings.physicalTimeStep = 0.01;
    cfg.Validate();
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D> second(mpi, cfg);
    second.ReadMeshAndInitialize();
    CHECK((second.GetState()[0] - initial).norm() < 1e-14);
    second.Run();
    CHECK(second.SimulationTime() == doctest::Approx(0.021));
    ACM::TurbulenceState expected = initial.tail<2>() / cfg.acmSettings.rho0;
    auto older = expected;
    for (int step = 0; step < 2; step++)
    {
        const auto coefficients = ACM::GetBDF2Coefficients(step);
        const ACM::TurbulenceState history = -coefficients.previous * expected - coefficients.previousPrevious * older;
        ACM::TurbulenceState next;
        next(1) = 2 * history(1) / (coefficients.current + std::sqrt(coefficients.current * coefficients.current +
            4 * 0.0708 * 0.01 * history(1)));
        next(0) = history(0) / (coefficients.current + 0.09 * 0.01 * next(1));
        older = expected;
        expected = next;
    }
    CHECK(second.GetState()[0](4) / cfg.acmSettings.rho0 == doctest::Approx(expected(0)).epsilon(1e-10));
    CHECK(second.GetState()[0](5) / cfg.acmSettings.rho0 == doctest::Approx(expected(1)).epsilon(1e-10));
    CHECK(std::filesystem::exists(output.path / "restart_00000003.vtkhdf"));
}

TEST_CASE("ACM conservative turbulence pseudo mass preserves constant scalar during a flow correction")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
    cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::BDF2DualTimeGMRES;
    // A deliberately incomplete physical solve tests pseudo-time constant-scalar
    // preservation; production cases still require converged physical histories.
    cfg.timeMarchSettings.maxImplicitIterations = 1;
    cfg.timeMarchSettings.nSteps = 1;
    cfg.timeMarchSettings.dualTimeRequireConvergence = false;
    cfg.timeMarchSettings.linearRelativeTolerance = 1e-12;
    cfg.turbulenceSettings.secondOrderReconstruction = false;
    cfg.turbulenceSettings.maximumEddyViscosityRatio = 1e-30;
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D> solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); cell++)
    {
        solver.GetState()[cell](0) = 0.3 + 0.1 * std::sin(1.0 + cell);
        solver.GetState()[cell](3) = 0.3 + 0.1 * std::cos(0.2 + cell);
    }
    solver.Run();
    for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); cell++)
    {
        CHECK(solver.GetState()[cell](4) == doctest::Approx(1.2).epsilon(1e-10));
        CHECK(solver.GetState()[cell](5) == doctest::Approx(0.24).epsilon(1e-10));
    }
}

TEST_CASE("ACM nonlinear mass continuation solves the original BDF2 equations")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    using Solver = ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D>;
    std::vector<Solver::TState> reference;
    for (int continuationSteps : {0, 1})
    {
        auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
        cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::BDF2DualTimeLUSGS;
        cfg.timeMarchSettings.dualTimeRANSContinuationSteps = continuationSteps;
        // This strict absolute-accuracy test solves all auxiliary stages without
        // Anderson acceleration; its budget is independent of production cases.
        cfg.timeMarchSettings.maxImplicitIterations = 3000;
        cfg.timeMarchSettings.implicitTolerance = 1e-10;
        cfg.turbulenceSettings.secondOrderReconstruction = false;
        cfg.turbulenceSettings.maximumEddyViscosityRatio = 1e-30;
        Solver solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); cell++)
        {
            solver.GetState()[cell](0) += 0.05 * std::sin(1.0 + cell);
            solver.GetState()[cell](1) += 0.03 * std::cos(1.0 + cell);
            solver.GetState()[cell](3) += 0.02 * std::sin(2.0 + cell);
            solver.GetState()[cell](4) += 0.02 * cell;
            solver.GetState()[cell](5) += 0.01 * cell;
        }
        solver.Run();
        REQUIRE(solver.CompletedPhysicalSteps() == 3);
        CHECK(solver.SimulationTime() == doctest::Approx(0.03));
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); cell++)
            if (continuationSteps == 0)
                reference.push_back(solver.GetState()[cell]);
            else
                CHECK((solver.GetState()[cell] - reference[cell]).norm() < 1e-9);
    }
}

TEST_CASE("ACM unfinished auxiliary stages cannot commit physical histories")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
    cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::BDF2DualTimeLUSGS;
    cfg.timeMarchSettings.dualTimeRANSContinuationSteps = 1;
    cfg.timeMarchSettings.maxImplicitIterations = 1;
    cfg.timeMarchSettings.nSteps = 1;
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D> solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    CHECK_THROWS_WITH_AS(solver.Run(), doctest::Contains("histories were not committed"), std::runtime_error);
    CHECK(solver.CompletedPhysicalSteps() == 0);
    CHECK(solver.SimulationTime() == 0);
}

TEST_CASE("ACM RANS prepared closure reuse preserves nonuniform LUSGS BDF2 transport")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    using Solver = ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D>;
    std::vector<Solver::TState> reference;
    DNDS::index referencePreparations = 0;
    for (bool reuse : {false, true})
    {
        auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
        cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::BDF2DualTimeLUSGS;
        cfg.timeMarchSettings.implicitTolerance = 1e-9;
        cfg.turbulenceSettings.secondOrderReconstruction = false;
        cfg.turbulenceSettings.maximumEddyViscosityRatio = 1e-30;
        Solver solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        const auto transport = solver.GetTurbulenceTransport();
        transport->SetPreparationReuse(reuse);
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); cell++)
        {
            solver.GetState()[cell](0) += 0.05 * std::sin(1.0 + cell);
            solver.GetState()[cell](1) += 0.03 * std::cos(1.0 + cell);
            solver.GetState()[cell](3) += 0.02 * std::sin(2.0 + cell);
            solver.GetState()[cell](4) += 0.02 * cell;
            solver.GetState()[cell](5) += 0.01 * cell;
        }
        solver.Run();
        REQUIRE(solver.CompletedPhysicalSteps() == 3);
        if (!reuse)
            referencePreparations = transport->PreparationCount();
        else
            CHECK(transport->PreparationCount() * 2 < referencePreparations);
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); cell++)
            if (!reuse)
                reference.push_back(solver.GetState()[cell]);
            else
                CHECK((solver.GetState()[cell] - reference[cell]).norm() < 1e-9);
    }
}

TEST_CASE("ACM RANS Anderson acceleration preserves converged BDF2 transport states")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    using Solver = ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D>;
    std::vector<Solver::TState> reference;
    for (int depth : {0, 6})
    {
        auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
        cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::BDF2DualTimeGMRES;
        cfg.timeMarchSettings.dualTimeRANSAndersonDepth = depth;
        // Explicit k stress introduces a pressure adjustment absent from passive transport.
        // Keep the strict tolerance and allow that additional mode to converge.
        cfg.timeMarchSettings.maxImplicitIterations = 500;
        cfg.timeMarchSettings.implicitTolerance = 1e-10;
        cfg.initialState[0] = 0.3;
        cfg.turbulenceSettings.secondOrderReconstruction = false;
        cfg.turbulenceSettings.maximumEddyViscosityRatio = 1e-30;
        Solver solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); cell++)
        {
            solver.GetState()[cell](4) += 0.02 * cell;
            solver.GetState()[cell](5) += 0.01 * cell;
        }
        solver.Run();
        REQUIRE(solver.CompletedPhysicalSteps() == 3);
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); cell++)
            if (depth == 0)
                reference.push_back(solver.GetState()[cell]);
            else
                CHECK((solver.GetState()[cell] - reference[cell]).norm() < 1e-9);
    }
}

TEST_CASE("ACM matrix free coupled transport preserves the BDF2 solution")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    using Solver = ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D>;
    std::vector<Solver::TState> reference;
    for (bool matrixFree : {false, true})
    {
        auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
        cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::BDF2DualTimeGMRES;
        cfg.timeMarchSettings.dualTimeRANSMatrixFreeJacobian = matrixFree;
        cfg.timeMarchSettings.implicitTolerance = 1e-9;
        cfg.turbulenceSettings.secondOrderReconstruction = false;
        cfg.turbulenceSettings.maximumEddyViscosityRatio = 1e-30;
        cfg.initialState[0] = 0.3;
        Solver solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); cell++)
        {
            solver.GetState()[cell](0) += 0.05 * std::sin(1.0 + cell);
            solver.GetState()[cell](1) += 0.03 * std::cos(1.0 + cell);
            solver.GetState()[cell](3) += 0.02 * std::sin(2.0 + cell);
            solver.GetState()[cell](4) += 0.02 * cell;
            solver.GetState()[cell](5) += 0.01 * cell;
        }
        solver.Run();
        REQUIRE(solver.CompletedPhysicalSteps() == 3);
        for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); cell++)
            if (!matrixFree)
                reference.push_back(solver.GetState()[cell]);
            else
                CHECK((solver.GetState()[cell] - reference[cell]).norm() < 1e-8);
    }
}

TEST_CASE("ACM matrix free probes do not add variational reconstruction sweeps")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
    cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::BDF2DualTimeGMRES;
    cfg.timeMarchSettings.dualTimeRANSMatrixFreeJacobian = true;
    cfg.timeMarchSettings.maxImplicitIterations = 1;
    cfg.timeMarchSettings.nSteps = 1;
    cfg.timeMarchSettings.dualTimeRequireConvergence = false;
    cfg.reconstructionSettings.type = ACM::ReconstructionType::Variational;
    cfg.reconstructionSettings.variationalIterations = 1;
    cfg.reconstructionSettings.variationalTolerance = 0;
    cfg.vfvSettings.maxOrder = 1;
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D> solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); cell++)
        solver.GetState()[cell](3) += 0.1 * std::sin(1.0 + cell);
    const auto before = solver.GetEvaluator()->GetTotalReconstructionSweeps();
    solver.Run();
    CHECK(solver.GetEvaluator()->GetTotalReconstructionSweeps() - before == 2);
}

TEST_CASE("ACM unified RANS keeps failed physical histories uncommitted and validates model choices")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
    cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::BDF2DualTimeLUSGS;
    cfg.turbulenceSettings.enableSourceTerms = true;
    cfg.timeMarchSettings.maxImplicitIterations = 1;
    cfg.timeMarchSettings.implicitTolerance = 1e-25;
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D> solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    CHECK_THROWS_WITH_AS(solver.Run(), doctest::Contains("histories were not committed"), std::runtime_error);
    CHECK(solver.CompletedPhysicalSteps() == 0);
    CHECK(solver.SimulationTime() == 0);
    cfg.turbulenceSettings.model = ACM::TurbulenceModel::SpalartAllmaras;
    CHECK_THROWS_AS(ACM::ValidateModelConfiguration<ACM::ACMModel::ConstantDensity2EQ2D>(cfg), std::runtime_error);
    cfg.initialState.resize(4);
    CHECK_THROWS_AS(cfg.Validate(), std::runtime_error);
}

TEST_CASE("ACM unified two-equation state supports Wilcox SST and realizable k-epsilon closures")
{
    MPIInfo mpi;
    mpi.setWorld();
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    for (auto closure : {ACM::TurbulenceModel::KOmegaWilcox, ACM::TurbulenceModel::KOmegaSST,
                         ACM::TurbulenceModel::RealizableKEpsilon})
    {
        auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
        cfg.turbulenceSettings.model = closure;
        cfg.turbulenceSettings.enableSourceTerms = true;
        cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::BDF2DualTimeLUSGS;
        cfg.timeMarchSettings.nSteps = 1;
        ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        const DNDS::real k0 = solver.GetState()[0](4);
        solver.Run();
        CHECK(solver.GetState()[0].allFinite());
        CHECK(solver.GetState()[0](4) < k0);
        CHECK(solver.GetState()[0](4) > 0);
        CHECK(solver.GetState()[0](5) > 0);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}

/// @test Positive cell floors must not reintroduce turbulent viscosity on no-slip faces.
TEST_CASE("ACM and variable ACM no-slip faces have exactly zero eddy viscosity")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path, false);
    const auto checkFaces = [](auto &transport, auto &solver)
    {
        transport.Initialize();
        transport.Prepare(solver.GetState());
        const auto mesh = solver.GetMesh();
        REQUIRE(mesh->NumFaceProc() == 4);
        for (DNDS::index face = 0; face < mesh->NumFaceProc(); ++face)
        {
            const auto quad = solver.GetReconstruction()->GetFaceQuad(face);
            for (int point = 0; point < quad.GetNumPoints(); ++point)
                CHECK(transport.FaceEddyViscosity(face, point) == 0);
        }
    };
    for (const auto wall : {ACM::BoundaryType::BCWall, ACM::BoundaryType::BCWallIsothermal})
    {
        auto cfg = MakeRANSPhysicalCase(meshFile.path, ACM::TurbulenceModel::KOmegaSST,
                                        ACM::TimeIntegratorType::BDF2DualTimeLUSGS, 0.05, 0);
        cfg.defaultBoundaryType = wall;
        cfg.turbulenceSettings.minimumValue[0] = 0.01;
        ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        auto boundary = std::make_shared<ACM::BoundaryHandler>(cfg.defaultBoundaryType,
            cfg.InitialState(), cfg.boundaryConditions);
        ACM::ACMTurbulenceTransport<2> transport(solver.GetMesh(), solver.GetReconstruction(),
            cfg.acmSettings, cfg.turbulenceSettings, boundary);
        checkFaces(transport, solver);
    }
    for (const auto wall : {ACMVariable::BoundaryType::BCWall, ACMVariable::BoundaryType::BCWallIsothermal})
    {
        ACMVariable::KernelConfiguration cfg;
        cfg.meshSettings.meshFile = meshFile.path.string();
        cfg.initialState = {1, 0, 0, 0, 0};
        cfg.defaultBoundaryType = wall;
        cfg.acmSettings.enableViscousFlux = true;
        cfg.acmSettings.dynamicViscosity = 0.001;
        cfg.turbulenceSettings.model = ACMVariable::TurbulenceModel::KOmegaSST;
        cfg.turbulenceSettings.initialValue = {0.2, 4};
        cfg.turbulenceSettings.minimumValue[0] = 0.01;
        ACMVariable::ACMSolver<ACMVariable::ACMModel::VariableDensity2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        auto boundary = std::make_shared<ACMVariable::BoundaryHandler>(cfg.defaultBoundaryType,
            cfg.InitialState(), cfg.boundaryConditions);
        ACMVariable::ACMTurbulenceTransport<2> transport(solver.GetMesh(), solver.GetReconstruction(),
            cfg.acmSettings, cfg.turbulenceSettings, boundary);
        checkFaces(transport, solver);
    }
}

/// @test A nonuniform k and compensating static pressure must not create velocity or continuity defects.
TEST_CASE("ACM explicit k stress preserves total pressure equilibrium and its cross Jacobian")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WritePeriodicQuadGrid(meshFile.path);
    for (const DNDS::real rho : {1.0, 3.0})
    {
        auto cfg = MakeRANSPhysicalCase(meshFile.path, ACM::TurbulenceModel::KOmegaSST,
            ACM::TimeIntegratorType::BDF2DualTimeLUSGS, 0.01, 0);
        cfg.meshSettings.periodicTranslation1 = {1, 0, 0};
        cfg.meshSettings.periodicTranslation2 = {0, 1, 0};
        cfg.acmSettings.rho0 = rho;
        cfg.reconstructionSettings.type = ACM::ReconstructionType::FirstOrder;
        cfg.turbulenceSettings.secondOrderReconstruction = false;
        cfg.turbulenceSettings.enableSourceTerms = false;
        ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        auto transport = solver.GetTurbulenceTransport();
        auto &q = transport->GetState();
        auto &u = solver.GetState();
        auto evaluator = solver.GetEvaluator();
        auto metrics = solver.GetReconstruction();
        auto mesh = solver.GetMesh();
        decltype(solver)::TDof rhs, plus, minus;
        for (auto field : {&rhs, &plus, &minus}) metrics->BuildUDof(*field, 4);
        std::vector<DNDS::real> k(mesh->NumCell()), direction(mesh->NumCell());
        for (DNDS::index cell = 0; cell < mesh->NumCell(); ++cell)
        {
            k[cell] = 0.2 + 0.1 * cell;
            direction[cell] = std::sin(0.4 + cell);
            q[cell] << k[cell], 4;
            u[cell] << 0, 0, 0, 1 - (2.0 / 3.0) * rho * k[cell];
        }
        evaluator->EvaluateRHS(rhs, u);
        for (DNDS::index cell = 0; cell < mesh->NumCell(); ++cell)
            CHECK(rhs[cell].norm() < 1e-11);
        for (DNDS::index face = 0; face < mesh->NumFaceProc(); ++face)
            CHECK(std::abs(evaluator->FaceVolumeFlux(u, face, 0)) < 1e-12);
        ACM::ACMEvaluator<2>::KineticPressureJacobianField blocks;
        evaluator->AssembleKineticPressureLinearization(u, blocks);
        std::vector<ACM::State> product(mesh->NumCell(), ACM::State::Zero());
        for (DNDS::index face = 0; face < mesh->NumFaceProc(); ++face)
        {
            const auto cells = mesh->face2cell[face];
            const ACM::State flux = blocks[face][0] * direction[cells[0]] + blocks[face][1] * direction[cells[1]];
            product[cells[0]] += flux / metrics->GetCellVol(cells[0]);
            product[cells[1]] -= flux / metrics->GetCellVol(cells[1]);
        }
        constexpr DNDS::real epsilon = 1e-6;
        for (DNDS::index cell = 0; cell < mesh->NumCell(); ++cell)
            q[cell](0) = k[cell] + epsilon * direction[cell] / rho;
        evaluator->EvaluateRHS(plus, u);
        for (DNDS::index cell = 0; cell < mesh->NumCell(); ++cell)
            q[cell](0) = k[cell] - epsilon * direction[cell] / rho;
        evaluator->EvaluateRHS(minus, u);
        for (DNDS::index cell = 0; cell < mesh->NumCell(); ++cell)
            CHECK((product[cell] + (plus[cell] - minus[cell]) / (2 * epsilon)).norm() < 1e-7);
    }
}

TEST_CASE("Variable ACM explicit k stress preserves total pressure equilibrium")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WritePeriodicQuadGrid(meshFile.path);
    ACMVariable::KernelConfiguration cfg;
    cfg.meshSettings.meshFile = meshFile.path.string();
    cfg.meshSettings.periodicTranslation1 = {1, 0, 0};
    cfg.meshSettings.periodicTranslation2 = {0, 1, 0};
    cfg.initialState = {1, 0, 0, 0, 0.3};
    cfg.acmSettings.enableViscousFlux = true;
    cfg.acmSettings.dynamicViscosity = 0.001;
    cfg.acmSettings.entropyFixRatio = 0; // Retain an exact stationary density contact.
    cfg.reconstructionSettings.type = ACMVariable::ReconstructionType::FirstOrder;
    cfg.turbulenceSettings.model = ACMVariable::TurbulenceModel::KOmegaSST;
    cfg.turbulenceSettings.secondOrderReconstruction = false;
    cfg.turbulenceSettings.enableSourceTerms = false;
    ACMVariable::ACMSolver<ACMVariable::ACMModel::VariableDensity2D> solver(mpi, cfg);
    solver.ReadMeshAndInitialize();
    auto boundary = std::make_shared<ACMVariable::BoundaryHandler>(cfg.defaultBoundaryType,
        cfg.InitialState(), cfg.boundaryConditions);
    auto transport = std::make_shared<ACMVariable::ACMTurbulenceTransport<2>>(
        solver.GetMesh(), solver.GetReconstruction(), cfg.acmSettings, cfg.turbulenceSettings, boundary);
    transport->Initialize();
    solver.GetEvaluator()->SetTurbulenceCoupling(
        [transport](auto &flow, DNDS::real time) { transport->Prepare(flow, time); },
        [transport](DNDS::index f, int p) { return transport->FaceEddyViscosity(f, p); },
        [transport](DNDS::index f, int p, const ACMVariable::State &l, const ACMVariable::State &rr,
                    DNDS::real dl, DNDS::real dr) { return transport->FaceTurbulentPressure(f, p, l, rr, dl, dr); });
    auto &q = transport->GetState();
    auto &u = solver.GetState();
    const auto mesh = solver.GetMesh();
    for (DNDS::index cell = 0; cell < mesh->NumCell(); ++cell)
    {
        const DNDS::real rho = 1 + 0.2 * cell, k = 0.2 + 0.1 * cell;
        q[cell] << k, 4;
        u[cell] << rho, 0, 0, 0, 1 - (2.0 / 3.0) * rho * k;
    }
    decltype(solver)::TDof rhs;
    solver.GetReconstruction()->BuildUDof(rhs, 5);
    solver.GetEvaluator()->EvaluateRHS(rhs, u);
    for (DNDS::index cell = 0; cell < mesh->NumCell(); ++cell)
        CHECK(rhs[cell].norm() < 1e-10);
}

TEST_CASE("ACM far field and no-slip walls use consistent static and turbulent pressure")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile meshFile;
    WriteSingleCellQuad(meshFile.path, false);
    for (const auto type : {ACM::BoundaryType::BCFar, ACM::BoundaryType::BCWall})
    {
        auto cfg = MakeRANSPhysicalCase(meshFile.path, ACM::TurbulenceModel::KOmegaSST,
            ACM::TimeIntegratorType::BDF2DualTimeLUSGS, 0.01, 0);
        cfg.defaultBoundaryType = type;
        cfg.acmSettings.rho0 = 3;
        cfg.turbulenceSettings.initialValue = {0.2, 4};
        cfg.turbulenceSettings.farFieldValue = {0.05, 4};
        cfg.turbulenceSettings.secondOrderReconstruction = false;
        cfg.turbulenceSettings.enableSourceTerms = true;
        cfg.initialState = {0, 0, 0, 0.6}; // p + (2/3)*rho*k = 1
        cfg.boundaryValue = {0, 0, 0, 0.9}; // p_far + (2/3)*rho*k_far = 1
        cfg.acmSettings.farFieldValue = cfg.boundaryValue;
        cfg.reconstructionSettings.type = ACM::ReconstructionType::FirstOrder;
        ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        decltype(solver)::TDof rhs;
        solver.GetReconstruction()->BuildUDof(rhs, 4);
        solver.GetEvaluator()->EvaluateRHS(rhs, solver.GetState());
        CHECK(rhs[0].norm() < 1e-10);
        if (type == ACM::BoundaryType::BCFar)
        {
            const auto source = solver.GetTurbulenceTransport()->CellSource(0);
            CHECK(source(0) == doctest::Approx(-0.09 * 0.2 * 4));
            CHECK(source(1) == doctest::Approx(-0.0828 * 16));
        }
        for (DNDS::index face = 0; face < solver.GetMesh()->NumFaceProc(); ++face)
            CHECK(std::abs(solver.GetEvaluator()->FaceVolumeFlux(solver.GetState(), face, 0)) < 1e-12);
    }
}

TEST_CASE("ACM modified-pressure restart converts to static pressure without double counting k")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    struct Output
    {
        std::filesystem::path path;
        ~Output() { std::error_code error; std::filesystem::remove_all(path, error); }
    } output{mesh.path.string() + ".pressure_restart"};
    auto cfg = MakeUnifiedCase<ACM::ACMModel::ConstantDensity2EQ2D>(mesh.path);
    cfg.turbulenceSettings.includeIsotropicStress = false;
    cfg.timeMarchSettings.integrator = ACM::TimeIntegratorType::ImplicitEulerGMRES;
    cfg.timeMarchSettings.nSteps = 0;
    cfg.outputSettings.interval = 1;
    cfg.outputSettings.writeInitial = true;
    cfg.outputSettings.directory = output.path.string();
    cfg.outputSettings.prefix = "legacy";
    using Solver = ACM::ACMSolver<ACM::ACMModel::ConstantDensity2EQ2D>;
    Solver source(mpi, cfg);
    source.ReadMeshAndInitialize();
    source.Run();
    cfg.restartSettings.flowFile = (output.path / "legacy_00000000.vtkhdf").string();
    cfg.restartSettings.writerRanks = 1;
    cfg.restartSettings.requireTurbulence = true;
    cfg.restartSettings.pressureIsModified = true;
    cfg.turbulenceSettings.includeIsotropicStress = true;
    Solver resumed(mpi, cfg);
    resumed.ReadMeshAndInitialize();
    for (DNDS::index cell = 0; cell < resumed.GetMesh()->NumCell(); ++cell)
    {
        CHECK(resumed.GetState()[cell](3) == doctest::Approx(
            source.GetState()[cell](3) - (2.0 / 3.0) * source.GetState()[cell](4)));
        CHECK(resumed.GetState()[cell](4) == source.GetState()[cell](4));
    }
}

/// Diffusion ghost k=2*0.001-0.01 is negative; it must never be the incoming advective state.
TEST_CASE("ACM prescribed turbulence inflow uses face data and its implicit derivative")
{
    MPIInfo mpi; mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile mesh;
    WriteSingleCellQuad(mesh.path, false);
    for (const auto boundaryType : {ACM::BoundaryType::BCIn, ACM::BoundaryType::BCInPsTs,
                                    ACM::BoundaryType::BCSpecial, ACM::BoundaryType::BCFar})
    {
        auto cfg = MakeRANSPhysicalCase(mesh.path, ACM::TurbulenceModel::KOmegaSST,
            ACM::TimeIntegratorType::BDF2DualTimeLUSGS, 0.01, 0);
        cfg.defaultBoundaryType = boundaryType;
        cfg.initialState = {0, 0, 0, 0};
        cfg.boundaryValue = {0, 0, 0, 0};
        cfg.acmSettings.farFieldValue = {0, 0, 0, 0};
        cfg.acmSettings.dynamicViscosity = 0.001;
        cfg.turbulenceSettings.initialValue = {0.01, 4};
        cfg.turbulenceSettings.farFieldValue = {0.001, 2};
        cfg.turbulenceSettings.secondOrderReconstruction = false;
        cfg.turbulenceSettings.enableSourceTerms = false;
        cfg.reconstructionSettings.type = ACM::ReconstructionType::FirstOrder;
        ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        auto transport = solver.GetTurbulenceTransport();
        const auto metrics = solver.GetReconstruction();
        ACM::ACMTurbulenceTransport<2>::TTurbulenceDof base, incoming;
        metrics->BuildUDof(base, 2); metrics->BuildUDof(incoming, 2);
        transport->SetFaceVolumeFlux([](DNDS::index, int, DNDS::real) { return 0.0; });
        transport->EvaluateRHS(base, solver.GetState());
        std::vector<Eigen::Matrix<DNDS::real, 2, 2>> noAdvection, withAdvection;
        transport->EvaluateImplicitDiagonal(solver.GetState(), noAdvection, 0);
        transport->SetFaceVolumeFlux([](DNDS::index, int, DNDS::real) { return -1.0; });
        transport->EvaluateRHS(incoming, solver.GetState());
        transport->EvaluateImplicitDiagonal(solver.GetState(), withAdvection, 0);
        CHECK((incoming[0] - base[0] - 4 * cfg.turbulenceSettings.FarFieldState()).norm() < 1e-11);
        // Prescribed incoming advection has zero derivative with respect to the owner state.
        CHECK((withAdvection[0] - noAdvection[0]).norm() < 1e-11);
        for (DNDS::index face = 0; face < solver.GetMesh()->NumFaceProc(); face++)
            CHECK((transport->FrozenFaceAdvectionValue(solver.GetState(), face) -
                cfg.turbulenceSettings.FarFieldState()).norm() < 1e-12);
        transport->SetFaceVolumeFlux([](DNDS::index, int, DNDS::real) { return 1.0; });
        transport->EvaluateRHS(incoming, solver.GetState());
        CHECK((incoming[0] - base[0] + 4 * cfg.turbulenceSettings.InitialState()).norm() < 1e-11);
    }
}

TEST_CASE("Variable ACM prescribed turbulence inflow uses face values with shared mass flux")
{
    MPIInfo mpi; mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile mesh;
    WriteSingleCellQuad(mesh.path, false);
    for (const auto boundaryType : {ACMVariable::BoundaryType::BCIn, ACMVariable::BoundaryType::BCInPsTs,
                                    ACMVariable::BoundaryType::BCSpecial, ACMVariable::BoundaryType::BCFar})
    {
        ACMVariable::KernelConfiguration cfg;
        cfg.meshSettings.meshFile = mesh.path.string();
        cfg.initialState = {3, 0, 0, 0, 0};
        cfg.boundaryValue = cfg.initialState;
        cfg.defaultBoundaryType = boundaryType;
        cfg.acmSettings.enableViscousFlux = true;
        cfg.acmSettings.dynamicViscosity = 0.001;
        cfg.reconstructionSettings.type = ACMVariable::ReconstructionType::FirstOrder;
        cfg.turbulenceSettings.model = ACMVariable::TurbulenceModel::KOmegaSST;
        cfg.turbulenceSettings.initialValue = {0.01, 4};
        cfg.turbulenceSettings.farFieldValue = {0.001, 2};
        cfg.turbulenceSettings.secondOrderReconstruction = false;
        cfg.turbulenceSettings.enableSourceTerms = false;
        ACMVariable::ACMSolver<ACMVariable::ACMModel::VariableDensity2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        auto boundary = std::make_shared<ACMVariable::BoundaryHandler>(cfg.defaultBoundaryType,
            cfg.InitialState(), cfg.boundaryConditions);
        ACMVariable::ACMTurbulenceTransport<2> transport(solver.GetMesh(), solver.GetReconstruction(),
            cfg.acmSettings, cfg.turbulenceSettings, boundary);
        transport.Initialize();
        ACMVariable::ACMTurbulenceTransport<2>::TTurbulenceDof base, incoming;
        solver.GetReconstruction()->BuildUDof(base, 2); solver.GetReconstruction()->BuildUDof(incoming, 2);
        transport.SetMassFluxProvider([](DNDS::index, int) { return 0.0; });
        transport.EvaluateRHS(base, solver.GetState());
        transport.SetMassFluxProvider([](DNDS::index, int) { return -3.0; });
        transport.EvaluateRHS(incoming, solver.GetState());
        CHECK((incoming[0] - base[0] - 12 * cfg.turbulenceSettings.FarFieldState()).norm() < 1e-11);
        transport.SetMassFluxProvider([](DNDS::index, int) { return 3.0; });
        transport.EvaluateRHS(incoming, solver.GetState());
        CHECK((incoming[0] - base[0] + 12 * cfg.turbulenceSettings.InitialState()).norm() < 1e-11);
    }
}

TEST_CASE("Euler SST impermeable faces have zero advective turbulence flux for either eigenvalue scheme")
{
    MPIInfo mpi; mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile mesh;
    WriteSingleCellQuad(mesh.path);
    // Reuse only the mesh/CFV fixture; the numerical flux is the production Euler evaluator.
    auto cfg = MakeRANSPhysicalCase(mesh.path, ACM::TurbulenceModel::Laminar,
        ACM::TimeIntegratorType::BDF2DualTimeLUSGS, 0.01, 0);
    cfg.meshSettings.periodicTranslation1 = {1, 0, 0};
    cfg.meshSettings.periodicTranslation2 = {0, 1, 0};
    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> fixture(mpi, cfg);
    fixture.ReadMeshAndInitialize();
    using Eval = Euler::EulerEvaluator<Euler::NS_2EQ>;
    Euler::EulerEvaluatorSettings<Euler::NS_2EQ> settings(7);
    settings.ransModel = Euler::RANSModel::RANS_KOSST;
    auto boundary = std::make_shared<Euler::BoundaryHandler<Euler::NS_2EQ>>(7);
    Eval eval(fixture.GetMesh(), fixture.GetReconstruction(), boundary, settings);
    eval.deltaLambdaCell.setConstant(1.0);
    Eval::TU left, right;
    left << 1, 0.2, 0, 0, 2.52, 0.01, 4;
    right = left;
    right(1) = -left(1);
    right(5) = -left(5);
    right(6) = 8 - left(6);
    Eval::TU_Batch l = left, r = right, flux, fixL, fixR;
    Eval::TDiffU_Batch gradient(3, 7); gradient.setZero();
    Eval::TDiffU_Batch primitiveGradient = gradient;
    Eval::TVec_Batch normal(3, 1), grid(3, 1);
    normal << 1, 0, 0; grid.setZero();
    Eval::TReal_Batch lambda0, lambdaMid, lambda4;
    for (const int scheme : {0, 1})
    {
        eval.settings.ransEigScheme = scheme;
        eval.fluxFace(l, r, left, right, gradient, primitiveGradient, normal, grid,
            Eval::TVec::UnitX(), Eval::TVec::Zero(), fixL, fixR, flux, lambda0, lambdaMid, lambda4,
            Geom::BC_ID_DEFAULT_WALL, Euler::Gas::RiemannSolverType::Roe, 0, true);
        CHECK(flux(0, 0) == 0);
        CHECK(flux(5, 0) == 0);
        CHECK(flux(6, 0) == 0);
    }
}

TEST_CASE("Both ACM modules recover affine shear production on a skew stretched cell")
{
    MPIInfo mpi; mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile mesh;
    WriteSingleCellQuad(mesh.path, false, true);
    constexpr DNDS::real rho = 3, k = 0.2, omega = 4;
    const DNDS::real gamma2 = 0.0828 / 0.09 - 0.856 * 0.41 * 0.41 / 0.3;
    // No walls, F1=F2=0: nu_t=k/omega, du/dy=2, all other derivatives zero.
    Eigen::Vector2d expected;
    expected << (k / omega) * 4 - 0.09 * k * omega, gamma2 * 4 - 0.0828 * omega * omega;
    {
        auto cfg = MakeRANSPhysicalCase(mesh.path, ACM::TurbulenceModel::KOmegaSST,
            ACM::TimeIntegratorType::BDF2DualTimeLUSGS, 0.01, 0);
        cfg.initialState = {0.01, 0, 0, 1};
        cfg.acmSettings.rho0 = rho;
        cfg.reconstructionSettings.type = ACM::ReconstructionType::FirstOrder;
        cfg.turbulenceSettings.initialValue = {k, omega};
        cfg.turbulenceSettings.minimumWallDistance = 1e20;
        cfg.turbulenceSettings.enableSourceTerms = true;
        ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        auto transport = solver.GetTurbulenceTransport();
        transport->SetFlowBoundaryProvider([](DNDS::index, const ACM::State &left,
            const ACM::Vector3 &, const ACM::Vector3 &point, DNDS::real)
        {
            ACM::State face; face << 2 * point(1), 0, 0, 1;
            return ACM::State(2 * face - left);
        });
        transport->SetFaceVolumeFlux([](DNDS::index, int, DNDS::real) { return 0.0; });
        ACM::ACMTurbulenceTransport<2>::TTurbulenceDof rhs;
        solver.GetReconstruction()->BuildUDof(rhs, 2);
        transport->EvaluateRHS(rhs, solver.GetState());
        CHECK((rhs[0] - expected).norm() < 1e-10);
    }
    {
        ACMVariable::KernelConfiguration cfg;
        cfg.meshSettings.meshFile = mesh.path.string();
        cfg.initialState = {rho, rho * 0.01, 0, 0, 1};
        cfg.defaultBoundaryType = ACMVariable::BoundaryType::BCSym;
        cfg.acmSettings.enableViscousFlux = true;
        cfg.acmSettings.dynamicViscosity = 0.001;
        cfg.reconstructionSettings.type = ACMVariable::ReconstructionType::FirstOrder;
        cfg.turbulenceSettings.model = ACMVariable::TurbulenceModel::KOmegaSST;
        cfg.turbulenceSettings.initialValue = {k, omega};
        cfg.turbulenceSettings.minimumWallDistance = 1e20;
        cfg.turbulenceSettings.secondOrderReconstruction = false;
        cfg.turbulenceSettings.enableSourceTerms = true;
        ACMVariable::ACMSolver<ACMVariable::ACMModel::VariableDensity2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        auto boundary = std::make_shared<ACMVariable::BoundaryHandler>(cfg.defaultBoundaryType,
            cfg.InitialState(), cfg.boundaryConditions);
        ACMVariable::ACMTurbulenceTransport<2> transport(solver.GetMesh(), solver.GetReconstruction(),
            cfg.acmSettings, cfg.turbulenceSettings, boundary);
        transport.Initialize();
        transport.SetFlowBoundaryProvider([=](DNDS::index, const ACMVariable::State &left,
            const ACMVariable::Vector3 &, const ACMVariable::Vector3 &point, DNDS::real)
        {
            ACMVariable::State face; face << rho, rho * 2 * point(1), 0, 0, 1;
            return ACMVariable::State(2 * face - left);
        });
        transport.SetMassFluxProvider([](DNDS::index, int) { return 0.0; });
        ACMVariable::ACMTurbulenceTransport<2>::TTurbulenceDof rhs;
        solver.GetReconstruction()->BuildUDof(rhs, 2);
        transport.EvaluateRHS(rhs, solver.GetState());
        CHECK((rhs[0] / rho - expected).norm() < 1e-10);
    }
}

TEST_CASE("SST near-floor diffusion shares conservative residual and frozen face matrix")
{
    MPIInfo mpi; mpi.setWorld();
    REQUIRE(mpi.size == 1);
    TempMeshFile mesh;
    WritePeriodicQuadGrid(mesh.path);
    for (const bool bounded : {false, true})
    {
        auto cfg = MakeRANSPhysicalCase(mesh.path, ACM::TurbulenceModel::KOmegaSST,
            ACM::TimeIntegratorType::BDF2DualTimeLUSGS, 0.01, 0);
        cfg.meshSettings.periodicTranslation1 = {1, 0, 0};
        cfg.meshSettings.periodicTranslation2 = {0, 1, 0};
        cfg.turbulenceSettings.boundedSSTDiffusion = bounded;
        cfg.turbulenceSettings.minimumValue = {1e-14, 1e-10};
        cfg.turbulenceSettings.enableSourceTerms = false;
        cfg.turbulenceSettings.maximumEddyViscosityRatio = 0;
        cfg.turbulenceSettings.secondOrderReconstruction = false;
        cfg.reconstructionSettings.type = ACM::ReconstructionType::FirstOrder;
        ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, cfg);
        solver.ReadMeshAndInitialize();
        auto transport = solver.GetTurbulenceTransport();
        transport->SetFaceVolumeFlux([](DNDS::index, int, DNDS::real) { return 0.0; });
        const auto meshData = solver.GetMesh();
        const auto metrics = solver.GetReconstruction();
        const ACM::TurbulenceState floor(1e-14, 1e-10);
        std::vector<ACM::TurbulenceState> difference(meshData->NumCell());
        for (DNDS::index cell = 0; cell < meshData->NumCell(); cell++)
        {
            transport->GetState()[cell] = floor * (1 + cell * 1e-13);
            difference[cell] = transport->GetState()[cell] - floor;
        }
        ACM::ACMTurbulenceTransport<2>::TTurbulenceDof rhs;
        metrics->BuildUDof(rhs, 2);
        transport->EvaluateRHS(rhs, solver.GetState());
        std::vector<Eigen::Matrix<DNDS::real, 2, 2>> diagonal;
        ACM::ACMTurbulenceTransport<2>::FaceJacobianField blocks;
        transport->EvaluateImplicitDiagonal(solver.GetState(), diagonal, 0, &blocks);
        std::vector<ACM::TurbulenceState> product(meshData->NumCell());
        for (DNDS::index cell = 0; cell < meshData->NumCell(); cell++)
            product[cell] = diagonal[cell] * difference[cell];
        for (DNDS::index face = 0; face < meshData->NumFaceProc(); face++)
        {
            const auto cells = meshData->face2cell[face];
            REQUIRE(cells[1] != UnInitIndex);
            if (cells[0] == cells[1]) continue;
            CHECK(blocks[face][0](0, 0) >= 0);
            CHECK(blocks[face][1](0, 0) <= 0);
            product[cells[0]] += blocks[face][1] * difference[cells[1]] / metrics->GetCellVol(cells[0]);
            product[cells[1]] -= blocks[face][0] * difference[cells[0]] / metrics->GetCellVol(cells[1]);
        }
        ACM::TurbulenceState balance = ACM::TurbulenceState::Zero();
        DNDS::real scale = 0;
        for (DNDS::index cell = 0; cell < meshData->NumCell(); cell++)
        {
            const DNDS::real norm = rhs[cell].norm();
            CHECK((product[cell] + rhs[cell]).norm() <= 1e-10 * std::max(norm, DNDS::real(1e-300)));
            balance += metrics->GetCellVol(cell) * rhs[cell];
            scale += metrics->GetCellVol(cell) * norm;
        }
        CHECK(balance.norm() <= 1e-12 * scale);
        CHECK((rhs[0].array() >= 0).all());

        ACMVariable::KernelConfiguration variable;
        variable.meshSettings.meshFile = mesh.path.string();
        variable.meshSettings.periodicTranslation1 = {1, 0, 0};
        variable.meshSettings.periodicTranslation2 = {0, 1, 0};
        variable.initialState = {1, 0, 0, 0, 0};
        variable.acmSettings.enableViscousFlux = true;
        variable.acmSettings.dynamicViscosity = cfg.acmSettings.dynamicViscosity;
        variable.reconstructionSettings.type = ACMVariable::ReconstructionType::FirstOrder;
        variable.turbulenceSettings.model = ACMVariable::TurbulenceModel::KOmegaSST;
        variable.turbulenceSettings.minimumValue = {1e-14, 1e-10};
        variable.turbulenceSettings.enableSourceTerms = false;
        variable.turbulenceSettings.maximumEddyViscosityRatio = 0;
        variable.turbulenceSettings.secondOrderReconstruction = false;
        variable.turbulenceSettings.boundedSSTDiffusion = bounded;
        ACMVariable::ACMSolver<ACMVariable::ACMModel::VariableDensity2D> other(mpi, variable);
        other.ReadMeshAndInitialize();
        auto boundary = std::make_shared<ACMVariable::BoundaryHandler>(variable.defaultBoundaryType,
            variable.InitialState(), variable.boundaryConditions);
        ACMVariable::ACMTurbulenceTransport<2> otherTransport(other.GetMesh(), other.GetReconstruction(),
            variable.acmSettings, variable.turbulenceSettings, boundary);
        otherTransport.Initialize();
        otherTransport.SetMassFluxProvider([](DNDS::index, int) { return 0.0; });
        for (DNDS::index cell = 0; cell < other.GetMesh()->NumCell(); cell++)
        {
            other.GetState()[cell](0) = 1 + 0.2 * cell;
            otherTransport.GetState()[cell] = transport->GetState()[cell];
        }
        ACMVariable::ACMTurbulenceTransport<2>::TTurbulenceDof variableRhs;
        other.GetReconstruction()->BuildUDof(variableRhs, 2);
        otherTransport.EvaluateRHS(variableRhs, other.GetState());
        for (DNDS::index cell = 0; cell < other.GetMesh()->NumCell(); cell++)
            CHECK((variableRhs[cell] - rhs[cell]).norm() <= 1e-10 * std::max(rhs[cell].norm(), DNDS::real(1e-300)));
    }
}
