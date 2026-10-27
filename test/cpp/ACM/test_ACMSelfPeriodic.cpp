/**
 * @file test_ACMSelfPeriodic.cpp
 * @brief Regression tests for periodic ACM implicit operators and primitive RANS physical time.
 */
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "ACM/ACMSolver.hpp"
#include "ACMVariable/ACMSolver.hpp"

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

    void WriteSingleCellQuad(const std::filesystem::path &path, bool periodic = true)
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

            const std::array<double, 4> x{0, 1, 0, 1};
            const std::array<double, 4> y{0, 0, 1, 1};
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
                // On a unit cell, inflow reflection plus convection/diffusion gives
                // dq/dt = 2*(1+D_in)*(q_in-q), D_in=(nu+q_in)/(2/3).
                constexpr DNDS::real inlet = 0.1;
                constexpr DNDS::real rate = 2 * (1 + (0.001 + inlet) / (2.0 / 3.0));
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
                constexpr DNDS::real rate = 2 * (1 + (0.001 + 0.1) / (2.0 / 3.0));
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
    void WritePeriodicQuadGrid(const std::filesystem::path &path)
    {
        int file = -1, base = 0, zone = 0, coordinate = 0, section = 0, boundary = 0;
        CheckCGNS(cg_open(path.c_str(), CG_MODE_WRITE, &file));
        try
        {
            CheckCGNS(cg_base_write(file, "Base", 2, 2, &base));
            cgsize_t size[3] = {9, 4, 0};
            CheckCGNS(cg_zone_write(file, base, "Zone", size, Unstructured, &zone));
            std::array<double, 9> x{}, y{};
            for (int j = 0; j < 3; j++)
                for (int i = 0; i < 3; i++)
                {
                    x[3 * j + i] = 0.5 * i;
                    y[3 * j + i] = 0.5 * j;
                }
            CheckCGNS(cg_coord_write(file, base, zone, RealDouble, "CoordinateX", x.data(), &coordinate));
            CheckCGNS(cg_coord_write(file, base, zone, RealDouble, "CoordinateY", y.data(), &coordinate));
            std::array<cgsize_t, 16> cells{};
            int offset = 0;
            for (int j = 0; j < 2; j++)
                for (int i = 0; i < 2; i++)
                {
                    const cgsize_t first = 3 * j + i + 1;
                    cells[offset++] = first;
                    cells[offset++] = first + 1;
                    cells[offset++] = first + 4;
                    cells[offset++] = first + 3;
                }
            CheckCGNS(cg_section_write(file, base, zone, "Cells", QUAD_4, 1, 4, 0, cells.data(), &section));
            const std::array<const char *, 4> names = {"PERIODIC_1", "PERIODIC_1_DONOR", "PERIODIC_2", "PERIODIC_2_DONOR"};
            const std::array<std::array<cgsize_t, 4>, 4> edges = {{{1, 4, 4, 7}, {3, 6, 6, 9},
                                                               {1, 2, 2, 3}, {7, 8, 8, 9}}};
            for (int i = 0; i < 4; i++)
            {
                const cgsize_t start = 5 + 2 * i;
                CheckCGNS(cg_section_write(file, base, zone, names[i], BAR_2, start, start + 1, 0, edges[i].data(), &section));
                cgsize_t range[2] = {start, start + 1};
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
