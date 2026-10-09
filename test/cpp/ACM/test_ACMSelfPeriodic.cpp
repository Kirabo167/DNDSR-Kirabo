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
#include <stdexcept>
#include <string>

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
            inlet.value = cfg.initialState;
            outlet.name = "Right";
            outlet.type = ACM::BoundaryType::BCOut;
            cfg.boundaryConditions = {inlet, outlet};
        }
        cfg.acmSettings.farFieldValue = cfg.initialState;
        cfg.turbulenceSettings.secondOrderReconstruction = secondOrder;
        cfg.turbulenceSettings.transportTimeScale = 1;
        cfg.timeMarchSettings.integrator = integrator;
        cfg.timeMarchSettings.nSteps = steps;
        cfg.timeMarchSettings.pseudoTimeStep = 0.01;
        cfg.timeMarchSettings.physicalTimeStep = dt;
        cfg.timeMarchSettings.maxImplicitIterations = 500;
        cfg.timeMarchSettings.implicitTolerance = 1e-12;
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
    configuration.boundaryValue = configuration.initialState;
    configuration.acmSettings.farFieldValue = configuration.initialState;
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
    configuration.boundaryValue = configuration.initialState;
    configuration.acmSettings.farFieldValue = configuration.initialState;
    configuration.reconstructionSettings.type =
        ACMVariable::ReconstructionType::FirstOrder;
    configuration.reconstructionSettings.enableLimiter = false;
    configuration.Validate();

    ACMVariable::ACMSolver<ACMVariable::ACMModel::VariableDensity2D> solver(
        mpi, configuration);
    VerifySelfPeriodicImplicitOperator<5>(solver);
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
            cfg.acmSettings.farFieldValue = cfg.initialState;
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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
