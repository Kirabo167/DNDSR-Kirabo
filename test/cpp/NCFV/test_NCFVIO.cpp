/**
 * @file test_NCFVIO.cpp
 * @brief MPI regression tests for NCFV boundary, initial-field and I/O paths.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "NCFV/NCFVSolver.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace
{
    using namespace DNDS;
    using namespace DNDS::NCFV;

    MPIInfo gMPI;

    std::filesystem::path ProjectRoot()
    {
        return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    }

    std::filesystem::path TestDirectory()
    {
        return std::filesystem::temp_directory_path() /
               fmt::format("dndsr_ncfv_io_np{}", gMPI.size);
    }

    Configuration MakeConfiguration()
    {
        Configuration configuration;
        configuration.dimension = 3;
        configuration.mesh.meshFile =
            (ProjectRoot() / "data/mesh/ACMVariable_verify3D.cgns").string();
        configuration.algorithm.mode = IntegrationMode::EfficientDifferential;
        configuration.algorithm.quadratureOrder = 4;
        configuration.algorithm.retainMicroGeometry = false;
        configuration.physics.initialPrimitive = {1.0, 0.1, 0.0, 0.0, 1.0};
        configuration.physics.farFieldPrimitive = {1.0, 0.1, 0.0, 0.0, 1.0};
        configuration.physics.requireBoundaryZoneCoverage = true;

        BoundaryZoneSettings farField;
        farField.name = "FAR";
        farField.mode = BoundaryMode::FarField;
        farField.primitive = configuration.physics.farFieldPrimitive;
        configuration.physics.boundaryZones = {farField};

        configuration.physics.viscous.enabled = true;
        configuration.physics.viscous.model = ViscosityModel::Constant;
        configuration.physics.viscous.dynamicViscosity = 1e-3;
        configuration.physics.viscous.gasConstant = 1.0;
        configuration.physics.viscous.prandtlNumber = 0.72;
        configuration.time.useCFLTimeStep = true;
        configuration.time.useLocalTimeStep = true;
        configuration.time.cfl = 0.05;
        configuration.time.minimumTimeStep = 1e-10;
        configuration.time.maximumTimeStep = 1.0;
        configuration.time.reportInterval = 1;
        return configuration;
    }

    void PrepareDirectory(const std::filesystem::path &directory)
    {
        if (gMPI.rank == 0)
        {
            std::filesystem::remove_all(directory);
            std::filesystem::create_directories(directory);
        }
        MPI_Barrier(gMPI.comm);
    }

    void WriteInitialField(const std::filesystem::path &fileName)
    {
        if (gMPI.rank == 0)
        {
            std::ofstream output(fileName);
            REQUIRE(output.good());
            output << "global_node,rho,u,v,w,p\n";
            for (DNDS::index globalNode = 0; globalNode < 125; globalNode++)
            {
                const DNDS::real density =
                    1.0 + 1e-4 * static_cast<DNDS::real>(globalNode);
                const DNDS::real pressure =
                    1.0 + 5e-5 * static_cast<DNDS::real>(globalNode);
                output << globalNode << ',' << density
                       << ",0.1,0,0," << pressure << '\n';
            }
        }
        MPI_Barrier(gMPI.comm);
    }

    void CheckInitialField(const Solver<3> &solver)
    {
        DNDS::real localMaximumError = 0;
        for (DNDS::index iNode = 0; iNode < solver.Mesh()->NumNode(); iNode++)
        {
            const DNDS::index globalNode =
                solver.Mesh()->node2nodeOrig(iNode, 0);
            const DNDS::real expectedDensity =
                1.0 + 1e-4 * static_cast<DNDS::real>(globalNode);
            const DNDS::real expectedPressure =
                1.0 + 5e-5 * static_cast<DNDS::real>(globalNode);
            const auto state = solver.StateField()[iNode];
            const DNDS::real pressure = 0.4 *
                                        (state(4, 0) -
                                         0.5 * (state(1, 0) * state(1, 0) +
                                                state(2, 0) * state(2, 0) +
                                                state(3, 0) * state(3, 0)) /
                                             state(0, 0));
            localMaximumError = std::max(
                localMaximumError,
                std::abs(state(0, 0) - expectedDensity));
            localMaximumError = std::max(
                localMaximumError,
                std::abs(state(1, 0) - 0.1 * expectedDensity));
            localMaximumError = std::max(
                localMaximumError,
                std::abs(pressure - expectedPressure));
        }
        DNDS::real globalMaximumError = 0;
        MPI_Allreduce(&localMaximumError, &globalMaximumError, 1,
                      DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        CHECK(globalMaximumError < 2e-12);
    }

    void CheckOutputFiles(const std::filesystem::path &directory)
    {
        MPI_Barrier(gMPI.comm);
        if (gMPI.rank != 0)
            return;
        CHECK(std::filesystem::is_regular_file(
            directory / "solution_00000001.pvtu"));
        CHECK(std::filesystem::is_directory(
            directory / "solution_00000001.vtu.dir"));
        CHECK(std::filesystem::is_regular_file(
            directory / "series.pvtu.series"));
        CHECK(std::filesystem::is_regular_file(
            directory / "solution.resolved.json"));
        CHECK(std::filesystem::is_regular_file(
            directory / "restart_00000001.dnds.h5"));
    }

    void VerifyRestartState(
        const Solver<3> &written,
        const Solver<3> &restarted)
    {
        DNDS::real localMaximumError = 0;
        for (DNDS::index iNode = 0; iNode < written.Mesh()->NumNode(); iNode++)
            localMaximumError = std::max(
                localMaximumError,
                (written.StateField()[iNode] - restarted.StateField()[iNode])
                    .cwiseAbs()
                    .maxCoeff());
        DNDS::real globalMaximumError = 0;
        MPI_Allreduce(&localMaximumError, &globalMaximumError, 1,
                      DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
        CHECK(globalMaximumError < 2e-13);
    }
}

TEST_CASE("NCFV node field, VTK and HDF5 restart form an MPI I/O cycle")
{
    const std::filesystem::path directory = TestDirectory();
    PrepareDirectory(directory);
    const std::filesystem::path initialFile = directory / "initial.csv";
    WriteInitialField(initialFile);

    Configuration writeConfiguration = MakeConfiguration();
    writeConfiguration.initialField.nodeFile = initialFile.string();
    writeConfiguration.initialField.nodeFileVariables =
        InitialFieldVariables::Primitive;
    writeConfiguration.initialField.requireCompleteNodeFile = true;
    writeConfiguration.time.iterations = 1;
    writeConfiguration.io.writeVTK = true;
    writeConfiguration.io.writeFinal = true;
    writeConfiguration.io.outputPrefix = (directory / "solution").string();
    writeConfiguration.io.vtkSeriesName = (directory / "series").string();
    writeConfiguration.io.writeFinalRestart = true;
    writeConfiguration.io.restartPrefix = (directory / "restart").string();
    writeConfiguration.io.restartSerializer.type = "H5";
    writeConfiguration.Validate();

    Solver<3> written(gMPI, writeConfiguration);
    written.Initialize();
    CheckInitialField(written);
    written.Run();
    CHECK(written.CurrentIteration() == 1);
    CHECK(written.SimulationTime() > 0);
    CheckOutputFiles(directory);

    Configuration readConfiguration = MakeConfiguration();
    readConfiguration.io.restartInput =
        (directory / "restart_00000001").string();
    readConfiguration.io.restartSerializer.type = "H5";
    readConfiguration.time.iterations = 0;
    readConfiguration.Validate();

    Solver<3> restarted(gMPI, readConfiguration);
    restarted.Initialize();
    CHECK(restarted.CurrentIteration() == written.CurrentIteration());
    CHECK(restarted.SimulationTime() ==
          doctest::Approx(written.SimulationTime()).epsilon(2e-14));
    VerifyRestartState(written, restarted);
    CHECK(std::isfinite(restarted.EvaluateResidual()));
}

TEST_CASE("NCFV named no-slip wall is imposed strongly and contributes viscous flux")
{
    Configuration configuration = MakeConfiguration();
    configuration.physics.boundaryZones.front().mode =
        BoundaryMode::NoSlipAdiabaticWall;
    configuration.physics.boundaryZones.front().wallVelocity = {0.0, 0.0, 0.0};
    configuration.physics.boundaryZones.front().strongState = true;
    configuration.time.iterations = 0;
    configuration.Validate();

    Solver<3> solver(gMPI, configuration);
    solver.Initialize();

    DNDS::index localBoundaryNodes = 0;
    DNDS::index localInteriorNodes = 0;
    DNDS::real localMaximumWallMomentum = 0;
    for (DNDS::index iNode = 0; iNode < solver.Mesh()->NumNode(); iNode++)
    {
        const bool isBoundary =
            !solver.Geometry().NodeVolume(iNode).boundaryPieces.empty();
        const auto state = solver.StateField()[iNode];
        const DNDS::real momentum =
            std::sqrt(state(1, 0) * state(1, 0) +
                      state(2, 0) * state(2, 0) +
                      state(3, 0) * state(3, 0));
        if (isBoundary)
        {
            localBoundaryNodes++;
            localMaximumWallMomentum =
                std::max(localMaximumWallMomentum, momentum);
            for (const auto &piece :
                 solver.Geometry().NodeVolume(iNode).boundaryPieces)
            {
                CHECK(solver.Boundaries().Name(piece.zone) == "FAR");
                CHECK(solver.Boundaries().Get(piece.zone).mode ==
                      BoundaryMode::NoSlipAdiabaticWall);
            }
        }
        else
        {
            localInteriorNodes++;
            CHECK(momentum > 0);
        }
    }

    DNDS::index localCounts[2]{localBoundaryNodes, localInteriorNodes};
    DNDS::index globalCounts[2]{};
    DNDS::real globalMaximumWallMomentum = 0;
    MPI_Allreduce(localCounts, globalCounts, 2,
                  DNDS_MPI_INDEX, MPI_SUM, gMPI.comm);
    MPI_Allreduce(&localMaximumWallMomentum, &globalMaximumWallMomentum, 1,
                  DNDS_MPI_REAL, MPI_MAX, gMPI.comm);
    CHECK(globalCounts[0] > 0);
    CHECK(globalCounts[1] > 0);
    CHECK(globalMaximumWallMomentum < 1e-14);

    const DNDS::real residual = solver.EvaluateResidual();
    CHECK(std::isfinite(residual));
    CHECK(residual > 1e-8);
    for (DNDS::index iNode = 0; iNode < solver.Mesh()->NumNode(); iNode++)
    {
        CHECK(std::isfinite(solver.LocalTimeStep(iNode)));
        CHECK(solver.LocalTimeStep(iNode) > 0);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    gMPI.setWorld();
    doctest::Context context(argc, argv);
    const int localResult = context.run();
    int globalResult = 0;
    MPI_Allreduce(&localResult, &globalResult, 1, MPI_INT, MPI_MAX, gMPI.comm);
    MPI_Finalize();
    return globalResult;
}
