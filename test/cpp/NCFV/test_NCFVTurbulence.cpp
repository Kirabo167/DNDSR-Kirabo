#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "NCFV/NCFVSolver.hpp"

#include <filesystem>
#include <fstream>

namespace
{
    using namespace DNDS;
    using namespace DNDS::NCFV;

    MPIInfo mpi;

    template <int dimension>
    Configuration MakeConfiguration(const std::string &model, IntegrationMode mode)
    {
        const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
        Configuration c;
        c.dimension = dimension;
        c.algorithm.mode = mode;
        c.reconstruction.enableLimiter = false;
        c.mesh.meshFile = (root / "data/mesh" / fmt::format("ACMVariable_verify{}D.cgns", dimension)).string();
        c.physics.viscous.enabled = true;
        c.physics.viscous.model = ViscosityModel::Constant;
        c.physics.viscous.dynamicViscosity = 1e-4;
        c.physics.turbulence.model = model;
        c.physics.turbulence.viscosityReference = .02;
        c.physics.boundaryMode = BoundaryMode::Extrapolation;
        c.physics.initialPrimitive = {1.3, .2, 0};
        if constexpr (dimension == 3)
            c.physics.initialPrimitive.push_back(0);
        c.physics.initialPrimitive.push_back(1.1);
        c.physics.initialPrimitive.push_back(.001);
        if (model != "SA")
            c.physics.initialPrimitive.push_back(model == "RealizableKE" ? .005 : 2.);
        c.physics.farFieldPrimitive = c.physics.initialPrimitive;
        c.solver.fieldNVariables = static_cast<int>(c.physics.initialPrimitive.size());
        c.time.iterations = 0;
        c.io.writeVTK = c.io.writeResolvedConfiguration = false;
        return c;
    }

    template <int dimension, int nTurbulence>
    void CheckUniformRHS(const Configuration &c)
    {
        CAPTURE(dimension);
        CAPTURE(c.physics.turbulence.model);
        CAPTURE(static_cast<int>(c.algorithm.mode));
        Solver<dimension, nTurbulence> solver(mpi, c);
        solver.Initialize();
        CHECK(std::isfinite(solver.EvaluateResidual()));

        Physics::RANS::Point point;
        const auto &primitive = c.physics.initialPrimitive;
        const DNDS::real rho = primitive[0];
        point.state[0] = rho;
        point.state[1] = rho * primitive[1];
        point.state[4] = primitive[dimension + 1] / (c.physics.gamma - 1) + .5 * rho * sqr(primitive[1]);
        point.state[5] = rho * primitive[dimension + 2];
        if constexpr (nTurbulence == 1)
            point.state[5] /= c.physics.turbulence.viscosityReference;
        else
            point.state[6] = rho * primitive[dimension + 3];
        point.molecularViscosity = c.physics.viscous.dynamicViscosity;
        point.wallDistance = std::pow(veryLargeReal, .25);
        point.cellLength = 1;
        const auto source = c.physics.MakeTurbulenceModel().Source(point);
        for (DNDS::index node = 0; node < solver.Mesh()->NumNode(); ++node)
        {
            const auto rhs = solver.ResidualField()[node];
            CHECK(rhs.template head<dimension + 2>().norm() < 1e-11);
            for (int j = 0; j < nTurbulence; ++j)
            {
                CHECK(solver.StateField()[node][dimension + 2 + j] == doctest::Approx(point.state[5 + j]));
                CHECK(std::abs(rhs[dimension + 2 + j] - source[j]) < 1e-11);
            }
        }
    }

    template <int dimension>
    void CheckCompressibleSAS(IntegrationMode mode)
    {
        CAPTURE(dimension);
        CAPTURE(static_cast<int>(mode));
        constexpr int axis = dimension - 1;
        constexpr int nVariables = dimension + 4;
        constexpr real k = .025, omega = 1.2;
        auto c = MakeConfiguration<dimension>("KOmegaSST", mode);
        c.time.useCFLTimeStep = true;
        c.time.useLocalTimeStep = true;
        Solver<dimension, 2> solver(mpi, c);
        solver.Initialize();
        const auto &halo = solver.NodeCommunication();
        NodeStatePair means, rhs;
        means.InitPair("NCFV.testSASMeans", mpi);
        means.father->Resize(solver.Mesh()->NumNode(), nVariables, 1);
        means.son->Resize(halo.NumNodeGhost(), nVariables, 1);
        means.BorrowSetup(halo.Layout());
        means.trans.initPersistentPull();
        CFV::BuildUDofOnMesh(rhs, "NCFV.testSASRHS", mpi, solver.Mesh(), nVariables, false, false, Geom::MeshLoc::Node);

        for (const auto &volume : solver.Geometry().NodeVolumes())
        {
            const auto node = volume.node;
            const auto &moment = volume.moments;
            const real anchor = halo.Coordinate(node)[axis];
            const real meanX = anchor + moment.first[axis] / moment.measure;
            const real meanX2 = sqr(anchor) +
                (2 * anchor * moment.first[axis] + moment.second(axis, axis)) / moment.measure;
            const real rho = 1.3 + .45 * meanX + .22 * meanX2;
            means[node].setZero();
            means[node][0] = rho;
            means[node][1] = .25 + .04 * meanX + .11 * meanX2;
            means[node][dimension + 1] = 3;
            means[node][dimension + 2] = rho * k;
            means[node][dimension + 3] = rho * omega;
        }

        std::vector<Eigen::Vector<real, nVariables>> sstRhs(solver.Mesh()->NumNode());
        std::vector<real> sstTimeStep(solver.Mesh()->NumNode());
        for (const char *model : {"KOmegaSST", "KOmegaSSTSAS"})
        {
            c.physics.turbulence.model = model;
            SpatialOperator<dimension, 2> spatial(mpi, solver.Mesh(), solver.EdgeTopology(), solver.Geometry(),
                                                  solver.ReconstructionData(), solver.Boundaries(), mode,
                                                  c.reconstruction, c.physics, c.time, halo);
            spatial.Initialize();
            spatial.EvaluateRHS(means, rhs);
            real localSASVolumeIntegral = 0;
            for (const auto &volume : solver.Geometry().NodeVolumes())
            {
                const auto node = volume.node;
                if (!c.physics.UsesSSTScaleAdaptivity())
                {
                    sstRhs[node] = rhs[node];
                    sstTimeStep[node] = spatial.LocalTimeStep(node);
                    continue;
                }

                const real minimumLength = .11 * std::sqrt(.41 * 3.51 / (.856 * sqr(.41) / .3)) * volume.lengthScale;
                const real modelLength = std::sqrt(k) / (std::pow(.09, .25) * omega);
                real expectedSource = 0, expectedRate = 0;
                for (const auto &sample : volume.volumeQuadrature)
                {
                    const real x = sample.coordinate[axis];
                    const real rho = 1.3 + .45 * x + .22 * sqr(x);
                    const real rhoGradient = .45 + .44 * x;
                    const real momentum = .25 + .04 * x + .11 * sqr(x);
                    const real velocity = momentum / rho;
                    const real velocityGradient = (.04 + .22 * x - velocity * rhoGradient) / rho;
                    const real velocityLaplacian = (.22 - .44 * velocity - 2 * rhoGradient * velocityGradient) / rho;
                    const real rate = std::min(std::abs(velocityLaplacian) / .41, std::abs(velocityGradient) / minimumLength);
                    const real sas = 3.51 * .41 * sqr(modelLength * rate);
                    expectedSource += sample.weight * rho * sas;
                    expectedRate = std::max(expectedRate, 2 * sas / omega);
                }
                const Eigen::Vector<real, nVariables> change = rhs[node] - sstRhs[node];
                CHECK(change.template head<nVariables - 1>().norm() < 1e-12);
                CHECK(std::abs(change[nVariables - 1] - expectedSource / volume.moments.measure) < 1e-11);
                const real rateChange = c.time.cfl * (1 / spatial.LocalTimeStep(node) - 1 / sstTimeStep[node]);
                CHECK(std::abs(rateChange - expectedRate) < 1e-10);
                localSASVolumeIntegral += change[nVariables - 1] * volume.moments.measure;
            }
            if (c.physics.UsesSSTScaleAdaptivity())
            {
                real sasVolumeIntegral = 0;
                MPI_Allreduce(&localSASVolumeIntegral, &sasVolumeIntegral, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                CHECK(sasVolumeIntegral > 1e-4);
            }
        }
    }
}

TEST_CASE("NCFV assembles all RANS equation layouts with both spatial integration methods")
{
    for (const auto mode : {IntegrationMode::TraditionalQuadrature, IntegrationMode::EfficientDifferential})
    {
        CheckUniformRHS<2, 1>(MakeConfiguration<2>("SA", mode));
        CheckUniformRHS<3, 1>(MakeConfiguration<3>("SA", mode));
        for (const auto *model : {"KOmegaWilcox", "KOmegaSST", "KOmegaSSTSAS", "RealizableKE"})
        {
            CheckUniformRHS<2, 2>(MakeConfiguration<2>(model, mode));
            CheckUniformRHS<3, 2>(MakeConfiguration<3>(model, mode));
        }
    }
}

TEST_CASE("NCFV SST-SAS integrates variable-density velocity curvature and limits the CFL step")
{
    for (const auto mode : {IntegrationMode::TraditionalQuadrature, IntegrationMode::EfficientDifferential})
    {
        CheckCompressibleSAS<2>(mode);
        CheckCompressibleSAS<3>(mode);
    }
}

TEST_CASE("NCFV integrates the quadratic Wilcox k sink from linearly varying turbulence means")
{
    constexpr DNDS::real kSlope = .004, omegaSlope = .1;
    for (const auto mode : {IntegrationMode::TraditionalQuadrature, IntegrationMode::EfficientDifferential})
    {
        auto c = MakeConfiguration<2>("KOmegaWilcox", mode);
        c.physics.boundaryMode = BoundaryMode::SlipWall;
        c.physics.initialPrimitive[1] = 0;
        Solver<2, 2> solver(mpi, c);
        solver.Initialize();
        const auto &halo = solver.NodeCommunication();
        SpatialOperator<2, 2> spatial(mpi, solver.Mesh(), solver.EdgeTopology(), solver.Geometry(),
                                      solver.ReconstructionData(), solver.Boundaries(), mode,
                                      c.reconstruction, c.physics, c.time, halo);
        spatial.Initialize();
        NodeStatePair means, rhs;
        means.InitPair("NCFV.testRANSMeans", mpi);
        means.father->Resize(solver.Mesh()->NumNode(), 6, 1);
        means.son->Resize(halo.NumNodeGhost(), 6, 1);
        means.BorrowSetup(halo.Layout());
        means.trans.initPersistentPull();
        CFV::BuildUDofOnMesh(rhs, "NCFV.testRANSRHS", mpi, solver.Mesh(), 6, false, false, Geom::MeshLoc::Node);

        DNDS::real integratedSink = 0;
        for (const auto &volume : solver.Geometry().NodeVolumes())
        {
            const auto node = volume.node;
            const auto &moment = volume.moments;
            const DNDS::real x = halo.Coordinate(node).x();
            const DNDS::real k = .02 + kSlope * x, omega = 1.5 + omegaSlope * x;
            means[node] = solver.StateField()[node];
            const DNDS::real rho = means[node][0];
            means[node][4] = rho * (k + kSlope * moment.first.x() / moment.measure);
            means[node][5] = rho * (omega + omegaSlope * moment.first.x() / moment.measure);
            integratedSink -= .09 * rho * (k * omega * moment.measure +
                (k * omegaSlope + omega * kSlope) * moment.first.x() + kSlope * omegaSlope * moment.second(0, 0));
        }
        spatial.EvaluateRHS(means, rhs);
        DNDS::real integratedRHS = 0;
        for (const auto &volume : solver.Geometry().NodeVolumes())
            integratedRHS += volume.moments.measure * rhs[volume.node][4];
        const DNDS::real local[]{integratedRHS, integratedSink};
        DNDS::real global[2]{};
        MPI_Allreduce(local, global, 2, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        CHECK(std::abs(global[0] - global[1]) < 1e-12);
    }
}

TEST_CASE("NCFV pressure outlet gets its turbulence reservoir only on backflow")
{
    auto c = MakeConfiguration<3>("KOmegaSST", IntegrationMode::EfficientDifferential);
    c.algorithm.macroFaceBoundary = true;
    BoundaryZoneSettings far;
    far.name = "FAR";
    far.mode = BoundaryMode::PressureOutlet;
    far.staticPressure = c.physics.initialPrimitive[4];
    c.physics.boundaryZones = {far};
    Solver<3, 2> uniform(mpi, c);
    uniform.Initialize();
    uniform.EvaluateResidual();

    c.physics.farFieldPrimitive[5] *= 2;
    Solver<3, 2> supplied(mpi, c);
    supplied.Initialize();
    supplied.EvaluateResidual();

    DNDS::index localIncoming = 0, globalIncoming = 0;
    for (const auto &volume : supplied.Geometry().NodeVolumes())
    {
        const bool backflow = std::any_of(volume.boundaryPieces.begin(), volume.boundaryPieces.end(),
                                         [](const auto &piece) { return piece.vectorMeasure.x() < -1e-12 * piece.measure; });
        const DNDS::real change = supplied.ResidualField()[volume.node][5] - uniform.ResidualField()[volume.node][5];
        if (backflow)
        {
            CHECK(change > 1e-9);
            ++localIncoming;
        }
        else
            CHECK(std::abs(change) < 1e-12);
    }
    MPI_Allreduce(&localIncoming, &globalIncoming, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
    CHECK(globalIncoming > 0);
}

TEST_CASE("NCFV no-slip SST evolves turbulent means and preserves the restart model")
{
    auto c = MakeConfiguration<3>("KOmegaSST", IntegrationMode::EfficientDifferential);
    c.physics.boundaryMode = BoundaryMode::NoSlipIsothermalWall;
    c.physics.initialPrimitive[1] = 0;
    c.physics.initialPrimitive[4] = c.physics.initialPrimitive[0]; // Match the wall temperature.
    c.physics.farFieldPrimitive = c.physics.initialPrimitive;
    c.time.iterations = 2;
    c.time.useCFLTimeStep = true;
    c.time.useLocalTimeStep = false;
    c.time.cfl = .05;
    const auto directory = std::filesystem::temp_directory_path() / fmt::format("dndsr_ncfv_rans_np{}", mpi.size);
    if (mpi.rank == 0)
        std::filesystem::create_directories(directory);
    MPI_Barrier(mpi.comm);
    c.io.writeVTK = true;
    c.io.writeFinal = true;
    c.io.outputPrefix = (directory / "solution").string();
    c.io.vtkSeriesName = (directory / "series").string();
    c.io.writeFinalRestart = true;
    c.io.restartPrefix = (directory / "restart").string();
    c.io.restartSerializer.type = "H5";
    Solver<3, 2> written(mpi, c);
    written.Initialize();
    written.EvaluateResidual();
    const auto &initial = c.physics.initialPrimitive;
    for (const auto &volume : written.Geometry().NodeVolumes())
    {
        DNDS::real expected = -.09 * initial[0] * initial[5] * initial[6];
        for (const auto &piece : volume.boundaryPieces)
        {
            const DNDS::real distance = -piece.vectorMeasure.dot(volume.moments.first) /
                                        (piece.measure * volume.moments.measure);
            CHECK(distance > 0);
            expected -= c.physics.viscous.dynamicViscosity * initial[5] * piece.measure /
                        (distance * volume.moments.measure);
        }
        CHECK(std::abs(written.ResidualField()[volume.node][5] - expected) < 1e-12);
    }
    written.Run();
    CHECK(written.CurrentIteration() == 2);
    CHECK(written.SimulationTime() > 0);
    for (const auto &volume : written.Geometry().NodeVolumes())
    {
        const auto state = written.StateField()[volume.node];
        CHECK(state[5] > 0);
        CHECK(state[6] > 0);
    }
    if (mpi.rank == 0)
    {
        std::ifstream vtk(directory / "solution_00000002.pvtu");
        const std::string metadata(std::istreambuf_iterator<char>{vtk}, {});
        CHECK(metadata.find("TurbulentKineticEnergy") != std::string::npos);
        CHECK(metadata.find("Omega") != std::string::npos);
    }

    c.io.writeVTK = c.io.writeFinalRestart = false;
    c.io.restartInput = (directory / "restart_00000002").string();
    Solver<3, 2> read(mpi, c);
    read.Initialize();
    CHECK(read.CurrentIteration() == written.CurrentIteration());
    for (DNDS::index node = 0; node < read.Mesh()->NumNode(); ++node)
        CHECK((read.StateField()[node] - written.StateField()[node]).norm() < 1e-13);
    c.physics.turbulence.model = "KOmegaWilcox";
    Solver<3, 2> wrongModel(mpi, c);
    CHECK_THROWS_WITH_AS(wrongModel.Initialize(), doctest::Contains("restart turbulence model differs"), std::exception);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    const int local = doctest::Context(argc, argv).run();
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, mpi.comm);
    MPI_Finalize();
    return global;
}
