/**
 * @file taylor_green_2d_rhs_probe.cpp
 * @brief Compare NCFV momentum RHS with the exact decaying 2-D Taylor--Green field.
 *
 * The 2-D incompressible field is embedded in the 3-D periodic mesh because NCFV
 * currently supports only a fully periodic 3-D quotient.  At every sampled time
 * the compressible momentum and continuity operators agree with the incompressible
 * equations for rho=1 and divergence-free velocity.  The compressible energy
 * equation does not share this exact solution; this probe does not time-march.
 */
#include "NCFV/NCFVSolver.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    constexpr double Nu = 0.01;
    constexpr double BasePressure = 100.0;

    std::string ExactNumber(double value)
    {
        std::ostringstream output;
        output << std::setprecision(17) << value;
        return output.str();
    }

    DNDS::NCFV::Configuration MakeConfiguration(
        const std::string &meshFile, DNDS::NCFV::IntegrationMode mode,
        double time, int maximumRings)
    {
        using namespace DNDS::NCFV;
        Configuration configuration;
        configuration.dimension = 3;
        configuration.mesh.meshFile = meshFile;
        configuration.mesh.periodicLengths = {2 * DNDS::pi, 2 * DNDS::pi, 2 * DNDS::pi};
        configuration.mesh.periodicBoundaryPairs = {
            "bc-2", "bc-2-1", "bc-3", "bc-3-1", "bc-4", "bc-4-1"};
        configuration.algorithm.mode = mode;
        configuration.algorithm.quadratureOrder = 4;
        configuration.algorithm.surfaceQuadratureOrder = 3;
        configuration.algorithm.retainMicroGeometry = false;
        configuration.reconstruction.method = ReconstructionMethod::SVDLeastSquares;
        configuration.reconstruction.enableLimiter = false;
        configuration.reconstruction.maximumStencilRings = maximumRings;
        configuration.physics.gamma = 1.4;
        configuration.physics.initialPrimitive = {1, 0, 0, 0, BasePressure};
        configuration.physics.farFieldPrimitive = configuration.physics.initialPrimitive;
        configuration.physics.requireBoundaryZoneCoverage = true;
        for (const auto &name : configuration.mesh.periodicBoundaryPairs)
        {
            BoundaryZoneSettings boundary;
            boundary.name = name;
            boundary.mode = BoundaryMode::Periodic;
            configuration.physics.boundaryZones.push_back(boundary);
        }
        configuration.physics.viscous.enabled = true;
        configuration.physics.viscous.model = ViscosityModel::Constant;
        configuration.physics.viscous.dynamicViscosity = Nu;
        configuration.physics.viscous.gasConstant = 1;
        configuration.physics.viscous.prandtlNumber = 0.71;
        const double amplitude = std::exp(-2 * Nu * time);
        const double pressureAmplitude = 0.25 * amplitude * amplitude;
        configuration.initialField.expressions = {{{
            "inRegion := 1;",
            "UPrim[0] := 1;",
            "UPrim[1] := " + ExactNumber(amplitude) + " * sin(x[0]) * cos(x[1]);",
            "UPrim[2] := -" + ExactNumber(amplitude) + " * cos(x[0]) * sin(x[1]);",
            "UPrim[3] := 0;",
            "UPrim[4] := " + ExactNumber(BasePressure) + " + " +
                ExactNumber(pressureAmplitude) + " * (cos(2*x[0]) + cos(2*x[1]));",
            "0;"}}};
        configuration.time.iterations = 0;
        configuration.time.useCFLTimeStep = false;
        configuration.time.useLocalTimeStep = false;
        configuration.time.timeStep = 0.001;
        configuration.io.writeVTK = false;
        configuration.io.writeInitial = false;
        configuration.io.writeFinal = false;
        configuration.io.writeResolvedConfiguration = false;
        configuration.Validate();
        return configuration;
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    int status = 0;
    try
    {
        DNDS_check_throw_info(
            argc == 4 || argc == 5,
            "usage: taylor_green_2d_rhs_probe mesh.cgns Efficient|Traditional time [maximum-rings]");
        const std::string modeName = argv[2];
        DNDS_check_throw_info(modeName == "Efficient" || modeName == "Traditional",
                              "mode must be Efficient or Traditional");
        const double time = std::stod(argv[3]);
        const int maximumRings = argc == 5 ? std::stoi(argv[4]) : 4;
        DNDS_check_throw_info(std::isfinite(time) && time >= 0,
                              "time must be finite and nonnegative");
        DNDS_check_throw_info(maximumRings >= 1 && maximumRings <= 8,
                              "maximum-rings must be between 1 and 8");
        const auto mode = modeName == "Efficient"
            ? DNDS::NCFV::IntegrationMode::EfficientDifferential
            : DNDS::NCFV::IntegrationMode::TraditionalQuadrature;
        DNDS::NCFV::Solver<3> solver(
            mpi, MakeConfiguration(argv[1], mode, time, maximumRings));
        solver.Initialize();
        if (mpi.rank == 0)
            std::cout << "TGV2D_RHS_CONFIG maximum_rings=" << maximumRings << '\n';
        solver.EvaluateResidual();

        double local[5]{};
        double localMaximum = 0;
        const double amplitude = std::exp(-2 * Nu * time);
        for (DNDS::index iNode = 0; iNode < solver.Mesh()->NumNode(); iNode++)
        {
            const auto &point = solver.Mesh()->coords[iNode];
            const double u = amplitude * std::sin(point(0)) * std::cos(point(1));
            const double v = -amplitude * std::cos(point(0)) * std::sin(point(1));
            const auto &rhs = solver.ResidualField()[iNode];
            const double errorSquared =
                std::pow(rhs(1) + 2 * Nu * u, 2) +
                std::pow(rhs(2) + 2 * Nu * v, 2) +
                std::pow(rhs(3), 2);
            const double volume = solver.Geometry().NodeVolume(iNode).moments.measure;
            local[0] += volume;
            local[1] += volume * errorSquared;
            local[2] += volume * 4 * Nu * Nu * (u * u + v * v);
            local[3] += volume * rhs(0) * rhs(0);
            local[4] += volume * std::sqrt(errorSquared);
            localMaximum = std::max(localMaximum, std::sqrt(errorSquared));
        }
        double global[5]{};
        double globalMaximum = 0;
        MPI_Allreduce(local, global, 5, MPI_DOUBLE, MPI_SUM, mpi.comm);
        MPI_Allreduce(&localMaximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(global[0] > 0 && global[2] > 0,
                              "invalid Taylor--Green norm denominator");
        const DNDS::index cells = solver.Mesh()->NumCellGlobal();
        if (mpi.rank == 0)
        {
            std::cout << std::setprecision(17)
                      << "TGV2D_RHS_RESULT " << modeName << ' ' << time << ' '
                      << cells << ' '
                      << std::sqrt(global[1] / global[0]) << ' '
                      << std::sqrt(global[1] / global[2]) << ' '
                      << global[4] / global[0] << ' '
                      << globalMaximum << ' '
                      << std::sqrt(global[3] / global[0]) << '\n';
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "TGV2D_RHS_ERROR rank=" << mpi.rank << ": "
                  << error.what() << '\n';
        status = 1;
    }
    int globalStatus = 0;
    MPI_Allreduce(&status, &globalStatus, 1, MPI_INT, MPI_MAX, mpi.comm);
    DNDS::MPI::Finalize();
    return globalStatus;
}
