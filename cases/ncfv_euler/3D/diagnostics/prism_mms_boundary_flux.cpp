/**
 * @brief Isolate the NCFV characteristic flux on each boundary dual of the unit-cube prism MMS.
 * Build with compile_reconstruction_probe.py --source prism_mms_boundary_flux.cpp --output /tmp/boundary_flux.
 * Run with mpiexec -n 4 /tmp/boundary_flux [--wall-outlet] case.json result.csv mesh12.cgns mesh16.cgns ...
 * --wall-outlet uses a source-free frozen smooth snapshot with impermeable y/z walls and p=1.35 at x=1.
 * It checks face-interior duals so that wall and outlet residuals are measured independently.
 * All integrals use the outward sign. Each CSV error is an area-weighted RMS of boundary-dual flux densities.
 */
#include "NCFV/NCFVSolver.hpp"
#include "Geom/Quadrature.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace
{
    using namespace DNDS::NCFV;
    using DNDS::real;
    using MMS = ExponentialMMS;
    using State = Eigen::Vector<real, 5>;
    using Quadrature = DNDS::Geom::Elem::Quadrature;
    constexpr real time = .5;
    constexpr real outletPressure = 1.35;

    State WallOutletPrimitive(const Vector3 &point)
    {
        const auto x = point.x(), y = point.y(), z = point.z();
        const real e = std::exp(.65 * x + .45 * y + .55 * z);
        return {1 + .15 * e, .22 + .07 * e, .09 * (1 + e) * y * (1 - y),
                .06 * (1 + e) * z * (1 - z), outletPressure + .18 * (1 - x) * e};
    }

    State WallOutletState(const Vector3 &point)
    {
        const State w = WallOutletPrimitive(point);
        const Vector3 u = w.segment<3>(1);
        State state;
        state << w[0], w[0] * u, w[4] / .4 + .5 * w[0] * u.squaredNorm();
        return state;
    }

    State ExactMean(const NodeControlVolume &volume, const Quadrature &rule, bool wallOutlet)
    {
        State integral = State::Zero();
        for (const auto &tet : volume.microVolumes)
        {
            const auto &x = tet.points;
            for (int q = 0; q < rule.GetNumPoints(); ++q)
            {
                const auto [p, weight] = rule.GetQuadraturePointInfo(q);
                const Vector3 point = x[0] + p[0] * (x[1] - x[0]) + p[1] * (x[2] - x[0]) + p[2] * (x[3] - x[0]);
                const State state = wallOutlet ? WallOutletState(point) :
                    MMS::State(MMS::Powers(MMS::direction.dot(point) + MMS::frequency * time));
                integral += (6 * tet.measure * weight) * state;
            }
        }
        return integral / volume.moments.measure;
    }

    // Direct projected Euler flux of the manufactured primitive field.
    State ExactFlux(const Vector3 &point, const Vector3 &normal, bool wallOutlet)
    {
        State w;
        if (wallOutlet)
            w = WallOutletPrimitive(point);
        else
        {
            const real e = std::exp(MMS::direction.dot(point) + MMS::frequency * time);
            w << 1 + .1 * e, .2 + .1 * e, .1 + .2 * e, .3 + .3 * e, 1.3 + 1.2 * e;
        }
        const real density = w[0], pressure = w[4];
        const Vector3 velocity = w.segment<3>(1);
        const real vn = velocity.dot(normal);
        const real energy = pressure / .4 + .5 * density * velocity.squaredNorm();
        State flux;
        flux << density * vn, (density * vn) * velocity + pressure * normal, (energy + pressure) * vn;
        return flux;
    }

    void WriteLevel(const DNDS::MPIInfo &mpi, Configuration cfg, bool wallOutlet, std::ostream &csv)
    {
        using namespace DNDS::Geom::Elem;
        Solver<3> solver(mpi, cfg);
        solver.Initialize();
        const auto &mesh = solver.Mesh();
        const auto &halo = solver.NodeCommunication();
        SpatialOperator<3> spatial(mpi, mesh, solver.EdgeTopology(), solver.Geometry(), solver.ReconstructionData(),
                                   solver.Boundaries(), cfg.algorithm.mode, cfg.reconstruction, cfg.physics, cfg.time, halo);
        spatial.Initialize();
        NodeStatePair means, rhs;
        for (auto *field : {&means, &rhs})
        {
            field->InitPair(field == &means ? "Probe.Mean" : "Probe.Rhs", mpi);
            field->father->Resize(mesh->NumNode(), 5, 1);
            field->son->Resize(halo.NumNodeGhost(), 5, 1);
            field->BorrowSetup(halo.Layout());
            field->trans.initPersistentPull();
        }
        const Quadrature tetRule(Element{Tet4}, 6), faceRule(Element{Tri3}, 6);
        for (const auto &volume : solver.Geometry().NodeVolumes())
            means[volume.node] = ExactMean(volume, tetRule, wallOutlet);
        spatial.EvaluateRHS(means, rhs, false, false, time);

        // Recover the applied boundary integral from the assembled production residual:
        // Phi_b = V*(S_discrete - RHS) - sum_e sign_e*Phi_e.
        // In wall/outlet mode columns 0 and 1 correspond to walls and the pressure outlet.
        // Rows 0..4 hold squared flux errors, row 5 area, row 6 squared normal traction error.
        using Sums = Eigen::Matrix<real, 7, 2>;
        Sums local = Sums::Zero(), global;
        const auto n = std::lround(std::cbrt(static_cast<real>(mesh->NumNodeGlobal()))) - 1;
        Eigen::Vector2i counts = Eigen::Vector2i::Zero(), globalCounts;
        for (const auto &volume : solver.Geometry().NodeVolumes())
        {
            if (volume.boundaryPieces.empty())
                continue;
            int group = 0;
            if (wallOutlet)
            {
                const Vector3 x = halo.Coordinate(volume.node);
                const int planes = (x.array().abs() < 1e-12).count() + ((x.array() - 1).abs() < 1e-12).count();
                if (planes != 1 || std::abs(x.x()) < 1e-12)
                    continue;
                group = std::abs(x.x() - 1) < 1e-12 ? 1 : 0;
                ++counts[group];
            }
            State discreteSource = State::Zero();
            if (!wallOutlet)
                discreteSource = MMS::sourceCoefficients.transpose() * spatial.ManufacturedBasisMean(volume.node, time);
            State numerical = volume.moments.measure * (discreteSource - rhs[volume.node]);
            for (const auto &incidence : solver.EdgeTopology().Node2Edge(volume.node))
                numerical -= incidence.outwardSign * State(spatial.EdgeFluxes()[incidence.edge]);

            State reference = State::Zero();
            real area = 0;
            Vector3 faceNormal = Vector3::Zero();
            for (const auto &piece : volume.boundaryPieces)
            {
                const Vector3 &x = piece.points[0].coordinate;
                const Vector3 u = piece.points[1].coordinate - x, v = piece.points[2].coordinate - x;
                const real jacobian = u.cross(v).norm();
                Vector3 normal = u.cross(v) / jacobian;
                if (normal.dot(x - Vector3::Constant(.5)) < 0)
                    normal = -normal;
                DNDS_check_throw_info((.5 * jacobian * normal - piece.vectorMeasure).norm() < 1e-12 * jacobian,
                                      "Reference face orientation or area disagrees with the outward dual geometry");
                if (wallOutlet)
                {
                    const auto expected = group == 0 ? BoundaryMode::SlipWall : BoundaryMode::PressureOutlet;
                    DNDS_check_throw_info(solver.Boundaries().Get(piece.zone).mode == expected,
                                          "Selected surface does not have the expected physical boundary type");
                    faceNormal = normal;
                }
                area += .5 * jacobian;
                for (int q = 0; q < faceRule.GetNumPoints(); ++q)
                {
                    const auto [p, weight] = faceRule.GetQuadraturePointInfo(q);
                    reference += (weight * jacobian) * ExactFlux(x + p[0] * u + p[1] * v, normal, wallOutlet);
                }
            }
            const State error = numerical - reference;
            local.col(group).head<5>() += error.array().square().matrix() / area;
            local(5, group) += area;
            if (wallOutlet && group == 0)
            {
                const real errorPressure = faceNormal.dot(error.segment<3>(1));
                local(6, group) += errorPressure * errorPressure / area;
            }
        }
        MPI_Allreduce(local.data(), global.data(), static_cast<int>(local.size()), DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        if (wallOutlet)
        {
            MPI_Allreduce(counts.data(), globalCounts.data(), 2, MPI_INT, MPI_SUM, mpi.comm);
            const auto faceNodes = (n - 1) * (n - 1);
            DNDS_check_throw_info(globalCounts[0] == 4 * faceNodes && globalCounts[1] == faceNodes,
                                  "Expected four wall interiors and one outlet interior on the unit cube");
        }
        else
            DNDS_check_throw_info(std::abs(global(5, 0) - 6) < 1e-10, "Boundary pieces must tile the six unit-cube faces");
        if (mpi.rank == 0)
        {
            if (wallOutlet)
            {
                csv << n << ',' << mpi.size << ',' << global(5, 0) << ',' << global(5, 1);
                for (int i : {6, 0, 4})
                    csv << ',' << std::sqrt(global(i, 0) / global(5, 0));
            }
            else
                csv << n << ',' << time << ',' << mpi.size << ',' << global(5, 0);
            const int group = wallOutlet ? 1 : 0;
            for (int i = 0; i < 5; ++i)
                csv << ',' << std::sqrt(global(i, group) / global(5, group));
            csv << std::endl;
        }
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    try
    {
        const bool wallOutlet = argc > 1 && std::string(argv[1]) == "--wall-outlet";
        const int first = wallOutlet ? 2 : 1;
        DNDS_check_throw_info(argc >= first + 3, "usage: prism_mms_boundary_flux [--wall-outlet] case.json result.csv mesh12.cgns [mesh16.cgns ...]");
        using namespace DNDS::NCFV;
        auto cfg = LoadConfiguration(argv[first], {}, {}).configuration;
        cfg.physics.boundaryMode = BoundaryMode::CharacteristicFarField;
        cfg.algorithm.retainMicroGeometry = true;
        DNDS_check_throw_info(cfg.physics.exponentialMMS && cfg.algorithm.macroFaceBoundary,
                              "This diagnostic requires the characteristic macroface prism MMS");
        if (wallOutlet)
        {
            cfg.physics.exponentialMMS = false;
            BoundaryZoneSettings outlet;
            outlet.name = "bc-2-1";
            outlet.mode = BoundaryMode::PressureOutlet;
            outlet.staticPressure = outletPressure;
            cfg.physics.boundaryZones = {outlet};
            for (const auto *name : {"bc-3", "bc-3-1", "bc-4", "bc-4-1"})
            {
                BoundaryZoneSettings wall;
                wall.name = name;
                wall.mode = BoundaryMode::SlipWall;
                cfg.physics.boundaryZones.push_back(wall);
            }
        }
        std::ofstream csv;
        if (mpi.rank == 0)
        {
            csv.open(argv[first + 1]);
            DNDS_check_throw_info(csv.good(), "Cannot open boundary-flux CSV");
            csv << std::setprecision(15);
            if (wallOutlet)
                csv << "n,ranks,wall_area,outlet_area,wall_pressure_L2,wall_mass_L2,wall_energy_L2,outlet_mass_L2,outlet_rhou_L2,outlet_rhov_L2,outlet_rhow_L2,outlet_energy_L2\n";
            else
                csv << "n,t,ranks,total_area,mass_flux_L2,x_momentum_flux_L2,y_momentum_flux_L2,z_momentum_flux_L2,energy_flux_L2\n";
        }
        for (int i = first + 2; i < argc; ++i)
        {
            cfg.mesh.meshFile = argv[i];
            WriteLevel(mpi, cfg, wallOutlet, csv);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    MPI_Finalize();
}
