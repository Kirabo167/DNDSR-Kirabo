#pragma once

#include "NCFV/NCFVSolver.hpp"

/// Inspect the actual reconstructed state, without changing the marching solver.
inline nlohmann::ordered_json RiemannFluxBudget(
    const DNDS::MPIInfo &mpi, const DNDS::NCFV::Configuration &cfg,
    const DNDS::NCFV::Solver<3> &solver)
{
    using namespace DNDS::NCFV;
    using State = Eigen::Vector<double, 5>;
    const auto &halo = solver.NodeCommunication();
    const auto &geometry = solver.Geometry();
    const auto &topology = solver.EdgeTopology();
    const auto &reconstruction = solver.ReconstructionData();
    const auto count = solver.Mesh()->NumNode();
    NodeStatePair means, rhs;
    for (auto *field : {&means, &rhs})
    {
        field->InitPair("riemannAudit", mpi);
        field->father->Resize(count, 5, 1);
        field->son->Resize(halo.NumNodeGhost(), 5, 1);
        field->BorrowSetup(halo.Layout());
        field->trans.initPersistentPull();
    }
    for (DNDS::index i = 0; i < count; i++)
        means[i] = solver.StateField()[i];
    SpatialOperator<3> spatial(mpi, solver.Mesh(), topology, geometry,
                              reconstruction, solver.Boundaries(), cfg.algorithm.mode,
                              cfg.reconstruction, cfg.physics, cfg.time, halo);
    spatial.Initialize();
    spatial.EvaluateRHS(means, rhs, false, false);
    Eigen::MatrixXd full(count, 5);
    for (DNDS::index i = 0; i < count; i++)
        full.row(i) = rhs[i].transpose();

    double localArea = 0, globalArea = 0;
    State localJump = State::Zero(), globalJump;
    const auto accumulate = [&](const State &left, const State &right, double area)
    {
        localArea += area;
        localJump += area * (right - left).cwiseAbs2();
    };
    for (DNDS::index e = 0; e < topology.NumEdge(); e++)
    {
        const auto &surface = geometry.EdgeSurface(e);
        if (cfg.algorithm.mode == IntegrationMode::TraditionalQuadrature)
        {
            for (const auto &point : surface.quadrature)
            {
                State traces[2];
                for (int side = 0; side < 2; side++)
                {
                    const auto node = surface.nodes[side];
                    traces[side] = State(spatial.PointValues()[node]) +
                        spatial.Coefficients()[node].transpose() * Reconstruction::EvaluateBasis(
                            halo.DisplacementToPoint(node, point.coordinate),
                            reconstruction.ReferenceLengths(node), 3);
                }
                accumulate(traces[0], traces[1], point.weight);
            }
        }
        else
        {
            State traces[2];
            for (int side = 0; side < 2; side++)
            {
                traces[side] = surface.measure * State(spatial.PointValues()[surface.nodes[side]]);
                for (const auto &entry : surface.efficientStencil)
                    traces[side] += spatial.Gradients()[entry.node].transpose() *
                        entry.stateGradientWeights[side].head<3>();
                traces[side] /= surface.measure;
            }
            accumulate(traces[0], traces[1], surface.measure);
        }
    }
    MPI_Allreduce(&localArea, &globalArea, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    MPI_Allreduce(localJump.data(), globalJump.data(), 5, MPI_DOUBLE, MPI_SUM, mpi.comm);
    spatial.SetInteriorInviscidCorrectionScalesForDiagnostics(State::Zero());
    spatial.EvaluateRHS(means, rhs, false, false);
    State localCorrection = State::Zero(), globalCorrection;
    double localVolume = 0, globalVolume = 0;
    for (DNDS::index i = 0; i < count; i++)
    {
        const double volume = geometry.NodeVolume(i).moments.measure;
        localVolume += volume;
        localCorrection += volume * (full.row(i).transpose() - State(rhs[i])).cwiseAbs2();
    }
    MPI_Allreduce(&localVolume, &globalVolume, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    MPI_Allreduce(localCorrection.data(), globalCorrection.data(), 5, MPI_DOUBLE, MPI_SUM, mpi.comm);
    nlohmann::ordered_json result;
    result["components"] = {"rho", "mx", "my", "mz", "E"};
    for (int c = 0; c < 5; c++)
    {
        result["trace_jump_area_RMS"].push_back(std::sqrt(globalJump(c) / globalArea));
        result["correction_rhs_volume_RMS"].push_back(std::sqrt(globalCorrection(c) / globalVolume));
    }
    return result;
}
