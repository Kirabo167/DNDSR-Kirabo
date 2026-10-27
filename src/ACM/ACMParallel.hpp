/**
 * @file ACMParallel.hpp
 * @brief OpenMP face-loop and MPI reduction adapters for the initial ACM kernels.
 *
 * @details These adapters retain the project's existing face-buffer parallel pattern: independent
 * faces are evaluated locally, then only explicitly requested diagnostics are reduced with MPI.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMFlux.hpp"
#include "DNDS/MPI.hpp"

#include <vector>

namespace DNDS::ACM
{
    /// Reconstructed two-sided state and geometry required to evaluate one local face flux.
    struct FaceInput
    {
        State left = State::Zero();  ///< Reconstructed global [u,v,w,p] state on the left side.
        State right = State::Zero(); ///< Right trace, or the generated exterior state on a boundary.
        Vector3 unitNormal = Vector3::UnitX(); ///< Left-to-right normal direction; flux code normalizes it.
    };

    /**
     * @brief Evaluate one inviscid flux per local face, using OpenMP when enabled by the build.
     * @param faces Read-only local face inputs; each entry is independent.
     * @param faceFluxBuffer Output buffer resized to `faces.size()` and written by face index.
     * @param settings Physical and Riemann-solver settings shared by all faces.
     */
    void EvaluateFaceFluxes(
        const std::vector<FaceInput> &faces,
        std::vector<FluxResult> &faceFluxBuffer,
        const Settings &settings);

    /**
     * @brief Sum all entries in a rank-local face-flux buffer.
     * @param faceFluxBuffer Local numerical flux results.
     * @return Component-wise local checksum; this helper does not communicate through MPI.
     */
    State LocalFluxSum(const std::vector<FluxResult> &faceFluxBuffer);

    /**
     * @brief Compute the component-wise MPI sum of rank-local flux checksums.
     * @param localFluxSum Four-component checksum owned by the calling rank.
     * @param mpi MPI communicator metadata; `mpi.comm` participates in the collective.
     * @return Global checksum available identically on every participating rank.
     */
    State GlobalFluxSum(const State &localFluxSum, const MPIInfo &mpi);
}
