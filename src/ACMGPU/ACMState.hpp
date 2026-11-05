/**
 * @file ACMState.hpp
 * @brief State accessors and face-local coordinate transformations for the ACM module.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACM.hpp"

namespace DNDS::ACMGPU
{
    /**
     * @brief Build a right-handed orthonormal face basis `[normal, tangent1, tangent2]`.
     * @param unitNormal Face normal direction; it is normalized internally and therefore need not
     * have unit length.
     * @return Rotation matrix whose columns are the local basis vectors in global coordinates.
     * @throws std::runtime_error If `unitNormal` is non-finite or has zero length.
     */
    Matrix3 BuildLocalBasis(const Vector3 &unitNormal);

    /**
     * @brief Extract the three velocity components from a four-component ACM state.
     * @param state Input state `[u,v,w,p]`.
     * @return Velocity vector `[u,v,w]`.
     */
    Vector3 Velocity(const State &state);

    /**
     * @brief Read physical pressure from an ACM state.
     * @param state Input state `[u,v,w,p]`.
     * @return Physical pressure stored in component three.
     */
    real PhysicalPressure(const State &state);

    /**
     * @brief Replace the physical-pressure component of an ACM state.
     * @param state State modified in place.
     * @param pressure New physical pressure.
     */
    void SetPhysicalPressure(State &state, real pressure);

    /**
     * @brief Rotate global velocity components into a face-local basis while preserving pressure.
     * @param state Global state `[u,v,w,p]`.
     * @param localBasis Orthonormal basis returned by BuildLocalBasis().
     * @return Face-local state `[u_n,u_t1,u_t2,p]`.
     */
    State ToLocalState(const State &state, const Matrix3 &localBasis);

    /**
     * @brief Rotate a face-local flux back to global coordinates while preserving its scalar entry.
     * @param localFlux Face-local flux `[F_n,F_t1,F_t2,F_c]`.
     * @param localBasis Orthonormal basis returned by BuildLocalBasis().
     * @return Flux represented in global coordinates.
     */
    State FromLocalFlux(const State &localFlux, const Matrix3 &localBasis);

    /**
     * @brief Rotate the velocity block of a state in place without changing pressure.
     * @param state State whose first three entries are modified.
     * @param rotation Three-dimensional rotation matrix applied to the velocity block.
     */
    void RotateStateInPlace(State &state, const Matrix3 &rotation);

    /**
     * @brief Check whether every state component is finite.
     * @param state State to inspect.
     * @return `true` when all four entries are finite; otherwise `false`.
     */
    bool IsFiniteState(const State &state);
}
