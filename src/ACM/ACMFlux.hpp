/**
 * @file ACMFlux.hpp
 * @brief Inviscid, viscous, preconditioning, and Riemann-flux interfaces for constant-density ACM.
 *
 * @details All inviscid algebra is evaluated in a face-local coordinate system. InviscidFlux()
 * performs the global/local transformations and exposes the resulting global face flux.
 *
 * @author Runzhi Ma
 * @date 2026-09-01
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMSettings.hpp"
#include "ACMState.hpp"

namespace DNDS::ACM
{
    /// Three distinct characteristic speeds of the four-variable ACM system.
    struct Eigenvalues
    {
        real lambdaMinus = 0;      ///< Signed lower acoustic speed of the preconditioned normal operator.
        real lambdaTangential = 0; ///< Signed repeated tangential speed, shared by two shear modes.
        real lambdaPlus = 0;       ///< Signed upper acoustic speed; speeds are not surface-integrated.

        /**
         * @brief Compute the largest characteristic-speed magnitude.
         * @return `max(abs(lambdaMinus), abs(lambdaTangential), abs(lambdaPlus))`.
         */
        real SpectralRadius() const;
    };

    /**
     * @brief Numerical face flux and the characteristic speeds used for its dissipation.
     * @details The flux uses global component ordering and a left-to-right normal convention.
     * Surface measure and quadrature weights are applied by the caller, not stored here.
     */
    struct FluxResult
    {
        State flux = State::Zero(); ///< Three velocity-equation fluxes followed by the continuity flux.
        Eigenvalues eigenvalues;   ///< Signed characteristic speeds at the flux linearization state.
    };

    /**
     * @brief Evaluate the physical inviscid flux in face-local coordinates.
     * @param localState Local state `[u_n,u_t1,u_t2,p]`.
     * @param rho0 Positive constant density.
     * @return Local physical flux `[u_n^2+p/rho0,u_n*u_t1,u_n*u_t2,u_n]`.
     */
    State PhysicalFluxLocal(const State &localState, real rho0);

    /**
     * @brief Evaluate the state Jacobian of PhysicalFluxLocal().
     * @param localState Local linearization state `[u_n,u_t1,u_t2,p]`.
     * @param rho0 Positive constant density.
     * @return Four-by-four physical-flux Jacobian in local coordinates.
     */
    Matrix4 PhysicalFluxJacobianLocal(const State &localState, real rho0);

    /**
     * @brief Construct the Scheme-A primitive preconditioning matrix Gamma.
     * @param meanLocalState Mean face state used by velocity-pressure coupling terms.
     * @param beta2 Positive artificial-compressibility parameter.
     * @param alpha Turkel coupling parameter.
     * @return Four-by-four local preconditioning matrix.
     */
    Matrix4 GammaLocal(const State &meanLocalState, real beta2, real alpha);

    /**
     * @brief Construct the analytic inverse of GammaLocal().
     * @param meanLocalState Mean face state used by velocity-pressure coupling terms.
     * @param beta2 Positive artificial-compressibility parameter.
     * @param alpha Turkel coupling parameter.
     * @return Four-by-four inverse preconditioning matrix.
     */
    Matrix4 GammaInvLocal(const State &meanLocalState, real beta2, real alpha);

    /**
     * @brief Differentiate `Gamma(U) * (U-Uold) / dTau` exactly.
     * @param state Current constant-density primitive state.
     * @param previous Frozen state at the beginning of the pseudo-time step.
     * @param pseudoTimeStep Positive pseudo-time interval.
     * @param settings Fixed artificial-compressibility parameters.
     * @return Four-by-four product-rule Jacobian in the supplied coordinate frame.
     */
    Matrix4 PseudoTimeProductJacobian(
        const State &state,
        const State &previous,
        real pseudoTimeStep,
        const Settings &settings);

    /**
     * @brief Apply the local preconditioning matrix to a state increment or eigenvector.
     * @param meanLocalState Mean state defining Gamma.
     * @param increment Vector to which Gamma is applied.
     * @param beta2 Positive artificial-compressibility parameter.
     * @param alpha Turkel coupling parameter.
     * @return `GammaLocal(meanLocalState, beta2, alpha) * increment`.
     */
    State ApplyGammaLocal(
        const State &meanLocalState,
        const State &increment,
        real beta2,
        real alpha);

    /**
     * @brief Build the preconditioned local flux Jacobian `Gamma^{-1} A`.
     * @param meanLocalState Local linearization state.
     * @param rho0 Positive constant density.
     * @param beta2 Positive artificial-compressibility parameter.
     * @param alpha Turkel coupling parameter.
     * @return Four-by-four preconditioned Jacobian.
     */
    Matrix4 PreconditionedJacobianLocal(
        const State &meanLocalState,
        real rho0,
        real beta2,
        real alpha);

    /**
     * @brief Try to construct global left/right eigenvectors of the general-alpha ACM operator.
     * @param meanState Mean global state `[u,v,w,p]`.
     * @param unitNormal Face-normal direction, normalized internally.
     * @param settings Validated ACM settings containing arbitrary finite Turkel alpha.
     * @param left Output matrix whose rows are the left characteristic vectors.
     * @param right Output matrix whose columns are minus-acoustic, two tangential, and plus-acoustic modes.
     * @param minimumRelativeSeparation Relative tolerance used to detect an acoustic/tangential
     * eigenvalue collision, where the general-alpha operator can become defective.
     * @return True when a complete, numerically invertible characteristic basis was constructed.
     * @note Modifier: Runzhi Ma.
     */
    bool TryCharacteristicMatricesGlobal(
        const State &meanState,
        const Vector3 &unitNormal,
        const Settings &settings,
        Matrix4 &left,
        Matrix4 &right,
        real minimumRelativeSeparation = 1e-11);

    /**
     * @brief Construct global right eigenvectors of the general-alpha ACM normal operator.
     * @param meanState Mean global state `[u,v,w,p]`.
     * @param unitNormal Face-normal direction, normalized internally.
     * @param settings Validated ACM settings containing arbitrary finite Turkel alpha.
     * @return Matrix whose columns are minus-acoustic, two tangential, and plus-acoustic modes.
     * @throws std::runtime_error If an acoustic eigenvalue collides with the tangential eigenvalue
     * and the operator has no complete eigenbasis.
     * @note Modifier: Runzhi Ma.
     */
    Matrix4 RightEigenvectorsGlobal(
        const State &meanState,
        const Vector3 &unitNormal,
        const Settings &settings);

    /**
     * @brief Construct global left eigenvectors of the general-alpha ACM normal operator.
     * @param meanState Mean global state `[u,v,w,p]`.
     * @param unitNormal Face-normal direction, normalized internally.
     * @param settings Validated ACM settings containing arbitrary finite Turkel alpha.
     * @return Inverse of RightEigenvectorsGlobal().
     * @throws std::runtime_error If the normal operator has no complete eigenbasis.
     * @note Modifier: Runzhi Ma.
     */
    Matrix4 LeftEigenvectorsGlobal(
        const State &meanState,
        const Vector3 &unitNormal,
        const Settings &settings);

    /**
     * @brief Compute the minus, repeated tangential, and plus characteristic speeds.
     * @param qn Mean normal velocity.
     * @param rho0 Positive constant density.
     * @param beta2 Positive artificial-compressibility parameter.
     * @param alpha Turkel coupling parameter.
     * @return Characteristic speeds ordered as minus, tangential, and plus.
     */
    Eigenvalues ComputeEigenvalues(real qn, real rho0, real beta2, real alpha);

    /**
     * @brief Apply a quadratic entropy fix to an absolute characteristic speed.
     * @param lambda Signed characteristic speed.
     * @param delta Positive entropy-fix width; a non-positive value disables the smoothing.
     * @return Entropy-corrected approximation of `abs(lambda)`.
     */
    real EntropyFixedAbs(real lambda, real delta);

    /**
     * @brief Evaluate scalar Rusanov dissipation in face-local coordinates.
     * @param leftLocal Left reconstructed local state.
     * @param rightLocal Right reconstructed local state.
     * @param settings Validated physical and numerical settings.
     * @param eigenvalues Output characteristic speeds evaluated at the arithmetic mean state.
     * @return Local dissipative vector to subtract from the centered two-state flux.
     */
    State RusanovDissipationLocal(
        const State &leftLocal,
        const State &rightLocal,
        const Settings &settings,
        Eigenvalues &eigenvalues);

    /**
     * @brief Evaluate characteristic Roe-type dissipation for the general-alpha ACM system.
     * @param leftLocal Left reconstructed local state.
     * @param rightLocal Right reconstructed local state.
     * @param settings Validated physical and Turkel-preconditioning settings.
     * @param eigenvalues Output characteristic speeds evaluated at the arithmetic mean state.
     * @return Local characteristic dissipative vector `Gamma R |Lambda| L deltaU` including the
     * configured entropy fix. A confluent-Hermite matrix function is used when the characteristic
     * basis is defective or ill-conditioned, including exact acoustic/tangential collisions.
     * @note Modifier: Runzhi Ma.
     */
    State RoeDissipationLocal(
        const State &leftLocal,
        const State &rightLocal,
        const Settings &settings,
        Eigenvalues &eigenvalues);

    /**
     * @brief Evaluate a complete two-state inviscid numerical flux on a three-dimensional face.
     * @param type Rusanov or Roe dissipation selection.
     * @param left Left reconstructed global state `[u,v,w,p]`.
     * @param right Right reconstructed global state `[u,v,w,p]`.
     * @param unitNormal Face-normal direction, normalized internally.
     * @param settings Physical and numerical settings.
     * @return Global numerical flux and its mean-state characteristic speeds.
     */
    FluxResult InviscidFlux(
        RiemannSolverType type,
        const State &left,
        const State &right,
        const Vector3 &unitNormal,
        const Settings &settings);

    /**
     * @brief Build a linearly exact over-relaxed face gradient from cell-center data.
     * @param leftGradient Reconstructed left gradient, with rows denoting spatial derivatives.
     * @param rightGradient Reconstructed right gradient in the same physical/periodic frame.
     * @param leftCellState Left cell-center state `[u,v,w,p]`.
     * @param rightCellState Right or boundary-ghost cell-center state in the left face frame.
     * @param centerDisplacement Vector from the left center to the mapped right/ghost center.
     * @param unitNormal Unit face normal directed from left to right; normalized internally.
     * @return Corrected gradient `Gbar+n*(deltaU-d^T*Gbar)/(d.n)`, which exactly reproduces
     * every linear state field and avoids the former `2*V_left/S_face` distance approximation.
     * @note Modifier: Runzhi Ma.
     */
    Eigen::Matrix<real, 3, 4> CorrectedFaceGradient(
        const Eigen::Matrix<real, 3, 4> &leftGradient,
        const Eigen::Matrix<real, 3, 4> &rightGradient,
        const State &leftCellState,
        const State &rightCellState,
        const Vector3 &centerDisplacement,
        const Vector3 &unitNormal);

    /**
     * @brief Evaluate the constant-density Newtonian laminar viscous flux through a face.
     * @param stateGradient Gradient matrix with convention `stateGradient(i,j)=d(state[j])/d(x[i])`.
     * The pressure-gradient column is accepted for layout compatibility and is not used.
     * @param unitNormal Face-normal direction, normalized internally.
     * @param rho0 Positive constant density used to convert stress to velocity flux.
     * @param dynamicViscosity Non-negative dynamic viscosity.
     * @return Global viscous flux; its continuity/pressure component is zero.
     */
    State ViscousFlux(
        const Eigen::Matrix<real, 3, 4> &stateGradient,
        const Vector3 &unitNormal,
        real rho0,
        real dynamicViscosity);
}
