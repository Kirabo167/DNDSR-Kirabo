/**
 * @file ACMFlux.hpp
 * @brief Inviscid, viscous, preconditioning, and Riemann-flux interfaces for variable-density ACM.
 *
 * @details All inviscid algebra is evaluated in a face-local coordinate system. InviscidFlux()
 * performs the global/local transformations and exposes the resulting global face flux.
 *
 * @author Runzhi Ma
 * @date 2026-09-03
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMSettings.hpp"
#include "ACMState.hpp"

namespace DNDS::ACMVariable
{
    /// Three distinct characteristic speeds of the five-variable ACM system.
    struct Eigenvalues
    {
        real lambdaMinus = 0;      ///< Signed lower acoustic speed of the preconditioned normal operator.
        real lambdaTangential = 0; ///< Repeated convective speed for the nonacoustic modes.
        real lambdaPlus = 0;       ///< Signed upper acoustic speed; no surface measure is included.

        /**
         * @brief Compute the largest characteristic-speed magnitude.
         * @return `max(abs(lambdaMinus), abs(lambdaTangential), abs(lambdaPlus))`.
         */
        real SpectralRadius() const;
    };

    /// Numerical face flux together with the characteristic speeds used by its dissipation.
    struct FluxResult
    {
        State flux = State::Zero(); ///< Global density, three momentum, and pressure-equation flux entries.
        Eigenvalues eigenvalues;   ///< Signed characteristic speeds used by the numerical dissipation.
    };

    /**
     * @brief Evaluate the physical inviscid flux in face-local coordinates.
     * @param localState Local state `[rho,m_n,m_t1,m_t2,p]`.
     * @return Local flux `[m_n,m_n*u_n+p,m_t1*u_n,m_t2*u_n,u_n]`.
     */
    State PhysicalFluxLocal(const State &localState);

    /**
     * @brief Evaluate the state Jacobian of PhysicalFluxLocal().
     * @param localState Local linearization state `[rho,m_n,m_t1,m_t2,p]`.
     * @return Five-by-five physical-flux Jacobian in local coordinates.
     */
    Matrix5 PhysicalFluxJacobianLocal(const State &localState);

    /**
     * @brief Construct Gamma for conservative variables [rho,mx,my,mz,p].
     * @param meanLocalState Mean face state used by velocity-pressure coupling terms.
     * @param beta2 Positive artificial-compressibility parameter.
     * @param alpha Turkel coupling parameter.
     * @return Five-by-five local preconditioning matrix.
     */
    Matrix5 GammaLocal(const State &meanLocalState, real beta2, real alpha);

    /**
     * @brief Construct the analytic inverse of GammaLocal().
     * @param meanLocalState Mean face state used by velocity-pressure coupling terms.
     * @param beta2 Positive artificial-compressibility parameter.
     * @param alpha Turkel coupling parameter.
     * @return Five-by-five inverse preconditioning matrix.
     */
    Matrix5 GammaInvLocal(const State &meanLocalState, real beta2, real alpha);

    /**
     * @brief Exact derivative of Gamma(U)*(U-Uold)/dTau, NOT just Gamma/dTau.
     * @param state Current positive-density conservative iterate.
     * @param previous Frozen state at the start of the pseudo-time step.
     * @param pseudoTimeStep Positive pseudo-time interval.
     * @param settings Fixed alpha and beta squared for this iteration.
     * @return Five-by-five product-rule Jacobian, in the supplied momentum frame.
     * @author Runzhi Ma
     */
    Matrix5 PseudoTimeProductJacobian(const State &state, const State &previous,
                                    real pseudoTimeStep, const Settings &settings);

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
     * @param beta2 Positive artificial-compressibility parameter.
     * @param alpha Turkel coupling parameter.
     * @return Five-by-five preconditioned Jacobian.
     */
    Matrix5 PreconditionedJacobianLocal(
        const State &meanLocalState,
        real beta2,
        real alpha);

    /**
     * @brief Try to construct global left/right eigenvectors of the general-alpha ACM operator.
     * @param meanState Mean global state `[rho,mx,my,mz,p]`.
     * @param unitNormal Face-normal direction, normalized internally.
     * @param settings Validated ACM settings containing arbitrary finite Turkel alpha.
     * @param left Output matrix whose rows are the left characteristic vectors.
     * @param right Output matrix whose columns are minus-acoustic, density-contact, two tangential, and plus-acoustic modes.
     * @param minimumRelativeSeparation Relative tolerance used to detect an acoustic/tangential
     * eigenvalue collision, where the general-alpha operator can become defective.
     * @return True when a complete, numerically invertible characteristic basis was constructed.
     * @note Modifier: Runzhi Ma.
     */
    bool TryCharacteristicMatricesGlobal(
        const State &meanState,
        const Vector3 &unitNormal,
        const Settings &settings,
        Matrix5 &left,
        Matrix5 &right,
        real minimumRelativeSeparation = 1e-11);

    /**
     * @brief Construct global right eigenvectors of the general-alpha ACM normal operator.
     * @param meanState Mean global state `[rho,mx,my,mz,p]`.
     * @param unitNormal Face-normal direction, normalized internally.
     * @param settings Validated ACM settings containing arbitrary finite Turkel alpha.
     * @return Matrix whose columns are minus-acoustic, density-contact, two tangential, and plus-acoustic modes.
     * @throws std::runtime_error If an acoustic eigenvalue collides with the tangential eigenvalue
     * and the operator has no complete eigenbasis.
     * @note Modifier: Runzhi Ma.
     */
    Matrix5 RightEigenvectorsGlobal(
        const State &meanState,
        const Vector3 &unitNormal,
        const Settings &settings);

    /**
     * @brief Construct global left eigenvectors of the general-alpha ACM normal operator.
     * @param meanState Mean global state `[rho,mx,my,mz,p]`.
     * @param unitNormal Face-normal direction, normalized internally.
     * @param settings Validated ACM settings containing arbitrary finite Turkel alpha.
     * @return Inverse of RightEigenvectorsGlobal().
     * @throws std::runtime_error If the normal operator has no complete eigenbasis.
     * @note Modifier: Runzhi Ma.
     */
    Matrix5 LeftEigenvectorsGlobal(
        const State &meanState,
        const Vector3 &unitNormal,
        const Settings &settings);

    /**
     * @brief Compute the minus, repeated tangential, and plus characteristic speeds.
     * @param qn Mean normal velocity.
     * @param rho0 Positive local density at the linearization state.
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
     * @brief Evaluate f(Gamma^-1 A), including the defective Jordan limit.
     * @param state Positive-density local conservative state.
     * @param settings Preconditioning parameters and relative entropy width.
     * @return Five-by-five entropy-fixed matrix absolute value (not Gamma times it).
     * @author Runzhi Ma
     */
    Matrix5 AbsolutePreconditionedJacobianLocal(const State &state, const Settings &settings);

    /**
     * @brief Evaluate scalar Rusanov dissipation in face-local coordinates.
     * @param leftLocal Left reconstructed local state.
     * @param rightLocal Right reconstructed local state.
     * @param settings Validated physical and numerical settings.
     * @param eigenvalues Output characteristic speeds evaluated at the square-root-density Roe state.
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
     * @param eigenvalues Output characteristic speeds evaluated at the square-root-density Roe state.
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
     * @param left Left reconstructed global state `[rho,mx,my,mz,p]`.
     * @param right Right reconstructed global state `[rho,mx,my,mz,p]`.
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
     * @param leftCellState Left cell-center state `[rho,mx,my,mz,p]`.
     * @param rightCellState Right or boundary-ghost cell-center state in the left face frame.
     * @param centerDisplacement Vector from the left center to the mapped right/ghost center.
     * @param unitNormal Unit face normal directed from left to right; normalized internally.
     * @return Corrected gradient `Gbar+n*(deltaU-d^T*Gbar)/(d.n)`, which exactly reproduces
     * every linear state field and avoids the former `2*V_left/S_face` distance approximation.
     * @note Modifier: Runzhi Ma.
     */
    Eigen::Matrix<real, 3, 5> CorrectedFaceGradient(
        const Eigen::Matrix<real, 3, 5> &leftGradient,
        const Eigen::Matrix<real, 3, 5> &rightGradient,
        const State &leftCellState,
        const State &rightCellState,
        const Vector3 &centerDisplacement,
        const Vector3 &unitNormal);

    /**
     * @brief Evaluate the variable-density Newtonian laminar viscous flux through a face.
     * @param stateGradient Gradient matrix with convention `stateGradient(i,j)=d(primitive[j])/d(x[i]), primitive=[rho,u,v,w,p]`.
     * The pressure-gradient column is accepted for layout compatibility and is not used.
     * @param unitNormal Face-normal direction, normalized internally.
     * @param dynamicViscosity Non-negative dynamic viscosity.
     * @return Global viscous flux; its continuity/pressure component is zero.
     */
    State ViscousFlux(
        const Eigen::Matrix<real, 3, 5> &stateGradient,
        const Vector3 &unitNormal,
        real dynamicViscosity);
}
