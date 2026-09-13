/**
 * @file ACM.cpp
 * @brief Implementation of constant-density ACM state, flux, boundary, parallel, and configuration kernels.
 *
 * @details This translation unit intentionally remains independent of the Euler evaluator. It provides
 * modular kernels that can later be assembled into a dedicated mesh-based ACM evaluator while reusing
 * DNDS configuration, OpenMP face loops, and MPI collectives.
 *
 * @author Runzhi Ma
 * @date 2026-09-01
 * @note Modifier: Runzhi Ma.
 */
#include "ACMBC.hpp"
#include "ACMBDF2.hpp"
#include "ACMConfig.hpp"
#include "ACMFlux.hpp"
#include "ACMParallel.hpp"

#include "DNDS/Errors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>

#include <Eigen/SVD>

namespace DNDS::ACM
{
    namespace
    {
        constexpr real normalTolerance = 1e-14;

        /**
         * @brief Validate and normalize a face-normal vector used internally by ACM kernels.
         * @param normal Finite non-zero face-normal direction.
         * @return Unit-length vector parallel to `normal`.
         * @throws std::runtime_error If the vector is non-finite or numerically zero.
         */
        Vector3 NormalizedNormal(const Vector3 &normal)
        {
            DNDS_check_throw_info(normal.allFinite(), "ACM face normal contains a non-finite value");
            const real normalNorm = normal.norm();
            DNDS_check_throw_info(normalNorm > normalTolerance, "ACM face normal has zero length");
            return normal / normalNorm;
        }
    }

    /** @copydoc BuildLocalBasis */
    Matrix3 BuildLocalBasis(const Vector3 &unitNormal)
    {
        const Vector3 normal = NormalizedNormal(unitNormal);
        Vector3 reference;
        const Vector3 normalAbs = normal.cwiseAbs();
        if (normalAbs(0) <= normalAbs(1) && normalAbs(0) <= normalAbs(2))
            reference = Vector3::UnitX();
        else if (normalAbs(1) <= normalAbs(2))
            reference = Vector3::UnitY();
        else
            reference = Vector3::UnitZ();

        Vector3 tangent1 = reference - normal * normal.dot(reference);
        tangent1.normalize();
        Vector3 tangent2 = normal.cross(tangent1);
        tangent2.normalize();

        Matrix3 basis;
        basis.col(0) = normal;
        basis.col(1) = tangent1;
        basis.col(2) = tangent2;
        return basis;
    }

    /** @copydoc Velocity */
    Vector3 Velocity(const State &state)
    {
        return state.head<3>();
    }

    /** @copydoc PhysicalPressure */
    real PhysicalPressure(const State &state)
    {
        return state(3);
    }

    /** @copydoc SetPhysicalPressure */
    void SetPhysicalPressure(State &state, real pressure)
    {
        state(3) = pressure;
    }

    /** @copydoc ToLocalState */
    State ToLocalState(const State &state, const Matrix3 &localBasis)
    {
        DNDS_check_throw_info(state.allFinite(), "ACM state contains a non-finite value");
        DNDS_check_throw_info(localBasis.allFinite(), "ACM local basis contains a non-finite value");
        State localState;
        localState.head<3>() = localBasis.transpose() * state.head<3>();
        localState(3) = state(3);
        return localState;
    }

    /** @copydoc FromLocalFlux */
    State FromLocalFlux(const State &localFlux, const Matrix3 &localBasis)
    {
        DNDS_check_throw_info(localFlux.allFinite(), "ACM local flux contains a non-finite value");
        State globalFlux;
        globalFlux.head<3>() = localBasis * localFlux.head<3>();
        globalFlux(3) = localFlux(3);
        return globalFlux;
    }

    /** @copydoc RotateStateInPlace */
    void RotateStateInPlace(State &state, const Matrix3 &rotation)
    {
        state.head<3>() = rotation * state.head<3>();
    }

    /** @copydoc IsFiniteState */
    bool IsFiniteState(const State &state)
    {
        return state.allFinite();
    }

    /** @copydoc Eigenvalues::SpectralRadius */
    real Eigenvalues::SpectralRadius() const
    {
        return std::max({std::abs(lambdaMinus), std::abs(lambdaTangential), std::abs(lambdaPlus)});
    }

    /** @copydoc PhysicalFluxLocal */
    State PhysicalFluxLocal(const State &localState, real rho0)
    {
        DNDS_check_throw_info(localState.allFinite(), "ACM local state contains a non-finite value");
        DNDS_check_throw_info(std::isfinite(rho0) && rho0 > 0, "ACM rho0 must be finite and positive");
        const real qn = localState(0);
        State flux;
        flux(0) = qn * qn + localState(3) / rho0;
        flux(1) = qn * localState(1);
        flux(2) = qn * localState(2);
        flux(3) = qn;
        return flux;
    }

    /** @copydoc PhysicalFluxJacobianLocal */
    Matrix4 PhysicalFluxJacobianLocal(const State &localState, real rho0)
    {
        DNDS_check_throw_info(localState.allFinite(), "ACM local state contains a non-finite value");
        DNDS_check_throw_info(std::isfinite(rho0) && rho0 > 0, "ACM rho0 must be finite and positive");
        Matrix4 jacobian = Matrix4::Zero();
        jacobian(0, 0) = 2 * localState(0);
        jacobian(0, 3) = 1 / rho0;
        jacobian(1, 0) = localState(1);
        jacobian(1, 1) = localState(0);
        jacobian(2, 0) = localState(2);
        jacobian(2, 2) = localState(0);
        jacobian(3, 0) = 1;
        return jacobian;
    }

    /** @copydoc GammaLocal */
    Matrix4 GammaLocal(const State &meanLocalState, real beta2, real alpha)
    {
        DNDS_check_throw_info(meanLocalState.allFinite(), "ACM mean state contains a non-finite value");
        DNDS_check_throw_info(std::isfinite(beta2) && beta2 > 0, "ACM beta2 must be finite and positive");
        const real gammaT = 1 + alpha;
        Matrix4 gamma = Matrix4::Identity();
        gamma.block<3, 1>(0, 3) = gammaT * meanLocalState.head<3>() / beta2;
        gamma(3, 3) = 1 / beta2;
        return gamma;
    }

    /** @copydoc GammaInvLocal */
    Matrix4 GammaInvLocal(const State &meanLocalState, real beta2, real alpha)
    {
        DNDS_check_throw_info(meanLocalState.allFinite(), "ACM mean state contains a non-finite value");
        DNDS_check_throw_info(std::isfinite(beta2) && beta2 > 0, "ACM beta2 must be finite and positive");
        const real gammaT = 1 + alpha;
        Matrix4 gammaInv = Matrix4::Identity();
        gammaInv.block<3, 1>(0, 3) = -gammaT * meanLocalState.head<3>();
        gammaInv(3, 3) = beta2;
        return gammaInv;
    }

    /** @copydoc PseudoTimeProductJacobian */
    Matrix4 PseudoTimeProductJacobian(
        const State &state,
        const State &previous,
        real pseudoTimeStep,
        const Settings &settings)
    {
        DNDS_check_throw_info(
            IsFiniteState(state) && IsFiniteState(previous) &&
                std::isfinite(pseudoTimeStep) && pseudoTimeStep > 0,
            "Invalid ACM pseudo-time product Jacobian input");
        Matrix4 result = GammaLocal(state, settings.beta2, settings.alpha);
        const real pressureJumpScaled =
            (state(3) - previous(3)) / settings.beta2;
        for (int component = 0; component < 3; component++)
            result(component, component) +=
                (settings.alpha + 1) * pressureJumpScaled;
        return result / pseudoTimeStep;
    }

    /** @copydoc ApplyGammaLocal */
    State ApplyGammaLocal(
        const State &meanLocalState,
        const State &increment,
        real beta2,
        real alpha)
    {
        return GammaLocal(meanLocalState, beta2, alpha) * increment;
    }

    /** @copydoc PreconditionedJacobianLocal */
    Matrix4 PreconditionedJacobianLocal(
        const State &meanLocalState,
        real rho0,
        real beta2,
        real alpha)
    {
        return GammaInvLocal(meanLocalState, beta2, alpha) *
               PhysicalFluxJacobianLocal(meanLocalState, rho0);
    }

    /** @copydoc ComputeEigenvalues */
    Eigenvalues ComputeEigenvalues(real qn, real rho0, real beta2, real alpha)
    {
        DNDS_check_throw_info(std::isfinite(qn), "ACM normal velocity must be finite");
        DNDS_check_throw_info(std::isfinite(rho0) && rho0 > 0, "ACM rho0 must be finite and positive");
        DNDS_check_throw_info(std::isfinite(beta2) && beta2 > 0, "ACM beta2 must be finite and positive");
        DNDS_check_throw_info(std::isfinite(alpha), "ACM alpha must be finite");
        const real centeredVelocity = (1 - alpha) * qn;
        DNDS_check_throw_info(std::isfinite(centeredVelocity),
                              "ACM centered characteristic velocity overflowed");
        // hypot(a,b) evaluates sqrt(a^2+b^2) without overflowing when either
        // representable input is large.  This is algebraically identical to the
        // characteristic discriminant and does not alter the ACM eigenvalues.
        // Modifier: Runzhi Ma.
        const real root = std::hypot(
            centeredVelocity,
            2.0 * std::sqrt(beta2 / rho0));
        DNDS_check_throw_info(std::isfinite(root),
                              "ACM characteristic root is non-finite");
        return {(centeredVelocity - root) * 0.5, qn, (centeredVelocity + root) * 0.5};
    }

    namespace
    {
        /**
         * @brief Construct one pressure-normalized general-alpha acoustic right eigenvector.
         * @param meanLocalState Local mean state `[q_n,q_t1,q_t2,p]`.
         * @param lambda Acoustic eigenvalue associated with the requested vector.
         * @param beta2 Positive artificial-compressibility parameter.
         * @param alpha Turkel coupling parameter.
         * @return Right eigenvector with pressure component equal to one.
         * @note Modifier: Runzhi Ma.
         */
        State AcousticRightEigenvectorLocal(
            const State &meanLocalState,
            real lambda,
            real beta2,
            real alpha)
        {
            const real qn = meanLocalState(0);
            const real separation = qn - lambda;
            DNDS_check_throw_info(
                std::abs(separation) > std::numeric_limits<real>::epsilon(),
                "ACM acoustic and tangential eigenvalues are inseparable");
            State right = State::Zero();
            right(0) = lambda / beta2;
            right(1) = alpha * meanLocalState(1) * right(0) / separation;
            right(2) = alpha * meanLocalState(2) * right(0) / separation;
            right(3) = 1;
            return right;
        }

        /**
         * @brief Build a complete local characteristic basis when the operator is diagonalizable.
         * @param meanLocalState Local mean state `[q_n,q_t1,q_t2,p]`.
         * @param settings General-alpha ACM settings.
         * @param left Output left characteristic matrix.
         * @param right Output right characteristic matrix.
         * @param minimumRelativeSeparation Relative eigenvalue and singular-value tolerance.
         * @return False at an acoustic/tangential collision or an ill-conditioned basis.
         * @note Modifier: Runzhi Ma.
         */
        bool TryCharacteristicMatricesLocal(
            const State &meanLocalState,
            const Settings &settings,
            Matrix4 &left,
            Matrix4 &right,
            real minimumRelativeSeparation)
        {
            const Eigenvalues eigenvalues = ComputeEigenvalues(
                meanLocalState(0), settings.rho0, settings.beta2, settings.alpha);
            const real scale = std::max(
                {real(1), std::abs(meanLocalState(0)), eigenvalues.SpectralRadius()});
            if (std::abs(meanLocalState(0) - eigenvalues.lambdaMinus) <=
                    minimumRelativeSeparation * scale ||
                std::abs(meanLocalState(0) - eigenvalues.lambdaPlus) <=
                    minimumRelativeSeparation * scale)
                return false;

            right.setZero();
            right.col(0) = AcousticRightEigenvectorLocal(
                meanLocalState, eigenvalues.lambdaMinus, settings.beta2, settings.alpha);
            right(1, 1) = 1;
            right(2, 2) = 1;
            right.col(3) = AcousticRightEigenvectorLocal(
                meanLocalState, eigenvalues.lambdaPlus, settings.beta2, settings.alpha);

            const Eigen::JacobiSVD<Matrix4> svd(right);
            const auto singularValues = svd.singularValues();
            if (!(singularValues(0) > 0) ||
                singularValues(3) <= minimumRelativeSeparation * singularValues(0))
                return false;
            left = right.fullPivLu().inverse();
            return left.allFinite() && right.allFinite();
        }
    }

    /** @copydoc TryCharacteristicMatricesGlobal */
    bool TryCharacteristicMatricesGlobal(
        const State &meanState,
        const Vector3 &unitNormal,
        const Settings &settings,
        Matrix4 &left,
        Matrix4 &right,
        real minimumRelativeSeparation)
    {
        settings.Validate();
        DNDS_check_throw_info(
            std::isfinite(minimumRelativeSeparation) && minimumRelativeSeparation > 0,
            "ACM characteristic separation tolerance must be finite and positive");
        const Matrix3 basis = BuildLocalBasis(unitNormal);
        const State meanLocal = ToLocalState(meanState, basis);
        Matrix4 leftLocal;
        Matrix4 rightLocal;
        if (!TryCharacteristicMatricesLocal(
                meanLocal, settings, leftLocal, rightLocal, minimumRelativeSeparation))
            return false;

        Matrix4 localToGlobal = Matrix4::Identity();
        localToGlobal.block<3, 3>(0, 0) = basis;
        Matrix4 globalToLocal = Matrix4::Identity();
        globalToLocal.block<3, 3>(0, 0) = basis.transpose();
        right = localToGlobal * rightLocal;
        left = leftLocal * globalToLocal;
        return left.allFinite() && right.allFinite();
    }

    /** @copydoc RightEigenvectorsGlobal */
    Matrix4 RightEigenvectorsGlobal(
        const State &meanState,
        const Vector3 &unitNormal,
        const Settings &settings)
    {
        Matrix4 left;
        Matrix4 right;
        DNDS_check_throw_info(
            TryCharacteristicMatricesGlobal(meanState, unitNormal, settings, left, right),
            "ACM normal operator is defective at an acoustic/tangential eigenvalue collision");
        return right;
    }

    /** @copydoc LeftEigenvectorsGlobal */
    Matrix4 LeftEigenvectorsGlobal(
        const State &meanState,
        const Vector3 &unitNormal,
        const Settings &settings)
    {
        Matrix4 left;
        Matrix4 right;
        DNDS_check_throw_info(
            TryCharacteristicMatricesGlobal(meanState, unitNormal, settings, left, right),
            "ACM normal operator is defective at an acoustic/tangential eigenvalue collision");
        return left;
    }

    /** @copydoc EntropyFixedAbs */
    real EntropyFixedAbs(real lambda, real delta)
    {
        const real lambdaAbs = std::abs(lambda);
        if (!(delta > 0) || lambdaAbs >= delta)
            return lambdaAbs;
        return (lambda * lambda + delta * delta) / (2 * delta);
    }

    namespace
    {
        /**
         * @brief Differentiate the Harten-smoothed absolute-value function.
         * @param lambda Signed characteristic speed.
         * @param delta Entropy-fix width used by EntropyFixedAbs().
         * @return First derivative; zero is selected at the unsmoothed origin.
         * @note Modifier: Runzhi Ma.
         */
        real EntropyFixedAbsDerivative(real lambda, real delta)
        {
            if (delta > 0 && std::abs(lambda) < delta)
                return lambda / delta;
            if (lambda > 0)
                return 1;
            if (lambda < 0)
                return -1;
            return 0;
        }

        /**
         * @brief Evaluate the second derivative used by a repeated Hermite node.
         * @param lambda Signed characteristic speed.
         * @param delta Entropy-fix width.
         * @return `1/delta` inside the quadratic entropy interval and zero outside.
         * @note Modifier: Runzhi Ma.
         */
        real EntropyFixedAbsSecondDerivative(real lambda, real delta)
        {
            return delta > 0 && std::abs(lambda) < delta ? 1 / delta : 0;
        }

        /**
         * @brief Compute `f[q,q,c]` for the entropy-fixed absolute-value function.
         * @param q Repeated tangential eigenvalue.
         * @param c Acoustic eigenvalue nearest to q.
         * @param delta Entropy-fix width.
         * @return Confluent second divided difference, evaluated without cancellation whenever
         * q and c lie in the same linear or quadratic branch.
         * @note Modifier: Runzhi Ma.
         */
        real EntropyFixedAbsRepeatedSecondDifference(real q, real c, real delta)
        {
            const real scale = std::max({real(1), std::abs(q), std::abs(c), std::abs(delta)});
            const real separation = c - q;
            if (std::abs(separation) <= 1e-8 * scale)
                return 0.5 * EntropyFixedAbsSecondDerivative(q, delta);

            if (delta > 0 && std::abs(q) < delta && std::abs(c) < delta)
                return 0.5 / delta;
            // Strict inequalities are required when delta=0: q=0 is the selected cusp
            // derivative, not a point on either differentiable linear branch.
            // Modifier: Runzhi Ma.
            if ((q > delta && c > delta) || (q < -delta && c < -delta))
                return 0;

            const real firstDifference =
                (EntropyFixedAbs(c, delta) - EntropyFixedAbs(q, delta)) / separation;
            return (firstDifference - EntropyFixedAbsDerivative(q, delta)) / separation;
        }

        /**
         * @brief Evaluate the entropy-fixed matrix absolute value without an eigenvector inverse.
         * @param preconditionedJacobian Local operator `B=Gamma^{-1}A`.
         * @param eigenvalues Its minus, repeated tangential, and plus eigenvalues.
         * @param entropyDelta Entropy-fix width.
         * @return Exact confluent-Hermite matrix function for the characteristic multiset
         * `{lambdaSeparated,q,q,lambdaClustered}`. At a defective collision this includes the
         * required Jordan derivative term.
         * @note Modifier: Runzhi Ma.
         */
        Matrix4 EntropyFixedAbsoluteJacobianHermite(
            const Matrix4 &preconditionedJacobian,
            const Eigenvalues &eigenvalues,
            real entropyDelta)
        {
            const real q = eigenvalues.lambdaTangential;
            const real minusSeparation = std::abs(q - eigenvalues.lambdaMinus);
            const real plusSeparation = std::abs(q - eigenvalues.lambdaPlus);
            const bool minusIsSeparated = minusSeparation >= plusSeparation;
            const real separated = minusIsSeparated
                                       ? eigenvalues.lambdaMinus
                                       : eigenvalues.lambdaPlus;
            const real clustered = minusIsSeparated
                                       ? eigenvalues.lambdaPlus
                                       : eigenvalues.lambdaMinus;

            const real qMinusSeparated = q - separated;
            const real acousticSpan = clustered - separated;
            const real scale = std::max(
                {real(1), std::abs(q), std::abs(separated), std::abs(clustered)});
            DNDS_check_throw_info(
                std::abs(qMinusSeparated) > std::numeric_limits<real>::epsilon() * scale &&
                    std::abs(acousticSpan) > std::numeric_limits<real>::epsilon() * scale,
                "ACM Hermite absolute Jacobian requires one separated acoustic eigenvalue");

            const real valueSeparated = EntropyFixedAbs(separated, entropyDelta);
            const real valueQ = EntropyFixedAbs(q, entropyDelta);
            const real derivativeQ = EntropyFixedAbsDerivative(q, entropyDelta);
            const real coefficient0 = valueSeparated;
            const real coefficient1 = (valueQ - valueSeparated) / qMinusSeparated;
            const real coefficient2 =
                (derivativeQ - coefficient1) / qMinusSeparated;
            const real repeatedSecondDifference =
                EntropyFixedAbsRepeatedSecondDifference(q, clustered, entropyDelta);
            const real coefficient3 =
                (repeatedSecondDifference - coefficient2) / acousticSpan;

            const Matrix4 identity = Matrix4::Identity();
            const Matrix4 shiftedSeparated =
                preconditionedJacobian - separated * identity;
            const Matrix4 shiftedTangential =
                preconditionedJacobian - q * identity;
            const Matrix4 absoluteJacobian =
                coefficient0 * identity +
                coefficient1 * shiftedSeparated +
                coefficient2 * shiftedSeparated * shiftedTangential +
                coefficient3 * shiftedSeparated * shiftedTangential * shiftedTangential;
            DNDS_check_throw_info(
                absoluteJacobian.allFinite(),
                "ACM Hermite absolute Jacobian is non-finite");
            return absoluteJacobian;
        }
    }

    /** @copydoc RusanovDissipationLocal */
    State RusanovDissipationLocal(
        const State &leftLocal,
        const State &rightLocal,
        const Settings &settings,
        Eigenvalues &eigenvalues)
    {
        settings.Validate();
        const State meanState = 0.5 * (leftLocal + rightLocal);
        eigenvalues = ComputeEigenvalues(meanState(0), settings.rho0, settings.beta2, settings.alpha);
        return eigenvalues.SpectralRadius() *
               ApplyGammaLocal(meanState, rightLocal - leftLocal, settings.beta2, settings.alpha);
    }

    /** @copydoc RoeDissipationLocal */
    State RoeDissipationLocal(
        const State &leftLocal,
        const State &rightLocal,
        const Settings &settings,
        Eigenvalues &eigenvalues)
    {
        settings.Validate();
        const State meanState = 0.5 * (leftLocal + rightLocal);
        const State increment = rightLocal - leftLocal;
        eigenvalues = ComputeEigenvalues(
            meanState(0), settings.rho0, settings.beta2, settings.alpha);

        const real entropyDelta = settings.entropyFixRatio *
                                  std::max({eigenvalues.SpectralRadius(),
                                            std::sqrt(settings.beta2 / settings.rho0)});
        const real lambdaMinusAbs = EntropyFixedAbs(eigenvalues.lambdaMinus, entropyDelta);
        const real lambdaTangentialAbs = EntropyFixedAbs(eigenvalues.lambdaTangential, entropyDelta);
        const real lambdaPlusAbs = EntropyFixedAbs(eigenvalues.lambdaPlus, entropyDelta);
        Matrix4 leftEigenvectors;
        Matrix4 rightEigenvectors;
        State preconditionedDissipation;
        if (TryCharacteristicMatricesLocal(
                meanState, settings, leftEigenvectors, rightEigenvectors, 1e-10))
        {
            Matrix4 absoluteEigenvalues = Matrix4::Zero();
            absoluteEigenvalues.diagonal() << lambdaMinusAbs,
                lambdaTangentialAbs,
                lambdaTangentialAbs,
                lambdaPlusAbs;
            preconditionedDissipation =
                rightEigenvectors * absoluteEigenvalues * leftEigenvectors * increment;
        }
        else
        {
            // A confluent-Hermite polynomial evaluates f(B), f=lambda->|lambda|_delta,
            // directly from B. It is identical to R*f(Lambda)*L away from a collision and,
            // at alpha*q_n^2=beta^2/rho, retains the f'(q) Jordan contribution that the former
            // equal-eigenvalue cluster approximation omitted. Modifier: Runzhi Ma.
            const Matrix4 preconditionedJacobian = PreconditionedJacobianLocal(
                meanState,
                settings.rho0,
                settings.beta2,
                settings.alpha);
            preconditionedDissipation =
                EntropyFixedAbsoluteJacobianHermite(
                    preconditionedJacobian,
                    eigenvalues,
                    entropyDelta) *
                increment;
        }

        const State dissipation = ApplyGammaLocal(
            meanState,
            preconditionedDissipation,
            settings.beta2,
            settings.alpha);
        DNDS_check_throw_info(dissipation.allFinite(), "ACM Roe dissipation is non-finite");
        return dissipation;
    }

    /** @copydoc InviscidFlux */
    FluxResult InviscidFlux(
        RiemannSolverType type,
        const State &left,
        const State &right,
        const Vector3 &unitNormal,
        const Settings &settings)
    {
        settings.Validate();
        DNDS_check_throw_info(left.allFinite() && right.allFinite(), "ACM interface state is non-finite");
        const Matrix3 localBasis = BuildLocalBasis(unitNormal);
        const State leftLocal = ToLocalState(left, localBasis);
        const State rightLocal = ToLocalState(right, localBasis);
        const State leftFlux = PhysicalFluxLocal(leftLocal, settings.rho0);
        const State rightFlux = PhysicalFluxLocal(rightLocal, settings.rho0);

        Eigenvalues eigenvalues;
        State dissipation;
        if (type == RiemannSolverType::Rusanov)
            dissipation = RusanovDissipationLocal(leftLocal, rightLocal, settings, eigenvalues);
        else if (type == RiemannSolverType::Roe)
            dissipation = RoeDissipationLocal(leftLocal, rightLocal, settings, eigenvalues);
        else
            DNDS_check_throw_info(false, "unknown ACM Riemann solver");

        FluxResult result;
        result.flux = FromLocalFlux(0.5 * (leftFlux + rightFlux - dissipation), localBasis);
        result.eigenvalues = eigenvalues;
        return result;
    }

    /** @copydoc CorrectedFaceGradient */
    Eigen::Matrix<real, 3, 4> CorrectedFaceGradient(
        const Eigen::Matrix<real, 3, 4> &leftGradient,
        const Eigen::Matrix<real, 3, 4> &rightGradient,
        const State &leftCellState,
        const State &rightCellState,
        const Vector3 &centerDisplacement,
        const Vector3 &unitNormal)
    {
        DNDS_check_throw_info(
            leftGradient.allFinite() && rightGradient.allFinite(),
            "ACM corrected face gradient received a non-finite reconstructed gradient");
        DNDS_check_throw_info(
            leftCellState.allFinite() && rightCellState.allFinite(),
            "ACM corrected face gradient received a non-finite cell state");
        DNDS_check_throw_info(
            centerDisplacement.allFinite(),
            "ACM corrected face gradient received a non-finite center displacement");

        const Vector3 normal = NormalizedNormal(unitNormal);
        const real displacementNorm = centerDisplacement.norm();
        DNDS_check_throw_info(
            displacementNorm > normalTolerance,
            "ACM corrected face gradient requires distinct cell/ghost centers");
        const real projectedDistance = centerDisplacement.dot(normal);
        DNDS_check_throw_info(
            std::abs(projectedDistance) >
                100 * std::numeric_limits<real>::epsilon() * displacementNorm,
            "ACM corrected face gradient has a center displacement tangent to the face");

        Eigen::Matrix<real, 3, 4> faceGradient =
            0.5 * (leftGradient + rightGradient);
        const Eigen::RowVector<real, 4> centerMismatch =
            (rightCellState - leftCellState).transpose() -
            centerDisplacement.transpose() * faceGradient;
        faceGradient += normal * centerMismatch / projectedDistance;
        DNDS_check_throw_info(
            faceGradient.allFinite(),
            "ACM corrected face gradient is non-finite");
        return faceGradient;
    }

    /** @copydoc ViscousFlux */
    State ViscousFlux(
        const Eigen::Matrix<real, 3, 4> &stateGradient,
        const Vector3 &unitNormal,
        real rho0,
        real dynamicViscosity)
    {
        DNDS_check_throw_info(stateGradient.allFinite(), "ACM state gradient contains a non-finite value");
        DNDS_check_throw_info(std::isfinite(rho0) && rho0 > 0, "ACM rho0 must be finite and positive");
        DNDS_check_throw_info(std::isfinite(dynamicViscosity) && dynamicViscosity >= 0,
                              "ACM dynamic viscosity must be finite and non-negative");
        const Vector3 normal = NormalizedNormal(unitNormal);

        // stateGradient(row, column) = d(state[column]) / d(x[row]).
        const Matrix3 velocityGradient = stateGradient.block<3, 3>(0, 0).transpose();
        const real divergence = velocityGradient.trace();
        const Matrix3 stress = dynamicViscosity *
                               (velocityGradient + velocityGradient.transpose() -
                                (2.0 / 3.0) * divergence * Matrix3::Identity());

        State flux = State::Zero();
        flux.head<3>() = stress * normal / rho0;
        return flux;
    }

    /** @copydoc GenerateBoundaryState */
    State BoundaryCondition::ValueState() const
    {
        return Eigen::Map<const State>(value.data());
    }

    /** @copydoc BoundaryHandler::BoundaryHandler */
    BoundaryHandler::BoundaryHandler(
        BoundaryType defaultType,
        const State &defaultValue,
        const std::vector<BoundaryCondition> &configuredConditions)
    {
        DNDS_check_throw_info(defaultType != BoundaryType::BCUnknown,
                              "ACM default boundary type cannot be BCUnknown");
        DNDS_check_throw_info(defaultValue.allFinite(), "ACM default boundary value is non-finite");

        _defaultCondition.type = defaultType;
        _defaultCondition.name = "__ACM_DEFAULT__";
        Eigen::Map<State>(_defaultCondition.value.data()) = defaultValue;
        _nameToID = Geom::GetFaceName2IDDefault();
        _conditions.assign(Geom::BC_ID_DEFAULT_MAX, _defaultCondition);

        // Preserve the same built-in zone semantics as Euler. A default wall is stationary,
        // whereas far/special zones inherit the configured ACM default state.
        BoundaryCondition stationaryWall = _defaultCondition;
        stationaryWall.type = BoundaryType::BCWall;
        stationaryWall.value = {0, 0, 0, 0};
        _conditions[Geom::BC_ID_DEFAULT_WALL] = stationaryWall;
        stationaryWall.type = BoundaryType::BCWallInvis;
        _conditions[Geom::BC_ID_DEFAULT_WALL_INVIS] = stationaryWall;
        _conditions[Geom::BC_ID_DEFAULT_FAR].type = BoundaryType::BCFar;
        _conditions[Geom::BC_ID_DEFAULT_SPECIAL_DMR_FAR].type = BoundaryType::BCSpecial;
        _conditions[Geom::BC_ID_DEFAULT_SPECIAL_RT_FAR].type = BoundaryType::BCSpecial;
        _conditions[Geom::BC_ID_DEFAULT_SPECIAL_IV_FAR].type = BoundaryType::BCSpecial;
        _conditions[Geom::BC_ID_DEFAULT_SPECIAL_2DRiemann_FAR].type = BoundaryType::BCSpecial;

        for (const auto &condition : configuredConditions)
        {
            DNDS_check_throw_info(!condition.name.empty(), "ACM boundary condition has an empty zone name");
            DNDS_check_throw_info(condition.type != BoundaryType::BCUnknown,
                                  "ACM configured boundary type cannot be BCUnknown");
            DNDS_check_throw_info(condition.ValueState().allFinite(),
                                  "ACM configured boundary value is non-finite: " + condition.name);
            DNDS_check_throw_info(
                std::all_of(condition.valueExtra.begin(), condition.valueExtra.end(),
                            [](real value)
                            { return std::isfinite(value); }),
                "ACM boundary valueExtra is non-finite: " + condition.name);

            Geom::t_index id;
            const auto found = _nameToID.find(condition.name);
            if (found != _nameToID.end())
                id = found->second;
            else
            {
                id = static_cast<Geom::t_index>(_conditions.size());
                _nameToID.emplace(condition.name, id);
                _conditions.push_back(_defaultCondition);
            }
            DNDS_check_throw_info(Geom::FaceIDIsExternalBC(id),
                                  "ACM boundary configuration cannot override an internal/periodic zone");
            if (id >= static_cast<Geom::t_index>(_conditions.size()))
                _conditions.resize(static_cast<std::size_t>(id + 1), _defaultCondition);
            _conditions[static_cast<std::size_t>(id)] = condition;
        }
    }

    /** @copydoc BoundaryHandler::GetIDFromName */
    Geom::t_index BoundaryHandler::GetIDFromName(const std::string &name)
    {
        const auto found = _nameToID.find(name);
        if (found != _nameToID.end())
            return found->second;
        const Geom::t_index id = static_cast<Geom::t_index>(_conditions.size());
        _nameToID.emplace(name, id);
        BoundaryCondition appended = _defaultCondition;
        appended.name = name;
        _conditions.push_back(std::move(appended));
        return id;
    }

    /** @copydoc BoundaryHandler::GetConditionFromID */
    const BoundaryCondition &BoundaryHandler::GetConditionFromID(Geom::t_index id) const
    {
        if (!Geom::FaceIDIsExternalBC(id) || id >= static_cast<Geom::t_index>(_conditions.size()))
            return _defaultCondition;
        return _conditions[static_cast<std::size_t>(id)];
    }

    /** @copydoc BoundaryHandler::GetTypeFromID */
    BoundaryType BoundaryHandler::GetTypeFromID(Geom::t_index id) const
    {
        return GetConditionFromID(id).type;
    }

    /** @copydoc BoundaryHandler::GetValueFromID */
    State BoundaryHandler::GetValueFromID(Geom::t_index id) const
    {
        return GetConditionFromID(id).ValueState();
    }

    /** @copydoc GenerateBoundaryState */
    State GenerateBoundaryState(
        const BoundaryCondition &condition,
        const State &interiorState,
        const Vector3 &unitNormal,
        const Settings &settings,
        const Vector3 &point,
        real time)
    {
        (void)point;
        (void)time;
        settings.Validate();
        const State boundaryValue = condition.ValueState();
        DNDS_check_throw_info(interiorState.allFinite() && boundaryValue.allFinite(),
                              "ACM boundary state contains a non-finite value");
        const Vector3 normal = NormalizedNormal(unitNormal);
        State ghost = interiorState;

        if (condition.type == BoundaryType::BCFar)
        {
            // Project onto the general-alpha modes of Gamma^{-1} A_n and import only
            // characteristics entering through the outward face. At a defective collision,
            // import the complete collided cluster and retain the separated outgoing mode.
            const Matrix3 basis = BuildLocalBasis(normal);
            const State interiorLocal = ToLocalState(interiorState, basis);
            const State farLocal = ToLocalState(boundaryValue, basis);
            const State increment = farLocal - interiorLocal;
            const Eigenvalues eigenvalues = ComputeEigenvalues(
                interiorLocal(0), settings.rho0, settings.beta2, settings.alpha);
            const real denominator = eigenvalues.lambdaPlus - eigenvalues.lambdaMinus;
            DNDS_check_throw_info(denominator > std::numeric_limits<real>::epsilon(),
                                  "ACM far-field eigenvalues are degenerate");
            const real amplitudeMinus =
                (eigenvalues.lambdaPlus * increment(3) - settings.beta2 * increment(0)) / denominator;
            const real amplitudePlus =
                (settings.beta2 * increment(0) - eigenvalues.lambdaMinus * increment(3)) / denominator;
            State exteriorLocal = interiorLocal;
            Matrix4 leftEigenvectors;
            Matrix4 rightEigenvectors;
            if (TryCharacteristicMatricesLocal(
                    interiorLocal, settings, leftEigenvectors, rightEigenvectors, 1e-10))
            {
                const State amplitudes = leftEigenvectors * increment;
                const std::array<real, 4> waveSpeeds{
                    eigenvalues.lambdaMinus,
                    eigenvalues.lambdaTangential,
                    eigenvalues.lambdaTangential,
                    eigenvalues.lambdaPlus};
                for (int wave = 0; wave < 4; wave++)
                    if (waveSpeeds[static_cast<std::size_t>(wave)] < 0)
                        exteriorLocal += amplitudes(wave) * rightEigenvectors.col(wave);
            }
            else
            {
                const real minusSeparation =
                    std::abs(interiorLocal(0) - eigenvalues.lambdaMinus);
                const real plusSeparation =
                    std::abs(interiorLocal(0) - eigenvalues.lambdaPlus);
                if (minusSeparation < plusSeparation)
                {
                    // q_n=lambda_minus<0: the collided acoustic/tangential cluster enters.
                    const State rightPlus = AcousticRightEigenvectorLocal(
                        interiorLocal,
                        eigenvalues.lambdaPlus,
                        settings.beta2,
                        settings.alpha);
                    exteriorLocal += increment - amplitudePlus * rightPlus;
                }
                else
                {
                    // q_n=lambda_plus>0: only the separated minus-acoustic mode enters.
                    const State rightMinus = AcousticRightEigenvectorLocal(
                        interiorLocal,
                        eigenvalues.lambdaMinus,
                        settings.beta2,
                        settings.alpha);
                    exteriorLocal += amplitudeMinus * rightMinus;
                }
            }
            ghost.head<3>() = basis * exteriorLocal.head<3>();
            ghost(3) = exteriorLocal(3);
        }
        else if (condition.type == BoundaryType::BCWall ||
                 condition.type == BoundaryType::BCWallIsothermal)
        {
            // Constant-density ACM has no temperature/energy state. BCWallIsothermal therefore
            // enforces the same velocity and zero-normal-pressure-gradient data as BCWall.
            ghost.head<3>() = 2 * boundaryValue.head<3>() - interiorState.head<3>();
        }
        else if (condition.type == BoundaryType::BCWallInvis)
        {
            const Vector3 relativeVelocity = interiorState.head<3>() - boundaryValue.head<3>();
            ghost.head<3>() = interiorState.head<3>() - 2 * relativeVelocity.dot(normal) * normal;
        }
        else if (condition.type == BoundaryType::BCOut)
            ghost = interiorState;
        else if (condition.type == BoundaryType::BCOutP)
            ghost(3) = 2 * boundaryValue(3) - interiorState(3);
        else if (condition.type == BoundaryType::BCIn)
            ghost = boundaryValue;
        else if (condition.type == BoundaryType::BCInPsTs)
        {
            // Total temperature is unavailable in `[u,v,w,p]`; `value[0:3]` is interpreted as
            // prescribed face velocity while pressure is extrapolated from the interior.
            ghost.head<3>() = 2 * boundaryValue.head<3>() - interiorState.head<3>();
        }
        else if (condition.type == BoundaryType::BCSym)
            ghost.head<3>() = interiorState.head<3>() -
                              2 * interiorState.head<3>().dot(normal) * normal;
        else if (condition.type == BoundaryType::BCSpecial)
        {
            DNDS_check_throw_info(condition.specialOption == 0,
                                  "ACM BCSpecial currently supports specialOption = 0 only");
            ghost = boundaryValue;
        }
        else
            DNDS_check_throw_info(false, "unknown ACM boundary type");
        return ghost;
    }

    /** @copydoc GenerateBoundaryState */
    State GenerateBoundaryState(
        BoundaryType type,
        const State &interiorState,
        const State &boundaryValue,
        const Vector3 &unitNormal)
    {
        BoundaryCondition condition;
        condition.type = type;
        Eigen::Map<State>(condition.value.data()) = boundaryValue;
        return GenerateBoundaryState(condition, interiorState, unitNormal, Settings{});
    }

    /** @copydoc EvaluateFaceFluxes */
    void EvaluateFaceFluxes(
        const std::vector<FaceInput> &faces,
        std::vector<FluxResult> &faceFluxBuffer,
        const Settings &settings)
    {
        settings.Validate();
        for (const auto &face : faces)
        {
            DNDS_check_throw_info(face.left.allFinite() && face.right.allFinite(),
                                  "ACM face state contains a non-finite value");
            static_cast<void>(NormalizedNormal(face.unitNormal));
        }
        faceFluxBuffer.resize(faces.size());
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < static_cast<index>(faces.size()); iFace++)
        {
            faceFluxBuffer[static_cast<std::size_t>(iFace)] = InviscidFlux(
                settings.riemannSolverType,
                faces[static_cast<std::size_t>(iFace)].left,
                faces[static_cast<std::size_t>(iFace)].right,
                faces[static_cast<std::size_t>(iFace)].unitNormal,
                settings);
        }
    }

    /** @copydoc LocalFluxSum */
    State LocalFluxSum(const std::vector<FluxResult> &faceFluxBuffer)
    {
        State sum = State::Zero();
        for (const auto &faceFlux : faceFluxBuffer)
            sum += faceFlux.flux;
        return sum;
    }

    /** @copydoc GlobalFluxSum */
    State GlobalFluxSum(const State &localFluxSum, const MPIInfo &mpi)
    {
        State globalFluxSum = State::Zero();
        MPI::Allreduce(
            localFluxSum.data(),
            globalFluxSum.data(),
            4,
            DNDS_MPI_REAL,
            MPI_SUM,
            mpi.comm);
        return globalFluxSum;
    }

    /** @copydoc KernelConfiguration::Validate */
    void KernelConfiguration::Validate() const
    {
        acmSettings.Validate();
        timeMarchSettings.Validate();
        turbulenceSettings.Validate();
        DNDS_check_throw_info(
            !IsBDF2DualTimeIntegrator(timeMarchSettings.integrator) ||
                TurbulenceVariableCount(turbulenceSettings.model) == 0,
            "ACM BDF2 dual-time marching currently supports Laminar flow only; "
            "segregated turbulence transport has no physical-time history");
        if (TurbulenceVariableCount(turbulenceSettings.model) > 0)
        {
            DNDS_check_throw_info(
                acmSettings.enableViscousFlux,
                "ACM turbulence models require acmSettings.enableViscousFlux=true");
            DNDS_check_throw_info(
                std::isfinite(acmSettings.dynamicViscosity) &&
                    acmSettings.dynamicViscosity > 0,
                "ACM turbulence models require positive dynamicViscosity");
        }
        const State left = LeftState();
        const State right = RightState();
        const State initial = InitialState();
        const State boundary = BoundaryValue();
        const Vector3 normal = UnitNormal();
        DNDS_check_throw_info(left.allFinite() && right.allFinite() &&
                                  initial.allFinite() && boundary.allFinite(),
                              "ACM configured state contains a non-finite value");
        DNDS_check_throw_info(normal.allFinite() && normal.norm() > normalTolerance,
                              "ACM preview unitNormal is invalid");
        const auto finiteArray = [](const auto &values)
        {
            return std::all_of(values.begin(), values.end(), [](real value)
                               { return std::isfinite(value); });
        };
        DNDS_check_throw_info(
            finiteArray(meshSettings.periodicTranslation1) &&
                finiteArray(meshSettings.periodicTranslation2) &&
                finiteArray(meshSettings.periodicTranslation3),
            "ACM periodic translation contains a non-finite value");
        DNDS_check_throw_info(
            reconstructionSettings.variationalIterations > 0,
            "ACM variationalIterations must be positive");
        DNDS_check_throw_info(
            std::isfinite(reconstructionSettings.variationalTolerance) &&
                reconstructionSettings.variationalTolerance >= 0 &&
                reconstructionSettings.variationalMaxIterations > 0 &&
                reconstructionSettings.variationalCheckInterval > 0 &&
                std::isfinite(reconstructionSettings.variationalRelaxation) &&
                reconstructionSettings.variationalRelaxation > 0 &&
                reconstructionSettings.variationalRelaxation <= 1 &&
                reconstructionSettings.variationalGMRESSubspace >= 2 &&
                reconstructionSettings.variationalGMRESRestarts >= 0 &&
                std::isfinite(reconstructionSettings.variationalGMRESRelativeTolerance) &&
                reconstructionSettings.variationalGMRESRelativeTolerance > 0 &&
                reconstructionSettings.variationalGMRESRelativeTolerance <= 1,
            "ACM reconstruction tolerance must be finite/non-negative and iteration controls positive");
        DNDS_check_throw_info(
            reconstructionSettings.variationalTolerance == 0 ||
                reconstructionSettings.variationalMaxIterations >= reconstructionSettings.variationalIterations,
            "ACM variationalMaxIterations must cover the minimum variationalIterations");
        DNDS_check_throw_info(
            reconstructionSettings.variationalTolerance == 0 ||
                !(vfvSettings.maxOrder == 1 && vfvSettings.subs2ndOrder != 0),
            "ACM reconstruction equation convergence requires the variational operator, not substituted second-order reconstruction");
        DNDS_check_throw_info(
            !reconstructionSettings.enableLimiter ||
                reconstructionSettings.limiterType == LimiterType::LocalExtrema ||
                reconstructionSettings.type == ReconstructionType::Variational,
            "ACM WBAP/CWBAP requires Variational reconstruction");
        for (const auto &condition : boundaryConditions)
        {
            DNDS_check_throw_info(!condition.name.empty(), "ACM boundary zone name cannot be empty");
            DNDS_check_throw_info(condition.type != BoundaryType::BCUnknown,
                                  "ACM boundary zone cannot use BCUnknown");
            DNDS_check_throw_info(condition.ValueState().allFinite(),
                                  "ACM configured boundary state is non-finite");
        }
        DNDS_check_throw_info(nFacesPerRank > 0, "ACM nFacesPerRank must be positive");
        DNDS_check_throw_info(
            restartSettings.flowFile.empty() || restartSettings.writerRanks > 0,
            "ACM VTK-HDF restart requires a positive writerRanks value");
        DNDS_check_throw_info(
            restartSettings.flowFile.empty() ||
                (timeMarchSettings.integrator != TimeIntegratorType::BDF2DualTimeLUSGS &&
                 timeMarchSettings.integrator != TimeIntegratorType::BDF2DualTimeGMRES),
            "ACM VTK-HDF restart contains no BDF2 history and cannot initialize dual-time marching");
    }

    /** @copydoc KernelConfiguration::LeftState */
    State KernelConfiguration::LeftState() const
    {
        return Eigen::Map<const State>(leftState.data());
    }

    /** @copydoc KernelConfiguration::RightState */
    State KernelConfiguration::RightState() const
    {
        return Eigen::Map<const State>(rightState.data());
    }

    /** @copydoc KernelConfiguration::UnitNormal */
    Vector3 KernelConfiguration::UnitNormal() const
    {
        return Eigen::Map<const Vector3>(unitNormal.data());
    }

    /** @copydoc KernelConfiguration::InitialState */
    State KernelConfiguration::InitialState() const
    {
        return Eigen::Map<const State>(initialState.data());
    }

    /** @copydoc KernelConfiguration::BoundaryValue */
    State KernelConfiguration::BoundaryValue() const
    {
        return Eigen::Map<const State>(boundaryValue.data());
    }

    /** @copydoc LoadConfiguration */
    LoadedConfiguration LoadConfiguration(
        const std::string &jsonName,
        const std::vector<std::string> &overwriteKeys,
        const std::vector<std::string> &overwriteValues)
    {
        DNDS_check_throw_info(overwriteKeys.size() == overwriteValues.size(),
                              "ACM overwrite keys and values have different lengths");

        std::ifstream input(jsonName);
        DNDS_check_throw_info(input.good(), "ACM case configuration file does not exist: " + jsonName);
        nlohmann::ordered_json resolved = nlohmann::ordered_json::parse(input, nullptr, true, true);

        for (std::size_t i = 0; i < overwriteKeys.size(); i++)
        {
            const nlohmann::ordered_json::json_pointer key(overwriteKeys[i]);
            try
            {
                resolved[key] = nlohmann::ordered_json::parse(overwriteValues[i]);
            }
            catch (const nlohmann::ordered_json::parse_error &)
            {
                resolved[key] = overwriteValues[i];
            }
        }

        // Keep pre-BDF2 case files valid without editing them in place.  The
        // normalized configuration and emitted schema still expose this key,
        // and a command-line JSON-pointer override may set it explicitly.
        auto &timeMarch = resolved.at("timeMarchSettings");
        if (!timeMarch.contains("physicalTimeStep"))
            timeMarch["physicalTimeStep"] = TimeMarchSettings{}.physicalTimeStep;
        const nlohmann::ordered_json timeDefaults = TimeMarchSettings{};
        for (const char *key : {"steadyRelativeTolerance", "steadyAdaptiveCFL", "steadyCFLMin", "steadyCFLMax",
                                "steadyCFLGrowth", "steadyCFLReduction"})
            if (!timeMarch.contains(key))
                timeMarch[key] = timeDefaults.at(key);
        auto &reconstruction = resolved.at("reconstructionSettings");
        const nlohmann::ordered_json reconstructionDefaults = ReconstructionSettings{};
        for (const char *key : {"variationalTolerance", "variationalMaxIterations", "variationalCheckInterval",
                                "variationalRelaxation", "variationalUseGMRES", "variationalGMRESSubspace",
                                "variationalGMRESRestarts", "variationalGMRESRelativeTolerance"})
            if (!reconstruction.contains(key))
                reconstruction[key] = reconstructionDefaults.at(key);
        if (!resolved.contains("restartSettings"))
            resolved["restartSettings"] = RestartSettings{};

        KernelConfiguration configuration = resolved.get<KernelConfiguration>();
        configuration.Validate();
        nlohmann::ordered_json normalized = configuration;
        return {configuration, normalized};
    }
}
