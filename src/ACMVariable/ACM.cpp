/**
 * @file ACM.cpp
 * @brief Implementation of variable-density ACM state, flux, boundary, parallel, and configuration kernels.
 *
 * @details This translation unit intentionally remains independent of the Euler evaluator. It provides
 * modular kernels that can later be assembled into a dedicated mesh-based ACM evaluator while reusing
 * DNDS configuration, OpenMP face loops, and MPI collectives.
 *
 * @author Runzhi Ma
 * @date 2026-09-03
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

namespace DNDS::ACMVariable
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
        DNDS_check_throw_info(IsFiniteState(state), "ACMVariable requires finite state and positive density");
        return state.segment<3>(1) / state(0);
    }

    /** @copydoc PhysicalPressure */
    real PhysicalPressure(const State &state) { return state(4); }

    /** @copydoc SetPhysicalPressure */
    void SetPhysicalPressure(State &state, real pressure) { state(4) = pressure; }

    /** @copydoc ToLocalState */
    State ToLocalState(const State &state, const Matrix3 &localBasis)
    {
        State local = state;
        local.segment<3>(1) = localBasis.transpose() * state.segment<3>(1);
        return local;
    }

    /** @copydoc FromLocalFlux */
    State FromLocalFlux(const State &localFlux, const Matrix3 &localBasis)
    {
        State flux = localFlux;
        flux.segment<3>(1) = localBasis * localFlux.segment<3>(1);
        return flux;
    }

    /** @copydoc RotateStateInPlace */
    void RotateStateInPlace(State &state, const Matrix3 &rotation)
    {
        state.segment<3>(1) = rotation * state.segment<3>(1);
    }

    /** @copydoc IsFiniteState */
    bool IsFiniteState(const State &state) { return state.allFinite() && state(0) > 0; }

    /** @copydoc RoeAverage */
    State RoeAverage(const State &left, const State &right)
    {
        const real sl = std::sqrt(left(0)), sr = std::sqrt(right(0));
        State mean;
        mean(0) = sl * sr;
        mean.segment<3>(1) = mean(0) * (sl * Velocity(left) + sr * Velocity(right)) / (sl + sr);
        mean(4) = 0.5 * (left(4) + right(4));
        return mean;
    }

    /** @copydoc PrimitiveGradient */
    State PrimitiveState(const State &state)
    {
        State primitive = state;
        primitive.segment<3>(1) = Velocity(state);
        return primitive;
    }

    /** @copydoc PrimitiveGradient */
    Eigen::Matrix<real, 3, 5> PrimitiveGradient(
        const State &state, const Eigen::Matrix<real, 3, 5> &gradient)
    {
        Eigen::Matrix<real, 3, 5> result = gradient;
        result.block<3, 3>(0, 1) =
            (gradient.block<3, 3>(0, 1) - gradient.col(0) * Velocity(state).transpose()) / state(0);
        return result;
    }

    /** @copydoc Eigenvalues::SpectralRadius */
    real Eigenvalues::SpectralRadius() const
    {
        return std::max({std::abs(lambdaMinus), std::abs(lambdaTangential), std::abs(lambdaPlus)});
    }

    /** @copydoc PhysicalFluxLocal */
    State PhysicalFluxLocal(const State &localState)
    {
        const Vector3 velocity = Velocity(localState);
        State flux;
        flux(0) = localState(1);
        flux.segment<3>(1) = localState.segment<3>(1) * velocity(0);
        flux(1) += localState(4);
        flux(4) = velocity(0);
        return flux;
    }

    /** @copydoc PhysicalFluxJacobianLocal */
    Matrix5 PhysicalFluxJacobianLocal(const State &localState)
    {
        const Vector3 u = Velocity(localState);
        Matrix5 jacobian = Matrix5::Zero();
        jacobian(0, 1) = 1;
        jacobian.block<3, 1>(1, 0) = -u * u(0);
        jacobian.block<3, 3>(1, 1) = u(0) * Matrix3::Identity();
        jacobian.block<3, 1>(1, 1) += u;
        jacobian(1, 4) = 1;
        jacobian(4, 0) = -u(0) / localState(0);
        jacobian(4, 1) = 1 / localState(0);
        return jacobian;
    }

    /** @copydoc GammaLocal */
    Matrix5 GammaLocal(const State &state, real beta2, real alpha)
    {
        DNDS_check_throw_info(IsFiniteState(state) && beta2 > 0, "Invalid ACMVariable Gamma state");
        Matrix5 gamma = Matrix5::Identity();
        gamma(0, 4) = (alpha + 1) * state(0) / beta2;
        gamma.block<3, 1>(1, 4) = (2 * alpha + 1) * state.segment<3>(1) / beta2;
        gamma(4, 4) = 1 / beta2;
        return gamma;
    }

    /** @copydoc GammaInvLocal */
    Matrix5 GammaInvLocal(const State &state, real beta2, real alpha)
    {
        DNDS_check_throw_info(IsFiniteState(state) && beta2 > 0, "Invalid ACMVariable inverse Gamma state");
        Matrix5 inverse = Matrix5::Identity();
        inverse(0, 4) = -(alpha + 1) * state(0);
        inverse.block<3, 1>(1, 4) = -(2 * alpha + 1) * state.segment<3>(1);
        inverse(4, 4) = beta2;
        return inverse;
    }

    /** @copydoc ApplyGammaLocal */
    State ApplyGammaLocal(const State &state, const State &increment, real beta2, real alpha)
    {
        return GammaLocal(state, beta2, alpha) * increment;
    }

    /** @copydoc PreconditionedJacobianLocal */
    Matrix5 PreconditionedJacobianLocal(const State &state, real beta2, real alpha)
    {
        return GammaInvLocal(state, beta2, alpha) * PhysicalFluxJacobianLocal(state);
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
        // The acoustic roots have product -beta2/rho. Recover the smaller root
        // from that product to avoid cancellation for large normal velocities.
        // Author: Runzhi Ma. qn and rho0 are local velocity and density, not references.
        const real largeRoot = 0.5 * centeredVelocity + 0.5 * std::copysign(root, centeredVelocity);
        const real smallRoot = -(beta2 / rho0) / largeRoot;
        return {std::min(largeRoot, smallRoot), qn, std::max(largeRoot, smallRoot)};
    }

    /** @copydoc PseudoTimeProductJacobian */
    Matrix5 PseudoTimeProductJacobian(const State &state, const State &previous,
                                    real pseudoTimeStep, const Settings &settings)
    {
        DNDS_check_throw_info(IsFiniteState(state) && IsFiniteState(previous) &&
                                 std::isfinite(pseudoTimeStep) && pseudoTimeStep > 0,
                             "Invalid variable ACM pseudo-time product Jacobian input");
        Matrix5 result = GammaLocal(state, settings.beta2, settings.alpha);
        const real pressureJumpScaled = (state(4) - previous(4)) / settings.beta2;
        result(0, 0) += (settings.alpha + 1) * pressureJumpScaled;
        for (int k = 1; k < 4; ++k)
            result(k, k) += (2 * settings.alpha + 1) * pressureJumpScaled;
        return result / pseudoTimeStep;
    }

    namespace
    {
        /**
         * @brief Construct one pressure-normalized general-alpha acoustic right eigenvector.
         * @param meanLocalState Local mean state `[rho,m_n,m_t1,m_t2,p]`.
         * @param lambda Acoustic eigenvalue associated with the requested vector.
         * @param beta2 Positive artificial-compressibility parameter.
         * @param alpha Turkel coupling parameter.
         * @return Right eigenvector with pressure component equal to one.
         * @note Modifier: Runzhi Ma.
         */
        State AcousticRightEigenvectorLocal(
            const State &state, real lambda, real beta2, real alpha)
        {
            const Vector3 u = Velocity(state);
            const real separation = lambda - u(0);
            DNDS_check_throw_info(std::abs(separation) > std::numeric_limits<real>::epsilon(),
                                  "ACMVariable acoustic eigenvector is singular");
            State primitive = State::Zero();
            primitive(0) = -alpha * state(0) * lambda / (beta2 * separation);
            primitive(1) = lambda / beta2;
            primitive(2) = -alpha * u(1) * lambda / (beta2 * separation);
            primitive(3) = -alpha * u(2) * lambda / (beta2 * separation);
            primitive(4) = 1;
            State conservative = primitive;
            conservative.segment<3>(1) =
                u * primitive(0) + state(0) * primitive.segment<3>(1);
            return conservative;
        }

        /**
         * @brief Build a complete local characteristic basis when the operator is diagonalizable.
         * @param meanLocalState Local mean state `[rho,m_n,m_t1,m_t2,p]`.
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
            Matrix5 &left,
            Matrix5 &right,
            real minimumRelativeSeparation)
        {
            const Eigenvalues eigenvalues = ComputeEigenvalues(
                Velocity(meanLocalState)(0), meanLocalState(0), settings.beta2, settings.alpha);
            const real scale = std::max(
                {real(1), std::abs(Velocity(meanLocalState)(0)), eigenvalues.SpectralRadius()});
            if (std::abs(Velocity(meanLocalState)(0) - eigenvalues.lambdaMinus) <=
                    minimumRelativeSeparation * scale ||
                std::abs(Velocity(meanLocalState)(0) - eigenvalues.lambdaPlus) <=
                    minimumRelativeSeparation * scale)
                return false;

            right.setZero();
            right.col(0) = AcousticRightEigenvectorLocal(
                meanLocalState, eigenvalues.lambdaMinus, settings.beta2, settings.alpha);
            right(0, 1) = 1;
            right.block<3, 1>(1, 1) = Velocity(meanLocalState);
            right(2, 2) = meanLocalState(0);
            right(3, 3) = meanLocalState(0);
            right.col(4) = AcousticRightEigenvectorLocal(
                meanLocalState, eigenvalues.lambdaPlus, settings.beta2, settings.alpha);

            const Eigen::JacobiSVD<Matrix5> svd(right);
            const auto singularValues = svd.singularValues();
            if (!(singularValues(0) > 0) ||
                singularValues(4) <= minimumRelativeSeparation * singularValues(0))
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
        Matrix5 &left,
        Matrix5 &right,
        real minimumRelativeSeparation)
    {
        settings.Validate();
        DNDS_check_throw_info(
            std::isfinite(minimumRelativeSeparation) && minimumRelativeSeparation > 0,
            "ACM characteristic separation tolerance must be finite and positive");
        const Matrix3 basis = BuildLocalBasis(unitNormal);
        const State meanLocal = ToLocalState(meanState, basis);
        Matrix5 leftLocal;
        Matrix5 rightLocal;
        if (!TryCharacteristicMatricesLocal(
                meanLocal, settings, leftLocal, rightLocal, minimumRelativeSeparation))
            return false;

        Matrix5 localToGlobal = Matrix5::Identity();
        localToGlobal.block<3, 3>(1, 1) = basis;
        Matrix5 globalToLocal = Matrix5::Identity();
        globalToLocal.block<3, 3>(1, 1) = basis.transpose();
        right = localToGlobal * rightLocal;
        left = leftLocal * globalToLocal;
        return left.allFinite() && right.allFinite();
    }

    /** @copydoc RightEigenvectorsGlobal */
    Matrix5 RightEigenvectorsGlobal(
        const State &meanState,
        const Vector3 &unitNormal,
        const Settings &settings)
    {
        Matrix5 left;
        Matrix5 right;
        DNDS_check_throw_info(
            TryCharacteristicMatricesGlobal(meanState, unitNormal, settings, left, right),
            "ACM normal operator is defective at an acoustic/tangential eigenvalue collision");
        return right;
    }

    /** @copydoc LeftEigenvectorsGlobal */
    Matrix5 LeftEigenvectorsGlobal(
        const State &meanState,
        const Vector3 &unitNormal,
        const Settings &settings)
    {
        Matrix5 left;
        Matrix5 right;
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
        Matrix5 EntropyFixedAbsoluteJacobianHermite(
            const Matrix5 &preconditionedJacobian,
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

            const Matrix5 identity = Matrix5::Identity();
            const Matrix5 shiftedSeparated =
                preconditionedJacobian - separated * identity;
            const Matrix5 shiftedTangential =
                preconditionedJacobian - q * identity;
            const Matrix5 absoluteJacobian =
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

    /** @copydoc AbsolutePreconditionedJacobianLocal */
    Matrix5 AbsolutePreconditionedJacobianLocal(const State &state, const Settings &settings)
    {
        const auto speeds = ComputeEigenvalues(Velocity(state)(0), state(0), settings.beta2, settings.alpha);
        const real delta = settings.entropyFixRatio *
            std::max(speeds.SpectralRadius(), std::sqrt(settings.beta2 / state(0)));
        return EntropyFixedAbsoluteJacobianHermite(
            PreconditionedJacobianLocal(state, settings.beta2, settings.alpha), speeds, delta);
    }

    /** @copydoc RusanovDissipationLocal */
    State RusanovDissipationLocal(
        const State &leftLocal,
        const State &rightLocal,
        const Settings &settings,
        Eigenvalues &eigenvalues)
    {
        settings.Validate();
        const State meanState = RoeAverage(leftLocal, rightLocal);
        const Eigenvalues meanEigenvalues = ComputeEigenvalues(
            Velocity(meanState)(0), meanState(0), settings.beta2, settings.alpha);
        const Eigenvalues leftEigenvalues = ComputeEigenvalues(
            Velocity(leftLocal)(0), leftLocal(0), settings.beta2, settings.alpha);
        const Eigenvalues rightEigenvalues = ComputeEigenvalues(
            Velocity(rightLocal)(0), rightLocal(0), settings.beta2, settings.alpha);
        // Bound both endpoint spectra; the Roe density alone can underestimate
        // the acoustic speed on the lower-density side. Author: Runzhi Ma.
        eigenvalues = meanEigenvalues;
        if (leftEigenvalues.SpectralRadius() > eigenvalues.SpectralRadius()) eigenvalues = leftEigenvalues;
        if (rightEigenvalues.SpectralRadius() > eigenvalues.SpectralRadius()) eigenvalues = rightEigenvalues;
        const real speed = eigenvalues.SpectralRadius();
        return speed *
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
        const State meanState = RoeAverage(leftLocal, rightLocal);
        const State increment = rightLocal - leftLocal;
        eigenvalues = ComputeEigenvalues(
            Velocity(meanState)(0), meanState(0), settings.beta2, settings.alpha);

        const real entropyDelta = settings.entropyFixRatio *
                                  std::max({eigenvalues.SpectralRadius(),
                                            std::sqrt(settings.beta2 / meanState(0))});
        const real lambdaMinusAbs = EntropyFixedAbs(eigenvalues.lambdaMinus, entropyDelta);
        const real lambdaTangentialAbs = EntropyFixedAbs(eigenvalues.lambdaTangential, entropyDelta);
        const real lambdaPlusAbs = EntropyFixedAbs(eigenvalues.lambdaPlus, entropyDelta);
        Matrix5 leftEigenvectors;
        Matrix5 rightEigenvectors;
        State preconditionedDissipation;
        if (TryCharacteristicMatricesLocal(
                meanState, settings, leftEigenvectors, rightEigenvectors, 1e-10))
        {
            Matrix5 absoluteEigenvalues = Matrix5::Zero();
            absoluteEigenvalues.diagonal() << lambdaMinusAbs,
                lambdaTangentialAbs,
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
            const Matrix5 preconditionedJacobian = PreconditionedJacobianLocal(
                meanState,
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
        DNDS_check_throw_info(IsFiniteState(left) && IsFiniteState(right), "ACMVariable invalid interface density/state");
        const Matrix3 localBasis = BuildLocalBasis(unitNormal);
        const State leftLocal = ToLocalState(left, localBasis);
        const State rightLocal = ToLocalState(right, localBasis);
        const State leftFlux = PhysicalFluxLocal(leftLocal);
        const State rightFlux = PhysicalFluxLocal(rightLocal);

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
    Eigen::Matrix<real, 3, 5> CorrectedFaceGradient(
        const Eigen::Matrix<real, 3, 5> &leftGradient,
        const Eigen::Matrix<real, 3, 5> &rightGradient,
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

        Eigen::Matrix<real, 3, 5> faceGradient =
            0.5 * (leftGradient + rightGradient);
        const Eigen::RowVector<real, 5> centerMismatch =
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
        const Eigen::Matrix<real, 3, 5> &velocityStateGradient,
        const Vector3 &unitNormal,
        real dynamicViscosity)
    {
        DNDS_check_throw_info(velocityStateGradient.allFinite() && dynamicViscosity >= 0,
                              "Invalid ACMVariable viscous data");
        const Matrix3 gradient = velocityStateGradient.block<3, 3>(0, 1).transpose();
        const Matrix3 stress = dynamicViscosity *
            (gradient + gradient.transpose() - (2.0 / 3.0) * gradient.trace() * Matrix3::Identity());
        State flux = State::Zero();
        flux.segment<3>(1) = stress * NormalizedNormal(unitNormal);
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
        stationaryWall.value = {1, 0, 0, 0, 0};
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
        settings.Validate();
        const State boundaryValue = condition.ValueState();
        DNDS_check_throw_info(interiorState.allFinite() && boundaryValue.allFinite(),
                              "ACM boundary state contains a non-finite value");
        const Vector3 normal = NormalizedNormal(unitNormal);
        State ghost = interiorState;

        const Vector3 ui = Velocity(interiorState), ub = Velocity(boundaryValue);
        if (condition.type == BoundaryType::BCFar)
        {
            const Matrix3 basis = BuildLocalBasis(normal);
            const State interiorLocal = ToLocalState(interiorState, basis);
            const State jump = ToLocalState(boundaryValue, basis) - interiorLocal;
            const auto eigenvalues = ComputeEigenvalues(ui.dot(normal), interiorState(0),
                                                        settings.beta2, settings.alpha);
            Matrix5 left, right;
            State incoming = State::Zero();
            if (TryCharacteristicMatricesLocal(interiorLocal, settings, left, right, 1e-10))
            {
                State amplitudes = left * jump;
                const std::array<real, 5> speeds{eigenvalues.lambdaMinus, eigenvalues.lambdaTangential,
                    eigenvalues.lambdaTangential, eigenvalues.lambdaTangential, eigenvalues.lambdaPlus};
                for (int k = 0; k < 5; k++)
                    if (speeds[k] < 0) incoming += amplitudes(k) * right.col(k);
            }
            else
            {
                // Spectral projector for the isolated acoustic wave. The collided cluster
                // has one sign; its projector is I-P. No singular basis is inverted.
                const real q = ui.dot(normal);
                const bool qNegative = q < 0;
                const real separated = qNegative ? eigenvalues.lambdaPlus : eigenvalues.lambdaMinus;
                const real clustered = qNegative ? eigenvalues.lambdaMinus : eigenvalues.lambdaPlus;
                const Matrix5 B = PreconditionedJacobianLocal(interiorLocal, settings.beta2, settings.alpha);
                const Matrix5 I = Matrix5::Identity();
                const Matrix5 P = (B - q * I) * (B - clustered * I) /
                    ((separated - q) * (separated - clustered));
                incoming = qNegative ? State((I - P) * jump) : State(P * jump);
            }
            ghost = FromLocalFlux(interiorLocal + incoming, basis);
        }
        else if (condition.type == BoundaryType::BCWall ||
                 condition.type == BoundaryType::BCWallIsothermal)
            ghost.segment<3>(1) = interiorState(0) * (2 * ub - ui);
        else if (condition.type == BoundaryType::BCWallInvis)
            ghost.segment<3>(1) = interiorState(0) * (ui - 2 * (ui - ub).dot(normal) * normal);
        else if (condition.type == BoundaryType::BCSym)
            ghost.segment<3>(1) = interiorState(0) * (ui - 2 * ui.dot(normal) * normal);
        else if (condition.type == BoundaryType::BCOut)
            ghost = interiorState;
        else if (condition.type == BoundaryType::BCOutP)
            ghost(4) = 2 * boundaryValue(4) - interiorState(4);
        else if (condition.type == BoundaryType::BCIn)
            ghost = boundaryValue;
        else if (condition.type == BoundaryType::BCInPsTs)
        {
            // No energy/EOS: this alias means density and velocity inlet, NOT total T.
            ghost(0) = boundaryValue(0);
            ghost.segment<3>(1) = ghost(0) * (2 * ub - ui);
        }
        else if (condition.type == BoundaryType::BCSpecial)
        {
            ghost = boundaryValue;
            if (condition.specialOption == 1)
            {
                // Exact smooth advected-density contact for manufactured verification.
                // valueExtra = [relative amplitude, kx, ky, kz], radians/length.
                // value supplies reference density, conservative momentum and pressure.
                // Author: Runzhi Ma. This is not Euler's compressible vortex boundary.
                DNDS_check_throw_info(condition.valueExtra.size() == 4 &&
                    std::abs(condition.valueExtra[0]) < 1 && point.allFinite() && std::isfinite(time),
                    "ACMVariable BCSpecial 1 requires finite [amplitude,kx,ky,kz], |amplitude|<1");
                const Vector3 wave = Eigen::Map<const Vector3>(condition.valueExtra.data() + 1);
                ghost(0) *= 1 + condition.valueExtra[0] * std::cos(wave.dot(point - ub * time));
                ghost.segment<3>(1) = ghost(0) * ub;
            }
            else
                DNDS_check_throw_info(condition.specialOption == 0, "Unknown ACMVariable special boundary");
        }
        else
            DNDS_check_throw_info(false, "Unknown ACMVariable boundary");
        DNDS_check_throw_info(IsFiniteState(ghost),
                              "ACMVariable boundary produced nonpositive density; revise boundary data");
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
            5,
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
        DNDS_check_throw_info(IsFiniteState(left) && IsFiniteState(right) &&
                                  IsFiniteState(initial) && IsFiniteState(boundary),
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
            !reconstructionSettings.enableLimiter ||
                reconstructionSettings.limiterType == LimiterType::LocalExtrema ||
                reconstructionSettings.type == ReconstructionType::Variational,
            "ACM WBAP/CWBAP requires Variational reconstruction");
        for (const auto &condition : boundaryConditions)
        {
            DNDS_check_throw_info(!condition.name.empty(), "ACM boundary zone name cannot be empty");
            DNDS_check_throw_info(condition.type != BoundaryType::BCUnknown,
                                  "ACM boundary zone cannot use BCUnknown");
            DNDS_check_throw_info(IsFiniteState(condition.ValueState()),
                                  "ACM configured boundary density/state is invalid");
            DNDS_check_throw_info(condition.frameOption == 0 && condition.anchorOption == 0 &&
                                     condition.integrationOption == 0 && condition.rectifyOption == 0,
                "Moving-frame/anchor/integration/rectification BC options are not implemented in ACMVariable");
        }
        DNDS_check_throw_info(std::isfinite(initialDensityAmplitude) && std::abs(initialDensityAmplitude) < 1 &&
                                 finiteArray(initialDensityWaveNumber) &&
                                 InitialState()(0) * (1 - std::abs(initialDensityAmplitude)) > acmSettings.densityFloor,
                             "Initial variable-density profile violates admissibility");
        DNDS_check_throw_info(nFacesPerRank > 0, "ACM nFacesPerRank must be positive");
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
        if (resolved.at("timeMarchSettings").contains("gmresRelativeTolerance"))
        {
            auto &timeMarch = resolved.at("timeMarchSettings");
            DNDS_check_throw_info(!timeMarch.contains("linearRelativeTolerance") ||
                                      timeMarch.at("linearRelativeTolerance") == timeMarch.at("gmresRelativeTolerance"),
                                  "Specify only linearRelativeTolerance; the legacy gmresRelativeTolerance conflicts");
            timeMarch["linearRelativeTolerance"] = timeMarch.at("gmresRelativeTolerance");
            timeMarch.erase("gmresRelativeTolerance");
        }

        for (std::size_t i = 0; i < overwriteKeys.size(); i++)
        {
            const std::string keyName = overwriteKeys[i] == "/timeMarchSettings/gmresRelativeTolerance"
                                            ? "/timeMarchSettings/linearRelativeTolerance" : overwriteKeys[i];
            const nlohmann::ordered_json::json_pointer key(keyName);
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
        if (!timeMarch.contains("physicalODECode"))
            timeMarch["physicalODECode"] = TimeMarchSettings{}.physicalODECode;
        if (!timeMarch.contains("pseudoODECode"))
            timeMarch["pseudoODECode"] = TimeMarchSettings{}.pseudoODECode;
        const nlohmann::ordered_json timeDefaults = TimeMarchSettings{};
        for (const char *key : {"implicitRelativeTolerance", "linearRelativeTolerance", "linearAbsoluteTolerance"})
            if (!timeMarch.contains(key))
                timeMarch[key] = timeDefaults.at(key);

        KernelConfiguration configuration = resolved.get<KernelConfiguration>();
        configuration.Validate();
        nlohmann::ordered_json normalized = configuration;
        return {configuration, normalized};
    }
}
