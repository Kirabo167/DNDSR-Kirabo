/**
 * @file ACMTime.cpp
 * @brief Implementation of explicit SSPRK3 and implicit backward-Euler ACM pseudo-time stepping.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#include "ACMTime.hpp"

#include "DNDS/Errors.hpp"

#include <Eigen/LU>

#include <cmath>
#include <limits>

namespace DNDS::ACM
{
    namespace
    {
        /** @brief Combine group convergence without allowing an inactive turbulence group to satisfy Either. */
        bool CombineDualTimeGroups(const std::string &criterion, bool flow, bool turbulence, int activeEquations)
        {
            if (criterion == "Both")
                return flow && (activeEquations == 4 || turbulence);
            if (criterion == "Either")
                return flow || (activeEquations > 4 && turbulence);
            if (criterion == "Flow")
                return flow;
            if (criterion == "Turbulence")
            {
                DNDS_check_throw_info(activeEquations > 4,
                                      "ACM Turbulence convergence requires an active turbulence model");
                return turbulence;
            }
            throw std::invalid_argument("unknown ACM dual-time convergence criterion: " + criterion);
        }

        /**
         * @brief Validate field sizes and all local pseudo-time-step values.
         * @param states Rank-local state field.
         * @param pseudoTimeStep Scalar field expected to match `states`.
         */
        void ValidateTimeStepField(const StateField &states, const ScalarField &pseudoTimeStep)
        {
            DNDS_check_throw_info(!states.empty(), "ACM time stepping requires at least one local state");
            DNDS_check_throw_info(
                pseudoTimeStep.size() == states.size(),
                "ACM pseudo-time-step field has a different size from the state field");
            for (std::size_t i = 0; i < states.size(); i++)
            {
                DNDS_check_throw_info(states[i].allFinite(), "ACM time-step input contains a non-finite state");
                DNDS_check_throw_info(
                    std::isfinite(pseudoTimeStep[i]) && pseudoTimeStep[i] > 0,
                    "ACM pseudo-time step must be finite and positive");
            }
        }

        /**
         * @brief Evaluate a callback and enforce the rank-local residual-field contract.
         * @param states Current state field.
         * @param residual Output residual field.
         * @param residualEvaluator User-supplied residual callback.
         */
        void EvaluateResidualChecked(
            const StateField &states,
            StateField &residual,
            const ResidualEvaluator &residualEvaluator)
        {
            DNDS_check_throw_info(static_cast<bool>(residualEvaluator), "ACM residual callback is empty");
            residualEvaluator(states, residual);
            DNDS_check_throw_info(
                residual.size() == states.size(),
                "ACM residual callback returned a field with an invalid size");
            for (const auto &value : residual)
                DNDS_check_throw_info(value.allFinite(), "ACM residual callback returned a non-finite value");
        }

        /**
         * @brief Compute a component-wise global RMS norm, reducing through MPI when requested.
         * @param field Rank-local vector field.
         * @param mpi Optional communicator metadata.
         * @return RMS norm over every component on every participating rank.
         */
        real GlobalRMSNorm(const StateField &field, const MPIInfo *mpi)
        {
            real localSquaredNorm = 0;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for reduction(+ : localSquaredNorm) schedule(runtime)
#endif
            for (index i = 0; i < static_cast<index>(field.size()); i++)
                localSquaredNorm += field[static_cast<std::size_t>(i)].squaredNorm();

            real globalSquaredNorm = localSquaredNorm;
            unsigned long long localComponents = static_cast<unsigned long long>(field.size()) * 4ULL;
            unsigned long long globalComponents = localComponents;
            if (mpi != nullptr)
            {
                int mpiInitialized = 0;
                MPI_Initialized(&mpiInitialized);
                if (mpiInitialized)
                {
                    MPI_Allreduce(
                        &localSquaredNorm,
                        &globalSquaredNorm,
                        1,
                        DNDS_MPI_REAL,
                        MPI_SUM,
                        mpi->comm);
                    MPI_Allreduce(
                        &localComponents,
                        &globalComponents,
                        1,
                        MPI_UNSIGNED_LONG_LONG,
                        MPI_SUM,
                        mpi->comm);
                }
            }
            DNDS_check_throw_info(globalComponents > 0, "ACM global RMS norm has no components");
            return std::sqrt(globalSquaredNorm / static_cast<real>(globalComponents));
        }

        /**
         * @brief Form the nonlinear backward-Euler defect `R(U)-Gamma(U)(U-Uold)/dTau`.
         * @param states Current nonlinear iterate.
         * @param statesOld State at the beginning of the physical pseudo-time step.
         * @param residual Raw spatial residual at `states`.
         * @param pseudoTimeStep Local pseudo-time steps.
         * @param settings ACM preconditioning settings.
         * @param defect Output backward-Euler defect.
         */
        void FormImplicitDefect(
            const StateField &states,
            const StateField &statesOld,
            const StateField &residual,
            const ScalarField &pseudoTimeStep,
            const Settings &settings,
            StateField &defect)
        {
            defect.resize(states.size());
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index i = 0; i < static_cast<index>(states.size()); i++)
            {
                const std::size_t ii = static_cast<std::size_t>(i);
                defect[ii] = residual[ii] -
                             GammaLocal(states[ii], settings.beta2, settings.alpha) *
                                 (states[ii] - statesOld[ii]) / pseudoTimeStep[ii];
            }
        }
    }

    /** @copydoc TimeMarchSettings::Validate */
    void TimeMarchSettings::Validate() const
    {
        DNDS_check_throw_info(nSteps >= 0, "ACM nSteps must be non-negative");
        DNDS_check_throw_info(
            std::isfinite(pseudoTimeStep) && pseudoTimeStep > 0,
            "ACM pseudoTimeStep must be finite and positive");
        DNDS_check_throw_info(
            std::isfinite(physicalTimeStep) && physicalTimeStep > 0,
            "ACM physicalTimeStep must be finite and positive");
        DNDS_check_throw_info(std::isfinite(cfl) && cfl > 0, "ACM cfl must be finite and positive");
        DNDS_check_throw_info(
            std::isfinite(maximumPseudoTimeStep) && maximumPseudoTimeStep > 0,
            "ACM maximumPseudoTimeStep must be finite and positive");
        DNDS_check_throw_info(maxImplicitIterations > 0, "ACM maxImplicitIterations must be positive");
        DNDS_check_throw_info(
            std::isfinite(implicitTolerance) && implicitTolerance >= 0,
            "ACM implicitTolerance must be finite and non-negative");
        DNDS_check_throw_info(
            std::isfinite(dualTimeFlowRelativeTolerance) && dualTimeFlowRelativeTolerance >= 0 && dualTimeFlowRelativeTolerance < 1 &&
                std::isfinite(dualTimeTurbulenceRelativeTolerance) && dualTimeTurbulenceRelativeTolerance >= 0 && dualTimeTurbulenceRelativeTolerance < 1,
            "ACM BDF grouped relative tolerances must be finite and in [0,1)");
        const bool physicalBDF = integrator == TimeIntegratorType::BDF2DualTimeLUSGS ||
                                 integrator == TimeIntegratorType::BDF2DualTimeGMRES;
        DNDS_check_throw_info(dualTimeConvergenceCriterion == "Both" || dualTimeConvergenceCriterion == "Either" ||
                                  dualTimeConvergenceCriterion == "Flow" || dualTimeConvergenceCriterion == "Turbulence",
                              "ACM dualTimeConvergenceCriterion must be Both, Either, Flow, or Turbulence");
        DNDS_check_throw_info(physicalBDF || dualTimeConvergenceCriterion == "Both",
                              "ACM dual-time convergence selection requires a BDF2 dual-time integrator");
        DNDS_check_throw_info(physicalBDF || (dualTimeFlowRelativeTolerance == 0 && dualTimeTurbulenceRelativeTolerance == 0),
                              "ACM grouped physical relative tolerances require a BDF2 dual-time integrator");
        for (std::size_t equation = 0; equation < dualTimeResidualScales.size(); equation++)
        {
            DNDS_check_throw_info(
                std::isfinite(dualTimeResidualScales[equation]) && dualTimeResidualScales[equation] > 0 &&
                    std::isfinite(dualTimeRelativeTolerances[equation]) &&
                    dualTimeRelativeTolerances[equation] >= 0 && dualTimeRelativeTolerances[equation] < 1,
                "ACM dual-time residual scales must be positive and relative tolerances must be in [0,1)");
            const real groupRelativeTolerance = equation < 4 ? dualTimeFlowRelativeTolerance : dualTimeTurbulenceRelativeTolerance;
            DNDS_check_throw_info(groupRelativeTolerance == 0 ||
                                      (dualTimeResidualScales[equation] == 1 && dualTimeRelativeTolerances[equation] == 0),
                                  "ACM grouped RMS and per-equation convergence controls cannot be combined in the same group");
        }
        DNDS_check_throw_info(
            std::isfinite(steadyRelativeTolerance) && steadyRelativeTolerance >= 0 && steadyRelativeTolerance < 1 &&
                std::isfinite(steadyCFLMin) && steadyCFLMin > 0 &&
                std::isfinite(steadyCFLMax) && steadyCFLMax >= steadyCFLMin &&
                std::isfinite(steadyCFLGrowth) && steadyCFLGrowth >= 1 &&
                std::isfinite(steadyCFLReduction) && steadyCFLReduction > 0 && steadyCFLReduction < 1,
            "ACM steady relative tolerance or adaptive CFL bounds/factors are invalid");
        const bool steadyImplicit = integrator == TimeIntegratorType::ImplicitEulerBlockJacobi ||
                                    integrator == TimeIntegratorType::ImplicitEulerLUSGS ||
                                    integrator == TimeIntegratorType::ImplicitEulerGMRES;
        DNDS_check_throw_info(
            (steadyRelativeTolerance == 0 && !steadyAdaptiveCFL) || steadyImplicit,
            "ACM steady convergence controls apply only to implicit steady pseudo-time integrators");
        DNDS_check_throw_info(
            !steadyAdaptiveCFL || (useCFLTimeStep && cfl >= steadyCFLMin && cfl <= steadyCFLMax),
            "ACM adaptive steady CFL requires CFL stepping and an initial CFL within the bounds");
        DNDS_check_throw_info(
            std::isfinite(implicitRelaxation) && implicitRelaxation > 0 && implicitRelaxation <= 1,
            "ACM implicitRelaxation must be in (0, 1]");
        DNDS_check_throw_info(lusgsSweeps > 0, "ACM lusgsSweeps must be positive");
        DNDS_check_throw_info(gmresSubspace >= 2, "ACM gmresSubspace must be at least two");
        DNDS_check_throw_info(gmresRestarts >= 0, "ACM gmresRestarts must be non-negative");
        DNDS_check_throw_info(
            std::isfinite(gmresRelativeTolerance) && gmresRelativeTolerance >= 0,
            "ACM gmresRelativeTolerance must be finite and non-negative");
    }

    real TimeMarchSettings::DualTimeGroupTarget(real initialDefect, bool turbulence) const
    {
        DNDS_check_throw_info(std::isfinite(initialDefect) && initialDefect >= 0,
                              "ACM initial physical group defect must be finite and non-negative");
        const real relativeTolerance = turbulence ? dualTimeTurbulenceRelativeTolerance : dualTimeFlowRelativeTolerance;
        return std::max(implicitTolerance, relativeTolerance * initialDefect);
    }

    std::array<real, 6> TimeMarchSettings::DualTimeTargets(const std::array<real, 6> &initialDefects) const
    {
        std::array<real, 6> targets{};
        for (std::size_t equation = 0; equation < targets.size(); equation++)
        {
            DNDS_check_throw_info(std::isfinite(initialDefects[equation]) && initialDefects[equation] >= 0,
                                  "ACM dual-time initial equation defect must be finite and non-negative");
            targets[equation] = std::max(implicitTolerance * dualTimeResidualScales[equation],
                                         dualTimeRelativeTolerances[equation] * initialDefects[equation]);
            DNDS_check_throw_info(std::isfinite(targets[equation]), "ACM dual-time equation target overflowed");
        }
        return targets;
    }

    bool TimeMarchSettings::DualTimeConverged(
        const std::array<real, 6> &initialDefects,
        const std::array<real, 6> &finalDefects, int activeEquations) const
    {
        DNDS_check_throw_info(activeEquations >= 4 && activeEquations <= 6,
                              "ACM dual-time convergence requires four to six active equations");
        const auto targets = DualTimeTargets(initialDefects);
        bool flowConverged = true;
        bool turbulenceConverged = true;
        for (int equation = 0; equation < activeEquations; equation++)
        {
            const auto ii = static_cast<std::size_t>(equation);
            if (!std::isfinite(finalDefects[ii]) || finalDefects[ii] < 0)
                return false;
            if (finalDefects[ii] > targets[ii])
                (equation < 4 ? flowConverged : turbulenceConverged) = false;
        }
        return CombineDualTimeGroups(dualTimeConvergenceCriterion, flowConverged, turbulenceConverged, activeEquations);
    }

    bool TimeMarchSettings::DualTimeConverged(const TimeStepReport &report, int activeEquations) const
    {
        DNDS_check_throw_info(activeEquations >= 4 && activeEquations <= 6,
                              "ACM dual-time convergence requires four to six active equations");
        const auto targets = DualTimeTargets(report.initialEquationDefectNorms);
        bool flowConverged = true;
        bool turbulenceConverged = true;
        for (int equation = 0; equation < activeEquations; equation++)
        {
            const auto ii = static_cast<std::size_t>(equation);
            const real value = report.equationDefectNorms[ii];
            if (!std::isfinite(value) || value < 0)
                return false;
            const real groupRelativeTolerance = equation < 4 ? dualTimeFlowRelativeTolerance : dualTimeTurbulenceRelativeTolerance;
            if (groupRelativeTolerance == 0 && value > targets[ii])
                (equation < 4 ? flowConverged : turbulenceConverged) = false;
        }
        if (!std::isfinite(report.finalDefectNorm) || report.finalDefectNorm < 0)
            return false;
        if (dualTimeFlowRelativeTolerance > 0 &&
            report.finalDefectNorm > DualTimeGroupTarget(report.initialDefectNorm, false))
            flowConverged = false;
        if (activeEquations > 4 &&
            (!std::isfinite(report.turbulenceDefectNorm) || report.turbulenceDefectNorm < 0))
            return false;
        if (activeEquations > 4 && dualTimeTurbulenceRelativeTolerance > 0 &&
            report.turbulenceDefectNorm > DualTimeGroupTarget(report.initialTurbulenceDefectNorm, true))
            turbulenceConverged = false;
        return CombineDualTimeGroups(dualTimeConvergenceCriterion, flowConverged, turbulenceConverged, activeEquations);
    }

    real TimeMarchSettings::SteadyImplicitTarget(real initialSpatialResidual) const
    {
        DNDS_check_throw_info(std::isfinite(initialSpatialResidual) && initialSpatialResidual >= 0,
                              "ACM steady initial residual must be finite and non-negative");
        return std::max(implicitTolerance, steadyRelativeTolerance * initialSpatialResidual);
    }

    real TimeMarchSettings::NextSteadyCFL(real currentCFL, real before, real after, bool innerConverged) const
    {
        if (!steadyAdaptiveCFL)
            return currentCFL;
        DNDS_check_throw_info(std::isfinite(currentCFL) && currentCFL > 0 &&
                                  std::isfinite(before) && before >= 0 && std::isfinite(after) && after >= 0,
                              "ACM adaptive CFL received a non-finite residual or invalid CFL");
        real factor = 1;
        if (!innerConverged || after > before * 1.05)
            factor = steadyCFLReduction;
        else if (before > 0 && after < before)
            factor = after == 0 ? steadyCFLGrowth
                                : std::min(steadyCFLGrowth, std::sqrt(before / after));
        return std::clamp(currentCFL * factor, steadyCFLMin, steadyCFLMax);
    }

    /** @copydoc ApplyGammaInverseToResidual */
    void ApplyGammaInverseToResidual(
        const StateField &states,
        const StateField &residual,
        StateField &pseudoDerivative,
        const Settings &settings)
    {
        settings.Validate();
        DNDS_check_throw_info(
            residual.size() == states.size(),
            "ACM residual field has a different size from the state field");
        pseudoDerivative.resize(states.size());
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index i = 0; i < static_cast<index>(states.size()); i++)
        {
            const std::size_t ii = static_cast<std::size_t>(i);
            pseudoDerivative[ii] =
                GammaInvLocal(states[ii], settings.beta2, settings.alpha) * residual[ii];
            DNDS_check_throw_info(
                pseudoDerivative[ii].allFinite(),
                "ACM inverse-Gamma residual contains a non-finite value");
        }
    }

    /** @copydoc AdvanceExplicitSSPRK3 */
    TimeStepReport AdvanceExplicitSSPRK3(
        StateField &states,
        const ScalarField &pseudoTimeStep,
        const Settings &settings,
        const ResidualEvaluator &residualEvaluator,
        const MPIInfo *mpi)
    {
        settings.Validate();
        ValidateTimeStepField(states, pseudoTimeStep);

        const StateField statesOld = states;
        StateField residual;
        StateField derivative;

        EvaluateResidualChecked(states, residual, residualEvaluator);
        ApplyGammaInverseToResidual(states, residual, derivative, settings);
        TimeStepReport report;
        report.iterations = 3;
        report.initialDefectNorm = GlobalRMSNorm(derivative, mpi);

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index i = 0; i < static_cast<index>(states.size()); i++)
        {
            const std::size_t ii = static_cast<std::size_t>(i);
            states[ii] = statesOld[ii] + pseudoTimeStep[ii] * derivative[ii];
        }

        EvaluateResidualChecked(states, residual, residualEvaluator);
        ApplyGammaInverseToResidual(states, residual, derivative, settings);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index i = 0; i < static_cast<index>(states.size()); i++)
        {
            const std::size_t ii = static_cast<std::size_t>(i);
            states[ii] = 0.75 * statesOld[ii] +
                         0.25 * (states[ii] + pseudoTimeStep[ii] * derivative[ii]);
        }

        EvaluateResidualChecked(states, residual, residualEvaluator);
        ApplyGammaInverseToResidual(states, residual, derivative, settings);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index i = 0; i < static_cast<index>(states.size()); i++)
        {
            const std::size_t ii = static_cast<std::size_t>(i);
            states[ii] = (1.0 / 3.0) * statesOld[ii] +
                         (2.0 / 3.0) * (states[ii] + pseudoTimeStep[ii] * derivative[ii]);
            DNDS_check_throw_info(states[ii].allFinite(), "ACM SSPRK3 produced a non-finite state");
        }

        EvaluateResidualChecked(states, residual, residualEvaluator);
        ApplyGammaInverseToResidual(states, residual, derivative, settings);
        report.finalDefectNorm = GlobalRMSNorm(derivative, mpi);
        report.converged = std::isfinite(report.finalDefectNorm);
        return report;
    }

    /** @copydoc AdvanceImplicitEulerBlockJacobi */
    TimeStepReport AdvanceImplicitEulerBlockJacobi(
        StateField &states,
        const ScalarField &pseudoTimeStep,
        const Settings &settings,
        const TimeMarchSettings &timeSettings,
        const ResidualEvaluator &residualEvaluator,
        const DiagonalJacobianEvaluator &diagonalJacobianEvaluator,
        const MPIInfo *mpi)
    {
        settings.Validate();
        timeSettings.Validate();
        ValidateTimeStepField(states, pseudoTimeStep);
        DNDS_check_throw_info(
            static_cast<bool>(diagonalJacobianEvaluator),
            "ACM implicit diagonal-Jacobian callback is empty");

        const StateField statesOld = states;
        StateField residual;
        StateField defect;
        MatrixField diagonalJacobian;
        TimeStepReport report;

        for (int iteration = 1; iteration <= timeSettings.maxImplicitIterations; iteration++)
        {
            EvaluateResidualChecked(states, residual, residualEvaluator);
            FormImplicitDefect(states, statesOld, residual, pseudoTimeStep, settings, defect);
            const real defectNorm = GlobalRMSNorm(defect, mpi);
            if (iteration == 1)
            {
                report.initialDefectNorm = defectNorm;
                report.defectTolerance = timeSettings.SteadyImplicitTarget(defectNorm);
            }
            report.finalDefectNorm = defectNorm;
            report.iterations = iteration;
            if (defectNorm <= report.defectTolerance)
            {
                report.converged = true;
                return report;
            }

            diagonalJacobianEvaluator(states, diagonalJacobian);
            DNDS_check_throw_info(
                diagonalJacobian.size() == states.size(),
                "ACM implicit Jacobian callback returned a field with an invalid size");

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index i = 0; i < static_cast<index>(states.size()); i++)
            {
                const std::size_t ii = static_cast<std::size_t>(i);
                DNDS_check_throw_info(
                    diagonalJacobian[ii].allFinite(),
                    "ACM implicit Jacobian callback returned a non-finite block");
                const Matrix4 diagonal =
                    PseudoTimeProductJacobian(
                        states[ii], statesOld[ii], pseudoTimeStep[ii], settings) -
                    diagonalJacobian[ii];
                const Eigen::FullPivLU<Matrix4> factorization(diagonal);
                DNDS_check_throw_info(factorization.isInvertible(), "ACM implicit diagonal block is singular");
                const State correction = factorization.solve(defect[ii]);
                DNDS_check_throw_info(correction.allFinite(), "ACM implicit correction is non-finite");
                states[ii] += timeSettings.implicitRelaxation * correction;
                DNDS_check_throw_info(states[ii].allFinite(), "ACM implicit update produced a non-finite state");
            }
        }

        EvaluateResidualChecked(states, residual, residualEvaluator);
        FormImplicitDefect(states, statesOld, residual, pseudoTimeStep, settings, defect);
        report.finalDefectNorm = GlobalRMSNorm(defect, mpi);
        report.converged = report.finalDefectNorm <= report.defectTolerance;
        return report;
    }

    /** @copydoc AdvancePseudoTimeStep */
    TimeStepReport AdvancePseudoTimeStep(
        StateField &states,
        const ScalarField &pseudoTimeStep,
        const Settings &settings,
        const TimeMarchSettings &timeSettings,
        const ResidualEvaluator &residualEvaluator,
        const DiagonalJacobianEvaluator &diagonalJacobianEvaluator,
        const MPIInfo *mpi)
    {
        if (timeSettings.integrator == TimeIntegratorType::ExplicitSSPRK3)
            return AdvanceExplicitSSPRK3(states, pseudoTimeStep, settings, residualEvaluator, mpi);
        if (timeSettings.integrator == TimeIntegratorType::ImplicitEulerBlockJacobi)
            return AdvanceImplicitEulerBlockJacobi(
                states,
                pseudoTimeStep,
                settings,
                timeSettings,
                residualEvaluator,
                diagonalJacobianEvaluator,
                mpi);
        DNDS_check_throw_info(false, "unknown ACM time integrator");
        return {};
    }
}
