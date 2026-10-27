/**
 * @file ACMBDF2.cpp
 * @brief Implementation of physical-time BDF2 kernels for ACM dual-time marching.
 * @author Runzhi Ma
 * @date 2026-09-02
 * @note Modifier: Runzhi Ma.
 */
#include "ACMBDF2.hpp"

#include "DNDS/Errors.hpp"
#include "DNDS/OMP.hpp"

#include <cmath>
#include <utility>

namespace DNDS::ACM
{
    namespace
    {
        /**
         * @brief Reject non-finite BDF coefficients or a time-discretization order other than one or two.
         * @param coefficients Physical-time derivative weights and startup/BDF2 order to validate.
         * @throws std::runtime_error If any coefficient or the selected order is invalid.
         */
        void ValidateCoefficients(const BDF2Coefficients &coefficients)
        {
            DNDS_check_throw_info(
                std::isfinite(coefficients.current) &&
                    std::isfinite(coefficients.previous) &&
                    std::isfinite(coefficients.previousPrevious),
                "ACM BDF coefficients must be finite");
            DNDS_check_throw_info(
                coefficients.order == 1 || coefficients.order == 2,
                "ACM BDF order must be one or two");
        }

        /**
         * @brief Check that a physical time interval is finite and strictly positive.
         * @param physicalTimeStep Physical interval used as the denominator of BDF derivatives.
         * @throws std::runtime_error If the time interval is non-finite or nonpositive.
         */
        void ValidatePhysicalTimeStep(real physicalTimeStep)
        {
            DNDS_check_throw_info(
                std::isfinite(physicalTimeStep) && physicalTimeStep > 0,
                "ACM BDF2 physical time step must be finite and positive");
        }

        /**
         * @brief Check that a local flow-history field is nonempty and finite.
         * @param field Owned-cell four-component [u,v,w,p] states to inspect.
         * @param message Failure text used for either an empty field or a non-finite state.
         */
        void ValidateStateField(const StateField &field, const char *message)
        {
            DNDS_check_throw_info(!field.empty(), message);
            for (const State &state : field)
                DNDS_check_throw_info(state.allFinite(), message);
        }

        /**
         * @brief Check the number of active entries in two-slot primitive turbulence storage.
         * @param count Active equation count; must be one for SA or two for a two-equation model.
         */
        void ValidateActiveVariableCount(int count)
        {
            DNDS_check_throw_info(count > 0 && count <= TurbulenceState::RowsAtCompileTime,
                                  "ACM BDF2 turbulence requires one or two active variables");
        }

        /**
         * @brief Validate active primitive turbulence history entries and clear inactive slots.
         * @param field Nonempty local turbulence history to copy without modifying the source.
         * @param activeVariableCount One or two finite positive entries to preserve per state.
         * @return Validated history copy with all inactive entries set to zero.
         * @throws std::runtime_error If the count, field size, or any active value is invalid.
         */
        TurbulenceStateField SanitizeTurbulenceHistory(
            const TurbulenceStateField &field, int activeVariableCount)
        {
            ValidateActiveVariableCount(activeVariableCount);
            DNDS_check_throw_info(!field.empty(), "ACM BDF2 turbulence history is empty");
            TurbulenceStateField sanitized;
            sanitized.reserve(field.size());
            for (const TurbulenceState &state : field)
            {
                DNDS_check_throw_info(state.head(activeVariableCount).allFinite() &&
                                          (state.head(activeVariableCount).array() > 0).all(),
                                      "ACM BDF2 active turbulence history must be finite and positive");
                TurbulenceState clean = TurbulenceState::Zero();
                clean.head(activeVariableCount) = state.head(activeVariableCount);
                sanitized.push_back(clean);
            }
            return sanitized;
        }
    }

    /** @copydoc GetBDF2Coefficients */
    BDF2Coefficients GetBDF2Coefficients(std::size_t completedPhysicalSteps)
    {
        if (completedPhysicalSteps == 0)
            return {1, -1, 0, 1};
        return {1.5, -2, 0.5, 2};
    }

    /** @copydoc PhysicalTimeMassMatrix */
    Matrix4 PhysicalTimeMassMatrix()
    {
        Matrix4 mass = Matrix4::Identity();
        mass(3, 3) = 0;
        return mass;
    }

    BDF2Coefficients GetPositiveTurbulenceBDFCoefficients(
        real previous, real previousPrevious,
        const BDF2Coefficients &coefficients, real minimumValue)
    {
        ValidateCoefficients(coefficients);
        DNDS_check_throw_info(std::isfinite(previous) && previous > 0 &&
                                  std::isfinite(previousPrevious) && previousPrevious > 0 &&
                                  std::isfinite(minimumValue) && minimumValue > 0,
                              "ACM positive RANS BDF requires positive finite histories and floor");
        const real history = -coefficients.previous * previous -
                             coefficients.previousPrevious * previousPrevious;
        if (coefficients.order == 2 && history < coefficients.current * minimumValue)
            return GetBDF2Coefficients(0);
        return coefficients;
    }

    /** @copydoc EvaluateBDF2PhysicalDerivative */
    State EvaluateBDF2PhysicalDerivative(
        const State &current,
        const State &previous,
        const State &previousPrevious,
        const BDF2Coefficients &coefficients,
        real physicalTimeStep)
    {
        DNDS_check_throw_info(
            current.allFinite() && previous.allFinite() && previousPrevious.allFinite(),
            "ACM BDF2 history contains a non-finite state");
        ValidateCoefficients(coefficients);
        ValidatePhysicalTimeStep(physicalTimeStep);
        return PhysicalTimeMassMatrix() *
               (coefficients.current * current +
                coefficients.previous * previous +
                coefficients.previousPrevious * previousPrevious) /
               physicalTimeStep;
    }

    // See ACMBDF2.hpp for the function contract.
    TurbulenceState EvaluateBDF2TurbulenceDerivative(
        const TurbulenceState &current,
        const TurbulenceState &previous,
        const TurbulenceState &previousPrevious,
        const BDF2Coefficients &coefficients,
        real physicalTimeStep,
        int activeVariableCount)
    {
        ValidateActiveVariableCount(activeVariableCount);
        DNDS_check_throw_info(
            current.head(activeVariableCount).allFinite() &&
                previous.head(activeVariableCount).allFinite() &&
                previousPrevious.head(activeVariableCount).allFinite(),
            "ACM BDF2 turbulence history contains a non-finite state");
        ValidateCoefficients(coefficients);
        ValidatePhysicalTimeStep(physicalTimeStep);
        TurbulenceState derivative = TurbulenceState::Zero();
        derivative.head(activeVariableCount) =
            (coefficients.current * current.head(activeVariableCount) +
             coefficients.previous * previous.head(activeVariableCount) +
             coefficients.previousPrevious * previousPrevious.head(activeVariableCount)) /
            physicalTimeStep;
        return derivative;
    }

    /** @copydoc FormBDF2PhysicalDefect */
    void FormBDF2PhysicalDefect(
        const StateField &current,
        const StateField &previous,
        const StateField &previousPrevious,
        const StateField &spatialResidual,
        const BDF2Coefficients &coefficients,
        real physicalTimeStep,
        StateField &defect)
    {
        DNDS_check_throw_info(
            !current.empty() && current.size() == previous.size() &&
                current.size() == previousPrevious.size() &&
                current.size() == spatialResidual.size(),
            "ACM BDF2 fields have incompatible sizes");
        ValidateCoefficients(coefficients);
        ValidatePhysicalTimeStep(physicalTimeStep);
        defect.resize(current.size());
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index i = 0; i < static_cast<index>(current.size()); i++)
        {
            const std::size_t ii = static_cast<std::size_t>(i);
            DNDS_check_throw_info(
                current[ii].allFinite() && previous[ii].allFinite() &&
                    previousPrevious[ii].allFinite() && spatialResidual[ii].allFinite(),
                "ACM BDF2 field contains a non-finite value");
            defect[ii] = spatialResidual[ii] - EvaluateBDF2PhysicalDerivative(
                                                   current[ii],
                                                   previous[ii],
                                                   previousPrevious[ii],
                                                   coefficients,
                                                   physicalTimeStep);
        }
    }

    /** @copydoc AddBDF2PhysicalDiagonal */
    void AddBDF2PhysicalDiagonal(
        MatrixField &diagonal,
        const BDF2Coefficients &coefficients,
        real physicalTimeStep)
    {
        DNDS_check_throw_info(!diagonal.empty(), "ACM BDF2 diagonal field is empty");
        ValidateCoefficients(coefficients);
        ValidatePhysicalTimeStep(physicalTimeStep);
        const Matrix4 physicalBlock =
            coefficients.current * PhysicalTimeMassMatrix() / physicalTimeStep;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index i = 0; i < static_cast<index>(diagonal.size()); i++)
        {
            Matrix4 &block = diagonal[static_cast<std::size_t>(i)];
            DNDS_check_throw_info(block.allFinite(), "ACM BDF2 diagonal contains a non-finite block");
            block += physicalBlock;
        }
    }

    // See ACMBDF2.hpp for the function contract.
    bool IsBDF2DualTimeIntegrator(TimeIntegratorType integrator)
    {
        return integrator == TimeIntegratorType::BDF2DualTimeLUSGS ||
               integrator == TimeIntegratorType::BDF2DualTimeGMRES;
    }

    // See ACMBDF2.hpp for the function contract.
    bool SupportsBDF2TurbulenceModel(TurbulenceModel model)
    {
        switch (model)
        {
        case TurbulenceModel::Laminar:
        case TurbulenceModel::SpalartAllmaras:
        case TurbulenceModel::KOmegaWilcox:
        case TurbulenceModel::KOmegaSST:
        case TurbulenceModel::RealizableKEpsilon:
            return true;
        }
        return false;
    }

    // See ACMBDF2.hpp for the function contract.
    bool BDF2UsesLUSGS(TimeIntegratorType integrator)
    {
        DNDS_check_throw_info(
            IsBDF2DualTimeIntegrator(integrator),
            "ACM BDF2 linear-solver query requires a BDF2 integrator");
        return integrator == TimeIntegratorType::BDF2DualTimeLUSGS;
    }

    void BDF2TurbulenceHistory::Initialize(
        const TurbulenceStateField &initialState, int activeVariableCount)
    {
        TurbulenceStateField sanitized = SanitizeTurbulenceHistory(initialState, activeVariableCount);
        _previous = sanitized;
        _previousPrevious = std::move(sanitized);
        _activeVariableCount = activeVariableCount;
        _completedPhysicalSteps = 0;
        _initialized = true;
    }

    void BDF2TurbulenceHistory::Commit(const TurbulenceStateField &completedState)
    {
        DNDS_check_throw_info(_initialized, "ACM BDF2 turbulence history is not initialized");
        DNDS_check_throw_info(completedState.size() == _previous.size(),
                              "ACM BDF2 completed turbulence field has an incompatible size");
        TurbulenceStateField sanitized = SanitizeTurbulenceHistory(completedState, _activeVariableCount);
        _previousPrevious = _previous;
        _previous = std::move(sanitized);
        ++_completedPhysicalSteps;
    }

    const TurbulenceStateField &BDF2TurbulenceHistory::Previous() const
    {
        DNDS_check_throw_info(_initialized, "ACM BDF2 turbulence history is not initialized");
        return _previous;
    }

    const TurbulenceStateField &BDF2TurbulenceHistory::PreviousPrevious() const
    {
        DNDS_check_throw_info(_initialized, "ACM BDF2 turbulence history is not initialized");
        return _previousPrevious;
    }

    BDF2Coefficients BDF2TurbulenceHistory::Coefficients() const
    {
        DNDS_check_throw_info(_initialized, "ACM BDF2 turbulence history is not initialized");
        return GetBDF2Coefficients(_completedPhysicalSteps);
    }

    // See ACMBDF2.hpp for the function contract.
    void BDF2History::Initialize(const StateField &initialState)
    {
        ValidateStateField(initialState, "ACM BDF2 initial history is empty or non-finite");
        _previous = initialState;
        _previousPrevious = initialState;
        _completedPhysicalSteps = 0;
        _initialized = true;
    }

    /** @copydoc BDF2History::Coefficients */
    BDF2Coefficients BDF2History::Coefficients() const
    {
        DNDS_check_throw_info(_initialized, "ACM BDF2 history is not initialized");
        return GetBDF2Coefficients(_completedPhysicalSteps);
    }

    /** @copydoc BDF2History::Previous */
    const StateField &BDF2History::Previous() const
    {
        DNDS_check_throw_info(_initialized, "ACM BDF2 history is not initialized");
        return _previous;
    }

    /** @copydoc BDF2History::PreviousPrevious */
    const StateField &BDF2History::PreviousPrevious() const
    {
        DNDS_check_throw_info(_initialized, "ACM BDF2 history is not initialized");
        return _previousPrevious;
    }

    // See ACMBDF2.hpp for the function contract.
    void BDF2History::Commit(const StateField &completedState)
    {
        DNDS_check_throw_info(_initialized, "ACM BDF2 history is not initialized");
        DNDS_check_throw_info(
            completedState.size() == _previous.size(),
            "ACM BDF2 completed state has an incompatible size");
        ValidateStateField(completedState, "ACM BDF2 completed state is empty or non-finite");
        _previousPrevious = _previous;
        _previous = completedState;
        _completedPhysicalSteps++;
    }
}
