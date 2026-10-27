/**
 * @file ACMBDF2.hpp
 * @brief Physical-time BDF2 kernels and history storage for ACM dual-time marching.
 *
 * @details The physical incompressible state is `Q=[u,v,w,0]`.  Consequently the
 * physical-time mass matrix is `diag(1,1,1,0)`: velocity receives the BDF term,
 * while pressure remains governed by artificial compressibility in pseudo-time.
 *
 * @author Runzhi Ma
 * @date 2026-09-02
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMTime.hpp"
#include "ACMTurbulence.hpp"

#include <cstddef>
#include <vector>

namespace DNDS::ACM
{
    /** @brief Constant-step BDF coefficients for one completed physical step. */
    struct BDF2Coefficients
    {
        real current = 1;          ///< Coefficient multiplying the new state.
        real previous = -1;        ///< Coefficient multiplying the last completed state.
        real previousPrevious = 0; ///< Coefficient multiplying the second history state.
        int order = 1;             ///< One for the backward-Euler startup, otherwise two.
    };

    /**
     * @brief Return backward-Euler startup or constant-step BDF2 coefficients.
     * @param completedPhysicalSteps Number of completed physical steps in the history.
     * @return `(1,-1,0)` before the first commit, otherwise `(3/2,-2,1/2)`.
     */
    BDF2Coefficients GetBDF2Coefficients(std::size_t completedPhysicalSteps);

    /**
     * @brief Use BE for one RANS equation if its BDF2 history predicts a value below its floor.
     * @details The choice depends only on completed histories and is fixed throughout the physical step.
     * Spatial defects remain unprojected; this fallback does not certify a floor-constrained state as converged.
     * @param previous Last completed value of one turbulence equation; use the same density convention as
     *     minimumValue.
     * @param previousPrevious Value of that equation at the second completed history level.
     * @param coefficients Nominal backward-Euler startup or BDF2 coefficients.
     * @param minimumValue Positive admissible floor in the history values' units.
     * @return Nominal coefficients, or backward-Euler coefficients if the extrapolated BDF history violates the
     *     floor.
     */
    BDF2Coefficients GetPositiveTurbulenceBDFCoefficients(
        real previous, real previousPrevious,
        const BDF2Coefficients &coefficients, real minimumValue);

    /**
     * @brief Return the physical-time mass matrix `diag(1,1,1,0)`.
     * @return Matrix applying physical time derivatives to velocity but not pressure.
     */
    Matrix4 PhysicalTimeMassMatrix();

    /**
     * @brief Form the physical BDF derivative of `Q=[u,v,w,0]` for one state.
     * @param current Current nonlinear iterate at the new physical time.
     * @param previous Last completed physical-time state.
     * @param previousPrevious Second completed history state; unused by startup BE.
     * @param coefficients Startup or BDF2 coefficients.
     * @param physicalTimeStep Positive uniform physical time step.
     * @return BDF derivative with an identically zero pressure component.
     */
    State EvaluateBDF2PhysicalDerivative(
        const State &current,
        const State &previous,
        const State &previousPrevious,
        const BDF2Coefficients &coefficients,
        real physicalTimeStep);

    /**
     *  @brief Form the physical BDF derivative of active primitive RANS variables; clear inactive entries.
     * @param current Current primitive turbulence iterate, with up to two active entries.
     * @param previous Last completed primitive turbulence state.
     * @param previousPrevious Second completed history state; ignored by backward-Euler startup.
     * @param coefficients Time-derivative coefficients for this physical step.
     * @param physicalTimeStep Finite positive physical time interval.
     * @param activeVariableCount Number of entries to differentiate, either one or two; defaults to two.
     * @return Weighted history derivative divided by physicalTimeStep; inactive entries are zero.
     */
    TurbulenceState EvaluateBDF2TurbulenceDerivative(
        const TurbulenceState &current,
        const TurbulenceState &previous,
        const TurbulenceState &previousPrevious,
        const BDF2Coefficients &coefficients,
        real physicalTimeStep,
        int activeVariableCount = TurbulenceState::RowsAtCompileTime);

    /**
     * @brief Form `R(U)-dQ/dt` for every rank-local state.
     * @param current Current nonlinear iterate.
     * @param previous Last completed state field.
     * @param previousPrevious Second completed state field.
     * @param spatialResidual Raw spatial residual `R(U)`.
     * @param coefficients Startup or BDF2 coefficients.
     * @param physicalTimeStep Positive uniform physical time step.
     * @param defect Output physical-time nonlinear defect.
     */
    void FormBDF2PhysicalDefect(
        const StateField &current,
        const StateField &previous,
        const StateField &previousPrevious,
        const StateField &spatialResidual,
        const BDF2Coefficients &coefficients,
        real physicalTimeStep,
        StateField &defect);

    /**
     * @brief Add the physical BDF Jacobian `a0 M_Q/dt` to cell diagonal blocks.
     * @param diagonal Cell-local implicit diagonal blocks, modified in place.
     * @param coefficients Startup or BDF2 coefficients.
     * @param physicalTimeStep Positive uniform physical time step.
     */
    void AddBDF2PhysicalDiagonal(
        MatrixField &diagonal,
        const BDF2Coefficients &coefficients,
        real physicalTimeStep);

    /**
     *  @brief Return whether an integrator selects physical BDF2 dual-time marching.
     * @param integrator Configured time-integration algorithm.
     * @return True for either BDF2 dual-time integrator.
     */
    bool IsBDF2DualTimeIntegrator(TimeIntegratorType integrator);

    /**
     *  @brief Return whether a closure has the physical-time transport required by BDF2.
     * @param model Configured turbulence closure.
     * @return True for laminar flow and closures with implemented physical-time transport.
     */
    bool SupportsBDF2TurbulenceModel(TurbulenceModel model);

    /**
     *  @brief Return whether a selected BDF2 integrator uses LU-SGS as its linear solver.
     * @param integrator Configured time-integration algorithm.
     * @return True for BDF2DualTimeLUSGS; false for the supported GMRES variant.
     */
    bool BDF2UsesLUSGS(TimeIntegratorType integrator);

    using TurbulenceStateField = std::vector<TurbulenceState>;

    /**
     * @brief Completed physical histories for one- or two-equation primitive RANS models.
     * @details Only active entries must be finite and positive. Inactive storage is zeroed.
     * Invalid commits leave both completed levels and the startup counter unchanged.
     */
    class BDF2TurbulenceHistory
    {
    public:
        /**
         * @brief Initialize both primitive RANS history levels and reset the completed-step count.
         * @param initialState Owned-cell primitive turbulence values copied into both history levels.
         * @param activeVariableCount One for SA or two for a two-equation closure; inactive storage is cleared.
         * @throws std::runtime_error If active values are non-finite/nonpositive or the variable count is invalid.
         */
        void Initialize(const TurbulenceStateField &initialState, int activeVariableCount);
        /**
         * @brief Validate a completed RANS step, shift histories, and increment the step count.
         * @param completedState New owned-cell primitive turbulence field, with the initialized field size.
         * @pre Initialize() has completed; failed validation leaves both history levels unchanged.
         */
        void Commit(const TurbulenceStateField &completedState);
        /**
         * @brief Return read-only last completed state field; Initialize() must have completed.
         * @return Read-only last completed state field; Initialize() must have completed.
         */
        const TurbulenceStateField &Previous() const;
        /**
         * @brief Return read-only second completed state field; Initialize() must have completed.
         * @return Read-only second completed state field; Initialize() must have completed.
         */
        const TurbulenceStateField &PreviousPrevious() const;
        /**
         * @brief Return backward-Euler startup coefficients before the first commit, then constant-step BDF2
         *     coefficients.
         * @return Backward-Euler startup coefficients before the first commit, then constant-step BDF2
         *     coefficients.
         */
        BDF2Coefficients Coefficients() const;
        /**
         * @brief Return number of active primitive turbulence equations in the history.
         * @return Number of active primitive turbulence equations in the history.
         */
        int ActiveVariableCount() const { return _activeVariableCount; }
        /**
         * @brief Return number of physical-step commits since initialization.
         * @return Number of physical-step commits since initialization.
         */
        std::size_t CompletedPhysicalSteps() const { return _completedPhysicalSteps; }
        /**
         * @brief Return true once both history levels have been initialized.
         * @return True once both history levels have been initialized.
         */
        bool IsInitialized() const { return _initialized; }

    private:
        TurbulenceStateField _previous;
        TurbulenceStateField _previousPrevious;
        int _activeVariableCount = 0;
        std::size_t _completedPhysicalSteps = 0;
        bool _initialized = false;
    };

    /**
     * @brief Two-level completed-step history with automatic backward-Euler startup.
     *
     * @details History advances only after all inner iterations for a physical step finish. This
     * avoids corrupting `U^n` and `U^{n-1}` during nonlinear dual-time iterations. The caller
     * decides whether reaching its inner-iteration limit constitutes completion.
     */
    class BDF2History
    {
    public:
        /**
         *  @brief Initialize both history levels from an initial flow field.
         * @param initialState Finite owned-cell [u,v,w,p] field copied into both levels; resets startup history.
         */
        void Initialize(const StateField &initialState);

        /**
         *  @brief Return coefficients for the next physical step.
         * @return Backward-Euler startup coefficients before the first commit, then constant-step BDF2
         *     coefficients.
         */
        BDF2Coefficients Coefficients() const;

        /**
         *  @brief Return the last completed physical-time state.
         * @return Read-only last completed state field; Initialize() must have completed.
         */
        const StateField &Previous() const;

        /**
         *  @brief Return the second completed physical-time state.
         * @return Read-only second completed state field; Initialize() must have completed.
         */
        const StateField &PreviousPrevious() const;

        /**
         *  @brief Commit a completed physical-step state and shift both history levels.
         * @param completedState Finite completed [u,v,w,p] field with the initialized owned-cell count.
         */
        void Commit(const StateField &completedState);

        /**
         *  @brief Return the number of completed physical steps.
         * @return Number of physical-step commits since initialization.
         */
        std::size_t CompletedPhysicalSteps() const { return _completedPhysicalSteps; }

    private:
        StateField _previous;         ///< `U^n` at the start of the next physical step.
        StateField _previousPrevious; ///< `U^{n-1}` at the start of the next physical step.
        std::size_t _completedPhysicalSteps = 0;
        bool _initialized = false;
    };
}
