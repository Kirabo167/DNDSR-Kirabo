/**
 * @file ACMTime.hpp
 * @brief Time-marching settings and pseudo-time integration interfaces for ACM states.
 *
 * @details The time integrators operate on rank-local state fields and obtain spatial residuals
 * through callbacks. A callback may perform mesh reconstruction and MPI ghost communication, so
 * the algorithms can later be connected to ACMEvaluator without changing their update formulas.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMFlux.hpp"
#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/MPI.hpp"

#include <array>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace DNDS::ACM
{
    /// Steady pseudo-time and physical dual-time algorithms exposed by the ACM solver.
    enum class TimeIntegratorType
    {
        ExplicitSSPRK3,
        ImplicitEulerBlockJacobi,
        ImplicitEulerLUSGS,
        ImplicitEulerGMRES,
        BDF2DualTimeLUSGS,
        BDF2DualTimeGMRES,
    };

    /**
     *  @brief Serialize an ACM time integrator without permitting an invalid enum value.
     * @param [out] json JSON value replaced by the canonical time-integrator string.
     * @param integrator Supported ACM integrator to serialize.
     * @tparam BasicJsonType nlohmann-compatible JSON value type.
     * @throws std::invalid_argument If integrator is not a supported enumeration value.
     */
    template <typename BasicJsonType>
    void to_json(BasicJsonType &json, const TimeIntegratorType &integrator)
    {
        switch (integrator)
        {
        case TimeIntegratorType::ExplicitSSPRK3:
            json = "ExplicitSSPRK3";
            return;
        case TimeIntegratorType::ImplicitEulerBlockJacobi:
            json = "ImplicitEulerBlockJacobi";
            return;
        case TimeIntegratorType::ImplicitEulerLUSGS:
            json = "ImplicitEulerLUSGS";
            return;
        case TimeIntegratorType::ImplicitEulerGMRES:
            json = "ImplicitEulerGMRES";
            return;
        case TimeIntegratorType::BDF2DualTimeLUSGS:
            json = "BDF2DualTimeLUSGS";
            return;
        case TimeIntegratorType::BDF2DualTimeGMRES:
            json = "BDF2DualTimeGMRES";
            return;
        }
        throw std::invalid_argument("unknown ACM time-integrator enum value");
    }

    /**
     *  @brief Deserialize an ACM time integrator and reject unknown JSON strings.
     * @param json JSON string containing an exact supported ACM time-integrator name.
     * @param [out] integrator Selected integrator, assigned after validating the input name.
     * @tparam BasicJsonType nlohmann-compatible JSON value type.
     * @throws std::invalid_argument If the input is not a string or its name is unsupported.
     */
    template <typename BasicJsonType>
    void from_json(const BasicJsonType &json, TimeIntegratorType &integrator)
    {
        if (!json.is_string())
            throw std::invalid_argument("ACM time integrator must be a JSON string");
        const std::string name = json.template get<std::string>();
        if (name == "ExplicitSSPRK3")
            integrator = TimeIntegratorType::ExplicitSSPRK3;
        else if (name == "ImplicitEulerBlockJacobi")
            integrator = TimeIntegratorType::ImplicitEulerBlockJacobi;
        else if (name == "ImplicitEulerLUSGS")
            integrator = TimeIntegratorType::ImplicitEulerLUSGS;
        else if (name == "ImplicitEulerGMRES")
            integrator = TimeIntegratorType::ImplicitEulerGMRES;
        else if (name == "BDF2DualTimeLUSGS")
            integrator = TimeIntegratorType::BDF2DualTimeLUSGS;
        else if (name == "BDF2DualTimeGMRES")
            integrator = TimeIntegratorType::BDF2DualTimeGMRES;
        else
            throw std::invalid_argument("unknown ACM time integrator: " + name);
    }

    /** @brief Return the strict JSON names emitted in the generated configuration schema. */
    inline std::vector<std::string> _dnds_enum_allowed_values_fn(TimeIntegratorType *)
    {
        return {
            "ExplicitSSPRK3",
            "ImplicitEulerBlockJacobi",
            "ImplicitEulerLUSGS",
            "ImplicitEulerGMRES",
            "BDF2DualTimeLUSGS",
            "BDF2DualTimeGMRES"};
    }

    /// Left preconditioners available to the generic GMRES algorithm.
    enum class GMRESPreconditionerType
    {
        BlockJacobi,
        LUSGS,
    };

    DNDS_DEFINE_ENUM_JSON(
        GMRESPreconditionerType,
        {
            {GMRESPreconditionerType::BlockJacobi, "BlockJacobi"},
            {GMRESPreconditionerType::LUSGS, "LUSGS"},
        })

    struct TimeStepReport;

    /// Runtime controls shared by steady pseudo-time and BDF2 dual-time stepping.
    struct TimeMarchSettings
    {
        TimeIntegratorType integrator = TimeIntegratorType::ExplicitSSPRK3; ///< Selected algorithm.
        int nSteps = 0;                                                     ///< Outer pseudo-time steps, or physical steps for BDF2.
        real pseudoTimeStep = 0.01;                                         ///< Positive fixed pseudo-time step.
        real physicalTimeStep = 0.01;                                       ///< Uniform physical step used by BDF2.
        bool useCFLTimeStep = false;                                        ///< Recompute spectral-radius time steps.
        bool useLocalTimeStep = true;                                       ///< Keep cell-local CFL steps instead of MPI minimum.
        real cfl = 0.5;                                                     ///< CFL multiplier for spectral-radius stepping.
        real maximumPseudoTimeStep = 1e100;                                 ///< Upper clamp for a CFL-derived step.
        int maxImplicitIterations = 20;                                     ///< Inner defect-correction iterations per implicit/physical step.
        real implicitTolerance = 1e-8;                                      ///< Global RMS steady or physical defect absolute tolerance.
        std::string dualTimeConvergenceCriterion = "Absolute";              ///< Defect criterion, or RelativeUpdate for laminar flow.
        real dualTimeUpdateRelativeTolerance = 1e-5;                       ///< Applied-update L2 / updated-state L2; zero disables early exit.
        int dualTimeUpdateMinimumIterations = 2;                           ///< Minimum corrections before testing RelativeUpdate.
        real dualTimeFlowRelativeTolerance = 0;                            ///< Flow RMS target relative to this physical step's initial defect; zero uses equation targets.
        real dualTimeTurbulenceRelativeTolerance = 0;                      ///< Active RANS RMS target relative to its own initial physical defect.
        std::array<real, 6> dualTimeResidualScales{1, 1, 1, 1, 1, 1};         ///< Characteristic residual scales for [u,v,w,p,q1,q2].
        std::array<real, 6> dualTimeRelativeTolerances{};                    ///< Per-equation target relative to the initial physical defect; zero disables it.
        bool dualTimeRequireConvergence = true;                            ///< Reject an unconverged physical step before committing histories.
        real steadyRelativeTolerance = 0;                                 ///< Steady inner target relative to this outer step's spatial RMS.
        bool steadyAdaptiveCFL = false;                                   ///< Adapt CFL using post-step spatial residual and inner success.
        real steadyCFLMin = 0.1;
        real steadyCFLMax = 20;
        real steadyCFLGrowth = 1.5;                                        ///< Maximum increase per successful outer step.
        real steadyCFLReduction = 0.5;                                    ///< Reduction after residual growth or inner failure.
        real implicitRelaxation = 1.0;                                      ///< Damping applied to every implicit correction.
        int lusgsSweeps = 2;                                                ///< Maximum standalone SGS sweeps; fixed GMRES preconditioner sweeps.
        int gmresSubspace = 10;                                             ///< Arnoldi vectors per GMRES restart.
        int gmresRestarts = 3;                                              ///< Maximum generic-GMRES restart count.
        real linearRelativeTolerance = 1e-6;                                ///< Relative true linear RMS target for LU-SGS and GMRES.
        real linearAbsoluteTolerance = 1e-10;                               ///< Absolute true linear RMS target; combined with relative tolerance by OR.
        GMRESPreconditionerType gmresPreconditioner =
            GMRESPreconditionerType::LUSGS; ///< Left preconditioner used by GMRES.

        DNDS_DECLARE_CONFIG(TimeMarchSettings)
        {
            DNDS_FIELD(
                integrator,
                "ACM steady pseudo-time or physical dual-time integrator",
                DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(TimeIntegratorType)));
            DNDS_FIELD(nSteps, "Number of outer pseudo-time steps, or physical steps for BDF2",
                       DNDS::Config::range(0));
            DNDS_FIELD(pseudoTimeStep, "Fixed pseudo-time step", DNDS::Config::range(0.0));
            DNDS_FIELD(physicalTimeStep, "Uniform physical time step for BDF2 dual-time marching",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(useCFLTimeStep, "Use Euler-style local CFL pseudo-time steps");
            DNDS_FIELD(useLocalTimeStep, "Use cell-local rather than globally uniform CFL steps");
            DNDS_FIELD(cfl, "ACM CFL number", DNDS::Config::range(0.0));
            DNDS_FIELD(maximumPseudoTimeStep, "Maximum CFL-derived pseudo-time step", DNDS::Config::range(0.0));
            DNDS_FIELD(maxImplicitIterations, "Maximum inner iterations per implicit or physical step",
                       DNDS::Config::range(1));
            DNDS_FIELD(implicitTolerance, "Implicit global RMS steady or physical defect tolerance",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(dualTimeConvergenceCriterion, "BDF convergence: Absolute, Relative, AbsoluteOrRelative, or laminar RelativeUpdate; residuals remain reported",
                       DNDS::Config::enum_values({"Absolute", "Relative", "AbsoluteOrRelative", "RelativeUpdate"}));
            DNDS_FIELD(dualTimeUpdateRelativeTolerance, "Laminar BDF applied-update L2 / updated-state L2 threshold; zero disables update early exit",
                       DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(dualTimeUpdateMinimumIterations, "Minimum BDF corrections before testing RelativeUpdate",
                       DNDS::Config::range(0));
            DNDS_FIELD(dualTimeFlowRelativeTolerance, "BDF flow RMS target relative to this physical step's initial flow defect; zero uses per-equation targets",
                       DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(dualTimeTurbulenceRelativeTolerance, "BDF active RANS RMS target relative to its own initial physical defect; zero uses per-equation targets",
                       DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(dualTimeResidualScales, "Positive characteristic residual scales for BDF [u,v,w,p,q1,q2]");
            DNDS_FIELD(dualTimeRelativeTolerances, "BDF per-equation relative defect targets in [0,1); zero uses only the scaled absolute target");
            DNDS_FIELD(dualTimeRequireConvergence, "Reject unconverged BDF physical steps; false explicitly enables diagnostic budget-limited marching");
            DNDS_FIELD(steadyRelativeTolerance, "Steady inner defect relative to current outer spatial residual; zero uses absolute tolerance",
                       DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(steadyAdaptiveCFL, "Adapt steady implicit CFL using actual spatial residual progress");
            DNDS_FIELD(steadyCFLMin, "Minimum adaptive steady CFL", DNDS::Config::range(0.0));
            DNDS_FIELD(steadyCFLMax, "Maximum adaptive steady CFL", DNDS::Config::range(0.0));
            DNDS_FIELD(steadyCFLGrowth, "Maximum steady CFL growth factor", DNDS::Config::range(1.0));
            DNDS_FIELD(steadyCFLReduction, "Steady CFL reduction factor", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(implicitRelaxation, "Implicit correction relaxation", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(lusgsSweeps, "Maximum standalone LU-SGS sweeps; fixed sweeps for GMRES preconditioning", DNDS::Config::range(1));
            DNDS_FIELD(gmresSubspace, "ACM GMRES Krylov subspace size", DNDS::Config::range(2));
            DNDS_FIELD(gmresRestarts, "ACM GMRES restart count", DNDS::Config::range(0));
            DNDS_FIELD(linearRelativeTolerance, "True linear RMS target relative to the initial right-hand side; OR with linearAbsoluteTolerance", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(linearAbsoluteTolerance, "Absolute true linear RMS target for LU-SGS and GMRES; OR with linearRelativeTolerance", DNDS::Config::range(0.0));
            DNDS_FIELD(gmresPreconditioner, "ACM GMRES left preconditioner",
                       DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(GMRESPreconditionerType)));
            config.post_read([](T &settings)
                             { settings.Validate(); });
        }

        /**
         * @brief Validate steady pseudo-time and BDF2 dual-time integration controls.
         * @throws std::runtime_error If a count, time step, tolerance, or relaxation is invalid.
         */
        void Validate() const;
        /**
         *  @brief Validate the relative targets of all active physical equations or their RMS groups.
         * @param activeEquations Number of active state equations, from four flow equations to six with RANS.
         * @throws std::runtime_error If active equation/group settings cannot define the selected convergence
         *     criterion.
         */
        void ValidateDualTimeConvergence(int activeEquations) const;
        /**
         *  @brief Return the absolute, relative, or maximum of both RMS targets for one group.
         * @param initialDefect Initial global RMS defect of the selected flow or turbulence group.
         * @param turbulence True to use the turbulence-group relative tolerance; false for the flow group.
         * @return Group RMS threshold for the configured absolute, relative, or absolute-or-relative criterion.
         */
        real DualTimeGroupTarget(real initialDefect, bool turbulence) const;
        /**
         *  @brief Per-equation targets, including the scaled absolute fallback in OR mode.
         * @param initialDefects Initial equation RMS defects ordered [u,v,w,p,q1,q2]; inactive entries may be
         *     zero.
         * @return Six equation thresholds formed from characteristic scales, absolute tolerance, and enabled
         *     relative tolerances.
         */
        std::array<real, 6> DualTimeTargets(const std::array<real, 6> &initialDefects) const;
        /**
         *  @brief Require every active group to pass the selected criterion, using grouped RMS where enabled.
         * @param initialDefects Initial physical-defect RMS values in [u,v,w,p,q1,q2] order.
         * @param finalDefects Current physical-defect RMS values in the same order.
         * @param activeEquations Number of active entries, four to six.
         * @return True if every active equation or enabled RMS group satisfies its selected defect criterion.
         */
        bool DualTimeConverged(const std::array<real, 6> &initialDefects,
                               const std::array<real, 6> &finalDefects, int activeEquations) const;
        /**
         *  @brief Use grouped RMS where enabled, otherwise per-equation targets; all active groups must pass.
         * @param report Physical-step report with initial/current equation and group norms and any applied-update
         *     diagnostics.
         * @param activeEquations Number of active flow-plus-turbulence equations, four to six.
         * @return True when the configured stopping criterion passes, including RelativeUpdate when selected.
         */
        bool DualTimeConverged(const TimeStepReport &report, int activeEquations) const;
        /**
         *  @brief Defect-only diagnostic; RelativeUpdate uses the absolute-or-relative residual targets.
         * @param report Physical-step report with initial/current equation and group norms and any applied-update
         *     diagnostics.
         * @param activeEquations Number of active flow-plus-turbulence equations, four to six.
         * @return True when residual-based targets pass; RelativeUpdate is evaluated using absolute-or-relative
         *     residual targets.
         */
        bool DualTimeResidualConverged(const TimeStepReport &report, int activeEquations) const;
        /**
         *  @brief Absolute/relative target for one steady implicit pseudo-time step.
         * @param initialSpatialResidual Global spatial RMS at the start of this outer steady iteration.
         * @return Maximum of the absolute tolerance and the enabled relative tolerance times the initial spatial
         *     RMS.
         */
        real SteadyImplicitTarget(real initialSpatialResidual) const;
        /**
         *  @brief Bounded CFL update; stalled steps cannot trigger a CFL increase.
         * @param currentCFL CFL used for the just-completed steady iteration.
         * @param before Global spatial residual norm before the step.
         * @param after Global spatial residual norm after the step.
         * @param innerConverged Whether the implicit inner defect solve met its target.
         * @return Next CFL, adjusted for actual progress and bounded by the configured limits when adaptation is
         *     active.
         */
        real NextSteadyCFL(real currentCFL, real before, real after, bool innerConverged) const;
    };

    using StateField = std::vector<State>;    ///< Rank-local owned ACM states.
    using MatrixField = std::vector<Matrix4>; ///< Rank-local block-diagonal matrices.
    using ScalarField = std::vector<real>;    ///< One scalar value per rank-local state.

    /**
     * @brief Callback that evaluates the raw spatial residual `R(U)` for all local states.
     * @param states Current rank-local states.
     * @param residual Output field resized or filled to `states.size()`.
     */
    using ResidualEvaluator = std::function<void(const StateField &states, StateField &residual)>;

    /**
     * @brief Callback that evaluates the block diagonal `dR_i/dU_i` used by the implicit solver.
     * @param states Current rank-local states.
     * @param diagonalJacobian Output matrix field resized or filled to `states.size()`.
     */
    using DiagonalJacobianEvaluator =
        std::function<void(const StateField &states, MatrixField &diagonalJacobian)>;

    /**
     * @brief Diagnostics returned by one outer pseudo-time step or physical BDF step.
     * @details Residual/defect norms use global owned-cell data. Equation arrays follow
     * [u,v,w,p,q1,q2], with inactive entries zero; grouped flow and RANS norms are reported
     * separately. Applied-update norms are unnormalized L2 values rather than RMS residuals.
     * @note converged refers to the selected stopping criterion. RelativeUpdate can pass
     * without satisfying the independently reported physical-defect convergence test.
     */
    struct TimeStepReport
    {
        int iterations = 0;         ///< Explicit stages or implicit nonlinear iterations.
        real initialDefectNorm = 0; ///< Global RMS derivative/defect before updating.
        real finalDefectNorm = 0;   ///< Global RMS derivative/defect after updating.
        bool converged = false;     ///< True when the selected stopping criterion was reached.
        real appliedUpdateNorm = std::numeric_limits<real>::infinity(); ///< Global owned-cell L2 of the actual damped flow update.
        real updatedStateNorm = 0;  ///< Global owned-cell L2 of [u,v,w,p] after the update.
        real defectTolerance = 0;   ///< Actual absolute or combined steady inner target.
        real turbulenceDefectNorm = 0; ///< Active RANS physical-time defect during BDF2 marching.
        real initialTurbulenceDefectNorm = 0; ///< RANS RMS before any update in this physical step.
        real turbulenceDefectTolerance = 0;   ///< Absolute/relative RANS group target fixed for this physical step.
        std::array<real, 6> initialEquationDefectNorms{}; ///< Raw initial RMS values [u,v,w,p,q1,q2].
        std::array<real, 6> equationDefectNorms{};        ///< Raw final RMS values; inactive entries remain zero.

        /** @brief Relative applied update; zero/zero is zero, an unavailable or nonzero/zero update is infinity. */
        real RelativeUpdateNorm() const
        {
            return updatedStateNorm > 0 ? appliedUpdateNorm / updatedStateNorm
                                        : (updatedStateNorm == 0 && appliedUpdateNorm == 0
                                               ? 0 : std::numeric_limits<real>::infinity());
        }

        /** @brief Ratio to this step's initial flow RMS; zero/zero is zero, nonzero/zero is infinity. */
        real FlowDefectRatio() const
        {
            return initialDefectNorm > 0 ? finalDefectNorm / initialDefectNorm
                                        : (finalDefectNorm == 0 ? 0 : std::numeric_limits<real>::infinity());
        }
        /** @brief Ratio to this step's own initial active RANS RMS. */
        real TurbulenceDefectRatio() const
        {
            return initialTurbulenceDefectNorm > 0 ? turbulenceDefectNorm / initialTurbulenceDefectNorm
                                                  : (turbulenceDefectNorm == 0 ? 0 : std::numeric_limits<real>::infinity());
        }
    };

    /**
     * @brief Apply the inverse ACM pseudo-time mass matrix to a raw spatial residual field.
     * @param states States defining the nonlinear preconditioning matrices.
     * @param residual Raw spatial residual field `R(U)`.
     * @param pseudoDerivative Output `Gamma(U)^{-1} R(U)` field.
     * @param settings ACM physical and preconditioning settings.
     */
    void ApplyGammaInverseToResidual(
        const StateField &states,
        const StateField &residual,
        StateField &pseudoDerivative,
        const Settings &settings);

    /**
     * @brief Advance one pseudo-time step with the three-stage third-order SSP Runge-Kutta method.
     * @param states Rank-local states updated in place.
     * @param pseudoTimeStep Positive local pseudo-time step for every state.
     * @param settings ACM physical and preconditioning settings.
     * @param residualEvaluator Callback evaluating the raw spatial residual.
     * @param mpi Optional communicator metadata used to report global RMS norms.
     * @return Stage count and global RMS derivative norms before and after the step.
     */
    TimeStepReport AdvanceExplicitSSPRK3(
        StateField &states,
        const ScalarField &pseudoTimeStep,
        const Settings &settings,
        const ResidualEvaluator &residualEvaluator,
        const MPIInfo *mpi = nullptr);

    /**
     * @brief Advance one backward-Euler pseudo-time step by nonlinear block-Jacobi iterations.
     * @param states Rank-local states updated in place.
     * @param pseudoTimeStep Positive local pseudo-time step for every state.
     * @param settings ACM physical and preconditioning settings.
     * @param timeSettings Implicit iteration count, tolerance, and damping controls.
     * @param residualEvaluator Callback evaluating the raw spatial residual `R(U)`.
     * @param diagonalJacobianEvaluator Callback evaluating block diagonals `dR_i/dU_i`.
     * @param mpi Optional communicator metadata used for global convergence norms.
     * @return Nonlinear iteration count, initial/final defect norms, and convergence status.
     */
    TimeStepReport AdvanceImplicitEulerBlockJacobi(
        StateField &states,
        const ScalarField &pseudoTimeStep,
        const Settings &settings,
        const TimeMarchSettings &timeSettings,
        const ResidualEvaluator &residualEvaluator,
        const DiagonalJacobianEvaluator &diagonalJacobianEvaluator,
        const MPIInfo *mpi = nullptr);

    /**
     * @brief Dispatch one pseudo-time step to the configured explicit or implicit algorithm.
     * @param states Rank-local states updated in place.
     * @param pseudoTimeStep Positive local pseudo-time steps.
     * @param settings ACM physical and preconditioning settings.
     * @param timeSettings Time-integration selection and nonlinear controls.
     * @param residualEvaluator Raw spatial-residual callback.
     * @param diagonalJacobianEvaluator Implicit block-diagonal callback; ignored by SSPRK3.
     * @param mpi Optional communicator metadata for global diagnostics.
     * @return Report produced by the selected algorithm.
     */
    TimeStepReport AdvancePseudoTimeStep(
        StateField &states,
        const ScalarField &pseudoTimeStep,
        const Settings &settings,
        const TimeMarchSettings &timeSettings,
        const ResidualEvaluator &residualEvaluator,
        const DiagonalJacobianEvaluator &diagonalJacobianEvaluator = {},
        const MPIInfo *mpi = nullptr);
}
