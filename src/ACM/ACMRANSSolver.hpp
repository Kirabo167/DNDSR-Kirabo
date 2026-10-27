/** @file ACMRANSSolver.hpp
 *  @brief Euler-style unified ACM state, residual and integration for SA/2EQ models.
 */
#pragma once

#include "ACMSolver.hpp"
#include "ACMModelState.hpp"

namespace DNDS::ACM
{
    /**
     * @brief Five/six-variable ACM specialization with density-weighted RANS transport.
     * @details The full distributed state is authoritative. Four-variable flow and
     * primitive closure fields are scratch projections refreshed at every RHS call.
     * All active equations share RK stages, nonlinear iterates and physical histories.
     * The implicit operator freezes flow/closure cross derivatives, uses existing
     * flow face blocks, and a local coupled RANS source/spectral-radius block.
     */
    template <ACMModel model>
    class ACMSolver<model, true>
    {
    public:
        using Traits = ModelTraits<model>;
        static constexpr int gDim = Traits::gDim;
        static constexpr int nVarsFixed = Traits::nVarsFixed;
        static constexpr int nTurbulence = Traits::nExtraVars;
        static constexpr ACMModel flowModel = gDim == 2 ? ACMModel::ConstantDensity2D : ACMModel::ConstantDensity3D;
        using TState = ModelState<model>;
        using TMatrix = Eigen::Matrix<real, nVarsFixed, nVarsFixed>;
        using TDof = CFV::tUDof<nVarsFixed>;
        using TFlowSolver = ACMSolver<flowModel, false>;
        using TFlowDof = typename TFlowSolver::TDof;
        using TEvaluator = typename TFlowSolver::TEvaluator;
        using TTurbulence = typename TFlowSolver::TTurbulence;
        using TStateField = std::vector<TState>;

        /**
         * @brief Construct the unified flow/RANS solver and configure its four-variable flow subsystem.
         * @param mpi Communicator metadata copied into the solver and used for mesh/field collectives.
         * @param configuration Validated five- or six-variable case settings; turbulence entries use rho*q
         *     storage.
         */
        ACMSolver(const MPIInfo &mpi, const KernelConfiguration &configuration);
        /**
         * @brief Read and partition the mesh, allocate unified fields, and load initial or restart states.
         * @details Connects turbulence advection to the ACM continuity flux and synchronizes initialized state
         *     ghosts.
         * @note Collective over the solver communicator. Call before accessing mesh-dependent fields or Run().
         */
        void ReadMeshAndInitialize();
        /**
         * @brief Advance all flow and turbulence equations using the configured steady or BDF2 integrator.
         * @details Shares stages and nonlinear iterations across every equation; applies convergence controls,
         *     commits completed physical histories, and writes scheduled output.
         * @pre ReadMeshAndInitialize() has completed on every rank.
         */
        void Run();
        /**
         *  @brief Evaluate the spatial residual for every conservative model component.
         * @details Projects the authoritative unified state into flow/closure scratch fields and refreshes halo
         *     data before evaluation.
         * @param [out] rhs Allocated unified residual: four flow components followed by rho times each primitive
         *     turbulence RHS.
         * @param time Boundary-evaluation time used by both flow and turbulence transport.
         */
        void EvaluateRHS(TDof &rhs, real time = 0);
        /**
         * @brief Return unified distributed state [u,v,w,p,rho*q...]; mutable access changes the authoritative
         *     solution.
         * @return Unified distributed state [u,v,w,p,rho*q...]; mutable access changes the authoritative solution.
         */
        TDof &GetState() { return _u; }
        /**
         * @brief Return unified distributed state [u,v,w,p,rho*q...]; mutable access changes the authoritative
         *     solution.
         * @return Unified distributed state [u,v,w,p,rho*q...]; mutable access changes the authoritative solution.
         */
        const TDof &GetState() const { return _u; }
        /**
         * @brief Return read-only most recently stored unified spatial residual.
         * @return Read-only most recently stored unified spatial residual.
         */
        const TDof &GetResidual() const { return _rhs; }
        /**
         * @brief Return shared distributed mesh of the flow subsystem; available after initialization.
         * @return Shared distributed mesh of the flow subsystem; available after initialization.
         */
        const auto &GetMesh() const { return _flowSolver.GetMesh(); }
        /**
         * @brief Return shared CFV reconstruction engine of the initialized flow subsystem.
         * @return Shared CFV reconstruction engine of the initialized flow subsystem.
         */
        const auto &GetReconstruction() const { return _flowSolver.GetReconstruction(); }
        /**
         * @brief Return shared four-variable ACM spatial evaluator used by the unified solver.
         * @return Shared four-variable ACM spatial evaluator used by the unified solver.
         */
        const auto &GetEvaluator() const { return _flowSolver.GetEvaluator(); }
        /**
         * @brief Return number of completed physical steps committed by this solver.
         * @return Number of completed physical steps committed by this solver.
         */
        std::size_t CompletedPhysicalSteps() const { return _completedPhysicalSteps; }
        /**
         * @brief Return current physical simulation time; steady pseudo-time iterations do not advance it.
         * @return Current physical simulation time; steady pseudo-time iterations do not advance it.
         */
        real SimulationTime() const { return _physicalTime; }

    private:
        MPIInfo _mpi;
        KernelConfiguration _configuration;
        TFlowSolver _flowSolver;
        TDof _u, _rhs, _defect, _increment;
        TFlowDof _flowRhs, _flowInput, _flowOutput;
        typename TTurbulence::TTurbulenceDof _turbulenceRhs;
        MatrixField _flowDiagonal;
        typename TEvaluator::FaceJacobianField _faceJacobians;
        std::vector<TMatrix> _diagonal;
        TStateField _previous, _previousPrevious;
        std::size_t _completedPhysicalSteps = 0;
        real _physicalTime = 0;

        /**
         * @brief Build four-variable flow settings while retaining primitive turbulence closure controls.
         * @param configuration Full unified-model settings to validate and project.
         * @return Configuration with the flow executable selection, four flow initial entries, and q =
         *     (rho*q)/rho0 closure values.
         */
        static KernelConfiguration FlowConfiguration(const KernelConfiguration &configuration);
        /**
         * @brief Synchronize the unified state and refresh owned flow and primitive turbulence scratch fields.
         * @details Copies [u,v,w,p] and divides each turbulence entry by rho0; rejects non-finite states or
         *     nonpositive turbulence values.
         */
        void ProjectState();
        /**
         * @brief Combine initialized flow and primitive closure fields into the unified conservative state.
         * @details Multiplies the turbulence tail by rho0 and pulls owned values to unified-state ghosts.
         */
        void ImportInitialState();
        /**
         * @brief Check and import optional density-weighted RANS datasets from the configured VTKHDF restart.
         * @details Returns immediately without a restart file. Validates density, complete turbulence dataset
         *     coverage, and agreement with primitive restart values; then synchronizes ghosts.
         */
        void ReadConservativeRestart();
        /**
         * @brief Copy the authoritative owned-cell state into a contiguous local history field.
         * @return One full model state per owned cell, excluding ghosts.
         */
        TStateField OwnedState() const;
        /**
         * @brief Replace owned unified-state entries without performing a halo exchange.
         * @param states Full model states in owned-cell order; length must equal the owned-cell count.
         */
        void AssignOwned(const TStateField &states);
        /**
         * @brief Limit the turbulence correction to stay within configured primitive bounds after density
         *     weighting.
         * @param state Current full model state [u,v,w,p,rho*q...].
         * @param increment Proposed finite full-state correction before nonlinear relaxation.
         * @return Correction with unchanged flow entries and turbulence entries limited to 90% of their available
         *     bound distance.
         */
        TState BoundedIncrement(const TState &state, const TState &increment) const;
        /**
         * @brief Reduce owned residual squares and report global equation and group RMS norms.
         * @param residual Unified distributed residual; only owned cells enter the reductions.
         * @return Report containing per-equation RMS plus separate four-equation flow and active-turbulence RMS
         *     values.
         */
        TimeStepReport NormReport(const TDof &residual) const;
        /**
         * @brief Choose physical BDF coefficients for one cell and equation.
         * @param cell Owned-cell index into completed physical history fields.
         * @param variable Full state-component index: 0..3 for flow, 4 and above for turbulence.
         * @param base Nominal startup or BDF2 coefficients for this physical step.
         * @return Base flow coefficients, or turbulence coefficients with the history-based positivity fallback.
         */
        BDF2Coefficients EquationCoefficients(index cell, int variable, const BDF2Coefficients &base) const;
        /**
         * @brief Evaluate and store the nonlinear steady or physical-time defect of the unified state.
         * @details Stores R(U) minus the appropriate time derivative in _defect; physical pressure has no BDF mass
         *     term.
         * @param old Frozen owned state at the start of a steady pseudo-time step; unused for physical history
         *     derivatives.
         * @param steps Positive pseudo-time intervals in owned-cell order.
         * @param physical True to include physical BDF history terms; false for nonlinear backward Euler in
         *     pseudo-time.
         * @param time Time supplied to boundary and closure evaluation.
         */
        void FormDefect(const TStateField &old, const ScalarField &steps, bool physical, real time);
        /**
         * @brief Assemble flow face couplings and local turbulence blocks for the current implicit correction.
         * @details Freezes flow/closure cross derivatives; the RANS block contains source and spectral-radius
         *     terms. Requires the current state to have been projected.
         * @param old Frozen owned state at the start of a steady pseudo-time step; unused for physical history
         *     derivatives.
         * @param steps Positive pseudo-time intervals in owned-cell order.
         * @param physical True to include physical BDF history terms; false for nonlinear backward Euler in
         *     pseudo-time.
         * @param time Time supplied to boundary and closure evaluation.
         */
        void BuildImplicitSystem(const TStateField &old, const ScalarField &steps, bool physical, real time);
        /**
         * @brief Apply the assembled flow-coupled, turbulence-local implicit operator.
         * @param [in] input Unified correction vector; the flow scratch projection is synchronized by the
         *     evaluator.
         * @param [out] output Allocated unified operator product on owned cells.
         */
        void ApplyOperator(TDof &input, TDof &output);
        /**
         * @brief Apply the flow preconditioner and invert each local active-turbulence block.
         * @param [in] input Unified linear residual to precondition.
         * @param [out] output Allocated unified preconditioned vector.
         * @param lusgs True for flow LU-SGS, false for independent 4x4 flow block Jacobi.
         * @param sweeps Positive LU-SGS sweep override; a nonpositive value selects the configured sweep count.
         */
        void ApplyPreconditioner(TDof &input, TDof &output, bool lusgs, int sweeps = -1);
        /**
         * @brief Solve the current assembled defect system into the unified correction workspace.
         * @details Uses true linear-residual tolerances for LU-SGS/GMRES and logs their convergence report on rank
         *     zero.
         * @param lusgs True for stationary LU-SGS; false for GMRES or the configured block-Jacobi integration
         *     branch.
         */
        void SolveCorrection(bool lusgs);
        /**
         * @brief Advance the unified state through three SSPRK3 stages with bounded turbulence increments.
         * @param steps Positive per-owned-cell flow pseudo-time steps; turbulence uses its configured transport
         *     time scale.
         */
        void AdvanceExplicit(const ScalarField &steps);
        /**
         * @brief Perform damped nonlinear corrections for a steady pseudo-step or one physical BDF step.
         * @param steps Positive per-owned-cell pseudo-time intervals for the implicit mass/preconditioner.
         * @param physical True for physical BDF defects and histories, false for steady backward-Euler defects.
         * @param time Boundary-evaluation time for the new iterate.
         * @return Global flow/turbulence convergence report after the inner iteration budget or tolerance is
         *     reached.
         */
        TimeStepReport AdvanceImplicit(const ScalarField &steps, bool physical, real time);
        /**
         * @brief Project the unified state and write flow, primitive closure, and density-weighted RANS output.
         * @param step Output step index used in the file name and series metadata.
         * @param time Physical or pseudo-time value attached to the output.
         */
        void WriteOutput(int step, real time);
    };

    extern template class ACMSolver<ACMModel::ConstantDensitySA2D>;
    extern template class ACMSolver<ACMModel::ConstantDensitySA3D>;
    extern template class ACMSolver<ACMModel::ConstantDensity2EQ2D>;
    extern template class ACMSolver<ACMModel::ConstantDensity2EQ3D>;
}
