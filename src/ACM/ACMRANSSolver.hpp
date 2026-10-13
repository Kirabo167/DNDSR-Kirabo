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

        ACMSolver(const MPIInfo &mpi, const KernelConfiguration &configuration);
        void ReadMeshAndInitialize();
        void Run();
        /** @brief Evaluate the spatial residual for every conservative model component. */
        void EvaluateRHS(TDof &rhs, real time = 0);
        TDof &GetState() { return _u; }
        const TDof &GetState() const { return _u; }
        const TDof &GetResidual() const { return _rhs; }
        const auto &GetMesh() const { return _flowSolver.GetMesh(); }
        const auto &GetReconstruction() const { return _flowSolver.GetReconstruction(); }
        const auto &GetEvaluator() const { return _flowSolver.GetEvaluator(); }
        std::size_t CompletedPhysicalSteps() const { return _completedPhysicalSteps; }
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

        static KernelConfiguration FlowConfiguration(const KernelConfiguration &configuration);
        void ProjectState();
        void ImportInitialState();
        void ReadConservativeRestart();
        TStateField OwnedState() const;
        void AssignOwned(const TStateField &states);
        TState BoundedIncrement(const TState &state, const TState &increment) const;
        TimeStepReport NormReport(const TDof &residual) const;
        BDF2Coefficients EquationCoefficients(index cell, int variable, const BDF2Coefficients &base) const;
        void FormDefect(const TStateField &old, const ScalarField &steps, bool physical, real time);
        void BuildImplicitSystem(const TStateField &old, const ScalarField &steps, bool physical, real time);
        void ApplyOperator(TDof &input, TDof &output);
        void ApplyPreconditioner(TDof &input, TDof &output, bool lusgs);
        void SolveCorrection(bool lusgs);
        void AdvanceExplicit(const ScalarField &steps);
        TimeStepReport AdvanceImplicit(const ScalarField &steps, bool physical, real time);
        void WriteOutput(int step, real time);
    };

    extern template class ACMSolver<ACMModel::ConstantDensitySA2D>;
    extern template class ACMSolver<ACMModel::ConstantDensitySA3D>;
    extern template class ACMSolver<ACMModel::ConstantDensity2EQ2D>;
    extern template class ACMSolver<ACMModel::ConstantDensity2EQ3D>;
}
