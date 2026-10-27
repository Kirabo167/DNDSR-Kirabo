/**
 * @file ACMPhysicalTime.hxx
 * @brief Separate physical DAE and pseudo-time ODE adapters for shared time engines.
 * @author Runzhi Ma
 * @date 2026-09-03
 * @details Pressure is NOT stored in differential histories. No Euler or Solver file is
 * changed. The 5x5 flow solve and segregated conservative RANS relaxation form an
 * approximate block nonlinear solver; accepted stages must satisfy all residuals.
 */
#pragma once
#include "ACMSolver.hpp"
#include "Solver/ODE.hpp"

namespace DNDS::ACMVariable
{
    template <ACMModel model>
    /** @copydoc ACMSolver::RunGenericPhysical */
    void ACMSolver<model>::RunGenericPhysical()
    {
        using TQ = CFV::tUDof<7>; // [rho,mx,my,mz,0,rho*phi1,rho*phi2].
        using TScalar = CFV::tUDof<1>;
        using TODE = ODE::ImplicitDualTimeStep<TQ, TScalar>;
        const auto &settings = _configuration.timeMarchSettings;
        const bool physical = settings.physicalODECode >= 0;
        const int code = physical ? settings.physicalODECode : settings.pseudoODECode;
        const real stepSize = physical ? settings.physicalTimeStep : settings.pseudoTimeStep;
        auto buildQ = [this](TQ &field) { _vfv->BuildUDof(field, 7); field.setConstant(0.0); };
        auto buildScalar = [this](TScalar &field) { _vfv->BuildUDof(field, 1); field.setConstant(0.0); };
        ssp<TODE> ode;
        if (code == 1 || code == 102 || code == 103)
            ode = std::make_shared<ODE::ImplicitVBDFDualTimeStep<TQ,TScalar>>(
                _mesh->NumCell(), buildQ, buildScalar, code == 103 ? 1 : 2);
        else if (code == 0 || code == 101 || (code >= 202 && code <= 204))
            ode = std::make_shared<ODE::ImplicitSDIRK4DualTimeStep<TQ,TScalar>>(
                _mesh->NumCell(), buildQ, buildScalar, code == 0 ? 1 : code == 101 ? 0 : code - 200);
        else if (code == 2 && !physical)
            ode = std::make_shared<ODE::ExplicitSSPRK3TimeStepAsImplicitDualTimeStep<TQ,TScalar>>(
                _mesh->NumCell(), buildQ, buildScalar, settings.useLocalTimeStep);
        else if (code == 401 || (code >= 411 && code <= 413))
            ode = std::make_shared<ODE::ImplicitHermite3SimpleJacobianDualStep<TQ,TScalar>>(
                _mesh->NumCell(), buildQ, buildScalar, 0.55, 0, 0, 0.9146, 0.0,
                code == 401 ? 0 : code - 411, 0); // No auxiliary p-multigrid stages.
        else
            DNDS_check_throw_info(false,
                "Unsupported physicalODECode: explicit/non-stiffly-accurate Euler ODE needs a separate projection");

        TQ q, increment;
        buildQ(q); buildQ(increment);
        std::array<TScalar,3> pressure;
        for (auto &p : pressure) buildScalar(p);
        typename TTurbulence::TTurbulenceDof turbulenceRHS;
        _vfv->BuildUDof(turbulenceRHS, 2);
        turbulenceRHS.setConstant(0.0);
        for (index cell=0; cell<_mesh->NumCell(); ++cell)
        {
            q[cell].template head<4>() = _u[cell].template head<4>();
            if (!physical) q[cell](4) = _u[cell](4);
            for (auto &p : pressure) p[cell](0) = _u[cell](4);
            if (_turbulence)
                q[cell].template tail<2>() = _u[cell](0) * _turbulence->GetState()[cell];
        }
        ScalarField steps(_mesh->NumCell(),settings.pseudoTimeStep);
        MatrixField diagonal;
        typename TEvaluator::FaceJacobianField faceJacobians;
        real physicalTime=0, lastNorm=0;
        std::array<real,3> constraintNorm{0,0,0};

        // Decode the differential variables using the pressure belonging to this stage.
        // Parameters: values = distributed Q, position = endpoint/midpoint stage slot.
        const auto decode = [&](const TQ &values, int position)
        {
            DNDS_check_throw_info(position >= 0 && position < 3, "Invalid physical stage slot");
            real valid=1;
            for (index cell=0; cell<_mesh->NumCell(); ++cell)
            {
                if (!values[cell].allFinite() || values[cell](0) <= _configuration.acmSettings.densityFloor)
                    valid=0;
                _u[cell].template head<4>() = values[cell].template head<4>();
                _u[cell](4) = physical ? pressure[position][cell](0) : values[cell](4);
                if (_turbulence && values[cell](0)>0)
                    _turbulence->GetState()[cell] = values[cell].template tail<2>() / values[cell](0);
            }
            real globalValid=0;
            MPI_Allreduce(&valid,&globalValid,1,DNDS_MPI_REAL,MPI_MIN,_mpi.comm);
            DNDS_check_throw_info(globalValid>0, "Physical stage rejected: invalid conservative state");
        };
        if (_configuration.outputSettings.interval>0 && _configuration.outputSettings.writeInitial)
            WriteFlowField(0,0);
        for (int step=1; step<=settings.nSteps; ++step)
        {
            std::vector<real> initialStageNorms;
            const auto frhs = [&](TQ &residual, TQ &values, TScalar &, int, real c, int position)
            {
                decode(values,position);
                const real boundaryTime = physical ? physicalTime + c * stepSize : 0;
                _evaluator->EvaluateRHS(_rhs,_u,boundaryTime);
                if (_turbulence)
                    _turbulence->EvaluateRHS(turbulenceRHS,_u,boundaryTime);
                real localConstraint=0;
                residual.setConstant(0.0);
                for (index cell=0; cell<_mesh->NumCell(); ++cell)
                {
                    if (physical)
                        residual[cell].template head<4>() = _rhs[cell].template head<4>();
                    else
                        residual[cell].template head<5>() = GammaInvLocal(
                            _u[cell], _configuration.acmSettings.beta2, _configuration.acmSettings.alpha) * _rhs[cell];
                    if (_turbulence) residual[cell].template tail<2>() = turbulenceRHS[cell];
                    if (physical) localConstraint += _rhs[cell](4)*_rhs[cell](4);
                }
                real globalConstraint=0;
                MPI_Allreduce(&localConstraint,&globalConstraint,1,DNDS_MPI_REAL,MPI_SUM,_mpi.comm);
                constraintNorm[position]=std::sqrt(globalConstraint/std::max(real(1),real(_mesh->NumCellGlobal())));
            };
            const auto fdt = [&](TQ &values, TScalar &dtau, real, int position)
            {
                decode(values,position);
                if (settings.useCFLTimeStep)
                    _evaluator->EvaluateTimeStep(steps,_u,settings.cfl,settings.maximumPseudoTimeStep,
                                                settings.useLocalTimeStep,physicalTime);
                for (index cell=0; cell<_mesh->NumCell(); ++cell) dtau[cell](0)=steps[cell];
            };
            const auto fsolve = [&](TQ &values, TQ &residual, TQ &, TScalar &dtau,
                                    real dt, real a, TQ &delta, int, real c, int position)
            {
                DNDS_check_throw_info(a>0, "DAE adapter needs a positive implicit stage coefficient");
                decode(values,position);
                const real boundaryTime = physical ? physicalTime + c * dt : 0;
                _evaluator->EvaluateRHS(_rhs,_u,boundaryTime);
                for (index cell=0; cell<_mesh->NumCell(); ++cell)
                {
                    steps[cell]=dtau[cell](0);
                    _linearRhs[cell] = residual[cell].template head<5>() / a;
                    if (physical)
                        _linearRhs[cell](4) = _rhs[cell](4);
                    else
                        _linearRhs[cell] = GammaLocal(_u[cell], _configuration.acmSettings.beta2,
                            _configuration.acmSettings.alpha) * State(_linearRhs[cell]);
                }
                _evaluator->AssembleImplicitLinearization(_u,steps,diagonal,faceJacobians,boundaryTime);
                for (index cell = 0; cell < _mesh->NumCell(); ++cell)
                    diagonal[cell] += (physical ? PhysicalTimeMassMatrix() :
                        GammaLocal(_u[cell], _configuration.acmSettings.beta2,
                                   _configuration.acmSettings.alpha)) / (a * dt);
                lastNorm = std::max(_linearRhs.norm2()/std::sqrt(std::max(real(1),real(5*_mesh->NumCellGlobal()))),
                                   residual.norm2()/std::sqrt(std::max(real(1),real(7*_mesh->NumCellGlobal()))));
                delta.setConstant(0.0);
                if (lastNorm<=settings.implicitTolerance) return;
                SolveImplicitCorrection(diagonal,faceJacobians,
                    settings.integrator==TimeIntegratorType::ImplicitEulerLUSGS ||
                    settings.integrator==TimeIntegratorType::BDF2DualTimeLUSGS);
                for (index cell=0; cell<_mesh->NumCell(); ++cell)
                {
                    delta[cell].template head<4>() = _linearIncrement[cell].template head<4>();
                    if (!physical) delta[cell](4) = _linearIncrement[cell](4);
                    if (_turbulence)
                        delta[cell].template tail<2>() = residual[cell].template tail<2>() /
                            (1/dt + a/(steps[cell]*_configuration.turbulenceSettings.transportTimeScale));
                    real relaxation=settings.implicitRelaxation;
                    if (delta[cell](0)<0)
                        relaxation=std::min(relaxation,0.9*(values[cell](0)-_configuration.acmSettings.densityFloor)/
                                             (-delta[cell](0)));
                    if (_turbulence)
                        for (int k=0;k<_turbulence->ActiveVariableCount();++k)
                        {
                            const real lower=_configuration.turbulenceSettings.minimumValue[k];
                            const real upper=_configuration.turbulenceSettings.maximumValue[k];
                            const real dLower=delta[cell](5+k)-lower*delta[cell](0);
                            const real dUpper=upper*delta[cell](0)-delta[cell](5+k);
                            if (dLower<0) relaxation=std::min(relaxation,0.9*(values[cell](5+k)-lower*values[cell](0))/(-dLower));
                            if (dUpper<0) relaxation=std::min(relaxation,0.9*(upper*values[cell](0)-values[cell](5+k))/(-dUpper));
                        }
                    relaxation=std::clamp(relaxation,real(0),real(1));
                    delta[cell] *= relaxation;
                    if (physical) pressure[position][cell](0) += relaxation*_linearIncrement[cell](4);
                }
            };
            const auto fstop = [&](int iteration, TQ &residual, int stage)
            {
                const real norm=std::max({lastNorm,
                    residual.norm2()/std::sqrt(std::max(real(1),real(7*_mesh->NumCellGlobal()))),
                    constraintNorm[0],constraintNorm[1]});
                if (initialStageNorms.size() <= static_cast<std::size_t>(stage))
                    initialStageNorms.resize(static_cast<std::size_t>(stage) + 1, -1);
                if (iteration == 1 || initialStageNorms[stage] < 0)
                    initialStageNorms[stage] = norm;
                const real target = settings.ImplicitTarget(initialStageNorms[stage]);
                DNDS_check_throw_info(iteration<settings.maxImplicitIterations || norm<=target,
                                      "Physical ODE stage rejected: nonlinear residual did not converge");
                return norm<=target;
            };
            const auto fincrement = [](TQ &values, TQ &delta, real scale, int)
            { values.addTo(delta,scale); };
            ode->Step(q,increment,frhs,fdt,fsolve,settings.maxImplicitIterations,fstop,fincrement,stepSize);
            decode(q,0);
            physicalTime+=stepSize;
            if (_mpi.rank==0) log()<<"ACMVariable "<<(physical ? "physical" : "pseudo")<<" ODE "<<code<<" step="<<step
                                    <<" time="<<physicalTime<<" defect="<<lastNorm<<std::endl;
            if (_configuration.outputSettings.interval>0 && step%_configuration.outputSettings.interval==0)
                WriteFlowField(step,physicalTime);
        }
    }
}
