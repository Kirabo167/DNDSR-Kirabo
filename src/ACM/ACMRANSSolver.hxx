/** @file ACMRANSSolver.hxx
 *  @brief Unified conservative ACM SA/two-equation marching implementation.
 */
#pragma once

#include "ACMRANSSolver.hpp"
#include "ACMLinear.hpp"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <hdf5.h>

namespace DNDS::ACM
{
    template <ACMModel model>
    KernelConfiguration ACMSolver<model, true>::FlowConfiguration(const KernelConfiguration &configuration)
    {
        ValidateModelConfiguration<model>(configuration);
        KernelConfiguration result = configuration;
        result.solver = ModelSelection<flowModel>();
        result.initialState.resize(4);
        for (int variable = 0; variable < nTurbulence; variable++)
            result.turbulenceSettings.initialValue[variable] =
                configuration.initialState[4 + variable] / configuration.acmSettings.rho0;
        return result;
    }

    template <ACMModel model>
    ACMSolver<model, true>::ACMSolver(const MPIInfo &mpi, const KernelConfiguration &configuration)
        : _mpi(mpi), _configuration(configuration), _flowSolver(mpi, FlowConfiguration(configuration))
    {
    }

    template <ACMModel model>
    void ACMSolver<model, true>::ReadMeshAndInitialize()
    {
        _flowSolver.ReadMeshAndInitialize();
        _flowSolver.GetTurbulenceTransport()->SetFaceVolumeFlux([this](index face, int point, real time)
        {
            return GetEvaluator()->FaceVolumeFlux(_flowSolver.GetState(), face, point, time);
        });
        auto &vfv = *GetReconstruction();
        for (TDof *field : {&_u, &_rhs, &_defect, &_increment})
        {
            vfv.BuildUDof(*field, nVarsFixed);
            field->setConstant(0.0);
        }
        for (TFlowDof *field : {&_flowRhs, &_flowInput, &_flowOutput})
        {
            vfv.BuildUDof(*field, 4);
            field->setConstant(0.0);
        }
        vfv.BuildUDof(_turbulenceRhs, 2);
        _turbulenceRhs.setConstant(0.0);
        ImportInitialState();
        ReadConservativeRestart();
        if (_mpi.rank == 0)
            log() << "ACM unified model=" << ModelName(model) << " nVars=" << nVarsFixed
                  << " state=[u,v,w,p,rho*q...]" << std::endl;
    }

    template <ACMModel model>
    void ACMSolver<model, true>::ImportInitialState()
    {
        const auto &flow = _flowSolver.GetState();
        const auto &turbulence = _flowSolver.GetTurbulenceTransport()->GetState();
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
        {
            _u[cell].template head<4>() = flow[cell];
            for (int variable = 0; variable < nTurbulence; variable++)
                _u[cell](4 + variable) = _configuration.acmSettings.rho0 * turbulence[cell](variable);
        }
        _u.trans.startPersistentPull();
        _u.trans.waitPersistentPull();
    }

    template <ACMModel model>
    void ACMSolver<model, true>::ReadConservativeRestart()
    {
        if (_configuration.restartSettings.flowFile.empty())
            return;
        const hid_t file = H5Fopen(_configuration.restartSettings.flowFile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        DNDS_check_throw_info(file >= 0, "Cannot reopen ACM conservative restart");
        try
        {
            const auto readScalar = [&](const std::string &name)
            {
                const std::string path = "/VTKHDF/CellData/" + name;
                const hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
                DNDS_check_throw_info(dataset >= 0, "Missing ACM restart scalar: " + name);
                const hid_t space = H5Dget_space(dataset);
                const hsize_t offset = static_cast<hsize_t>(GetMesh()->vtkCellOffset);
                const hsize_t count = static_cast<hsize_t>(GetMesh()->NumCell());
                hsize_t length = 0;
                const bool valid = H5Sget_simple_extent_ndims(space) == 1 &&
                    H5Sget_simple_extent_dims(space, &length, nullptr) >= 0 && length >= offset + count;
                if (!valid)
                {
                    H5Sclose(space);
                    H5Dclose(dataset);
                    throw std::runtime_error("Incompatible ACM restart scalar: " + name);
                }
                std::vector<real> values(static_cast<std::size_t>(count));
                herr_t status = 0;
                if (count > 0)
                {
                    status = H5Sselect_hyperslab(space, H5S_SELECT_SET, &offset, nullptr, &count, nullptr);
                    const hid_t memory = H5Screate_simple(1, &count, nullptr);
                    if (status >= 0 && memory >= 0)
                        status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memory, space, H5P_DEFAULT, values.data());
                    else
                        status = -1;
                    if (memory >= 0) H5Sclose(memory);
                }
                H5Sclose(space);
                H5Dclose(dataset);
                DNDS_check_throw_info(status >= 0, "Cannot read ACM restart scalar: " + name);
                return values;
            };
            if (H5Lexists(file, "/VTKHDF/CellData/Density", H5P_DEFAULT) > 0)
            {
                const auto density = readScalar("Density");
                for (real value : density)
                    DNDS_check_throw_info(std::isfinite(value) &&
                        std::abs(value - _configuration.acmSettings.rho0) <=
                            1e-12 * std::max<real>(1, _configuration.acmSettings.rho0),
                        "ACM conservative restart density does not match rho0");
            }
            int present = 0;
            for (int variable = 0; variable < nTurbulence; variable++)
            {
                const std::string path = std::string("/VTKHDF/CellData/rho_") +
                    TurbulenceVariableName(_configuration.turbulenceSettings.model, variable);
                present += H5Lexists(file, path.c_str(), H5P_DEFAULT) > 0;
            }
            DNDS_check_throw_info(present == 0 || present == nTurbulence,
                                  "ACM restart contains only part of the conservative RANS state");
            if (present == nTurbulence)
                for (int variable = 0; variable < nTurbulence; variable++)
                {
                    const std::string name = std::string("rho_") +
                        TurbulenceVariableName(_configuration.turbulenceSettings.model, variable);
                    const auto values = readScalar(name);
                    for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
                    {
                        const real value = values[static_cast<std::size_t>(cell)];
                        DNDS_check_throw_info(std::isfinite(value) && value > 0,
                                              "ACM conservative restart has an invalid RANS value");
                        const real primitiveCopy = _u[cell](4 + variable);
                        DNDS_check_throw_info(std::abs(value - primitiveCopy) <=
                                                  1e-12 * std::max<real>(1, std::abs(value)),
                                              "ACM restart primitive and conservative RANS datasets disagree");
                        _u[cell](4 + variable) = value;
                    }
                }
        }
        catch (...)
        {
            H5Fclose(file);
            throw;
        }
        H5Fclose(file);
        _u.trans.startPersistentPull();
        _u.trans.waitPersistentPull();
    }

    template <ACMModel model>
    void ACMSolver<model, true>::ProjectState()
    {
        auto &flow = _flowSolver.GetState();
        auto &turbulence = _flowSolver.GetTurbulenceTransport()->GetState();
        _u.trans.startPersistentPull();
        _u.trans.waitPersistentPull();
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
        {
            DNDS_check_throw_info(_u[cell].allFinite(), "ACM unified state is non-finite");
            flow[cell] = _u[cell].template head<4>();
            turbulence[cell].setZero();
            for (int variable = 0; variable < nTurbulence; variable++)
            {
                const real value = _u[cell](4 + variable) / _configuration.acmSettings.rho0;
                DNDS_check_throw_info(value > 0, "ACM conservative turbulence state must be positive");
                turbulence[cell](variable) = value;
            }
        }
    }

    template <ACMModel model>
    void ACMSolver<model, true>::EvaluateRHS(TDof &rhs, real time)
    {
        ProjectState();
        auto &flow = _flowSolver.GetState();
        GetEvaluator()->EvaluateRHS(_flowRhs, flow, time);
        _flowSolver.GetTurbulenceTransport()->EvaluateRHS(_turbulenceRhs, flow, time);
        rhs.setConstant(0.0);
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
        {
            rhs[cell].template head<4>() = _flowRhs[cell];
            for (int variable = 0; variable < nTurbulence; variable++)
                rhs[cell](4 + variable) = _configuration.acmSettings.rho0 * _turbulenceRhs[cell](variable);
            DNDS_check_throw_info(rhs[cell].allFinite(), "ACM conservative model RHS is non-finite");
        }
    }

    template <ACMModel model>
    typename ACMSolver<model, true>::TStateField ACMSolver<model, true>::OwnedState() const
    {
        TStateField result(static_cast<std::size_t>(GetMesh()->NumCell()));
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
            result[static_cast<std::size_t>(cell)] = _u[cell];
        return result;
    }

    template <ACMModel model>
    void ACMSolver<model, true>::AssignOwned(const TStateField &states)
    {
        DNDS_check_throw_info(states.size() == static_cast<std::size_t>(GetMesh()->NumCell()),
                              "ACM model state count differs from owned mesh cells");
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
            _u[cell] = states[static_cast<std::size_t>(cell)];
    }

    template <ACMModel model>
    typename ACMSolver<model, true>::TState ACMSolver<model, true>::BoundedIncrement(
        const TState &state, const TState &increment) const
    {
        DNDS_check_throw_info(increment.allFinite(), "ACM model correction is non-finite");
        TState result = increment;
        const real rho = _configuration.acmSettings.rho0;
        for (int variable = 0; variable < nTurbulence; variable++)
        {
            const int entry = variable + 4;
            const real lower = rho * _configuration.turbulenceSettings.minimumValue[variable];
            const real upper = rho * _configuration.turbulenceSettings.maximumValue[variable];
            if (result(entry) < 0)
                result(entry) = std::max(result(entry), -0.9 * std::max(state(entry) - lower, 0.0));
            else
                result(entry) = std::min(result(entry), 0.9 * std::max(upper - state(entry), 0.0));
        }
        return result;
    }

    template <ACMModel model>
    TimeStepReport ACMSolver<model, true>::NormReport(const TDof &residual) const
    {
        std::array<real, 6> local{}, global{};
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
            for (int variable = 0; variable < nVarsFixed; variable++)
                local[variable] += residual[cell](variable) * residual[cell](variable);
        MPI::Allreduce(local.data(), global.data(), nVarsFixed, DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
        TimeStepReport report;
        const real count = std::max<real>(1, GetMesh()->NumCellGlobal());
        real flowSquare = 0, turbulenceSquare = 0;
        for (int variable = 0; variable < nVarsFixed; variable++)
        {
            report.equationDefectNorms[variable] = std::sqrt(global[variable] / count);
            (variable < 4 ? flowSquare : turbulenceSquare) += global[variable];
        }
        report.finalDefectNorm = std::sqrt(flowSquare / (4 * count));
        report.turbulenceDefectNorm = std::sqrt(turbulenceSquare / (nTurbulence * count));
        return report;
    }

    template <ACMModel model>
    BDF2Coefficients ACMSolver<model, true>::EquationCoefficients(
        index cell, int variable, const BDF2Coefficients &base) const
    {
        if (variable < 4)
            return base;
        const auto indexCell = static_cast<std::size_t>(cell);
        return GetPositiveTurbulenceBDFCoefficients(
            _previous[indexCell](variable), _previousPrevious[indexCell](variable), base,
            _configuration.acmSettings.rho0 * _configuration.turbulenceSettings.minimumValue[variable - 4]);
    }

    template <ACMModel model>
    void ACMSolver<model, true>::FormDefect(
        const TStateField &old, const ScalarField &steps, bool physical, real time)
    {
        EvaluateRHS(_rhs, time);
        _defect = _rhs;
        const auto base = GetBDF2Coefficients(_completedPhysicalSteps);
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
        {
            const auto i = static_cast<std::size_t>(cell);
            if (physical)
            {
                for (int variable = 0; variable < nVarsFixed; variable++)
                    if (variable != 3)
                    {
                        const auto coefficients = EquationCoefficients(cell, variable, base);
                        _defect[cell](variable) -=
                            (coefficients.current * _u[cell](variable) +
                             coefficients.previous * _previous[i](variable) +
                             coefficients.previousPrevious * _previousPrevious[i](variable)) /
                            _configuration.timeMarchSettings.physicalTimeStep;
                    }
            }
            else
            {
                _defect[cell].template head<4>() -=
                    GammaLocal(State(_u[cell].template head<4>()), _configuration.acmSettings.beta2,
                               _configuration.acmSettings.alpha) *
                    (_u[cell].template head<4>() - old[i].template head<4>()) / steps[i];
                _defect[cell].template tail<nTurbulence>() -=
                    (_u[cell].template tail<nTurbulence>() - old[i].template tail<nTurbulence>()) /
                    (steps[i] * _configuration.turbulenceSettings.transportTimeScale);
            }
        }
    }

    template <ACMModel model>
    void ACMSolver<model, true>::BuildImplicitSystem(
        const TStateField &old, const ScalarField &steps, bool physical, real time)
    {
        GetEvaluator()->AssembleImplicitLinearization(
            _flowSolver.GetState(), steps, _flowDiagonal, _faceJacobians, time);
        std::vector<Eigen::Matrix<real, 2, 2>> turbulenceDiagonal;
        _flowSolver.GetTurbulenceTransport()->EvaluateImplicitDiagonal(
            _flowSolver.GetState(), turbulenceDiagonal, time);
        _diagonal.assign(static_cast<std::size_t>(GetMesh()->NumCell()), TMatrix::Zero());
        const auto base = GetBDF2Coefficients(_completedPhysicalSteps);
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
        {
            const auto i = static_cast<std::size_t>(cell);
            if (physical)
                _flowDiagonal[i] += base.current * PhysicalTimeMassMatrix() /
                                    _configuration.timeMarchSettings.physicalTimeStep;
            else
                _flowDiagonal[i] +=
                    PseudoTimeProductJacobian(State(_u[cell].template head<4>()),
                                              State(old[i].template head<4>()), steps[i], _configuration.acmSettings) -
                    GammaLocal(State(_u[cell].template head<4>()), _configuration.acmSettings.beta2,
                               _configuration.acmSettings.alpha) / steps[i];
            _diagonal[i].template topLeftCorner<4, 4>() = _flowDiagonal[i];
            _diagonal[i].template bottomRightCorner<nTurbulence, nTurbulence>() =
                turbulenceDiagonal[i].template topLeftCorner<nTurbulence, nTurbulence>();
            for (int variable = 4; variable < nVarsFixed; variable++)
            {
                _diagonal[i](variable, variable) +=
                    1 / (steps[i] * _configuration.turbulenceSettings.transportTimeScale);
                if (physical)
                    _diagonal[i](variable, variable) += EquationCoefficients(cell, variable, base).current /
                                                        _configuration.timeMarchSettings.physicalTimeStep;
            }
        }
    }

    template <ACMModel model>
    void ACMSolver<model, true>::ApplyOperator(TDof &input, TDof &output)
    {
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
            _flowInput[cell] = input[cell].template head<4>();
        GetEvaluator()->ApplyImplicitLinearization(_flowInput, _flowDiagonal, _faceJacobians, _flowOutput);
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
        {
            output[cell].template head<4>() = _flowOutput[cell];
            output[cell].template tail<nTurbulence>() =
                _diagonal[static_cast<std::size_t>(cell)].template bottomRightCorner<nTurbulence, nTurbulence>() *
                input[cell].template tail<nTurbulence>();
        }
    }

    template <ACMModel model>
    void ACMSolver<model, true>::ApplyPreconditioner(TDof &input, TDof &output, bool lusgs, int sweeps)
    {
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
            _flowInput[cell] = input[cell].template head<4>();
        if (lusgs)
            GetEvaluator()->SolveLUSGS(_flowInput, _flowDiagonal, _faceJacobians, _flowOutput,
                                      sweeps > 0 ? sweeps : _configuration.timeMarchSettings.lusgsSweeps);
        else
            GetEvaluator()->ApplyBlockJacobi(_flowInput, _flowDiagonal, _flowOutput);
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
        {
            output[cell].template head<4>() = _flowOutput[cell];
            const auto decomposition = _diagonal[static_cast<std::size_t>(cell)]
                                           .template bottomRightCorner<nTurbulence, nTurbulence>().fullPivLu();
            DNDS_check_throw_info(decomposition.isInvertible(), "ACM implicit RANS diagonal is singular");
            output[cell].template tail<nTurbulence>() = decomposition.solve(input[cell].template tail<nTurbulence>());
        }
    }

    template <ACMModel model>
    void ACMSolver<model, true>::SolveCorrection(bool lusgs)
    {
        _increment.setConstant(0.0);
        const auto integrator = _configuration.timeMarchSettings.integrator;
        if (integrator == TimeIntegratorType::ImplicitEulerBlockJacobi)
        {
            ApplyPreconditioner(_defect, _increment, lusgs);
            return;
        }
        const auto &settings = _configuration.timeMarchSettings;
        const auto report = SolveImplicitLinearSystem(
            _defect, _increment, lusgs,
            settings.lusgsSweeps, settings.gmresSubspace, settings.gmresRestarts,
            settings.linearAbsoluteTolerance, settings.linearRelativeTolerance,
            std::sqrt(static_cast<real>(GetMesh()->NumCellGlobal()) * nVarsFixed),
            [this](TDof &input, TDof &output) { ApplyOperator(input, output); },
            [&](TDof &input, TDof &output)
            {
                ApplyPreconditioner(input, output,
                    lusgs || settings.gmresPreconditioner == GMRESPreconditionerType::LUSGS,
                    lusgs ? 1 : settings.lusgsSweeps);
            },
            [this](TDof &field)
            {
                GetReconstruction()->BuildUDof(field, nVarsFixed);
                field.setConstant(0.0);
            });
        if (_mpi.rank == 0)
            log() << "ACM linear solver=" << (lusgs ? "LUSGS" : "GMRES")
                  << " iterations=" << report.iterations << " trueResidual=" << report.finalResidual
                  << " initial=" << report.initialResidual << " target=" << report.tolerance
                  << " relativeResidual=" << (report.initialResidual > 0 ? report.finalResidual / report.initialResidual : 0)
                  << " converged=" << report.converged << std::endl;
    }

    template <ACMModel model>
    void ACMSolver<model, true>::AdvanceExplicit(const ScalarField &steps)
    {
        const auto base = OwnedState();
        for (int stage = 0; stage < 3; stage++)
        {
            EvaluateRHS(_rhs);
            auto next = OwnedState();
            for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
            {
                const auto i = static_cast<std::size_t>(cell);
                const TState current = _u[cell];
                TState increment = TState::Zero();
                increment.template head<4>() = steps[i] *
                    GammaLocal(State(current.template head<4>()), _configuration.acmSettings.beta2,
                               _configuration.acmSettings.alpha).fullPivLu().solve(_rhs[cell].template head<4>());
                increment.template tail<nTurbulence>() = steps[i] *
                    _configuration.turbulenceSettings.transportTimeScale * _rhs[cell].template tail<nTurbulence>();
                const TState forward = current + BoundedIncrement(current, increment);
                next[i] = stage == 0 ? forward : stage == 1 ? TState(0.75 * base[i] + 0.25 * forward)
                                                            : TState(base[i] / 3 + (2.0 / 3) * forward);
            }
            AssignOwned(next);
        }
    }

    template <ACMModel model>
    TimeStepReport ACMSolver<model, true>::AdvanceImplicit(const ScalarField &steps, bool physical, real time)
    {
        const auto old = OwnedState();
        FormDefect(old, steps, physical, time);
        TimeStepReport initial = NormReport(_defect), report = initial;
        report.initialDefectNorm = initial.finalDefectNorm;
        report.initialTurbulenceDefectNorm = initial.turbulenceDefectNorm;
        report.initialEquationDefectNorms = initial.equationDefectNorms;
        const auto &settings = _configuration.timeMarchSettings;
        report.defectTolerance = physical ? settings.DualTimeGroupTarget(report.initialDefectNorm, false)
                                         : settings.SteadyImplicitTarget(report.initialDefectNorm);
        report.turbulenceDefectTolerance = physical ? settings.DualTimeGroupTarget(report.initialTurbulenceDefectNorm, true)
                                                   : settings.SteadyImplicitTarget(report.initialTurbulenceDefectNorm);
        const auto converged = [&]()
        {
            return physical ? settings.DualTimeConverged(report, nVarsFixed)
                            : report.finalDefectNorm <= report.defectTolerance &&
                              report.turbulenceDefectNorm <= report.turbulenceDefectTolerance;
        };
        for (int iteration = 0; iteration < settings.maxImplicitIterations && !converged(); iteration++)
        {
            BuildImplicitSystem(old, steps, physical, time);
            SolveCorrection(physical ? BDF2UsesLUSGS(settings.integrator)
                                     : settings.integrator == TimeIntegratorType::ImplicitEulerLUSGS);
            for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
            {
                const TState state = _u[cell];
                _u[cell] += BoundedIncrement(state, TState(settings.implicitRelaxation * _increment[cell]));
            }
            FormDefect(old, steps, physical, time);
            const auto norms = NormReport(_defect);
            report.finalDefectNorm = norms.finalDefectNorm;
            report.turbulenceDefectNorm = norms.turbulenceDefectNorm;
            report.equationDefectNorms = norms.equationDefectNorms;
            report.iterations = iteration + 1;
        }
        report.converged = converged();
        return report;
    }

    template <ACMModel model>
    void ACMSolver<model, true>::WriteOutput(int step, real time)
    {
        ProjectState();
        const auto &output = _configuration.outputSettings;
        std::ostringstream stamp;
        stamp << std::setw(8) << std::setfill('0') << step;
        const std::filesystem::path directory(output.directory);
        MPI_Comm comm = MPI_COMM_NULL;
        MPI_Comm_dup(_mpi.comm, &comm);
        GetMesh()->PrintParallelVTKHDFDataArray(
            (directory / (output.prefix + "_" + stamp.str())).string(), (directory / output.prefix).string(),
            2 + 2 * nTurbulence, 1, 0, 0,
            [this](int array) -> std::string
            {
                if (array == 0) return "Pressure";
                if (array == 1) return "Density";
                const int variable = (array - 2) % nTurbulence;
                const std::string name = TurbulenceVariableName(_configuration.turbulenceSettings.model, variable);
                return array < 2 + nTurbulence ? name : "rho_" + name;
            },
            [this](int array, index cell) -> real
            {
                if (array == 0) return _u[cell](3);
                if (array == 1) return _configuration.acmSettings.rho0;
                const int variable = (array - 2) % nTurbulence;
                return _u[cell](4 + variable) / (array < 2 + nTurbulence ? _configuration.acmSettings.rho0 : 1);
            },
            [](int) -> std::string { return "Velocity"; },
            [this](int, index cell, rowsize component) -> real { return _u[cell](component); },
            [](int) -> std::string { return {}; }, [](int, index) -> real { return 0; },
            [](int) -> std::string { return {}; }, [](int, index, rowsize) -> real { return 0; }, time, comm);
        MPI_Comm_free(&comm);
    }

    template <ACMModel model>
    void ACMSolver<model, true>::Run()
    {
        DNDS_check_throw_info(GetEvaluator() != nullptr, "ACM RANS Run requires mesh initialization");
        const auto &settings = _configuration.timeMarchSettings;
        const bool physical = IsBDF2DualTimeIntegrator(settings.integrator);
        ScalarField steps(static_cast<std::size_t>(GetMesh()->NumCell()), settings.pseudoTimeStep);
        _previous = OwnedState();
        _previousPrevious = _previous;
        real cfl = settings.cfl;
        if (_configuration.outputSettings.interval > 0 && _configuration.outputSettings.writeInitial)
            WriteOutput(_configuration.restartSettings.completedSteps, 0);
        for (int step = 1; step <= settings.nSteps; step++)
        {
            const auto usedCoefficients = GetBDF2Coefficients(_completedPhysicalSteps);
            const real time = physical ? _physicalTime + settings.physicalTimeStep : 0;
            ProjectState();
            if (settings.useCFLTimeStep)
                GetEvaluator()->EvaluateTimeStep(steps, _flowSolver.GetState(), cfl,
                                                settings.maximumPseudoTimeStep, settings.useLocalTimeStep, time);
            TimeStepReport report;
            if (settings.integrator == TimeIntegratorType::ExplicitSSPRK3)
            {
                EvaluateRHS(_rhs);
                const auto initial = NormReport(_rhs);
                AdvanceExplicit(steps);
                EvaluateRHS(_rhs);
                report = NormReport(_rhs);
                report.initialDefectNorm = initial.finalDefectNorm;
                report.initialTurbulenceDefectNorm = initial.turbulenceDefectNorm;
                report.converged = true;
            }
            else
                report = AdvanceImplicit(steps, physical, time);
            if (physical)
            {
                DNDS_check_throw_info(report.converged || !settings.dualTimeRequireConvergence,
                    fmt::format("ACM unified physical step {} did not converge after {} iterations: flow={}, turbulence={}; histories were not committed",
                                step, report.iterations, report.finalDefectNorm, report.turbulenceDefectNorm));
                _previousPrevious = _previous;
                _previous = OwnedState();
                _completedPhysicalSteps++;
                _physicalTime = time;
            }
            else if (settings.steadyAdaptiveCFL)
            {
                EvaluateRHS(_rhs);
                cfl = settings.NextSteadyCFL(cfl, report.initialDefectNorm, NormReport(_rhs).finalDefectNorm,
                                           report.converged);
            }
            const int absoluteStep = _configuration.restartSettings.completedSteps + step;
            real localMinimum = steps.empty() ? veryLargeReal : *std::min_element(steps.begin(), steps.end());
            real globalMinimum = 0;
            MPI::Allreduce(&localMinimum, &globalMinimum, 1, DNDS_MPI_REAL, MPI_MIN, _mpi.comm);
            if (_mpi.rank == 0)
            {
                log() << std::scientific;
                if (physical)
                    log() << "ACM physical step " << std::setw(8) << absoluteStep << " time=" << _physicalTime
                          << " BDF" << usedCoefficients.order;
                else
                    log() << "ACM step " << std::setw(8) << absoluteStep;
                log() << " residual " << report.initialDefectNorm << " -> " << report.finalDefectNorm
                      << " dtauMin=" << globalMinimum << " turbResidual=" << report.turbulenceDefectNorm
                      << " inner=" << report.iterations << " converged=" << report.converged
                      << " model=" << ModelName(model) << " nVars=" << nVarsFixed
                      << " eqResidual=" << nlohmann::ordered_json(std::vector<real>(
                          report.equationDefectNorms.begin(), report.equationDefectNorms.begin() + nVarsFixed)).dump();
                if (physical)
                    log() << " convergenceCriterion=" << settings.dualTimeConvergenceCriterion
                          << " flowRatio=" << report.FlowDefectRatio()
                          << " turbInitial=" << report.initialTurbulenceDefectNorm
                          << " turbRatio=" << report.TurbulenceDefectRatio();
                log() << std::endl;
            }
            if (_configuration.outputSettings.interval > 0 && absoluteStep % _configuration.outputSettings.interval == 0)
                WriteOutput(absoluteStep, physical ? _physicalTime : static_cast<real>(absoluteStep));
        }
        ProjectState();
    }
}
