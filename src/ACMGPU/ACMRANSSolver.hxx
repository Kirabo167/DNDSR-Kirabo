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

namespace DNDS::ACMGPU
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
        _flowSolver.GetTurbulenceTransport()->SetPreparationReuse(true);
        _flowSolver.GetTurbulenceTransport()->SetFaceVolumeFlux([this](index face, int point, real time)
        {
            return GetEvaluator()->FaceVolumeFlux(_flowSolver.GetState(), face, point, time);
        });
        auto &vfv = *GetReconstruction();
        for (TDof *field : {&_u, &_rhs, &_defect, &_increment, &_scaledLinearInput})
        {
            vfv.BuildUDof(*field, nVarsFixed);
            field->setConstant(0.0);
        }
        if (_configuration.timeMarchSettings.dualTimeRANSMatrixFreeJacobian)
            for (TDof *field : {&_matrixFreeRhs, &_matrixFreeMinusRhs})
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
        _physicalTime = _configuration.restartSettings.physicalTime;
        if (_mpi.rank == 0 && !_configuration.restartSettings.flowFile.empty() &&
            IsBDF2DualTimeIntegrator(_configuration.timeMarchSettings.integrator))
            log() << "ACM field-only restart at t=" << _physicalTime
                  << "; BDF history reset: first new step uses BE, then constant-step BDF2" << std::endl;
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
        _flowSolver.GetTurbulenceTransport()->InvalidatePreparation();
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
    void ACMSolver<model, true>::EvaluateRHS(TDof &rhs, real time, bool rebuildReconstruction)
    {
        ProjectState();
        auto &flow = _flowSolver.GetState();
        GetEvaluator()->EvaluateRHS(_flowRhs, flow, time, rebuildReconstruction);
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
                            _configuration.timeMarchSettings.physicalTimeStep +
                            _continuationMass * (_u[cell](variable) - old[i](variable));
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
        _matrixFreePhysical = physical && _configuration.timeMarchSettings.dualTimeRANSMatrixFreeJacobian;
        if (_matrixFreePhysical)
        {
            _matrixFreeBaseState = OwnedState();
            _matrixFreeSteps = steps;
            _matrixFreeTime = time;
            TState local = TState::Zero(), global;
            for (const auto &state : _matrixFreeBaseState)
                local += state.array().square().matrix();
            MPI::Allreduce(local.data(), global.data(), nVarsFixed, DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
            for (int variable = 0; variable < nVarsFixed; variable++)
                _matrixFreeStateScales(variable) = std::max(std::sqrt(global(variable) /
                    std::max<real>(1, GetMesh()->NumCellGlobal())), real(1e-12));
            // Zero transverse velocity or gauge pressure is not a vanishing
            // characteristic scale; use the common velocity/pressure units.
            const real velocityScale = std::max(real(1), _matrixFreeStateScales.template head<3>().norm());
            _matrixFreeStateScales.template head<3>().setConstant(velocityScale);
            _matrixFreeStateScales(3) = std::max(_matrixFreeStateScales(3),
                _configuration.acmSettings.rho0 * velocityScale * velocityScale);
            _matrixFreeBaseNorm = 0;
            for (int variable = 0; variable < nVarsFixed; variable++)
                _matrixFreeBaseNorm += global(variable) / std::pow(_matrixFreeStateScales(variable), 2);
            _matrixFreeBaseNorm = std::sqrt(_matrixFreeBaseNorm);
        }
        GetEvaluator()->AssembleImplicitLinearization(
            _flowSolver.GetState(), steps, _flowDiagonal, _faceJacobians, time);
        std::vector<Eigen::Matrix<real, 2, 2>> turbulenceDiagonal;
        _flowSolver.GetTurbulenceTransport()->EvaluateImplicitDiagonal(
            _flowSolver.GetState(), turbulenceDiagonal, time, &_turbulenceFaceJacobians);
        _diagonal.assign(static_cast<std::size_t>(GetMesh()->NumCell()), TMatrix::Zero());
        _turbulencePressureMass.assign(static_cast<std::size_t>(GetMesh()->NumCell()),
                                       Eigen::Vector<real, nTurbulence>::Zero());
        const auto base = GetBDF2Coefficients(_completedPhysicalSteps);
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
        {
            const auto i = static_cast<std::size_t>(cell);
            if (physical)
                _flowDiagonal[i] += (base.current / _configuration.timeMarchSettings.physicalTimeStep +
                                     _continuationMass) * PhysicalTimeMassMatrix();
            else
                _flowDiagonal[i] +=
                    PseudoTimeProductJacobian(State(_u[cell].template head<4>()),
                                              State(old[i].template head<4>()), steps[i], _configuration.acmSettings) -
                    GammaLocal(State(_u[cell].template head<4>()), _configuration.acmSettings.beta2,
                               _configuration.acmSettings.alpha) / steps[i];
            _diagonal[i].template topLeftCorner<4, 4>() = _flowDiagonal[i];
            _diagonal[i].template bottomRightCorner<nTurbulence, nTurbulence>() =
                turbulenceDiagonal[i].template topLeftCorner<nTurbulence, nTurbulence>();
            if (physical)
            {
                // Primitive-scalar pseudo-time preconditioning in conservative variables:
                // add (rho*q)/beta^2 * dp/dtau, matching the continuity pseudo mass.
                // This changes the iteration matrix, never the physical BDF defect.
                _turbulencePressureMass[i] = _u[cell].template tail<nTurbulence>() /
                    (_configuration.acmSettings.beta2 * steps[i]);
                _diagonal[i].template block<nTurbulence, 1>(4, 3) = _turbulencePressureMass[i];
            }
            for (int variable = 4; variable < nVarsFixed; variable++)
            {
                _diagonal[i](variable, variable) +=
                    1 / (steps[i] * _configuration.turbulenceSettings.transportTimeScale);
                if (physical)
                    _diagonal[i](variable, variable) += EquationCoefficients(cell, variable, base).current /
                                                        _configuration.timeMarchSettings.physicalTimeStep + _continuationMass;
            }
        }
        const auto transport = _flowSolver.GetTurbulenceTransport();
        _faceAdvectedConservativeTurbulence.resize(static_cast<std::size_t>(GetMesh()->NumFaceProc()));
        for (index face = 0; face < GetMesh()->NumFaceProc(); face++)
        {
            const TurbulenceState value = _configuration.acmSettings.rho0 * transport->FrozenFaceAdvectionValue(
                _flowSolver.GetState(), face, time);
            _faceAdvectedConservativeTurbulence[face] = value;
            const auto cells = GetMesh()->face2cell[face];
            if (cells[0] == cells[1])
                continue;
            const auto &blocks = _faceJacobians[face];
            if (cells[0] < GetMesh()->NumCell())
                _diagonal[cells[0]].template bottomLeftCorner<nTurbulence, 4>() +=
                    value.template head<nTurbulence>() * blocks.left.row(3) / GetReconstruction()->GetCellVol(cells[0]);
            if (cells[1] != UnInitIndex && cells[1] < GetMesh()->NumCell())
                _diagonal[cells[1]].template bottomLeftCorner<nTurbulence, 4>() -=
                    value.template head<nTurbulence>() * blocks.right.row(3) / GetReconstruction()->GetCellVol(cells[1]);
        }
    }

    template <ACMModel model>
    void ACMSolver<model, true>::ApplyMatrixFreeOperator(TDof &input, TDof &output)
    {
        real localNorm = 0, globalNorm = 0;
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
            localNorm += TState(input[cell]).cwiseQuotient(_matrixFreeStateScales).squaredNorm();
        MPI::Allreduce(&localNorm, &globalNorm, 1, DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
        if (globalNorm == 0)
        {
            output.setConstant(0.0);
            return;
        }
        real localSafeStep = 1e100, safeStep = 0;
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
            for (int variable = 0; variable < nVarsFixed; variable++)
                if (input[cell](variable) != 0)
                {
                    const real reference = variable < 4
                        ? std::max(std::abs(_matrixFreeBaseState[cell](variable)), _matrixFreeStateScales(variable))
                        : _matrixFreeBaseState[cell](variable);
                    localSafeStep = std::min(localSafeStep, 1e-3 * reference / std::abs(input[cell](variable)));
                }
        MPI::Allreduce(&localSafeStep, &safeStep, 1, DNDS_MPI_REAL, MPI_MIN, _mpi.comm);
        const real epsilon = CentralJacobianProbeStep(_matrixFreeBaseNorm, std::sqrt(globalNorm), safeStep);
        const auto turbulence = _flowSolver.GetTurbulenceTransport();
        turbulence->SetLimiterFrozen(true);
        try
        {
            // No extra variational sweep: only cell means, closure gradients,
            // eddy viscosity, numerical fluxes and sources are perturbed.
            for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
                _u[cell] = _matrixFreeBaseState[cell] + epsilon * input[cell];
            EvaluateRHS(_matrixFreeRhs, _matrixFreeTime, false);
            for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
                _u[cell] = _matrixFreeBaseState[cell] - epsilon * input[cell];
            EvaluateRHS(_matrixFreeMinusRhs, _matrixFreeTime, false);
        }
        catch (...)
        {
            turbulence->SetLimiterFrozen(false);
            AssignOwned(_matrixFreeBaseState);
            throw;
        }
        turbulence->SetLimiterFrozen(false);
        const auto base = GetBDF2Coefficients(_completedPhysicalSteps);
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
        {
            output[cell] = (_matrixFreeMinusRhs[cell] - _matrixFreeRhs[cell]) / (2 * epsilon);
            output[cell].template head<4>() += GammaLocal(
                State(_matrixFreeBaseState[cell].template head<4>()), _configuration.acmSettings.beta2,
                _configuration.acmSettings.alpha) * input[cell].template head<4>() / _matrixFreeSteps[cell];
            output[cell].template tail<nTurbulence>() += input[cell].template tail<nTurbulence>() /
                (_matrixFreeSteps[cell] * _configuration.turbulenceSettings.transportTimeScale) +
                _turbulencePressureMass[cell] * input[cell](3);
            for (int variable = 0; variable < nVarsFixed; variable++)
                if (variable != 3)
                    output[cell](variable) += (EquationCoefficients(cell, variable, base).current /
                                             _configuration.timeMarchSettings.physicalTimeStep + _continuationMass) * input[cell](variable);
        }
        AssignOwned(_matrixFreeBaseState);
    }

    template <ACMModel model>
    void ACMSolver<model, true>::ApplyOperator(TDof &input, TDof &output)
    {
        if (_matrixFreePhysical)
        {
            ApplyMatrixFreeOperator(input, output);
            return;
        }
        input.trans.startPersistentPull();
        input.trans.waitPersistentPull();
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
            _flowInput[cell] = input[cell].template head<4>();
        GetEvaluator()->ApplyImplicitLinearization(_flowInput, _flowDiagonal, _faceJacobians, _flowOutput);
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
        {
            output[cell].template head<4>() = _flowOutput[cell];
            output[cell].template tail<nTurbulence>() =
                _diagonal[static_cast<std::size_t>(cell)].template bottomRightCorner<nTurbulence, nTurbulence>() *
                input[cell].template tail<nTurbulence>();
            output[cell].template tail<nTurbulence>() +=
                _turbulencePressureMass[cell] * input[cell](3);
        }
        const auto &mesh = GetMesh();
        for (index face = 0; face < mesh->NumFaceProc(); face++)
        {
            const auto cells = mesh->face2cell[face];
            if (cells[0] == cells[1])
                continue;
            const auto &flowBlocks = _faceJacobians[face];
            real volumeFluxIncrement = (flowBlocks.left.row(3) * input[cells[0]].template head<4>())(0);
            if (cells[1] != UnInitIndex)
                volumeFluxIncrement += (flowBlocks.right.row(3) * input[cells[1]].template head<4>())(0);
            const Eigen::Vector<real, nTurbulence> fluxIncrement = volumeFluxIncrement *
                _faceAdvectedConservativeTurbulence[face].template head<nTurbulence>();
            if (cells[0] < mesh->NumCell())
                output[cells[0]].template tail<nTurbulence>() += fluxIncrement / GetReconstruction()->GetCellVol(cells[0]);
            if (cells[1] != UnInitIndex && cells[1] < mesh->NumCell())
                output[cells[1]].template tail<nTurbulence>() -= fluxIncrement / GetReconstruction()->GetCellVol(cells[1]);
            if (cells[1] == UnInitIndex)
                continue;
            const auto &blocks = _turbulenceFaceJacobians[static_cast<std::size_t>(face)];
            if (cells[0] < mesh->NumCell())
                output[cells[0]].template tail<nTurbulence>() +=
                    blocks[1].template topLeftCorner<nTurbulence, nTurbulence>() *
                    input[cells[1]].template tail<nTurbulence>() / GetReconstruction()->GetCellVol(cells[0]);
            if (cells[1] < mesh->NumCell())
                output[cells[1]].template tail<nTurbulence>() -=
                    blocks[0].template topLeftCorner<nTurbulence, nTurbulence>() *
                    input[cells[0]].template tail<nTurbulence>() / GetReconstruction()->GetCellVol(cells[1]);
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
        using TurbulenceMatrix = Eigen::Matrix<real, nTurbulence, nTurbulence>;
        using TurbulenceVector = Eigen::Vector<real, nTurbulence>;
        std::vector<TurbulenceMatrix> inverse(static_cast<std::size_t>(GetMesh()->NumCell()));
        std::vector<TurbulenceVector> rhs(static_cast<std::size_t>(GetMesh()->NumCell()));
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
        {
            output[cell].template head<4>() = _flowOutput[cell];
            const Eigen::Matrix<real, nTurbulence, nTurbulence> block =
                _diagonal[static_cast<std::size_t>(cell)].template bottomRightCorner<nTurbulence, nTurbulence>();
            const auto decomposition = block.fullPivLu();
            if (decomposition.isInvertible())
                inverse[cell] = decomposition.inverse();
            else
            {
                for (int column = 0; column < nTurbulence; column++)
                    inverse[cell].col(column) = SolveEquilibratedImplicitBlock<nTurbulence>(
                        block, TurbulenceVector(TurbulenceVector::Unit(column)));
                if (!_reportedScaledTurbulenceBlock)
                {
                    nlohmann::ordered_json entries = nlohmann::ordered_json::array();
                    for (int row = 0; row < nTurbulence; row++)
                    {
                        nlohmann::ordered_json values = nlohmann::ordered_json::array();
                        for (int column = 0; column < nTurbulence; column++)
                            values.push_back(block(row, column));
                        entries.push_back(std::move(values));
                    }
                    std::cerr << "ACM RANS block equilibration rank=" << _mpi.rank << " localCell=" << cell
                              << " unscaledRank=" << decomposition.rank() << " block=" << entries.dump() << std::endl;
                    _reportedScaledTurbulenceBlock = true;
                }
            }
            rhs[cell] = input[cell].template tail<nTurbulence>() - _turbulencePressureMass[cell] * _flowOutput[cell](3);
            output[cell].template tail<nTurbulence>().setZero();
        }
        output.trans.startPersistentPull();
        output.trans.waitPersistentPull();
        const auto mesh = GetMesh();
        // Block-triangular elimination uses the complete shared-flux cross operator,
        // including neighbours, rather than only its cell-local diagonal.
        for (index face = 0; face < mesh->NumFaceProc(); face++)
        {
            const auto cells = mesh->face2cell[face];
            if (cells[0] == cells[1])
                continue;
            const auto &blocks = _faceJacobians[face];
            real flux = (blocks.left.row(3) * output[cells[0]].template head<4>())(0);
            if (cells[1] != UnInitIndex)
                flux += (blocks.right.row(3) * output[cells[1]].template head<4>())(0);
            const TurbulenceVector crossFlux = flux * _faceAdvectedConservativeTurbulence[face].template head<nTurbulence>();
            if (cells[0] < mesh->NumCell())
                rhs[cells[0]] -= crossFlux / GetReconstruction()->GetCellVol(cells[0]);
            if (cells[1] != UnInitIndex && cells[1] < mesh->NumCell())
                rhs[cells[1]] += crossFlux / GetReconstruction()->GetCellVol(cells[1]);
        }
        if (!lusgs)
        {
            for (index cell = 0; cell < mesh->NumCell(); cell++)
                output[cell].template tail<nTurbulence>() = inverse[cell] * rhs[cell];
            return;
        }
        const int count = sweeps > 0 ? sweeps : _configuration.timeMarchSettings.lusgsSweeps;
        const auto sweepCell = [&](index cell)
        {
            TurbulenceVector value = rhs[cell];
            const auto faces = mesh->cell2face[cell];
            for (rowsize incidence = 0; incidence < faces.size(); incidence++)
            {
                const index face = faces[incidence];
                const auto cells = mesh->face2cell[face];
                if (cells[1] == UnInitIndex || cells[0] == cells[1])
                    continue;
                const auto &blocks = _turbulenceFaceJacobians[face];
                if (cells[0] == cell)
                    value -= blocks[1].template topLeftCorner<nTurbulence, nTurbulence>() *
                             output[cells[1]].template tail<nTurbulence>() / GetReconstruction()->GetCellVol(cell);
                else
                    value += blocks[0].template topLeftCorner<nTurbulence, nTurbulence>() *
                             output[cells[0]].template tail<nTurbulence>() / GetReconstruction()->GetCellVol(cell);
            }
            output[cell].template tail<nTurbulence>() = inverse[cell] * value;
        };
        for (int sweep = 0; sweep < count; sweep++)
        {
            for (index cell = 0; cell < mesh->NumCell(); cell++)
                sweepCell(cell);
            for (index cell = mesh->NumCell(); cell-- > 0;)
                sweepCell(cell);
            if (sweep + 1 < count)
            {
                output.trans.startPersistentPull();
                output.trans.waitPersistentPull();
            }
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
        const bool useSGS = lusgs || settings.gmresPreconditioner == GMRESPreconditionerType::LUSGS;
        TState variableScales = TState::Ones();
        if (!lusgs)
        {
            ApplyPreconditioner(_defect, _increment, useSGS, settings.lusgsSweeps);
            TState localScales = TState::Zero(), globalScales;
            for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
                localScales += TState(_increment[cell]).array().square().matrix();
            MPI::Allreduce(localScales.data(), globalScales.data(), nVarsFixed, DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
            for (int variable = 0; variable < nVarsFixed; variable++)
                variableScales(variable) = std::max(std::sqrt(globalScales(variable) /
                    std::max<real>(1, GetMesh()->NumCellGlobal())), real(1e-14));
        }
        if (_matrixFreePhysical)
        {
            // A cold omega field can be orders of magnitude below its first
            // implicit correction. Include that correction in the FD units so
            // resolving omega does not round transverse flow perturbations away.
            _matrixFreeStateScales = _matrixFreeStateScales.cwiseMax(variableScales);
            real localNorm = 0, globalNorm = 0;
            for (const auto &state : _matrixFreeBaseState)
                localNorm += state.cwiseQuotient(_matrixFreeStateScales).squaredNorm();
            MPI::Allreduce(&localNorm, &globalNorm, 1, DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
            _matrixFreeBaseNorm = std::sqrt(globalNorm);
        }
        const auto initial = NormReport(_defect);
        std::array<real, 6> linearTargets{}, linearResiduals{};
        for (int variable = 0; variable < nVarsFixed; variable++)
            linearTargets[variable] = std::max(settings.linearAbsoluteTolerance,
                settings.linearRelativeTolerance * initial.equationDefectNorms[variable]);
        const auto report = SolveImplicitLinearSystem(
            _defect, _increment, lusgs,
            settings.lusgsSweeps, settings.gmresSubspace, settings.gmresRestarts,
            settings.linearAbsoluteTolerance, settings.linearRelativeTolerance,
            std::sqrt(static_cast<real>(GetMesh()->NumCellGlobal()) * nVarsFixed),
            [&](TDof &input, TDof &output)
            {
                for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
                    _scaledLinearInput[cell] = TState(input[cell]).cwiseProduct(variableScales);
                ApplyOperator(_scaledLinearInput, output);
            },
            [&](TDof &input, TDof &output)
            {
                ApplyPreconditioner(input, output, useSGS, lusgs ? 1 : settings.lusgsSweeps);
                for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
                    output[cell] = TState(output[cell]).cwiseQuotient(variableScales);
            },
            [this](TDof &field)
            {
                GetReconstruction()->BuildUDof(field, nVarsFixed);
                field.setConstant(0.0);
            },
            [&](const TDof &residual)
            {
                const auto norms = NormReport(residual);
                linearResiduals = norms.equationDefectNorms;
                for (int variable = 0; variable < nVarsFixed; variable++)
                    if (linearResiduals[variable] > linearTargets[variable])
                        return false;
                return true;
            });
        for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
            _increment[cell] = TState(_increment[cell]).cwiseProduct(variableScales);
        if (_mpi.rank == 0)
            log() << "ACM linear solver=" << (lusgs ? "LUSGS" : "GMRES")
                  << " iterations=" << report.iterations << " trueResidual=" << report.finalResidual
                  << " initial=" << report.initialResidual << " target=" << report.tolerance
                  << " relativeResidual=" << (report.initialResidual > 0 ? report.finalResidual / report.initialResidual : 0)
                  << " converged=" << report.converged
                  << " equationResiduals=" << nlohmann::ordered_json(std::vector<real>(
                      linearResiduals.begin(), linearResiduals.begin() + nVarsFixed)).dump()
                  << " equationTargets=" << nlohmann::ordered_json(std::vector<real>(
                      linearTargets.begin(), linearTargets.begin() + nVarsFixed)).dump() << std::endl;
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
        const auto &settings = _configuration.timeMarchSettings;
        real continuationFactor = physical && _completedPhysicalSteps < settings.dualTimeRANSContinuationSteps
            ? settings.dualTimeRANSContinuationFactor : 1;
        _continuationMass = physical ? (continuationFactor - 1) / settings.physicalTimeStep : 0;
        FormDefect(old, steps, physical, time);
        TimeStepReport initial = NormReport(_defect), report = initial;
        report.initialDefectNorm = initial.finalDefectNorm;
        report.initialTurbulenceDefectNorm = initial.turbulenceDefectNorm;
        report.initialEquationDefectNorms = initial.equationDefectNorms;
        report.defectTolerance = physical ? settings.DualTimeGroupTarget(report.initialDefectNorm, false)
                                         : settings.SteadyImplicitTarget(report.initialDefectNorm);
        report.turbulenceDefectTolerance = physical ? settings.DualTimeGroupTarget(report.initialTurbulenceDefectNorm, true)
                                                   : settings.SteadyImplicitTarget(report.initialTurbulenceDefectNorm);
        const int iterationLimit = physical ? settings.RANSInnerIterationLimit(_completedPhysicalSteps)
                                            : settings.maxImplicitIterations;
        if (physical && _mpi.rank == 0)
            log() << "ACM physical step=" << _completedPhysicalSteps + 1
                  << " innerIterationLimit=" << iterationLimit << std::endl;
        const auto stageConverged = [&]()
        {
            return physical ? settings.DualTimeConverged(report, nVarsFixed)
                            : report.finalDefectNorm <= report.defectTolerance &&
                              report.turbulenceDefectNorm <= report.turbulenceDefectTolerance;
        };
        const auto converged = [&]() { return _continuationMass == 0 && stageConverged(); };
        const auto logDefect = [&]()
        {
            if (!physical)
                return;
            if (_continuationMass == 0 &&
                (report.iterations == 0 || report.iterations == 10 || report.iterations == iterationLimit))
            {
                std::array<real, 6> local{}, global{};
                const auto transport = _flowSolver.GetTurbulenceTransport();
                std::array<real, 2> largest{-1, -1}, maximum{}, nearLower{}, globalNearLower{};
                std::array<index, 2> worst{};
                for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
                {
                    const auto source = transport->CellSource(cell);
                    for (int variable = 0; variable < nTurbulence; variable++)
                    {
                        const real s = _configuration.acmSettings.rho0 * source(variable);
                        const real t = _rhs[cell](4 + variable) - s;
                        const real b = _rhs[cell](4 + variable) - _defect[cell](4 + variable);
                        local[variable] += s * s;
                        local[2 + variable] += t * t;
                        local[4 + variable] += b * b;
                        if (std::abs(_defect[cell](4 + variable)) > largest[variable])
                        {
                            largest[variable] = std::abs(_defect[cell](4 + variable));
                            worst[variable] = cell;
                        }
                        if (_u[cell](4 + variable) <= 1.1 * _configuration.acmSettings.rho0 *
                            _configuration.turbulenceSettings.minimumValue[variable])
                            nearLower[variable]++;
                    }
                }
                MPI::Allreduce(local.data(), global.data(), 6, DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
                MPI::Allreduce(largest.data(), maximum.data(), nTurbulence, DNDS_MPI_REAL, MPI_MAX, _mpi.comm);
                MPI::Allreduce(nearLower.data(), globalNearLower.data(), nTurbulence, DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
                for (int variable = 0; variable < nTurbulence; variable++)
                if (largest[variable] == maximum[variable] && GetMesh()->NumCell() > 0)
                {
                    const index cell = worst[variable];
                    const auto point = GetReconstruction()->GetCellQuadraturePPhys(cell, -1);
                    const TurbulenceState source = _configuration.acmSettings.rho0 * transport->CellSource(cell);
                    TurbulenceState spatialTransport = TurbulenceState::Zero(), physicalDerivative = TurbulenceState::Zero();
                    for (int entry = 0; entry < nTurbulence; entry++)
                    {
                        spatialTransport(entry) = _rhs[cell](4 + entry) - source(entry);
                        physicalDerivative(entry) = _rhs[cell](4 + entry) - _defect[cell](4 + entry);
                    }
                    std::cerr << "ACM turbulence defect hotspot step=" << (_completedPhysicalSteps + 1)
                              << " inner=" << report.iterations << " variable=" << (4 + variable)
                              << " rank=" << _mpi.rank << " cell=" << cell
                              << " xyz=" << point.transpose() << " wallDistance=" << transport->CellWallDistance(cell)
                              << " state=" << _u[cell].transpose() << " defect=" << _defect[cell].transpose()
                              << " source=" << source.transpose()
                              << " transport=" << spatialTransport.transpose() << " BDF=" << physicalDerivative.transpose()
                              << " nearLowerBoundCells=" << globalNearLower[variable] << std::endl;
                }
                if (_mpi.rank == 0)
                {
                    for (real &value : global)
                        value = std::sqrt(value / std::max<real>(1, GetMesh()->NumCellGlobal()));
                    log() << "ACM turbulence defect terms step=" << (_completedPhysicalSteps + 1)
                          << " inner=" << report.iterations << " source_transport_BDF_RMS="
                          << nlohmann::ordered_json(global).dump() << std::endl;
                }
            }
            if (_mpi.rank != 0)
                return;
            const auto equationTargets = settings.DualTimeTargets(report.initialEquationDefectNorms);
            std::vector<real> equationRatios(nVarsFixed);
            for (int variable = 0; variable < nVarsFixed; variable++)
                equationRatios[variable] = report.equationDefectNorms[variable] /
                    std::max(report.initialEquationDefectNorms[variable], real(1e-30));
            log() << "ACM dual-time defect step=" << (_completedPhysicalSteps + 1)
                  << " inner=" << report.iterations
                  << " eqResidual=" << nlohmann::ordered_json(std::vector<real>(
                      report.equationDefectNorms.begin(), report.equationDefectNorms.begin() + nVarsFixed)).dump()
                  << " flow=" << report.finalDefectNorm << " flowTarget=" << report.defectTolerance
                  << " turbulence=" << report.turbulenceDefectNorm
                  << " turbulenceTarget=" << report.turbulenceDefectTolerance
                  << " continuationFactor=" << continuationFactor
                  << " turbulenceCriterion=" << (settings.dualTimeTurbulenceRelativeTolerance > 0 ? "GroupRMS" : "PerEquation")
                  << " turbulenceEquationTargets=" << nlohmann::ordered_json(std::vector<real>(
                      equationTargets.begin() + 4, equationTargets.begin() + nVarsFixed)).dump()
                  << " equationRatios=" << nlohmann::ordered_json(equationRatios).dump() << std::endl;
        };
        logDefect();
        std::vector<std::pair<TStateField, TStateField>> mixingHistory;
        const int mixingDepth = physical ? settings.dualTimeRANSAndersonDepth : 0;
        real previousMixingNorm = std::numeric_limits<real>::infinity();
        TState mixingScales = TState::Ones();
        bool haveMixingScales = false, canRejectMixing = false;
        TStateField unmixedFallback;
        real mixingBaseMerit = 0;
        const auto physicalMerit = [&]()
        {
            const real flow = report.finalDefectNorm / std::max(report.defectTolerance, real(1e-30));
            real result = flow * flow;
            const auto targets = settings.DualTimeTargets(report.initialEquationDefectNorms);
            if (settings.dualTimeTurbulenceRelativeTolerance > 0)
            {
                const real ratio = report.turbulenceDefectNorm / std::max(report.turbulenceDefectTolerance, real(1e-30));
                result += ratio * ratio;
            }
            else
                for (int variable = 4; variable < nVarsFixed; variable++)
                {
                    const real ratio = report.equationDefectNorms[variable] / std::max(targets[variable], real(1e-30));
                    result += ratio * ratio / nTurbulence;
                }
            return result;
        };
        for (int iteration = 0; iteration < iterationLimit && !converged(); iteration++)
        {
            if (_continuationMass > 0 && stageConverged())
            {
                continuationFactor = std::max(real(1), continuationFactor * 0.5);
                _continuationMass = (continuationFactor - 1) / settings.physicalTimeStep;
                mixingHistory.clear();
                previousMixingNorm = std::numeric_limits<real>::infinity();
                haveMixingScales = canRejectMixing = false;
                FormDefect(old, steps, physical, time);
                const auto norms = NormReport(_defect);
                report.finalDefectNorm = norms.finalDefectNorm;
                report.turbulenceDefectNorm = norms.turbulenceDefectNorm;
                report.equationDefectNorms = norms.equationDefectNorms;
                report.iterations = iteration + 1;
                if (_mpi.rank == 0)
                    log() << "ACM nonlinear continuation step=" << _completedPhysicalSteps + 1
                          << " inner=" << report.iterations << " factor=" << continuationFactor << std::endl;
                logDefect();
                continue;
            }
            // Reject a bad accelerated candidate using the next available residual evaluation.
            // The fallback consumes an inner iteration, preserving the reconstruction budget.
            if (mixingDepth > 0 && canRejectMixing && physicalMerit() > 1.5 * mixingBaseMerit)
            {
                AssignOwned(unmixedFallback);
                FormDefect(old, steps, physical, time);
                const auto norms = NormReport(_defect);
                report.finalDefectNorm = norms.finalDefectNorm;
                report.turbulenceDefectNorm = norms.turbulenceDefectNorm;
                report.equationDefectNorms = norms.equationDefectNorms;
                report.iterations = iteration + 1;
                mixingHistory.clear();
                previousMixingNorm = std::numeric_limits<real>::infinity();
                canRejectMixing = false;
                if (_mpi.rank == 0)
                    log() << "ACM RANS Anderson rejected step=" << (_completedPhysicalSteps + 1)
                          << " inner=" << report.iterations << std::endl;
                logDefect();
                continue;
            }
            canRejectMixing = false;
            BuildImplicitSystem(old, steps, physical, time);
            SolveCorrection(physical ? BDF2UsesLUSGS(settings.integrator)
                                     : settings.integrator == TimeIntegratorType::ImplicitEulerLUSGS);
            const auto current = mixingDepth > 0 ? OwnedState() : TStateField{};
            for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
            {
                const TState state = _u[cell];
                _u[cell] += BoundedIncrement(state, TState(settings.implicitRelaxation * _increment[cell]));
            }
            if (mixingDepth > 0)
            {
                // Standard Anderson fixed-point residual is G(U)-U, rather than the
                // raw stiff PDE defect. Freeze one RMS scale per variable for this step.
                TStateField scaledDefect = OwnedState();
                if (!haveMixingScales)
                {
                    TState localScales = TState::Zero(), globalScales;
                    for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
                        localScales += TState(_u[cell] - current[cell]).array().square().matrix();
                    MPI::Allreduce(localScales.data(), globalScales.data(), nVarsFixed, DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
                    for (int variable = 0; variable < nVarsFixed; variable++)
                        mixingScales(variable) = std::max(std::sqrt(globalScales(variable) /
                            std::max<real>(1, GetMesh()->NumCellGlobal())), real(1e-14));
                    haveMixingScales = true;
                }
                for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
                    scaledDefect[cell] = TState(_u[cell] - current[cell]).cwiseQuotient(mixingScales);
                real localNorm = 0, globalNorm = 0;
                for (const auto &value : scaledDefect)
                    localNorm += value.squaredNorm();
                MPI::Allreduce(&localNorm, &globalNorm, 1, DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
                if (globalNorm > 4 * previousMixingNorm)
                    mixingHistory.clear();
                previousMixingNorm = globalNorm;
                mixingHistory.emplace_back(OwnedState(), std::move(scaledDefect));
                if (mixingHistory.size() > static_cast<std::size_t>(mixingDepth + 1))
                    mixingHistory.erase(mixingHistory.begin());
                const int depth = static_cast<int>(mixingHistory.size()) - 1;
                if (depth > 0)
                {
                    Eigen::MatrixXd localGram = Eigen::MatrixXd::Zero(depth, depth + 1);
                    const auto &latest = mixingHistory.back();
                    for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
                        for (int i = 0; i < depth; i++)
                        {
                            const TState difference = mixingHistory[i].second[cell] - latest.second[cell];
                            localGram(i, depth) += difference.dot(latest.second[cell]);
                            for (int j = 0; j <= i; j++)
                                localGram(i, j) += difference.dot(mixingHistory[j].second[cell] - latest.second[cell]);
                        }
                    Eigen::MatrixXd globalGram(depth, depth + 1);
                    MPI::Allreduce(localGram.data(), globalGram.data(), static_cast<int>(localGram.size()),
                                   DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
                    Eigen::MatrixXd gram = globalGram.leftCols(depth).selfadjointView<Eigen::Lower>();
                    const real trace = gram.trace();
                    if (trace > std::numeric_limits<real>::min() && std::isfinite(trace))
                    {
                        gram.diagonal().array() += 1e-12 * trace / depth;
                        const Eigen::VectorXd weights = gram.fullPivLu().solve(-globalGram.col(depth));
                        const real l1 = weights.lpNorm<1>() + std::abs(1 - weights.sum());
                        if (weights.allFinite() && l1 <= 100)
                        {
                            unmixedFallback = latest.first;
                            mixingBaseMerit = physicalMerit();
                            canRejectMixing = true;
                            for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
                            {
                                TState candidate = latest.first[cell];
                                for (int i = 0; i < depth; i++)
                                    candidate += weights(i) * (mixingHistory[i].first[cell] - latest.first[cell]);
                                _u[cell] = current[cell] + BoundedIncrement(current[cell], TState(candidate - current[cell]));
                            }
                            if (_mpi.rank == 0)
                                log() << "ACM RANS Anderson step=" << (_completedPhysicalSteps + 1)
                                      << " inner=" << (iteration + 1) << " depth=" << depth << " coefficientL1=" << l1 << std::endl;
                        }
                    }
                }
            }
            FormDefect(old, steps, physical, time);
            const auto norms = NormReport(_defect);
            report.finalDefectNorm = norms.finalDefectNorm;
            report.turbulenceDefectNorm = norms.turbulenceDefectNorm;
            report.equationDefectNorms = norms.equationDefectNorms;
            report.iterations = iteration + 1;
            logDefect();
        }
        const bool physicalEquationReached = _continuationMass == 0;
        if (!physicalEquationReached)
        {
            // Report the original BDF defect if the auxiliary solve exhausted its budget.
            for (index cell = 0; cell < GetMesh()->NumCell(); cell++)
                for (int variable = 0; variable < nVarsFixed; variable++)
                    if (variable != 3)
                        _defect[cell](variable) += _continuationMass * (_u[cell](variable) - old[cell](variable));
            _continuationMass = 0;
            continuationFactor = 1;
            const auto norms = NormReport(_defect);
            report.finalDefectNorm = norms.finalDefectNorm;
            report.turbulenceDefectNorm = norms.turbulenceDefectNorm;
            report.equationDefectNorms = norms.equationDefectNorms;
            logDefect();
        }
        report.converged = physicalEquationReached && stageConverged();
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
        if (TryHostSnapshotAudit()) return;
        if (GPU::Enabled())
        {
            RunResident();
            return;
        }
        DNDS_check_throw_info(GetEvaluator() != nullptr, "ACM RANS Run requires mesh initialization");
        const auto &settings = _configuration.timeMarchSettings;
        const bool physical = IsBDF2DualTimeIntegrator(settings.integrator);
        DNDS_check_throw_info(!settings.steadyUntilConverged,
                              "Until-converged steady control is implemented by the resident CUDA SST path");
        ScalarField steps(static_cast<std::size_t>(GetMesh()->NumCell()), settings.pseudoTimeStep);
        _previous = OwnedState();
        _previousPrevious = _previous;
        real cfl = settings.cfl;
        if (_configuration.outputSettings.interval > 0 && _configuration.outputSettings.writeInitial)
            WriteOutput(_configuration.restartSettings.completedSteps, _physicalTime);
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
