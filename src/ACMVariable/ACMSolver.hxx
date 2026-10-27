/**
 * @file ACMSolver.hxx
 * @brief Template implementation of the standalone ACM solver assembly.
 * @author Runzhi Ma
 * @date 2026-09-03
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMSolver.hpp"
#include "ACM/ACMLinear.hpp"

#include "DNDS/Errors.hpp"
#include "DNDS/OMP.hpp"

#include <filesystem>
#include <iomanip>
#include <sstream>

namespace DNDS::ACMVariable
{
    template <ACMModel model>
    ACMSolver<model>::ACMSolver(
        const MPIInfo &mpi,
        const KernelConfiguration &configuration)
        : _mpi(mpi), _configuration(configuration)
    {
        _configuration.Validate();
    }

    template <ACMModel model>
    void ACMSolver<model>::ReadMeshAndInitialize()
    {
        DNDS_check_throw_info(
            !_configuration.meshSettings.meshFile.empty(),
            "ACM meshSettings.meshFile must name a CGNS mesh");

        DNDS_MAKE_SSP(_mesh, _mpi, gDim);
        DNDS_MAKE_SSP(_reader, _mesh, 0);
        _mesh->periodicInfo.translation[1].map() = Eigen::Map<const Vector3>(
            _configuration.meshSettings.periodicTranslation1.data());
        _mesh->periodicInfo.translation[2].map() = Eigen::Map<const Vector3>(
            _configuration.meshSettings.periodicTranslation2.data());
        _mesh->periodicInfo.translation[3].map() = Eigen::Map<const Vector3>(
            _configuration.meshSettings.periodicTranslation3.data());
        DNDS_MAKE_SSP(
            _boundaryHandler,
            _configuration.defaultBoundaryType,
            _configuration.BoundaryValue(),
            _configuration.boundaryConditions);
        Geom::ReadMeshFromCGNS(
            _mesh,
            _reader,
            _configuration.meshSettings.meshFile,
            _configuration.meshSettings.partitionOptions,
            _configuration.meshSettings.periodicTolerance,
            _configuration.meshSettings.meshElevation,
            _configuration.meshSettings.meshDirectBisect,
            [this](const std::string &name) -> Geom::t_index
            { return _boundaryHandler->GetIDFromName(name); });

        Geom::PrepareMeshOptions preparationOptions;
        preparationOptions.reorderCells = _configuration.meshSettings.meshReorderCells;
#ifdef DNDS_USE_OMP
        preparationOptions.reorderParts = std::max(get_env_DNDS_DIST_OMP_NUM_THREADS(), 1);
#else
        preparationOptions.reorderParts = 1;
#endif
        preparationOptions.buildSerialOut = false;
        Geom::PrepareMesh(*_mesh, *_reader, preparationOptions);
        _mesh->RecreatePeriodicNodes();
        if (_configuration.outputSettings.interval > 0)
            _mesh->BuildVTKConnectivity();

        DNDS_MAKE_SSP(_vfv, _mpi, _mesh);
        _vfv->parseSettings(_configuration.vfvSettings);
        TEvaluator::InitializeFV(
            _mesh,
            _vfv,
            _boundaryHandler);

        _vfv->BuildUDof(_u, Traits::nVarsFixed);
        _vfv->BuildUDof(_rhs, Traits::nVarsFixed);
        _vfv->BuildUDof(_linearRhs, Traits::nVarsFixed);
        _vfv->BuildUDof(_linearIncrement, Traits::nVarsFixed);
        _u.setConstant(_configuration.InitialState());
        // Integrate rho exactly to quadrature accuracy; preserve a uniform primitive velocity.
        // Author: Runzhi Ma. The JSON profile is optional and does not alter old modules.
        for (index cell = 0; cell < _mesh->NumCell(); ++cell)
        {
            real densityIntegral = 0;
            const Vector3 wave = Eigen::Map<const Vector3>(_configuration.initialDensityWaveNumber.data());
            _vfv->GetCellQuad(cell).IntegrationSimple(densityIntegral,
                [&](real &value, int point, real)
                {
                    const auto x = _vfv->GetCellQuadraturePPhys(cell, point);
                    value = _configuration.initialState[0] *
                        (1 + _configuration.initialDensityAmplitude * std::cos(wave.dot(x))) *
                        _vfv->GetCellJacobiDet(cell, point);
                });
            const real density = densityIntegral / _vfv->GetCellVol(cell);
            _u[cell](0) = density;
            _u[cell].template segment<3>(1) = density * Velocity(_configuration.InitialState());
        }
        _rhs.setConstant(0.0);
        _linearRhs.setConstant(0.0);
        _linearIncrement.setConstant(0.0);

        DNDS_MAKE_SSP(
            _evaluator,
            _mesh,
            _vfv,
            _configuration.acmSettings,
            _configuration.reconstructionSettings,
            _boundaryHandler);

        if (TurbulenceVariableCount(_configuration.turbulenceSettings.model) > 0)
        {
            DNDS_MAKE_SSP(
                _turbulence,
                _mesh,
                _vfv,
                _configuration.acmSettings,
                _configuration.turbulenceSettings,
                _boundaryHandler);
            _turbulence->Initialize();
            _turbulence->SetMassFluxProvider([evaluator = _evaluator](index face, int point)
                                            { return evaluator->FaceMassFlux(face, point); });
            _evaluator->SetTurbulenceCoupling(
                [turbulence = _turbulence](TDof &flow, real time)
                {
                    turbulence->Prepare(flow, time);
                },
                [turbulence = _turbulence](index iFace, int iG)
                {
                    return turbulence->FaceEddyViscosity(iFace, iG);
                });
        }

        if (_mpi.rank == 0)
            log() << "ACM mesh/reconstruction initialized: dim=" << gDim
                  << ", global cells=" << _mesh->NumCellGlobal()
                  << ", reconstruction order=" << _configuration.vfvSettings.maxOrder
                  << ", turbulence model="
                  << TurbulenceModelName(_configuration.turbulenceSettings.model)
                  << ", turbulence equations="
                  << TurbulenceVariableCount(_configuration.turbulenceSettings.model)
                  << std::endl;
    }

    template <ACMModel model>
    void ACMSolver<model>::CopyOwnedToStateField(
        const TDof &source,
        StateField &destination) const
    {
        destination.resize(static_cast<std::size_t>(_mesh->NumCell()));
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
            destination[static_cast<std::size_t>(iCell)] = source[iCell];
    }

    template <ACMModel model>
    void ACMSolver<model>::CopyStateFieldToOwned(
        const StateField &source,
        TDof &destination) const
    {
        DNDS_check_throw_info(
            source.size() == static_cast<std::size_t>(_mesh->NumCell()),
            "ACM rank-local state field does not match the number of owned mesh cells");
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
            destination[iCell] = source[static_cast<std::size_t>(iCell)];
    }

    template <ACMModel model>
    void ACMSolver<model>::WriteFlowField(int iStep, real outputTime)
    {
        const auto &output = _configuration.outputSettings;
        DNDS_check_throw_info(output.interval > 0, "ACM flow-field output is disabled");
        DNDS_check_throw_info(!output.directory.empty(), "ACM output directory must not be empty");
        DNDS_check_throw_info(!output.prefix.empty(), "ACM output prefix must not be empty");

        std::ostringstream stamp;
        stamp << std::setw(8) << std::setfill('0') << iStep;
        const std::filesystem::path outputDirectory(output.directory);
        const std::string fileBase =
            (outputDirectory / (output.prefix + "_" + stamp.str())).string();
        const std::string seriesBase =
            (outputDirectory / output.prefix).string();

        MPI_Comm outputComm = MPI_COMM_NULL;
        MPI_Comm_dup(_mpi.comm, &outputComm);
        _mesh->PrintParallelVTKHDFDataArray(
            fileBase,
            seriesBase,
            2,
            1,
            0,
            0,
            [](int scalar) -> std::string
            { return scalar == 0 ? "Density" : "Pressure"; },
            [this](int scalar, index iCell) -> real
            { return _u[iCell](scalar == 0 ? 0 : 4); },
            [](int) -> std::string
            { return "Velocity"; },
            [this](int, index iCell, rowsize component) -> real
            { return _u[iCell](component + 1) / _u[iCell](0); },
            [](int) -> std::string
            { return {}; },
            [](int, index) -> real
            { return 0; },
            [](int) -> std::string
            { return {}; },
            [](int, index, rowsize) -> real
            { return 0; },
            static_cast<double>(outputTime),
            outputComm);
        MPI_Comm_free(&outputComm);

        if (_mpi.rank == 0)
            log() << "ACM flow field written at step " << iStep
                  << " to " << fileBase << ".vtkhdf" << std::endl;
    }

    template <ACMModel model>
    /** @copydoc ACMSolver::SolveImplicitCorrection */
    void ACMSolver<model>::SolveImplicitCorrection(
        const MatrixField &diagonal,
        const typename TEvaluator::FaceJacobianField &faceJacobians,
        bool useLUSGS)
    {
        const auto &settings = _configuration.timeMarchSettings;
        const auto report = ACM::SolveImplicitLinearSystem(
            _linearRhs, _linearIncrement, useLUSGS,
            settings.lusgsSweeps, settings.gmresSubspace, settings.gmresRestarts,
            settings.linearAbsoluteTolerance, settings.linearRelativeTolerance,
            std::sqrt(static_cast<real>(_mesh->NumCellGlobal()) * Traits::nVarsFixed),
            [&](TDof &input, TDof &output)
            { _evaluator->ApplyImplicitLinearization(input, diagonal, faceJacobians, output); },
            [&](TDof &input, TDof &output)
            {
                if (useLUSGS || settings.gmresPreconditioner == GMRESPreconditionerType::LUSGS)
                    _evaluator->SolveLUSGS(input, diagonal, faceJacobians, output, useLUSGS ? 1 : settings.lusgsSweeps);
                else
                    _evaluator->ApplyBlockJacobi(input, diagonal, output);
            },
            [this](TDof &field)
            {
                _vfv->BuildUDof(field, Traits::nVarsFixed);
                field.setConstant(0.0);
            });
        if (_mpi.rank == 0)
            log() << "ACMVariable linear solver=" << (useLUSGS ? "LUSGS" : "GMRES")
                  << " iterations=" << report.iterations << " trueResidual=" << report.finalResidual
                  << " initial=" << report.initialResidual << " target=" << report.tolerance
                  << " relativeResidual=" << (report.initialResidual > 0 ? report.finalResidual / report.initialResidual : 0)
                  << " converged=" << report.converged << std::endl;
    }

    template <ACMModel model>
    /** @copydoc ACMSolver::AdvanceImplicitDistributed */
    TimeStepReport ACMSolver<model>::AdvanceImplicitDistributed(
        StateField &states,
        const ScalarField &pseudoTimeStep,
        real time)
    {
        DNDS_check_throw_info(
            states.size() == pseudoTimeStep.size() &&
                states.size() == static_cast<std::size_t>(_mesh->NumCell()),
            "ACM distributed implicit fields have incompatible sizes");
        const StateField statesOld = states;
        MatrixField diagonal;
        typename TEvaluator::FaceJacobianField faceJacobians;
        TimeStepReport report;

        const auto evaluateDefect = [&]()
        {
            CopyStateFieldToOwned(states, _u);
            _evaluator->EvaluateRHS(_rhs, _u, time);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
            {
                const std::size_t ii = static_cast<std::size_t>(iCell);
                _linearRhs[iCell] = _rhs[iCell] -
                                    GammaLocal(
                                        _u[iCell],
                                        _configuration.acmSettings.beta2,
                                        _configuration.acmSettings.alpha) *
                                        (states[ii] - statesOld[ii]) /
                                        pseudoTimeStep[ii];
            }
            return _linearRhs.norm2() /
                   std::sqrt(std::max<real>(
                       1.0,
                       static_cast<real>(_mesh->NumCellGlobal()) * Traits::nVarsFixed));
        };

        report.initialDefectNorm = evaluateDefect();
        report.finalDefectNorm = report.initialDefectNorm;
        const real target = _configuration.timeMarchSettings.ImplicitTarget(report.initialDefectNorm);
        for (int iteration = 0;
             iteration < _configuration.timeMarchSettings.maxImplicitIterations;
             iteration++)
        {
            if (report.finalDefectNorm <= target)
            {
                report.converged = true;
                break;
            }

            _evaluator->AssembleImplicitLinearization(
                _u,
                pseudoTimeStep,
                diagonal,
                faceJacobians,
                time);
            // Exact product rule for backward Euler in pseudo-time. In dual-time
            // defect correction there is no finite pseudo-time history term.
            // Author: Runzhi Ma. Preserve the independent physical mass matrix.
            for (index cell = 0; cell < _mesh->NumCell(); ++cell)
                diagonal[cell] += PseudoTimeProductJacobian(
                    states[cell], statesOld[cell], pseudoTimeStep[cell], _configuration.acmSettings) -
                    GammaLocal(states[cell], _configuration.acmSettings.beta2,
                               _configuration.acmSettings.alpha) / pseudoTimeStep[cell];
            SolveImplicitCorrection(
                diagonal,
                faceJacobians,
                _configuration.timeMarchSettings.integrator ==
                    TimeIntegratorType::ImplicitEulerLUSGS);

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
            {
                State &state = states[static_cast<std::size_t>(iCell)];
                const State increment = _linearIncrement[iCell];
                real relaxation = _configuration.timeMarchSettings.implicitRelaxation;
                if (increment(0) < 0)
                    relaxation = std::min(relaxation,
                        0.9 * (state(0) - _configuration.acmSettings.densityFloor) / (-increment(0)));
                state += std::max(real(0), relaxation) * increment;
                DNDS_check_throw_info(IsFiniteState(state), "Invalid ACMVariable implicit state");
            }

            report.iterations = iteration + 1;
            report.finalDefectNorm = evaluateDefect();
        }
        report.converged =
            report.finalDefectNorm <= target;
        return report;
    }

    template <ACMModel model>
    /** @copydoc ACMSolver::AdvanceBDF2DualTimeDistributed */
    TimeStepReport ACMSolver<model>::AdvanceBDF2DualTimeDistributed(
        StateField &states,
        const BDF2History &history,
        const ScalarField &pseudoTimeStep,
        real physicalTime)
    {
        DNDS_check_throw_info(
            IsBDF2DualTimeIntegrator(_configuration.timeMarchSettings.integrator),
            "ACM BDF2 advance requires a BDF2 dual-time integrator");
        DNDS_check_throw_info(
            states.size() == pseudoTimeStep.size() &&
                states.size() == history.Previous().size() &&
                states.size() == history.PreviousPrevious().size() &&
                states.size() == static_cast<std::size_t>(_mesh->NumCell()),
            "ACM BDF2 distributed fields have incompatible sizes");
        DNDS_check_throw_info(
            std::isfinite(physicalTime),
            "ACM BDF2 physical time must be finite");

        const BDF2Coefficients coefficients = history.Coefficients();
        const StateField &previous = history.Previous();
        const StateField &previousPrevious = history.PreviousPrevious();
        const real physicalTimeStep = _configuration.timeMarchSettings.physicalTimeStep;
        MatrixField diagonal;
        typename TEvaluator::FaceJacobianField faceJacobians;
        TimeStepReport report;

        const auto evaluateDefect = [&]()
        {
            CopyStateFieldToOwned(states, _u);
            _evaluator->EvaluateRHS(_rhs, _u, physicalTime);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
            {
                const std::size_t ii = static_cast<std::size_t>(iCell);
                _linearRhs[iCell] = _rhs[iCell] - EvaluateBDF2PhysicalDerivative(
                                                      states[ii],
                                                      previous[ii],
                                                      previousPrevious[ii],
                                                      coefficients,
                                                      physicalTimeStep);
            }
            const real flowNorm = _linearRhs.norm2() /
                   std::sqrt(std::max<real>(1.0, real(_mesh->NumCellGlobal()) * Traits::nVarsFixed));
            const real turbulenceNorm = _turbulence ?
                _turbulence->PhysicalDefect(_u, coefficients, physicalTimeStep, nullptr, physicalTime) : 0;
            return std::max(flowNorm, turbulenceNorm);
        };

        report.initialDefectNorm = evaluateDefect();
        report.finalDefectNorm = report.initialDefectNorm;
        const real target = _configuration.timeMarchSettings.ImplicitTarget(report.initialDefectNorm);
        for (int iteration = 0;
             iteration < _configuration.timeMarchSettings.maxImplicitIterations;
             iteration++)
        {
            if (report.finalDefectNorm <= target)
            {
                report.converged = true;
                break;
            }

            _evaluator->AssembleImplicitLinearization(
                _u,
                pseudoTimeStep,
                diagonal,
                faceJacobians,
                physicalTime);
            AddBDF2PhysicalDiagonal(diagonal, coefficients, physicalTimeStep);
            if (_turbulence)
                _turbulence->PhysicalDefect(_u, coefficients, physicalTimeStep, &pseudoTimeStep, physicalTime);
            SolveImplicitCorrection(
                diagonal,
                faceJacobians,
                BDF2UsesLUSGS(_configuration.timeMarchSettings.integrator));

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
            {
                State &state = states[static_cast<std::size_t>(iCell)];
                const State increment = _linearIncrement[iCell];
                real relaxation = _configuration.timeMarchSettings.implicitRelaxation;
                if (increment(0) < 0)
                    relaxation = std::min(relaxation,
                        0.9 * (state(0) - _configuration.acmSettings.densityFloor) / (-increment(0)));
                state += std::max(real(0), relaxation) * increment;
                DNDS_check_throw_info(IsFiniteState(state), "Invalid ACMVariable implicit state");
            }

            report.iterations = iteration + 1;
            report.finalDefectNorm = evaluateDefect();
        }
        report.converged =
            report.finalDefectNorm <= target;
        return report;
    }

    template <ACMModel model>
    void ACMSolver<model>::Run()
    {
        DNDS_check_throw_info(_evaluator != nullptr, "ACM Run requires ReadMeshAndInitialize first");
        if (_configuration.timeMarchSettings.physicalODECode >= 0 ||
            _configuration.timeMarchSettings.pseudoODECode >= 0)
        {
            RunGenericPhysical();
            return;
        }

        StateField states;
        CopyOwnedToStateField(_u, states);
        ScalarField pseudoTimeStep(
            states.size(),
            _configuration.timeMarchSettings.pseudoTimeStep);
        const TimeIntegratorType integrator = _configuration.timeMarchSettings.integrator;
        const bool useBDF2 = IsBDF2DualTimeIntegrator(integrator);
        BDF2History bdf2History;
        if (useBDF2)
        {
            bdf2History.Initialize(states);
            if (_turbulence) _turbulence->CommitPhysicalHistory(_u, true);
        }
        real physicalTime = 0;
        real residualTime = 0;

        const ResidualEvaluator residualEvaluator =
            [this, &residualTime](const StateField &input, StateField &output)
        {
            CopyStateFieldToOwned(input, _u);
            _evaluator->EvaluateRHS(_rhs, _u, residualTime);
            CopyOwnedToStateField(_rhs, output);
        };
        const DiagonalJacobianEvaluator diagonalJacobianEvaluator =
            [this, &residualTime](const StateField &input, MatrixField &output)
        {
            CopyStateFieldToOwned(input, _u);
            _evaluator->EvaluateDiagonalJacobian(_u, output, residualTime);
        };

        if (_configuration.outputSettings.interval > 0 &&
            _configuration.outputSettings.writeInitial)
            WriteFlowField(0, 0);

        for (int iStep = 1; iStep <= _configuration.timeMarchSettings.nSteps; iStep++)
        {
            const BDF2Coefficients bdf2Coefficients =
                useBDF2 ? bdf2History.Coefficients() : BDF2Coefficients{};
            residualTime = useBDF2
                               ? physicalTime + _configuration.timeMarchSettings.physicalTimeStep
                               : 0;
            real minimumTimeStep = _configuration.timeMarchSettings.pseudoTimeStep;
            if (_configuration.timeMarchSettings.useCFLTimeStep)
            {
                CopyStateFieldToOwned(states, _u);
                minimumTimeStep = _evaluator->EvaluateTimeStep(
                    pseudoTimeStep,
                    _u,
                    _configuration.timeMarchSettings.cfl,
                    _configuration.timeMarchSettings.maximumPseudoTimeStep,
                    _configuration.timeMarchSettings.useLocalTimeStep,
                    residualTime);
            }
            const bool useDistributedImplicit =
                integrator == TimeIntegratorType::ImplicitEulerLUSGS ||
                integrator == TimeIntegratorType::ImplicitEulerGMRES;
            TimeStepReport report;
            if (useBDF2)
            {
                report = AdvanceBDF2DualTimeDistributed(
                    states,
                    bdf2History,
                    pseudoTimeStep,
                    residualTime);
                DNDS_check_throw_info(report.converged,
                    "ACMVariable physical step rejected: increase inner iterations or reduce physical dt");
                bdf2History.Commit(states);
                if (_turbulence) _turbulence->CommitPhysicalHistory(_u, false);
                physicalTime = residualTime;
            }
            else if (useDistributedImplicit)
                report = AdvanceImplicitDistributed(states, pseudoTimeStep, residualTime);
            else
                report = AdvancePseudoTimeStep(
                    states,
                    pseudoTimeStep,
                    _configuration.acmSettings,
                    _configuration.timeMarchSettings,
                    residualEvaluator,
                    diagonalJacobianEvaluator,
                    &_mpi);
            real turbulenceResidual = 0;
            if (_turbulence && !useBDF2)
            {
                CopyStateFieldToOwned(states, _u);
                turbulenceResidual = _turbulence->Advance(
                    _u,
                    pseudoTimeStep,
                    residualTime);
            }
            if (_mpi.rank == 0)
            {
                log() << std::scientific;
                if (useBDF2)
                    log() << "ACM physical step " << std::setw(8) << iStep
                          << " time=" << physicalTime
                          << " BDF" << bdf2Coefficients.order;
                else
                    log() << "ACM step " << std::setw(8) << iStep;
                log() << " residual " << report.initialDefectNorm
                      << " -> " << report.finalDefectNorm
                      << " dtauMin=" << minimumTimeStep
                      << " turbResidual=" << turbulenceResidual
                      << " inner=" << report.iterations
                      << " converged=" << report.converged
                      << std::endl;
            }
            if (_configuration.outputSettings.interval > 0 &&
                iStep % _configuration.outputSettings.interval == 0)
            {
                CopyStateFieldToOwned(states, _u);
                WriteFlowField(iStep, useBDF2 ? physicalTime : static_cast<real>(iStep));
            }
        }
        CopyStateFieldToOwned(states, _u);
    }
}
