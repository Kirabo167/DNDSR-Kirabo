/**
 * @file ACMSolver.hxx
 * @brief Template implementation of the standalone ACM solver assembly.
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMSolver.hpp"

#include "DNDS/Errors.hpp"
#include "DNDS/OMP.hpp"

#include <hdf5.h>

#include <filesystem>
#include <iomanip>
#include <sstream>

namespace DNDS::ACM
{
    namespace
    {
        template <class TValue>
        std::vector<TValue> ReadVTKHDFSlice(
            hid_t file,
            const char *path,
            hsize_t offset,
            hsize_t count,
            hsize_t width,
            hid_t memoryType)
        {
            const hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
            DNDS_check_throw_info(dataset >= 0, std::string("ACM restart dataset is missing: ") + path);
            const hid_t fileSpace = H5Dget_space(dataset);
            DNDS_check_throw_info(fileSpace >= 0, std::string("Cannot inspect ACM restart dataset: ") + path);
            const int expectedDimensions = width == 1 ? 1 : 2;
            const int dimensions = H5Sget_simple_extent_ndims(fileSpace);
            std::array<hsize_t, 2> shape{0, 1};
            DNDS_check_throw_info(
                dimensions == expectedDimensions && H5Sget_simple_extent_dims(fileSpace, shape.data(), nullptr) >= 0 &&
                    shape[0] >= offset + count && (width == 1 || shape[1] == width),
                std::string("ACM restart dataset has an incompatible shape: ") + path);
            std::array<hsize_t, 2> start{offset, 0};
            std::array<hsize_t, 2> selection{count, width};
            DNDS_check_throw_info(
                H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start.data(), nullptr, selection.data(), nullptr) >= 0,
                std::string("Cannot select ACM restart dataset slice: ") + path);
            const hid_t memorySpace = H5Screate_simple(expectedDimensions, selection.data(), nullptr);
            DNDS_check_throw_info(memorySpace >= 0, "Cannot create ACM restart memory selection");
            std::vector<TValue> values(static_cast<std::size_t>(count * width));
            const herr_t readStatus = H5Dread(
                dataset, memoryType, memorySpace, fileSpace, H5P_DEFAULT, values.data());
            H5Sclose(memorySpace);
            H5Sclose(fileSpace);
            H5Dclose(dataset);
            DNDS_check_throw_info(readStatus >= 0, std::string("Cannot read ACM restart dataset: ") + path);
            return values;
        }
    }

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
        if (_configuration.outputSettings.interval > 0 || !_configuration.restartSettings.flowFile.empty())
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

        ReadInitialFlowField();

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
    void ACMSolver<model>::ReadInitialFlowField()
    {
        const auto &restart = _configuration.restartSettings;
        if (restart.flowFile.empty())
            return;
        DNDS_check_throw_info(
            restart.writerRanks == _mpi.size,
            fmt::format("ACM VTK-HDF restart requires the original MPI size: file ranks={}, current ranks={}",
                        restart.writerRanks, _mpi.size));

        const hid_t access = H5Pcreate(H5P_FILE_ACCESS);
        DNDS_check_throw_info(access >= 0, "Cannot create ACM restart HDF5 access properties");
        DNDS_check_throw_info(
            H5Pset_fapl_mpio(access, _mpi.comm, MPI_INFO_NULL) >= 0,
            "Cannot enable parallel HDF5 for ACM restart");
        const hid_t file = H5Fopen(restart.flowFile.c_str(), H5F_ACC_RDONLY, access);
        H5Pclose(access);
        DNDS_check_throw_info(file >= 0, "Cannot open ACM restart flow file: " + restart.flowFile);

        const hsize_t cellOffset = static_cast<hsize_t>(_mesh->vtkCellOffset);
        const hsize_t localCells = static_cast<hsize_t>(_mesh->NumCell());
        const auto numberOfCells = ReadVTKHDFSlice<long long>(
            file, "/VTKHDF/NumberOfCells", 0, 1, 1, H5T_NATIVE_LLONG);
        DNDS_check_throw_info(
            numberOfCells[0] == _mesh->NumCellGlobal(),
            fmt::format("ACM restart cell count mismatch: file={}, mesh={}",
                        numberOfCells[0], _mesh->NumCellGlobal()));

        // These arrays prove that this rank owns the same cells in the same
        // order used by the original VTK-HDF writer.
        const auto cellTypes = ReadVTKHDFSlice<uint8_t>(
            file, "/VTKHDF/Types", cellOffset, localCells, 1, H5T_NATIVE_UINT8);
        DNDS_check_throw_info(cellTypes == _mesh->vtkCellType,
                              "ACM restart cell types/order differ from the current partition");
        const auto offsets = ReadVTKHDFSlice<long long>(
            file, "/VTKHDF/Offsets", cellOffset, localCells, 1, H5T_NATIVE_LLONG);
        for (std::size_t i = 0; i < offsets.size(); i++)
            DNDS_check_throw_info(
                offsets[i] == _mesh->vtkCell2nodeOffsets[i],
                "ACM restart cell offsets/order differ from the current partition");
        const hsize_t connectivityOffset =
            static_cast<hsize_t>(_mesh->vtkCell2nodeOffsets.front());
        const auto connectivity = ReadVTKHDFSlice<long long>(
            file, "/VTKHDF/Connectivity", connectivityOffset,
            static_cast<hsize_t>(_mesh->vtkCell2node.size()), 1, H5T_NATIVE_LLONG);
        for (std::size_t i = 0; i < connectivity.size(); i++)
            DNDS_check_throw_info(
                connectivity[i] == _mesh->vtkCell2node[i],
                "ACM restart connectivity/order differ from the current partition");

        const auto velocity = ReadVTKHDFSlice<real>(
            file, "/VTKHDF/CellData/Velocity", cellOffset, localCells, 3, H5T_NATIVE_DOUBLE);
        const auto pressure = ReadVTKHDFSlice<real>(
            file, "/VTKHDF/CellData/Pressure", cellOffset, localCells, 1, H5T_NATIVE_DOUBLE);
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            for (int component = 0; component < 3; component++)
                _u[iCell](component) = velocity[static_cast<std::size_t>(iCell) * 3 + component];
            _u[iCell](3) = pressure[static_cast<std::size_t>(iCell)];
            DNDS_check_throw_info(_u[iCell].allFinite(), "ACM restart flow field contains a non-finite value");
        }
        _u.trans.startPersistentPull();
        _u.trans.waitPersistentPull();

        bool turbulenceLoaded = false;
        if (_turbulence)
        {
            const bool hasK = H5Lexists(file, "/VTKHDF/CellData/TurbulenceK", H5P_DEFAULT) > 0;
            const bool hasOmega = H5Lexists(file, "/VTKHDF/CellData/TurbulenceOmega", H5P_DEFAULT) > 0;
            DNDS_check_throw_info(
                !restart.requireTurbulence || (hasK && hasOmega),
                "ACM restart requires turbulence datasets, but the file contains only the flow field");
            if (hasK && hasOmega)
            {
                const auto k = ReadVTKHDFSlice<real>(
                    file, "/VTKHDF/CellData/TurbulenceK", cellOffset, localCells, 1, H5T_NATIVE_DOUBLE);
                const auto omega = ReadVTKHDFSlice<real>(
                    file, "/VTKHDF/CellData/TurbulenceOmega", cellOffset, localCells, 1, H5T_NATIVE_DOUBLE);
                auto &turbulence = _turbulence->GetState();
                for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
                {
                    turbulence[iCell] << k[static_cast<std::size_t>(iCell)],
                        omega[static_cast<std::size_t>(iCell)];
                    DNDS_check_throw_info(
                        turbulence[iCell].allFinite() && (turbulence[iCell].array() > 0).all(),
                        "ACM restart turbulence field is non-finite or non-positive");
                }
                turbulence.trans.startPersistentPull();
                turbulence.trans.waitPersistentPull();
                turbulenceLoaded = true;
            }
        }
        H5Fclose(file);
        if (_mpi.rank == 0)
        {
            log() << "ACM restart loaded flow field from " << restart.flowFile
                  << " at completed step " << restart.completedSteps << std::endl;
            if (_turbulence && !turbulenceLoaded)
                log() << "ACM restart warning: turbulence datasets are absent; using configured initial k/omega"
                      << std::endl;
        }
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
        const int turbulenceVariables = _turbulence ? _turbulence->ActiveVariableCount() : 0;
        _mesh->PrintParallelVTKHDFDataArray(
            fileBase,
            seriesBase,
            1 + turbulenceVariables,
            1,
            0,
            0,
            [](int iArray) -> std::string
            {
                if (iArray == 0)
                    return "Pressure";
                return iArray == 1 ? "TurbulenceK" : "TurbulenceOmega";
            },
            [this](int iArray, index iCell) -> real
            {
                if (iArray == 0)
                    return _u[iCell](3);
                return _turbulence->GetState()[iCell](iArray - 1);
            },
            [](int) -> std::string
            { return "Velocity"; },
            [this](int, index iCell, rowsize component) -> real
            { return _u[iCell](component); },
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
        _linearIncrement.setConstant(0.0);
        if (useLUSGS)
        {
            _evaluator->SolveLUSGS(
                _linearRhs,
                diagonal,
                faceJacobians,
                _linearIncrement,
                _configuration.timeMarchSettings.lusgsSweeps);
            return;
        }

        Linear::GMRES_LeftPreconditioned<TDof> gmres(
            static_cast<uint32_t>(_configuration.timeMarchSettings.gmresSubspace),
            [this](TDof &field)
            {
                _vfv->BuildUDof(field, Traits::nVarsFixed);
                field.setConstant(0.0);
            });
        gmres.solve(
            [&](TDof &input, TDof &output)
            {
                _evaluator->ApplyImplicitLinearization(
                    input,
                    diagonal,
                    faceJacobians,
                    output);
            },
            [&](TDof &input, TDof &output)
            {
                if (_configuration.timeMarchSettings.gmresPreconditioner ==
                    GMRESPreconditionerType::LUSGS)
                    _evaluator->SolveLUSGS(
                        input,
                        diagonal,
                        faceJacobians,
                        output,
                        _configuration.timeMarchSettings.lusgsSweeps);
                else
                    _evaluator->ApplyBlockJacobi(input, diagonal, output);
            },
            [](TDof &left, TDof &right)
            { return left.dot(right); },
            _linearRhs,
            _linearIncrement,
            static_cast<uint32_t>(_configuration.timeMarchSettings.gmresRestarts),
            [&](uint32_t restart, real residual, real initialResidual)
            {
                if (_mpi.rank == 0 && restart > 0)
                    log() << "ACM GMRES restart=" << restart
                          << " residual=" << residual
                          << " initial=" << initialResidual << std::endl;
                return residual <=
                       _configuration.timeMarchSettings.gmresRelativeTolerance *
                           std::max(initialResidual, verySmallReal);
            });
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
        report.defectTolerance = _configuration.timeMarchSettings.SteadyImplicitTarget(report.initialDefectNorm);
        for (int iteration = 0;
             iteration < _configuration.timeMarchSettings.maxImplicitIterations;
             iteration++)
        {
            if (report.finalDefectNorm <= report.defectTolerance)
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
            // AssembleImplicitLinearization already contains Gamma(U)/dTau. Add only
            // the product-rule contribution from d[Gamma(U)(U-Uold)]/dU here.
            // BDF2 deliberately does not use this finite pseudo-time history term.
            for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
            {
                const std::size_t ii = static_cast<std::size_t>(iCell);
                diagonal[ii] += PseudoTimeProductJacobian(
                                    states[ii],
                                    statesOld[ii],
                                    pseudoTimeStep[ii],
                                    _configuration.acmSettings) -
                                GammaLocal(
                                    states[ii],
                                    _configuration.acmSettings.beta2,
                                    _configuration.acmSettings.alpha) /
                                    pseudoTimeStep[ii];
            }
            SolveImplicitCorrection(
                diagonal,
                faceJacobians,
                _configuration.timeMarchSettings.integrator ==
                    TimeIntegratorType::ImplicitEulerLUSGS);

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
                states[static_cast<std::size_t>(iCell)] +=
                    _configuration.timeMarchSettings.implicitRelaxation *
                    State(_linearIncrement[iCell]);

            report.iterations = iteration + 1;
            report.finalDefectNorm = evaluateDefect();
        }
        report.converged =
            report.finalDefectNorm <= report.defectTolerance;
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
            return _linearRhs.norm2() /
                   std::sqrt(std::max<real>(
                       1.0,
                       static_cast<real>(_mesh->NumCellGlobal()) * Traits::nVarsFixed));
        };

        report.initialDefectNorm = evaluateDefect();
        report.finalDefectNorm = report.initialDefectNorm;
        for (int iteration = 0;
             iteration < _configuration.timeMarchSettings.maxImplicitIterations;
             iteration++)
        {
            if (report.finalDefectNorm <= _configuration.timeMarchSettings.implicitTolerance)
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
            SolveImplicitCorrection(
                diagonal,
                faceJacobians,
                BDF2UsesLUSGS(_configuration.timeMarchSettings.integrator));

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
                states[static_cast<std::size_t>(iCell)] +=
                    _configuration.timeMarchSettings.implicitRelaxation *
                    State(_linearIncrement[iCell]);

            report.iterations = iteration + 1;
            report.finalDefectNorm = evaluateDefect();
        }
        report.converged =
            report.finalDefectNorm <= _configuration.timeMarchSettings.implicitTolerance;
        return report;
    }

    template <ACMModel model>
    void ACMSolver<model>::Run()
    {
        DNDS_check_throw_info(_evaluator != nullptr, "ACM Run requires ReadMeshAndInitialize first");

        StateField states;
        CopyOwnedToStateField(_u, states);
        ScalarField pseudoTimeStep(
            states.size(),
            _configuration.timeMarchSettings.pseudoTimeStep);
        const TimeIntegratorType integrator = _configuration.timeMarchSettings.integrator;
        const bool useBDF2 = IsBDF2DualTimeIntegrator(integrator);
        DNDS_check_throw_info(
            !useBDF2 || _turbulence == nullptr,
            "ACM BDF2 dual-time marching currently supports Laminar flow only; "
            "segregated turbulence equations do not yet have physical-time history");

        BDF2History bdf2History;
        if (useBDF2)
            bdf2History.Initialize(states);
        real physicalTime = 0;
        real currentCFL = _configuration.timeMarchSettings.cfl;
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
            WriteFlowField(
                _configuration.restartSettings.completedSteps,
                static_cast<real>(_configuration.restartSettings.completedSteps));

        for (int iStep = 1; iStep <= _configuration.timeMarchSettings.nSteps; iStep++)
        {
            const int absoluteStep = _configuration.restartSettings.completedSteps + iStep;
            const auto reconstructionSweepsBefore = _evaluator->GetTotalReconstructionSweeps();
            const real usedCFL = currentCFL;
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
                    currentCFL,
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
                // Match the reference fixed-pseudo-iteration policy: reaching the
                // inner limit completes the physical step, while report.converged
                // records whether the requested defect tolerance was also reached.
                bdf2History.Commit(states);
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
            if (_turbulence)
            {
                CopyStateFieldToOwned(states, _u);
                turbulenceResidual = _turbulence->Advance(
                    _u,
                    pseudoTimeStep,
                    residualTime);
            }
            const bool steadyControls = _configuration.timeMarchSettings.steadyAdaptiveCFL ||
                                        _configuration.timeMarchSettings.steadyRelativeTolerance > 0;
            real steadyResidual = 0;
            if (steadyControls)
            {
                // This is R(U), after the segregated turbulence update, not the
                // finite pseudo-time history defect used by the inner solve.
                CopyStateFieldToOwned(states, _u);
                _evaluator->EvaluateRHS(_rhs, _u, residualTime);
                steadyResidual = _rhs.norm2() /
                                 std::sqrt(std::max<real>(1, static_cast<real>(_mesh->NumCellGlobal()) * Traits::nVarsFixed));
                currentCFL = _configuration.timeMarchSettings.NextSteadyCFL(
                    currentCFL, report.initialDefectNorm, steadyResidual, report.converged);
            }
            if (_mpi.rank == 0)
            {
                log() << std::scientific;
                if (useBDF2)
                    log() << "ACM physical step " << std::setw(8) << absoluteStep
                          << " time=" << physicalTime
                          << " BDF" << bdf2Coefficients.order;
                else
                    log() << "ACM step " << std::setw(8) << absoluteStep;
                log() << " residual " << report.initialDefectNorm
                      << " -> " << report.finalDefectNorm
                      << " dtauMin=" << minimumTimeStep
                      << " turbResidual=" << turbulenceResidual
                      << " inner=" << report.iterations
                      << " converged=" << report.converged;
                if (steadyControls)
                    log() << " innerTarget=" << report.defectTolerance
                          << " steadyResidual=" << steadyResidual
                          << " cfl=" << usedCFL << " nextCFL=" << currentCFL;
                if (_configuration.reconstructionSettings.variationalTolerance > 0)
                    log() << " recSweeps=" << _evaluator->GetTotalReconstructionSweeps() - reconstructionSweepsBefore
                          << " recDefect=" << _evaluator->GetReconstructionReport().equationDefect
                          << " recConverged=" << _evaluator->GetReconstructionReport().converged;
                log() << std::endl;
            }
            if (_configuration.outputSettings.interval > 0 &&
                absoluteStep % _configuration.outputSettings.interval == 0)
            {
                CopyStateFieldToOwned(states, _u);
                WriteFlowField(
                    absoluteStep,
                    useBDF2 ? physicalTime : static_cast<real>(absoluteStep));
            }
        }
        CopyStateFieldToOwned(states, _u);
    }
}
