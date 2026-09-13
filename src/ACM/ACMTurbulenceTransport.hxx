/**
 * @file ACMTurbulenceTransport.hxx
 * @brief Template implementation of segregated ACM turbulence transport.
 * @author Runzhi Ma
 * @date 2026-09-02
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMTurbulenceTransport.hpp"

#include "DNDS/Errors.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace DNDS::ACM
{
    template <int gDim>
    ACMTurbulenceTransport<gDim>::ACMTurbulenceTransport(
        ssp<Geom::UnstructuredMesh> mesh,
        TpVFV vfv,
        const Settings &flowSettings,
        const TurbulenceSettings &turbulenceSettings,
        ssp<BoundaryHandler> boundaryHandler)
        : _mesh(std::move(mesh)),
          _vfv(std::move(vfv)),
          _flowSettings(flowSettings),
          _turbulenceSettings(turbulenceSettings),
          _boundaryHandler(std::move(boundaryHandler))
    {
        DNDS_check_throw_info(_mesh != nullptr && _vfv != nullptr,
                              "ACM turbulence transport requires mesh and CFV objects");
        DNDS_check_throw_info(_boundaryHandler != nullptr,
                              "ACM turbulence transport requires a boundary handler");
        DNDS_check_throw_info(_mesh->getDim() == gDim,
                              "ACM turbulence transport dimension does not match mesh");
        _flowSettings.Validate();
        _turbulenceSettings.Validate();
        DNDS_check_throw_info(ActiveVariableCount() > 0,
                              "Do not construct ACM turbulence transport for Laminar mode");
        DNDS_check_throw_info(
            _flowSettings.enableViscousFlux && _flowSettings.dynamicViscosity > 0,
            "ACM RANS transport requires enabled positive molecular viscosity");

        _vfv->BuildUDof(_state, nStoredVariables);
        _vfv->BuildUDof(_rhs, nStoredVariables);
        _vfv->BuildUGrad(_gradient, nStoredVariables);
        _vfv->BuildUGrad(_flowGradient, nFlowVariables);
        _vfv->BuildUDof(_limiter, nStoredVariables);
        _vfv->BuildUDof(_flowLimiter, 1);
        _state.setConstant(0.0);
        _rhs.setConstant(0.0);
        _gradient.setConstant(0.0);
        _flowGradient.setConstant(0.0);
        _limiter.setConstant(1.0);
        _flowLimiter.setConstant(1.0);
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::Initialize */
    void ACMTurbulenceTransport<gDim>::Initialize()
    {
        BuildWallDistance();
        const TurbulenceState initial = _turbulenceSettings.InitialState();
        _state.setConstant(initial);
        _state.trans.startPersistentPull();
        _state.trans.waitPersistentPull();
        _rhs.setConstant(0.0);
        _gradient.setConstant(0.0);
        _flowGradient.setConstant(0.0);
        _limiter.setConstant(1.0);
        _flowLimiter.setConstant(1.0);
        _initialized = true;
        _prepared = false;
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::BuildWallDistance */
    void ACMTurbulenceTransport<gDim>::BuildWallDistance()
    {
        Geom::UnstructuredMesh::WallDistOptions options;
        options.subdivide_quad = _turbulenceSettings.wallDistanceSubdivide;
        options.method = _turbulenceSettings.wallDistanceMethod;
        options.wallDistExecution = _turbulenceSettings.wallDistanceExecution;
        options.minWallDist = _turbulenceSettings.minimumWallDistance;
        options.verbose = _turbulenceSettings.wallDistanceVerbose;
        _mesh->BuildNodeWallDist(
            [boundaryHandler = _boundaryHandler](Geom::t_index zone)
            {
                const BoundaryType type = boundaryHandler->GetTypeFromID(zone);
                return type == BoundaryType::BCWall ||
                       type == BoundaryType::BCWallIsothermal;
            },
            options);

        _cellWallDistance.assign(
            static_cast<std::size_t>(_mesh->NumCellProc()),
            _turbulenceSettings.minimumWallDistance);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCellProc(); iCell++)
        {
            Vector3 interpolated = Vector3::Zero();
            const auto nodes = _mesh->cell2node[iCell];
            for (rowsize iNode = 0; iNode < nodes.size(); iNode++)
                interpolated += ToVector3(_mesh->GetCoordWallDistOnCell(iCell, iNode));
            if (nodes.size() > 0)
                interpolated /= static_cast<real>(nodes.size());
            _cellWallDistance[static_cast<std::size_t>(iCell)] = std::max(
                interpolated.norm(),
                _turbulenceSettings.minimumWallDistance);
        }

        _faceWallDistance.assign(
            static_cast<std::size_t>(_mesh->NumFaceProc()),
            _turbulenceSettings.minimumWallDistance);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < _mesh->NumFaceProc(); iFace++)
        {
            const auto faceToCell = _mesh->face2cell[iFace];
            real distance = _cellWallDistance[static_cast<std::size_t>(faceToCell[0])];
            if (faceToCell[1] != UnInitIndex)
                distance = 0.5 *
                           (distance +
                            _cellWallDistance[static_cast<std::size_t>(faceToCell[1])]);
            _faceWallDistance[static_cast<std::size_t>(iFace)] = std::max(
                distance,
                _turbulenceSettings.minimumWallDistance);
        }
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::ToVector3 */
    Vector3 ACMTurbulenceTransport<gDim>::ToVector3(const Geom::tPoint &value)
    {
        Vector3 result = Vector3::Zero();
        result.template head<gDim>() = value.template head<gDim>();
        return result;
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::GenerateBoundary */
    TurbulenceState ACMTurbulenceTransport<gDim>::GenerateBoundary(
        index iFace,
        const TurbulenceState &interior,
        index ownerCell) const
    {
        return GenerateTurbulenceBoundaryState(
            _boundaryHandler->GetTypeFromID(_mesh->GetFaceZone(iFace)),
            interior,
            _cellWallDistance.at(static_cast<std::size_t>(ownerCell)),
            _flowSettings.rho0,
            _flowSettings.dynamicViscosity,
            _turbulenceSettings);
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::EvaluateGradients */
    void ACMTurbulenceTransport<gDim>::EvaluateGradients(TFlowDof &flow, real time)
    {
        flow.trans.startPersistentPull();
        flow.trans.waitPersistentPull();
        _state.trans.startPersistentPull();
        _state.trans.waitPersistentPull();

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            Eigen::Matrix<real, nFlowVariables, gDim> flowGradient =
                Eigen::Matrix<real, nFlowVariables, gDim>::Zero();
            Eigen::Matrix<real, nStoredVariables, gDim> turbulenceGradient =
                Eigen::Matrix<real, nStoredVariables, gDim>::Zero();
            const auto cellFaces = _mesh->cell2face[iCell];
            for (rowsize iCellFace = 0; iCellFace < cellFaces.size(); iCellFace++)
            {
                const index iFace = cellFaces[iCellFace];
                const index otherCell = _mesh->CellFaceOther(iCell, iFace, iCellFace);
                const rowsize if2c = _mesh->CellIsFaceBack(iCell, iFace, iCellFace) ? 0 : 1;
                const Geom::tPoint faceCenter =
                    _vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, -1);
                real thisLength =
                    (faceCenter - _vfv->GetCellQuadraturePPhys(iCell, -1)).norm();
                real otherLength = thisLength;
                State otherFlow;
                TurbulenceState otherTurbulence;
                Vector3 outward = ToVector3(
                    _vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1));
                outward *= if2c == 0 ? 1.0 : -1.0;

                if (otherCell != UnInitIndex)
                {
                    otherLength =
                        (_vfv->GetOtherCellPointFromCell(
                             iCell,
                             otherCell,
                             iFace,
                             if2c,
                             _vfv->GetCellQuadraturePPhys(otherCell, -1)) -
                         faceCenter)
                            .norm();
                    otherFlow = flow[otherCell];
                    Eigen::RowVector<real, nFlowVariables> flowRow =
                        otherFlow.transpose();
                    _vfv->ApplyPeriodicTransform(
                        if2c,
                        _mesh->GetFaceZone(iFace),
                        flowRow);
                    otherFlow = flowRow.transpose();
                    otherTurbulence = _state[otherCell];
                }
                else
                {
                    const Vector3 point = ToVector3(faceCenter);
                    otherFlow = GenerateBoundaryState(
                        _boundaryHandler->GetConditionFromID(_mesh->GetFaceZone(iFace)),
                        State(flow[iCell]),
                        outward,
                        _flowSettings,
                        point,
                        time);
                    otherTurbulence = GenerateBoundary(
                        iFace,
                        TurbulenceState(_state[iCell]),
                        iCell);
                    if (_boundaryHandler->GetTypeFromID(
                            _mesh->GetFaceZone(iFace)) == BoundaryType::BCFar &&
                        State(flow[iCell]).template head<3>().dot(outward) >= 0)
                        otherTurbulence = _state[iCell];
                }

                const real interpolation = otherCell == UnInitIndex
                                               ? 0.5
                                               : thisLength /
                                                     (thisLength + otherLength + verySmallReal);
                const real area = _vfv->GetFaceArea(iFace);
                flowGradient +=
                    (otherFlow - State(flow[iCell])) * interpolation *
                    area * outward.template head<gDim>().transpose();
                turbulenceGradient +=
                    (otherTurbulence - TurbulenceState(_state[iCell])) * interpolation *
                    area * outward.template head<gDim>().transpose();
            }
            const real inverseVolume = 1.0 / _vfv->GetCellVol(iCell);
            _flowGradient[iCell] = (flowGradient * inverseVolume).transpose();
            _gradient[iCell] = (turbulenceGradient * inverseVolume).transpose();
        }
        _flowGradient.trans.startPersistentPull();
        _flowGradient.trans.waitPersistentPull();
        _gradient.trans.startPersistentPull();
        _gradient.trans.waitPersistentPull();
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::EvaluateLimiter */
    void ACMTurbulenceTransport<gDim>::EvaluateLimiter(
        const TFlowDof &flow,
        real time)
    {
        _limiter.setConstant(1.0);
        _flowLimiter.setConstant(1.0);
        if (!_turbulenceSettings.secondOrderReconstruction)
        {
            _limiter.trans.startPersistentPull();
            _limiter.trans.waitPersistentPull();
            _flowLimiter.trans.startPersistentPull();
            _flowLimiter.trans.waitPersistentPull();
            return;
        }

        constexpr real tolerance = 1e-14;
        const int nActive = ActiveVariableCount();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            const TurbulenceState mean = _state[iCell];
            TurbulenceState minimum = mean;
            TurbulenceState maximum = mean;
            const auto cellFaces = _mesh->cell2face[iCell];
            for (rowsize iCellFace = 0; iCellFace < cellFaces.size(); iCellFace++)
            {
                const index iFace = cellFaces[iCellFace];
                const index otherCell = _mesh->CellFaceOther(iCell, iFace, iCellFace);
                const rowsize if2c =
                    _mesh->CellIsFaceBack(iCell, iFace, iCellFace) ? 0 : 1;
                TurbulenceState neighbour =
                    otherCell == UnInitIndex
                        ? GenerateBoundary(iFace, mean, iCell)
                        : TurbulenceState(_state[otherCell]);
                if (otherCell == UnInitIndex &&
                    _boundaryHandler->GetTypeFromID(
                        _mesh->GetFaceZone(iFace)) == BoundaryType::BCFar)
                {
                    Vector3 outward = ToVector3(
                        _vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1));
                    outward *= if2c == 0 ? 1.0 : -1.0;
                    if (State(flow[iCell]).template head<3>().dot(outward) >= 0)
                        neighbour = mean;
                }
                minimum = minimum.cwiseMin(neighbour);
                maximum = maximum.cwiseMax(neighbour);
            }
            for (int variable = 0; variable < nActive; variable++)
                minimum(variable) = std::max(
                    minimum(variable),
                    _turbulenceSettings.minimumValue[static_cast<std::size_t>(variable)]);

            TurbulenceState theta = TurbulenceState::Ones();
            for (rowsize iCellFace = 0; iCellFace < cellFaces.size(); iCellFace++)
            {
                const index iFace = cellFaces[iCellFace];
                const rowsize if2c = _mesh->CellIsFaceBack(iCell, iFace, iCellFace) ? 0 : 1;
                const auto quadrature = _vfv->GetFaceQuad(iFace);
                for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
                {
                    const Eigen::Matrix<real, gDim, 1> displacement =
                        (_vfv->GetFaceQuadraturePPhysFromCell(
                             iFace, iCell, if2c, iG) -
                         _vfv->GetCellQuadraturePPhys(iCell, -1))
                            .template head<gDim>();
                    const TurbulenceState increment =
                        _gradient[iCell].transpose() * displacement;
                    for (int variable = 0; variable < nActive; variable++)
                    {
                        if (increment(variable) > tolerance)
                            theta(variable) = std::min(
                                theta(variable),
                                (maximum(variable) - mean(variable)) /
                                    increment(variable));
                        else if (increment(variable) < -tolerance)
                            theta(variable) = std::min(
                                theta(variable),
                                (minimum(variable) - mean(variable)) /
                                    increment(variable));
                    }
                }
            }
            for (int variable = 0; variable < nActive; variable++)
                _limiter[iCell](variable) =
                    std::clamp(theta(variable), 0.0, 1.0);
            for (int variable = nActive; variable < nStoredVariables; variable++)
                _limiter[iCell](variable) = 0.0;

            const State flowMean = flow[iCell];
            State flowMinimum = flowMean;
            State flowMaximum = flowMean;
            for (rowsize iCellFace = 0; iCellFace < cellFaces.size(); iCellFace++)
            {
                const index iFace = cellFaces[iCellFace];
                const index otherCell = _mesh->CellFaceOther(iCell, iFace, iCellFace);
                const rowsize if2c = _mesh->CellIsFaceBack(iCell, iFace, iCellFace) ? 0 : 1;
                State neighbourFlow;
                if (otherCell != UnInitIndex)
                {
                    neighbourFlow = flow[otherCell];
                    Eigen::RowVector<real, nFlowVariables> neighbourRow =
                        neighbourFlow.transpose();
                    _vfv->ApplyPeriodicTransform(
                        if2c,
                        _mesh->GetFaceZone(iFace),
                        neighbourRow);
                    neighbourFlow = neighbourRow.transpose();
                }
                else
                {
                    Vector3 outward = ToVector3(
                        _vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1));
                    outward *= if2c == 0 ? 1.0 : -1.0;
                    neighbourFlow = GenerateBoundaryState(
                        _boundaryHandler->GetConditionFromID(_mesh->GetFaceZone(iFace)),
                        flowMean,
                        outward,
                        _flowSettings,
                        ToVector3(_vfv->GetFaceQuadraturePPhysFromCell(
                            iFace, iCell, if2c, -1)),
                        time);
                }
                flowMinimum = flowMinimum.cwiseMin(neighbourFlow);
                flowMaximum = flowMaximum.cwiseMax(neighbourFlow);
            }

            real flowTheta = 1.0;
            for (rowsize iCellFace = 0; iCellFace < cellFaces.size(); iCellFace++)
            {
                const index iFace = cellFaces[iCellFace];
                const rowsize if2c = _mesh->CellIsFaceBack(iCell, iFace, iCellFace) ? 0 : 1;
                const auto quadrature = _vfv->GetFaceQuad(iFace);
                for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
                {
                    const Eigen::Matrix<real, gDim, 1> displacement =
                        (_vfv->GetFaceQuadraturePPhysFromCell(
                             iFace, iCell, if2c, iG) -
                         _vfv->GetCellQuadraturePPhys(iCell, -1))
                            .template head<gDim>();
                    const State increment =
                        _flowGradient[iCell].transpose() * displacement;
                    for (int variable = 0; variable < nFlowVariables; variable++)
                    {
                        if (increment(variable) > tolerance)
                            flowTheta = std::min(
                                flowTheta,
                                (flowMaximum(variable) - flowMean(variable)) /
                                    increment(variable));
                        else if (increment(variable) < -tolerance)
                            flowTheta = std::min(
                                flowTheta,
                                (flowMinimum(variable) - flowMean(variable)) /
                                    increment(variable));
                    }
                }
            }
            _flowLimiter[iCell](0) = std::clamp(flowTheta, 0.0, 1.0);
        }
        _limiter.trans.startPersistentPull();
        _limiter.trans.waitPersistentPull();
        _flowLimiter.trans.startPersistentPull();
        _flowLimiter.trans.waitPersistentPull();
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::ReconstructTurbulenceState */
    TurbulenceState ACMTurbulenceTransport<gDim>::ReconstructTurbulenceState(
        index iCell,
        index iFace,
        rowsize if2c,
        int iG) const
    {
        TurbulenceState reconstructed = _state[iCell];
        if (_turbulenceSettings.secondOrderReconstruction)
        {
            const Eigen::Matrix<real, gDim, 1> displacement =
                (_vfv->GetFaceQuadraturePPhysFromCell(
                     iFace, iCell, if2c, iG) -
                 _vfv->GetCellQuadraturePPhys(iCell, -1))
                    .template head<gDim>();
            reconstructed +=
                _limiter[iCell].asDiagonal() *
                (_gradient[iCell].transpose() * displacement);
        }
        return ClampTurbulenceState(reconstructed, _turbulenceSettings);
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::ReconstructFlowState */
    State ACMTurbulenceTransport<gDim>::ReconstructFlowState(
        const TFlowDof &flow,
        index iCell,
        index iFace,
        rowsize if2c,
        int iG) const
    {
        State reconstructed = flow[iCell];
        if (_turbulenceSettings.secondOrderReconstruction)
        {
            const Eigen::Matrix<real, gDim, 1> displacement =
                (_vfv->GetFaceQuadraturePPhysFromCell(
                     iFace, iCell, if2c, iG) -
                 _vfv->GetCellQuadraturePPhys(iCell, -1))
                    .template head<gDim>();
            reconstructed += _flowLimiter[iCell](0) *
                             (_flowGradient[iCell].transpose() * displacement);
        }
        return reconstructed;
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::CellTurbulenceGradient */
    TurbulenceGradient ACMTurbulenceTransport<gDim>::CellTurbulenceGradient(
        index iCell) const
    {
        TurbulenceGradient result = TurbulenceGradient::Zero();
        result.template topRows<gDim>() = _gradient[iCell];
        return result;
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::CellVelocityGradient */
    VelocityGradient ACMTurbulenceTransport<gDim>::CellVelocityGradient(
        index iCell) const
    {
        VelocityGradient result = VelocityGradient::Zero();
        result.template leftCols<gDim>() =
            _flowGradient[iCell].template leftCols<3>().transpose();
        return result;
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::Prepare */
    void ACMTurbulenceTransport<gDim>::Prepare(TFlowDof &flow, real time)
    {
        DNDS_check_throw_info(_initialized,
                              "ACM turbulence Prepare requires Initialize first");
        EvaluateGradients(flow, time);
        EvaluateLimiter(flow, time);
        _faceEddyViscosity.clear();
        _faceEddyViscosity.resize(static_cast<std::size_t>(_mesh->NumFaceProc()));

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < _mesh->NumFaceProc(); iFace++)
        {
            const auto faceToCell = _mesh->face2cell[iFace];
            auto quadrature = _vfv->GetFaceQuad(iFace);
            auto &faceValues = _faceEddyViscosity[static_cast<std::size_t>(iFace)];
            faceValues.assign(static_cast<std::size_t>(quadrature.GetNumPoints()), 0.0);
            for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
            {
                const Vector3 normal = ToVector3(_vfv->GetFaceNorm(iFace, iG));
                const Vector3 point = ToVector3(
                    _vfv->GetFaceQuadraturePPhys(iFace, iG));
                const Vector3 leftCenter = ToVector3(
                    _vfv->GetCellQuadraturePPhys(faceToCell[0], -1));

                TurbulenceState leftTurbulence = ReconstructTurbulenceState(
                    faceToCell[0], iFace, 0, iG);
                State leftFlow = ReconstructFlowState(
                    flow, faceToCell[0], iFace, 0, iG);
                TurbulenceGradient gradientLeft = CellTurbulenceGradient(faceToCell[0]);
                Eigen::Matrix<real, 3, 4> flowGradientLeft =
                    Eigen::Matrix<real, 3, 4>::Zero();
                flowGradientLeft.template topRows<gDim>() = _flowGradient[faceToCell[0]];

                TurbulenceState rightTurbulence;
                State rightFlow;
                TurbulenceGradient gradientRight = gradientLeft;
                Eigen::Matrix<real, 3, 4> flowGradientRight = flowGradientLeft;
                TurbulenceState rightCellTurbulence;
                State rightCellFlow;
                Vector3 centerDisplacement;

                if (faceToCell[1] != UnInitIndex)
                {
                    rightTurbulence = ReconstructTurbulenceState(
                        faceToCell[1], iFace, 1, iG);
                    rightFlow = ReconstructFlowState(
                        flow, faceToCell[1], iFace, 1, iG);
                    rightCellTurbulence = _state[faceToCell[1]];
                    rightCellFlow = flow[faceToCell[1]];
                    gradientRight = CellTurbulenceGradient(faceToCell[1]);
                    flowGradientRight.setZero();
                    flowGradientRight.template topRows<gDim>() =
                        _flowGradient[faceToCell[1]];

                    Eigen::RowVector<real, nFlowVariables> rightFlowRow =
                        rightFlow.transpose();
                    Eigen::RowVector<real, nFlowVariables> rightCellFlowRow =
                        rightCellFlow.transpose();
                    _vfv->ApplyPeriodicTransform(
                        1, _mesh->GetFaceZone(iFace), rightFlowRow, rightCellFlowRow);
                    rightFlow = rightFlowRow.transpose();
                    rightCellFlow = rightCellFlowRow.transpose();

                    _vfv->ApplyPeriodicTransform(
                        1, _mesh->GetFaceZone(iFace), flowGradientRight);
                    Eigen::Matrix<real, nFlowVariables, gDim> flowSpatialTranspose =
                        flowGradientRight.template topRows<gDim>().transpose();
                    _vfv->ApplyPeriodicTransform(
                        1, _mesh->GetFaceZone(iFace), flowSpatialTranspose);
                    flowGradientRight.template topRows<gDim>() =
                        flowSpatialTranspose.transpose();

                    Eigen::Matrix<real, nStoredVariables, gDim> turbulenceSpatialTranspose =
                        gradientRight.template topRows<gDim>().transpose();
                    _vfv->ApplyPeriodicTransform(
                        1, _mesh->GetFaceZone(iFace), turbulenceSpatialTranspose);
                    gradientRight.template topRows<gDim>() =
                        turbulenceSpatialTranspose.transpose();

                    centerDisplacement = ToVector3(
                        _vfv->GetOtherCellPointFromCell(
                            faceToCell[0],
                            faceToCell[1],
                            iFace,
                            0,
                            _vfv->GetCellQuadraturePPhys(faceToCell[1], -1)) -
                        _vfv->GetCellQuadraturePPhys(faceToCell[0], -1));
                }
                else
                {
                    rightTurbulence = GenerateBoundary(
                        iFace, leftTurbulence, faceToCell[0]);
                    rightFlow = GenerateBoundaryState(
                        _boundaryHandler->GetConditionFromID(_mesh->GetFaceZone(iFace)),
                        leftFlow,
                        normal,
                        _flowSettings,
                        point,
                        time);
                    rightCellTurbulence = GenerateBoundary(
                        iFace,
                        TurbulenceState(_state[faceToCell[0]]),
                        faceToCell[0]);
                    rightCellFlow = GenerateBoundaryState(
                        _boundaryHandler->GetConditionFromID(_mesh->GetFaceZone(iFace)),
                        State(flow[faceToCell[0]]),
                        normal,
                        _flowSettings,
                        point,
                        time);
                    centerDisplacement =
                        2.0 * normal * (point - leftCenter).dot(normal);

                    // A characteristic far field prescribes turbulence only on inflow.
                    // On outflow, use the owner value for both the face state and the
                    // center-connection correction so that the normal diffusive gradient
                    // is exactly zero.  Modifier: Runzhi Ma.
                    if (_boundaryHandler->GetTypeFromID(
                            _mesh->GetFaceZone(iFace)) == BoundaryType::BCFar &&
                        leftFlow.template head<3>().dot(normal) >= 0)
                    {
                        rightTurbulence = leftTurbulence;
                        rightCellTurbulence = _state[faceToCell[0]];
                    }
                }

                TurbulenceGradient faceTurbulenceGradient =
                    0.5 * (gradientLeft + gradientRight);
                const real connectionProjection = centerDisplacement.dot(normal);
                if (std::abs(connectionProjection) > 1e-14)
                    faceTurbulenceGradient +=
                        normal *
                        (rightCellTurbulence - TurbulenceState(_state[faceToCell[0]]) -
                         faceTurbulenceGradient.transpose() * centerDisplacement)
                            .transpose() /
                        connectionProjection;

                const Eigen::Matrix<real, 3, 4> faceFlowGradient =
                    CorrectedFaceGradient(
                        flowGradientLeft,
                        flowGradientRight,
                        State(flow[faceToCell[0]]),
                        rightCellFlow,
                        centerDisplacement,
                        normal);
                VelocityGradient velocityGradient = VelocityGradient::Zero();
                velocityGradient.template leftCols<gDim>() =
                    faceFlowGradient.template topRows<gDim>()
                        .template leftCols<3>()
                        .transpose();
                const TurbulenceState faceTurbulence = ClampTurbulenceState(
                    0.5 * (leftTurbulence + rightTurbulence),
                    _turbulenceSettings);
                faceValues[static_cast<std::size_t>(iG)] =
                    TurbulentDynamicViscosity(
                        faceTurbulence,
                        velocityGradient,
                        faceTurbulenceGradient,
                        _faceWallDistance[static_cast<std::size_t>(iFace)],
                        _flowSettings.rho0,
                        _flowSettings.dynamicViscosity,
                        _turbulenceSettings);
            }
        }
        _prepared = true;
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::FaceEddyViscosity */
    real ACMTurbulenceTransport<gDim>::FaceEddyViscosity(
        index iFace,
        int iG) const
    {
        DNDS_check_throw_info(_prepared,
                              "ACM face eddy viscosity requested before Prepare");
        const auto &values = _faceEddyViscosity.at(static_cast<std::size_t>(iFace));
        DNDS_check_throw_info(!values.empty(),
                              "ACM face eddy-viscosity cache is empty");
        if (iG < 0)
            return std::accumulate(values.begin(), values.end(), 0.0) /
                   static_cast<real>(values.size());
        return values.at(static_cast<std::size_t>(iG));
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::EvaluateRHS */
    void ACMTurbulenceTransport<gDim>::EvaluateRHS(
        TTurbulenceDof &rhs,
        TFlowDof &flow,
        real time)
    {
        Prepare(flow, time);
        rhs.setConstant(0.0);

        for (index iFace = 0; iFace < _mesh->NumFaceProc(); iFace++)
        {
            const auto faceToCell = _mesh->face2cell[iFace];
            auto quadrature = _vfv->GetFaceQuad(iFace);
            TurbulenceState integratedResidual = TurbulenceState::Zero();
            quadrature.IntegrationSimple(
                integratedResidual,
                [&](TurbulenceState &contribution, int iG, real weight)
                {
                    (void)weight;
                    const Vector3 normal = ToVector3(_vfv->GetFaceNorm(iFace, iG));
                    const Vector3 point = ToVector3(
                        _vfv->GetFaceQuadraturePPhys(iFace, iG));
                    const Vector3 leftCenter = ToVector3(
                        _vfv->GetCellQuadraturePPhys(faceToCell[0], -1));
                    TurbulenceState leftTurbulence = ReconstructTurbulenceState(
                        faceToCell[0], iFace, 0, iG);
                    State leftFlow = ReconstructFlowState(
                        flow, faceToCell[0], iFace, 0, iG);
                    TurbulenceGradient gradientLeft =
                        CellTurbulenceGradient(faceToCell[0]);
                    TurbulenceGradient gradientRight = gradientLeft;
                    TurbulenceState rightTurbulence;
                    State rightFlow;
                    TurbulenceState rightCellTurbulence;
                    Vector3 centerDisplacement;

                    if (faceToCell[1] != UnInitIndex)
                    {
                        rightTurbulence = ReconstructTurbulenceState(
                            faceToCell[1], iFace, 1, iG);
                        rightFlow = ReconstructFlowState(
                            flow, faceToCell[1], iFace, 1, iG);
                        rightCellTurbulence = _state[faceToCell[1]];
                        gradientRight = CellTurbulenceGradient(faceToCell[1]);
                        Eigen::RowVector<real, nFlowVariables> rightFlowRow =
                            rightFlow.transpose();
                        _vfv->ApplyPeriodicTransform(
                            1, _mesh->GetFaceZone(iFace), rightFlowRow);
                        rightFlow = rightFlowRow.transpose();
                        Eigen::Matrix<real, nStoredVariables, gDim> spatialTranspose =
                            gradientRight.template topRows<gDim>().transpose();
                        _vfv->ApplyPeriodicTransform(
                            1, _mesh->GetFaceZone(iFace), spatialTranspose);
                        gradientRight.template topRows<gDim>() =
                            spatialTranspose.transpose();
                        centerDisplacement = ToVector3(
                            _vfv->GetOtherCellPointFromCell(
                                faceToCell[0],
                                faceToCell[1],
                                iFace,
                                0,
                                _vfv->GetCellQuadraturePPhys(faceToCell[1], -1)) -
                            _vfv->GetCellQuadraturePPhys(faceToCell[0], -1));
                    }
                    else
                    {
                        rightTurbulence = GenerateBoundary(
                            iFace, leftTurbulence, faceToCell[0]);
                        rightFlow = GenerateBoundaryState(
                            _boundaryHandler->GetConditionFromID(_mesh->GetFaceZone(iFace)),
                            leftFlow,
                            normal,
                            _flowSettings,
                            point,
                            time);
                        rightCellTurbulence = GenerateBoundary(
                            iFace,
                            TurbulenceState(_state[faceToCell[0]]),
                            faceToCell[0]);
                        centerDisplacement =
                            2.0 * normal * (point - leftCenter).dot(normal);
                        if (_boundaryHandler->GetTypeFromID(
                                _mesh->GetFaceZone(iFace)) == BoundaryType::BCFar &&
                            leftFlow.template head<3>().dot(normal) >= 0)
                        {
                            rightTurbulence = leftTurbulence;
                            rightCellTurbulence = _state[faceToCell[0]];
                        }
                    }

                    TurbulenceGradient faceGradient =
                        0.5 * (gradientLeft + gradientRight);
                    const real connectionProjection = centerDisplacement.dot(normal);
                    if (std::abs(connectionProjection) > 1e-14)
                        faceGradient +=
                            normal *
                            (rightCellTurbulence -
                             TurbulenceState(_state[faceToCell[0]]) -
                             faceGradient.transpose() * centerDisplacement)
                                .transpose() /
                            connectionProjection;

                    const TurbulenceState faceTurbulence = ClampTurbulenceState(
                        0.5 * (leftTurbulence + rightTurbulence),
                        _turbulenceSettings);
                    const real normalVelocityLeft = leftFlow.head<3>().dot(normal);
                    const real normalVelocityRight = rightFlow.head<3>().dot(normal);
                    TurbulenceState advectiveFlux =
                        0.5 *
                        (normalVelocityLeft * leftTurbulence +
                         normalVelocityRight * rightTurbulence -
                         std::max(std::abs(normalVelocityLeft),
                                  std::abs(normalVelocityRight)) *
                             (rightTurbulence - leftTurbulence));
                    if (faceToCell[1] == UnInitIndex)
                    {
                        const BoundaryType boundaryType =
                            _boundaryHandler->GetTypeFromID(
                                _mesh->GetFaceZone(iFace));
                        if (boundaryType == BoundaryType::BCWall ||
                            boundaryType == BoundaryType::BCWallIsothermal ||
                            boundaryType == BoundaryType::BCWallInvis ||
                            boundaryType == BoundaryType::BCSym)
                            advectiveFlux.setZero();
                    }
                    const TurbulenceState diffusiveFlux =
                        TurbulenceDiffusiveFlux(
                            faceTurbulence,
                            faceGradient,
                            normal,
                            _faceWallDistance[static_cast<std::size_t>(iFace)],
                            _flowSettings.rho0,
                            _flowSettings.dynamicViscosity,
                            FaceEddyViscosity(iFace, iG),
                            _turbulenceSettings);
                    contribution = -(advectiveFlux - diffusiveFlux) *
                                   _vfv->GetFaceJacobiDet(iFace, iG);
                });

            rhs[faceToCell[0]] +=
                integratedResidual / _vfv->GetCellVol(faceToCell[0]);
            if (faceToCell[1] != UnInitIndex)
                rhs[faceToCell[1]] -=
                    integratedResidual / _vfv->GetCellVol(faceToCell[1]);
        }

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            const TurbulenceState source = TurbulenceSource(
                TurbulenceState(_state[iCell]),
                CellVelocityGradient(iCell),
                CellTurbulenceGradient(iCell),
                _cellWallDistance[static_cast<std::size_t>(iCell)],
                _flowSettings.rho0,
                _flowSettings.dynamicViscosity,
                _turbulenceSettings);
            rhs[iCell] += source;
            DNDS_check_throw_info(
                rhs[iCell].allFinite(),
                "ACM turbulence transport produced a non-finite total residual");
        }
    }

    template <int gDim>
    /** @copydoc ACMTurbulenceTransport::Advance */
    real ACMTurbulenceTransport<gDim>::Advance(
        TFlowDof &flow,
        const ScalarField &flowPseudoTimeStep,
        real time)
    {
        DNDS_check_throw_info(
            flowPseudoTimeStep.size() ==
                static_cast<std::size_t>(_mesh->NumCell()),
            "ACM turbulence time-step field does not match owned cells");
        for (const real value : flowPseudoTimeStep)
            DNDS_check_throw_info(std::isfinite(value) && value > 0,
                                  "ACM turbulence requires positive finite time steps");

        const std::size_t nOwned = static_cast<std::size_t>(_mesh->NumCell());
        std::vector<TurbulenceState> state0(nOwned);
        std::vector<TurbulenceState> state1(nOwned);
        std::vector<TurbulenceState> state2(nOwned);
        real lastResidualNorm = 0;

        // Limit each explicit stage by the distance to the configured admissible interval.
        // This residual-based local step is especially important for the stiff wall-omega
        // condition and leaves the four-equation ACM flow time step unchanged.
        // Modifier: Runzhi Ma.
        const auto boundedStageTimeStep = [this](
                                                  const TurbulenceState &state,
                                                  const TurbulenceState &residual,
                                                  real requestedTimeStep)
        {
            constexpr real safety = 0.9;
            real boundedTimeStep = requestedTimeStep;
            for (int variable = 0; variable < ActiveVariableCount(); variable++)
            {
                const std::size_t component = static_cast<std::size_t>(variable);
                if (residual(variable) > 0)
                {
                    const real available = std::max(
                        _turbulenceSettings.maximumValue[component] - state(variable),
                        0.0);
                    boundedTimeStep = std::min(
                        boundedTimeStep,
                        safety * available / residual(variable));
                }
                else if (residual(variable) < 0)
                {
                    const real available = std::max(
                        state(variable) - _turbulenceSettings.minimumValue[component],
                        0.0);
                    boundedTimeStep = std::min(
                        boundedTimeStep,
                        safety * available / (-residual(variable)));
                }
            }
            return std::max(boundedTimeStep, 0.0);
        };

        for (int substep = 0;
             substep < _turbulenceSettings.transportSubsteps;
             substep++)
        {
            for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
                state0[static_cast<std::size_t>(iCell)] =
                    ClampTurbulenceState(_state[iCell], _turbulenceSettings);

            EvaluateRHS(_rhs, flow, time);
            lastResidualNorm = _rhs.norm2();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
            {
                const std::size_t ii = static_cast<std::size_t>(iCell);
                const real requestedDt = flowPseudoTimeStep[ii] *
                                         _turbulenceSettings.transportTimeScale /
                                         static_cast<real>(_turbulenceSettings.transportSubsteps);
                const real dt = boundedStageTimeStep(
                    state0[ii], TurbulenceState(_rhs[iCell]), requestedDt);
                state1[ii] = ClampTurbulenceState(
                    state0[ii] + dt * TurbulenceState(_rhs[iCell]),
                    _turbulenceSettings);
                _state[iCell] = state1[ii];
            }

            EvaluateRHS(_rhs, flow, time);
            lastResidualNorm = _rhs.norm2();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
            {
                const std::size_t ii = static_cast<std::size_t>(iCell);
                const real requestedDt = flowPseudoTimeStep[ii] *
                                         _turbulenceSettings.transportTimeScale /
                                         static_cast<real>(_turbulenceSettings.transportSubsteps);
                const real dt = boundedStageTimeStep(
                    state1[ii], TurbulenceState(_rhs[iCell]), requestedDt);
                state2[ii] = ClampTurbulenceState(
                    0.75 * state0[ii] +
                        0.25 * (state1[ii] + dt * TurbulenceState(_rhs[iCell])),
                    _turbulenceSettings);
                _state[iCell] = state2[ii];
            }

            EvaluateRHS(_rhs, flow, time);
            lastResidualNorm = _rhs.norm2();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
            {
                const std::size_t ii = static_cast<std::size_t>(iCell);
                const real requestedDt = flowPseudoTimeStep[ii] *
                                         _turbulenceSettings.transportTimeScale /
                                         static_cast<real>(_turbulenceSettings.transportSubsteps);
                const real dt = boundedStageTimeStep(
                    state2[ii], TurbulenceState(_rhs[iCell]), requestedDt);
                _state[iCell] = ClampTurbulenceState(
                    (1.0 / 3.0) * state0[ii] +
                        (2.0 / 3.0) *
                            (state2[ii] + dt * TurbulenceState(_rhs[iCell])),
                    _turbulenceSettings);
            }
        }

        _state.trans.startPersistentPull();
        _state.trans.waitPersistentPull();
        _prepared = false;
        const real normalizer = std::sqrt(std::max<real>(
            1.0,
            static_cast<real>(_mesh->NumCellGlobal()) *
                static_cast<real>(ActiveVariableCount())));
        return lastResidualNorm / normalizer;
    }
}
