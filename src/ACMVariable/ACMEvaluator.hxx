/**
 * @file ACMEvaluator.hxx
 * @brief Template implementation of the ACM high-order mesh evaluator.
 * @author Runzhi Ma
 * @date 2026-09-03
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMEvaluator.hpp"

#include "DNDS/Errors.hpp"
#include "ACMPeriodic.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace DNDS::ACMVariable
{

    template <int gDim>
    FluxResult ACMEvaluator<gDim>::InviscidFaceFlux(
        const State &left, const State &right, const Vector3 &normal,
        index face, int point, real time, real leftKIncrement, real rightKIncrement) const
    {
        if (!_turbulentPressure)
            return InviscidFlux(_settings.riemannSolverType, left, right, normal, _settings);
        const auto pressure = _turbulentPressure(face, point, left, right, leftKIncrement, rightKIncrement);
        State totalLeft = left, totalRight = right;
        totalLeft(4) += pressure[0];
        totalRight(4) += pressure[1];
        if (_mesh->face2cell(face, 1) == UnInitIndex)
        {
            auto condition = _boundaryHandler->GetConditionFromID(_mesh->GetFaceZone(face));
            condition.value[4] += pressure[2];
            // Apply pressure data as static p; impose extrapolation on total normal stress.
            // In particular, an impermeable wall cannot acquire a mass flux from a k jump.
            totalRight = GenerateBoundaryState(condition, totalLeft, normal, _settings,
                ToVector3(_vfv->GetFaceQuadraturePPhys(face, point)), time);
        }
        return InviscidFlux(_settings.riemannSolverType, totalLeft, totalRight, normal, _settings);
    }

    template <int gDim>
    void ACMEvaluator<gDim>::AssembleKineticPressureLinearization(
        TDof &u, KineticPressureJacobianField &faces, real time)
    {
        faces.assign(static_cast<std::size_t>(_mesh->NumFaceProc()), {State::Zero(), State::Zero()});
        if (!_turbulentPressure)
            return;
        for (index face = 0; face < _mesh->NumFaceProc(); ++face)
        {
            const auto cells = _mesh->face2cell[face];
            if (cells[0] == cells[1])
                continue;
            const auto quad = _vfv->GetFaceQuad(face);
            for (int point = 0; point < quad.GetNumPoints(); ++point)
            {
                const State left = u[cells[0]];
                const Vector3 normal = ToVector3(_vfv->GetFaceNorm(face, point));
                State right = left;
                if (cells[1] != UnInitIndex)
                {
                    right = u[cells[1]];
                    Eigen::RowVector<real, nVarsFixed> row = right.transpose();
                    _vfv->ApplyPeriodicTransform(1, _mesh->GetFaceZone(face), row);
                    right = row.transpose();
                }
                else
                    right = GenerateBoundaryForFace(face, left, normal,
                        ToVector3(_vfv->GetFaceQuadraturePPhys(face, point)), time);
                const auto pressure = _turbulentPressure(face, point, left, right, 0, 0);
                if (pressure[0] == 0 && pressure[1] == 0)
                    continue; // SA, legacy pressure mode, or no kinetic stress.
                const real epsilon = 1e-6 * std::max({real(1e-8), std::abs(pressure[0]), std::abs(pressure[1])});
                const real factor = quad.GetWeight(point) * _vfv->GetFaceJacobiDet(face, point) / (2 * epsilon);
                for (int side = 0; side < (cells[1] == UnInitIndex ? 1 : 2); ++side)
                {
                    const State plus = InviscidFaceFlux(left, right, normal, face, point, time,
                        side == 0 ? epsilon : 0, side == 1 ? epsilon : 0).flux;
                    const State minus = InviscidFaceFlux(left, right, normal, face, point, time,
                        side == 0 ? -epsilon : 0, side == 1 ? -epsilon : 0).flux;
                    faces[face][side] += factor * (plus - minus);
                }
            }
        }
    }

    template <int gDim>
    ACMEvaluator<gDim>::ACMEvaluator(
        ssp<Geom::UnstructuredMesh> mesh,
        TpVFV vfv,
        const Settings &settings,
        const ReconstructionSettings &reconstructionSettings,
        ssp<BoundaryHandler> boundaryHandler)
        : _mesh(std::move(mesh)),
          _vfv(std::move(vfv)),
          _settings(settings),
          _reconstructionSettings(reconstructionSettings),
          _boundaryHandler(std::move(boundaryHandler))
    {
        DNDS_check_throw_info(_mesh != nullptr && _vfv != nullptr, "ACM evaluator requires mesh and VFV objects");
        DNDS_check_throw_info(_mesh->getDim() == gDim, "ACM evaluator dimension does not match the mesh");
        _settings.Validate();
        DNDS_check_throw_info(_boundaryHandler != nullptr, "ACM evaluator requires a boundary handler");

        _vfv->BuildURec(_uRec, nVarsFixed);
        _vfv->BuildURec(_uRecWork, nVarsFixed);
        _vfv->BuildURec(_uRecLimited, nVarsFixed);
        _vfv->BuildUGrad(_uGrad, nVarsFixed);
        _vfv->BuildUDof(_limiter, 1);
        _vfv->BuildScalar(_smoothIndicator);
        _vfv->BuildUDof(_lusgsOperatorProduct, nVarsFixed);
        _vfv->BuildUDof(_lusgsCorrection, nVarsFixed);
        _uRec.setConstant(0.0);
        _uRecWork.setConstant(0.0);
        _uRecLimited.setConstant(0.0);
        _uGrad.setConstant(0.0);
        _limiter.setConstant(1.0);
        _lusgsOperatorProduct.setConstant(0.0);
        _lusgsCorrection.setConstant(0.0);
    }

    template <int gDim>
    /** @copydoc ACMEvaluator::SetTurbulenceCoupling */
    void ACMEvaluator<gDim>::SetTurbulenceCoupling(
        TurbulencePrepareFunction prepareFunction,
        TurbulentViscosityFunction viscosityFunction,
        TurbulentPressureFunction pressureFunction)
    {
        DNDS_check_throw_info(
            static_cast<bool>(prepareFunction) == static_cast<bool>(viscosityFunction),
            "ACM turbulence prepare and viscosity callbacks must be attached together");
        _prepareTurbulence = std::move(prepareFunction);
        _turbulentViscosity = std::move(viscosityFunction);
        _turbulentPressure = std::move(pressureFunction);
    }

    template <int gDim>
    /** @copydoc ACMEvaluator::GetTurbulentViscosity */
    real ACMEvaluator<gDim>::GetTurbulentViscosity(index iFace, int iG) const
    {
        if (!_turbulentViscosity)
            return 0;
        const real value = _turbulentViscosity(iFace, iG);
        DNDS_check_throw_info(std::isfinite(value) && value >= 0,
                              "ACM turbulence callback returned invalid eddy viscosity");
        return value;
    }

    template <int gDim>
    void ACMEvaluator<gDim>::InitializeFV(
        const ssp<Geom::UnstructuredMesh> &mesh,
        const TpVFV &vfv,
        const ssp<BoundaryHandler> &boundaryHandler)
    {
        DNDS_check_throw_info(mesh != nullptr && vfv != nullptr, "ACM InitializeFV requires mesh and VFV objects");
        DNDS_check_throw_info(boundaryHandler != nullptr, "ACM InitializeFV requires a boundary handler");
        if constexpr (gDim == 2)
            vfv->SetPeriodicTransformations(std::array<int, 2>{1, 2});
        else
            vfv->SetPeriodicTransformations(std::array<int, 3>{1, 2, 3});

        vfv->ConstructMetrics();
        vfv->ConstructBaseAndWeight(
            [boundaryHandler](Geom::t_index faceID, int derivativeOrder) -> real
            {
                if (Geom::FaceIDIsPeriodic(faceID))
                    return 1.0;
                (void)boundaryHandler->GetTypeFromID(faceID);
                return derivativeOrder == 0 ? 1.0 : 0.0;
            });
        vfv->ConstructRecCoeff();
    }

    template <int gDim>
    typename ACMEvaluator<gDim>::TBoundaryFunction
    ACMEvaluator<gDim>::GetBoundaryFunction(real time) const
    {
        return [this, time](
                   const State &interior,
                   const State &cellMean,
                   index iCell,
                   index iFace,
                   int iG,
                   const Geom::tPoint &normal,
                   const Geom::tPoint &point,
                   Geom::t_index faceType) -> State
        {
            (void)cellMean;
            (void)iCell;
            (void)iG;
            (void)faceType;
            return GenerateBoundaryForFace(
                iFace,
                IsFiniteState(interior) ? interior : cellMean,
                ToVector3(normal),
                ToVector3(point),
                time, iG);
        };
    }

    template <int gDim>
    State ACMEvaluator<gDim>::GenerateBoundaryForFace(
        index face, const State &interior, const Vector3 &normal,
        const Vector3 &point, real time, int quadrature) const
    {
        auto condition = _boundaryHandler->GetConditionFromID(_mesh->GetFaceZone(face));
        const State raw = GenerateBoundaryState(condition, interior, normal, _settings, point, time);
        if (!_turbulentPressure)
            return raw;
        const auto pressure = _turbulentPressure(face, quadrature, interior, raw, 0, 0);
        State total = interior;
        total(4) += pressure[0];
        condition.value[4] += pressure[2];
        State ghost = GenerateBoundaryState(condition, total, normal, _settings, point, time);
        ghost(4) -= pressure[1];
        return ghost; // Static p; its effective total-pressure state is used by InviscidFaceFlux.
    }

    template <int gDim>
    Vector3 ACMEvaluator<gDim>::ToVector3(const Geom::tPoint &normal)
    {
        Vector3 result = Vector3::Zero();
        result.template head<gDim>() = normal.template head<gDim>();
        return result;
    }

    template <int gDim>
    void ACMEvaluator<gDim>::Reconstruct(TDof &u, real time)
    {
        u.trans.startPersistentPull();
        u.trans.waitPersistentPull();

        if (_turbulentPressure && _prepareTurbulence)
            _prepareTurbulence(u, time);
        const auto boundary = GetBoundaryFunction(time);
        if (_reconstructionSettings.type == ReconstructionType::FirstOrder)
        {
            _uRec.setConstant(0.0);
            _uGrad.setConstant(0.0);
        }
        else if (_reconstructionSettings.type == ReconstructionType::GreenGauss)
        {
            _vfv->template DoReconstruction2ndGrad<nVarsFixed>(_uGrad, u, boundary, 1);
            _uGrad.trans.startPersistentPull();
            _uGrad.trans.waitPersistentPull();
        }
        else
        {
            if (_reconstructionSettings.resetVariationalCoefficients)
                _uRec.setConstant(0.0);
            for (int iteration = 0; iteration < _reconstructionSettings.variationalIterations; iteration++)
            {
                _vfv->template DoReconstructionIter<nVarsFixed>(
                    _uRec,
                    _uRecWork,
                    u,
                    boundary,
                    false);
                _uRec.trans.startPersistentPull();
                _uRec.trans.waitPersistentPull();
            }
        }
        if (_reconstructionSettings.enableLimiter &&
            _reconstructionSettings.type == ReconstructionType::Variational &&
            _reconstructionSettings.limiterType != LimiterType::LocalExtrema)
            ApplyCharacteristicLimiter(u);
        else
            EvaluateLimiter(u, time);
    }

    template <int gDim>
    State ACMEvaluator<gDim>::ReconstructFaceState(
        const TDof &u,
        index iCell,
        index iFace,
        rowsize if2c,
        int iG) const
    {
        State state = u[iCell];
        if (_reconstructionSettings.type == ReconstructionType::FirstOrder)
            return state;

        State increment = State::Zero();
        if (_reconstructionSettings.type == ReconstructionType::GreenGauss)
        {
            const auto displacement =
                (_vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, iG) -
                 _vfv->GetCellQuadraturePPhys(iCell, -1))
                    .template head<gDim>();
            increment = _uGrad[iCell].transpose() * displacement;
        }
        else
        {
            increment =
                (_vfv->GetIntPointDiffBaseValue(
                     iCell,
                     iFace,
                     if2c,
                     iG,
                     std::array<int, 1>{0},
                     1) *
                 _uRec[iCell])
                    .transpose();
        }
        real theta = _limiter[iCell](0);
        if (increment(0) < 0)
            theta = std::min(theta, std::max(real(0), (state(0) - _settings.densityFloor) / (-increment(0))));
        state += theta * increment;
        return state;
    }

    template <int gDim>
    Eigen::Matrix<real, 3, 5> ACMEvaluator<gDim>::ReconstructFaceGradient(
        index iCell,
        index iFace,
        rowsize if2c,
        int iG) const
    {
        Eigen::Matrix<real, 3, 5> gradient = Eigen::Matrix<real, 3, 5>::Zero();
        if (_reconstructionSettings.type == ReconstructionType::GreenGauss)
            gradient.template topRows<gDim>() = _uGrad[iCell];
        else if (_reconstructionSettings.type == ReconstructionType::Variational)
        {
            const auto derivativeIndices = Eigen::seq(Eigen::fix<1>, Eigen::fix<gDim>);
            gradient.template topRows<gDim>() =
                _vfv->GetIntPointDiffBaseValue(
                    iCell,
                    iFace,
                    if2c,
                    iG,
                    derivativeIndices,
                    gDim + 1) *
                _uRec[iCell];
        }
        gradient *= _limiter[iCell](0);
        return gradient;
    }

    template <int gDim>
    void ACMEvaluator<gDim>::EvaluateLimiter(TDof &u, real time)
    {
        _limiter.setConstant(1.0);
        if (!_reconstructionSettings.enableLimiter ||
            _reconstructionSettings.type == ReconstructionType::FirstOrder)
        {
            _limiter.trans.startPersistentPull();
            _limiter.trans.waitPersistentPull();
            return;
        }

        constexpr real tolerance = 1e-14;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            const State mean = u[iCell];
            State minimum = mean;
            State maximum = mean;
            const auto cellFaces = _mesh->cell2face[iCell];

            for (rowsize iCellFace = 0; iCellFace < cellFaces.size(); iCellFace++)
            {
                const index iFace = cellFaces[iCellFace];
                const index otherCell = _mesh->CellFaceOther(iCell, iFace, iCellFace);
                const rowsize if2c = _mesh->CellIsFaceBack(iCell, iFace, iCellFace) ? 0 : 1;
                State neighbour;
                if (otherCell != UnInitIndex)
                {
                    neighbour = u[otherCell];
                    Eigen::RowVector<real, 5> neighbourRow = neighbour.transpose();
                    _vfv->ApplyPeriodicTransform(if2c, _mesh->GetFaceZone(iFace), neighbourRow);
                    neighbour = neighbourRow.transpose();
                }
                else
                {
                    const Vector3 outward = ToVector3(
                        _vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1));
                    neighbour = GenerateBoundaryForFace(
                        iFace,
                        mean,
                        outward,
                        ToVector3(_vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, -1)),
                        time);
                }
                minimum = minimum.cwiseMin(neighbour);
                maximum = maximum.cwiseMax(neighbour);
            }

            real theta = 1.0;
            for (rowsize iCellFace = 0; iCellFace < cellFaces.size(); iCellFace++)
            {
                const index iFace = cellFaces[iCellFace];
                const rowsize if2c = _mesh->CellIsFaceBack(iCell, iFace, iCellFace) ? 0 : 1;
                const auto quadrature = _vfv->GetFaceQuad(iFace);
                for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
                {
                    State unlimited = mean;
                    if (_reconstructionSettings.type == ReconstructionType::GreenGauss)
                    {
                        const auto displacement =
                            (_vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, iG) -
                             _vfv->GetCellQuadraturePPhys(iCell, -1))
                                .template head<gDim>();
                        unlimited += _uGrad[iCell].transpose() * displacement;
                    }
                    else
                    {
                        unlimited +=
                            (_vfv->GetIntPointDiffBaseValue(
                                 iCell, iFace, if2c, iG,
                                 std::array<int, 1>{0}, 1) *
                             _uRec[iCell])
                                .transpose();
                    }

                    for (int variable = 0; variable < nVarsFixed; variable++)
                    {
                        const real increment = unlimited(variable) - mean(variable);
                        if (increment > tolerance)
                            theta = std::min(theta, (maximum(variable) - mean(variable)) / increment);
                        else if (increment < -tolerance)
                            theta = std::min(theta, (minimum(variable) - mean(variable)) / increment);
                    }
                }
            }
            _limiter[iCell](0) = std::clamp(theta, 0.0, 1.0);
        }
        _limiter.trans.startPersistentPull();
        _limiter.trans.waitPersistentPull();
    }

    template <int gDim>
    /** @copydoc ACMEvaluator::ApplyCharacteristicLimiter */
    void ACMEvaluator<gDim>::ApplyCharacteristicLimiter(TDof &u)
    {
        using TLimitBatch = typename TVFV::template tLimitBatch<nVarsFixed>;
        const real velocityScale = std::sqrt(_settings.beta2 / _settings.rho0);
        const real pressureScale = _settings.beta2;

        _vfv->template DoCalculateSmoothIndicatorV1<nVarsFixed>(
            _smoothIndicator,
            _uRec,
            u,
            State::Ones(),
            [velocityScale, pressureScale](Eigen::Matrix<real, 1, nVarsFixed> &value)
            {
                const State state = value.transpose();
                const real scaledVelocitySquared =
                    state.segment<3>(1).squaredNorm() / (velocityScale * velocityScale);
                const real scaledPressure = state(4) / pressureScale;
                value.setConstant(std::sqrt(state(0) * state(0) + scaledVelocitySquared + scaledPressure * scaledPressure));
            });

        const auto toCharacteristic = [this](
                                                const State &left,
                                                const State &right,
                                                const Geom::tPoint &normal,
                                                const Eigen::Ref<TLimitBatch> &coefficients) -> TLimitBatch
        {
            Matrix5 leftTransform;
            Matrix5 rightTransform;
            if (!TryCharacteristicMatricesGlobal(
                    RoeAverage(left, right),
                    ToVector3(normal),
                    _settings,
                    leftTransform,
                    rightTransform))
                return coefficients;
            return (leftTransform * coefficients.transpose()).transpose();
        };
        const auto fromCharacteristic = [this](
                                                  const State &left,
                                                  const State &right,
                                                  const Geom::tPoint &normal,
                                                  const Eigen::Ref<TLimitBatch> &coefficients) -> TLimitBatch
        {
            Matrix5 leftTransform;
            Matrix5 rightTransform;
            if (!TryCharacteristicMatricesGlobal(
                    RoeAverage(left, right),
                    ToVector3(normal),
                    _settings,
                    leftTransform,
                    rightTransform))
                return coefficients;
            return (rightTransform * coefficients.transpose()).transpose();
        };

        if (_reconstructionSettings.limiterType == LimiterType::CWBAP)
            _vfv->template DoLimiterWBAP_C<nVarsFixed>(
                u,
                _uRec,
                _uRecLimited,
                _uRecWork,
                _smoothIndicator,
                false,
                toCharacteristic,
                fromCharacteristic,
                false);
        else
            _vfv->template DoLimiterWBAP_3<nVarsFixed>(
                u,
                _uRec,
                _uRecLimited,
                _uRecWork,
                _smoothIndicator,
                false,
                toCharacteristic,
                fromCharacteristic,
                false);

        // WBAP/CWBAP modifies polynomial coefficients directly, so the scalar face limiter must
        // remain one to avoid applying the local-extrema limiter a second time.
        _limiter.setConstant(1.0);
        _limiter.trans.startPersistentPull();
        _limiter.trans.waitPersistentPull();
    }

    template <int gDim>
    void ACMEvaluator<gDim>::EvaluateRHS(TDof &rhs, TDof &u, real time)
    {
        Reconstruct(u, time);
        if (_prepareTurbulence)
            _prepareTurbulence(u, time);
        rhs.setConstant(0.0);
        _faceMassFlux.resize(static_cast<std::size_t>(_mesh->NumFaceProc()));

        for (index iFace = 0; iFace < _mesh->NumFaceProc(); iFace++)
        {
            const auto faceToCell = _mesh->face2cell[iFace];
            auto quadrature = _vfv->GetFaceQuad(iFace);
            _faceMassFlux[iFace].resize(quadrature.GetNumPoints());
            State integratedFlux = State::Zero();
            quadrature.IntegrationSimple(
                integratedFlux,
                [&](State &contribution, int iG, real weight)
                {
                    (void)weight;
                    const Vector3 unitNormal = ToVector3(_vfv->GetFaceNorm(iFace, iG));
                    State left = ReconstructFaceState(u, faceToCell[0], iFace, 0, iG);
                    State right;
                    Eigen::Matrix<real, 3, 5> gradientLeft =
                        ReconstructFaceGradient(faceToCell[0], iFace, 0, iG);
                    Eigen::Matrix<real, 3, 5> gradientRight = gradientLeft;

                    if (faceToCell[1] != UnInitIndex)
                    {
                        right = ReconstructFaceState(u, faceToCell[1], iFace, 1, iG);
                        gradientRight = ReconstructFaceGradient(faceToCell[1], iFace, 1, iG);
                        Eigen::RowVector<real, 5> rightRow = right.transpose();
                        _vfv->ApplyPeriodicTransform(1, _mesh->GetFaceZone(iFace), rightRow);
                        right = rightRow.transpose();

                        // A periodic rotation acts on both the state-vector columns and the
                        // spatial-derivative rows of the gradient tensor. Apply the same CFV
                        // face-frame map in both directions before averaging gradients.
                        // Modifier: Runzhi Ma.
                        _vfv->ApplyPeriodicTransform(
                            1,
                            _mesh->GetFaceZone(iFace),
                            gradientRight);
                        Eigen::Matrix<real, 5, gDim> spatialTranspose =
                            gradientRight.template topRows<gDim>().transpose();
                        TransformSpatialGradient<gDim>(_mesh, 1, _mesh->GetFaceZone(iFace), spatialTranspose);
                        gradientRight.template topRows<gDim>() = spatialTranspose.transpose();
                    }
                    else
                    {
                        right = GenerateBoundaryForFace(
                            iFace,
                            left,
                            unitNormal,
                            ToVector3(_vfv->GetFaceQuadraturePPhys(iFace, iG)),
                            time, iG);
                    }

                    State totalFlux = InviscidFaceFlux(left, right, unitNormal, iFace, iG, time)
                                          .flux;
                    _faceMassFlux[iFace][iG] = totalFlux(0);
                    if (_settings.enableViscousFlux)
                    {
                        const State leftCellState = u[faceToCell[0]];
                        State rightCellState;
                        const Vector3 leftCenter = ToVector3(
                            _vfv->GetCellQuadraturePPhys(faceToCell[0], -1));
                        Vector3 centerDisplacement;
                        if (faceToCell[1] != UnInitIndex)
                        {
                            rightCellState = u[faceToCell[1]];
                            Eigen::RowVector<real, 5> rightCellRow = rightCellState.transpose();
                            _vfv->ApplyPeriodicTransform(
                                1,
                                _mesh->GetFaceZone(iFace),
                                rightCellRow);
                            rightCellState = rightCellRow.transpose();
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
                            const Vector3 facePoint = ToVector3(
                                _vfv->GetFaceQuadraturePPhys(iFace, iG));
                            rightCellState = GenerateBoundaryForFace(
                                iFace,
                                leftCellState,
                                unitNormal,
                                facePoint,
                                time);
                            centerDisplacement =
                                2 * unitNormal * (facePoint - leftCenter).dot(unitNormal);
                        }

                        const Eigen::Matrix<real, 3, 5> faceGradient =
                            CorrectedFaceGradient(
                                PrimitiveGradient(left, gradientLeft),
                                PrimitiveGradient(right, gradientRight),
                                PrimitiveState(leftCellState),
                                PrimitiveState(rightCellState),
                                centerDisplacement,
                                unitNormal);
                        totalFlux -= ViscousFlux(
                            faceGradient,
                            unitNormal,
                            _settings.dynamicViscosity +
                                GetTurbulentViscosity(iFace, iG));
                    }
                    contribution = -totalFlux * _vfv->GetFaceJacobiDet(iFace, iG);
                });

            rhs[faceToCell[0]] += integratedFlux / _vfv->GetCellVol(faceToCell[0]);
            if (faceToCell[1] != UnInitIndex)
                rhs[faceToCell[1]] -= integratedFlux / _vfv->GetCellVol(faceToCell[1]);
        }
    }

    template <int gDim>
    void ACMEvaluator<gDim>::EvaluateDiagonalJacobian(
        TDof &u,
        std::vector<Matrix5> &diagonal,
        real time)
    {
        ScalarField timeSteps(static_cast<std::size_t>(_mesh->NumCell()), 1.0);
        FaceJacobianField faces;
        AssembleImplicitLinearization(u, timeSteps, diagonal, faces, time);
        for (index i = 0; i < _mesh->NumCell(); ++i)
            diagonal[static_cast<std::size_t>(i)] =
                GammaLocal(u[i], _settings.beta2, _settings.alpha) - diagonal[static_cast<std::size_t>(i)];
    }

    template <int gDim>
    real ACMEvaluator<gDim>::EvaluateTimeStep(
        ScalarField &pseudoTimeStep,
        TDof &u,
        real cfl,
        real maximumTimeStep,
        bool useLocalTimeStep,
        real time)
    {
        DNDS_check_throw_info(std::isfinite(cfl) && cfl > 0, "ACM CFL must be finite and positive");
        DNDS_check_throw_info(std::isfinite(maximumTimeStep) && maximumTimeStep > 0,
                              "ACM maximum pseudo-time step must be finite and positive");
        u.trans.startPersistentPull();
        u.trans.waitPersistentPull();
        if (_prepareTurbulence)
            _prepareTurbulence(u, time);

        std::vector<real> faceSpectralRadius(static_cast<std::size_t>(_mesh->NumFaceProc()), 0.0);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < _mesh->NumFaceProc(); iFace++)
        {
            const auto faceToCell = _mesh->face2cell[iFace];
            const Vector3 unitNormal = ToVector3(_vfv->GetFaceNorm(iFace, -1));
            const Vector3 point = ToVector3(_vfv->GetFaceQuadraturePPhys(iFace, -1));
            State left = u[faceToCell[0]];
            State right;
            if (faceToCell[1] != UnInitIndex)
            {
                right = u[faceToCell[1]];
                Eigen::RowVector<real, 5> rightRow = right.transpose();
                _vfv->ApplyPeriodicTransform(1, _mesh->GetFaceZone(iFace), rightRow);
                right = rightRow.transpose();
            }
            else
                right = GenerateBoundaryForFace(
                    iFace, left, unitNormal, point, time);

            const real normalVelocityLeft = Velocity(left).dot(unitNormal);
            const real normalVelocityRight = Velocity(right).dot(unitNormal);
            const real convectiveRadius = std::max(
                ComputeEigenvalues(normalVelocityLeft, left(0), _settings.beta2, _settings.alpha)
                    .SpectralRadius(),
                ComputeEigenvalues(normalVelocityRight, right(0), _settings.beta2, _settings.alpha)
                    .SpectralRadius());

            real viscousRadius = 0;
            if (_settings.enableViscousFlux && _settings.dynamicViscosity > 0)
            {
                const real kinematicViscosity =
                    (_settings.dynamicViscosity +
                     GetTurbulentViscosity(iFace, -1)) /
                    std::min(left(0), right(0));
                const real area = _vfv->GetFaceArea(iFace);
                const real leftVolume = _vfv->GetCellVol(faceToCell[0]);
                const real rightVolume = faceToCell[1] == UnInitIndex
                                             ? leftVolume
                                             : _vfv->GetCellVol(faceToCell[1]);
                viscousRadius = (4.0 / 3.0) * kinematicViscosity * area *
                                (1.0 / leftVolume + 1.0 / rightVolume);
            }
            faceSpectralRadius[static_cast<std::size_t>(iFace)] = convectiveRadius + viscousRadius;
        }

        pseudoTimeStep.resize(static_cast<std::size_t>(_mesh->NumCell()));
        real localMinimum = maximumTimeStep;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime) reduction(min : localMinimum)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            real cellSpectralSum = 0;
            for (const index iFace : _mesh->cell2face[iCell])
                cellSpectralSum += faceSpectralRadius[static_cast<std::size_t>(iFace)] *
                                   _vfv->GetFaceArea(iFace);
            const real dt = std::min(
                cfl * _vfv->GetCellVol(iCell) * _vfv->GetCellSmoothScaleRatio(iCell) /
                    (cellSpectralSum + 1e-100),
                maximumTimeStep);
            pseudoTimeStep[static_cast<std::size_t>(iCell)] = dt;
            localMinimum = std::min(localMinimum, dt);
        }

        real globalMinimum = localMinimum;
        MPI::Allreduce(
            &localMinimum,
            &globalMinimum,
            1,
            DNDS_MPI_REAL,
            MPI_MIN,
            _mesh->getMPI().comm);
        if (!useLocalTimeStep)
            std::fill(pseudoTimeStep.begin(), pseudoTimeStep.end(), globalMinimum);
        return globalMinimum;
    }

    template <int gDim>
    /** @copydoc ACMEvaluator::AssembleImplicitLinearization */
    void ACMEvaluator<gDim>::AssembleImplicitLinearization(
        TDof &u,
        const ScalarField &pseudoTimeStep,
        MatrixField &diagonal,
        FaceJacobianField &faceJacobians,
        real time)
    {
        DNDS_check_throw_info(
            pseudoTimeStep.size() == static_cast<std::size_t>(_mesh->NumCell()),
            "ACM implicit time-step field does not match owned cells");
        u.trans.startPersistentPull();
        u.trans.waitPersistentPull();
        if (_prepareTurbulence)
            _prepareTurbulence(u, time);

        faceJacobians.assign(
            static_cast<std::size_t>(_mesh->NumFaceProc()),
            FaceJacobianBlocks{});
        constexpr real relativeEpsilon = 1e-7;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < _mesh->NumFaceProc(); iFace++)
        {
            const auto faceToCell = _mesh->face2cell[iFace];
            const State leftBase = u[faceToCell[0]];
            const State rightNativeBase = faceToCell[1] == UnInitIndex
                                              ? State::Zero()
                                              : State(u[faceToCell[1]]);
            const auto quadrature = _vfv->GetFaceQuad(iFace);
            FaceJacobianBlocks integrated;

            for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
            {
                const Vector3 unitNormal = ToVector3(_vfv->GetFaceNorm(iFace, iG));
                const Vector3 point = ToVector3(_vfv->GetFaceQuadraturePPhys(iFace, iG));
                const Vector3 leftCenter = ToVector3(
                    _vfv->GetCellQuadraturePPhys(faceToCell[0], -1));
                Vector3 centerDisplacement;
                if (faceToCell[1] != UnInitIndex)
                    centerDisplacement = ToVector3(
                        _vfv->GetOtherCellPointFromCell(
                            faceToCell[0],
                            faceToCell[1],
                            iFace,
                            0,
                            _vfv->GetCellQuadraturePPhys(faceToCell[1], -1)) -
                        _vfv->GetCellQuadraturePPhys(faceToCell[0], -1));
                else
                    centerDisplacement =
                        2 * unitNormal * (point - leftCenter).dot(unitNormal);

                const auto transformRight = [&](const State &rightNative)
                {
                    State transformed = rightNative;
                    Eigen::RowVector<real, 5> row = transformed.transpose();
                    _vfv->ApplyPeriodicTransform(1, _mesh->GetFaceZone(iFace), row);
                    return State(row.transpose());
                };
                const auto evaluateFlux = [&](const State &left, const State &right)
                {
                    State flux = InviscidFaceFlux(left, right, unitNormal, iFace, iG, time)
                                     .flux;
                    if (_settings.enableViscousFlux)
                    {
                        const Eigen::Matrix<real, 3, 5> zeroGradient =
                            Eigen::Matrix<real, 3, 5>::Zero();
                        const Eigen::Matrix<real, 3, 5> jumpGradient =
                            CorrectedFaceGradient(
                                zeroGradient,
                                zeroGradient,
                                PrimitiveState(left),
                                PrimitiveState(right),
                                centerDisplacement,
                                unitNormal);
                        flux -= ViscousFlux(
                            jumpGradient,
                            unitNormal,
                            _settings.dynamicViscosity +
                                GetTurbulentViscosity(iFace, iG));
                    }
                    return flux;
                };

                Matrix5 leftJacobian = Matrix5::Zero();
                Matrix5 rightJacobian = Matrix5::Zero();
                for (int variable = 0; variable < nVarsFixed; variable++)
                {
                    const real epsilonLeft =
                        relativeEpsilon * (variable == 0 ? leftBase(0) : std::max(1.0, std::abs(leftBase(variable))));
                    State leftPlus = leftBase;
                    State leftMinus = leftBase;
                    leftPlus(variable) += epsilonLeft;
                    leftMinus(variable) -= epsilonLeft;

                    if (faceToCell[1] == UnInitIndex)
                    {
                        const State ghostPlus = GenerateBoundaryForFace(
                            iFace, leftPlus, unitNormal, point, time);
                        const State ghostMinus = GenerateBoundaryForFace(
                            iFace, leftMinus, unitNormal, point, time);
                        leftJacobian.col(variable) =
                            (evaluateFlux(leftPlus, ghostPlus) -
                             evaluateFlux(leftMinus, ghostMinus)) /
                            (2.0 * epsilonLeft);
                    }
                    else
                    {
                        const State rightBase = transformRight(rightNativeBase);
                        leftJacobian.col(variable) =
                            (evaluateFlux(leftPlus, rightBase) -
                             evaluateFlux(leftMinus, rightBase)) /
                            (2.0 * epsilonLeft);

                        const real epsilonRight =
                            relativeEpsilon * (variable == 0 ? rightNativeBase(0) : std::max(1.0, std::abs(rightNativeBase(variable))));
                        State rightPlus = rightNativeBase;
                        State rightMinus = rightNativeBase;
                        rightPlus(variable) += epsilonRight;
                        rightMinus(variable) -= epsilonRight;
                        rightJacobian.col(variable) =
                            (evaluateFlux(leftBase, transformRight(rightPlus)) -
                             evaluateFlux(leftBase, transformRight(rightMinus))) /
                            (2.0 * epsilonRight);
                    }
                }

                const real integrationFactor =
                    quadrature.GetWeight(iG) * _vfv->GetFaceJacobiDet(iFace, iG);
                integrated.left += integrationFactor * leftJacobian;
                integrated.right += integrationFactor * rightJacobian;
            }
            faceJacobians[static_cast<std::size_t>(iFace)] = integrated;
        }

        diagonal.resize(static_cast<std::size_t>(_mesh->NumCell()));
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            DNDS_check_throw_info(
                std::isfinite(pseudoTimeStep[static_cast<std::size_t>(iCell)]) &&
                    pseudoTimeStep[static_cast<std::size_t>(iCell)] > 0,
                "ACM implicit pseudo-time step must be finite and positive");
            Matrix5 block = GammaLocal(
                                u[iCell],
                                _settings.beta2,
                                _settings.alpha) /
                            pseudoTimeStep[static_cast<std::size_t>(iCell)];
            const real inverseVolume = 1.0 / _vfv->GetCellVol(iCell);
            const auto cellFaces = _mesh->cell2face[iCell];
            for (rowsize iCellFace = 0; iCellFace < cellFaces.size(); iCellFace++)
            {
                const index iFace = cellFaces[iCellFace];
                const index otherCell = _mesh->CellFaceOther(iCell, iFace, iCellFace);
                const rowsize if2c =
                    _mesh->CellIsFaceBack(iCell, iFace, iCellFace) ? 0 : 1;
                const auto &faceBlock = faceJacobians[static_cast<std::size_t>(iFace)];
                if (if2c == 0)
                    block += inverseVolume * faceBlock.left;
                else
                    block -= inverseVolume * faceBlock.right;

                // A self-periodic face couples the opposite incidence back to
                // the same algebraic unknown.  Store that contribution in the
                // diagonal so block Jacobi and LU-SGS see the same local block
                // as the complete matrix-vector product.
                if (otherCell == iCell)
                    block += if2c == 0
                                 ? inverseVolume * faceBlock.right
                                 : -inverseVolume * faceBlock.left;
            }
            DNDS_check_throw_info(block.allFinite(), "ACM implicit diagonal block is non-finite");
            diagonal[static_cast<std::size_t>(iCell)] = block;
        }
    }

    template <int gDim>
    /** @copydoc ACMEvaluator::ApplyImplicitLinearization */
    void ACMEvaluator<gDim>::ApplyImplicitLinearization(
        TDof &increment,
        const MatrixField &diagonal,
        const FaceJacobianField &faceJacobians,
        TDof &result) const
    {
        DNDS_check_throw_info(
            diagonal.size() == static_cast<std::size_t>(_mesh->NumCell()) &&
                faceJacobians.size() == static_cast<std::size_t>(_mesh->NumFaceProc()),
            "ACM implicit operator storage has an invalid size");
        increment.trans.startPersistentPull();
        increment.trans.waitPersistentPull();
        result.setConstant(0.0);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            State value = diagonal[static_cast<std::size_t>(iCell)] * increment[iCell];
            const real inverseVolume = 1.0 / _vfv->GetCellVol(iCell);
            const auto cellFaces = _mesh->cell2face[iCell];
            for (rowsize iCellFace = 0; iCellFace < cellFaces.size(); iCellFace++)
            {
                const index iFace = cellFaces[iCellFace];
                const index otherCell = _mesh->CellFaceOther(iCell, iFace, iCellFace);
                // Self-periodic coupling is already folded into diagonal during
                // assembly; treating it again as off-diagonal would double count it.
                if (otherCell == UnInitIndex || otherCell == iCell)
                    continue;
                const rowsize if2c =
                    _mesh->CellIsFaceBack(iCell, iFace, iCellFace) ? 0 : 1;
                const auto &faceBlock = faceJacobians[static_cast<std::size_t>(iFace)];
                const Matrix5 coupling = if2c == 0
                                             ? inverseVolume * faceBlock.right
                                             : -inverseVolume * faceBlock.left;
                value += coupling * increment[otherCell];
            }
            result[iCell] = value;
        }
    }

    template <int gDim>
    /** @copydoc ACMEvaluator::ApplyBlockJacobi */
    void ACMEvaluator<gDim>::ApplyBlockJacobi(
        const TDof &rhs,
        const MatrixField &diagonal,
        TDof &result) const
    {
        DNDS_check_throw_info(
            diagonal.size() == static_cast<std::size_t>(_mesh->NumCell()),
            "ACM block-Jacobi diagonal has an invalid size");
        result.setConstant(0.0);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            const auto decomposition =
                diagonal[static_cast<std::size_t>(iCell)].partialPivLu();
            result[iCell] = decomposition.solve(State(rhs[iCell]));
            DNDS_check_throw_info(result[iCell].allFinite(),
                                  "ACM block-Jacobi solve produced a non-finite value");
        }
    }

    template <int gDim>
    /** @copydoc ACMEvaluator::SolveLUSGS */
    void ACMEvaluator<gDim>::SolveLUSGS(
        const TDof &rhs,
        const MatrixField &diagonal,
        const FaceJacobianField &faceJacobians,
        TDof &result,
        int nSweeps)
    {
        DNDS_check_throw_info(nSweeps > 0, "ACM LU-SGS requires at least one sweep");
        DNDS_check_throw_info(
            rhs.father.get() != result.father.get(),
            "ACM LU-SGS right-hand side and result must not alias");
        result.setConstant(0.0);
        _lusgsCorrection.setConstant(0.0);
        const index nOwned = _mesh->NumCell();

        for (int sweep = 0; sweep < nSweeps; sweep++)
        {
            if (sweep == 0)
            {
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
                for (index iCell = 0; iCell < nOwned; iCell++)
                    _lusgsCorrection[iCell] = rhs[iCell];
            }
            else
            {
                ApplyImplicitLinearization(
                    result,
                    diagonal,
                    faceJacobians,
                    _lusgsOperatorProduct);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
                for (index iCell = 0; iCell < nOwned; iCell++)
                    _lusgsCorrection[iCell] =
                        rhs[iCell] - _lusgsOperatorProduct[iCell];
            }

            for (index iCell = 0; iCell < nOwned; iCell++)
            {
                State value = _lusgsCorrection[iCell];
                const real inverseVolume = 1.0 / _vfv->GetCellVol(iCell);
                const auto cellFaces = _mesh->cell2face[iCell];
                for (rowsize iCellFace = 0; iCellFace < cellFaces.size(); iCellFace++)
                {
                    const index iFace = cellFaces[iCellFace];
                    const index otherCell = _mesh->CellFaceOther(iCell, iFace, iCellFace);
                    if (otherCell == UnInitIndex || otherCell >= nOwned || otherCell >= iCell)
                        continue;
                    const rowsize if2c =
                        _mesh->CellIsFaceBack(iCell, iFace, iCellFace) ? 0 : 1;
                    const auto &faceBlock = faceJacobians[static_cast<std::size_t>(iFace)];
                    const Matrix5 coupling = if2c == 0
                                                 ? inverseVolume * faceBlock.right
                                                 : -inverseVolume * faceBlock.left;
                    value -= coupling * _lusgsCorrection[otherCell];
                }
                _lusgsCorrection[iCell] =
                    diagonal[static_cast<std::size_t>(iCell)].partialPivLu().solve(value);
            }

            for (index iScan = nOwned; iScan > 0; iScan--)
            {
                const index iCell = iScan - 1;
                State correction = State::Zero();
                const real inverseVolume = 1.0 / _vfv->GetCellVol(iCell);
                const auto cellFaces = _mesh->cell2face[iCell];
                for (rowsize iCellFace = 0; iCellFace < cellFaces.size(); iCellFace++)
                {
                    const index iFace = cellFaces[iCellFace];
                    const index otherCell = _mesh->CellFaceOther(iCell, iFace, iCellFace);
                    if (otherCell == UnInitIndex || otherCell >= nOwned || otherCell <= iCell)
                        continue;
                    const rowsize if2c =
                        _mesh->CellIsFaceBack(iCell, iFace, iCellFace) ? 0 : 1;
                    const auto &faceBlock = faceJacobians[static_cast<std::size_t>(iFace)];
                    const Matrix5 coupling = if2c == 0
                                                 ? inverseVolume * faceBlock.right
                                                 : -inverseVolume * faceBlock.left;
                    correction -= coupling * _lusgsCorrection[otherCell];
                }
                _lusgsCorrection[iCell] += diagonal[static_cast<std::size_t>(iCell)]
                                               .partialPivLu()
                                               .solve(correction);
            }

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < nOwned; iCell++)
                result[iCell] += _lusgsCorrection[iCell];
        }
        result.trans.startPersistentPull();
        result.trans.waitPersistentPull();
    }
}
