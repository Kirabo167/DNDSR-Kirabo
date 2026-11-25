#pragma once

#include "Geom/BaseFunction.hpp"

namespace DNDS::ACM
{
    template <int gDim>
    void ACMTurbulenceTransport<gDim>::ReconstructVariational(TFlowDof &flow, real time)
    {
        if (!_scalarVFV)
        {
            _scalarVFV = std::make_shared<TVFV>(_mesh->getMPI(), _mesh);
            _scalarVFV->parseSettings(_vfv->getSettings());
            _scalarVFV->SetPeriodicTransformations(); // k, omega and nuTilde are scalars.
            _scalarVFV->ConstructMetrics();
            _scalarVFV->ConstructBaseAndWeight([](Geom::t_index zone, int order)
                { return Geom::FaceIDIsPeriodic(zone) || order == 0 ? 1.0 : 0.0; });
            _scalarVFV->ConstructRecCoeff();
            _scalarVFV->BuildURec(_qRec, 2);
            _scalarVFV->BuildURec(_qRecWork, 2);
            _vfv->BuildURec(_flowRec, 4);
            _vfv->BuildURec(_flowRecWork, 4);
        }
        typename TVFV::template TFBoundary<2> qBoundary =
            [this, &flow](const TurbulenceState &value, const TurbulenceState &, index cell,
                         index face, int, const Geom::tPoint &normal, const Geom::tPoint &, Geom::t_index)
            {
                if (_boundaryHandler->GetTypeFromID(_mesh->GetFaceZone(face)) == BoundaryType::BCFar &&
                    State(flow[cell]).template head<3>().dot(ToVector3(normal)) >= 0)
                    return value;
                return GenerateBoundary(face, value, cell);
            };
        typename TVFV::template TFBoundary<4> flowBoundary =
            [this, time](const State &value, const State &, index, index face, int,
                         const Geom::tPoint &normal, const Geom::tPoint &point, Geom::t_index)
            { return FlowBoundaryState(face, value, ToVector3(normal), _flowSettings, ToVector3(point), time); };
        const auto reconstruct = [&](auto &engine, auto &mean, auto &rec, auto &work, const auto &boundary, int variables)
        {
            // Reset each evaluation: the spatial residual must not depend on a
            // warm-start history, especially during finite-difference J*v probes.
            rec.setConstant(0.0);
            rec.trans.startPersistentPull();
            rec.trans.waitPersistentPull();
            Eigen::VectorXd localScale = Eigen::VectorXd::Zero(variables), scale(variables);
            for (index cell = 0; cell < _mesh->NumCell(); ++cell)
                for (int variable = 0; variable < variables; ++variable)
                    localScale(variable) = std::max(localScale(variable), std::abs(mean[cell](variable)));
            MPI::Allreduce(localScale.data(), scale.data(), variables, DNDS_MPI_REAL, MPI_MAX, _mesh->getMPI().comm);
            scale = scale.cwiseMax(real(1e-30));
            if (variables == 4)
            {
                const real velocityScale = std::max(scale.head<3>().norm(),
                    std::sqrt(_flowSettings.beta2 / _flowSettings.rho0));
                scale.head<3>().setConstant(velocityScale);
                scale(3) = std::max(scale(3), _flowSettings.beta2);
            }
            else
                for (int variable = 0; variable < variables; ++variable)
                    scale(variable) = std::max(scale(variable), variable < ActiveVariableCount()
                        ? _turbulenceSettings.minimumValue[variable] : real(1));
            bool converged = false;
            real finalDefect = 0;
            for (int iteration = 0; iteration <= _turbulenceSettings.variationalMaxIterations; ++iteration)
            {
                engine.DoReconstructionIter(rec, work, mean, boundary, true, true);
                real local = 0, global = 0;
                for (index cell = 0; cell < _mesh->NumCell(); ++cell)
                    for (int variable = 0; variable < variables; ++variable)
                        local = std::max(local, work[cell].col(variable).cwiseAbs().maxCoeff() / scale(variable));
                MPI::Allreduce(&local, &global, 1, DNDS_MPI_REAL, MPI_MAX, _mesh->getMPI().comm);
                finalDefect = global;
                if (global <= _turbulenceSettings.variationalTolerance)
                { converged = true; break; }
                if (iteration == _turbulenceSettings.variationalMaxIterations)
                    break;
                rec.addTo(work, -0.7);
                rec.trans.startPersistentPull();
                rec.trans.waitPersistentPull();
            }
            DNDS_check_throw_info(converged, fmt::format("ACM RANS {} variational reconstruction did not converge: defect={:.6e}",
                variables == 4 ? "flow" : "turbulence", finalDefect));
        };
        reconstruct(*_scalarVFV, _state, _qRec, _qRecWork, qBoundary, 2);
        reconstruct(*_vfv, flow, _flowRec, _flowRecWork, flowBoundary, 4);
    }

    template <int gDim>
    TurbulenceGradient ACMTurbulenceTransport<gDim>::PointTurbulenceGradient(
        index cell, index face, rowsize side, int point) const
    {
        if (!_turbulenceSettings.variationalReconstruction)
            return CellTurbulenceGradient(cell);
        TurbulenceGradient gradient = TurbulenceGradient::Zero();
        gradient.template topRows<gDim>() = _scalarVFV->GetIntPointDiffBaseValue(
            cell, face, side, point, Eigen::seq(Eigen::fix<1>, Eigen::fix<gDim>), gDim + 1) * _qRec[cell];
        for (int variable = 0; variable < 2; ++variable)
            gradient.col(variable) *= _limiter[cell](variable);
        return gradient;
    }

    template <int gDim>
    Eigen::Matrix<real, 3, 4> ACMTurbulenceTransport<gDim>::PointFlowGradient(
        index cell, index face, rowsize side, int point) const
    {
        Eigen::Matrix<real, 3, 4> gradient = Eigen::Matrix<real, 3, 4>::Zero();
        if (_turbulenceSettings.variationalReconstruction)
            gradient.template topRows<gDim>() = _vfv->GetIntPointDiffBaseValue(
                cell, face, side, point, Eigen::seq(Eigen::fix<1>, Eigen::fix<gDim>), gDim + 1) * _flowRec[cell];
        else
            gradient.template topRows<gDim>() = _flowGradient[cell];
        return gradient;
    }

    template <int gDim>
    SASKinematics ACMTurbulenceTransport<gDim>::PointSASKinematics(index cell, int point, real filterWidth) const
    {
        const auto &secondDerivatives = std::get<2>(Geom::Base::getDiffOperatorIJK2I<gDim>());
        const int diffCount = Geom::Base::ndiffSizS<gDim>(2);
        std::array<int, gDim> traceRows{};
        for (int axis = 0; axis < gDim; ++axis)
            traceRows[axis] = secondDerivatives[axis][axis];
        const MatrixXR secondBasis = _vfv->GetIntPointDiffBaseValue(cell, -1, -1, point, traceRows, diffCount);
        const Eigen::RowVector3d laplacian = secondBasis.colwise().sum() * _flowRec[cell].template leftCols<3>();
        return {laplacian.stableNorm(), filterWidth};
    }
}
