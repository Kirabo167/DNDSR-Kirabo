#pragma once
#include "ACMSolver.hpp"
#include "GPU/ResidentACM.hpp"
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <fstream>

namespace DNDS::ACMGPU
{
    template <int gDim>
    GPU::ResidentACMData ACMEvaluator<gDim>::ExportResidentData(TDof &u)
    {
        GPU::ResidentACMData data;
        auto &s = data.settings;
        s.dimension = gDim;
        s.flow.model = 1;
        s.flow.dimension = gDim;
        s.flow.rho0 = _settings.rho0;
        s.flow.beta2 = _settings.beta2;
        s.flow.alpha = _settings.alpha;
        s.flow.entropyRatio = _settings.entropyFixRatio;
        s.flow.solver = _settings.riemannSolverType == RiemannSolverType::Rusanov ? 1 : 0;
        s.viscous = _settings.enableViscousFlux;
        s.viscosity = _settings.dynamicViscosity;
        s.reconstruction = _reconstructionSettings.type == ReconstructionType::FirstOrder ? 0 : _reconstructionSettings.type == ReconstructionType::GreenGauss ? 1
                                                                                                                                                               : 2;
        s.limiter = _reconstructionSettings.enableLimiter;
        s.limiterType = _reconstructionSettings.limiterType == LimiterType::LocalExtrema ? 0 : _reconstructionSettings.limiterType == LimiterType::CWBAP ? 2
                                                                                                                                                         : 1;
        s.sweeps = _reconstructionSettings.variationalIterations;
        s.resetCoefficients = _reconstructionSettings.resetVariationalCoefficients;
        s.controlled = _reconstructionSettings.variationalTolerance > 0;
        s.maximumSweeps = _reconstructionSettings.variationalMaxIterations;
        s.reconstructionTolerance = _reconstructionSettings.variationalTolerance;
        s.checkInterval = _reconstructionSettings.variationalCheckInterval;
        s.reconstructionRelaxation = _reconstructionSettings.variationalRelaxation;
        s.reconstructionGMRES = _reconstructionSettings.variationalUseGMRES;
        s.sor = _vfv->getSettings().SORInstead;
        data.owned = _mesh->NumCell();
        data.cells.resize(_mesh->NumCellProc());
        data.globals.resize(data.cells.size());
        data.initialState.resize(data.cells.size() * 4);
        std::size_t nLinks = 0, nCoupling = 0, nPoints = 0;
        for (index cell = 0; cell < _mesh->NumCell(); ++cell)
        {
            const auto cellFaces = _mesh->cell2face[cell];
            nLinks += cellFaces.size();
            if (s.reconstruction == 2)
                for (rowsize j = 0; j < cellFaces.size(); ++j)
                {
                    const index other = _mesh->CellFaceOther(cell, cellFaces[j], j);
                    if (other != UnInitIndex)
                        nCoupling += (_vfv->GetCellAtr(cell).NDOF - 1) * (_vfv->GetCellAtr(other).NDOF - 1);
                }
        }
        for (index face = 0; face < _mesh->NumFaceProc(); ++face)
            nPoints += _vfv->GetFaceQuad(face).GetNumPoints();
        log() << "ACM GPU static export rank " << _mesh->getMPI().rank << ": cells=" << data.cells.size()
              << " links=" << nLinks << " quadraturePoints=" << nPoints << " payloadGiB="
              << (data.cells.size() * sizeof(GPU::ResidentACMCell) + nLinks * sizeof(GPU::ResidentACMLink) +
                  nCoupling * sizeof(double) + nPoints * sizeof(GPU::ResidentACMPoint)) / double(std::uint64_t(1) << 30) << std::endl;
        data.links.reserve(nLinks);
        data.coupling.reserve(nCoupling);
        data.points.reserve(nPoints);
        const auto rotation = [&](int side, Geom::t_index zone, double *out)
        {
            for (int col = 0; col < 3; ++col)
            {
                Eigen::RowVector<real, 4> row = Eigen::RowVector<real, 4>::Zero();
                row(col) = 1;
                _vfv->ApplyPeriodicTransform(side, zone, row);
                for (int r = 0; r < 3; ++r)
                    out[r * 3 + col] = row(r);
            }
        };
        DNDS_check_throw_info(_vfv->getSettings().functionalSettings.greenGauss1Weight == 0,
                              "Resident ACM Green-Gauss augmented variational functional is not yet available; CPU fallback is disabled");
        for (index cell = 0; cell < _mesh->NumCellProc(); ++cell)
        {
            auto &out = data.cells[cell];
            data.globals[cell] = _mesh->CellIndexLocal2Global(cell);
            out.volume = _vfv->GetCellVol(cell);
            out.smoothScale = _vfv->GetCellSmoothScaleRatio(cell);
            out.relaxation = _vfv->GetCellAtr(cell).relax;
            out.modes = _vfv->GetCellAtr(cell).NDOF - 1;
            DNDS_check_throw_info(out.modes <= GPU::ACMMaximumModes, "ACM device polynomial mode capacity exceeded");
            for (int k = 0; k < 4; ++k)
                data.initialState[cell * 4 + k] = u[cell](k);
            if (cell >= _mesh->NumCell())
                continue;
            out.begin = data.links.size();
            const auto cellFaces = _mesh->cell2face[cell];
            for (rowsize j = 0; j < cellFaces.size(); ++j)
            {
                const index face = cellFaces[j], other = _mesh->CellFaceOther(cell, face, j);
                const int side = _mesh->CellIsFaceBack(cell, face, j) ? 0 : 1;
                const auto zone = _mesh->GetFaceZone(face);
                GPU::ResidentACMLink link;
                link.neighbor = other == UnInitIndex ? -1 : other;
                link.face = face;
                link.side = side;
                rotation(side, zone, link.rotation);
                {
                    const auto center = _vfv->GetFaceQuadraturePPhysFromCell(face, cell, side, -1);
                    const double ownLength = (center - _vfv->GetCellQuadraturePPhys(cell, -1)).norm();
                    const double otherLength = other == UnInitIndex ? ownLength :
                        (_vfv->GetOtherCellPointFromCell(cell, other, face, side,
                            _vfv->GetCellQuadraturePPhys(other, -1)) - center).norm();
                    const double interpolation = other == UnInitIndex ? .5 : ownLength / (ownLength + otherLength + verySmallReal);
                    const auto normal = _vfv->GetFaceNormFromCell(face, cell, side, -1);
                    for (int d = 0; d < gDim; ++d)
                        link.turbulenceWeight[d] = interpolation * _vfv->GetFaceArea(face) * normal(d) * (side ? -1 : 1) / out.volume;
                }
                if (s.reconstruction == 2 && other != UnInitIndex)
                {
                    const auto matrix = _vfv->GetCellRecMatAInvB(cell, j);
                    const auto weight = _vfv->get_vectorAInvB()(cell, j);
                    link.coupling = data.coupling.size();
                    for (int row = 0; row < matrix.rows(); ++row)
                    {
                        link.meanWeight[row] = weight(row, 0);
                        for (int col = 0; col < matrix.cols(); ++col)
                            data.coupling.push_back(matrix(row, col));
                    }
                }
                if (s.reconstruction == 2 && other == UnInitIndex)
                {
                    link.boundaryBegin = data.boundaries.size();
                    const auto &bc = _boundaryHandler->GetConditionFromID(zone);
                    const auto inverse = _vfv->GetCellRecMatAInv(cell);
                    const auto &settings = _vfv->getSettings();
                    auto quadrature = Geom::Elem::Quadrature(_mesh->GetFaceElement(face), settings.intOrderVRBCValue());
                    Geom::tSmallCoords coordinates;
                    _mesh->GetCoordsOnFace(face, coordinates);
                    double unused = 0;
                    quadrature.Integration(unused, [&](auto &increment, int q, const Geom::tPoint &, const Geom::Elem::tD01Nj &derivatives)
                                           {
                        GPU::ResidentACMBoundarySample sample; sample.type = static_cast<int>(bc.type);
                        Geom::tPoint point, normal; double jacobian;
                        Eigen::RowVectorXd basis;
                        if (settings.intOrderVRBCIsSame())
                        {
                            point = _vfv->GetFaceQuadraturePPhysFromCell(face, cell, 0, q);
                            normal = _vfv->GetFaceNorm(face, q); jacobian = _vfv->GetFaceJacobiDet(face, q);
                            basis = _vfv->GetIntPointDiffBaseValue(cell, face, 0, q, std::array<int, 1>{0}, 1);
                        }
                        else
                        {
                            point = Geom::Elem::PPhysicsCoordD01Nj(coordinates, derivatives);
                            const auto jac = Geom::Elem::ShapeJacobianCoordD01Nj(coordinates, derivatives);
                            jacobian = Geom::Elem::JacobiDetFace<gDim>(jac);
                            normal = Geom::FacialJacobianToNormVec<gDim>(jac).normalized();
                            point = _vfv->GetFacePointFromCell(face, cell, 0, point);
                            Eigen::MatrixXd dbv(1, out.modes + 1); _vfv->FDiffBaseValue(dbv, point, cell, face, -2, 0);
                            basis = dbv.rightCols(out.modes);
                        }
                        for (int d = 0; d < 3; ++d) sample.normal[d] = normal(d);
                        for (int k = 0; k < 4; ++k) sample.prescribed[k] = bc.value[k];
                        const Eigen::VectorXd response = inverse * _vfv->FFaceFunctional(basis, Eigen::MatrixXd::Ones(1, 1), face, cell, cell) *
                            (quadrature.GetWeight(q) * jacobian);
                        for (int m = 0; m < out.modes; ++m) { sample.basis[m] = basis(m); sample.response[m] = response(m); }
                        data.boundaries.push_back(sample); increment = 0; });
                    link.boundaryEnd = data.boundaries.size();
                }
                if (s.reconstruction == 1)
                {
                    const auto center = _vfv->GetFaceQuadraturePPhysFromCell(face, cell, side, -1);
                    double leftDistance = (center - _vfv->GetCellBary(cell)).norm(), rightDistance = leftDistance;
                    if (other != UnInitIndex)
                        rightDistance = (_vfv->GetOtherCellBaryFromCell(cell, other, face, side) - center).norm();
                    if (_vfv->getSettings().subs2ndOrderGGScheme == 0)
                        leftDistance = rightDistance = 1;
                    const auto normal = _vfv->GetFaceNormFromCell(face, cell, side, -1);
                    const double scale = other == UnInitIndex ? .5 : leftDistance / (leftDistance + rightDistance + verySmallReal) * (side ? -1 : 1);
                    for (int d = 0; d < gDim; ++d)
                        link.gradientWeight[d] = scale * _vfv->GetFaceArea(face) * normal(d) / out.volume;
                }
                data.links.push_back(link);
            }
            out.end = data.links.size();
        }
        data.faces.resize(_mesh->NumFaceProc());
        for (index face = 0; face < _mesh->NumFaceProc(); ++face)
        {
            auto &out = data.faces[face];
            const auto cells = _mesh->face2cell[face];
            const auto zone = _mesh->GetFaceZone(face);
            out.left = cells[0];
            out.right = cells[1] == UnInitIndex ? -1 : cells[1];
            out.area = _vfv->GetFaceArea(face);
            for (int d = 0; d < 3; ++d)
                out.normal[d] = _vfv->GetFaceNorm(face, -1)(d);
            rotation(1, zone, out.rotation);
            if (out.right < 0)
            {
                const auto &bc = _boundaryHandler->GetConditionFromID(zone);
                out.type = static_cast<int>(bc.type);
                for (int k = 0; k < 4; ++k)
                    out.prescribed[k] = bc.value[k];
            }
            out.begin = data.points.size();
            auto quadrature = _vfv->GetFaceQuad(face);
            for (int q = 0; q < quadrature.GetNumPoints(); ++q)
            {
                GPU::ResidentACMPoint point;
                point.measure = quadrature.GetWeight(q) * _vfv->GetFaceJacobiDet(face, q);
                const auto normal = _vfv->GetFaceNorm(face, q);
                for (int d = 0; d < 3; ++d)
                    point.normal[d] = normal(d);
                for (int side = 0; side < 2; ++side)
                {
                    const index cell = cells[side];
                    if (cell == UnInitIndex)
                        continue;
                    const Vector3 delta = _vfv->GetFaceQuadraturePPhysFromCell(face, cell, side, q) - _vfv->GetCellQuadraturePPhys(cell, -1);
                    for (int d = 0; d < 3; ++d)
                        point.delta[side][d] = delta(d);
                    if (s.reconstruction == 2)
                    {
                        const Eigen::MatrixXd basis = _vfv->GetIntPointDiffBaseValue(cell, face, side, q, std::array<int, 1>{0}, 1);
                        const Eigen::MatrixXd gradient = _vfv->GetIntPointDiffBaseValue(cell, face, side, q, Eigen::seq(Eigen::fix<1>, Eigen::fix<gDim>), gDim + 1);
                        for (int m = 0; m < data.cells[cell].modes; ++m)
                        {
                            point.basis[side][m] = basis(0, m);
                            for (int d = 0; d < gDim; ++d)
                                point.gradient[side][m * 3 + d] = gradient(d, m);
                        }
                    }
                }
                Vector3 displacement;
                const auto leftCenter = _vfv->GetCellQuadraturePPhys(cells[0], -1);
                if (out.right >= 0)
                    displacement = _vfv->GetOtherCellPointFromCell(cells[0], cells[1], face, 0,
                                                                   _vfv->GetCellQuadraturePPhys(cells[1], -1)) -
                                   leftCenter;
                else
                    displacement = 2 * normal * (_vfv->GetFaceQuadraturePPhys(face, q) - leftCenter).dot(normal);
                for (int d = 0; d < 3; ++d)
                    point.displacement[d] = displacement(d);
                data.points.push_back(point);
            }
            out.end = data.points.size();
        }
        std::vector<index> levels(data.owned, 0);
        index maximum = 0;
        for (index cell = 0; cell < data.owned; ++cell)
        {
            for (index j = data.cells[cell].begin; j < data.cells[cell].end; ++j)
                if (data.links[j].neighbor >= 0 && data.links[j].neighbor < cell)
                    levels[cell] = std::max(levels[cell], levels[data.links[j].neighbor] + 1);
            maximum = std::max(maximum, levels[cell]);
        }
        std::vector<std::vector<index>> groups(maximum + 1);
        for (index cell = 0; cell < data.owned; ++cell)
            groups[levels[cell]].push_back(cell);
        data.reconstructionLevels.push_back(0);
        for (const auto &group : groups)
        {
            data.reconstructionNodes.insert(data.reconstructionNodes.end(), group.begin(), group.end());
            data.reconstructionLevels.push_back(data.reconstructionNodes.size());
        }
        return data;
    }

    inline void ConfigureResidentTime(GPU::ResidentACMSettings &s, const TimeMarchSettings &time)
    {
        s.integrator = static_cast<int>(time.integrator);
        s.fixedStep = time.pseudoTimeStep;
        s.cflStep = time.useCFLTimeStep;
        s.localStep = time.useLocalTimeStep;
        s.cfl = time.cfl;
        s.maximumStep = time.maximumPseudoTimeStep;
        s.implicitIterations = time.maxImplicitIterations;
        s.startupSteps = time.dualTimeRANSStartupSteps;
        s.startupIterations = time.dualTimeRANSStartupIterations;
        DNDS_check_throw_info(time.dualTimeRANSContinuationSteps == 0 && time.dualTimeRANSAndersonDepth == 0 &&
                                  !time.dualTimeRANSMatrixFreeJacobian,
                              "Resident CUDA does not yet support RANS continuation, Anderson mixing or matrix-free GMRES; CPU fallback is disabled");
        s.implicitTolerance = time.implicitTolerance;
        s.implicitRelaxation = time.implicitRelaxation;
        s.lusgsSweeps = time.lusgsSweeps;
        s.linearRelativeTolerance = time.linearRelativeTolerance;
        s.linearAbsoluteTolerance = time.linearAbsoluteTolerance;
        s.physicalStep = time.physicalTimeStep;
        s.requireConvergence = time.dualTimeRequireConvergence;
        s.convergenceCriterion = time.dualTimeConvergenceCriterion == "Absolute" ? 0 : time.dualTimeConvergenceCriterion == "Relative" ? 1 : 2;
        DNDS_check_throw_info(time.dualTimeConvergenceCriterion != "RelativeUpdate", "Resident RANS does not support RelativeUpdate");
        s.flowRelativeTolerance = time.dualTimeFlowRelativeTolerance;
        s.turbulenceRelativeTolerance = time.dualTimeTurbulenceRelativeTolerance;
        for (int k = 0; k < 6; ++k)
        {
            s.residualScales[k] = time.dualTimeResidualScales[k];
            s.relativeTolerances[k] = time.dualTimeRelativeTolerances[k];
            s.steadyAbsoluteTolerances[k] = time.steadyConvergenceAbsoluteTolerances[k];
        }
        s.steadyLineSearch = time.steadyLineSearch;
        s.steadyLineSearchWindow = time.steadyLineSearchWindow;
        DNDS_check_throw_info(s.integrator == 2 || (!time.steadyAdaptiveCFL && time.steadyRelativeTolerance == 0),
                              "Resident ACM adaptive steady controls are not available");
    }

    template <int gDim>
    void ACMTurbulenceTransport<gDim>::ExportResidentData(GPU::ResidentACMData &data) const
    {
        DNDS_check_throw_info(_turbulenceSettings.model == TurbulenceModel::KOmegaSST,
                              "Resident RANS currently requires KOmegaSST");
        auto &s = data.settings;
        s.sst = true;
        s.turbulenceSecondOrder = _turbulenceSettings.secondOrderReconstruction;
        s.turbulenceSource = _turbulenceSettings.enableSourceTerms;
        s.eddyRatioMaximum = _turbulenceSettings.maximumEddyViscosityRatio;
        s.wallOmegaCoefficient = _turbulenceSettings.wallOmegaCoefficient;
        s.minimumWallDistance = _turbulenceSettings.minimumWallDistance;
        s.turbulenceTimeScale = _turbulenceSettings.transportTimeScale;
        for (int k = 0; k < 2; ++k)
        {
            s.turbulenceMinimum[k] = _turbulenceSettings.minimumValue[k];
            s.turbulenceMaximum[k] = _turbulenceSettings.maximumValue[k];
            s.turbulenceFar[k] = _turbulenceSettings.FarFieldState()(k);
        }
        data.initialTurbulence.resize(data.cells.size() * 2);
        for (index cell = 0; cell < _mesh->NumCellProc(); ++cell)
        {
            data.cells[cell].wallDistance = _cellWallDistance[cell];
            for (int k = 0; k < 2; ++k)
                data.initialTurbulence[cell * 2 + k] = _state[cell](k);
        }
        for (index face = 0; face < _mesh->NumFaceProc(); ++face)
            data.faces[face].wallDistance = _faceWallDistance[face];
    }

    template <ACMModel model>
    bool ACMSolver<model, true>::TryHostSnapshotAudit()
    {
        if (const char *audit = std::getenv("DNDS_GPU_SST_HOST_AUDIT_FILE"))
        {
            DNDS_check_throw_info(nTurbulence == 2, "Host SST audit requires two turbulence equations");
            DNDS_check_throw_info(_mpi.size == 1, "Host snapshot audit currently requires one MPI rank");
            EvaluateRHS(_rhs, _physicalTime);
            nlohmann::ordered_json report;
            report["backend"] = "host reference, one spatial residual evaluation, no state advance";
            report["cells"] = GetMesh()->NumCell();
            std::array<real, 6> square{};
            std::array<std::vector<index>, 6> worst;
            std::array<index, 2> floorCount{}, floorNegative{};
            std::array<real, 2> floorNegativeSquare{};
            const auto sample = [&](index cell)
            {
                const auto center = GetReconstruction()->GetCellQuadraturePPhys(cell, -1);
                std::array<real, 6> state{}, rhs{};
                for (int k = 0; k < 6; ++k) { state[k] = _u[cell](k); rhs[k] = _rhs[cell](k); }
                return nlohmann::ordered_json{{"cell", cell}, {"center", {center(0), center(1), center(2)}},
                    {"state", state}, {"rhs", rhs}, {"volume", GetReconstruction()->GetCellVol(cell)}};
            };
            for (index cell = 0; cell < GetMesh()->NumCell(); ++cell)
            {
                for (int k = 0; k < 6; ++k)
                {
                    square[k] += _rhs[cell](k) * _rhs[cell](k);
                    auto &ids = worst[k];
                    if (ids.size() < 10 || std::abs(_rhs[cell](k)) > std::abs(_rhs[ids.back()](k)))
                    {
                        ids.push_back(cell);
                        std::sort(ids.begin(), ids.end(), [&](index a, index b) { return std::abs(_rhs[a](k)) > std::abs(_rhs[b](k)); });
                        if (ids.size() > 10) ids.pop_back();
                    }
                }
                for (int k = 0; k < 2; ++k)
                    if (_u[cell](4 + k) <= 1.01 * _configuration.acmSettings.rho0 * _configuration.turbulenceSettings.minimumValue[k])
                    {
                        ++floorCount[k];
                        if (_rhs[cell](4 + k) < 0)
                        {
                            ++floorNegative[k];
                            floorNegativeSquare[k] += _rhs[cell](4 + k) * _rhs[cell](4 + k);
                        }
                    }
            }
            for (int k = 0; k < 6; ++k)
            {
                report["equation_rms"].push_back(std::sqrt(square[k] / GetMesh()->NumCell()));
                report["worst_cells"][k] = nlohmann::ordered_json::array();
                for (index cell : worst[k]) report["worst_cells"][k].push_back(sample(cell));
            }
            report["floor_cells"] = floorCount;
            report["floor_negative_rhs_cells"] = floorNegative;
            for (int k = 0; k < 2; ++k) report["floor_negative_rhs_rms"].push_back(std::sqrt(floorNegativeSquare[k] / GetMesh()->NumCell()));
            std::ofstream stream(audit);
            DNDS_check_throw_info(bool(stream), "Cannot open host snapshot audit output");
            stream << report.dump(2) << std::endl;
            DNDS_check_throw_info(bool(stream), "Cannot write host snapshot audit output");
            log() << "Host SST snapshot audit saved without advancing state: " << audit << std::endl;
            return true;
        }
        return false;
    }

    template <ACMModel model>
    void ACMSolver<model, true>::RunResident()
    {
        DNDS_check_throw_info(nTurbulence == 2 && _configuration.turbulenceSettings.model == TurbulenceModel::KOmegaSST,
                              "Resident conservative RANS requires the two-equation SST model");
        ProjectState();
        auto data = GetEvaluator()->ExportResidentData(_flowSolver.GetState());
        _flowSolver.GetTurbulenceTransport()->ExportResidentData(data);
        ConfigureResidentTime(data.settings, _configuration.timeMarchSettings);
        GPU::ResidentACM resident(data, _mpi.comm);
        const auto &time = _configuration.timeMarchSettings;
        data = GPU::ResidentACMData{};
        if (const char *audit = std::getenv("DNDS_GPU_SST_AUDIT_FILE"))
        {
            DNDS_check_throw_info(_mpi.size == 1, "SST snapshot audit currently requires one MPI rank");
            resident.WriteSSTAudit(audit);
            log() << "SST snapshot audit saved without advancing state: " << audit << std::endl;
            return;
        }
        const auto &output = _configuration.outputSettings;
        if (output.interval > 0 && output.writeInitial)
            WriteOutput(_configuration.restartSettings.completedSteps, _physicalTime);
        if (time.integrator == TimeIntegratorType::ImplicitEulerLUSGS)
        {
            std::array<real, 6> reference{}, targets{};
            int stable = 0;
            real cfl = time.cfl;
            const auto save = [&](int absolute)
            {
                std::vector<double> flow, turbulence;
                resident.Download(flow);
                resident.DownloadTurbulence(turbulence);
                for (index cell = 0; cell < GetMesh()->NumCell(); ++cell)
                {
                    for (int k = 0; k < 4; ++k) _u[cell](k) = flow[cell * 4 + k];
                    for (int k = 0; k < 2; ++k) _u[cell](4 + k) = _configuration.acmSettings.rho0 * turbulence[cell * 2 + k];
                }
                // There is no physical-time advance in a steady solve.
                WriteOutput(absolute, _physicalTime);
            };
            for (int step = 1; time.steadyUntilConverged || step <= time.nSteps; ++step)
            {
                DNDS_check_throw_info(step < std::numeric_limits<int>::max() - _configuration.restartSettings.completedSteps,
                                      "Steady iteration index exhausted; convergence has not been established");
                resident.SetCFL(cfl);
                const auto report = resident.Advance();
                const int absolute = step + _configuration.restartSettings.completedSteps;
                if (step == 1)
                {
                    reference = report.initialEquationNorms;
                    for (int k = 0; k < 6; ++k)
                        targets[k] = std::max(time.steadyConvergenceAbsoluteTolerances[k],
                                              reference[k] * time.steadyConvergenceRelativeTolerance);
                    if (_mpi.rank == 0)
                        log() << "ACM resident steady reference eqInitial=" << nlohmann::ordered_json(reference).dump()
                              << " eqTarget=" << nlohmann::ordered_json(targets).dump()
                              << " window=" << time.steadyConvergenceWindow << std::endl;
                }
                bool all = true;
                real before = 0, after = 0;
                for (int k = 0; k < 6; ++k)
                {
                    all = all && report.equationNorms[k] <= targets[k];
                    const real scale = std::max(reference[k], time.steadyConvergenceAbsoluteTolerances[k]);
                    before += std::pow(report.initialEquationNorms[k] / scale, 2);
                    after += std::pow(report.equationNorms[k] / scale, 2);
                }
                stable = all ? stable + 1 : 0;
                const bool converged = stable >= time.steadyConvergenceWindow;
                real nextCFL = cfl;
                if (time.steadyAdaptiveCFL)
                {
                    if (!report.updateAccepted || report.acceptedRelaxation < .5 * time.implicitRelaxation)
                        nextCFL *= time.steadyCFLReduction;
                    else if (after < before && report.acceptedRelaxation >= .99 * time.implicitRelaxation)
                        nextCFL *= time.steadyCFLGrowth;
                    nextCFL = std::clamp(nextCFL, time.steadyCFLMin, time.steadyCFLMax);
                }
                if (_mpi.rank == 0)
                    log() << std::scientific << std::setprecision(10)
                          << "ACM resident steady iteration " << absolute
                          << " residual " << report.initialNorm << " -> " << report.finalNorm
                          << " dtauMin=" << report.minimumStep << " turbResidual=" << report.turbulenceNorm
                          << " cfl=" << cfl << " nextCFL=" << nextCFL
                          << " relaxation=" << report.acceptedRelaxation
                          << " accepted=" << report.updateAccepted << " stable=" << stable
                          << " converged=" << converged
                          << " eqResidual=" << nlohmann::ordered_json(report.equationNorms).dump() << std::endl;
                if (output.interval > 0 && (absolute % output.interval == 0 || converged || (!time.steadyUntilConverged && step == time.nSteps)))
                    save(absolute);
                if (converged)
                {
                    if (_mpi.rank == 0) log() << "ACM resident steady spatial convergence reached" << std::endl;
                    break;
                }
                cfl = nextCFL;
            }
            resident.Report(_mpi.rank);
            return;
        }
        for (int step = 1; step <= time.nSteps; ++step)
        {
            const auto report = resident.Advance();
            const int absolute = step + _configuration.restartSettings.completedSteps;
            _physicalTime = _configuration.restartSettings.physicalTime + time.physicalTimeStep * step;
            _completedPhysicalSteps = step;
            if (_mpi.rank == 0)
                log() << "ACM resident physical step " << absolute << " time=" << _physicalTime
                      << " BDF" << (step == 1 ? 1 : 2) << " residual " << report.initialNorm << " -> " << report.finalNorm
                      << " dtauMin=" << report.minimumStep << " turbResidual=" << report.turbulenceNorm
                      << " inner=" << report.iterations << " converged=" << report.converged
                      << " eqResidual=" << nlohmann::ordered_json(report.equationNorms).dump() << std::endl;
            if (output.interval > 0 && (absolute % output.interval == 0 || step == time.nSteps))
            {
                std::vector<double> flow, turbulence;
                resident.Download(flow);
                resident.DownloadTurbulence(turbulence);
                for (index cell = 0; cell < GetMesh()->NumCell(); ++cell)
                {
                    for (int k = 0; k < 4; ++k)
                        _u[cell](k) = flow[cell * 4 + k];
                    for (int k = 0; k < 2; ++k)
                        _u[cell](4 + k) = _configuration.acmSettings.rho0 * turbulence[cell * 2 + k];
                }
                WriteOutput(absolute, _physicalTime);
            }
        }
        resident.Report(_mpi.rank);
    }

    template <ACMModel model>
    void ACMSolver<model, false>::RunResident()
    {
        DNDS_check_throw_info(!_turbulence, "Resident ACM turbulence transport is not yet available; CPU fallback is disabled");
        auto data = _evaluator->ExportResidentData(_u);
        auto &s = data.settings;
        const auto &time = _configuration.timeMarchSettings;
        const auto &output = _configuration.outputSettings;
        s.integrator = static_cast<int>(time.integrator);
        s.fixedStep = time.pseudoTimeStep;
        s.cflStep = time.useCFLTimeStep;
        s.localStep = time.useLocalTimeStep;
        s.cfl = time.cfl;
        s.maximumStep = time.maximumPseudoTimeStep;
        s.implicitIterations = time.maxImplicitIterations;
        s.implicitTolerance = time.implicitTolerance;
        s.implicitRelaxation = time.implicitRelaxation;
        s.lusgsSweeps = time.lusgsSweeps;
        s.gmresSubspace = time.gmresSubspace;
        s.gmresRestarts = time.gmresRestarts;
        s.linearRelativeTolerance = time.linearRelativeTolerance;
        s.linearAbsoluteTolerance = time.linearAbsoluteTolerance;
        s.physicalStep = time.physicalTimeStep;
        DNDS_check_throw_info(!time.steadyAdaptiveCFL && time.steadyRelativeTolerance == 0,
                              "Resident ACM adaptive steady controls are not yet available; CPU fallback is disabled");
        GPU::ResidentACM resident(data, _mpi.comm);
        if (output.interval > 0 && output.writeInitial)
            WriteFlowField(_configuration.restartSettings.completedSteps, _configuration.restartSettings.completedSteps);
        for (int step = 1; step <= time.nSteps; ++step)
        {
            const auto report = resident.Advance();
            const int absolute = step + _configuration.restartSettings.completedSteps;
            if (_mpi.rank == 0)
                log() << "ACM resident step " << absolute << " residual " << report.initialNorm
                      << " -> " << report.finalNorm << " dtauMin=" << report.minimumStep << " inner=" << report.iterations << std::endl;
            if (output.interval > 0 && absolute % output.interval == 0)
            {
                std::vector<double> snapshot;
                resident.Download(snapshot);
                for (index cell = 0; cell < _mesh->NumCell(); ++cell)
                    for (int k = 0; k < 4; ++k)
                        _u[cell](k) = snapshot[cell * 4 + k];
                WriteFlowField(absolute, absolute);
            }
        }
        resident.Report(_mpi.rank);
    }
}
