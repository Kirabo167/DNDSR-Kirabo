#pragma once

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include "NCFVLocalLimiterMath.hxx"
#include "NCFVPlanarCompression.hpp"

namespace DNDS::NCFV
{
    // Transient forward-mode jets account for BOTH A(U)*H(U) and dA(U)*G(U)
    // in nonlinear Euler fluxes. No per-node P2 or flux-jet solution cache.
    template <int dim>
    struct DifferentialJet
    {
        static constexpr int Modes = dim + dim * (dim + 1) / 2;
        Eigen::Matrix<real, Modes + 1, 1> v = Eigen::Matrix<real, Modes + 1, 1>::Zero();
        static DifferentialJet Product(const DifferentialJet &a, const DifferentialJet &b)
        {
            DifferentialJet c;
            c.v(0) = a.v(0) * b.v(0);
            for (int d = 0; d < dim; d++)
                c.v(1 + d) = a.v(1 + d) * b.v(0) + a.v(0) * b.v(1 + d);
            int k = 1 + dim;
            for (int d = 0; d < dim; d++)
                for (int e = d; e < dim; e++, k++)
                    c.v(k) = a.v(k) * b.v(0) + a.v(0) * b.v(k) +
                             a.v(1 + d) * b.v(1 + e) + a.v(1 + e) * b.v(1 + d);
            return c;
        }
        static DifferentialJet Reciprocal(const DifferentialJet &a)
        {
            DifferentialJet c;
            const real r = 1 / a.v(0);
            c.v(0) = r;
            for (int d = 0; d < dim; d++)
                c.v(1 + d) = -a.v(1 + d) * r * r;
            int k = 1 + dim;
            for (int d = 0; d < dim; d++)
                for (int e = d; e < dim; e++, k++)
                    c.v(k) = 2 * a.v(1 + d) * a.v(1 + e) * r * r * r - a.v(k) * r * r;
            return c;
        }
    };

    inline Eigen::MatrixXd DifferentialWBAP(const std::vector<Eigen::MatrixXd> &candidates,
                                           const Eigen::VectorXd &metric)
    {
        Eigen::MatrixXd result = candidates.front();
        for (Eigen::Index v = 0; v < result.cols(); v++)
        {
            real scale = 0;
            for (const auto &candidate : candidates)
                scale = std::max(scale, (candidate.col(v).array().square() * metric.array()).sum());
            if (scale == 0)
            {
                result.col(v).setZero();
                continue;
            }
            std::vector<real> beta;
            for (const auto &candidate : candidates)
                beta.push_back((candidate.col(v).array().square() * metric.array()).sum() / scale + 1e-24);
            const real smallest = *std::min_element(beta.begin(), beta.end());
            real total = 0;
            result.col(v).setZero();
            for (std::size_t s = 0; s < candidates.size(); s++)
            {
                const real weight = (s == 0 ? 10 : 1) * sqr(smallest / beta[s]);
                result.col(v) += weight * candidates[s].col(v);
                total += weight;
            }
            result.col(v) /= total;
        }
        return result;
    }

    template <int dimension>
    void SpatialOperator<dimension>::InitializeCellLocal()
    {
        _differentialBounds = _reconstructionSettings.enableLimiter;
        _differentialWBAP = _cellPairBounds = false;
        _localLinearLimiter.Initialize(_geometry, _reconstructionSettings.enableLimiter);
        _p1FallbackSides.resize(_topology.NumEdgeProc());
        _p1BothSides = !std::getenv("NCFV_P1_BOTH_SIDES") || std::atoi(std::getenv("NCFV_P1_BOTH_SIDES")) != 0;
        _p1ComponentWise = std::getenv("NCFV_P1_COMPONENTWISE") && std::atoi(std::getenv("NCFV_P1_COMPONENTWISE")) != 0;
        _p1LocalComponentWise = std::getenv("NCFV_P1_LOCAL_COMPONENTWISE") &&
                                std::atoi(std::getenv("NCFV_P1_LOCAL_COMPONENTWISE")) != 0;
        const std::string slopeLimiter = std::getenv("NCFV_P1_SLOPE_LIMITER") ? std::getenv("NCFV_P1_SLOPE_LIMITER") : "Barth";
        _reportFourPoint = slopeLimiter == "ReportFourPoint" && _reconstructionSettings.enableLimiter;
        if (_reportFourPoint)
        {
            DNDS_check_throw_info(_p1BothSides && !_p1ComponentWise && _physics.viscous.gasConstant > 0,
                                  "ReportFourPoint requires paired traces, scalar P1 positivity and positive R");
            _reportP1Scales.resize(_topology.NumEdgeProc());
        }
        const std::vector<std::string> names = {"Barth", "Minmod", "MC", "VanLeer", "VanAlbada1", "Superbee", "Koren"};
        const auto selected = std::find(names.begin(), names.end(), slopeLimiter);
        DNDS_check_throw_info(selected != names.end() || slopeLimiter == "ReportFourPoint", "Unknown P1 slope limiter");
        _p1SlopeLimiter = selected == names.end() ? 0 : static_cast<int>(selected - names.begin());
        if (_p1SlopeLimiter && _reconstructionSettings.enableLimiter)
        {
            // Only one geometry exchange at initialization. No extra per-RHS halo.
            AllocateNodeField(_p1CentroidOffsets, "NCFV.p1CentroidOffsets", dimension);
            for (index node = 0; node < _mesh->NumNode(); node++)
                _p1CentroidOffsets[node] = _localLinearLimiter.Weight(node).template head<dimension>();
            _p1CentroidOffsets.trans.startPersistentPull();
            _p1CentroidOffsets.trans.waitPersistentPull();
        }
        _p1UseLLF = std::getenv("NCFV_P1_FLUX") && std::string(std::getenv("NCFV_P1_FLUX")) == "LLF";
        _fluxDensityOnly = std::getenv("NCFV_FLUX_DENSITY_ONLY") && std::atoi(std::getenv("NCFV_FLUX_DENSITY_ONLY")) != 0;
        _fluxComponentWise = std::getenv("NCFV_FLUX_COMPONENTWISE") && std::atoi(std::getenv("NCFV_FLUX_COMPONENTWISE")) != 0;
        DNDS_check_throw_info(!_p1LocalComponentWise ||
                                  (_reconstructionSettings.enableLimiter && slopeLimiter == "Barth" &&
                                   !_p1ComponentWise && !_fluxComponentWise && _reconstructionSettings.limitFluxCorrection),
                              "Local componentwise P1 requires Barth reconstruction and shared flux correction");
        _fluxPressureFloorOnly = std::getenv("NCFV_FLUX_PRESSURE_FLOOR_ONLY") && std::atoi(std::getenv("NCFV_FLUX_PRESSURE_FLOOR_ONLY")) != 0;
        _fluxConstraintTolerance = std::getenv("NCFV_FLUX_RELAXATION") ? std::atof(std::getenv("NCFV_FLUX_RELAXATION")) : 0;
        DNDS_check_throw_info(std::isfinite(_fluxConstraintTolerance) && _fluxConstraintTolerance >= 0 &&
            _fluxConstraintTolerance <= 2.5e-11, "Flux relaxation must fit inside the 1e-10 audit tolerance");
        DNDS_check_throw_info(!_fluxComponentWise || !_reconstructionSettings.fluxCorrectionBinary,
                              "Vector flux correction uses continuous factors");
        DNDS_check_throw_info(!_fluxComponentWise || !_reconstructionSettings.fluxCorrectionPairBounds,
                              "Vector flux correction uses cell-neighborhood bar-state bounds");
        if (_p1BothSides && _reconstructionSettings.enableLimiter)
            _p1Traces.resize(_topology.NumEdgeProc());
        DNDS_check_throw_info(_physics.riemannSolver == Euler::Gas::Roe,
                              "P1 midpoint high flux uses Roe");
        _cellShockSensing = std::getenv("NCFV_CELL_SHOCK_SENSOR") && std::atoi(std::getenv("NCFV_CELL_SHOCK_SENSOR")) != 0;
        if (_cellShockSensing)
        {
            DNDS_check_throw_info(_reconstructionSettings.limitFluxCorrection, "Shock sensor needs full-flux correction");
            AllocateNodeField(_cellShockSensor, "NCFV.diffShockSensor", 1);
        }
        DNDS_check_throw_info(_mode == IntegrationMode::EfficientDifferential && !_physics.viscous.enabled,
                              "Differential WBAP experiment is inviscid EfficientDifferential");
        AllocateNodeMatrix(_cellBounds, "NCFV.diffMeanBounds", dimension + 2, 2);
        _cellFaceBasis.resize(_topology.NumEdgeProc());
        _cellFaceFluxBasis.resize(_topology.NumEdgeProc());
        _differentialTraces.resize(_topology.NumEdgeProc());
        _differentialFluxes.resize(_topology.NumEdgeProc());
        for (index edge = 0; edge < _topology.NumEdgeProc(); edge++)
        {
            const auto &surface = _geometry.EdgeSurface(edge);
            for (int side = 0; side < 2; side++)
            {
                auto &basis = _cellFaceBasis[edge][side];
                auto &fluxBasis = _cellFaceFluxBasis[edge][side];
                basis.setZero();
                fluxBasis.setZero();
                const index node = surface.nodes[side];
                const auto lengths = _nodeHalo.ReferenceLengths(node);
                for (const auto &entry : surface.efficientStencil)
                {
                    const auto gradient = Reconstruction::EvaluateBasisGradient(
                        _nodeHalo.Displacement(node, entry.node), lengths, dimension);
                    basis.noalias() += gradient.transpose() * entry.stateGradientWeights[side].template head<dimension>() / surface.measure;
                    fluxBasis.noalias() += gradient.transpose() * entry.fluxGradientWeights[side].template topLeftCorner<dimension, dimension>();
                }
            }
        }
        _cellVolumeBasis.resize(_mesh->NumNode());
        _cellBoundaryBasis.resize(_mesh->NumNode());
        _differentialStencils.resize(_mesh->NumNode());
        _differentialBoundary.resize(_mesh->NumNode());
        if (_reconstructionSettings.limitFluxCorrection)
            _boundaryHighFlux.resize(_mesh->NumNode());
        constexpr int nCurv = dimension * (dimension + 1) / 2;
        real localCondition = 0;
        for (index node = 0; node < _mesh->NumNode(); node++)
        {
            if (!_boundaryHighFlux.empty())
                _boundaryHighFlux[node].resize(_geometry.NodeVolume(node).boundaryPieces.size(), State::Zero());
            auto &fit = _differentialStencils[node];
            fit.length = std::pow(_geometry.NodeVolume(node).moments.measure, 1.0 / dimension);
            std::vector<SpatialVector> offsets;
            for (const auto &incidence : _topology.Node2Edge(node))
            {
                const auto &face = _geometry.EdgeSurface(incidence.edge);
                const index other = face.nodes[face.nodes[0] == node ? 1 : 0];
                fit.nodes.push_back(other);
                offsets.push_back(_nodeHalo.Displacement(node, other).template head<dimension>() / fit.length);
            }
            Eigen::MatrixXd design = Eigen::MatrixXd::Zero(dimension * offsets.size(), nCurv);
            for (std::size_t m = 0; m < offsets.size(); m++)
            {
                int k = 0;
                for (int d = 0; d < dimension; d++)
                    for (int e = d; e < dimension; e++, k++)
                    {
                        design(dimension * m + d, k) = offsets[m](e);
                        if (d != e) design(dimension * m + e, k) = offsets[m](d);
                    }
            }
            // Geometry-directed positive-weight fits on the same complete
            // stencil. Directions rotate with the mesh; no preferred z axis.
            for (std::size_t s = 0; s <= (_differentialWBAP ? offsets.size() : 0); s++)
            {
                Eigen::VectorXd weights(design.rows());
                for (std::size_t m = 0; m < offsets.size(); m++)
                {
                    const real bias = s == 0 ? 0 : offsets[m].normalized().dot(offsets[s - 1].normalized());
                    for (int d = 0; d < dimension; d++) weights(dimension * m + d) = std::exp(2 * bias);
                }
                const Eigen::MatrixXd weighted = weights.asDiagonal() * design;
                const Eigen::JacobiSVD<Eigen::MatrixXd> svd(weighted, Eigen::ComputeThinU | Eigen::ComputeThinV);
                DNDS_check_throw_info(svd.rank() == nCurv, "Differential curvature stencil is rank deficient");
                const real condition = svd.singularValues()(0) / svd.singularValues()(nCurv - 1);
                DNDS_check_throw_info(condition < 1e8, "Differential curvature stencil is ill conditioned");
                localCondition = std::max(localCondition, condition);
                fit.maps.push_back(svd.solve(weights.asDiagonal().toDenseMatrix()));
            }
            _cellVolumeBasis[node] = _reconstruction.Operator(node).targetBasisMean;
            const auto lengths = _nodeHalo.ReferenceLengths(node);
            for (const auto &piece : _geometry.NodeVolume(node).boundaryPieces)
            {
                std::vector<Vector3> points;
                for (int k = 0; k < piece.nPoints; k++)
                    points.push_back(_nodeHalo.DisplacementToPoint(node, piece.points[k].coordinate));
                _cellBoundaryBasis[node].push_back(Reconstruction::MeanBasis(
                    DualGeometry::ExactSimplexMoments(points, piece.measure), Vector3::Zero(), lengths, dimension));
            }
        }
        // Route one completed side (mean, integrated physical flux) from the
        // anchor owner to the edge owner. Counts and IDs are initialized once.
        std::vector<std::vector<std::pair<index, int>>> routes(_mpi.size);
        for (index node = 0; node < _mesh->NumNode(); node++)
            for (const auto &incidence : _topology.Node2Edge(node))
                if (incidence.edge >= _topology.NumEdge())
                {
                    const index global = _topology.EdgeIndexLocal2Global(incidence.edge);
                    MPI_int owner = -1;
                    index ownerLocal = -1;
                    const bool found = _topology.Edge2Node().father->pLGlobalMapping->search(global, owner, ownerLocal);
                    DNDS_check_throw_info(found && owner != _mpi.rank, "Invalid differential edge-side route");
                    const auto &surface = _geometry.EdgeSurface(incidence.edge);
                    routes[owner].push_back({incidence.edge, surface.nodes[0] == node ? 0 : 1});
                }
        std::vector<int> sendCounts(_mpi.size), receiveCounts(_mpi.size), sendOffsets(_mpi.size), receiveOffsets(_mpi.size);
        std::vector<index> sendIDs;
        for (int rank = 0; rank < _mpi.size; rank++)
        {
            sendOffsets[rank] = sendIDs.size();
            sendCounts[rank] = routes[rank].size();
            for (const auto &slot : routes[rank])
            {
                _differentialSendSlots.push_back(slot);
                sendIDs.push_back(2 * _topology.EdgeIndexLocal2Global(slot.first) + slot.second);
            }
        }
        MPI_Alltoall(sendCounts.data(), 1, MPI_INT, receiveCounts.data(), 1, MPI_INT, _mpi.comm);
        int received = 0;
        for (int rank = 0; rank < _mpi.size; rank++)
        {
            receiveOffsets[rank] = received;
            received += receiveCounts[rank];
        }
        std::vector<index> receiveIDs(received);
        MPI_Alltoallv(sendIDs.data(), sendCounts.data(), sendOffsets.data(), DNDS_MPI_INDEX,
                      receiveIDs.data(), receiveCounts.data(), receiveOffsets.data(), DNDS_MPI_INDEX, _mpi.comm);
        for (index id : receiveIDs)
        {
            MPI_int owner = -1;
            index local = -1;
            const bool found = _topology.Edge2Node().father->pLGlobalMapping->search(id / 2, owner, local);
            DNDS_check_throw_info(found && owner == _mpi.rank, "Received side for a foreign edge");
            _differentialReceiveSlots.push_back({local, static_cast<int>(id % 2)});
        }
        const int width = (2 + !_p1Traces.empty()) * (dimension + 2) + 1 + int(_reportFourPoint);
        for (int rank = 0; rank < _mpi.size; rank++)
        {
            _differentialSendCounts.push_back(width * sendCounts[rank]);
            _differentialReceiveCounts.push_back(width * receiveCounts[rank]);
            _differentialSendOffsets.push_back(width * sendOffsets[rank]);
            _differentialReceiveOffsets.push_back(width * receiveOffsets[rank]);
        }
        real maximumCondition = 0;
        MPI_Allreduce(&localCondition, &maximumCondition, 1, DNDS_MPI_REAL, MPI_MAX, _mpi.comm);
        if (_mpi.rank == 0)
            log() << (_p1BothSides ? "BARTH_P1_BOTH_SIDES" : "BARTH_P1_EDGE_MIDPOINT")
                  << ": R1-only recovery; full Roe interface fallback; max condition="
                  << maximumCondition << "; P1 weight bytes per rank=" << _localLinearLimiter.WeightStorageBytes()
                  << "; paired P1 traces=" << !_p1Traces.empty()
                  << "; componentwise=" << _p1ComponentWise << "; local componentwise=" << _p1LocalComponentWise
                  << "; fallback LLF=" << _p1UseLLF << std::endl;
    }

    template <int dimension>
    void SpatialOperator<dimension>::ExchangeDifferentialSides()
    {
        constexpr int nVars = dimension + 2;
        const int width = (2 + !_p1Traces.empty()) * nVars + 1 + int(_reportFourPoint);
        std::vector<real> send(width * _differentialSendSlots.size());
        std::vector<real> receive(width * _differentialReceiveSlots.size());
        for (std::size_t i = 0; i < _differentialSendSlots.size(); i++)
        {
            const auto [edge, side] = _differentialSendSlots[i];
            for (int v = 0; v < nVars; v++)
            {
                send[width * i + v] = _differentialTraces[edge][side](v);
                send[width * i + nVars + v] = _differentialFluxes[edge][side](v);
                if (!_p1Traces.empty())
                    send[width * i + 2 * nVars + 1 + v] = _p1Traces[edge][side](v);
            }
            send[width * i + 2 * nVars] = _p1FallbackSides[edge][side];
            if (_reportFourPoint) send[width * i + width - 1] = _reportP1Scales[edge][side];
        }
        MPI_Alltoallv(send.data(), _differentialSendCounts.data(), _differentialSendOffsets.data(), DNDS_MPI_REAL,
                      receive.data(), _differentialReceiveCounts.data(), _differentialReceiveOffsets.data(), DNDS_MPI_REAL, _mpi.comm);
        for (std::size_t i = 0; i < _differentialReceiveSlots.size(); i++)
        {
            const auto [edge, side] = _differentialReceiveSlots[i];
            for (int v = 0; v < nVars; v++)
            {
                _differentialTraces[edge][side](v) = receive[width * i + v];
                _differentialFluxes[edge][side](v) = receive[width * i + nVars + v];
                if (!_p1Traces.empty())
                    _p1Traces[edge][side](v) = receive[width * i + 2 * nVars + 1 + v];
            }
            _p1FallbackSides[edge][side] = receive[width * i + 2 * nVars] > .5;
            if (_reportFourPoint) _reportP1Scales[edge][side] = receive[width * i + width - 1];
        }
        // Make the decision once on the edge owner after both trigger flags
        // and both mean-preserving P1 alternatives have arrived. No recursive
        // propagation to adjacent cells or modification of shared gradients.
        if (!_p1Traces.empty())
            for (index edge = 0; edge < _topology.NumEdge(); edge++)
                if (_p1FallbackSides[edge][0] || _p1FallbackSides[edge][1])
                {
                    if (_reportFourPoint) ApplyReportFourPoint(edge);
                    else _differentialTraces[edge] = _p1Traces[edge];
                }
    }

    template <int dimension>
    void SpatialOperator<dimension>::ApplyReportFourPoint(index edge)
    {
        const auto &nodes = _geometry.EdgeSurface(edge).nodes;
        const SpatialVector halfEdge = .5 * _nodeHalo.Displacement(nodes[0], nodes[1]).template head<dimension>();
        const auto variables = [&](const State &u) -> State
        {
            DNDS_check_throw_info(IsPhysical(u), "Nonphysical report sampling state");
            State w;
            w.template head<dimension>() = u.template segment<dimension>(1) / u(0);
            w(dimension) = Pressure(u);
            w(dimension + 1) = Temperature(u);
            return w;
        };
        const auto &anchor = _p1Traces[edge];
        const State wi = variables(anchor[0]), wj = variables(anchor[1]);
        const State wp = variables(anchor[0] - _reportP1Scales[edge][0] * _stateGradients[nodes[0]].transpose() * halfEdge);
        const State wq = variables(anchor[1] + _reportP1Scales[edge][1] * _stateGradients[nodes[1]].transpose() * halfEdge);
        const State jump = wj - wi;
        std::array<State, 2> traces{wi, wj};
        for (int v = 0; v < dimension + 2; ++v)
        {
            traces[0](v) += .5 * LinearBarthLimiter::ReportIncrement(jump(v), 2 * (wi(v) - wp(v)));
            traces[1](v) -= .5 * LinearBarthLimiter::ReportIncrement(jump(v), 2 * (wq(v) - wj(v)));
        }
        for (int side = 0; side < 2; ++side)
        {
            const State &w = traces[side];
            if (!w.allFinite() || w(dimension) <= 0 || w(dimension + 1) <= 0)
            {
                _differentialTraces[edge][side] = anchor[side];
                continue;
            }
            State u;
            u(0) = w(dimension) / (_physics.viscous.gasConstant * w(dimension + 1));
            u.template segment<dimension>(1) = u(0) * w.template head<dimension>();
            u(dimension + 1) = w(dimension) / (_physics.gamma - 1)
                              + .5 * u(0) * w.template head<dimension>().squaredNorm();
            _differentialTraces[edge][side] = IsPhysical(u) ? u : anchor[side];
        }
    }

    template <int dimension>
    void SpatialOperator<dimension>::ReconstructCellLocal(NodeStatePair &means)
    {
        constexpr int nVars = dimension + 2;
        constexpr int nCurv = dimension * (dimension + 1) / 2;
        double start = MPI_Wtime();
        means.trans.startPersistentPull();
        means.trans.waitPersistentPull();
        _lastRhsTiming.meanHaloSeconds += MPI_Wtime() - start;
        start = MPI_Wtime();
        _reconstruction.ComputeCoefficients(means, _stateGradients, _coefficients);
        _lastRhsTiming.coefficientComputeSeconds += MPI_Wtime() - start;
        start = MPI_Wtime();
        _stateGradients.trans.startPersistentPull();
        _stateGradients.trans.waitPersistentPull();
        _lastRhsTiming.coefficientHaloSeconds += MPI_Wtime() - start;
        start = MPI_Wtime();
        _differentialMeanDefect = _differentialMaximumDz = 0;
        _twoParameterObjectiveExcess = 0;
        _twoParameterMinimumAlpha = _twoParameterMinimumBeta = 1;
        _twoParameterSplitCells = _twoParameterPositiveCells = 0;
        _differentialBoundaryViolations = _differentialBoundaryNonphysical = 0;
        _differentialBoundaryExcess = 0;
        _differentialBoundaryMinDensity = _differentialBoundaryMinPressure = veryLargeReal;
        const char *tablePrefix = std::getenv("NCFV_CELL_TABLE_PREFIX");
        const char *jetPrefix = std::getenv("NCFV_CELL_JET_PREFIX");
        const index tableCall = std::getenv("NCFV_CELL_TABLE_CALL") ? std::atoll(std::getenv("NCFV_CELL_TABLE_CALL")) : 0;
        nlohmann::json rows = nlohmann::json::array();
        nlohmann::json jets = nlohmann::json::array();
        std::vector<real> sensorPressures;
        _p1GatedNodes = _p1FullWeightNodes = _p1ChangedFallbackNodes = 0;
        if (_p1LocalComponentWise)
        {
            sensorPressures.resize(_nodeHalo.NumNodeProc());
            for (index node = 0; node < _nodeHalo.NumNodeProc(); ++node)
                sensorPressures[node] = Pressure(means[node]);
        }
        const auto jsonVector = [](const auto &value) { return std::vector<real>(value.data(), value.data() + value.size()); };
        Eigen::VectorXd metric(nCurv);
        int mode = 0;
        for (int d = 0; d < dimension; d++)
            for (int e = d; e < dimension; e++, mode++) metric(mode) = d == e ? 1 : 2;
        for (index node = 0; node < _mesh->NumNode(); node++)
        {
            const auto &fit = _differentialStencils[node];
            const auto lengths = _nodeHalo.ReferenceLengths(node);
            const State mean = means[node];
            const Eigen::MatrixXd rawGradient = _stateGradients[node];
            Eigen::MatrixXd differences(dimension * fit.nodes.size(), nVars);
            for (std::size_t m = 0; m < fit.nodes.size(); m++)
                differences.middleRows(dimension * m, dimension) = _stateGradients[fit.nodes[m]] - rawGradient;
            std::vector<Eigen::MatrixXd> curvatures;
            for (const auto &map : fit.maps) curvatures.push_back(map * differences);
            const Eigen::MatrixXd curvature = _differentialWBAP && _reconstructionSettings.enableLimiter ?
                DifferentialWBAP(curvatures, metric) : curvatures.front();
            std::array<Eigen::Matrix<real, dimension, dimension>, nVars> hessians;
            for (int v = 0; v < nVars; v++)
            {
                int k = 0;
                for (int d = 0; d < dimension; d++)
                    for (int e = d; e < dimension; e++, k++)
                        hessians[v](d, e) = hessians[v](e, d) = curvature(k, v) / fit.length;
            }
            std::vector<Eigen::MatrixXd> slopes{rawGradient};
            if (_differentialWBAP)
                for (index other : fit.nodes)
                {
                    const SpatialVector d = _nodeHalo.Displacement(node, other).template head<dimension>();
                    Eigen::MatrixXd candidate = _stateGradients[other];
                    for (int v = 0; v < nVars; v++) candidate.col(v) -= hessians[v] * d;
                    slopes.push_back(std::move(candidate));
                }
            const Eigen::MatrixXd gradient = _differentialWBAP && _reconstructionSettings.enableLimiter ?
                DifferentialWBAP(slopes, Eigen::VectorXd::Ones(dimension)) : rawGradient;
            // Temporary normalized action representation, only for this cell's
            // contraction, constraints and automatic differentiation. Never
            // allocated as a solution field or exchanged through MPI.
            LocalCoefficients candidate, raw;
            candidate.template topRows<dimension>() = lengths.template head<dimension>().asDiagonal() * gradient;
            raw.template topRows<dimension>() = lengths.template head<dimension>().asDiagonal() * rawGradient;
            int k = dimension;
            for (int d = 0; d < dimension; d++)
                for (int e = d; e < dimension; e++, k++)
                {
                    candidate.row(k) = curvature.row(k - dimension) * (lengths(d) * lengths(e) / fit.length);
                    raw.row(k) = curvatures.front().row(k - dimension) * (lengths(d) * lengths(e) / fit.length);
                }
            State lower = mean, upper = mean;
            nlohmann::json neighbors = nlohmann::json::array();
            for (index other : fit.nodes)
            {
                lower = lower.cwiseMin(State(means[other]));
                upper = upper.cwiseMax(State(means[other]));
                if (tablePrefix && _cellAuditCalls == tableCall) neighbors.push_back(jsonVector(State(means[other])));
            }
            _cellBounds[node].col(0) = lower;
            _cellBounds[node].col(1) = upper;
            std::vector<LocalBasis> checks;
            std::vector<real> faceWeights;
            std::vector<Vector3> midpointOffsets;
            for (const auto &incidence : _topology.Node2Edge(node))
            {
                const auto &face = _geometry.EdgeSurface(incidence.edge);
                const int side = face.nodes[0] == node ? 0 : 1;
                checks.push_back(_cellFaceBasis[incidence.edge][side] - _cellVolumeBasis[node]);
                faceWeights.push_back(face.measure);
                midpointOffsets.push_back(.5 * _nodeHalo.Displacement(node, face.nodes[1 - side]));
            }
            const auto &volume = _geometry.NodeVolume(node);
            for (std::size_t b = 0; b < volume.boundaryPieces.size(); b++)
                if (_boundaries.Get(volume.boundaryPieces[b].zone).mode != BoundaryMode::Periodic)
                {
                    checks.push_back(_cellBoundaryBasis[node][b] - _cellVolumeBasis[node]);
                    faceWeights.push_back(volume.boundaryPieces[b].measure);
                    midpointOffsets.push_back(lengths.cwiseProduct(_cellBoundaryBasis[node][b].template head<dimension>()));
                }
            using Math = LocalLimiterMath<dimension>;
            const auto controls = Math::BernsteinBasis(volume.lowerOffset, volume.upperOffset, lengths, _cellVolumeBasis[node]);
            std::vector<LocalBasis> positiveChecks(controls.begin(), controls.end());
            positiveChecks.insert(positiveChecks.end(), checks.begin(), checks.end());
            positiveChecks.push_back(-_cellVolumeBasis[node]);
            bool fallback = false;
            if (_reconstructionSettings.enableLimiter)
            {
                for (const auto &basis : checks)
                {
                    const State value = mean + candidate.transpose() * basis;
                    fallback |= (value - upper).maxCoeff() > 1e-10 || (lower - value).maxCoeff() > 1e-10;
                }
                for (const auto &basis : positiveChecks)
                    fallback |= !IsPhysical(mean + candidate.transpose() * basis);
            }
            struct LocalResult
            {
                LocalCoefficients coefficients;
                real alpha = 1, beta = 1, positivity = 1, repair = 1;
                real objective = 0, uniformObjective = 0, uniformFactor = 1;
            } limited;
            limited.coefficients = candidate;
            State r1 = State::Zero();
            LocalCoefficients p1Coefficients = LocalCoefficients::Zero();
            State p1Point = mean;
            State p1Factors = State::Ones();
            State projectedFactors = State::Ones();
            StateGradient p1Gradient = StateGradient::Zero();
            real p1Factor = 1, p1Positivity = 1;
            const auto positivityFactor = [&](const LocalCoefficients &values)
            {
                real factor = 1;
                for (const auto &basis : positiveChecks)
                {
                    const State delta = values.transpose() * basis;
                    if (!IsPhysical(mean + factor * delta))
                    {
                        real lo = 0, hi = factor;
                        for (int iteration = 0; iteration < 48; iteration++)
                        {
                            const real mid = .5 * (lo + hi);
                            if (IsPhysical(mean + mid * delta)) lo = mid;
                            else hi = mid;
                        }
                        factor = .95 * lo;
                    }
                }
                return factor;
            };
            DNDS_check_throw_info(IsPhysical(mean), "P1 fallback requires a physical mean");
            real localP1Weight = 0;
            if (fallback || !_p1Traces.empty())
            {
                if (_p1LocalComponentWise)
                {
                    using Sensor = PlanarCompression<dimension>;
                    const auto offset = [&](index other) -> SpatialVector
                    {
                        return _nodeHalo.Displacement(node, other).template head<dimension>();
                    };
                    const auto pressureGradient = [&](index other) -> SpatialVector
                    {
                        return Sensor::PressureGradient(means[other], _stateGradients[other], _physics.gamma);
                    };
                    localP1Weight = Sensor::Weight(node, mean, rawGradient, _physics.gamma, fit.nodes,
                                                   sensorPressures, offset, pressureGradient);
                    _p1GatedNodes += localP1Weight > 0;
                    _p1FullWeightNodes += localP1Weight == 1;
                }
                if (_reportFourPoint)
                {
                    r1 = -rawGradient.transpose() * _localLinearLimiter.Weight(node).template head<dimension>();
                    p1Gradient = rawGradient;
                }
                else if (_p1ComponentWise || localP1Weight > 0)
                {
                    const auto linear = LinearBarthLimiter::LimitComponentsWithWeight(mean, rawGradient,
                        _localLinearLimiter.Weight(node), midpointOffsets, lower, upper);
                    r1 = linear.linearConstant;
                    p1Factors = linear.factors;
                    p1Gradient = linear.gradient;
                    if (_p1LocalComponentWise)
                    {
                        const real shared = p1Factors.minCoeff();
                        if (localP1Weight < 1)
                        {
                            p1Factors = ((1 - localP1Weight) * shared + localP1Weight * p1Factors.array()).matrix();
                            p1Gradient = rawGradient * p1Factors.asDiagonal();
                        }
                        _p1ChangedFallbackNodes += fallback && (p1Factors.maxCoeff() - shared > 1e-12);
                    }
                }
                else
                {
                    const auto linear = _localLinearLimiter.LimitCell(node, mean, rawGradient, midpointOffsets, lower, upper);
                    r1 = linear.linearConstant;
                    p1Factors.setConstant(linear.factor);
                    p1Gradient = linear.gradient;
                }
                if (_p1SlopeLimiter)
                {
                    for (const auto &incidence : _topology.Node2Edge(node))
                    {
                        const auto &surface = _geometry.EdgeSurface(incidence.edge);
                        const index other = surface.nodes[surface.nodes[0] == node ? 1 : 0];
                        const SpatialVector displacement = _nodeHalo.Displacement(node, other).template head<dimension>() +
                            _p1CentroidOffsets[other] - _p1CentroidOffsets[node];
                        const State prediction = rawGradient.transpose() * displacement;
                        const State jump = State(means[other]) - mean;
                        for (int v = 0; v < nVars; v++)
                            projectedFactors(v) = std::min(projectedFactors(v),
                                LinearBarthLimiter::ProjectedSlopeFactor(prediction(v), jump(v), _p1SlopeLimiter));
                    }
                    // Both tests use the ORIGINAL gradient; take a minimum,
                    // rather than applying two sequential contractions.
                    p1Factors = p1Factors.cwiseMin(projectedFactors);
                    if (!_p1ComponentWise)
                        p1Factors.setConstant(p1Factors.minCoeff());
                    p1Gradient = rawGradient;
                    for (int v = 0; v < nVars; v++)
                        p1Gradient.col(v) *= p1Factors(v);
                }
                p1Factor = p1Factors.minCoeff();
                p1Coefficients.template topRows<dimension>() = lengths.template head<dimension>().asDiagonal() * p1Gradient;
                p1Positivity = positivityFactor(p1Coefficients);
                if (_reportFourPoint)
                {
                    // Every outward half-edge sample belongs to this SAME P1.
                    for (const auto &incidence : _topology.Node2Edge(node))
                    {
                        const auto &nodes = _geometry.EdgeSurface(incidence.edge).nodes;
                        const auto other = nodes[nodes[0] == node ? 1 : 0];
                        const State delta = r1 - .5 * rawGradient.transpose() *
                            _nodeHalo.Displacement(node, other).template head<dimension>();
                        if (IsPhysical(mean + p1Positivity * delta)) continue;
                        real lo = 0, hi = p1Positivity;
                        for (int iteration = 0; iteration < 48; ++iteration)
                        {
                            const real mid = .5 * (lo + hi);
                            if (IsPhysical(mean + mid * delta)) lo = mid;
                            else hi = mid;
                        }
                        p1Positivity = .95 * lo;
                    }
                }
                p1Coefficients *= p1Positivity;
                p1Gradient *= p1Positivity;
                p1Point = mean + p1Positivity * p1Factors.cwiseProduct(r1);
                _differentialMeanDefect = std::max(_differentialMeanDefect,
                    (p1Point + p1Coefficients.transpose() * _cellVolumeBasis[node] - mean).cwiseAbs().maxCoeff());
            }
            if (fallback)
            {
                limited.alpha = p1Factor;
                limited.beta = 0;
                limited.positivity = p1Positivity;
                limited.coefficients = p1Coefficients;
            }
            else
            {
                limited.positivity = positivityFactor(limited.coefficients);
                limited.coefficients *= limited.positivity;
            }
            LocalCoefficients coefficients = limited.coefficients;
            _twoParameterObjectiveExcess = std::max(_twoParameterObjectiveExcess,
                (limited.objective - limited.uniformObjective) / std::max(real(1), limited.uniformObjective));
            _twoParameterMinimumAlpha = std::min(_twoParameterMinimumAlpha, limited.alpha);
            _twoParameterMinimumBeta = std::min(_twoParameterMinimumBeta, limited.beta);
            _twoParameterSplitCells += std::abs(limited.alpha - limited.beta) > 1e-10;
            _twoParameterPositiveCells += limited.positivity < 1;
            real pairFactor = 1;
            if (_cellPairBounds && _reconstructionSettings.enableLimiter)
                for (const auto &incidence : _topology.Node2Edge(node))
                {
                    const auto &face = _geometry.EdgeSurface(incidence.edge);
                    const int side = face.nodes[0] == node ? 0 : 1;
                    const State other = means[face.nodes[1 - side]];
                    const State delta = coefficients.transpose() * (_cellFaceBasis[incidence.edge][side] - _cellVolumeBasis[node]);
                    const State lo = mean.cwiseMin(other), hi = mean.cwiseMax(other);
                    for (int v = 0; v < nVars; v++)
                    {
                        if (delta(v) > 5e-11) pairFactor = std::min(pairFactor, std::clamp((hi(v) - mean(v)) / delta(v), real(0), real(1)));
                        else if (delta(v) < -5e-11) pairFactor = std::min(pairFactor, std::clamp((lo(v) - mean(v)) / delta(v), real(0), real(1)));
                    }
                }
            coefficients *= pairFactor;
            const State point = fallback ? p1Point :
                                          State(mean - coefficients.transpose() * _cellVolumeBasis[node]);
            _pointValues[node] = point;
            _limiterFactors[node](0) = fallback ? 1 - 1e-12 : 1;
            _differentialMeanDefect = std::max(_differentialMeanDefect,
                (point + coefficients.transpose() * _cellVolumeBasis[node] - mean).cwiseAbs().maxCoeff());
            if constexpr (dimension == 3)
                _differentialMaximumDz = std::max(_differentialMaximumDz, (coefficients.row(2) / lengths(2)).cwiseAbs().maxCoeff());
            Eigen::Matrix<real, nVars, dimension * (LocalModes + 1)> flux;
            flux.setZero();
            if (!fallback)
            {
            using Jet = DifferentialJet<dimension>;
            std::array<Jet, nVars> state;
            for (int v = 0; v < nVars; v++)
            {
                state[v].v(0) = point(v);
                state[v].v.template tail<LocalModes>() = coefficients.col(v);
            }
            const Jet reciprocal = Jet::Reciprocal(state[0]);
            Jet kinetic;
            for (int d = 0; d < dimension; d++) kinetic.v += 0.5 * Jet::Product(state[d + 1], state[d + 1]).v;
            kinetic = Jet::Product(kinetic, reciprocal);
            Jet pressure, enthalpy;
            pressure.v = (_physics.gamma - 1) * (state[nVars - 1].v - kinetic.v);
            enthalpy.v = state[nVars - 1].v + pressure.v;
            for (int d = 0; d < dimension; d++)
            {
                const Jet velocity = Jet::Product(state[d + 1], reciprocal);
                flux.row(0).template segment<LocalModes + 1>(d * (LocalModes + 1)) = state[d + 1].v.transpose();
                for (int e = 0; e < dimension; e++)
                {
                    Jet momentum = Jet::Product(state[e + 1], velocity);
                    if (d == e) momentum.v += pressure.v;
                    flux.row(e + 1).template segment<LocalModes + 1>(d * (LocalModes + 1)) = momentum.v.transpose();
                }
                flux.row(nVars - 1).template segment<LocalModes + 1>(d * (LocalModes + 1)) = Jet::Product(enthalpy, velocity).v.transpose();
            }
            }
            std::vector<State> selectedTraces;
            std::vector<State> p1SelectedTraces;
            for (const auto &incidence : _topology.Node2Edge(node))
            {
                const auto &face = _geometry.EdgeSurface(incidence.edge);
                const int side = face.nodes[0] == node ? 0 : 1;
                _p1FallbackSides[incidence.edge][side] = fallback;
                if (!_p1Traces.empty())
                {
                    _p1Traces[incidence.edge][side] = p1Point + p1Gradient.transpose() *
                        (.5 * _nodeHalo.Displacement(node, face.nodes[1 - side]).template head<dimension>());
                    if (tablePrefix) p1SelectedTraces.push_back(_p1Traces[incidence.edge][side]);
                    if (_reportFourPoint)
                    {
                        _p1Traces[incidence.edge][side] = p1Point;
                        _reportP1Scales[incidence.edge][side] = p1Positivity;
                    }
                }
                if (fallback)
                    _differentialTraces[incidence.edge][side] = point + p1Gradient.transpose() *
                        (.5 * _nodeHalo.Displacement(node, face.nodes[1 - side]).template head<dimension>());
                else
                    _differentialTraces[incidence.edge][side] = point + coefficients.transpose() * _cellFaceBasis[incidence.edge][side];
                if (tablePrefix) selectedTraces.push_back(_differentialTraces[incidence.edge][side]);
                State integral = State::Zero();
                for (int d = 0; d < dimension; d++)
                {
                    integral += face.vectorMeasure(d) * flux.col(d * (LocalModes + 1));
                    integral.noalias() += flux.template block<nVars, LocalModes>(0, d * (LocalModes + 1) + 1) * _cellFaceFluxBasis[incidence.edge][side].col(d);
                }
                _differentialFluxes[incidence.edge][side] = integral;
            }
            _differentialBoundary[node].setZero();
            for (std::size_t b = 0; b < volume.boundaryPieces.size(); b++)
            {
                const auto &piece = volume.boundaryPieces[b];
                const auto &boundary = _boundaries.Get(piece.zone);
                if (boundary.mode == BoundaryMode::Periodic) continue;
                const State trace = point + coefficients.transpose() * _cellBoundaryBasis[node][b];
                if (tablePrefix) selectedTraces.push_back(trace);
                const real excess = std::max(real(0), std::max((lower - trace).maxCoeff(), (trace - upper).maxCoeff()));
                _differentialBoundaryViolations += excess > 1e-10;
                _differentialBoundaryNonphysical += !IsPhysical(trace);
                _differentialBoundaryExcess = std::max(_differentialBoundaryExcess, excess);
                _differentialBoundaryMinDensity = std::min(_differentialBoundaryMinDensity, trace(0));
                _differentialBoundaryMinPressure = std::min(_differentialBoundaryMinPressure, Pressure(trace));
                const SpatialVector normal = piece.vectorMeasure.template head<dimension>() / piece.measure;
                State pieceFlux = State::Zero();
                if (fallback)
                    pieceFlux += piece.measure * BoundaryNumericalFlux(trace, normal, boundary);
                else
                {
                for (int d = 0; d < dimension; d++)
                    pieceFlux += piece.vectorMeasure(d) *
                        (State(flux.col(d * (LocalModes + 1))) + flux.template block<nVars, LocalModes>(0, d * (LocalModes + 1) + 1) * _cellBoundaryBasis[node][b]);
                pieceFlux += piece.measure * (BoundaryNumericalFlux(trace, normal, boundary) - PhysicalFlux(trace, normal));
                }
                _differentialBoundary[node] += pieceFlux;
                if (!_boundaryHighFlux.empty()) _boundaryHighFlux[node][b] = pieceFlux;
            }
            if (tablePrefix && _cellAuditCalls == tableCall)
            {
                nlohmann::json offsetsJSON = nlohmann::json::array(), gradientsJSON = nlohmann::json::array();
                for (index other : fit.nodes)
                {
                    offsetsJSON.push_back(jsonVector(_nodeHalo.Displacement(node, other)));
                    gradientsJSON.push_back(jsonVector(Eigen::MatrixXd(_stateGradients[other])));
                }
                nlohmann::json checkJSON = nlohmann::json::array(), controlJSON = nlohmann::json::array();
                nlohmann::json midpointJSON = nlohmann::json::array(), traceJSON = nlohmann::json::array();
                nlohmann::json p1TraceJSON = nlohmann::json::array();
                for (const auto &offset : midpointOffsets) midpointJSON.push_back(jsonVector(offset));
                for (const auto &trace : selectedTraces) traceJSON.push_back(jsonVector(trace));
                for (const auto &trace : p1SelectedTraces) p1TraceJSON.push_back(jsonVector(trace));
                for (const auto &basis : checks) checkJSON.push_back(jsonVector(basis));
                for (const auto &basis : controls) controlJSON.push_back(jsonVector(basis));
                rows.push_back({{"original", _mesh->node2nodeOrig(node, 0)}, {"mean", jsonVector(mean)},
                    {"lower", jsonVector(lower)}, {"upper", jsonVector(upper)}, {"neighbors", neighbors},
                    {"point", jsonVector(point)}, {"raw_coefficients", jsonVector(raw)},
                    {"coefficients", jsonVector(coefficients)}, {"candidate_coefficients", jsonVector(candidate)},
                    {"basis_mean", jsonVector(_cellVolumeBasis[node])}, {"check_basis", checkJSON}, {"control_basis", controlJSON},
                    {"raw_gradient", jsonVector(rawGradient)}, {"gradient_differences", jsonVector(differences)},
                    {"curvature", jsonVector(curvature)}, {"linear_gradient", jsonVector(gradient)},
                    {"stencil_offsets", offsetsJSON}, {"stencil_gradients", gradientsJSON}, {"fit_length", fit.length},
                    {"reference_lengths", jsonVector(lengths)},
                    {"alpha", limited.alpha}, {"beta", limited.beta}, {"positivity_factor", limited.positivity},
                    {"repair_factor", limited.repair}, {"face_weights", faceWeights},
                    {"qp_objective", limited.objective}, {"uniform_objective", limited.uniformObjective},
                    {"uniform_factor", limited.uniformFactor},
                    {"pair_factor", pairFactor}, {"coordinate", jsonVector(_nodeHalo.Coordinate(node))},
                    {"fallback", fallback}, {"r1", jsonVector(r1)}, {"midpoint_offsets", midpointJSON},
                    {"local_p1_weight", localP1Weight},
                    {"selected_face_states", traceJSON}, {"p1_point", jsonVector(p1Point)},
                    {"p1_coefficients", jsonVector(p1Coefficients)}, {"p1_factor", p1Factor},
                    {"p1_factors", jsonVector(p1Factors)}, {"p1_gradient", jsonVector(p1Gradient)},
                    {"p1_slope_limiter", _p1SlopeLimiter}, {"p1_projected_factors", jsonVector(projectedFactors)},
                    {"p1_positivity", p1Positivity}, {"p1_face_states", p1TraceJSON}});
            }
            if (jetPrefix && _cellAuditCalls == tableCall)
                jets.push_back({{"original", _mesh->node2nodeOrig(node, 0)}, {"flux", jsonVector(flux)}});
        }
        if (tablePrefix && _cellAuditCalls == tableCall)
        {
            std::ofstream stream(std::string(tablePrefix) + ".rank" + std::to_string(_mpi.rank) + ".json");
            stream << rows.dump() << '\n';
        }
        if (jetPrefix && _cellAuditCalls == tableCall)
        {
            std::ofstream stream(std::string(jetPrefix) + ".rank" + std::to_string(_mpi.rank) + ".json");
            stream << jets.dump() << '\n';
        }
        _lastRhsTiming.limiterSeconds += MPI_Wtime() - start;
        start = MPI_Wtime();
        ExchangeDifferentialSides();
        _pointValues.trans.startPersistentPull();
        _limiterFactors.trans.startPersistentPull();
        _pointValues.trans.waitPersistentPull();
        _limiterFactors.trans.waitPersistentPull();
        _lastRhsTiming.coefficientHaloSeconds += MPI_Wtime() - start;
        AuditCellLocal(means);
    }

    template <int dimension>
    void SpatialOperator<dimension>::ComputeCellLocalFluxJets() {}

    template <int dimension>
    typename SpatialOperator<dimension>::State SpatialOperator<dimension>::CellLocalSurfaceMean(index edge, int side) const
    {
        return _differentialTraces[edge][side];
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State SpatialOperator<dimension>::CellLocalIntegratedFlux(index edge, int side) const
    {
        return _differentialFluxes[edge][side];
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State SpatialOperator<dimension>::CellLocalBoundaryFlux(index node) const
    {
        return _differentialBoundary[node];
    }

    template <int dimension>
    void SpatialOperator<dimension>::AuditCellLocal(const NodeStatePair &means)
    {
        ++_cellAuditCalls;
        const char *path = std::getenv("NCFV_CELL_AUDIT_FILE");
        const int interval = std::getenv("NCFV_CELL_AUDIT_INTERVAL") ? std::atoi(std::getenv("NCFV_CELL_AUDIT_INTERVAL")) : 300;
        if (!path || (_cellAuditCalls > 6 && _cellAuditCalls % std::max(1, interval) != 0)) return;
        _cellBounds.trans.startPersistentPull();
        _cellBounds.trans.waitPersistentPull();
        index counts[13]{}, globalCounts[13]{};
        const char *faceTable = std::getenv("NCFV_P1_FACE_TABLE_PREFIX");
        nlohmann::json faceRows = nlohmann::json::array();
        counts[7] = _twoParameterSplitCells;
        counts[8] = _twoParameterPositiveCells;
        counts[9] = _p1GatedNodes;
        counts[10] = _p1FullWeightNodes;
        counts[11] = _p1ChangedFallbackNodes;
        counts[12] = _mesh->NumNode();
        counts[1] = _differentialBoundaryNonphysical;
        counts[5] = _differentialBoundaryViolations;
        real maxima[4]{_differentialMeanDefect, _differentialBoundaryExcess, _differentialMaximumDz, _twoParameterObjectiveExcess}, globalMaxima[4]{};
        real minima[4]{_differentialBoundaryMinDensity, _differentialBoundaryMinPressure,
                       _twoParameterMinimumAlpha, _twoParameterMinimumBeta}, globalMinima[4]{};
        for (index node = 0; node < _mesh->NumNode(); node++)
        {
            const State point = _pointValues[node];
            counts[0] += _limiterFactors[node](0) < 1;
            counts[1] += !IsPhysical(point);
            minima[0] = std::min(minima[0], point(0));
            minima[1] = std::min(minima[1], Pressure(point));
        }
        for (index edge = 0; edge < _topology.NumEdge(); edge++)
        {
            bool limited = false, violated = false;
            const auto &face = _geometry.EdgeSurface(edge);
            if (faceTable)
            {
                nlohmann::json sides = nlohmann::json::array();
                for (int side = 0; side < 2; side++)
                {
                    const auto x = _nodeHalo.Coordinate(face.nodes[side]);
                    const auto u = _differentialTraces[edge][side];
                    sides.push_back({{"coordinate", std::vector<real>(x.data(), x.data() + dimension)},
                        {"state", std::vector<real>(u.data(), u.data() + dimension + 2)},
                        {"triggered", _p1FallbackSides[edge][side]}});
                }
                faceRows.push_back({{"sides", sides}, {"uses_p1_pair", !_p1Traces.empty() &&
                    (_p1FallbackSides[edge][0] || _p1FallbackSides[edge][1])}});
            }
            for (int side = 0; side < 2; side++)
            {
                const index node = face.nodes[side];
                const State value = _differentialTraces[edge][side];
                const State lo = _cellBounds[node].col(0), hi = _cellBounds[node].col(1);
                const real excess = std::max(real(0), std::max((lo - value).maxCoeff(), (value - hi).maxCoeff()));
                maxima[1] = std::max(maxima[1], excess);
                minima[0] = std::min(minima[0], value(0));
                minima[1] = std::min(minima[1], Pressure(value));
                counts[1] += !IsPhysical(value);
                limited |= _limiterFactors[node](0) < 1;
                violated |= excess > 1e-10;
            }
            counts[2]++;
            counts[3] += limited;
            counts[4] += violated;
            counts[6] += violated && limited;
        }
        MPI_Allreduce(counts, globalCounts, 13, DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
        MPI_Allreduce(maxima, globalMaxima, 4, DNDS_MPI_REAL, MPI_MAX, _mpi.comm);
        MPI_Allreduce(minima, globalMinima, 4, DNDS_MPI_REAL, MPI_MIN, _mpi.comm);
        if (_mpi.rank == 0)
        {
            const nlohmann::json row{{"method", _reportFourPoint ? "REPORT_FOUR_POINT" : (_p1BothSides ? "BARTH_P1_BOTH_SIDES" : "BARTH_P1_EDGE_MIDPOINT")},
                {"componentwise_p1", _p1ComponentWise}, {"fallback_llf", _p1UseLLF},
                {"local_componentwise_p1", _p1LocalComponentWise}, {"componentwise_flux", _fluxComponentWise},
                {"local_p1_gated_nodes", globalCounts[9]}, {"local_p1_full_weight_nodes", globalCounts[10]},
                {"local_p1_changed_fallback_nodes", globalCounts[11]}, {"total_nodes", globalCounts[12]},
                {"p1_slope_limiter", _p1SlopeLimiter},
                {"p1_weight_bytes_per_rank0", _localLinearLimiter.WeightStorageBytes()},
                {"p1_pair_trace_bytes_per_rank0", _p1Traces.capacity() * sizeof(_p1Traces[0])},
                {"rhs_call", _cellAuditCalls}, {"limited_nodes", globalCounts[0]},
                {"nonphysical_traces", globalCounts[1]}, {"total_faces", globalCounts[2]},
                {"limited_faces", globalCounts[3]}, {"violating_faces", globalCounts[4]},
                {"violating_boundary_pieces", globalCounts[5]}, {"maximum_mean_defect", globalMaxima[0]},
                {"maximum_bound_excess", globalMaxima[1]}, {"maximum_abs_dU_dz_at_node", globalMaxima[2]},
                {"minimum_density", globalMinima[0]}, {"minimum_pressure", globalMinima[1]},
                {"violating_limited_faces", globalCounts[6]},
                {"violating_limited_fraction", globalCounts[3] ? real(globalCounts[6]) / globalCounts[3] : 0},
                {"different_alpha_beta_cells", globalCounts[7]}, {"positivity_contracted_cells", globalCounts[8]},
                {"minimum_alpha", globalMinima[2]}, {"minimum_beta", globalMinima[3]},
                {"maximum_normalized_qp_objective_excess", globalMaxima[3]},
                {"audit_boundary_traces", true}, {"gradient_only_solution_halo", true}};
            std::ofstream stream(path, std::ios::app);
            stream << row.dump() << '\n';
        }
        if (faceTable)
        {
            std::ofstream stream(std::string(faceTable) + ".rank" + std::to_string(_mpi.rank) + ".json");
            stream << faceRows.dump() << '\n';
        }
    }
}
