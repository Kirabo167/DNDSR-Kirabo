#include "NCFVSolver.hpp"
#include <algorithm>
#include <iomanip>

namespace DNDS::NCFVGPU
{
    template <int dimension>
    GPU::ResidentNCFVData Solver<dimension>::ExportResidentData() const
    {
        const auto &r = _configuration.reconstruction;
        GPU::ResidentNCFVData data;
        auto &s = data.settings;
        s.dimension = dimension;
        s.traditional = _configuration.algorithm.mode == IntegrationMode::TraditionalQuadrature;
        s.limiter = r.enableLimiter;
        s.limitRecovery = r.limitPointRecovery;
        s.binaryBarth = r.barthZeroOrderFallback;
        s.limitPressure = r.limitPressure;
        s.directionalBounds = r.directionalPairBounds;
        s.fluxCorrection = r.limitFluxCorrection;
        s.binaryFluxCorrection = r.fluxCorrectionBinary;
        s.entropyFluxCorrection = r.fluxCorrectionEntropy;
        s.pairFluxBounds = r.fluxCorrectionPairBounds;
        s.maximumFluxCorrection = r.maximumFluxCorrection;
        s.variational = r.method == ReconstructionMethod::Variational;
        s.variationalIterations = r.variationalIterations;
        s.variationalRelaxation = r.variationalRelaxation;
        s.sharedGradient = r.sharedGradientFaceSweep;
        s.sharedParallel = r.sharedGradientFaceSweepParallel;
        s.sharedBudget = r.sharedGradientBudget;
        const auto &viscous = _configuration.physics.viscous;
        s.viscous = viscous.enabled;
        s.viscosityModel = static_cast<int>(viscous.model);
        s.viscosity = viscous.dynamicViscosity;
        s.prandtl = viscous.prandtlNumber;
        s.gasConstant = viscous.gasConstant;
        s.referenceTemperature = viscous.referenceTemperature;
        s.sutherlandConstant = viscous.sutherlandConstant;
        s.viscousSpectralFactor = viscous.spectralRadiusFactor;
        const auto &time = _configuration.time;
        s.cflStep = time.useCFLTimeStep;
        s.localStep = time.useLocalTimeStep;
        s.cfl = time.cfl;
        s.fixedStep = time.timeStep;
        s.minimumStep = time.minimumTimeStep;
        s.maximumStep = time.maximumTimeStep;
        s.rejectNonphysical = time.rejectNonPhysicalSteps;
        s.maximumRetries = time.maximumStepRetries;
        s.gamma = _configuration.physics.gamma;
        for (int d = 0; d < 3; ++d)
            s.periodic[d] = _configuration.mesh.periodicLengths[d];
        data.ownedNodes = _mesh->NumNode();
        data.ownedEdges = _topology->NumEdge();
        data.nodes.resize(_nodeHalo->NumNodeProc());
        data.nodeGlobals.resize(data.nodes.size());
        data.initialState.resize(data.nodes.size() * 5);
        const auto addQuadrature = [&](const auto &points)
        {
            for (const auto &p : points)
            {
                GPU::ResidentQuadrature q;
                q.weight = p.weight;
                for (int d = 0; d < 3; ++d)
                {
                    q.coordinate[d] = p.coordinate(d);
                    q.normal[d] = p.vectorWeight(d) / p.weight;
                }
                data.quadrature.push_back(q);
            }
        };
        for (index node = 0; node < _nodeHalo->NumNodeProc(); ++node)
        {
            auto &out = data.nodes[node];
            data.nodeGlobals[node] = _nodeHalo->LocalToGlobal(node);
            out.volume = _nodeHalo->Volume(node);
            for (int d = 0; d < 3; ++d)
            {
                out.reference[d] = _nodeHalo->ReferenceLengths(node)(d);
                out.coordinate[d] = _nodeHalo->Coordinate(node)(d);
            }
            for (int k = 0; k < dimension + 2; ++k)
                data.initialState[node * 5 + k] = _state[node](k);
            if (node >= _mesh->NumNode())
                continue;
            const auto &op = _reconstruction->Operator(node);
            for (int m = 0; m < op.targetBasisMean.size(); ++m)
                out.basisMean[m] = op.targetBasisMean(m);
            out.recBegin = data.reconstruction.size();
            for (std::size_t j = 0; j < op.stencil.size(); ++j)
            {
                GPU::ResidentRecEntry entry;
                entry.node = op.stencil[j];
                for (int row = 0; row < op.inverseRows.rows(); ++row)
                    entry.weights[row] = op.inverseRows(row, j);
                data.reconstruction.push_back(entry);
            }
            out.recEnd = data.reconstruction.size();
            out.variationalBegin = data.variational.size();
            if (s.variational)
                for (const auto &p : _reconstruction->ResidentVariationalOperators(node))
                {
                    GPU::ResidentVariationalEntry entry;
                    entry.node = p.node;
                    for (int row = 0; row < p.meanJump.size(); ++row)
                    {
                        entry.jump[row] = p.meanJump(row);
                        for (int col = 0; col < p.coupling.cols(); ++col)
                            entry.coupling[row * 9 + col] = p.coupling(row, col);
                    }
                    data.variational.push_back(entry);
                }
            out.variationalEnd = data.variational.size();
            const auto &volume = _geometry->NodeVolume(node);
            out.recoveryBegin = data.recovery.size();
            for (const auto &p : volume.pointRecoveryStencil)
            {
                GPU::ResidentRecoveryEntry entry;
                entry.node = p.node;
                for (int d = 0; d < 3; ++d)
                    entry.weight[d] = p.gradientWeight(d);
                data.recovery.push_back(entry);
            }
            out.recoveryEnd = data.recovery.size();
            out.edgeBegin = data.incidence.size();
            for (const auto &p : _topology->Node2Edge(node))
            {
                const auto endpoints = _topology->Edge2Node()[p.edge];
                const index node0 = _nodeHalo->MeshLocalToLocal(endpoints[0]);
                const index node1 = _nodeHalo->MeshLocalToLocal(endpoints[1]);
                GPU::ResidentIncidence entry;
                entry.edge = p.edge;
                entry.sign = p.outwardSign;
                entry.side = node0 == node ? 0 : 1;
                entry.neighbor = entry.side == 0 ? node1 : node0;
                data.incidence.push_back(entry);
            }
            out.edgeEnd = data.incidence.size();
            out.boundaryBegin = data.boundaries.size();
            for (const auto &p : volume.boundaryPieces)
            {
                const auto &bc = _boundaries->Get(p.zone);
                GPU::ResidentBoundary entry;
                entry.anchor = node;
                entry.mode = static_cast<int>(bc.mode);
                entry.zone = p.zone;
                entry.measure = p.measure;
                entry.pressure = bc.staticPressure;
                entry.wallTemperature = bc.wallTemperature;
                entry.strong = bc.strongState;
                const State prescribed = PrimitiveToConservative(bc.primitive);
                for (int k = 0; k < dimension + 2; ++k)
                    entry.prescribed[k] = prescribed(k);
                for (int d = 0; d < 3; ++d)
                {
                    entry.normal[d] = p.vectorMeasure(d) / p.measure;
                    if (d < dimension)
                        entry.wallVelocity[d] = bc.wallVelocity[d];
                }
                entry.begin = data.boundarySupport.size();
                for (const auto &support : p.efficientStencil)
                    data.boundarySupport.push_back({support.node, support.integrationWeight});
                entry.end = data.boundarySupport.size();
                entry.quadBegin = data.quadrature.size();
                addQuadrature(p.quadrature);
                entry.quadEnd = data.quadrature.size();
                data.boundaries.push_back(entry);
            }
            out.boundaryEnd = data.boundaries.size();
        }
        data.edges.resize(_topology->NumEdgeProc());
        data.edgeGlobals.resize(data.edges.size());
        for (index edge = 0; edge < _topology->NumEdgeProc(); ++edge)
        {
            if (s.traditional && edge >= _topology->NumEdge())
            {
                auto &out = data.edges[edge];
                const auto endpoints = _topology->Edge2Node()[edge];
                data.edgeGlobals[edge] = _topology->EdgeIndexLocal2Global(edge);
                out.nodes[0] = _nodeHalo->MeshLocalToLocal(endpoints[0]);
                out.nodes[1] = _nodeHalo->MeshLocalToLocal(endpoints[1]);
                for (int d = 0; d < 3; ++d)
                    out.vectorMeasure[d] = _geometry->EdgeMetrics()[edge](d);
                out.measure = _geometry->EdgeMetrics()[edge](3);
                continue;
            }
            const auto &surface = _geometry->EdgeSurface(edge);
            auto &out = data.edges[edge];
            data.edgeGlobals[edge] = _topology->EdgeIndexLocal2Global(edge);
            out.nodes[0] = surface.nodes[0];
            out.nodes[1] = surface.nodes[1];
            out.measure = surface.measure;
            for (int d = 0; d < 3; ++d)
                out.vectorMeasure[d] = surface.vectorMeasure(d);
            out.begin = data.surface.size();
            for (const auto &p : surface.efficientStencil)
            {
                GPU::ResidentSurfaceEntry entry;
                entry.node = p.node;
                entry.gradientWeight = p.gradientWeight;
                for (int side = 0; side < 2; ++side)
                    for (int d = 0; d < 3; ++d)
                    {
                        entry.stateWeight[side][d] = p.stateGradientWeights[side](d);
                        for (int direction = 0; direction < 3; ++direction)
                            entry.fluxWeight[side][d * 3 + direction] = p.fluxGradientWeights[side](d, direction);
                    }
                data.surface.push_back(entry);
            }
            out.end = data.surface.size();
            out.quadBegin = data.quadrature.size();
            addQuadrature(surface.quadrature);
            out.quadEnd = data.quadrature.size();
        }
        if (s.variational)
        {
            std::vector<index> level(data.ownedNodes, 0);
            index maximum = 0;
            for (index node = 0; node < data.ownedNodes; ++node)
            {
                for (index p = data.nodes[node].variationalBegin; p < data.nodes[node].variationalEnd; ++p)
                    if (data.variational[p].node < node)
                        level[node] = std::max(level[node], level[data.variational[p].node] + 1);
                maximum = std::max(maximum, level[node]);
            }
            std::vector<std::vector<index>> groups(maximum + 1);
            for (index node = 0; node < data.ownedNodes; ++node)
                groups[level[node]].push_back(node);
            data.variationalLevels.push_back(0);
            for (const auto &group : groups)
            {
                data.variationalNodes.insert(data.variationalNodes.end(), group.begin(), group.end());
                data.variationalLevels.push_back(data.variationalNodes.size());
            }
        }
        if (s.sharedGradient)
            _spatial->ExportResidentSharedGradient(data);
        return data;
    }

    template <int dimension>
    void Solver<dimension>::DownloadResidentOutput()
    {
        std::vector<double> state, steps;
        _resident->Download(state, &steps);
        for (index i = 0; i < _mesh->NumNode(); ++i)
            for (int k = 0; k < dimension + 2; ++k)
                _state[i](k) = state[i * 5 + k];
        _spatial->SetOutputTimeSteps(steps);
    }

    template <int dimension>
    void Solver<dimension>::RunResident()
    {
        _resident = std::make_unique<GPU::ResidentNCFV>(ExportResidentData(), _mpi.comm);
        const auto &io = _configuration.io;
        const auto &time = _configuration.time;
        double residual = _resident->EvaluateResidual();
        if (io.writeVTK && io.writeInitial)
        {
            DownloadResidentOutput();
            WriteOutput(StampedName(io.outputPrefix, _currentIteration));
        }
        for (index i = 0; i < time.iterations; ++i)
        {
            if (time.endTime >= 0 && _simulationTime >= time.endTime - 1e-13)
                break;
            const double maximumStep = time.endTime >= 0 ? time.endTime - _simulationTime : veryLargeReal;
            const auto report = _resident->Advance(maximumStep);
            residual = report.residual;
            ++_currentIteration;
            _simulationTime += report.minimumStep;
            if (_mpi.rank == 0 && (_currentIteration == 1 || _currentIteration % time.reportInterval == 0 || i + 1 == time.iterations))
                log() << "Resident CUDA NCFV iteration=" << _currentIteration << ", residual RMS=" << std::setprecision(12) << residual
                      << ", dt[min,max]=[" << report.minimumStep << "," << report.maximumStep << "] time=" << _simulationTime << std::endl;
            const bool write = io.writeVTK && io.outputInterval > 0 && _currentIteration % io.outputInterval == 0;
            const bool restart = io.restartInterval > 0 && _currentIteration % io.restartInterval == 0;
            if (write || restart)
                DownloadResidentOutput();
            if (write)
                WriteOutput(StampedName(io.outputPrefix, _currentIteration));
            if (restart)
                WriteRestart(StampedName(io.restartPrefix, _currentIteration));
        }
        const bool finalOutput = io.writeVTK && io.writeFinal && _lastOutputIteration != _currentIteration;
        const bool finalRestart = io.writeFinalRestart && _lastRestartIteration != _currentIteration;
        if (finalOutput || finalRestart)
            DownloadResidentOutput();
        if (finalOutput)
            WriteOutput(StampedName(io.outputPrefix, _currentIteration));
        if (finalRestart)
            WriteRestart(StampedName(io.restartPrefix, _currentIteration));
        _resident->Report(_mpi.rank);
    }

    template GPU::ResidentNCFVData Solver<DNDS_NCFV_INSTANTIATION_DIM>::ExportResidentData() const;
    template void Solver<DNDS_NCFV_INSTANTIATION_DIM>::DownloadResidentOutput();
    template void Solver<DNDS_NCFV_INSTANTIATION_DIM>::RunResident();
}
