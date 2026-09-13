#include "NCFVSolver.hpp"
#include "NCFVAnalytic.hpp"

#include "DNDS/Errors.hpp"
#include "DNDS/ExprtkWrapper.hpp"
#include "DNDS/OMP.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace DNDS::NCFV
{
    template <int dimension>
    void Solver<dimension>::BroadcastBoundaryNameMap()
    {
        std::string packed;
        if (_mpi.rank == 0)
        {
            nlohmann::ordered_json object = nlohmann::ordered_json::object();
            for (const auto &[name, id] : _boundaryNameToID)
                object[name] = id;
            packed = object.dump();
        }
        index length = static_cast<index>(packed.size());
        MPI_Bcast(&length, 1, DNDS_MPI_INDEX, 0, _mpi.comm);
        DNDS_check_throw_info(length >= 0,
                              "NCFV boundary-name broadcast has a negative length");
        packed.resize(static_cast<std::size_t>(length));
        if (length > 0)
            MPI_Bcast(packed.data(), static_cast<int>(length), MPI_CHAR, 0, _mpi.comm);

        if (_mpi.rank != 0)
        {
            _boundaryNameToID.clear();
            const auto object = nlohmann::ordered_json::parse(packed);
            for (const auto &[name, value] : object.items())
                _boundaryNameToID.emplace(name, value.get<Geom::t_index>());
        }
    }

    template <int dimension>
    void Solver<dimension>::ReadMesh()
    {
        DNDS_MAKE_SSP(_mesh, _mpi, dimension);
        DNDS_MAKE_SSP(_reader, _mesh, 0);

        Geom::AutoAppendName2ID nameMapper;
        Geom::t_FBCName_2_ID resolveBoundaryName =
            [&nameMapper](const std::string &name)
        {
            return nameMapper(name);
        };
        _reader->ReadFromCGNSSerial(
            _configuration.mesh.meshFile, resolveBoundaryName);
        _boundaryNameToID = nameMapper.n2id_map;
        BroadcastBoundaryNameMap();

        _reader->Deduplicate1to1Periodic(_configuration.mesh.periodicTolerance);
        _reader->BuildCell2Cell();
        _reader->MeshPartitionCell2Cell(_configuration.mesh.partitionOptions);
        _reader->PartitionReorderToMeshCell2Cell();
        Geom::BuildGhostPrimary(*_mesh, _configuration.mesh.ghostLayers);

        Geom::PrepareMeshOptions options;
        options.reorderCells = _configuration.mesh.reorderCells;
#ifdef DNDS_USE_OMP
        options.reorderParts = std::max(get_env_DNDS_DIST_OMP_NUM_THREADS(), 1);
#else
        options.reorderParts = 1;
#endif
        options.buildSerialOut = false;
        Geom::PrepareMesh(*_mesh, *_reader, options);

        DNDS_check_throw_info(
            !_mesh->isPeriodic,
            "NCFV rejects periodic meshes until edge-frame transforms are implemented");
    }

    template <int dimension>
    typename Solver<dimension>::State
    Solver<dimension>::PrimitiveToConservative(
        const std::vector<real> &primitive) const
    {
        State conservative = State::Zero();
        const real density = primitive[0];
        Eigen::Vector<real, dimension> velocity;
        for (int i = 0; i < dimension; i++)
            velocity(i) = primitive[static_cast<std::size_t>(i + 1)];
        const real pressure = primitive[static_cast<std::size_t>(dimension + 1)];
        conservative(0) = density;
        conservative.template segment<dimension>(1) = density * velocity;
        conservative(dimension + 1) =
            pressure / (_configuration.physics.gamma - 1.0) +
            0.5 * density * velocity.squaredNorm();
        return conservative;
    }

    template <int dimension>
    typename Solver<dimension>::State
    Solver<dimension>::ConservativeToPrimitive(
        const State &conservative) const
    {
        State primitive = State::Zero();
        primitive(0) = conservative(0);
        primitive.template segment<dimension>(1) =
            conservative.template segment<dimension>(1) / conservative(0);
        primitive(dimension + 1) =
            (_configuration.physics.gamma - 1.0) *
            (conservative(dimension + 1) -
             0.5 * conservative.template segment<dimension>(1).squaredNorm() /
                 conservative(0));
        return primitive;
    }

    template <int dimension>
    bool Solver<dimension>::StateIsPhysical(const State &state) const
    {
        if (!state.allFinite() || state(0) <= 1e-12)
            return false;
        const real pressure = (_configuration.physics.gamma - 1.0) *
                              (state(dimension + 1) -
                               0.5 * state.template segment<dimension>(1).squaredNorm() /
                                   state(0));
        return pressure > 1e-12;
    }

    template <int dimension>
    void Solver<dimension>::CheckOwnedState(
        const NodeStatePair &state,
        const std::string &where) const
    {
        index localBad = 0;
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            localBad += StateIsPhysical(State(state[iNode])) ? 0 : 1;
        index globalBad = 0;
        MPI_Allreduce(&localBad, &globalBad, 1,
                      DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
        DNDS_check_throw_info(
            globalBad == 0,
            fmt::format("NCFV has {} nonphysical owned states after {}",
                        globalBad, where));
    }

    template <int dimension>
    void Solver<dimension>::AllocateFields()
    {
        CFV::BuildUDofOnMesh(
            _state, "NCFV.state", _mpi, _mesh, dimension + 2,
            true, true, Geom::MeshLoc::Node);
        CFV::BuildUDofOnMesh(
            _stageState, "NCFV.stageState", _mpi, _mesh, dimension + 2,
            true, true, Geom::MeshLoc::Node);
        CFV::BuildUDofOnMesh(
            _baseState, "NCFV.baseState", _mpi, _mesh, dimension + 2,
            false, false, Geom::MeshLoc::Node);
        CFV::BuildUDofOnMesh(
            _rhs, "NCFV.rhs", _mpi, _mesh, dimension + 2,
            false, false, Geom::MeshLoc::Node);
        for (index iNode = 0; iNode < _state.Size(); iNode++)
            _state[iNode].setZero();
        for (index iNode = 0; iNode < _stageState.Size(); iNode++)
            _stageState[iNode].setZero();
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            _baseState[iNode].setZero();
            _rhs[iNode].setZero();
        }
    }

    template <int dimension>
    void Solver<dimension>::ApplyInitialRegions()
    {
        for (const auto &box : _configuration.initialField.boxes)
        {
            const State value = PrimitiveToConservative(box.primitive);
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            {
                const Vector3 point = _mesh->coords[iNode];
                if (point.x() >= box.xMin && point.x() <= box.xMax &&
                    point.y() >= box.yMin && point.y() <= box.yMax &&
                    point.z() >= box.zMin && point.z() <= box.zMax)
                    _state[iNode] = value;
            }
        }

        for (const auto &plane : _configuration.initialField.planes)
        {
            const State value = PrimitiveToConservative(plane.primitive);
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            {
                const Vector3 point = _mesh->coords[iNode];
                if (plane.a * point.x() + plane.b * point.y() +
                        plane.c * point.z() + plane.h >=
                    0)
                    _state[iNode] = value;
            }
        }

        for (const auto &expression : _configuration.initialField.expressions)
        {
            ExprtkWrapperEvaluator evaluator;
            evaluator.AddScalar("inRegion");
            evaluator.AddScalar("globalNode");
            evaluator.AddVector("x", 3);
            evaluator.AddVector("UPrim", dimension + 2);
            evaluator.Compile(expression.GetProgram());
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            {
                const Vector3 point = _mesh->coords[iNode];
                State primitive = ConservativeToPrimitive(State(_state[iNode]));
                evaluator.Var("inRegion") = 0;
                evaluator.Var("globalNode") =
                    static_cast<real>(_mesh->node2nodeOrig(iNode, 0));
                for (int i = 0; i < 3; i++)
                    evaluator.VarVec("x", i) = point(i);
                for (int i = 0; i < dimension + 2; i++)
                    evaluator.VarVec("UPrim", i) = primitive(i);
                const real returnCode = evaluator.Evaluate();
                DNDS_check_throw_info(
                    returnCode == 0,
                    "NCFV initial-field ExprTk program must return zero:\n" +
                        expression.GetProgram());
                if (evaluator.Var("inRegion") == 0)
                    continue;
                std::vector<real> values(static_cast<std::size_t>(dimension + 2));
                for (int i = 0; i < dimension + 2; i++)
                    values[static_cast<std::size_t>(i)] =
                        evaluator.VarVec("UPrim", i);
                const State value = PrimitiveToConservative(values);
                DNDS_check_throw_info(
                    StateIsPhysical(value),
                    fmt::format("NCFV initial expression produced a nonphysical state at global node {}",
                                _mesh->NodeIndexLocal2Global(iNode)));
                _state[iNode] = value;
            }
        }
    }

    template <int dimension>
    void Solver<dimension>::ReadInitialNodeFile()
    {
        const auto &settings = _configuration.initialField;
        if (settings.nodeFile.empty())
            return;

        std::ifstream input(settings.nodeFile);
        DNDS_check_throw_info(
            input.good(),
            "NCFV initial node field does not exist: " + settings.nodeFile);

        std::unordered_map<index, index> globalToLocal;
        globalToLocal.reserve(static_cast<std::size_t>(_mesh->NumNode()));
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            globalToLocal.emplace(_mesh->node2nodeOrig(iNode, 0), iNode);
        std::vector<bool> loaded(static_cast<std::size_t>(_mesh->NumNode()), false);

        std::string line;
        index lineNumber = 0;
        while (std::getline(input, line))
        {
            lineNumber++;
            const auto comment = line.find('#');
            if (comment != std::string::npos)
                line.erase(comment);
            std::replace(line.begin(), line.end(), ',', ' ');
            std::replace(line.begin(), line.end(), ';', ' ');
            std::istringstream parser(line);
            index fileNode = 0;
            if (!(parser >> fileNode))
                continue; // blank line or a single optional header line
            const index globalNode =
                fileNode - static_cast<index>(settings.nodeFileIndexBase);
            const auto found = globalToLocal.find(globalNode);
            std::vector<real> values(static_cast<std::size_t>(dimension + 2));
            bool complete = true;
            for (real &value : values)
                complete = complete && static_cast<bool>(parser >> value);
            DNDS_check_throw_info(
                complete,
                fmt::format("NCFV initial node field line {} has fewer than {} values",
                            lineNumber, dimension + 2));
            if (found == globalToLocal.end())
                continue;
            const index iNode = found->second;
            DNDS_check_throw_info(
                !loaded[static_cast<std::size_t>(iNode)],
                fmt::format("NCFV initial node field repeats global node {}",
                            globalNode));
            State value;
            if (settings.nodeFileVariables == InitialFieldVariables::Primitive)
                value = PrimitiveToConservative(values);
            else
                for (int i = 0; i < dimension + 2; i++)
                    value(i) = values[static_cast<std::size_t>(i)];
            DNDS_check_throw_info(
                StateIsPhysical(value),
                fmt::format("NCFV initial node field is nonphysical at global node {}",
                            globalNode));
            _state[iNode] = value;
            loaded[static_cast<std::size_t>(iNode)] = true;
        }

        index localMissing = 0;
        index localLoaded = 0;
        for (bool value : loaded)
        {
            localLoaded += value ? 1 : 0;
            localMissing += value ? 0 : 1;
        }
        index globalMissing = 0;
        index globalLoaded = 0;
        MPI_Allreduce(&localMissing, &globalMissing, 1,
                      DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
        MPI_Allreduce(&localLoaded, &globalLoaded, 1,
                      DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
        DNDS_check_throw_info(
            !settings.requireCompleteNodeFile || globalMissing == 0,
            fmt::format("NCFV initial node field is missing {} owned global nodes",
                        globalMissing));
        if (_mpi.rank == 0)
            log() << "NCFV read initial node field: " << settings.nodeFile
                  << ", loaded owned nodes=" << globalLoaded << std::endl;
    }

    template <int dimension>
    void Solver<dimension>::InitializeFreshField()
    {
        const State initial = PrimitiveToConservative(
            _configuration.physics.initialPrimitive);
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            _state[iNode] = initial;
        ApplyInitialRegions();
        ReadInitialNodeFile();
        if (_configuration.initialField.isentropicVortex)
            InitializeVortex();
        if (_periodic)
            _periodic->Average(_state);
        _currentIteration = 0;
        _simulationTime = 0;
    }

    template <int dimension>
    void Solver<dimension>::SynchronizeState(NodeStatePair &state)
    {
        state.trans.startPersistentPull();
        state.trans.waitPersistentPull();
    }

    template <int dimension>
    void Solver<dimension>::InitializeVortex()
    {
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const auto &volume = _geometry->NodeVolume(iNode);
            if (_configuration.algorithm.mode == IntegrationMode::EfficientDifferential)
            {
                _state[iNode] = IsentropicVortex<dimension>(_configuration, _mesh->coords[iNode], 0).first;
                for (const auto &weight : volume.pointRecoveryWeights)
                    _state[iNode] += IsentropicVortex<dimension>(
                        _configuration, _mesh->coords[weight.node], 0).second.transpose() *
                        weight.value.head(dimension);
            }
            else
            {
                _state[iNode].setZero();
                for (const auto &point : volume.volumeQuadrature)
                    _state[iNode] += point.weight / volume.moments.measure *
                        IsentropicVortex<dimension>(_configuration, point.coordinate, 0).first;
            }
        }
    }

    template <int dimension>
    void Solver<dimension>::WriteVortexDiagnostics(bool initial)
    {
        if (!_configuration.initialField.isentropicVortex)
            return;
        EvaluateResidual(); // Refresh point recovery and the final-time output fields.
        const std::filesystem::path nodeFile = StampedName(_configuration.io.outputPrefix, _currentIteration) +
            fmt::format(".nodes.rank{:04d}.csv", _mpi.rank);
        std::filesystem::create_directories(nodeFile.parent_path() / ".");
        std::ofstream nodeOutput(nodeFile);
        nodeOutput << "original_node,x,y,z,partial_volume,rho_mean,rho_point,rho_exact,rhou,rhov,rhow,rhoE\n";
        nodeOutput << std::setprecision(17);
        real local[9]{};
        real localMaximum = 0;
        for (index i = 0; i < _mesh->NumNode(); i++)
        {
            const real volume = _geometry->NodeVolume(i).moments.measure;
            const State recovered = _spatial->PointValues()[i];
            const State exact = IsentropicVortex<dimension>(_configuration, _mesh->coords[i], _simulationTime).first;
            const real error = recovered(0) - exact(0);
            const Vector3 coordinate = _mesh->coords[i];
            nodeOutput << _mesh->node2nodeOrig(i, 0) << ',' << coordinate(0) << ',' << coordinate(1) << ','
                       << coordinate(2) << ',' << volume << ',' << _state[i](0) << ',' << recovered(0) << ',' << exact(0);
            for (int d = 0; d < 3; d++)
                nodeOutput << ',' << (d < dimension ? _state[i](d + 1) : 0.0);
            nodeOutput << ',' << _state[i](dimension + 1) << '\n';
            local[0] += volume;
            local[1] += volume * std::abs(error);
            local[2] += volume * error * error;
            localMaximum = std::max(localMaximum, std::abs(error));
            local[3] += volume * _state[i](0);
            for (int d = 0; d < dimension; d++)
                local[4 + d] += volume * _state[i](1 + d);
            local[7] += volume * _state[i](dimension + 1);
            const real p = (_configuration.physics.gamma - 1.0) *
                (recovered(dimension + 1) - 0.5 * recovered.template segment<dimension>(1).squaredNorm() / recovered(0));
            local[8] += volume * std::abs(p / std::pow(recovered(0), _configuration.physics.gamma) - 1.0);
        }
        real global[9]{}, maximum = 0;
        MPI_Allreduce(local, global, 9, DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
        MPI_Allreduce(&localMaximum, &maximum, 1, DNDS_MPI_REAL, MPI_MAX, _mpi.comm);
        if (_mpi.rank == 0)
        {
            const std::filesystem::path file = _configuration.io.outputPrefix + ".diagnostics.csv";
            std::filesystem::create_directories(file.parent_path() / ".");
            std::ofstream out(file, initial ? std::ios::out : std::ios::app);
            if (initial)
                out << "iteration,time,volume,rho_point_L1,rho_point_L2,rho_point_Linf,mass,momentum_x,momentum_y,momentum_z,total_energy,entropy_L1\n";
            out << std::setprecision(17) << _currentIteration << ',' << _simulationTime << ',' << global[0]
                << ',' << global[1] / global[0] << ',' << std::sqrt(global[2] / global[0]) << ',' << maximum;
            for (int v = 0; v < 5; v++)
                out << ',' << global[3 + v];
            out << ',' << global[8] / global[0] << '\n';
            log() << "NCFV vortex t=" << _simulationTime << ", recovered-point density L2="
                  << std::sqrt(global[2] / global[0]) << ", mass=" << global[3] << std::endl;
        }
    }

    template <int dimension>
    std::string Solver<dimension>::StampedName(
        const std::string &prefix,
        index iteration) const
    {
        std::ostringstream result;
        result << prefix << "_" << std::setfill('0') << std::setw(8)
               << iteration;
        return result.str();
    }

    template <int dimension>
    void Solver<dimension>::WriteRestart(const std::string &baseName)
    {
        const auto [fileName, displayName] =
            _configuration.io.restartSerializer.ModifyFilePath(
                baseName, _mpi, "%06d", false);
        const auto serializer =
            _configuration.io.restartSerializer.BuildSerializer(_mpi);
        serializer->OpenFile(fileName, false);
        serializer->WriteInt("dimension", dimension);
        serializer->WriteIndex("iteration", _currentIteration);
        serializer->WriteReal("simulationTime", _simulationTime);
        serializer->WriteReal("gamma", _configuration.physics.gamma);
        serializer->WriteString("nodeIdentity", "CGNSOriginalNodeIndex");
        serializer->WriteString(
            "integrationMode",
            nlohmann::json(_configuration.algorithm.mode).get<std::string>());
        serializer->WriteString(
            "configuration", nlohmann::ordered_json(_configuration).dump());

        std::vector<index> globalNodes(static_cast<std::size_t>(_mesh->NumNode()));
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            globalNodes[static_cast<std::size_t>(iNode)] =
                _mesh->node2nodeOrig(iNode, 0);
        if (serializer->IsPerRank())
            serializer->WriteIndexVector("nodeIdentityOrder", globalNodes, Serializer::ArrayGlobalOffset_Parts);
        _state.WriteSerialize(
            serializer, "state", globalNodes,
            /*includePIG*/ false, /*includeSon*/ false);
        serializer->CloseFile();
        _lastRestartIteration = _currentIteration;
        if (_mpi.rank == 0)
            log() << "NCFV wrote restart: " << displayName << std::endl;
    }

    template <int dimension>
    void Solver<dimension>::ReadRestart(const std::string &baseName)
    {
        const auto [fileName, displayName] =
            _configuration.io.restartSerializer.ModifyFilePath(
                baseName, _mpi, "%06d", true);
        const auto serializer =
            _configuration.io.restartSerializer.BuildSerializer(_mpi);
        serializer->OpenFile(fileName, true);
        int storedDimension = 0;
        real storedGamma = 0;
        serializer->ReadInt("dimension", storedDimension);
        serializer->ReadIndex("iteration", _currentIteration);
        serializer->ReadReal("simulationTime", _simulationTime);
        serializer->ReadReal("gamma", storedGamma);
        std::string identity;
        serializer->ReadString("nodeIdentity", identity);
        DNDS_check_throw_info(identity == "CGNSOriginalNodeIndex", "NCFV restart node identity is incompatible");
        DNDS_check_throw_info(storedDimension == dimension,
                              "NCFV restart dimension differs from the current solver");
        DNDS_check_throw_info(
            std::abs(storedGamma - _configuration.physics.gamma) < 1e-13,
            "NCFV restart gamma differs from the current configuration");

        if (serializer->IsPerRank())
        {
            std::vector<index> originalIndices;
            Serializer::ArrayGlobalOffset offset = Serializer::ArrayGlobalOffset_Parts;
            serializer->ReadIndexVector("nodeIdentityOrder", originalIndices, offset);
            DNDS_check_throw_info(originalIndices.size() == static_cast<std::size_t>(_mesh->NumNode()),
                                  "NCFV JSON restart node count differs");
            for (index i = 0; i < _mesh->NumNode(); i++)
                DNDS_check_throw_info(originalIndices[i] == _mesh->node2nodeOrig(i, 0),
                                      "NCFV JSON restart partition differs; use H5 to repartition");
            NodeStatePair readState;
            readState.InitPair("NCFV.restartRead", _mpi);
            readState.ReadSerialize(
                serializer, "state",
                /*includePIG*/ false, /*includeSon*/ false);
            DNDS_check_throw_info(
                readState.father->Size() == _state.father->Size() &&
                    readState.father->MatRowSize() == dimension + 2,
                "NCFV JSON restart requires the same MPI partition and variable count");
            _state.CopyFather(readState);
        }
        else
        {
            std::vector<index> globalNodes(static_cast<std::size_t>(_mesh->NumNode()));
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
                globalNodes[static_cast<std::size_t>(iNode)] =
                    _mesh->node2nodeOrig(iNode, 0);
            _state.ReadSerializeRedistributed(
                serializer, "state", globalNodes);
        }
        serializer->CloseFile();
        if (_mpi.rank == 0)
            log() << "NCFV read restart: " << displayName
                  << ", iteration=" << _currentIteration
                  << ", time=" << _simulationTime << std::endl;
    }

    template <int dimension>
    void Solver<dimension>::WriteOutput(const std::string &baseName)
    {
        SynchronizeState(_state);
        _reader->SetASCIIPrecision(_configuration.io.asciiPrecision);
        _reader->SetVTKFloatEncodeMode(_configuration.io.vtkFloatEncoding);

        const std::array<std::string, 6> scalarNames{
            "Density", "Pressure", "Temperature", "Mach",
            "TotalEnergy", "LocalTimeStep"};
        const auto scalarValue = [&](int field, index iNode)
        {
            const State conservative = _state[iNode];
            const State primitive = ConservativeToPrimitive(conservative);
            const auto velocity = primitive.template segment<dimension>(1);
            const real pressure = primitive(dimension + 1);
            const real temperature =
                pressure /
                (primitive(0) * _configuration.physics.viscous.gasConstant);
            const real acousticSpeed = std::sqrt(
                _configuration.physics.gamma * pressure / primitive(0));
            switch (field)
            {
            case 0:
                return primitive(0);
            case 1:
                return pressure;
            case 2:
                return temperature;
            case 3:
                return velocity.norm() / acousticSpeed;
            case 4:
                return conservative(dimension + 1);
            case 5:
                return _spatial->LocalTimeStep(iNode);
            default:
                return 0.0;
            }
        };

        _reader->PrintSerialPartVTKDataArray(
            baseName, _configuration.io.vtkSeriesName,
            0, 0, static_cast<int>(scalarNames.size()), 1,
            [](int) { return std::string{}; },
            [](int, index) { return 0.0; },
            [](int) { return std::string{}; },
            [](int, index, int) { return 0.0; },
            [&](int field) { return scalarNames.at(static_cast<std::size_t>(field)); },
            scalarValue,
            [](int) { return std::string("Velocity"); },
            [&](int, index iNode, int component)
            {
                if (component >= dimension)
                    return 0.0;
                const State conservative = _state[iNode];
                return conservative(1 + component) / conservative(0);
            },
            _simulationTime, 1);
        _lastOutputIteration = _currentIteration;
        if (_mpi.rank == 0)
            log() << "NCFV wrote distributed VTK: " << baseName
                  << ".pvtu" << std::endl;
    }

    template <int dimension>
    void Solver<dimension>::WriteResolvedConfiguration() const
    {
        const auto &io = _configuration.io;
        if (!io.writeResolvedConfiguration ||
            !(io.writeVTK || io.restartInterval > 0 || io.writeFinalRestart))
            return;
        if (_mpi.rank == 0)
        {
            const std::filesystem::path fileName =
                io.outputPrefix + ".resolved.json";
            std::filesystem::create_directories(fileName.parent_path() / ".");
            std::ofstream output(fileName);
            DNDS_check_throw_info(output.good(),
                                  "NCFV cannot write resolved configuration");
            output << std::setw(4) << nlohmann::ordered_json(_configuration);
        }
        MPI_Barrier(_mpi.comm);
    }

    template <int dimension>
    void Solver<dimension>::Initialize()
    {
        _configuration.Validate();
        ReadMesh();

        _topology = std::make_unique<Topology>(_mpi, _mesh);
        _topology->Build();
        _geometry = std::make_unique<DualGeometry>(
            _mpi, _mesh, *_topology, _configuration.algorithm,
            _configuration.reconstruction.enableLimiter);
        _geometry->Build();
        if (_configuration.mesh.periodicLengths[0] > 0)
        {
            _periodic = std::make_unique<PeriodicNodes>(_mpi, _mesh, *_geometry, _configuration.mesh);
            _periodic->Build(*_topology, _configuration.mesh.periodicTolerance);
        }
        _reconstruction = std::make_unique<Reconstruction>(
            _mpi, _mesh, *_topology, *_geometry,
            _configuration.algorithm.mode,
            _configuration.reconstruction, _periodic.get());
        _reconstruction->Build();
        _boundaries = std::make_unique<BoundaryRegistry>(
            _mpi, dimension, _configuration.physics);
        _boundaries->Build(
            _boundaryNameToID, *_topology, _configuration.physics);

        AllocateFields();
        _spatial = std::make_unique<SpatialOperator<dimension>>(
            _mpi, _mesh, *_topology, *_geometry, *_reconstruction,
            *_boundaries, _configuration.algorithm.mode,
            _configuration.reconstruction,
            _configuration.physics, _configuration.time, _periodic.get());
        _spatial->Initialize();

        if (_configuration.io.restartInput.empty())
            InitializeFreshField();
        else
            ReadRestart(_configuration.io.restartInput);
        _spatial->ApplyStrongBoundaryConditions(_state);
        SynchronizeState(_state);
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            _stageState[iNode] = _state[iNode];
            _baseState[iNode] = _state[iNode];
            _rhs[iNode].setZero();
        }
        CheckOwnedState(_state, "initialization");
        WriteResolvedConfiguration();

        if (_mpi.rank == 0)
            log() << "NCFV (" << MethodName << ") initialized: dimension=" << dimension
                  << ", global nodes=" << _mesh->NumNodeGlobal()
                  << ", global cells=" << _mesh->NumCellGlobal()
                  << ", global edges=" << _topology->NumEdgeGlobal()
                  << ", viscous=" << _configuration.physics.viscous.enabled
                  << std::endl;
    }

    template <int dimension>
    void Solver<dimension>::Run()
    {
        const index additionalIterations = _configuration.time.iterations;
        WriteVortexDiagnostics(true);
        real residual = 0;
        if (_configuration.io.writeInitial || additionalIterations == 0)
            residual = EvaluateResidual();

        if (_configuration.io.writeVTK && _configuration.io.writeInitial)
            WriteOutput(StampedName(
                _configuration.io.outputPrefix, _currentIteration));

        for (index additional = 0; additional < additionalIterations; additional++)
        {
            if (_configuration.time.endTime >= 0 &&
                _simulationTime >= _configuration.time.endTime - 1e-13)
                break;
            _lastStepTiming = {};
            const double stepStart = MPI_Wtime();
            _spatial->SetMaximumStep(_configuration.time.endTime >= 0 ?
                _configuration.time.endTime - _simulationTime : veryLargeReal);
            const double baseCopyStart = MPI_Wtime();
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
                _baseState[iNode] = _state[iNode];
            _lastStepTiming.baseStateCopySeconds = MPI_Wtime() - baseCopyStart;

            _spatial->ResetRiemannSolverCallCount();
            residual = _spatial->EvaluateRHS(_state, _rhs);
            _lastStepTiming.rhs += _spatial->LastRhsTiming();
            const real physicalStep = _spatial->LastMinimumTimeStep();
            std::vector<real> stepSize(static_cast<std::size_t>(_mesh->NumNode()));
            double stageUpdateStart = MPI_Wtime();
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            {
                stepSize[static_cast<std::size_t>(iNode)] =
                    _spatial->LocalTimeStep(iNode);
                _stageState[iNode] =
                    _baseState[iNode] +
                    stepSize[static_cast<std::size_t>(iNode)] * _rhs[iNode];
            }
            _spatial->ApplyStrongBoundaryConditions(_stageState);
            CheckOwnedState(_stageState, "SSPRK3 stage 1");
            _lastStepTiming.stageUpdateSeconds += MPI_Wtime() - stageUpdateStart;

            residual = _spatial->EvaluateRHS(_stageState, _rhs);
            _lastStepTiming.rhs += _spatial->LastRhsTiming();
            stageUpdateStart = MPI_Wtime();
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
                _stageState[iNode] =
                    0.75 * _baseState[iNode] +
                    0.25 * (_stageState[iNode] +
                            stepSize[static_cast<std::size_t>(iNode)] * _rhs[iNode]);
            _spatial->ApplyStrongBoundaryConditions(_stageState);
            CheckOwnedState(_stageState, "SSPRK3 stage 2");
            _lastStepTiming.stageUpdateSeconds += MPI_Wtime() - stageUpdateStart;

            residual = _spatial->EvaluateRHS(_stageState, _rhs);
            _lastStepTiming.rhs += _spatial->LastRhsTiming();
            stageUpdateStart = MPI_Wtime();
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
                _state[iNode] =
                    (1.0 / 3.0) * _baseState[iNode] +
                    (2.0 / 3.0) *
                        (_stageState[iNode] +
                         stepSize[static_cast<std::size_t>(iNode)] * _rhs[iNode]);
            _spatial->ApplyStrongBoundaryConditions(_state);
            CheckOwnedState(_state, "SSPRK3 stage 3");
            _lastStepRiemannSolverCalls = _spatial->RiemannSolverCallCount();
            _lastStepTiming.stageUpdateSeconds += MPI_Wtime() - stageUpdateStart;
            _lastStepTiming.totalSeconds = MPI_Wtime() - stepStart;

            _currentIteration++;
            _simulationTime += physicalStep;
            if (_mpi.rank == 0 &&
                (_currentIteration == 1 ||
                 _currentIteration % _configuration.time.reportInterval == 0 ||
                 additional + 1 == additionalIterations))
                log() << "NCFV iteration=" << _currentIteration
                      << ", residual RMS=" << std::setprecision(12) << residual
                      << ", dt[min,max]=[" << _spatial->LastMinimumTimeStep()
                      << "," << _spatial->LastMaximumTimeStep() << "]"
                      << std::endl;

            if (_configuration.io.writeVTK &&
                _configuration.io.outputInterval > 0 &&
                _currentIteration % _configuration.io.outputInterval == 0)
                WriteOutput(StampedName(
                    _configuration.io.outputPrefix, _currentIteration));
            if (_configuration.io.restartInterval > 0 &&
                _currentIteration % _configuration.io.restartInterval == 0)
                WriteRestart(StampedName(
                    _configuration.io.restartPrefix, _currentIteration));
        }

        WriteVortexDiagnostics(false);

        if (additionalIterations == 0 && _mpi.rank == 0)
            log() << "NCFV initialization-only residual RMS="
                  << std::setprecision(12) << residual
                  << ", dt[min,max]=[" << _spatial->LastMinimumTimeStep()
                  << "," << _spatial->LastMaximumTimeStep() << "]"
                  << std::endl;

        if (_configuration.io.writeVTK && _configuration.io.writeFinal &&
            _lastOutputIteration != _currentIteration)
            WriteOutput(StampedName(
                _configuration.io.outputPrefix, _currentIteration));
        if (_configuration.io.writeFinalRestart &&
            _lastRestartIteration != _currentIteration)
            WriteRestart(StampedName(
                _configuration.io.restartPrefix, _currentIteration));
    }

    template class Solver<2>;
    template class Solver<3>;
}
