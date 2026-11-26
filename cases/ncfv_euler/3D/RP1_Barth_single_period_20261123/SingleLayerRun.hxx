// Body of the case-local Solver<3>::Run(). Only this compact field is advanced.
const int side = std::stoi(std::getenv("NCFV_LAYER_SIDE"));
const int images = std::stoi(std::getenv("NCFV_LAYER_IMAGES"));
PeriodicLayer layer(_mpi, *_mesh, *_geometry, _state, side, images, _configuration.mesh.periodicLengths[2]);
const auto &time = _configuration.time;
const auto &io = _configuration.io;
DNDS_check_throw_info(!time.useLocalTimeStep && time.endTime >= 0 && !io.writeVTK,
                      "Single-period RK requires global physical time and H5 field output");
NodeStatePair compact;
compact.InitPair("PhysicalLayer", _mpi);
compact.father->Resize(layer.means.size(), 5, 1);
compact.son->Resize(0, 5, 1);
if (!io.restartInput.empty())
{
    auto reader = io.restartSerializer.BuildSerializer(_mpi);
    const auto [filename, displayName] = io.restartSerializer.ModifyFilePath(io.restartInput, _mpi, "%06d", true);
    reader->OpenFile(filename, true);
    std::string stored;
    reader->ReadString("configuration", stored);
    const auto cfg = nlohmann::json::parse(stored);
    DNDS_check_throw_info(cfg.at("mesh").at("meshFile") == _configuration.mesh.meshFile &&
                          cfg.at("physics").at("gamma") == _configuration.physics.gamma,
                          "Physical restart mesh or gas mismatch");
    reader->ReadIndex("iteration", _currentIteration);
    reader->ReadReal("simulationTime", _simulationTime);
    compact.ReadSerializeRedistributed(reader, "state", layer.physicalIds);
    reader->CloseFile();
    for (std::size_t k = 0; k < layer.means.size(); ++k)
        layer.means[k] = compact[k];
}
auto save = [&]()
{
    const auto [filename, displayName] = io.restartSerializer.ModifyFilePath(StampedName(io.restartPrefix, _currentIteration),
                                                    _mpi, "%06d", false);
    auto writer = io.restartSerializer.BuildSerializer(_mpi);
    writer->OpenFile(filename, false);
    writer->WriteString("configuration", nlohmann::ordered_json(_configuration).dump());
    writer->WriteString("nodeIdentity", "CGNSOriginalNodeIndex");
    writer->WriteString("integrationMode", "EfficientDifferential");
    writer->WriteInt("dimension", 3);
    writer->WriteReal("gamma", _configuration.physics.gamma);
    writer->WriteInt("operatorImagePeriods", images);
    const char *localMode = std::getenv("NCFV_P1_LOCAL_COMPONENTWISE");
    writer->WriteInt("localComponentwiseBarth", localMode && std::atoi(localMode) != 0);
    writer->WriteIndex("iteration", _currentIteration);
    writer->WriteReal("simulationTime", _simulationTime);
    for (std::size_t k = 0; k < layer.means.size(); ++k)
        compact[k] = layer.means[k];
    compact.WriteSerialize(writer, "state", layer.physicalIds, false, false);
    writer->CloseFile();
    _lastRestartIteration = _currentIteration;
    if (_mpi.rank == 0)
        log() << "NCFV wrote restart: " << displayName << std::endl;
};
std::vector<State> stage(layer.means.size()), derivative(layer.means.size());
if (io.writeInitial && io.restartInterval > 0 && io.restartInput.empty())
    save();
for (index it = 0; it < time.iterations && _simulationTime < time.endTime - 1e-13; ++it)
{
    const double start = MPI_Wtime();
    real cap = time.endTime - _simulationTime, dt = 0;
    for (int retry = 0;; ++retry)
    {
        _spatial->SetMaximumStep(cap);
        stage = layer.means;
        bool good = true;
        for (int s = 0; s < 3; ++s)
        {
            layer.Lift(stage, _state);
            _spatial->EvaluateRHS(_state, _rhs, s == 0, false);
            if (s == 0)
                dt = _spatial->LastMinimumTimeStep();
            layer.FoldResidual(_rhs, derivative);
            int bad = 0;
            for (std::size_t k = 0; k < stage.size(); ++k)
            {
                const State forward = stage[k] + dt * derivative[k];
                if (s == 0)
                    stage[k] = forward;
                else if (s == 1)
                    stage[k] = .75 * layer.means[k] + .25 * forward;
                else
                    stage[k] = layer.means[k] / 3 + (2. / 3) * forward;
                bad |= !StateIsPhysical(stage[k]);
            }
            MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, _mpi.comm);
            if (bad)
            {
                good = false;
                break;
            }
        }
        if (good)
            break;
        cap = .5 * dt;
        DNDS_check_throw_info(time.rejectNonPhysicalSteps && retry < time.maximumStepRetries &&
                              cap >= time.minimumTimeStep, "Single-layer physical-stage retry failed");
        if (_mpi.rank == 0)
            log() << "Single-period SSPRK3 retry dt=" << cap << std::endl;
    }
    layer.means.swap(stage);
    _simulationTime += dt;
    ++_currentIteration;
    if (_mpi.rank == 0 && (_currentIteration % time.reportInterval == 0 || _simulationTime >= time.endTime - 1e-13))
        log() << "NCFV iteration=" << _currentIteration << ", dt=" << std::setprecision(12) << dt
              << ", time=" << _simulationTime << ", target=" << time.endTime
              << ", step wall[s]=" << MPI_Wtime() - start << std::endl;
    if (io.restartInterval > 0 && _currentIteration % io.restartInterval == 0)
        save();
}
if (io.writeFinalRestart && _lastRestartIteration != _currentIteration)
    save();
