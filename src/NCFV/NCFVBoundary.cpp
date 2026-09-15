#include "NCFVBoundary.hpp"

#include "DNDS/Errors.hpp"

#include <set>
#include <sstream>

namespace DNDS::NCFV
{
    BoundaryRegistry::BoundaryRegistry(
        const MPIInfo &mpi,
        int dimension,
        const PhysicsSettings &physics)
        : _mpi(mpi), _dimension(dimension),
          _requireCoverage(physics.requireBoundaryZoneCoverage)
    {
        _default.name = "<default>";
        _default.mode = physics.boundaryMode;
        _default.primitive = physics.farFieldPrimitive;
        _default.wallVelocity.assign(static_cast<std::size_t>(_dimension), 0.0);
        _default = Normalize(std::move(_default));
    }

    BoundaryZoneSettings BoundaryRegistry::Normalize(
        BoundaryZoneSettings settings) const
    {
        if (settings.primitive.empty())
            settings.primitive = _default.primitive;
        if (settings.wallVelocity.empty())
            settings.wallVelocity.assign(static_cast<std::size_t>(_dimension), 0.0);
        if (settings.wallVelocity.size() == 3 && _dimension == 2)
            settings.wallVelocity.resize(2);
        return settings;
    }

    void BoundaryRegistry::Build(
        const std::unordered_map<std::string, Geom::t_index> &nameToID,
        const Topology &topology,
        const PhysicsSettings &physics)
    {
        _settings.clear();
        _idToName.clear();
        for (const auto &[name, id] : nameToID)
            _idToName.try_emplace(id, name);

        for (const BoundaryZoneSettings &input : physics.boundaryZones)
        {
            const auto found = nameToID.find(input.name);
            DNDS_check_throw_info(
                found != nameToID.end(),
                "NCFV boundary zone is absent from the CGNS mesh: " + input.name);
            const bool validZone = input.mode == BoundaryMode::Periodic
                                       ? Geom::FaceIDIsPeriodic(found->second)
                                       : Geom::FaceIDIsExternalBC(found->second);
            DNDS_check_throw_info(
                validZone,
                input.mode == BoundaryMode::Periodic
                    ? "NCFV Periodic boundary name is not part of a merged periodic pair: " + input.name
                    : "NCFV boundary name does not identify an external CGNS zone: " + input.name);
            DNDS_check_throw_info(
                _settings.count(found->second) == 0,
                "NCFV boundary aliases configure the same zone more than once: " + input.name);
            _settings.emplace(found->second, Normalize(input));
            _idToName[found->second] = input.name;
        }

        std::set<Geom::t_index> localUsedZones;
        for (index iFace = 0; iFace < topology.NumFaceProc(); iFace++)
        {
            const Geom::t_index zone = topology.FaceElemInfo()(iFace, 0).zone;
            if (Geom::FaceIDIsExternalBC(zone))
                localUsedZones.insert(zone);
        }

        index localMissing = 0;
        std::ostringstream localMissingNames;
        for (Geom::t_index zone : localUsedZones)
            if (_settings.count(zone) == 0)
            {
                localMissing++;
                localMissingNames << Name(zone) << "(id=" << zone << ") ";
            }
        index globalMissing = 0;
        MPI_Allreduce(&localMissing, &globalMissing, 1,
                      DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
        DNDS_check_throw_info(
            !_requireCoverage || globalMissing == 0,
            "NCFV requires explicit settings for every boundary zone; locally missing: " +
                localMissingNames.str());

        if (_mpi.rank == 0)
        {
            log() << "NCFV boundary registry: default="
                  << nlohmann::json(_default.mode).get<std::string>()
                  << ", explicit zones=" << _settings.size() << std::endl;
            for (const auto &[zone, settings] : _settings)
                log() << "  zone " << settings.name << " (id=" << zone << ") -> "
                      << nlohmann::json(settings.mode).get<std::string>() << std::endl;
        }
    }

    const BoundaryZoneSettings &BoundaryRegistry::Get(Geom::t_index zone) const
    {
        const auto found = _settings.find(zone);
        return found == _settings.end() ? _default : found->second;
    }

    std::string BoundaryRegistry::Name(Geom::t_index zone) const
    {
        const auto found = _idToName.find(zone);
        return found == _idToName.end()
                   ? fmt::format("UnNamedBoundary{}", zone)
                   : found->second;
    }

    bool BoundaryRegistry::IsWall(BoundaryMode mode)
    {
        return mode == BoundaryMode::SlipWall ||
               mode == BoundaryMode::Symmetry ||
               mode == BoundaryMode::NoSlipAdiabaticWall ||
               mode == BoundaryMode::NoSlipIsothermalWall;
    }

    bool BoundaryRegistry::IsStrongType(BoundaryMode mode)
    {
        return mode == BoundaryMode::SupersonicInlet ||
               mode == BoundaryMode::NoSlipAdiabaticWall ||
               mode == BoundaryMode::NoSlipIsothermalWall;
    }
}
