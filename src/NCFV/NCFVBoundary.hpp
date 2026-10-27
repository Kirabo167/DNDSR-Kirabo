/**
 * @file NCFVBoundary.hpp
 * @brief CGNS-name to boundary-condition registry private to NCFV.
 */
#pragma once

#include "NCFVConfig.hpp"
#include "NCFVTopology.hpp"

#include <unordered_map>

namespace DNDS::NCFV
{
    /**
     * @brief Resolves user boundary names after CGNS assigns integer zone IDs.
     *
     * The registry intentionally does not alter Geom's boundary tables.  It
     * owns a normalized copy of every NCFV setting and provides a default
     * for unlisted zones when strict coverage is disabled.
     */
    class BoundaryRegistry
    {
        MPIInfo _mpi;
        int _dimension = 0;
        bool _requireCoverage = false;
        BoundaryZoneSettings _default;
        std::unordered_map<Geom::t_index, BoundaryZoneSettings> _settings;
        std::unordered_map<Geom::t_index, std::string> _idToName;

        /**
         * @brief Fill optional boundary data and adapt wall velocity to the active dimension.
         * @param settings Boundary settings copied by value; missing primitive data inherits the registry default.
         * @return Normalized copy with complete primitive and wall-velocity vectors.
         */
        BoundaryZoneSettings Normalize(BoundaryZoneSettings settings) const;

    public:
        /**
         * @brief Construct fallback boundary data from the global physics settings.
         * @param mpi MPI communicator metadata; collective operations must be entered by every participating rank.
         * @param dimension Active spatial dimension, either 2 or 3; the third coordinate is ignored in 2-D.
         * @param physics Default boundary family, far-field primitive values, and strict zone-coverage policy.
         */
        BoundaryRegistry(
            const MPIInfo &mpi,
            int dimension,
            const PhysicsSettings &physics);

        /**
         * @brief Bind configured boundary names to CGNS zone IDs and validate coverage across ranks.
         * @param nameToID Mesh-reader mapping from CGNS boundary names and aliases to integer zone IDs.
         * @param topology Built face topology used to find external zones present on this rank.
         * @param physics Named boundary overrides to normalize and register.
         * @throws std::runtime_error If names are missing, aliases conflict, periodic classification disagrees, or
         *     required coverage is incomplete.
         */
        void Build(
            const std::unordered_map<std::string, Geom::t_index> &nameToID,
            const Topology &topology,
            const PhysicsSettings &physics);

        /**
         * @brief Look up boundary settings.
         * @param zone Integer mesh face-zone ID.
         * @return Read-only explicitly configured setting, or the normalized fallback for an unlisted zone.
         */
        [[nodiscard]] const BoundaryZoneSettings &Get(Geom::t_index zone) const;
        /**
         * @brief Look up a boundary-zone display name.
         * @param zone Integer mesh face-zone ID.
         * @return Configured/mesh zone name, or a generated UnNamedBoundary label for an unknown ID.
         */
        [[nodiscard]] std::string Name(Geom::t_index zone) const;
        /**
         * @brief Look up explicit boundary coverage.
         * @param zone Integer mesh face-zone ID.
         * @return True when this zone has an explicit configured boundary entry.
         */
        [[nodiscard]] bool HasExplicit(Geom::t_index zone) const
        {
            return _settings.count(zone) != 0;
        }

        /**
         * @brief Classify a reflecting or no-slip boundary family.
         * @param mode Boundary-condition family to inspect.
         * @return True for slip, symmetry, no-slip adiabatic, or no-slip isothermal walls.
         */
        [[nodiscard]] static bool IsWall(BoundaryMode mode);
        /**
         * @brief Identify boundary families eligible for direct owned-node state constraints.
         * @param mode Boundary-condition family to inspect.
         * @return True for supersonic inlets and either no-slip wall family.
         */
        [[nodiscard]] static bool IsStrongType(BoundaryMode mode);
    };
}
