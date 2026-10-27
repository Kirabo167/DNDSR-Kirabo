/**
 * @file ACMBC.hpp
 * @brief Ghost-state construction interface for constant-density ACM boundary conditions.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMFlux.hpp"
#include "DNDS/Config/ConfigParam.hpp"
#include "Geom/BoundaryCondition.hpp"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace DNDS::ACM
{
    /**
     * @brief Configuration of one named ACM boundary zone.
     * @details `value` always uses ACM ordering `[u,v,w,p]`. Euler-compatible option fields are
     * retained so case files can share a modular layout; options unsupported by constant-density
     * ACM are validated and documented instead of being interpreted as Euler thermodynamic data.
     * @note Modifier: Runzhi Ma.
     */
    struct BoundaryCondition
    {
        BoundaryType type = BoundaryType::BCFar; ///< ACM boundary family.
        std::string name;                         ///< CGNS boundary-zone name.
        std::array<real, 4> value{0, 0, 0, 0};   ///< Prescribed `[u,v,w,p]` data.
        int frameOption = 0;                      ///< Reserved moving-frame option.
        int anchorOption = 0;                     ///< Reserved anchor option.
        int integrationOption = 0;                ///< Reserved boundary-integration option.
        int specialOption = 0;                    ///< ACM special-boundary subtype.
        int rectifyOption = 0;                    ///< Reserved symmetry rectification option.
        std::vector<real> valueExtra;              ///< Optional module-specific extra values.

        DNDS_DECLARE_CONFIG(BoundaryCondition)
        {
            DNDS_FIELD(type, "ACM boundary type",
                       DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(BoundaryType)));
            DNDS_FIELD(name, "CGNS boundary-zone name");
            DNDS_FIELD(value, "ACM boundary data [u,v,w,p]");
            DNDS_FIELD(frameOption, "Reserved frame option", DNDS::Config::range(0));
            DNDS_FIELD(anchorOption, "Reserved anchor option", DNDS::Config::range(0));
            DNDS_FIELD(integrationOption, "Reserved integration option", DNDS::Config::range(0));
            DNDS_FIELD(specialOption, "ACM special-boundary subtype", DNDS::Config::range(0));
            DNDS_FIELD(rectifyOption, "Reserved symmetry rectification option", DNDS::Config::range(0));
            DNDS_FIELD(valueExtra, "Optional ACM boundary data");
        }

        /**
         * @brief Convert the JSON-compatible value array to an ACM state.
         * @return Boundary data in `[u,v,w,p]` ordering.
         */
        State ValueState() const;
    };

    /**
     * @brief Map CGNS zone names/IDs to independently configured ACM boundary conditions.
     * @note Modifier: Runzhi Ma.
     */
    class BoundaryHandler
    {
    public:
        /**
         * @brief Construct default and user-defined boundary mappings.
         * @param defaultType Type assigned to otherwise unmapped external zones.
         * @param defaultValue Default `[u,v,w,p]` value.
         * @param configuredConditions Per-zone overrides read from the existing case JSON path.
         */
        BoundaryHandler(
            BoundaryType defaultType,
            const State &defaultValue,
            const std::vector<BoundaryCondition> &configuredConditions);

        /**
         * @brief Map a CGNS boundary-zone name to a stable face-zone ID.
         * @param name Zone name supplied by the mesh reader.
         * @return Existing reserved/configured ID, or a newly appended default-condition ID.
         */
        Geom::t_index GetIDFromName(const std::string &name);

        /**
         * @brief Return the complete condition for a face-zone ID.
         * @param id Mesh face-zone ID.
         * @return Configured condition, or the default condition for unknown external IDs.
         */
        const BoundaryCondition &GetConditionFromID(Geom::t_index id) const;

        /**
         * @brief Return only the boundary type for a face-zone ID.
         * @param id Integer face-zone ID assigned by the mesh reader.
         * @return Boundary family for this zone, using the default condition when unmapped.
         */
        BoundaryType GetTypeFromID(Geom::t_index id) const;

        /**
         * @brief Return only the `[u,v,w,p]` boundary value for a face-zone ID.
         * @param id Integer face-zone ID assigned by the mesh reader.
         * @return Prescribed state [u,v,w,p], using the default condition when unmapped.
         */
        State GetValueFromID(Geom::t_index id) const;

    private:
        BoundaryCondition _defaultCondition;                         ///< Fallback external condition.
        std::vector<BoundaryCondition> _conditions;                  ///< Conditions indexed by zone ID.
        std::unordered_map<std::string, Geom::t_index> _nameToID;    ///< CGNS name-to-ID map.
    };

    /**
     * @brief Construct an ACM ghost state from one fully configured boundary condition.
     * @param condition Per-zone condition using ACM variable ordering.
     * @param interiorState Reconstructed interior state `[u,v,w,p]`.
     * @param unitNormal Outward face-normal direction.
     * @param settings ACM density and artificial-compressibility parameters.
     * @param point Physical boundary point, reserved for time/space-dependent special conditions.
     * @param time Current pseudo/physical time.
     * @return State supplied to the exterior side of the Riemann solver.
     */
    State GenerateBoundaryState(
        const BoundaryCondition &condition,
        const State &interiorState,
        const Vector3 &unitNormal,
        const Settings &settings,
        const Vector3 &point = Vector3::Zero(),
        real time = 0);

    /**
     * @brief Construct a ghost state that imposes the requested boundary condition at the face.
     * @param type Boundary-condition family to apply.
     * @param interiorState Reconstructed state on the interior side of the boundary face.
     * @param boundaryValue Prescribed `[u,v,w,p]`; individual entries are used according to `type`.
     * @param unitNormal Outward face-normal direction, normalized internally.
     * @return Ghost state paired with `interiorState` by the numerical flux.
     * @throws std::runtime_error If an input is invalid or `type` is unsupported.
     */
    State GenerateBoundaryState(
        BoundaryType type,
        const State &interiorState,
        const State &boundaryValue,
        const Vector3 &unitNormal);
}
