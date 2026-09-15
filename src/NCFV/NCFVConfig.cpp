#include "NCFVConfig.hpp"

#include "DNDS/Errors.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>

namespace DNDS::NCFV
{
    namespace
    {
        void MergeJsonObject(
            nlohmann::ordered_json &destination,
            const nlohmann::ordered_json &source)
        {
            if (!destination.is_object() || !source.is_object())
            {
                destination = source;
                return;
            }
            for (const auto &[key, value] : source.items())
            {
                if (destination.contains(key) &&
                    destination[key].is_object() && value.is_object())
                    MergeJsonObject(destination[key], value);
                else
                    destination[key] = value;
            }
        }

        template <class T>
        void MergeArrayItemDefaults(
            nlohmann::ordered_json &array,
            const T &defaults)
        {
            if (!array.is_array())
                return;
            const nlohmann::ordered_json defaultJson = defaults;
            for (auto &item : array)
            {
                nlohmann::ordered_json normalized = defaultJson;
                MergeJsonObject(normalized, item);
                item = std::move(normalized);
            }
        }
    }

    void Configuration::Validate() const
    {
        const std::size_t requiredPrimitiveSize = static_cast<std::size_t>(dimension + 2);
        const auto checkPrimitive = [&](const std::vector<real> &primitive,
                                        const std::string &name,
                                        bool allowEmpty = false)
        {
            if (allowEmpty && primitive.empty())
                return;
            DNDS_check_throw_info(primitive.size() == requiredPrimitiveSize,
                                  name + " must contain [rho,u,(v,w),p]");
            DNDS_check_throw_info(primitive.front() > 0 && primitive.back() > 0,
                                  name + " density and pressure must be positive");
            for (real value : primitive)
                DNDS_check_throw_info(std::isfinite(value), name + " must contain finite values");
        };

        DNDS_check_throw_info(dimension == 2 || dimension == 3,
                              "NCFV dimension must be 2 or 3");
        DNDS_check_throw_info(!mesh.meshFile.empty(),
                              "NCFV mesh.meshFile must name a CGNS mesh");
        DNDS_check_throw_info(mesh.periodicLengths.size() == 3,
                              "NCFV periodicLengths must have three entries");
        const bool periodic = mesh.periodicLengths[0] > 0;
        if (periodic)
        {
            DNDS_check_throw_info(dimension == 3 && mesh.periodicLengths[1] > 0 && mesh.periodicLengths[2] > 0,
                                  "NCFV translational quotient currently requires a fully periodic 3-D box");
            DNDS_check_throw_info(!reconstruction.enableLimiter,
                                  "NCFV periodic reconstruction currently requires limiter disabled");
            DNDS_check_throw_info(physics.requireBoundaryZoneCoverage && !physics.boundaryZones.empty(),
                                  "NCFV periodic box requires explicit complete boundary coverage");
            for (const auto &boundary : physics.boundaryZones)
                DNDS_check_throw_info(boundary.mode == BoundaryMode::Periodic,
                                      "NCFV fully periodic box requires every boundary to be Periodic");
            DNDS_check_throw_info(
                mesh.periodicBoundaryPairs.empty() ||
                    mesh.periodicBoundaryPairs.size() == 6,
                "NCFV exact periodic topology requires three main/donor boundary pairs");
            if (mesh.periodicBoundaryPairs.empty())
                DNDS_check_throw_info(
                    physics.boundaryZones.size() == 6,
                    "NCFV infers exact periodic pairs only when exactly six Periodic boundaryZones are listed");
            else
            {
                std::set<std::string> pairNames;
                for (const std::string &name : mesh.periodicBoundaryPairs)
                    DNDS_check_throw_info(
                        !name.empty() && pairNames.insert(name).second,
                        "NCFV periodicBoundaryPairs names must be nonempty and unique");
                for (const std::string &name : mesh.periodicBoundaryPairs)
                {
                    const auto found = std::find_if(
                        physics.boundaryZones.begin(), physics.boundaryZones.end(),
                        [&](const BoundaryZoneSettings &zone)
                        { return zone.name == name && zone.mode == BoundaryMode::Periodic; });
                    DNDS_check_throw_info(
                        found != physics.boundaryZones.end(),
                        "NCFV periodicBoundaryPairs name is not configured as Periodic: " + name);
                }
            }
        }
        else
        {
            for (real length : mesh.periodicLengths)
                DNDS_check_throw_info(length == 0, "NCFV partial periodic boxes are not implemented");
            for (const auto &boundary : physics.boundaryZones)
                DNDS_check_throw_info(boundary.mode != BoundaryMode::Periodic,
                                      "NCFV Periodic boundary requires periodicLengths");
        }
        DNDS_check_throw_info(algorithm.quadratureOrder >= 2 &&
                                  algorithm.quadratureOrder <= Geom::Elem::INT_ORDER_MAX,
                              "NCFV quadrature order is outside DNDSR's supported range");
        DNDS_check_throw_info(reconstruction.stencilSizeFactor >= 1.0,
                              "NCFV stencil size factor must be at least one");
        DNDS_check_throw_info(reconstruction.svdTolerance > 0,
                              "NCFV SVD tolerance must be positive");
        DNDS_check_throw_info(physics.gamma > 1,
                              "NCFV ideal-gas gamma must exceed one");
        checkPrimitive(physics.initialPrimitive, "NCFV physics.initialPrimitive");
        checkPrimitive(physics.farFieldPrimitive, "NCFV physics.farFieldPrimitive");
        DNDS_check_throw_info(physics.riemannSolver != Euler::Gas::UnknownRS &&
                                  physics.riemannSolver != Euler::Gas::Roe_M9,
                              "NCFV selected an unavailable Riemann solver");

        const auto &viscous = physics.viscous;
        DNDS_check_throw_info(viscous.gasConstant > 0,
                              "NCFV viscous.gasConstant must be positive");
        if (viscous.enabled)
        {
            DNDS_check_throw_info(viscous.dynamicViscosity > 0,
                                  "NCFV viscous.dynamicViscosity must be positive");
            DNDS_check_throw_info(viscous.prandtlNumber > 0,
                                  "NCFV viscous.prandtlNumber must be positive");
            if (viscous.model == ViscosityModel::Sutherland)
                DNDS_check_throw_info(viscous.referenceTemperature > 0,
                                      "NCFV Sutherland reference temperature must be positive");
        }

        std::set<std::string> boundaryNames;
        for (const auto &boundary : physics.boundaryZones)
        {
            DNDS_check_throw_info(!boundary.name.empty(),
                                  "NCFV boundary-zone names must not be empty");
            DNDS_check_throw_info(boundaryNames.insert(boundary.name).second,
                                  "NCFV boundary-zone names must be unique: " + boundary.name);
            checkPrimitive(boundary.primitive,
                           "NCFV boundaryZones[" + boundary.name + "].primitive", true);
            DNDS_check_throw_info(
                boundary.wallVelocity.empty() ||
                    boundary.wallVelocity.size() == static_cast<std::size_t>(dimension) ||
                    boundary.wallVelocity.size() == 3,
                "NCFV wallVelocity must be empty, dimension-sized, or three-dimensional");
            for (real value : boundary.wallVelocity)
                DNDS_check_throw_info(std::isfinite(value),
                                      "NCFV wallVelocity must contain finite values");
            if (boundary.mode == BoundaryMode::NoSlipIsothermalWall)
                DNDS_check_throw_info(boundary.wallTemperature > 0,
                                      "NCFV isothermal-wall temperature must be positive");
            if (boundary.mode == BoundaryMode::PressureOutlet)
                DNDS_check_throw_info(boundary.staticPressure > 0,
                                      "NCFV pressure-outlet pressure must be positive");
        }

        for (const auto &box : initialField.boxes)
        {
            DNDS_check_throw_info(box.xMin <= box.xMax && box.yMin <= box.yMax &&
                                      box.zMin <= box.zMax,
                                  "NCFV initial-field box bounds are inverted");
            checkPrimitive(box.primitive, "NCFV initial-field box primitive");
        }
        for (const auto &plane : initialField.planes)
        {
            DNDS_check_throw_info(sqr(plane.a) + sqr(plane.b) + sqr(plane.c) >
                                      sqr(verySmallReal),
                                  "NCFV initial-field plane has a zero normal");
            checkPrimitive(plane.primitive, "NCFV initial-field plane primitive");
        }
        for (const auto &expression : initialField.expressions)
            DNDS_check_throw_info(!expression.program.empty(),
                                  "NCFV initial-field expression program is empty");
        if (initialField.isentropicVortex)
        {
            DNDS_check_throw_info(initialField.vortexCenter.size() == 3 && initialField.vortexStrength >= 0,
                                  "NCFV vortex center/strength is invalid");
            DNDS_check_throw_info(!physics.viscous.enabled &&
                                      physics.initialPrimitive.front() == 1 && physics.initialPrimitive.back() == 1 &&
                                      (dimension == 2 || physics.initialPrimitive[3] == 0),
                                  "NCFV analytic vortex requires inviscid rho=p=1, w=0 background");
            DNDS_check_throw_info(initialField.nodeFile.empty() && initialField.expressions.empty() &&
                                      initialField.boxes.empty() && initialField.planes.empty(),
                                  "NCFV builtin vortex cannot be combined with other initializers");
        }

        DNDS_check_throw_info(
            time.iterations == 0 || time.useCFLTimeStep || time.timeStep > 0,
            "NCFV fixed timeStep must be positive when time marching is enabled");
        if (time.useCFLTimeStep)
            DNDS_check_throw_info(time.cfl > 0,
                                  "NCFV CFL must be positive when CFL stepping is enabled");
        DNDS_check_throw_info(time.reportInterval > 0 && time.iterations >= 0,
                              "NCFV iterations/report interval is invalid");
        DNDS_check_throw_info(time.endTime < 0 || !time.useCFLTimeStep || !time.useLocalTimeStep,
                              "NCFV physical endTime cannot be used with local pseudo-time stepping");
        DNDS_check_throw_info(time.minimumTimeStep > 0 &&
                                  time.maximumTimeStep >= time.minimumTimeStep,
                              "NCFV time-step clamps are invalid");

        if (io.writeVTK)
            DNDS_check_throw_info(!io.outputPrefix.empty(),
                                  "NCFV outputPrefix must not be empty when VTK output is enabled");
        if (io.restartInterval > 0 || io.writeFinalRestart)
            DNDS_check_throw_info(!io.restartPrefix.empty(),
                                  "NCFV restartPrefix must not be empty when restart output is enabled");
    }

    LoadedConfiguration LoadConfiguration(
        const std::string &jsonName,
        const std::vector<std::string> &overwriteKeys,
        const std::vector<std::string> &overwriteValues)
    {
        DNDS_check_throw_info(overwriteKeys.size() == overwriteValues.size(),
                              "NCFV overwrite keys and values have different lengths");

        std::ifstream input(jsonName);
        DNDS_check_throw_info(input.good(),
                              "NCFV configuration file does not exist: " + jsonName);
        const nlohmann::ordered_json supplied =
            nlohmann::ordered_json::parse(input, nullptr, true, true);
        nlohmann::ordered_json resolved = Configuration{};
        MergeJsonObject(resolved, supplied);

        MergeArrayItemDefaults(
            resolved["physics"]["boundaryZones"], BoundaryZoneSettings{});
        MergeArrayItemDefaults(
            resolved["initialField"]["boxes"], BoxInitializer{});
        MergeArrayItemDefaults(
            resolved["initialField"]["planes"], PlaneInitializer{});
        MergeArrayItemDefaults(
            resolved["initialField"]["expressions"], ExpressionInitializer{});

        for (std::size_t i = 0; i < overwriteKeys.size(); i++)
        {
            const nlohmann::ordered_json::json_pointer key(overwriteKeys[i]);
            try
            {
                resolved[key] = nlohmann::ordered_json::parse(overwriteValues[i]);
            }
            catch (const nlohmann::ordered_json::parse_error &)
            {
                resolved[key] = overwriteValues[i];
            }
        }

        // An overwrite may replace an entire array after the first default
        // normalization, so normalize item-level optional fields once more.
        MergeArrayItemDefaults(
            resolved["physics"]["boundaryZones"], BoundaryZoneSettings{});
        MergeArrayItemDefaults(
            resolved["initialField"]["boxes"], BoxInitializer{});
        MergeArrayItemDefaults(
            resolved["initialField"]["planes"], PlaneInitializer{});
        MergeArrayItemDefaults(
            resolved["initialField"]["expressions"], ExpressionInitializer{});

        Configuration configuration = resolved.get<Configuration>();
        configuration.Validate();
        nlohmann::ordered_json normalized = configuration;
        return {configuration, normalized};
    }
}
