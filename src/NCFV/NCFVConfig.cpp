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
        /**
         * @brief Recursively overlay supplied object members onto typed default configuration data.
         * @param [in,out] destination Defaults/resolved JSON object whose matching leaves are replaced.
         * @param source User JSON values; nested objects are merged and non-object values replace their
         *     destinations.
         */
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

        /**
         * @brief Fill optional fields of each configured array item from a typed default object.
         * @param [in,out] array JSON array whose items are normalized against defaults; non-array input is
         *     returned unchanged and non-object items are preserved for later validation.
         * @param defaults Default configuration item serialized as the initial value for each normalized object.
         * @tparam T Serializable configuration-item type, such as a boundary zone or region initializer.
         */
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
        DNDS_check_throw_info(!reconstruction.limitPointRecovery ||
                                  (reconstruction.enableLimiter && algorithm.mode == IntegrationMode::EfficientDifferential),
                              "NCFV limitPointRecovery requires efficient mode with the limiter enabled");
        DNDS_check_throw_info(!reconstruction.barthZeroOrderFallback ||
                                  (reconstruction.limitPointRecovery && !reconstruction.sharedGradientFaceSweep &&
                                   !reconstruction.limitFluxCorrection && !physics.viscous.enabled),
                              "NCFV barthZeroOrderFallback requires mean-anchored efficient inviscid limiting without shared-gradient sweeps or flux correction");
        DNDS_check_throw_info(!reconstruction.sharedGradientFaceSweep ||
                                  (reconstruction.enableLimiter && algorithm.mode == IntegrationMode::EfficientDifferential),
                              "NCFV sharedGradientFaceSweep requires efficient mode with the limiter enabled");
        DNDS_check_throw_info(!reconstruction.sharedGradientFaceSweep ||
                                  (!reconstruction.limitFluxCorrection && !reconstruction.directionalPairBounds),
                              "NCFV sharedGradientFaceSweep cannot be combined with limitFluxCorrection or directionalPairBounds");
        DNDS_check_throw_info(!reconstruction.limitPressure || reconstruction.limitPointRecovery,
                              "NCFV limitPressure requires mean-anchored point recovery limiting");
        DNDS_check_throw_info(!reconstruction.directionalPairBounds || reconstruction.limitPointRecovery,
                              "NCFV directionalPairBounds requires mean-anchored point recovery limiting");
        DNDS_check_throw_info(!reconstruction.limitFluxCorrection ||
                                  (algorithm.mode == IntegrationMode::EfficientDifferential && algorithm.checkGeometryClosure &&
                                   !physics.viscous.enabled && time.useCFLTimeStep &&
                                   !time.useLocalTimeStep && time.cfl <= 1.0),
                              "NCFV limitFluxCorrection requires efficient inviscid integration with global CFL time steps and CFL <= 1");
        DNDS_check_throw_info(std::isfinite(reconstruction.maximumFluxCorrection) &&
                                  reconstruction.maximumFluxCorrection >= 0 && reconstruction.maximumFluxCorrection <= 1,
                              "NCFV maximumFluxCorrection must be finite and between zero and one");
        DNDS_check_throw_info(!reconstruction.fluxCorrectionPairBounds || reconstruction.limitFluxCorrection,
                              "NCFV fluxCorrectionPairBounds requires limitFluxCorrection");
        DNDS_check_throw_info(!time.rejectNonPhysicalSteps || !time.useLocalTimeStep,
                              "NCFV positivity step retries require a global physical time step");
        DNDS_check_throw_info(mesh.periodicLengths.size() == 3,
                              "NCFV periodicLengths must have three entries");
        std::size_t periodicAxisCount = 0;
        for (real length : mesh.periodicLengths)
        {
            DNDS_check_throw_info(std::isfinite(length) && length >= 0,
                                  "NCFV periodicLengths must be finite and nonnegative");
            if (length > 0)
                periodicAxisCount++;
        }
        const bool periodic = periodicAxisCount > 0;
        if (periodic)
        {
            DNDS_check_throw_info(dimension == 3,
                                  "NCFV translational quotient currently requires a 3-D mesh");
            DNDS_check_throw_info(!reconstruction.enableLimiter ||
                                      algorithm.mode == IntegrationMode::EfficientDifferential,
                                  "NCFV periodic limiting requires efficient integration");
            DNDS_check_throw_info(physics.requireBoundaryZoneCoverage && !physics.boundaryZones.empty(),
                                  "NCFV periodic box requires explicit complete boundary coverage");
            std::vector<std::string> periodicZoneNames;
            for (const auto &boundary : physics.boundaryZones)
                if (boundary.mode == BoundaryMode::Periodic)
                    periodicZoneNames.push_back(boundary.name);
            DNDS_check_throw_info(
                periodicZoneNames.size() == 2 * periodicAxisCount,
                "NCFV requires exactly two Periodic boundary zones for each active periodic axis");
            DNDS_check_throw_info(
                mesh.periodicBoundaryPairs.empty() ||
                    mesh.periodicBoundaryPairs.size() == 2 * periodicAxisCount,
                "NCFV exact periodic topology requires one main/donor pair per active axis, ordered x,y,z");
            if (!mesh.periodicBoundaryPairs.empty())
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
            DNDS_check_throw_info(mesh.periodicBoundaryPairs.empty(),
                                  "NCFV periodicBoundaryPairs requires positive periodicLengths");
            for (const auto &boundary : physics.boundaryZones)
                DNDS_check_throw_info(boundary.mode != BoundaryMode::Periodic,
                                      "NCFV Periodic boundary requires periodicLengths");
        }
        DNDS_check_throw_info(algorithm.quadratureOrder >= 2 &&
                                  algorithm.quadratureOrder <= Geom::Elem::INT_ORDER_MAX,
                              "NCFV volume quadrature order is outside DNDSR's supported range");
        DNDS_check_throw_info(algorithm.surfaceQuadratureOrder >= 3 &&
                                  algorithm.surfaceQuadratureOrder <= Geom::Elem::INT_ORDER_MAX + 1,
                              "NCFV surface quadrature order must be at least three and within DNDSR's supported range");
        DNDS_check_throw_info(reconstruction.stencilSizeFactor >= 1.0,
                              "NCFV stencil size factor must be at least one");
        DNDS_check_throw_info(
            reconstruction.method != ReconstructionMethod::Unknown,
            "NCFV reconstruction.method must be LeastSquares, SVDLeastSquares, or Variational");
        DNDS_check_throw_info(reconstruction.variationalIterations >= 1 &&
                                  reconstruction.variationalIterations <= 10000 &&
                                  reconstruction.variationalRelaxation > 0 &&
                                  reconstruction.variationalRelaxation <= 1,
                              "NCFV variational iteration settings are invalid");
        DNDS_check_throw_info(std::isfinite(reconstruction.variationalWeight) &&
                                  reconstruction.variationalWeight > 0,
                              "NCFV reconstruction.variationalWeight must be finite and positive");
        DNDS_check_throw_info(reconstruction.svdTolerance > 0,
                              "NCFV least-squares rank tolerance must be positive");
        DNDS_check_throw_info(physics.gamma > 1,
                              "NCFV ideal-gas gamma must exceed one");
        checkPrimitive(physics.initialPrimitive, "NCFV physics.initialPrimitive");
        checkPrimitive(physics.farFieldPrimitive, "NCFV physics.farFieldPrimitive");
        DNDS_check_throw_info(physics.riemannSolver != Euler::Gas::UnknownRS &&
                                  physics.riemannSolver != Euler::Gas::Roe_M9,
                              "NCFV selected an unavailable Riemann solver");
        DNDS_check_throw_info(
            physics.riemannSolver != Euler::Gas::Roe_M2,
            "NCFV Roe_M2 is the deprecated scalar Rusanov path; use Roe for "
            "characteristic Roe flux with Harten--Yee entropy fixing");

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
        DNDS_check_throw_info(time.maximumStepRetries >= 0 && time.maximumStepRetries <= 40,
                              "NCFV maximumStepRetries must be between zero and forty");
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
