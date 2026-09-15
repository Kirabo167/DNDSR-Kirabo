/**
 * @file NCFVConfig.hpp
 * @brief Typed JSON configuration for NCFV (Node Center Finite Volume Method).
 */
#pragma once

#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Config/ConfigEnum.hpp"
#include "DNDS/Serializer/SerializerFactory.hpp"
#include "Euler/Gas.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Quadratures/QuadratureBase.hpp"

#include <string>
#include <vector>

namespace DNDS::NCFV
{
    inline constexpr char MethodName[] = "Node Center Finite Volume Method";

    /** @brief Runtime choice between the thesis algorithm and ordinary quadrature. */
    enum class IntegrationMode
    {
        EfficientDifferential,
        TraditionalQuadrature,
    };

    DNDS_DEFINE_ENUM_JSON(
        IntegrationMode,
        {
            {IntegrationMode::EfficientDifferential, "EfficientDifferential"},
            {IntegrationMode::TraditionalQuadrature, "TraditionalQuadrature"},
        })

    /** @brief Boundary models supported by the standalone Navier--Stokes solver. */
    enum class BoundaryMode
    {
        FarField,
        SlipWall,
        Symmetry,
        NoSlipAdiabaticWall,
        NoSlipIsothermalWall,
        SupersonicInlet,
        SupersonicOutlet,
        PressureOutlet,
        Periodic,
    };

    DNDS_DEFINE_ENUM_JSON(
        BoundaryMode,
        {
            {BoundaryMode::FarField, "FarField"},
            {BoundaryMode::SlipWall, "SlipWall"},
            {BoundaryMode::Symmetry, "Symmetry"},
            {BoundaryMode::NoSlipAdiabaticWall, "NoSlipAdiabaticWall"},
            {BoundaryMode::NoSlipIsothermalWall, "NoSlipIsothermalWall"},
            {BoundaryMode::SupersonicInlet, "SupersonicInlet"},
            {BoundaryMode::SupersonicOutlet, "SupersonicOutlet"},
            {BoundaryMode::PressureOutlet, "PressureOutlet"},
            {BoundaryMode::Periodic, "Periodic"},
        })

    enum class ViscosityModel
    {
        Constant,
        Sutherland,
        DensityProportional,
    };

    DNDS_DEFINE_ENUM_JSON(
        ViscosityModel,
        {
            {ViscosityModel::Constant, "Constant"},
            {ViscosityModel::Sutherland, "Sutherland"},
            {ViscosityModel::DensityProportional, "DensityProportional"},
        })

    enum class InitialFieldVariables
    {
        Primitive,
        Conservative,
    };

    DNDS_DEFINE_ENUM_JSON(
        InitialFieldVariables,
        {
            {InitialFieldVariables::Primitive, "Primitive"},
            {InitialFieldVariables::Conservative, "Conservative"},
        })

    struct MeshSettings
    {
        std::string meshFile;
        int reorderCells = 0;
        real periodicTolerance = 1e-9;
        std::vector<real> periodicLengths{0.0, 0.0, 0.0};
        std::vector<std::string> periodicBoundaryPairs;
        Geom::PartitionOptions partitionOptions;

        DNDS_DECLARE_CONFIG(MeshSettings)
        {
            DNDS_FIELD(meshFile, "Input O1 CGNS mesh file");
            DNDS_FIELD(reorderCells, "Reorder local primal cells", DNDS::Config::range(0, 1));
            DNDS_FIELD(periodicTolerance, "Periodic-node matching tolerance", DNDS::Config::range(0.0));
            DNDS_FIELD(periodicLengths, "Translational periodic box lengths [Lx,Ly,Lz]; zeros disable");
            DNDS_FIELD(periodicBoundaryPairs,
                       "Flat [main1,donor1,...] CGNS zone names; empty infers pairs from Periodic boundaryZones order");
            config.field_section(&T::partitionOptions, "partitionOptions", "Existing DNDSR partition options");
        }
    };

    struct AlgorithmSettings
    {
        IntegrationMode mode = IntegrationMode::EfficientDifferential;
        int quadratureOrder = 4;
        bool retainMicroGeometry = true;
        bool checkGeometryClosure = true;
        real closureTolerance = 2e-10;
        // Diagnostic-only switch set programmatically by initialization probes.
        // It is intentionally absent from the serialized configuration schema.
        bool profileIntegrationInitialization = false;

        DNDS_DECLARE_CONFIG(AlgorithmSettings)
        {
            DNDS_FIELD(mode, "NCFV integration implementation",
                       DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(IntegrationMode)));
            DNDS_FIELD(quadratureOrder, "Traditional Gauss/Hammer integration order",
                       DNDS::Config::range(2, Geom::Elem::INT_ORDER_MAX));
            DNDS_FIELD(retainMicroGeometry, "Retain construction-simplex geometry for diagnostics");
            DNDS_FIELD(checkGeometryClosure, "Reject a dual grid that fails vector-area closure");
            DNDS_FIELD(closureTolerance, "Relative dual-area closure tolerance", DNDS::Config::range(0.0));
        }
    };

    struct ReconstructionSettings
    {
        real stencilSizeFactor = 1.7;
        int maximumStencilRings = 4;
        real distanceWeightPower = 1.0;
        real distanceWeightFloor = 0.15;
        real svdTolerance = 1e-11;
        real maximumConditionNumber = 1e12;
        bool enableLimiter = true;

        DNDS_DECLARE_CONFIG(ReconstructionSettings)
        {
            DNDS_FIELD(stencilSizeFactor, "Stencil size divided by quadratic basis size",
                       DNDS::Config::range(1.0, 8.0));
            DNDS_FIELD(maximumStencilRings, "Maximum breadth-first node rings", DNDS::Config::range(1, 8));
            DNDS_FIELD(distanceWeightPower, "Inverse-distance least-squares exponent", DNDS::Config::range(0.0, 8.0));
            DNDS_FIELD(distanceWeightFloor, "Minimum normalized stencil distance", DNDS::Config::range(1e-8, 1.0));
            DNDS_FIELD(svdTolerance, "Relative singular-value cutoff", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(maximumConditionNumber, "Maximum accepted reconstruction condition estimate",
                       DNDS::Config::range(1.0));
            DNDS_FIELD(enableLimiter, "Enable one-coefficient Barth--Jespersen limiting");
        }
    };

    struct ViscousSettings
    {
        bool enabled = false;
        ViscosityModel model = ViscosityModel::Sutherland;
        real dynamicViscosity = 1e-5;
        real gasConstant = 1.0;
        real prandtlNumber = 0.72;
        real referenceTemperature = 273.15;
        real sutherlandConstant = 110.4;
        real spectralRadiusFactor = 4.0;

        DNDS_DECLARE_CONFIG(ViscousSettings)
        {
            DNDS_FIELD(enabled, "Enable laminar Navier--Stokes viscous and heat fluxes");
            DNDS_FIELD(model, "Molecular viscosity law",
                       DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(ViscosityModel)));
            DNDS_FIELD(dynamicViscosity, "Constant/reference dynamic viscosity",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(gasConstant, "Ideal-gas specific gas constant",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(prandtlNumber, "Laminar Prandtl number",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(referenceTemperature, "Sutherland reference temperature",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(sutherlandConstant, "Sutherland constant");
            DNDS_FIELD(spectralRadiusFactor, "Explicit viscous spectral-radius multiplier",
                       DNDS::Config::range(0.0));
        }
    };

    /** @brief One CGNS boundary-zone override; an empty state uses the far field. */
    struct BoundaryZoneSettings
    {
        std::string name;
        BoundaryMode mode = BoundaryMode::FarField;
        std::vector<real> primitive;
        std::vector<real> wallVelocity;
        real wallTemperature = 1.0;
        real staticPressure = 1.0;
        bool strongState = true;

        DNDS_DECLARE_CONFIG(BoundaryZoneSettings)
        {
            DNDS_FIELD(name, "CGNS boundary-zone name (case-sensitive)");
            DNDS_FIELD(mode, "Boundary-condition model",
                       DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(BoundaryMode)));
            DNDS_FIELD(primitive, "Prescribed primitive state [rho,u,(v,w),p]; empty uses far field");
            DNDS_FIELD(wallVelocity, "Wall velocity [u,v,(w)]; empty means a stationary wall");
            DNDS_FIELD(wallTemperature, "Isothermal-wall temperature",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(staticPressure, "Pressure-outlet static pressure",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(strongState, "Strongly impose inlet/wall nodal state after every RK stage");
        }
    };

    struct PhysicsSettings
    {
        real gamma = 1.4;
        Euler::Gas::RiemannSolverType riemannSolver = Euler::Gas::Roe_M2;
        BoundaryMode boundaryMode = BoundaryMode::FarField;
        std::vector<real> initialPrimitive{1.0, 0.1, 0.0, 1.0};
        std::vector<real> farFieldPrimitive{1.0, 0.1, 0.0, 1.0};
        bool requireBoundaryZoneCoverage = false;
        std::vector<BoundaryZoneSettings> boundaryZones;
        ViscousSettings viscous;

        DNDS_DECLARE_CONFIG(PhysicsSettings)
        {
            DNDS_FIELD(gamma, "Ideal-gas heat-capacity ratio", DNDS::Config::range(1.0));
            DNDS_FIELD(riemannSolver, "DNDSR inviscid Riemann solver",
                       DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(Euler::Gas::RiemannSolverType)));
            DNDS_FIELD(boundaryMode, "Boundary model applied to all external zones",
                       DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(BoundaryMode)));
            DNDS_FIELD(initialPrimitive, "Uniform primitive state [rho,u,(v,w),p]");
            DNDS_FIELD(farFieldPrimitive, "Far-field primitive state [rho,u,(v,w),p]");
            DNDS_FIELD(requireBoundaryZoneCoverage,
                       "Require every CGNS external zone to have an explicit boundaryZones entry");
            config.template field_array_of<BoundaryZoneSettings>(
                &T::boundaryZones, "boundaryZones", "Per-CGNS-zone boundary conditions");
            config.field_section(&T::viscous, "viscous", "Laminar viscous and heat-flux model");
        }
    };

    struct BoxInitializer
    {
        real xMin = 0;
        real xMax = 0;
        real yMin = 0;
        real yMax = 0;
        real zMin = 0;
        real zMax = 0;
        std::vector<real> primitive;

        DNDS_DECLARE_CONFIG(BoxInitializer)
        {
            DNDS_FIELD(xMin, "Box x minimum");
            DNDS_FIELD(xMax, "Box x maximum");
            DNDS_FIELD(yMin, "Box y minimum");
            DNDS_FIELD(yMax, "Box y maximum");
            DNDS_FIELD(zMin, "Box z minimum");
            DNDS_FIELD(zMax, "Box z maximum");
            DNDS_FIELD(primitive, "Primitive state assigned inside the box");
        }
    };

    struct PlaneInitializer
    {
        real a = 0;
        real b = 0;
        real c = 0;
        real h = 0;
        std::vector<real> primitive;

        DNDS_DECLARE_CONFIG(PlaneInitializer)
        {
            DNDS_FIELD(a, "Plane normal x coefficient");
            DNDS_FIELD(b, "Plane normal y coefficient");
            DNDS_FIELD(c, "Plane normal z coefficient");
            DNDS_FIELD(h, "Half-space offset in a*x+b*y+c*z+h >= 0");
            DNDS_FIELD(primitive, "Primitive state assigned in the selected half-space");
        }
    };

    struct ExpressionInitializer
    {
        std::vector<std::string> program;

        [[nodiscard]] std::string GetProgram() const
        {
            std::string result;
            for (const auto &line : program)
                result += line + "\n";
            return result;
        }

        DNDS_DECLARE_CONFIG(ExpressionInitializer)
        {
            DNDS_FIELD(program,
                       "ExprTk program using inRegion, globalNode, x[3], and UPrim[dimension+2]");
        }
    };

    struct InitialFieldSettings
    {
        std::string nodeFile;
        InitialFieldVariables nodeFileVariables = InitialFieldVariables::Primitive;
        int nodeFileIndexBase = 0;
        bool requireCompleteNodeFile = true;
        std::vector<BoxInitializer> boxes;
        std::vector<PlaneInitializer> planes;
        std::vector<ExpressionInitializer> expressions;
        bool isentropicVortex = false;
        real vortexStrength = 5.0;
        std::vector<real> vortexCenter{5.0, 5.0, 0.0};

        DNDS_DECLARE_CONFIG(InitialFieldSettings)
        {
            DNDS_FIELD(nodeFile,
                       "Optional CSV/text field: globalNode,rho,u,(v,w),p-or-rhoE");
            DNDS_FIELD(isentropicVortex, "Initialize dual means using the analytic isentropic vortex");
            DNDS_FIELD(vortexStrength, "Isentropic vortex beta");
            DNDS_FIELD(vortexCenter, "Initial vortex center [x,y,z]");
            DNDS_FIELD(nodeFileVariables, "Variables stored by nodeFile",
                       DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(InitialFieldVariables)));
            DNDS_FIELD(nodeFileIndexBase, "Global node-index base used by nodeFile");
            DNDS_FIELD(requireCompleteNodeFile,
                       "Require nodeFile to contain every locally-owned global node");
            config.template field_array_of<BoxInitializer>(
                &T::boxes, "boxes", "Axis-aligned primitive-state initializers");
            config.template field_array_of<PlaneInitializer>(
                &T::planes, "planes", "Half-space primitive-state initializers");
            config.template field_array_of<ExpressionInitializer>(
                &T::expressions, "expressions", "ExprTk primitive-state initializers");
        }
    };

    struct TimeSettings
    {
        index iterations = 0;
        real timeStep = 1e-4;
        bool useCFLTimeStep = false;
        bool useLocalTimeStep = true;
        real cfl = 0.5;
        real minimumTimeStep = 1e-12;
        real maximumTimeStep = 1e10;
        real endTime = -1.0;
        int reportInterval = 10;

        DNDS_DECLARE_CONFIG(TimeSettings)
        {
            DNDS_FIELD(iterations, "Number of explicit SSPRK3 steps", DNDS::Config::range(0));
            DNDS_FIELD(timeStep, "Fixed physical/pseudo time step", DNDS::Config::range(0.0));
            DNDS_FIELD(useCFLTimeStep, "Derive the explicit step from convective/viscous spectra");
            DNDS_FIELD(useLocalTimeStep, "Use one CFL step per dual control volume");
            DNDS_FIELD(cfl, "Explicit CFL number", DNDS::Config::range(0.0));
            DNDS_FIELD(minimumTimeStep, "Lower time-step clamp", DNDS::Config::range(0.0));
            DNDS_FIELD(maximumTimeStep, "Upper time-step clamp", DNDS::Config::range(0.0));
            DNDS_FIELD(endTime, "Stop at this physical time; negative disables time-based stopping");
            DNDS_FIELD(reportInterval, "Residual reporting interval", DNDS::Config::range(1));
        }
    };

    struct IOSettings
    {
        bool writeVTK = false;
        bool writeInitial = false;
        bool writeFinal = false;
        index outputInterval = 0;
        std::string outputPrefix = "../data/out/NCFV/solution";
        std::string vtkSeriesName = "../data/out/NCFV/solution_series";
        std::string vtkFloatEncoding = "binary";
        int asciiPrecision = 12;

        std::string restartInput;
        index restartInterval = 0;
        bool writeFinalRestart = false;
        std::string restartPrefix = "../data/out/NCFV/restart";
        Serializer::SerializerFactory restartSerializer{"H5"};
        bool writeResolvedConfiguration = true;

        DNDS_DECLARE_CONFIG(IOSettings)
        {
            DNDS_FIELD(writeVTK, "Enable distributed VTU/PVTU result output");
            DNDS_FIELD(writeInitial, "Write the initialized/restarted state");
            DNDS_FIELD(writeFinal, "Write the state after the requested iterations");
            DNDS_FIELD(outputInterval, "VTK output interval; zero disables periodic output",
                       DNDS::Config::range(0));
            DNDS_FIELD(outputPrefix, "VTK output path prefix");
            DNDS_FIELD(vtkSeriesName, "PVTU series path prefix; empty disables series");
            DNDS_FIELD(vtkFloatEncoding, "VTU float encoding",
                       DNDS::Config::enum_values({"binary", "ascii"}));
            DNDS_FIELD(asciiPrecision, "VTU ASCII precision", DNDS::Config::range(1, 17));
            DNDS_FIELD(restartInput,
                       "Restart input base path without .dnds.h5/.dir suffix; empty starts fresh");
            DNDS_FIELD(restartInterval, "Restart-write interval; zero disables periodic checkpoints",
                       DNDS::Config::range(0));
            DNDS_FIELD(writeFinalRestart, "Write a restart after the requested iterations");
            DNDS_FIELD(restartPrefix, "Restart output path prefix");
            config.field_section(&T::restartSerializer, "restartSerializer",
                                 "DNDS JSON/HDF5 restart backend");
            DNDS_FIELD(writeResolvedConfiguration,
                       "Write the resolved NCFV JSON next to result output");
        }
    };

    struct Configuration
    {
        int dimension = 2;
        MeshSettings mesh;
        AlgorithmSettings algorithm;
        ReconstructionSettings reconstruction;
        PhysicsSettings physics;
        InitialFieldSettings initialField;
        TimeSettings time;
        IOSettings io;

        DNDS_DECLARE_CONFIG(Configuration)
        {
            DNDS_FIELD(dimension, "Spatial dimension", DNDS::Config::range(2, 3));
            config.field_section(&T::mesh, "mesh", "Primal mesh input");
            config.field_section(&T::algorithm, "algorithm", "Dual-grid integration implementation");
            config.field_section(&T::reconstruction, "reconstruction", "Third-order reconstruction controls");
            config.field_section(&T::physics, "physics", "Ideal-gas Euler/Navier--Stokes model");
            config.field_section(&T::initialField, "initialField", "Fresh-start field input and overrides");
            config.field_section(&T::time, "time", "Explicit time integration");
            config.field_section(&T::io, "io", "VTK result and DNDS restart I/O");
            config.post_read([](T &configuration)
                             { configuration.Validate(); });
        }

        void Validate() const;
    };

    struct LoadedConfiguration
    {
        Configuration configuration;
        nlohmann::ordered_json resolvedJson;
    };

    LoadedConfiguration LoadConfiguration(
        const std::string &jsonName,
        const std::vector<std::string> &overwriteKeys = {},
        const std::vector<std::string> &overwriteValues = {});
}
