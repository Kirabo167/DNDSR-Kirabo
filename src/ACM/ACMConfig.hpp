/**
 * @file ACMConfig.hpp
 * @brief DNDS-compatible JSON configuration model and loader for the ACM preview driver.
 *
 * @details The loader reads one complete case JSON document, applies optional command-line
 * JSON-pointer overrides, and validates the resulting typed configuration.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMBC.hpp"
#include "ACMTime.hpp"
#include "ACMTurbulence.hpp"
#include "CFV/VRSettings.hpp"
#include "DNDS/Config/SolverSelection.hpp"
#include "Geom/Mesh/Mesh.hpp"

#include <array>
#include <string>
#include <vector>

namespace DNDS::ACM
{
    /**
     * @brief High-order reconstruction algorithms available to the ACM evaluator.
     * @note Modifier: Runzhi Ma.
     */
    enum class ReconstructionType
    {
        FirstOrder,
        GreenGauss,
        Variational,
    };

    DNDS_DEFINE_ENUM_JSON(
        ReconstructionType,
        {
            {ReconstructionType::FirstOrder, "FirstOrder"},
            {ReconstructionType::GreenGauss, "GreenGauss"},
            {ReconstructionType::Variational, "Variational"},
        })

    /**
     * @brief Limiter applied after ACM variational reconstruction.
     * @note Modifier: Runzhi Ma.
     */
    enum class LimiterType
    {
        LocalExtrema,
        WBAP,
        CWBAP,
    };

    DNDS_DEFINE_ENUM_JSON(
        LimiterType,
        {
            {LimiterType::LocalExtrema, "LocalExtrema"},
            {LimiterType::WBAP, "WBAP"},
            {LimiterType::CWBAP, "CWBAP"},
        })

    /**
     * @brief Mesh input controls reused by the two- and three-dimensional ACM solvers.
     * @note Modifier: Runzhi Ma.
     */
    struct MeshSettings
    {
        std::string meshFile;                              ///< CGNS mesh path.
        int meshElevation = 0;                             ///< O1-to-O2 geometry elevation switch.
        int meshDirectBisect = 0;                          ///< Number of direct h-refinement passes.
        int meshReorderCells = 0;                          ///< Local cell reordering switch.
        real periodicTolerance = 1e-9;                     ///< Periodic-node matching tolerance.
        std::array<real, 3> periodicTranslation1{0, 0, 0}; ///< Periodic pair-1 translation.
        std::array<real, 3> periodicTranslation2{0, 0, 0}; ///< Periodic pair-2 translation.
        std::array<real, 3> periodicTranslation3{0, 0, 0}; ///< Periodic pair-3 translation.
        Geom::PartitionOptions partitionOptions;           ///< Existing DNDS METIS settings.

        DNDS_DECLARE_CONFIG(MeshSettings)
        {
            DNDS_FIELD(meshFile, "Input CGNS mesh file");
            DNDS_FIELD(meshElevation, "Elevate O1 mesh geometry to O2", DNDS::Config::range(0, 1));
            DNDS_FIELD(meshDirectBisect, "Direct mesh bisection passes", DNDS::Config::range(0, 4));
            DNDS_FIELD(meshReorderCells, "Reorder local cells", DNDS::Config::range(0, 1));
            DNDS_FIELD(periodicTolerance, "Periodic matching tolerance", DNDS::Config::range(0.0));
            DNDS_FIELD(periodicTranslation1, "Periodic pair-1 translation vector");
            DNDS_FIELD(periodicTranslation2, "Periodic pair-2 translation vector");
            DNDS_FIELD(periodicTranslation3, "Periodic pair-3 translation vector");
            config.field_section(&T::partitionOptions, "partitionOptions", "Existing DNDS mesh partition settings");
        }
    };

    /**
     * @brief Controls the CFV reconstruction stage executed before every ACM residual evaluation.
     * @note Modifier: Runzhi Ma.
     */
    struct ReconstructionSettings
    {
        ReconstructionType type = ReconstructionType::GreenGauss; ///< Selected reconstruction family.
        int variationalIterations = 3;                            ///< Fixed-point VR sweeps per residual call.
        real variationalTolerance = 0;                            ///< Scaled max equation defect; zero keeps fixed sweeps.
        int variationalMaxIterations = 10000;                     ///< Hard cap for convergence-controlled reconstruction.
        int variationalCheckInterval = 10;                        ///< Sweeps between equation-defect checks.
        real variationalRelaxation = 0.7;                         ///< Damping for simultaneous equation-defect updates.
        bool variationalUseGMRES = false;                         ///< Solve the consistent reconstruction defect with GMRES.
        int variationalGMRESSubspace = 20;                        ///< Reconstruction Arnoldi vectors per restart.
        int variationalGMRESRestarts = 2;                         ///< Reconstruction GMRES restart count.
        real variationalGMRESRelativeTolerance = 0.1;             ///< Relative linear defect target per nonlinear update.
        bool resetVariationalCoefficients = false;                ///< Reset VR coefficients before each solve.
        bool enableLimiter = true;                                ///< Apply the limiter selected by limiterType.
        LimiterType limiterType = LimiterType::LocalExtrema;      ///< Local-extrema, WBAP, or CWBAP procedure.

        DNDS_DECLARE_CONFIG(ReconstructionSettings)
        {
            DNDS_FIELD(type, "ACM reconstruction method",
                       DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(ReconstructionType)));
            DNDS_FIELD(variationalIterations, "Variational reconstruction sweeps", DNDS::Config::range(1));
            DNDS_FIELD(variationalTolerance, "Scaled maximum reconstruction equation defect; zero disables convergence control",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(variationalMaxIterations, "Maximum convergence-controlled reconstruction sweeps", DNDS::Config::range(1));
            DNDS_FIELD(variationalCheckInterval, "Reconstruction equation-defect check interval", DNDS::Config::range(1));
            DNDS_FIELD(variationalRelaxation, "Convergence-controlled reconstruction defect relaxation", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(variationalUseGMRES, "Use GMRES for convergence-controlled reconstruction");
            DNDS_FIELD(variationalGMRESSubspace, "Reconstruction GMRES subspace size", DNDS::Config::range(2));
            DNDS_FIELD(variationalGMRESRestarts, "Reconstruction GMRES restart count", DNDS::Config::range(0));
            DNDS_FIELD(variationalGMRESRelativeTolerance, "Reconstruction GMRES relative linear tolerance",
                       DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(resetVariationalCoefficients, "Reset variational coefficients at every residual call");
            DNDS_FIELD(enableLimiter, "Enable the selected reconstruction limiter");
            DNDS_FIELD(limiterType, "ACM reconstruction limiter",
                       DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(LimiterType)));
        }
    };

    /** @brief Controls parallel VTK-HDF flow-field output from the ACM driver. */
    struct OutputSettings
    {
        int interval = 0;                         ///< Outer-step interval; zero disables output.
        std::string directory = "../data/outACM"; ///< Output directory, relative to the launch directory.
        std::string prefix = "acm";               ///< File and VTK series prefix.
        bool writeInitial = true;                 ///< Write the initialized field at step zero.

        DNDS_DECLARE_CONFIG(OutputSettings)
        {
            DNDS_FIELD(interval, "Outer-step flow-field output interval; zero disables output",
                       DNDS::Config::range(0));
            DNDS_FIELD(directory, "Flow-field output directory");
            DNDS_FIELD(prefix, "Flow-field file prefix");
            DNDS_FIELD(writeInitial, "Write the initial flow field at step zero");
        }
    };

    /** @brief Controls initialization from an ACM VTK-HDF flow output. */
    struct RestartSettings
    {
        std::string flowFile;          ///< Empty selects the configured uniform initial state.
        int completedSteps = 0;        ///< Absolute steady-step index represented by flowFile.
        int writerRanks = 0;           ///< Required original MPI size; zero is valid only without a file.
        bool requireTurbulence = false; ///< Reject a RANS restart without turbulence datasets.

        DNDS_DECLARE_CONFIG(RestartSettings)
        {
            DNDS_FIELD(flowFile, "VTK-HDF flow field used for initialization; empty disables restart");
            DNDS_FIELD(completedSteps, "Completed steady steps represented by the initial field",
                       DNDS::Config::range(0));
            DNDS_FIELD(writerRanks, "MPI ranks that wrote the VTK-HDF field",
                       DNDS::Config::range(0));
            DNDS_FIELD(requireTurbulence, "Require turbulence variables in a RANS restart file");
        }
    };

    /// Complete kernel-preview configuration read by the `acm3D` application.
    struct KernelConfiguration
    {
        SolverSelection solver{"ACM", "CFV", "ConstantDensity3D", 4};
        Settings acmSettings;
        TimeMarchSettings timeMarchSettings;
        TurbulenceSettings turbulenceSettings; ///< Independent runtime RANS selection and transport controls.
        MeshSettings meshSettings;
        ReconstructionSettings reconstructionSettings;
        OutputSettings outputSettings;
        RestartSettings restartSettings;
        CFV::VRSettings vfvSettings{3};
        BoundaryType defaultBoundaryType = BoundaryType::FarField;
        std::vector<BoundaryCondition> boundaryConditions; ///< Per-zone Euler-style ACM boundaries.
        std::array<real, 4> initialState{1, 0, 0, 0};
        std::array<real, 4> boundaryValue{1, 0, 0, 0};
        std::array<real, 4> leftState{1, 0, 0, 0};
        std::array<real, 4> rightState{0, 0, 0, 0};
        std::array<real, 3> unitNormal{1, 0, 0};
        int nFacesPerRank = 1;

        DNDS_DECLARE_CONFIG(KernelConfiguration)
        {
            config.field_section(&T::solver, "solver", "Unified executable dispatch settings");
            config.field_section(&T::acmSettings, "acmSettings", "Constant-density ACM settings");
            config.field_section(
                &T::timeMarchSettings,
                "timeMarchSettings",
                "ACM steady pseudo-time and physical dual-time integration settings");
            config.field_section(
                &T::turbulenceSettings,
                "turbulenceSettings",
                "Segregated constant-density ACM turbulence settings");
            config.field_section(&T::meshSettings, "meshSettings", "Distributed mesh input settings");
            config.field_section(&T::reconstructionSettings, "reconstructionSettings", "ACM high-order reconstruction settings");
            config.field_section(&T::outputSettings, "outputSettings", "Parallel VTK-HDF flow-field output settings");
            config.field_section(&T::restartSettings, "restartSettings", "Optional same-partition VTK-HDF flow restart");
            config.field_section(&T::vfvSettings, "vfvSettings", "Existing CFV variational-reconstruction settings");
            DNDS_FIELD(defaultBoundaryType, "Boundary type applied to unmapped external zones",
                       DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(BoundaryType)));
            config.field_array_of<BoundaryCondition>(
                &T::boundaryConditions,
                "boundaryConditions",
                "Per-zone ACM boundary conditions using Euler-compatible names");
            DNDS_FIELD(initialState, "Uniform initial ACM state [u,v,w,p]");
            DNDS_FIELD(boundaryValue, "Prescribed default boundary state [u,v,w,p]");
            DNDS_FIELD(leftState, "Preview left state [u,v,w,p]");
            DNDS_FIELD(rightState, "Preview right state [u,v,w,p]");
            DNDS_FIELD(unitNormal, "Preview unit normal");
            DNDS_FIELD(nFacesPerRank, "Preview faces evaluated on each MPI rank", DNDS::Config::range(1));
            config.post_read([](T &configuration)
                             { configuration.Validate(); });
        }

        /**
         * @brief Validate nested ACM settings, preview states, normal, and face count.
         * @throws std::runtime_error If any configured value is invalid or unsupported.
         */
        void Validate() const;

        /**
         * @brief Convert the JSON-compatible left-state array to the fixed-size Eigen state type.
         * @return Left preview state `[u,v,w,p]`.
         */
        State LeftState() const;

        /**
         * @brief Convert the JSON-compatible right-state array to the fixed-size Eigen state type.
         * @return Right preview state `[u,v,w,p]`.
         */
        State RightState() const;

        /**
         * @brief Convert the JSON-compatible normal array to the ACM geometry-vector type.
         * @return Configured face-normal vector; validation ensures it has non-zero length.
         */
        Vector3 UnitNormal() const;

        /**
         * @brief Convert the configured uniform initial value to an ACM state.
         * @return Initial state `[u,v,w,p]`.
         */
        State InitialState() const;

        /**
         * @brief Convert the configured default boundary value to an ACM state.
         * @return Boundary value `[u,v,w,p]`.
         */
        State BoundaryValue() const;
    };

    /// Validated typed configuration paired with its normalized JSON representation.
    struct LoadedConfiguration
    {
        KernelConfiguration configuration;
        nlohmann::ordered_json resolvedJson;
    };

    /**
     * @brief Load, override, deserialize, and validate one complete ACM case configuration.
     * @param jsonName Path to the complete case JSON configuration.
     * @param overwriteKeys JSON-pointer paths supplied by command-line `-k` options.
     * @param overwriteValues Values paired with `overwriteKeys`; valid JSON text is parsed, while
     * non-JSON text is stored as a string.
     * @return Validated typed configuration and normalized resolved JSON.
     * @throws std::runtime_error If the file cannot be read, override counts differ, or validation fails.
     * @note One case is intentionally represented by one self-contained JSON file; no adjacent
     * base configuration participates in loading.
     * @note Modifier: Runzhi Ma.
     */
    LoadedConfiguration LoadConfiguration(
        const std::string &jsonName,
        const std::vector<std::string> &overwriteKeys = {},
        const std::vector<std::string> &overwriteValues = {});
}
