/**
 * @file ACM.hpp
 * @brief Common model identifiers, enumerations, traits, and fixed-size types for the ACM module.
 *
 * @details This header defines the public vocabulary shared by the constant-density three-dimensional
 * artificial-compressibility kernels without introducing dependencies on the existing Euler model.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "DNDS/Config/ConfigEnum.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/EigenUtil.hpp"

namespace DNDS::ACM
{
    /// Available artificial-compressibility equation sets.
    enum class ACMModel
    {
        ConstantDensity2D,
        ConstantDensity3D,
        ConstantDensitySA2D,
        ConstantDensitySA3D,
        ConstantDensity2EQ2D,
        ConstantDensity2EQ3D,
    };

    /// Policy used to store the pressure-like component in an ACM state.
    enum class PressureStorage
    {
        PhysicalP,
        ScaledPOverBeta2,
    };

    DNDS_DEFINE_ENUM_JSON(
        PressureStorage,
        {
            {PressureStorage::PhysicalP, "PhysicalP"},
            {PressureStorage::ScaledPOverBeta2, "ScaledPOverBeta2"},
        })

    /// Numerical flux families supported by the initial ACM implementation.
    enum class RiemannSolverType
    {
        Rusanov,
        Roe,
    };

    DNDS_DEFINE_ENUM_JSON(
        RiemannSolverType,
        {
            {RiemannSolverType::Rusanov, "Rusanov"},
            {RiemannSolverType::Roe, "Roe"},
        })

    /**
     * @brief Boundary families mirrored from the Euler solver and adapted to `[u,v,w,p]`.
     * @details The legacy names are aliases retained for existing ACM case files. They do not
     * introduce density, energy, temperature, or turbulence variables into the ACM state.
     * @note Modifier: Runzhi Ma.
     */
    enum class BoundaryType
    {
        BCUnknown = 0,
        BCFar,
        BCWall,
        BCWallInvis,
        BCWallIsothermal,
        BCOut,
        BCOutP,
        BCIn,
        BCInPsTs,
        BCSym,
        BCSpecial,

        FarField = BCFar,
        NoSlipWall = BCWall,
        SlipWall = BCWallInvis,
        PressureOutlet = BCOutP,
        VelocityInlet = BCInPsTs,
        Symmetry = BCSym,
    };

    DNDS_DEFINE_ENUM_JSON(
        BoundaryType,
        {
            {BoundaryType::BCUnknown, nullptr},
            {BoundaryType::BCFar, "BCFar"},
            {BoundaryType::BCWall, "BCWall"},
            {BoundaryType::BCWallInvis, "BCWallInvis"},
            {BoundaryType::BCWallIsothermal, "BCWallIsothermal"},
            {BoundaryType::BCOut, "BCOut"},
            {BoundaryType::BCOutP, "BCOutP"},
            {BoundaryType::BCIn, "BCIn"},
            {BoundaryType::BCInPsTs, "BCInPsTs"},
            {BoundaryType::BCSym, "BCSym"},
            {BoundaryType::BCSpecial, "BCSpecial"},
            {BoundaryType::FarField, "FarField"},
            {BoundaryType::VelocityInlet, "VelocityInlet"},
            {BoundaryType::PressureOutlet, "PressureOutlet"},
            {BoundaryType::NoSlipWall, "NoSlipWall"},
            {BoundaryType::SlipWall, "SlipWall"},
            {BoundaryType::Symmetry, "Symmetry"},
        })

    /**
     * @brief Compile-time properties of an ACM equation set.
     * @tparam model ACM equation-set identifier.
     */
    template <ACMModel model>
    struct ModelTraits;

    /// Compile-time layout of the constant-density three-dimensional state `[u, v, w, p]`.
    template <>
    struct ModelTraits<ACMModel::ConstantDensity2D>
    {
        static constexpr int dim = 2;
        static constexpr int gDim = 2;
        static constexpr int nVarsFixed = 4;
        static constexpr int velocityBegin = 0;
        static constexpr int pressureIndex = 3;
        static constexpr int nExtraVars = 0;
    };

    /// Compile-time layout of the constant-density three-dimensional state `[u, v, w, p]`.
    template <>
    struct ModelTraits<ACMModel::ConstantDensity3D>
    {
        static constexpr int dim = 3;
        static constexpr int gDim = 3;
        static constexpr int nVarsFixed = 4;
        static constexpr int velocityBegin = 0;
        static constexpr int pressureIndex = 3;
        static constexpr int nExtraVars = 0;
    };

    /** @brief Compile-time extension of the four-equation ACM flow block. */
    template <int dimension, int turbulenceVariables>
    struct RANSModelTraits
    {
        static constexpr int dim = dimension;
        static constexpr int gDim = dimension;
        static constexpr int nExtraVars = turbulenceVariables;
        static constexpr int nVarsFixed = 4 + turbulenceVariables;
        static constexpr int velocityBegin = 0;
        static constexpr int pressureIndex = 3;
        static constexpr int turbulenceBegin = 4;
        static constexpr bool hasSA = turbulenceVariables == 1;
        static constexpr bool has2EQ = turbulenceVariables == 2;
    };

    template <>
    struct ModelTraits<ACMModel::ConstantDensitySA2D> : RANSModelTraits<2, 1> {};
    template <>
    struct ModelTraits<ACMModel::ConstantDensitySA3D> : RANSModelTraits<3, 1> {};
    template <>
    struct ModelTraits<ACMModel::ConstantDensity2EQ2D> : RANSModelTraits<2, 2> {};
    template <>
    struct ModelTraits<ACMModel::ConstantDensity2EQ3D> : RANSModelTraits<3, 2> {};

    /**
     * @brief Return the stable name of an ACM equation set.
     * @param model Equation-set identifier, including its dimension and turbulence extension.
     * @return Static model-name string, or "UnknownACMModel" for an unrecognized value.
     */
    constexpr const char *ModelName(ACMModel model)
    {
        switch (model)
        {
        case ACMModel::ConstantDensity2D: return "ConstantDensity2D";
        case ACMModel::ConstantDensity3D: return "ConstantDensity3D";
        case ACMModel::ConstantDensitySA2D: return "ConstantDensitySA2D";
        case ACMModel::ConstantDensitySA3D: return "ConstantDensitySA3D";
        case ACMModel::ConstantDensity2EQ2D: return "ConstantDensity2EQ2D";
        case ACMModel::ConstantDensity2EQ3D: return "ConstantDensity2EQ3D";
        }
        return "UnknownACMModel";
    }

    template <ACMModel model>
    using ModelState = Eigen::Vector<real, ModelTraits<model>::nVarsFixed>;

    using State = Eigen::Vector<real, 4>;      ///< ACM flow block `[u, v, w, p]`.
    using Vector3 = Eigen::Vector<real, 3>;    ///< Three-dimensional geometry or velocity vector.
    using Matrix3 = Eigen::Matrix<real, 3, 3>; ///< Three-dimensional rotation or tensor matrix.
    using Matrix4 = Eigen::Matrix<real, 4, 4>; ///< Four-variable ACM operator matrix.
}
