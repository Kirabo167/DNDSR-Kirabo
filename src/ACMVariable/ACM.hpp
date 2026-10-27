/**
 * @file ACM.hpp
 * @brief Common model identifiers, enumerations, traits, and fixed-size types for the ACM module.
 *
 * @details This header defines the public vocabulary shared by the variable-density three-dimensional
 * artificial-compressibility kernels without introducing dependencies on the existing Euler model.
 *
 * @author Runzhi Ma
 * @date 2026-09-03
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "DNDS/Config/ConfigEnum.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/EigenUtil.hpp"

namespace DNDS::ACMVariable
{
    /// Available artificial-compressibility equation sets.
    enum class ACMModel
    {
        VariableDensity2D,
        VariableDensity3D,
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
     * @brief Boundary families mirrored from the Euler solver and adapted to `[rho,mx,my,mz,p]`.
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

    /**
     * @brief Five-entry variable-density layout on a two-dimensional primal mesh.
     * @details [rho,mx,my,mz,p] retains all three momentum slots; only x/y derivatives are active.
     * Momentum is rho times velocity, and the last entry is physical pressure rather than energy.
     */
    template <>
    struct ModelTraits<ACMModel::VariableDensity2D>
    {
        static constexpr int dim = 2;
        static constexpr int gDim = 2;
        static constexpr int nVarsFixed = 5;    ///< Density, three momenta, and physical pressure.
        static constexpr int velocityBegin = 1; ///< First momentum slot; velocity requires division by rho.
        static constexpr int pressureIndex = 4; ///< Physical-pressure component, not total energy.
        static constexpr int nExtraVars = 0;    ///< Turbulence transport is stored outside this flow state.
    };

    /// Compile-time layout of the variable-density three-dimensional state `[rho, mx, my, mz, p]`.
    template <>
    struct ModelTraits<ACMModel::VariableDensity3D>
    {
        static constexpr int dim = 3;
        static constexpr int gDim = 3;
        static constexpr int nVarsFixed = 5;    ///< Density, three momenta, and physical pressure.
        static constexpr int velocityBegin = 1; ///< First momentum slot; velocity requires division by rho.
        static constexpr int pressureIndex = 4; ///< Physical-pressure component, not total energy.
        static constexpr int nExtraVars = 0;    ///< Turbulence transport is stored outside this flow state.
    };

    using State = Eigen::Vector<real, 5>;      ///< ACM state or flux vector `[rho, mx, my, mz, p]`.
    using Vector3 = Eigen::Vector<real, 3>;    ///< Three-dimensional geometry or velocity vector.
    using Matrix3 = Eigen::Matrix<real, 3, 3>; ///< Three-dimensional rotation or tensor matrix.
    using Matrix5 = Eigen::Matrix<real, 5, 5>; ///< Five-variable ACM operator matrix.
}
