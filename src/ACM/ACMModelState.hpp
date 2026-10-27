/** @file ACMModelState.hpp
 *  @brief Unified flow/RANS state conversions and executable selection.
 */
#pragma once

#include "ACMConfig.hpp"
#include "DNDS/Errors.hpp"

namespace DNDS::ACM
{
    /**
     * @brief Describe the solver executable selected by a compile-time ACM model.
     * @return ACM/CFV selection with the model name and total flow-plus-turbulence variable count.
     * @tparam model ACM equation set, including dimension and turbulence extension.
     */
    template <ACMModel model>
    SolverSelection ModelSelection()
    {
        return {"ACM", "CFV", ModelName(model), ModelTraits<model>::nVarsFixed};
    }

    /**
     *  @brief Convert [u,v,w,p,q...] to [u,v,w,p,rho*q...].
     * @param primitive Model state [u,v,w,p,q...] with unweighted primitive turbulence variables.
     * @param rho Finite positive constant density used only to weight the turbulence tail.
     * @return State [u,v,w,p,rho*q...]; the four flow entries are copied unchanged.
     * @tparam model ACM equation set defining the state length.
     * @throws std::runtime_error If rho is invalid or any state entry is non-finite.
     */
    template <ACMModel model>
    ModelState<model> ModelPrimitiveToConservative(const ModelState<model> &primitive, real rho)
    {
        DNDS_check_throw_info(std::isfinite(rho) && rho > 0 && primitive.allFinite(),
                              "ACM model conversion requires finite state and positive density");
        ModelState<model> result = primitive;
        if constexpr (ModelTraits<model>::nExtraVars > 0)
            result.template tail<ModelTraits<model>::nExtraVars>() *= rho;
        return result;
    }

    /**
     * @brief Remove density weighting from the turbulence tail of an ACM model state.
     * @param conservative Model state [u,v,w,p,rho*q...] containing density-weighted turbulence variables.
     * @param rho Finite positive constant density used to divide the turbulence tail.
     * @return State [u,v,w,p,q...] with unchanged flow entries.
     * @tparam model ACM equation set defining the state length.
     * @throws std::runtime_error If rho is invalid or any state entry is non-finite.
     */
    template <ACMModel model>
    ModelState<model> ModelConservativeToPrimitive(const ModelState<model> &conservative, real rho)
    {
        DNDS_check_throw_info(std::isfinite(rho) && rho > 0 && conservative.allFinite(),
                              "ACM model conversion requires finite state and positive density");
        ModelState<model> result = conservative;
        if constexpr (ModelTraits<model>::nExtraVars > 0)
            result.template tail<ModelTraits<model>::nExtraVars>() /= rho;
        return result;
    }

    /**
     * @brief Check that runtime settings match the selected flow/RANS executable.
     * @details Checks the equation count, closure family, transport substeps, and primitive bounds of initial
     *     turbulence values.
     * @param configuration Case settings, including solver selection, initial conservative state, and turbulence
     *     bounds.
     * @tparam model Compile-time equation set whose layout the case must satisfy.
     * @throws std::runtime_error If generic validation or a model-specific constraint fails.
     */
    template <ACMModel model>
    void ValidateModelConfiguration(const KernelConfiguration &configuration)
    {
        using Traits = ModelTraits<model>;
        configuration.Validate();
        configuration.solver.Require(ModelSelection<model>());
        const int active = TurbulenceVariableCount(configuration.turbulenceSettings.model);
        DNDS_check_throw_info(active == Traits::nExtraVars,
                              "ACM executable turbulence equation count does not match turbulenceSettings.model");
        if constexpr (Traits::nExtraVars == 1)
            DNDS_check_throw_info(configuration.turbulenceSettings.model == TurbulenceModel::SpalartAllmaras,
                                  "ACM SA executable requires SpalartAllmaras");
        if constexpr (Traits::nExtraVars > 0)
        {
            DNDS_check_throw_info(configuration.turbulenceSettings.transportSubsteps == 1,
                                  "Unified ACM RANS advances every variable in each stage; transportSubsteps must be 1");
            for (int variable = 0; variable < Traits::nExtraVars; variable++)
            {
                const real value = configuration.initialState[4 + variable] / configuration.acmSettings.rho0;
                DNDS_check_throw_info(value >= configuration.turbulenceSettings.minimumValue[variable] &&
                                          value <= configuration.turbulenceSettings.maximumValue[variable],
                                      "ACM conservative initial turbulence value lies outside its primitive bounds");
            }
        }
    }
}
