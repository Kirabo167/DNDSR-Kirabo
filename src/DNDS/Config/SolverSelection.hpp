/**
 * @file SolverSelection.hpp
 * @brief Common run-time selector for the unified DNDSR solver executable.
 */
#pragma once

#include "ConfigParam.hpp"

#include <string>

namespace DNDS
{
    /**
     * @brief Select the equation family, spatial discretization and compiled model.
     *
     * `type` selects the governing-equation implementation, while
     * `discretization` distinguishes the cell-centred CFV path from NCFV. The
     * model name replaces the legacy executable suffix.
     */
    struct SolverSelection
    {
        std::string type = "Euler";
        std::string discretization = "CFV";
        std::string model = "NS";
        int fieldNVariables = 5;

        DNDS_DECLARE_CONFIG(SolverSelection)
        {
            DNDS_FIELD(type, "Governing-equation family",
                       DNDS::Config::enum_values({"Euler", "ACM", "ACMVariable"}));
            DNDS_FIELD(discretization, "Spatial discretization",
                       DNDS::Config::enum_values({"CFV", "NCFV"}));
            DNDS_FIELD(model, "Compiled equation/model specialization");
            DNDS_FIELD(fieldNVariables,
                       "Runtime state size for dynamic Euler models NS_EX and NS_EX_3D",
                       DNDS::Config::range(1));
        }
    };
}
