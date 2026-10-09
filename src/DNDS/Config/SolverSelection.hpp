/**
 * @file SolverSelection.hpp
 * @brief Solver metadata checked against each independent executable.
 */
#pragma once

#include "ConfigParam.hpp"

#include <string>
#include <stdexcept>

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
                       DNDS::Config::enum_values({"Euler", "ACM", "ACMVariable", "ncfv_euler"}));
            DNDS_FIELD(discretization, "Spatial discretization",
                       DNDS::Config::enum_values({"CFV", "NCFV"}));
            DNDS_FIELD(model, "Compiled equation/model specialization");
            DNDS_FIELD(fieldNVariables,
                       "Runtime state size for dynamic Euler models NS_EX and NS_EX_3D",
                       DNDS::Config::range(1));
        }

        void Require(const SolverSelection &expected) const
        {
            if (type != expected.type || discretization != expected.discretization ||
                model != expected.model || fieldNVariables != expected.fieldNVariables)
                throw std::runtime_error(
                    "solver metadata does not match this executable; expected " +
                    expected.type + "/" + expected.discretization + "/" +
                    expected.model + " with fieldNVariables=" +
                    std::to_string(expected.fieldNVariables));
        }

        static void ConstrainSchema(nlohmann::ordered_json &schema,
                                    const SolverSelection &expected,
                                    bool dynamicState = false)
        {
            auto &properties = schema["properties"]["solver"]["properties"];
            for (const auto &key : {"type", "discretization", "model"})
            {
                const nlohmann::ordered_json values = expected;
                properties[key]["const"] = values[key];
                properties[key]["default"] = values[key];
            }
            if (!dynamicState)
                properties["fieldNVariables"]["const"] = expected.fieldNVariables;
            properties["fieldNVariables"]["default"] = expected.fieldNVariables;
        }
    };
}
