#pragma once

#include "DNDS/Config/JsonOverrides.hpp"

#include <fstream>
#include <utility>

namespace DNDS::ACMCommon
{
    /// Read the complete ACM case. Both density models accept the old GMRES
    /// tolerance name in a file or a CLI override. Typed defaults are model-specific.
    inline t_jsonconfig ReadCaseDocument(const std::string &path, const std::vector<std::string> &keys,
                                         const std::vector<std::string> &values)
    {
        std::ifstream input(path);
        DNDS_check_throw_info(input.good(), "ACM case configuration file does not exist: " + path);
        auto document = t_jsonconfig::parse(input, nullptr, true, true);
        auto &time = document.at("timeMarchSettings");
        if (time.contains("gmresRelativeTolerance"))
        {
            DNDS_check_throw_info(!time.contains("linearRelativeTolerance") ||
                                      time.at("linearRelativeTolerance") == time.at("gmresRelativeTolerance"),
                                  "Specify only linearRelativeTolerance; the legacy gmresRelativeTolerance conflicts");
            // Copy before inserting: ordered_json insertion can move sibling values.
            auto tolerance = time.at("gmresRelativeTolerance");
            time["linearRelativeTolerance"] = std::move(tolerance);
            time.erase("gmresRelativeTolerance");
        }
        const t_jsonconfig::json_pointer legacy("/timeMarchSettings/gmresRelativeTolerance");
        const t_jsonconfig::json_pointer canonical("/timeMarchSettings/linearRelativeTolerance");
        Config::ForEachJsonOverride(keys, values, Config::OverrideSyntax::JsonOrText,
                                    [&](const auto &key, t_jsonconfig value)
                                    { document[key == legacy ? canonical : key] = std::move(value); });
        return document;
    }
}
