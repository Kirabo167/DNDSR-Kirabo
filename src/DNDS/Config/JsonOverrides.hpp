#pragma once

#include "DNDS/Serializer/JsonUtil.hpp"

#include <string>
#include <vector>

namespace DNDS::Config
{
    // The accepted spelling of a string value is part of each solver's public CLI.
    enum class OverrideSyntax { Json, JsonOrText, EulerJsonC };

    inline t_jsonconfig ParseOverrideValue(const std::string &text, OverrideSyntax syntax)
    {
        if (syntax == OverrideSyntax::Json)
            return t_jsonconfig::parse(text);
        try
        {
            return t_jsonconfig::parse(text, nullptr, true, syntax == OverrideSyntax::EulerJsonC);
        }
        catch (const t_jsonconfig::parse_error &)
        {
            // Euler historically decodes JSON escapes even in an unquoted CLI string.
            if (syntax == OverrideSyntax::EulerJsonC)
                return t_jsonconfig::parse('"' + text + '"');
            return text;
        }
    }

    /// Visit overrides in CLI order, including repeated keys. The receiver owns
    /// path creation, aliases and application to the appropriate configuration layer.
    template <class Receiver>
    void ForEachJsonOverride(const std::vector<std::string> &keys, const std::vector<std::string> &values,
                             OverrideSyntax syntax, Receiver &&receive)
    {
        DNDS_check_throw_info(keys.size() == values.size(), "overwrite keys and values must be paired");
        for (std::size_t i = 0; i < keys.size(); ++i)
        {
            const t_jsonconfig::json_pointer pointer(keys[i]);
            receive(pointer, ParseOverrideValue(values[i], syntax));
        }
    }
}
