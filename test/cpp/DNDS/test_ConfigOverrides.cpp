#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Config/JsonOverrides.hpp"

using DNDS::Config::ForEachJsonOverride;
using DNDS::Config::OverrideSyntax;
using DNDS::Config::ParseOverrideValue;
using DNDS::t_jsonconfig;

TEST_CASE("Case overrides preserve pointer escaping, value types and occurrence order")
{
    t_jsonconfig document = {{"a/b", {1, 2}}};
    const auto assign = [&](const auto &pointer, t_jsonconfig value)
    { document[pointer] = std::move(value); };
    ForEachJsonOverride({"/a~1b/1", "/a~1b/1", "/tilde~0", "/text"}, {"3", "false", R"({"x":[0,null]})", "Roe"},
                        OverrideSyntax::JsonOrText, assign);
    CHECK(document["a/b"][1] == false);
    CHECK(document["tilde~"]["x"][1].is_null());
    CHECK(document["text"] == "Roe");
    CHECK_THROWS(ForEachJsonOverride({"/extra"}, {}, OverrideSyntax::JsonOrText, assign));
    CHECK_FALSE(document.contains("extra"));
    CHECK_THROWS(ForEachJsonOverride({"invalid"}, {"1"}, OverrideSyntax::JsonOrText, assign));
}

TEST_CASE("Explicit override syntax preserves the Cauchy Euler and ACM string conventions")
{
    CHECK_THROWS(ParseOverrideValue("Roe", OverrideSyntax::Json));
    CHECK(ParseOverrideValue(R"("Roe")", OverrideSyntax::Json) == "Roe");
    CHECK(ParseOverrideValue(R"(a\nb)", OverrideSyntax::JsonOrText) == R"(a\nb)");
    CHECK(ParseOverrideValue(R"(a\nb)", OverrideSyntax::EulerJsonC) == "a\nb");
    CHECK(ParseOverrideValue("2 /* comment */", OverrideSyntax::EulerJsonC) == 2);
    CHECK(ParseOverrideValue("2 /* comment */", OverrideSyntax::JsonOrText) == "2 /* comment */");
}
