#pragma once

#include <argparse.hpp>
#include <string>
#include <vector>

namespace DNDS::Config
{
    /// Register the case-inspection and override switches shared by solver frontends.
    /// Each driver owns its positional arguments, default case and MPI lifecycle.
    inline void AddSolverArguments(argparse::ArgumentParser &parser)
    {
        parser.add_argument("--check-config").flag().default_value(false)
            .help("Validate the configuration and exit before mesh initialization");
        parser.add_argument("--emit-schema").flag().default_value(false)
            .help("Print the configuration schema and exit");
        parser.add_argument("-k", "--overwrite_key").append().default_value<std::vector<std::string>>({})
            .help("JSON pointer to override, paired in order with -v (repeatable)");
        parser.add_argument("-v", "--overwrite_value").append().default_value<std::vector<std::string>>({})
            .help("Value for the corresponding -k. Use JSON quotes for string values");
    }
}
