#pragma once

#include "DNDS/Defines_bind.hpp"
#include "DNDS/Array_bind.hpp"
#include <pybind11/functional.h>
#include <pybind11_json/pybind11_json.hpp>

#include "NativeCFV/Legacy/VariationalReconstruction.hpp"

#include "NativeCFV/Legacy/ModelEvaluator.hpp"

namespace DNDS::NativeCFV
{
    using tPy_ModelEvaluator = py_class_ssp<ModelEvaluator>;

    void pybind11_ModelEvaluator_define(py::module_ &m);
}