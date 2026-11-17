#pragma once

#include "DNDS/Defines_bind.hpp"
#include "NativeCFV/Legacy/FiniteVolume.hpp"
#include "NativeCFV/Legacy/VRDefines_bind.hpp"

namespace DNDS::NativeCFV
{
    using tPy_FiniteVolume = py_class_ssp<FiniteVolume>;

    void pybind11_FiniteVolume_define(py::module_ &m);
}