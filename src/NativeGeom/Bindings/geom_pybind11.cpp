
#include "NativeGeom/Bindings/Boundary_bind.hpp"
#include "NativeGeom/Elements/Elements_bind.hpp"
#include "NativeGeom/Unstructured/UnstructuredMesh_bind.hpp"
#include "NativeGeom/Compat/AdjPairTracked_bind.hpp"
#include "NativeGeom/Compat/AdjPairTracked_bind_inst/AdjPairTracked_bind_fwd.hpp"

namespace DNDS::NativeGeom
{
    void pybind11_BoundaryInterface_define(py::module_ &m);
}

PYBIND11_MODULE(native_geom_pybind11, m)
{
    //! hard coded dependency here
    //! better solution?
    py::module_::import("DNDSR.DNDS");

    using namespace DNDS::NativeGeom;
    auto m_elem = m.def_submodule("Elem");
    Elem::pybind11_ElemType_define(m_elem);

    pybind11_AutoAppendName2ID_define(m);
    pybind11_ElemInfo_define(m);
    pybind11_ArrayElemInfo_define(m);

    // AdjPairTracked types (must precede UnstructuredMesh which exposes them)
    pybind11_MeshAdjState_define(m);
    pybind11_AdjIndexInfo_define(m);
    pybind11_AdjPairTracked_I_I_N_define(m);
    pybind11_AdjPairTracked_2_2_N_define(m);
    pybind11_AdjPairTracked_1_1_N_define(m);

    pybind11_MeshLocDefine(m);
    pybind11_UnstructuredMesh_define(m);
    pybind11_UnstructuredMeshSerialRW_define(m);
    pybind11_BoundaryInterface_define(m);
}
