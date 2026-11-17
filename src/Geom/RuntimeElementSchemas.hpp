#pragma once

#include "Geom/Mesh/Mesh_DeviceView.hpp"
#include "DNDS/Runtime/Storage/ElementSchema.hpp"

namespace DNDS::Runtime
{
    /// Native communication schemas for the existing mesh PODs; no layout change.
    template <>
    struct ElementSchemaProvider<Geom::ElemInfo>
    {
        static ElementSchema Get()
        {
            using T = Geom::ElemInfo;
            static_assert(sizeof(Geom::t_index) == sizeof(int32_t));
            return ElementSchema::Record("Geom.ElemInfo.v1", sizeof(T),
                                         {{"type", offsetof(T, type), ElementScalar::I32},
                                          {"zone", offsetof(T, zone), ElementScalar::I32}},
                                         "Geom::ElemInfo", alignof(T));
        }
    };
    template <>
    struct ElementSchemaProvider<Geom::NodePeriodicBits>
    {
        static ElementSchema Get()
        {
            using T = Geom::NodePeriodicBits;
            return ElementSchema::Record("Geom.NodePeriodicBits.v1", sizeof(T),
                                         {{"bits", offsetof(T, _v), ElementScalar::U8}},
                                         "Geom::NodePeriodicBits", alignof(T));
        }
    };
}
