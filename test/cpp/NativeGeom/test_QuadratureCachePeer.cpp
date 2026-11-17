#include "NativeGeom/Quadratures/Quadrature.hpp"

const void *QuadratureCachePeer()
{
    return &DNDS::NativeGeom::Elem::detail::NBufferAtQuadrature;
}
