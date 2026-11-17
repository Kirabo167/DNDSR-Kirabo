#pragma once

#include "NativeGeom/Geometry/NativeMeshSmoothing.hpp"
#include "NativeGeom/Geometry/NativeMeshSmoothingMath.hpp"
#include "NativeGeom/Transition/FieldTransfer.hpp"

namespace DNDS::NativeGeom::NativeSmoothingFacade
{
    template <class T>
    NativeSmoothingOptions Options(const T &legacy)
    {
        NativeSmoothingOptions result;
        result.RBFRadius = legacy.RBFRadius;
        result.MaxIncludedAngle = legacy.MaxIncludedAngle;
        result.RBFPower = legacy.RBFPower;
        result.refDWall = legacy.refDWall;
        result.nIter = legacy.nIter;
        result.nSearch = legacy.nSearch;
        result.kernel = legacy.kernel;
        return result;
    }
}
