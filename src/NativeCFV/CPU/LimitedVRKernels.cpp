#include "NativeCFV/Detail/LimitedVRKernels.hpp"
#include "NativeCFV/Math/LimitedVRStencil.hpp"
#include "DNDS/OMP.hpp"

namespace DNDS::NativeCFV::Detail
{
    void BuildLimitedVRSeeds(LimitedVRStencilView stencil, DeviceWorkSelection selection, LVRField scalars, LVRField gates,
                             LVRInput previous, LVROutput output, LimitedVRKernelParameters settings)
    {
        Runtime::ValidateViewDevices(Runtime::Device::Host(), stencil, selection, scalars, gates, previous, output);
        OMP::ForRows(0, selection.count, [&](index iWork) noexcept
                     { Math::BuildLimitedVRSeedCell(stencil, scalars, gates, previous, output, settings, selection.Cell(iWork)); });
    }
    void LimitedVRMaximum(LimitedVRStencilView stencil, DeviceWorkSelection selection, LVRField input, LVROutput output,
                          std::array<bool, 3> active, bool excludeBoundary, bool saturatedOnly, std::array<real, 3> caps)
    {
        Runtime::ValidateViewDevices(Runtime::Device::Host(), stencil, selection, input, output);
        OMP::ForRows(0, selection.count, [&](index iWork) noexcept
                     { Math::LimitedVRMaximumCell(stencil, input, output, selection.Cell(iWork), active, excludeBoundary, saturatedOnly, caps); });
    }
}
