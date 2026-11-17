#pragma once
#include "NativeCFV/Detail/ReconstructionDeviceView.hpp"

namespace DNDS::NativeCFV::Detail
{
    /// Verify finite matrices and both inverse products on the execution stream.
    /// On failure set sticky status to nonzero. Consumers must read that status
    /// after this launch, with the same retained matrix read reservation.
    void LaunchValidateCharacteristics(Runtime::ExecutionContext, DeviceCharacteristicView, Runtime::ArrayView<int> status);
}
