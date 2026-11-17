#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Input/FlowProfile.hpp"
#include "Cauchy/Motion/RigidMotion.hpp"
#include "NativeCFV/Plans/FiniteVolumePlan.hpp"
#include "NativeGeom/Access/MeshView.hpp"
namespace DNDS::Cauchy::CPU
{
    struct FlowErrorStatistics
    {
        std::array<FlowState, 3> errors{};
        real volume = 0;
    };
    NumericalStatus InitializeFlowProfile(NativeGeom::UnstructuredMeshView mesh, EquationBlockArrayView<real> flow,
                                          const FlowProfileSettings &profile, const FlowState &primitive, const PerfectGas &gas, const RigidFrame &frame) noexcept;
    NumericalResult<FlowErrorStatistics> MeasureFlowError(NativeGeom::UnstructuredMeshView mesh, NativeCFV::FiniteVolumeHostView metric,
                                                          EquationBlockArrayView<const real> flow, const FlowProfileSettings &profile,
                                                          const FlowState &primitive, const PerfectGas &gas, real time, const RigidFrame &frame) noexcept;
    bool AdmissibleMeans(Algebra::DofComponentView<const real> values, AdmissibilityView bounds);
    NumericalResult<real> MinimumStableStep(Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> sums,
                                            NativeCFV::FiniteVolumeHostView metric, real cfl) noexcept;
}
