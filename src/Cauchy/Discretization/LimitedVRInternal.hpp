#pragma once

// Initial compressible CPU model adapter. CFV owns equation-independent gate
// stencils/penalties; this private module supplies physical sampling and O2 PP.
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "NativeCFV/Plans/LimitedVRPlan.hpp"
#include "Cauchy/Discretization/LimitedVRFace.hpp"

namespace DNDS::Cauchy
{
    struct FiniteVolume::LimitedVRGeometry
    {
        NativeCFV::GradientPlan gradient;
        NativeCFV::LimitedVRPlan gates;
        Runtime::Array<LimitedVRFace> faces;
        /// Owned-cell incidence CSR for deterministic CUDA gather of face
        /// sensors. Incidences preserve both slots of a self-periodic face.
        Runtime::Array<index> cellFaceOffsets, cellFaces;
        std::optional<NativeCFV::PolynomialSamplingPlan> sampling;
        explicit LimitedVRGeometry(const Data &data);
    };
    struct FiniteVolume::LimitedVRWorkspace
    {
        NativeCFV::ReconstructionWorkspace target;
        NativeCFV::FVField targetCoefficients;
        NativeCFV::FVField gradients, scalars, branches, seeds, temporary;
        NativeCFV::GradientWorkspace gradientExecution;
        Runtime::Array<real> base, alpha, sampledFaces, sampledPP;
        Runtime::TransferBinding targetHalo, branchHalo, seedHalo;
        explicit LimitedVRWorkspace(const Data &data, const NativeCFV::ReconstructionWorkspace &prototype);
        /// Means are host-visible/current. Gradients and BC values were prepared
        /// for this input; target,alpha are frozen until the next Build. History
        /// mutation is explicit here and must be backed up by the domain driver.
        void Build(const Data &data, const NativeCFV::FVField &mean,
                   Runtime::Array<real> &previous,
                   NativeCFV::FaceCharacteristicTransformRef transform,
                   Runtime::EagerEngine &engine,
                   const Runtime::ExecutionStream &compute,
                   const Runtime::ExecutionStream &communication);
    };
}
