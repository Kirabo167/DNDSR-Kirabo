/**
 * @file acm3D.cpp
 * @brief Explicit template instantiation for the three-dimensional high-order ACM solver.
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#include "CFV/VariationalReconstruction_Reconstruction.hxx"
#include "CFV/VariationalReconstruction_LimiterProcedure.hxx"

#include "ACMEvaluator.hxx"
#include "ACMTurbulenceTransport.hxx"
#include "ACMSolver.hxx"
#include "ACMResident.hxx"

namespace DNDS::ACMGPU
{
    template class ACMEvaluator<3>;
    template class ACMTurbulenceTransport<3>;
    template class ACMSolver<ACMModel::ConstantDensity3D>;
}
