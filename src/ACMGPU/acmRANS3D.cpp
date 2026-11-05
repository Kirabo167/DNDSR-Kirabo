#include "ACMRANSSolver.hxx"
#include "ACMResident.hxx"

namespace DNDS::ACMGPU
{
    template class ACMSolver<ACMModel::ConstantDensitySA3D>;
    template class ACMSolver<ACMModel::ConstantDensity2EQ3D>;
}
