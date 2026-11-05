#include "ACMRANSSolver.hxx"
#include "ACMResident.hxx"

namespace DNDS::ACMGPU
{
    template class ACMSolver<ACMModel::ConstantDensitySA2D>;
    template class ACMSolver<ACMModel::ConstantDensity2EQ2D>;
}
