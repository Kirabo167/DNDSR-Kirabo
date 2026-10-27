#pragma once

#include "Solver/Linear.hpp"
#include "DNDS/Errors.hpp"

#include <algorithm>
#include <cmath>

namespace DNDS::ACM
{
    /// True, unpreconditioned RMS residuals of one implicit correction system.
    struct ImplicitLinearReport
    {
        real initialResidual = 0;
        real finalResidual = 0;
        real tolerance = 0;
        int iterations = 0;
        bool converged = false;
    };

    /**
     * @brief Solve an implicit correction with an absolute-or-relative true-residual test.
     * @details SGS applies one residual correction per iteration; GMRES checks the true
     * residual at cycle boundaries. A spent iteration budget is reported as unconverged.
     * The supplied preconditioner uses one SGS sweep in stationary mode and the configured
     * fixed sweep count in GMRES mode. Damping of the nonlinear update happens afterwards.
     * @param [in] rhs Distributed right-hand side of A * increment = rhs; used for the initial global norm.
     * @param [out] increment Correction vector, cleared on entry and filled with the computed approximation.
     * @param useLUSGS True for stationary SGS residual corrections; false for restarted left-preconditioned GMRES.
     * @param maximumSweeps Positive maximum number of stationary SGS corrections; unused by GMRES.
     * @param subspace GMRES Krylov dimension, at least two.
     * @param restarts Nonnegative GMRES restart allowance; at most restarts + 1 cycles are requested.
     * @param absoluteTolerance Nonnegative absolute threshold for the true, unpreconditioned RMS residual.
     * @param relativeTolerance Relative threshold in [0,1), multiplied by the initial RMS residual.
     * @param rmsDivisor Positive divisor converting Field::norm2() to RMS, normally sqrt(global number of
     *     unknowns).
     * @param applyOperator Callback (input, output) applying the assembled distributed linear operator A.
     * @param precondition Callback (input, output) applying an approximate inverse to a residual.
     * @param buildField Callback allocating a compatible distributed workspace field, including communication
     *     mappings.
     * @return Initial/final true residuals, combined tolerance, iteration count, and convergence status.
     * @tparam Field Distributed vector type supporting norm, dot, and additive updates.
     * @tparam ApplyOperator Callable applying the linear operator to a field.
     * @tparam Precondition Callable applying the selected approximate inverse.
     * @tparam BuildField Callable allocating compatible distributed work fields.
     * @pre At least one tolerance is positive; every rank participating in the field collectives calls this
     *     function.
     */
    template <class Field, class ApplyOperator, class Precondition, class BuildField>
    ImplicitLinearReport SolveImplicitLinearSystem(
        Field &rhs, Field &increment, bool useLUSGS,
        int maximumSweeps, int subspace, int restarts,
        real absoluteTolerance, real relativeTolerance, real rmsDivisor,
        ApplyOperator &&applyOperator, Precondition &&precondition, BuildField &&buildField)
    {
        DNDS_check_throw_info(
            std::isfinite(absoluteTolerance) && absoluteTolerance >= 0 &&
                std::isfinite(relativeTolerance) && relativeTolerance >= 0 && relativeTolerance < 1 &&
                (absoluteTolerance > 0 || relativeTolerance > 0) &&
                std::isfinite(rmsDivisor) && rmsDivisor > 0 && maximumSweeps > 0 &&
                subspace >= 2 && restarts >= 0,
            "Invalid ACM implicit linear convergence settings");
        Field product, residual, correction;
        for (auto *field : {&product, &residual, &correction})
            buildField(*field);
        increment.setConstant(0.0);
        ImplicitLinearReport report;
        report.initialResidual = rhs.norm2() / rmsDivisor;
        DNDS_check_throw_info(std::isfinite(report.initialResidual), "Non-finite ACM linear right-hand side");
        report.finalResidual = report.initialResidual;
        report.tolerance = std::max(absoluteTolerance, relativeTolerance * report.initialResidual);
        if (report.finalResidual <= report.tolerance)
        {
            report.converged = true;
            return report;
        }
        residual.setConstant(0.0);
        residual.addTo(rhs, 1.0);
        const auto measure = [&]()
        {
            applyOperator(increment, product);
            residual.setConstant(0.0);
            residual.addTo(rhs, 1.0);
            residual.addTo(product, -1.0);
            report.finalResidual = residual.norm2() / rmsDivisor;
            DNDS_check_throw_info(std::isfinite(report.finalResidual), "Non-finite ACM true linear residual");
            report.converged = report.finalResidual <= report.tolerance;
            return report.converged;
        };
        if (useLUSGS)
        {
            for (int sweep = 0; sweep < maximumSweeps && !report.converged; sweep++)
            {
                precondition(residual, correction);
                increment.addTo(correction, 1.0);
                report.iterations = sweep + 1;
                measure();
            }
        }
        else
        {
            Linear::GMRES_LeftPreconditioned<Field> gmres(static_cast<uint32_t>(subspace), buildField);
            gmres.solve(applyOperator, precondition,
                        [](Field &left, Field &right) -> real { return left.dot(right); },
                        rhs, increment, static_cast<uint32_t>(restarts) + 1U,
                        [&](uint32_t cycle, real, real)
                        {
                            report.iterations = static_cast<int>(cycle);
                            return measure();
                        });
        }
        return report;
    }
}
