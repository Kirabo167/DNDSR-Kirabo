#pragma once

#include "Solver/Linear.hpp"
#include "DNDS/Errors.hpp"

#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <limits>

namespace DNDS::ACM
{
    struct AcceptAnyImplicitLinearResidual
    {
        template <class Field>
        bool operator()(const Field &) const { return true; }
    };

    /** @brief Central FD step bounded before probing, never by projecting a probe state. */
    inline real CentralJacobianProbeStep(real baseNorm, real directionNorm, real safeStep)
    {
        DNDS_check_throw_info(std::isfinite(baseNorm) && baseNorm >= 0 &&
                                  std::isfinite(directionNorm) && directionNorm > 0 &&
                                  std::isfinite(safeStep) && safeStep > 0,
                              "Invalid ACM central Jacobian probe scales");
        const real result = std::min(std::cbrt(std::numeric_limits<real>::epsilon()) *
                                     (1 + baseNorm) / directionNorm, safeStep);
        DNDS_check_throw_info(std::isfinite(result) && result > 0, "ACM central Jacobian step underflowed");
        return result;
    }

    /**
     * @brief Solve a small implicit block after row and column equilibration.
     * @details Independent equation/variable units must not determine numerical rank. Scaling
     *     preserves the original system; genuinely singular or non-finite blocks still fail.
     * @param matrix Finite local implicit block.
     * @param rhs Finite local right-hand side in the original equation units.
     * @return Correction in the original variable units.
     * @tparam size Number of active entries in the local block.
     */
    template <int size>
    Eigen::Vector<real, size> SolveEquilibratedImplicitBlock(
        const Eigen::Matrix<real, size, size> &matrix,
        const Eigen::Vector<real, size> &rhs)
    {
        DNDS_check_throw_info(matrix.allFinite() && rhs.allFinite(),
                              "ACM implicit block or right-hand side is non-finite");
        Eigen::Matrix<real, size, size> scaled = matrix;
        Eigen::Vector<real, size> scaledRhs = rhs, columnScale;
        for (int row = 0; row < size; row++)
        {
            const real scale = scaled.row(row).cwiseAbs().maxCoeff();
            DNDS_check_throw_info(scale > 0, "ACM implicit block has a zero row");
            scaled.row(row) /= scale;
            scaledRhs(row) /= scale;
        }
        for (int column = 0; column < size; column++)
        {
            columnScale(column) = scaled.col(column).cwiseAbs().maxCoeff();
            DNDS_check_throw_info(columnScale(column) > 0, "ACM implicit block has a zero column");
            scaled.col(column) /= columnScale(column);
        }
        DNDS_check_throw_info(scaledRhs.allFinite(), "ACM scaled implicit right-hand side is non-finite");
        const auto decomposition = scaled.fullPivLu();
        DNDS_check_throw_info(decomposition.isInvertible(), "ACM equilibrated implicit block is singular");
        const Eigen::Vector<real, size> result = decomposition.solve(scaledRhs).cwiseQuotient(columnScale);
        DNDS_check_throw_info(result.allFinite(), "ACM equilibrated implicit correction is non-finite");
        return result;
    }

    /**
     * @brief Convergence diagnostics for one linear correction system A*increment = rhs.
     * @details Norms are true unpreconditioned residual RMS values, measured before nonlinear
     * damping. Exhausting an iteration budget leaves converged false unless the target was met.
     */
    struct ImplicitLinearReport
    {
        real initialResidual = 0; ///< RHS RMS for the zero initial correction.
        real finalResidual = 0;   ///< Latest checked RMS of rhs - A*increment.
        real tolerance = 0;       ///< max(absoluteTolerance, relativeTolerance*initialResidual).
        int iterations = 0;       ///< Completed SGS corrections, or the GMRES cycle index reported by its callback.
        bool converged = false;   ///< Whether the latest true residual is at or below tolerance.
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
    template <class Field, class ApplyOperator, class Precondition, class BuildField,
              class CheckResidual = AcceptAnyImplicitLinearResidual>
    ImplicitLinearReport SolveImplicitLinearSystem(
        Field &rhs, Field &increment, bool useLUSGS,
        int maximumSweeps, int subspace, int restarts,
        real absoluteTolerance, real relativeTolerance, real rmsDivisor,
        ApplyOperator &&applyOperator, Precondition &&precondition, BuildField &&buildField,
        CheckResidual &&checkResidual = CheckResidual{})
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
        if (report.finalResidual <= report.tolerance && checkResidual(rhs))
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
            const bool accepted = checkResidual(residual);
            report.converged = report.finalResidual <= report.tolerance && accepted;
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
