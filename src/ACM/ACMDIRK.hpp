#pragma once

#include "ACMTime.hpp"
#include "Solver/TimeIntegration/DIRKTableau.hpp"

namespace DNDS::ACM
{
    inline bool IsESDIRK3Integrator(TimeIntegratorType integrator)
    {
        return integrator == TimeIntegratorType::ESDIRK3DualTimeGMRES;
    }

    inline bool IsPhysicalTimeIntegrator(TimeIntegratorType integrator)
    {
        return IsESDIRK3Integrator(integrator) ||
               integrator == TimeIntegratorType::BDF2DualTimeGMRES ||
               integrator == TimeIntegratorType::BDF2DualTimeLUSGS;
    }

    /**
     * @brief Advance differential components with stiffly accurate ESDIRK3.
     * @details The stage solver enforces the algebraic continuity equation without
     * physical pressure mass. Its source is U_n + dt*sum_{j<s} a_sj R(U_j),
     * and shift=1/(dt*a_ss). Stage states are private until all solves succeed.
     * Boundary callbacks receive the tableau stage time, including the
     * nonmonotone interior abscissae. Pressure entries of source are ignored.
     */
    template <class Field, class Evaluate, class Solve>
    TimeStepReport AdvanceESDIRK3(Field &accepted, real time, real dt,
                                 Evaluate &&evaluate, Solve &&solve)
    {
        DNDS_check_throw_info(std::isfinite(dt) && dt > 0 && std::isfinite(time) &&
                                  std::isfinite(time + dt) && time + dt > time,
                              "ACM ESDIRK3 requires a finite advancing physical interval");
        const ODE::DIRKTableau tableau("esdirk3");
        const Field initial = accepted;
        Field stage = initial;
        std::vector<Field> rates(tableau.c.size());
        evaluate(initial, rates[0], time);
        TimeStepReport report;
        int iterations = 0;
        for (std::size_t s = 1; s < tableau.c.size(); ++s)
        {
            Field source = initial;
            for (std::size_t j = 0; j < s; ++j)
                for (std::size_t cell = 0; cell < source.size(); ++cell)
                    source[cell] += dt * tableau.a[s][j] * rates[j][cell];
            report = solve(stage, source, 1 / (dt * tableau.a[s][s]), time + tableau.c[s] * dt);
            DNDS_check_throw_info(report.converged,
                                  "ACM ESDIRK3 stage did not converge; physical step was not accepted");
            iterations += report.iterations;
            if (s + 1 < tableau.c.size())
                evaluate(stage, rates[s], time + tableau.c[s] * dt);
        }
        accepted = std::move(stage);
        report.iterations = iterations;
        return report;
    }
}
