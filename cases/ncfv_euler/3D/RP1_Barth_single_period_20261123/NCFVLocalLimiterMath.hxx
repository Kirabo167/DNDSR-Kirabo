#pragma once

namespace DNDS::NCFV
{
    /** Pure local algebra: inputs are immutable; all modes have zero volume mean. */
    template <int dim>
    struct LocalLimiterMath
    {
        static constexpr int Modes = dim + dim * (dim + 1) / 2;
        static constexpr int Controls = dim == 3 ? 27 : 9;
        using State = Eigen::Matrix<real, dim + 2, 1>;
        using Basis = Eigen::Matrix<real, Modes, 1>;
        using Coeff = Eigen::Matrix<real, Modes, dim + 2>;
        struct Result
        {
            Coeff coefficients;
            State linearFactors = State::Ones();
            State quadraticFactors = State::Ones();
            real positivityFactor = 1;
        };

        static std::array<Basis, Controls> BernsteinBasis(
            const Vector3 &lower, const Vector3 &upper,
            const Vector3 &lengths, const Basis &volumeMean)
        {
            std::array<Basis, Controls> controls;
            for (int code = 0; code < Controls; code++)
            {
                int digits = code;
                std::array<int, dim> indices;
                Vector3 q = Vector3::Zero();
                for (int d = 0; d < dim; d++)
                {
                    indices[d] = digits % 3;
                    digits /= 3;
                    q(d) = (lower(d) + 0.5 * indices[d] * (upper(d) - lower(d))) / lengths(d);
                }
                Basis basis;
                basis.template head<dim>() = q.template head<dim>();
                int mode = dim;
                for (int d = 0; d < dim; d++)
                    for (int e = d; e < dim; e++, mode++)
                    {
                        basis(mode) = q(d) * q(e) * (d == e ? 0.5 : 1);
                        if (d == e && indices[d] == 1)
                            basis(mode) -= sqr((upper(d) - lower(d)) / lengths(d)) / 8;
                    }
                controls[code] = basis - volumeMean;
            }
            return controls;
        }

        static real StrictVenkat(real r)
        {
            if (r <= 0)
                return 0;
            if (r >= 2)
                return 1;
            // epsilon=0 form: 0 <= phi(r) <= min(1,r). No relaxed extrema.
            return r * (r + 2) / (r * r + r + 2);
        }

        static State Factors(const Coeff &coefficients, const State &mean,
                             const State &lower, const State &upper,
                             const std::vector<Basis> &checks, bool venkat)
        {
            State factors = State::Ones();
            for (const auto &basis : checks)
            {
                const State increment = coefficients.transpose() * basis;
                for (int v = 0; v < dim + 2; v++)
                {
                    real ratio = 1;
                    if (increment(v) > 1e-10)
                        ratio = (upper(v) - mean(v)) / increment(v);
                    else if (increment(v) < -1e-10)
                        ratio = (lower(v) - mean(v)) / increment(v);
                    else
                        continue;
                    factors(v) = std::min(factors(v), venkat ? StrictVenkat(ratio) : std::clamp(ratio, real(0), real(1)));
                }
            }
            return factors;
        }

        template <typename PhysicalPredicate>
        static Result Apply(const Coeff &candidate, const State &mean,
                            const State &lower, const State &upper,
                            const std::vector<Basis> &faceChecks,
                            const std::array<Basis, Controls> &controls,
                            int kind, bool enabled, PhysicalPredicate physical)
        {
            Result result;
            result.coefficients = candidate;
            std::vector<Basis> checks = faceChecks;
            if (kind == 2 || kind == 5 || kind == 6)
                checks.insert(checks.end(), controls.begin(), controls.end());
            if (enabled)
            {
                if (kind == 2)
                {
                    // Hierarchy: bound the linear polynomial, then add as much of
                    // the zero-mean quadratic remainder as is admissible.
                    Coeff linear = candidate;
                    linear.template bottomRows<Modes - dim>().setZero();
                    result.linearFactors = Factors(linear, mean, lower, upper, checks, false);
                    for (int v = 0; v < dim + 2; v++)
                        linear.col(v) *= result.linearFactors(v);
                    Coeff quadratic = candidate;
                    quadratic.template topRows<dim>().setZero();
                    for (const auto &basis : checks)
                    {
                        const State base = mean + linear.transpose() * basis;
                        const State increment = quadratic.transpose() * basis;
                        for (int v = 0; v < dim + 2; v++)
                        {
                            real ratio = 1;
                            if (increment(v) > 1e-10)
                                ratio = (upper(v) - base(v)) / increment(v);
                            else if (increment(v) < -1e-10)
                                ratio = (lower(v) - base(v)) / increment(v);
                            else
                                continue;
                            result.quadraticFactors(v) = std::min(result.quadraticFactors(v), std::clamp(ratio, real(0), real(1)));
                        }
                    }
                    for (int v = 0; v < dim + 2; v++)
                        quadratic.col(v) *= result.quadraticFactors(v);
                    result.coefficients = linear + quadratic;
                }
                else
                {
                    result.linearFactors = Factors(candidate, mean, lower, upper, checks, kind == 1);
                    if (kind == 0 || kind == 5 || kind == 6)
                        result.linearFactors.setConstant(result.linearFactors.minCoeff());
                    result.quadraticFactors = result.linearFactors;
                    for (int v = 0; v < dim + 2; v++)
                        result.coefficients.col(v) *= result.linearFactors(v);
                }
                // Floating-point guard: compare differences, not upper+tolerance.
                // A further local contraction preserves all previous bounds.
                State repair = State::Ones();
                const real repairTolerance = (kind == 5 || kind == 6) ? 5e-11 : 1e-10;
                for (const auto &basis : checks)
                {
                    const State state = mean + result.coefficients.transpose() * basis;
                    for (int v = 0; v < dim + 2; v++)
                    {
                        const real delta = state(v) - mean(v);
                        if (state(v) - upper(v) > repairTolerance && delta > 0)
                            repair(v) = std::min(repair(v), std::max(real(0), (upper(v) - mean(v)) / delta) * (1 - 32 * std::numeric_limits<real>::epsilon()));
                        if (lower(v) - state(v) > repairTolerance && delta < 0)
                            repair(v) = std::min(repair(v), std::max(real(0), (lower(v) - mean(v)) / delta) * (1 - 32 * std::numeric_limits<real>::epsilon()));
                    }
                }
                if (kind == 5 || kind == 6)
                    repair.setConstant(repair.minCoeff());
                for (int v = 0; v < dim + 2; v++)
                    result.coefficients.col(v) *= repair(v);
                result.linearFactors = result.linearFactors.cwiseProduct(repair);
                result.quadraticFactors = result.quadraticFactors.cwiseProduct(repair);
            }
            DNDS_check_throw_info(physical(mean), "Local limiter received a nonphysical control-volume mean");
            for (const auto &basis : controls)
            {
                const State increment = result.coefficients.transpose() * basis;
                if (!physical(mean + result.positivityFactor * increment))
                {
                    real lo = 0, hi = result.positivityFactor;
                    for (int iteration = 0; iteration < 48; iteration++)
                    {
                        const real mid = 0.5 * (lo + hi);
                        if (physical(mean + mid * increment))
                            lo = mid;
                        else
                            hi = mid;
                    }
                    result.positivityFactor = 0.95 * lo;
                }
            }
            result.coefficients *= result.positivityFactor;
            return result;
        }
    };
}
