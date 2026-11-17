/**
 * @file NCFVLinearLimiter.hpp
 * @brief Local Barth P1 fallback with initialization-only linear recovery weights.
 */
#pragma once

#include "NCFVDualGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace DNDS::NCFV
{
    class LinearBarthLimiter
    {
        bool _enabled = false;
        std::vector<Vector3> _recoveryWeights;

    public:
        /** Cache omega_ii^(1)=centroid_i-x_i. All other P1 weights are zero. */
        void Initialize(const DualGeometry &geometry, bool enabled)
        {
            _enabled = enabled;
            if (!enabled)
            {
                std::vector<Vector3>{}.swap(_recoveryWeights);
                return;
            }
            _recoveryWeights.resize(geometry.NodeVolumes().size());
            for (std::size_t i = 0; i < _recoveryWeights.size(); i++)
            {
                const auto &moments = geometry.NodeVolume(static_cast<index>(i)).moments;
                _recoveryWeights[i] = moments.first / moments.measure;
            }
        }

        [[nodiscard]] bool Enabled() const { return _enabled; }
        [[nodiscard]] std::size_t WeightCount() const { return _recoveryWeights.size(); }
        [[nodiscard]] std::size_t WeightStorageBytes() const { return _recoveryWeights.capacity() * sizeof(Vector3); }
        [[nodiscard]] const Vector3 &Weight(index node) const
        {
            DNDS_check_throw_info(_enabled, "P1 limiter recovery weights were not enabled at initialization");
            return _recoveryWeights.at(static_cast<std::size_t>(node));
        }

        template <class TMean, class TGradient>
        [[nodiscard]] static auto RecoverPointValue(
            const Eigen::MatrixBase<TMean> &mean,
            const Eigen::MatrixBase<TGradient> &gradient,
            const Vector3 &weight, real factor = 1)
        {
            DNDS_assert(mean.cols() == 1 && gradient.cols() == mean.rows());
            DNDS_assert(gradient.rows() == 2 || gradient.rows() == 3);
            DNDS_check_throw_info(std::isfinite(factor) && factor >= 0 && factor <= 1,
                                  "NCFV linear point recovery requires a finite factor in [0,1]");
            return (mean - factor * gradient.transpose() * weight.head(gradient.rows())).eval();
        }

        template <class TState, class TGradient>
        struct Result
        {
            real factor = 1;
            TState linearConstant; ///< R1 before contraction, without the discarded R2.
            TState point;
            TGradient gradient;
        };

        /**
         * Bound the complete increment R1+G^T*(x_f-x_i), then form a consistent
         * point/gradient pair. Offsets may be edge midpoints, boundary centroids,
         * or extra verification points. Only the supplied locations are bounded.
         */
        template <class TMean, class TGradient, class TLower, class TUpper>
        [[nodiscard]] static auto LimitWithWeight(
            const Eigen::MatrixBase<TMean> &mean,
            const Eigen::MatrixBase<TGradient> &gradient,
            const Vector3 &weight, const std::vector<Vector3> &offsets,
            const Eigen::MatrixBase<TLower> &lower,
            const Eigen::MatrixBase<TUpper> &upper,
            real tolerance = 1e-10)
        {
            DNDS_assert(mean.cols() == 1 && gradient.cols() == mean.rows());
            DNDS_assert(gradient.rows() == 2 || gradient.rows() == 3);
            DNDS_assert(lower.rows() == mean.rows() && upper.rows() == mean.rows());
            DNDS_check_throw_info(std::isfinite(tolerance) && tolerance >= 0 &&
                                  mean.allFinite() && gradient.allFinite() && weight.allFinite() &&
                                  lower.allFinite() && upper.allFinite() &&
                                  (lower.array() <= mean.array()).all() && (mean.array() <= upper.array()).all(),
                                  "Invalid P1 Barth state or bounds");
            Result<typename TMean::PlainObject, typename TGradient::PlainObject> result;
            result.linearConstant = -gradient.transpose() * weight.head(gradient.rows());
            for (const Vector3 &offset : offsets)
            {
                DNDS_check_throw_info(offset.allFinite(), "Nonfinite P1 Barth check location");
                const auto delta = (result.linearConstant + gradient.transpose() * offset.head(gradient.rows())).eval();
                for (Eigen::Index v = 0; v < mean.rows(); v++)
                {
                    if (delta(v) > tolerance)
                        result.factor = std::min(result.factor, std::clamp((upper(v) - mean(v)) / delta(v), real(0), real(1)));
                    else if (delta(v) < -tolerance)
                        result.factor = std::min(result.factor, std::clamp((lower(v) - mean(v)) / delta(v), real(0), real(1)));
                }
            }
            result.point = mean + result.factor * result.linearConstant;
            result.gradient = result.factor * gradient;
            return result;
        }

        template <class TMean, class TGradient, class TLower, class TUpper>
        [[nodiscard]] auto LimitCell(
            index node, const Eigen::MatrixBase<TMean> &mean,
            const Eigen::MatrixBase<TGradient> &gradient, const std::vector<Vector3> &offsets,
            const Eigen::MatrixBase<TLower> &lower, const Eigen::MatrixBase<TUpper> &upper,
            real tolerance = 1e-10) const
        {
            return LimitWithWeight(mean, gradient, Weight(node), offsets, lower, upper, tolerance);
        }
    };
}
