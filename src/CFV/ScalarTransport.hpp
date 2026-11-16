#pragma once

#include "DNDS/Defines.hpp"
#include "DNDS/Errors.hpp"
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>

namespace DNDS::CFV
{
    /** SST robust diffusion: a non-negative shared coefficient in [0,2/dn].
     * This is a frozen-coefficient Picard operator, not the exact derivative
     * of the nonlinear limiter. Some affine normal derivatives on skew meshes
     * lie outside this interval; boundedness takes precedence in this mode.
     * projection must be oriented from the left centre towards the right.
     */
    inline real BoundedScalarConductance(real derivative, real jump, real projection, real normalSquared)
    {
        // Do not turn a reversed/degenerate connection into valid geometry via abs().
        if (!(projection > 0) || !(normalSquared > 0))
            return 0;
        const real twoPoint = normalSquared / projection;
        if (jump == 0)
            return twoPoint;
        return std::clamp(derivative / jump, real(0), 2 * twoPoint);
    }

    /** Geometry-only, coordinate-scaled weighted least-squares coefficients.
     * Each displacement connects the cell centre to a neighbour (including a
     * periodic image), or to a ghost reflected through a prescribed face point.
     * SVD avoids squared conditioning on stretched boundary-layer stencils.
     */
    template <int dim>
    std::vector<Eigen::Vector<real, dim>> ScalarGradientWeights(
        const std::vector<Eigen::Vector<real, dim>> &displacements)
    {
        Eigen::Vector<real, dim> scale = Eigen::Vector<real, dim>::Zero();
        for (const auto &d : displacements)
            scale = scale.cwiseMax(d.cwiseAbs());
        DNDS_check_throw_info((scale.array() > 0).all(), "Rank-deficient scalar gradient stencil");
        Eigen::MatrixXd matrix(displacements.size(), dim);
        Eigen::VectorXd weights(displacements.size());
        for (std::size_t i = 0; i < displacements.size(); i++)
        {
            const auto scaled = displacements[i].cwiseQuotient(scale).eval();
            DNDS_check_throw_info(scaled.allFinite() && scaled.norm() > 0,
                                  "Invalid scalar gradient displacement");
            weights(i) = 1 / scaled.norm();
            matrix.row(i) = scaled.transpose() * weights(i);
        }
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        DNDS_check_throw_info(svd.singularValues().size() == dim &&
                                  svd.singularValues()(dim - 1) > 1e-12 * svd.singularValues()(0),
                              "Rank-deficient scalar gradient stencil");
        const Eigen::MatrixXd inverse = scale.cwiseInverse().asDiagonal() * svd.matrixV() *
            svd.singularValues().cwiseInverse().asDiagonal() * svd.matrixU().transpose();
        std::vector<Eigen::Vector<real, dim>> result(displacements.size());
        for (std::size_t i = 0; i < result.size(); i++)
            result[i] = inverse.col(i) * weights(i);
        return result;
    }

    /** Bits: 1 = stencil minimum, 2 = stencil maximum. Boundary ghosts are
     * included in the stencil. Flags on the right refer to the right cell's
     * own stencil; -1 means no right cell. A single limited face flux is shared
     * by both cells. This protects extrema, not a universal discrete maximum
     * principle for the complete high-order convection/source/time operator.
     */
    inline real BoundScalarNormalDerivative(real derivative, int left, int right)
    {
        if (((left & 1) && derivative < 0) || ((left & 2) && derivative > 0) ||
            (right >= 0 && (((right & 1) && derivative > 0) || ((right & 2) && derivative < 0))))
            return 0;
        return derivative;
    }
}
