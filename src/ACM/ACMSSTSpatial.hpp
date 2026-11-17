#pragma once
#include "DNDS/Errors.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <limits>

namespace DNDS::ACM
{
    inline bool SSTPrescribedInflow(int boundary)
    {
        return boundary == 1 || boundary == 7 || boundary == 8 || boundary == 10;
    }

    inline int SSTExtremaFlags(double value, double lower, double upper)
    {
        return (value <= lower ? 1 : 0) | (value >= upper ? 2 : 0);
    }

    /** Constrain only diffusion at current stencil extrema, preserving face conservation. */
    inline double SSTBoundNormalDerivative(double derivative, int left, int right)
    {
        if (((left & 1) || (right & 2)) && derivative < 0) derivative = 0;
        if (((left & 2) || (right & 1)) && derivative > 0) derivative = 0;
        return derivative;
    }

    struct SSTGradientStencil
    {
        std::vector<Eigen::Vector3d> displacement, weight;
    };

    /** Static, coordinate-scaled inverse-distance weighted least squares.
     * Solve the rectangular design matrix directly, avoiding squared conditioning
     * in a normal-matrix inverse on stretched boundary-layer cells.
     */
    template <int dim>
    void BuildSSTGradientWeights(SSTGradientStencil &stencil)
    {
        Eigen::Matrix<double, dim, 1> scale = Eigen::Matrix<double, dim, 1>::Zero();
        for (const auto &delta : stencil.displacement)
            scale = scale.cwiseMax(delta.template head<dim>().cwiseAbs());
        DNDS_check_throw_info((scale.array() > 0).all(), "Degenerate SST gradient stencil extent");
        const Eigen::Index count = static_cast<Eigen::Index>(stencil.displacement.size());
        Eigen::MatrixXd design(count, dim);
        Eigen::VectorXd length(count);
        for (Eigen::Index j = 0; j < count; ++j)
        {
            const auto delta = (stencil.displacement[j].template head<dim>().array() / scale.array()).matrix().eval();
            length[j] = delta.norm();
            DNDS_check_throw_info(length[j] > 0, "Zero SST neighbor displacement");
            design.row(j) = delta.transpose() / length[j];
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> solver(design, Eigen::ComputeThinU | Eigen::ComputeThinV);
        solver.setThreshold(32 * std::numeric_limits<double>::epsilon());
        DNDS_check_throw_info(solver.rank() == dim, "Rank-deficient SST gradient stencil");
        const Eigen::MatrixXd inverse = solver.solve(Eigen::MatrixXd::Identity(count, count));
        stencil.weight.assign(static_cast<std::size_t>(count), Eigen::Vector3d::Zero());
        for (Eigen::Index j = 0; j < count; ++j)
            stencil.weight[j].template head<dim>() = inverse.col(j).array() / (length[j] * scale.array());
    }
}
