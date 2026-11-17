#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "ACMGPU/ACMSSTSpatial.hpp"
#include "ACM/ACMSSTSpatial.hpp"
#include "GPU/SSTSpatial.hpp"
#include <random>

namespace
{
    template <int dim>
    void CheckAffineStencils()
    {
        std::mt19937 engine(42);
        std::uniform_real_distribution<double> random(-1, 1);
        for (int sample = 0; sample < 30; ++sample)
        {
            DNDS::ACMGPU::SSTGradientStencil stencil;
            for (int neighbor = 0; neighbor < 10; ++neighbor)
            {
                Eigen::Vector3d delta(random(engine), random(engine), dim == 3 ? random(engine) : 0);
                delta.x() *= sample % 2 ? 1e-5 : 1;
                stencil.displacement.push_back(delta);
            }
            DNDS::ACMGPU::BuildSSTGradientWeights<dim>(stencil);
            DNDS::ACM::SSTGradientStencil cpuStencil;
            cpuStencil.displacement = stencil.displacement;
            DNDS::ACM::BuildSSTGradientWeights<dim>(cpuStencil);
            Eigen::Matrix<double, dim, dim> response = Eigen::Matrix<double, dim, dim>::Zero();
            Eigen::Matrix<double, dim, 1> reconstructed = Eigen::Matrix<double, dim, 1>::Zero();
            const auto exact = Eigen::Matrix<double, dim, 1>::LinSpaced(-2, 3).eval();
            for (std::size_t j = 0; j < stencil.weight.size(); ++j)
            {
                CHECK((cpuStencil.weight[j] - stencil.weight[j]).norm() < 1e-14);
                response += stencil.weight[j].template head<dim>() * stencil.displacement[j].template head<dim>().transpose();
                const double difference = exact.dot(stencil.displacement[j].template head<dim>());
                reconstructed += stencil.weight[j].template head<dim>() * difference;
            }
            CHECK((response - Eigen::Matrix<double, dim, dim>::Identity()).cwiseAbs().maxCoeff() < 1e-8);
            CHECK((reconstructed - exact).cwiseAbs().maxCoeff() < 1e-8);
        }
    }
}

TEST_CASE("SST least squares reproduces affine fields on skewed and stretched stencils")
{
    CheckAffineStencils<2>();
    CheckAffineStencils<3>();
}

TEST_CASE("SST rank-deficient geometry is rejected rather than silently using inconsistent gradients")
{
    DNDS::ACMGPU::SSTGradientStencil stencil;
    stencil.displacement = {Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(2, 2, 2)};
    CHECK_THROWS_AS(DNDS::ACMGPU::BuildSSTGradientWeights<3>(stencil), std::runtime_error);
}

TEST_CASE("SST extremum diffusion is conservative and cannot deepen local extrema")
{
    for (int left = 0; left < 4; ++left)
        for (int right = 0; right < 4; ++right)
            for (double normalDerivative : {-10., -1., 0., 1., 10.})
            {
                const double bounded = DNDS::GPU::SSTBoundNormalDerivative(normalDerivative, left, right);
                CHECK(bounded == DNDS::ACM::SSTBoundNormalDerivative(normalDerivative, left, right));
                CHECK(bounded == -DNDS::GPU::SSTBoundNormalDerivative(-normalDerivative, right, left));
                if ((left & 1) || (right & 2)) CHECK(bounded >= 0);
                if ((left & 2) || (right & 1)) CHECK(bounded <= 0);
                if (left == 0 && right == 0) CHECK(bounded == normalDerivative);
            }
}

TEST_CASE("SST prescribed turbulence inflow identifies each supported Dirichlet boundary")
{
    for (int boundary : {1, 7, 8, 10}) CHECK(DNDS::GPU::SSTPrescribedInflow(boundary));
    for (int boundary : {2, 3, 4, 5, 6, 9}) CHECK_FALSE(DNDS::GPU::SSTPrescribedInflow(boundary));
}

TEST_CASE("SST shared diffusion stays bounded on a near-floor plateau")
{
    const double left = 1.0000000000005337e-10, right = 1.0000000000000247e-10;
    const double jump = right - left;
    for (double raw : {-1e20, -1e10, -1., 0., 1., 1e10, 1e20})
        for (double projection : {1e-5, 1., 10.})
        {
            const double coefficient = DNDS::GPU::SSTNormalConductance(raw, jump, projection, 1);
            const double flux = jump * coefficient;
            CHECK(coefficient >= 0);
            CHECK(coefficient <= 2 / projection);
            CHECK(flux * jump >= 0);
            CHECK(std::abs(flux) <= 2 * std::abs(jump) / projection);
            CHECK(coefficient == DNDS::GPU::SSTNormalConductance(-raw, -jump, projection, 1));
        }
}

TEST_CASE("SST equal states have exactly zero diffusion despite arbitrary reconstructed gradients")
{
    for (double raw : {-1e30, -1., 0., 1., 1e30})
    {
        const double coefficient = DNDS::GPU::SSTNormalConductance(raw, 0, 1e-5, 1);
        CHECK(coefficient == doctest::Approx(1e5));
        CHECK(0 * coefficient == 0);
    }
    CHECK(DNDS::GPU::SSTNormalConductance(1e20, 1e-300, 1e-5, 1) == doctest::Approx(2e5));
    CHECK(DNDS::GPU::SSTNormalConductance(-1e20, 1e-300, 1e-5, 1) == 0);
}

TEST_CASE("SST admissible nonorthogonal affine flux retains its physical normal derivative")
{
    const Eigen::Vector3d normal(1, 0, 0), displacement(2, 1, 0), gradient(1, -.5, 0);
    const double jump = gradient.dot(displacement), derivative = gradient.dot(normal);
    const double coefficient = DNDS::GPU::SSTNormalConductance(
        derivative, jump, displacement.dot(normal), normal.squaredNorm());
    CHECK(jump * coefficient == doctest::Approx(derivative));
    CHECK(coefficient >= 0);
}
