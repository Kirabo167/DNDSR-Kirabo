#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "NCFVPlanarCompression.hpp"
#include <array>

namespace
{
    using Sensor = DNDS::NCFV::PlanarCompression<3>;
    using Vector = Sensor::Vector;
    using DNDS::real;

    // Prescribe primitive gradients, then supply the corresponding conserved gradients.
    real Sense(Vector normal, real dp, real compression, bool crossing = false, real boost = 0)
    {
        normal.normalize();
        const Vector tangent = normal.unitOrthogonal();
        const Vector velocity = Vector(0.4, -0.3, 0.1) + boost * normal;
        const Vector gradRho = 0.7 * normal;
        constexpr real gamma = 1.4;
        Sensor::State state;
        state << 1., velocity, 1 / (gamma - 1) + 0.5 * velocity.squaredNorm();
        const Eigen::Matrix3d gradVelocity = -compression * normal * normal.transpose() +
                                             0.2 * normal * tangent.transpose();
        Sensor::Gradient gradient;
        gradient.col(0) = gradRho;
        gradient.middleCols<3>(1) = gradRho * velocity.transpose() + gradVelocity;
        gradient.col(4) = dp * normal / (gamma - 1) + 0.5 * velocity.squaredNorm() * gradRho +
                          gradVelocity * velocity;
        CHECK((Sensor::PressureGradient(state, gradient, gamma) - dp * normal).norm() < 1e-12);
        std::array<Vector, 7> offsets{Vector::Zero(), Vector::UnitX(), -Vector::UnitX(),
                                      Vector::UnitY(), -Vector::UnitY(), Vector::UnitZ(), -Vector::UnitZ()};
        std::array<real, 7> pressures{};
        for (size_t i = 0; i < offsets.size(); ++i)
            pressures[i] = 1. + dp * normal.dot(offsets[i]);
        const std::array<DNDS::index, 6> neighbors{1, 2, 3, 4, 5, 6};
        return Sensor::Weight(0, state, gradient, gamma, neighbors, pressures,
                              [&](DNDS::index i) { return offsets[i]; },
                              [&](DNDS::index i) -> Vector
                              { return dp * (crossing && i % 2 == 0 ? tangent : normal); });
    }
}

TEST_CASE("Only a coherent planar pressure jump with normal compression changes the P1 slopes")
{
    const Vector oblique(1, 1, 0.7);
    CHECK(Sense(Vector::UnitX(), 0.2, 0.25) == 1.);
    CHECK(Sense(Vector::UnitY(), 0.2, 0.25) == 1.);
    CHECK(Sense(oblique, 0.2, 0.25) == 1.);
    CHECK(Sense(oblique, 0.2, 0.25, false, 3.0) == 1.);
    CHECK(Sense(oblique, 0., 0.) == 0.);       // constant-pressure contact with shear
    CHECK(Sense(oblique, 0.2, 0.) == 0.);      // tangential velocity gradient
    CHECK(Sense(oblique, 0.2, -0.25) == 0.);   // expansion
    CHECK(Sense(oblique, 0.002, 0.25) == 0.);  // smooth small pressure variations
    CHECK(Sense(oblique, 0.2, 0.25, true) == 0.); // two distinct pressure-gradient directions
    const real transition = Sense(Vector::UnitX(), 0.2, 0.04);
    CHECK(transition > 0.);
    CHECK(transition < 1.);
}
