/**
 * @file test_NCFVGeometry.cpp
 * @brief Polynomial-exactness tests for the median-dual differential rules.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "NCFV/NCFVReconstruction.hpp"
#include "NCFV/NCFVAnalytic.hpp"
#include "NCFV/NCFVBoundary.hpp"
#include "Geom/Quadrature.hpp"

#include <cmath>
#include <initializer_list>
#include <vector>

namespace
{
    using DNDS::index;
    using DNDS::real;
    using DNDS::NCFV::AffinePoint;
    using DNDS::NCFV::DualGeometry;
    using DNDS::NCFV::Matrix3;
    using DNDS::NCFV::RawMoments;
    using DNDS::NCFV::Vector3;

    AffinePoint MakeAverage(
        const std::vector<Vector3> &nodeCoordinates,
        std::initializer_list<index> nodes)
    {
        AffinePoint result;
        const real coefficient = 1.0 / static_cast<real>(nodes.size());
        for (index node : nodes)
        {
            result.coordinate += coefficient *
                                 nodeCoordinates.at(static_cast<std::size_t>(node));
            result.support.push_back({node, coefficient});
        }
        return result;
    }

    real ExactQuadraticIntegral(
        const RawMoments &moments,
        const Vector3 &anchor,
        real anchorValue,
        const Vector3 &anchorGradient,
        const Matrix3 &hessian)
    {
        const Vector3 centredFirst =
            moments.first - moments.measure * anchor;
        Matrix3 centredSecond = moments.second;
        centredSecond -= moments.first * anchor.transpose();
        centredSecond -= anchor * moments.first.transpose();
        centredSecond += moments.measure * anchor * anchor.transpose();
        return moments.measure * anchorValue +
               anchorGradient.dot(centredFirst) +
               0.5 * (hessian.cwiseProduct(centredSecond)).sum();
    }

    void CheckDifferentialRule(
        const std::vector<Vector3> &nodeCoordinates,
        const std::vector<AffinePoint> &simplex,
        int simplexDimension)
    {
        constexpr index anchorNode = 0;
        const Vector3 anchor = nodeCoordinates[0];
        std::vector<Vector3> simplexCoordinates;
        simplexCoordinates.reserve(simplex.size());
        for (const AffinePoint &point : simplex)
            simplexCoordinates.push_back(point.coordinate);

        const auto [measure, signedJacobian] =
            DualGeometry::SimplexMeasure(simplexCoordinates, simplexDimension);
        CHECK(measure > 0);
        CHECK(std::abs(signedJacobian) > 0);
        const RawMoments moments =
            DualGeometry::ExactSimplexMoments(simplexCoordinates, measure);

        const real anchorValue = 0.73;
        const Vector3 anchorGradient{0.31, -0.47, 0.29};
        Matrix3 hessian;
        hessian << 0.8, -0.2, 0.13,
            -0.2, -0.5, 0.17,
            0.13, 0.17, 0.4;

        const auto weights = DualGeometry::DifferentialWeights(
            simplex, anchorNode, anchor, measure);
        real differentialIntegral = measure * anchorValue;
        for (const auto &weight : weights)
        {
            const Vector3 displacement =
                nodeCoordinates.at(static_cast<std::size_t>(weight.node)) - anchor;
            const Vector3 gradient = anchorGradient + hessian * displacement;
            differentialIntegral += weight.value.dot(gradient);
        }

        const real exactIntegral = ExactQuadraticIntegral(
            moments, anchor, anchorValue, anchorGradient, hessian);
        CHECK(differentialIntegral == doctest::Approx(exactIntegral).epsilon(2e-13));
    }
}

TEST_CASE("NCFV exact simplex moments")
{
    const std::vector<Vector3> triangle{
        Vector3{0, 0, 0}, Vector3{2, 0, 0}, Vector3{0, 3, 0}};
    const auto [triangleArea, triangleJacobian] =
        DualGeometry::SimplexMeasure(triangle, 2);
    CHECK(triangleArea == doctest::Approx(3.0));
    CHECK(triangleJacobian == doctest::Approx(6.0));
    const RawMoments triangleMoments =
        DualGeometry::ExactSimplexMoments(triangle, triangleArea);
    CHECK(triangleMoments.first.x() == doctest::Approx(2.0));
    CHECK(triangleMoments.first.y() == doctest::Approx(3.0));
    CHECK(triangleMoments.second(0, 0) == doctest::Approx(2.0));
    CHECK(triangleMoments.second(1, 1) == doctest::Approx(4.5));
    CHECK(triangleMoments.second(0, 1) == doctest::Approx(1.5));

    const std::vector<Vector3> tetrahedron{
        Vector3{0, 0, 0}, Vector3{2, 0, 0},
        Vector3{0, 3, 0}, Vector3{0, 0, 4}};
    const auto [tetrahedronVolume, tetrahedronJacobian] =
        DualGeometry::SimplexMeasure(tetrahedron, 3);
    CHECK(tetrahedronVolume == doctest::Approx(4.0));
    CHECK(tetrahedronJacobian == doctest::Approx(24.0));
    const RawMoments tetrahedronMoments =
        DualGeometry::ExactSimplexMoments(tetrahedron, tetrahedronVolume);
    CHECK(tetrahedronMoments.first.x() == doctest::Approx(2.0));
    CHECK(tetrahedronMoments.first.y() == doctest::Approx(3.0));
    CHECK(tetrahedronMoments.first.z() == doctest::Approx(4.0));
    CHECK(tetrahedronMoments.second(0, 0) == doctest::Approx(1.6));
    CHECK(tetrahedronMoments.second(0, 1) == doctest::Approx(1.2));
    CHECK(tetrahedronMoments.second(1, 2) == doctest::Approx(2.4));
}

TEST_CASE("NCFV all-differential rules integrate quadratics exactly")
{
    const std::vector<Vector3> nodes{
        Vector3{0.2, -0.1, 0.3}, Vector3{1.7, 0.2, 0.1},
        Vector3{0.1, 1.5, 0.4}, Vector3{0.3, 0.4, 2.0}};

    const AffinePoint node = MakeAverage(nodes, {0});
    const AffinePoint edge = MakeAverage(nodes, {0, 1});
    const AffinePoint face = MakeAverage(nodes, {0, 1, 2});
    const AffinePoint cell = MakeAverage(nodes, {0, 1, 2, 3});

    SUBCASE("dual micro-tetrahedron volume rule")
    {
        CheckDifferentialRule(nodes, {node, edge, face, cell}, 3);
    }
    SUBCASE("dual micro-triangle surface rule")
    {
        CheckDifferentialRule(nodes, {edge, face, cell}, 2);
    }
    SUBCASE("dual micro-line surface rule")
    {
        CheckDifferentialRule(nodes, {edge, cell}, 1);
    }
}

TEST_CASE("NCFV integration mode JSON names are stable")
{
    CHECK(std::string(DNDS::NCFV::MethodName) == "Node Center Finite Volume Method");
    using DNDS::NCFV::BoundaryMode;
    using DNDS::NCFV::InitialFieldVariables;
    using DNDS::NCFV::IntegrationMode;
    using DNDS::NCFV::ReconstructionMethod;
    using DNDS::NCFV::ViscosityModel;
    const nlohmann::json efficient = "EfficientDifferential";
    const nlohmann::json traditional = "TraditionalQuadrature";
    CHECK(efficient.get<IntegrationMode>() == IntegrationMode::EfficientDifferential);
    CHECK(traditional.get<IntegrationMode>() == IntegrationMode::TraditionalQuadrature);
    CHECK(nlohmann::json("LeastSquares").get<ReconstructionMethod>() ==
          ReconstructionMethod::LeastSquares);
    CHECK(nlohmann::json("SVDLeastSquares").get<ReconstructionMethod>() ==
          ReconstructionMethod::SVDLeastSquares);
    CHECK(nlohmann::json("Variational").get<ReconstructionMethod>() ==
          ReconstructionMethod::Variational);
    CHECK(nlohmann::json("变分重构").get<ReconstructionMethod>() ==
          ReconstructionMethod::Variational);
    CHECK(nlohmann::json("最小二乘重构").get<ReconstructionMethod>() ==
          ReconstructionMethod::LeastSquares);
    CHECK(nlohmann::json("SVD最小二乘重构").get<ReconstructionMethod>() ==
          ReconstructionMethod::SVDLeastSquares);
    CHECK(nlohmann::json("invalid").get<ReconstructionMethod>() ==
          ReconstructionMethod::Unknown);
    DNDS::NCFV::ReconstructionSettings reconstruction;
    CHECK(reconstruction.method == ReconstructionMethod::SVDLeastSquares);
    CHECK(reconstruction.variationalWeight == doctest::Approx(1.0));
    nlohmann::ordered_json reconstructionJson = reconstruction;
    CHECK(reconstructionJson.at("method") == "SVDLeastSquares");
    reconstructionJson["method"] = "LeastSquares";
    CHECK(reconstructionJson.get<DNDS::NCFV::ReconstructionSettings>().method ==
          ReconstructionMethod::LeastSquares);
    reconstructionJson["method"] = "Variational";
    reconstructionJson["variationalWeight"] = 5.0;
    const auto variationalSettings =
        reconstructionJson.get<DNDS::NCFV::ReconstructionSettings>();
    CHECK(variationalSettings.method == ReconstructionMethod::Variational);
    CHECK(variationalSettings.variationalWeight == doctest::Approx(5.0));
    CHECK(nlohmann::json("NoSlipAdiabaticWall").get<BoundaryMode>() ==
          BoundaryMode::NoSlipAdiabaticWall);
    CHECK(nlohmann::json("PressureOutlet").get<BoundaryMode>() ==
          BoundaryMode::PressureOutlet);
    CHECK(nlohmann::json("Sutherland").get<ViscosityModel>() ==
          ViscosityModel::Sutherland);
    CHECK(nlohmann::json("Conservative").get<InitialFieldVariables>() ==
          InitialFieldVariables::Conservative);
}

TEST_CASE("NCFV coplanar boundary macroface integrates a quadratic and separates corners and zones")
{
    using namespace DNDS::NCFV;
    const std::vector<Vector3> x{{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}, {0,0,1}};
    NodeControlVolume volume;
    volume.node = 0;
    const auto add = [&](int j, int k, int zone)
    {
        BoundaryPiece piece;
        piece.zone = zone;
        piece.nPoints = 3;
        piece.points = {MakeAverage(x, {0}), MakeAverage(x, {j}), MakeAverage(x, {k})};
        piece.vectorMeasure = .5 * (x[j] - x[0]).cross(x[k] - x[0]);
        piece.measure = piece.vectorMeasure.norm();
        volume.boundaryPieces.push_back(piece);
    };
    add(1, 2, 1);
    add(2, 3, 1);
    add(3, 4, 1); // Perpendicular normal.
    add(2, 1, 1); // Opposite normal.
    add(1, 2, 2); // Same normal, distinct boundary data.
    const auto faces = volume.BuildBoundaryMacroFaces();
    REQUIRE(faces.size() == 4);
    const auto &face = faces.front();
    REQUIRE(face.weights.size() == 4);
    CHECK(face.measure == doctest::Approx(1));
    real integral = face.measure;
    for (const auto &entry : face.weights)
    {
        const auto &point = x[entry.node];
        const Vector3 gradient{2 * point.x() + 3 * point.y(), 2 + 3 * point.x(), 0};
        integral += gradient.dot(entry.value);
    }
    CHECK(integral == doctest::Approx(37. / 12.).epsilon(2e-14));
}

TEST_CASE("NCFV P2 and quadrature volume sources give the same exact quadratic mean")
{
    using namespace DNDS::NCFV;
    const std::vector<Vector3> x{{0,0,0}, {2,0,0}, {0,3,0}, {0,0,4}};
    const std::vector<AffinePoint> points{MakeAverage(x, {0}), MakeAverage(x, {1}), MakeAverage(x, {2}), MakeAverage(x, {3})};
    NodeControlVolume volume;
    volume.moments.measure = 4;
    for (const auto &w : DualGeometry::DifferentialWeights(points, 0, x[0], volume.moments.measure))
        volume.pointRecoveryStencil.push_back({w.node, w.value / volume.moments.measure});
    const auto value = [](const Vector3 &p) -> Eigen::Vector2d { return {1 + p.x() + p.z() * p.z(), 2.3}; };
    const auto gradient = [&](DNDS::index node)
    {
        Eigen::Matrix<real, 3, 2> g = Eigen::Matrix<real, 3, 2>::Zero();
        g.col(0) << 1, 0, 2 * x[node].z();
        return g;
    };
    const Eigen::Vector2d expected{3.1, 2.3};
    CHECK((volume.VolumeMean<3>(x[0], value, gradient) - expected).norm() < 1e-13);

    volume.pointRecoveryStencil.clear();
    DNDS::Geom::Elem::Quadrature quadrature(DNDS::Geom::Elem::Element{DNDS::Geom::Elem::Tet4}, 2);
    for (int i = 0; i < quadrature.GetNumPoints(); ++i)
    {
        const auto [p, w] = quadrature.GetQuadraturePointInfo(i);
        volume.volumeQuadrature.push_back({p[0] * x[1] + p[1] * x[2] + p[2] * x[3], 24 * w});
    }
    CHECK((volume.VolumeMean<3>(x[0], value, gradient) - expected).norm() < 1e-13);
}

TEST_CASE("NCFV wall flux satisfies the shock or expansion relation and differentiates the applied flux")
{
    using namespace DNDS::NCFV;
    using State = Eigen::Vector<DNDS::real, 5>;
    constexpr DNDS::real gamma = 1.4, rho = 1.2, pressure = 1.3, eps = 1e-5;
    const DNDS::real sound = std::sqrt(gamma * pressure / rho);
    const Vector3 normal = Vector3{2, -1, 3}.normalized();
    const Vector3 tangent = .24 * normal.unitOrthogonal();
    const State increment{.14, .2, -.13, .19, -.16};
    for (const DNDS::real speed : {0., .09})
        for (const DNDS::real relative : {-.7, .8})
        {
            const Vector3 velocity = tangent + (speed + relative) * normal;
            State inside;
            inside << rho, rho * velocity, pressure / (gamma - 1) + .5 * rho * velocity.squaredNorm();
            const auto [flux, derivative] = WallCharacteristicFlux<3>(inside, increment, normal, gamma, speed);
            // Subtract the normal convective momentum flux to recover the wall pressure.
            const DNDS::real wallPressure = normal.dot(flux.segment<3>(1)) - speed * flux[0];
            REQUIRE(wallPressure > 0);
            DNDS::real wallRho;
            if (relative > 0)
            {
                const DNDS::real beta = (gamma - 1) / (gamma + 1);
                const DNDS::real jump = (wallPressure - pressure) *
                    std::sqrt(2 / ((gamma + 1) * rho * (wallPressure + beta * pressure)));
                CHECK(jump == doctest::Approx(relative).epsilon(1e-12));
                wallRho = rho * (wallPressure + beta * pressure) / (beta * wallPressure + pressure);
            }
            else
            {
                wallRho = rho * std::pow(wallPressure / pressure, 1 / gamma);
                const DNDS::real wallSound = std::sqrt(gamma * wallPressure / wallRho);
                CHECK(2 * (wallSound - sound) / (gamma - 1) == doctest::Approx(relative).epsilon(1e-12));
            }
            const Vector3 wallVelocity = tangent + speed * normal;
            State expected;
            expected << wallRho * speed, wallRho * speed * wallVelocity + wallPressure * normal,
                speed * (gamma * wallPressure / (gamma - 1) + .5 * wallRho * wallVelocity.squaredNorm());
            CHECK((flux - expected).norm() < 2e-12);
            const auto plus = WallCharacteristicFlux<3>(inside + eps * increment, State::Zero(), normal, gamma, speed).first;
            const auto minus = WallCharacteristicFlux<3>(inside - eps * increment, State::Zero(), normal, gamma, speed).first;
            CHECK((derivative - (plus - minus) / (2 * eps)).norm() < 2e-9);
        }

    const Vector3 separated = -2.1 * sound / (gamma - 1) * normal;
    State vacuumInterior;
    vacuumInterior << rho, rho * separated, pressure / (gamma - 1) + .5 * rho * separated.squaredNorm();
    const auto vacuum = WallCharacteristicFlux<3>(vacuumInterior, increment, normal, gamma, 0);
    CHECK(vacuum.first.isZero());
    CHECK(vacuum.second.isZero());
}

TEST_CASE("NCFV pressure outlet selects acoustic and material data for each flow regime")
{
    using namespace DNDS::NCFV;
    using State = Eigen::Vector<DNDS::real, 5>;
    constexpr DNDS::real gamma = 1.4, rho = 1.2, pressure = 1.3, eps = 1e-5;
    const Vector3 normal = Vector3{2, -1, 3}.normalized();
    const State increment{.14, .2, -.13, .19, -.16};
    BoundaryZoneSettings boundary;
    boundary.mode = BoundaryMode::PressureOutlet;
    boundary.staticPressure = 1.1;
    boundary.primitive = {1.08, -.4, .3, -.5, .95};
    const Eigen::Map<const Vector3> reservoir(boundary.primitive.data() + 1);
    const DNDS::real sound = std::sqrt(gamma * pressure / rho);
    for (const DNDS::real vn : {-1.8, -.82, .45, 1.8})
    {
        const Vector3 velocity = .24 * normal.unitOrthogonal() + vn * normal;
        State inside;
        inside << rho, rho * velocity, pressure / (gamma - 1) + .5 * rho * velocity.squaredNorm();
        const auto [trace, derivative] = PressureOutletCharacteristicTrace<3>(inside, increment, normal, gamma, boundary);
        if (vn > sound)
        {
            CHECK((trace - inside).norm() == 0);
            CHECK((derivative - increment).norm() == 0);
            continue;
        }
        const Vector3 u = trace.segment<3>(1) / trace[0];
        const DNDS::real p = (gamma - 1) * (trace[4] - .5 * trace[0] * u.squaredNorm());
        CHECK(p == doctest::Approx(boundary.staticPressure).epsilon(1e-12));
        if (vn < -sound)
        {
            CHECK(trace[0] == doctest::Approx(boundary.primitive.front()).epsilon(1e-12));
            CHECK((u - reservoir).norm() < 2e-12);
            CHECK(derivative.isZero());
            continue;
        }
        CHECK(u.dot(normal) + 2 * std::sqrt(gamma * p / trace[0]) / (gamma - 1) ==
              doctest::Approx(vn + 2 * sound / (gamma - 1)).epsilon(1e-12));
        const DNDS::real entropy = vn < 0 ? boundary.primitive.back() / std::pow(boundary.primitive.front(), gamma)
                                         : pressure / std::pow(rho, gamma);
        CHECK(p / std::pow(trace[0], gamma) == doctest::Approx(entropy).epsilon(1e-12));
        const Vector3 expectedVelocity = vn < 0 ? Vector3(reservoir) : velocity;
        const Vector3 difference = u - expectedVelocity;
        CHECK((difference - normal * normal.dot(difference)).norm() < 2e-12);
        const auto plus = PressureOutletCharacteristicTrace<3>(inside + eps * increment, State::Zero(), normal, gamma, boundary).first;
        const auto minus = PressureOutletCharacteristicTrace<3>(inside - eps * increment, State::Zero(), normal, gamma, boundary).first;
        CHECK((derivative - (plus - minus) / (2 * eps)).norm() < 2e-9);
    }
}

TEST_CASE("NCFV exponential manufactured source equals U_t plus the divergence of the Euler flux")
{
    using MMS = DNDS::NCFV::ExponentialMMS;
    for (real phase : {-.4, 0., .7, 1.5})
    {
        const auto basis = MMS::Powers(phase);
        const real e = std::exp(phase), rho = 1 + .1 * e, drho = .1 * e;
        const real p = 1.3 + 1.2 * e, dp = 1.2 * e;
        const Vector3 dv = Vector3{.1,.2,.3} * e, v = Vector3{.2,.1,.3} + dv;
        const real energy = p / .4 + .5 * rho * v.squaredNorm();
        MMS::Basis state;
        state << rho, rho * v, energy;
        CHECK((MMS::State(basis) - state).norm() < 1e-13);
        const Vector3 dm = drho * v + rho * dv;
        const real dEnergy = dp / .4 + .5 * drho * v.squaredNorm() + rho * v.dot(dv);
        MMS::Basis du;
        du << drho, dm, dEnergy;
        CHECK((MMS::State(MMS::PhaseDerivative(basis)) - du).norm() < 1e-13);
        const Vector3 a{.75,.35,.65};
        MMS::Basis divF;
        divF << dm.dot(a), dm * v.dot(a) + rho * v * dv.dot(a) + dp * a,
            (dEnergy + dp) * v.dot(a) + (energy + p) * dv.dot(a);
        CHECK((MMS::sourceCoefficients.transpose() * basis - (.65 * du + divF)).norm() < 2e-13);
    }
}

TEST_CASE("NCFV manufactured characteristic trace uses incoming data and reconstructed outgoing data")
{
    using MMS = DNDS::NCFV::ExponentialMMS;
    using State = MMS::Basis;
    constexpr DNDS::real gamma = 1.4, exponential = 1.7, dPhase = .37;
    const State exact = MMS::State(MMS::PowersOf(exponential));
    const State increment{.07, .03, -.08, .1, -.15};
    State inside = exact;
    inside += State{.06, -.18, .11, .04, -.26};

    for (int sign : {-1, 1})
    {
        const Vector3 normal = sign * Vector3{.3, .4, .8}.normalized();
        const Vector3 tangent = normal.unitOrthogonal();
        const Vector3 bitangent = normal.cross(tangent);
        const auto characteristics = [&](const State &state) -> State
        {
            const DNDS::real rho = state[0];
            const Vector3 v = state.segment<3>(1) / rho;
            const DNDS::real p = (gamma - 1) * (state[4] - .5 * rho * v.squaredNorm());
            const DNDS::real un = normal.dot(v), a = std::sqrt(gamma * p / rho);
            REQUIRE(std::abs(un) < a);
            return {un + 2 * a / (gamma - 1), un - 2 * a / (gamma - 1),
                    p / std::pow(rho, gamma), tangent.dot(v), bitangent.dot(v)};
        };
        const auto [trace, differential] = MMS::CharacteristicTrace(inside, increment, normal, exponential, dPhase);
        CHECK((trace - MMS::CharacteristicTrace(inside, normal, exponential)).norm() < 1e-13);
        const State computed = characteristics(trace);
        const State inner = characteristics(inside), prescribed = characteristics(exact);
        CHECK(std::abs(computed[0] - inner[0]) < 2e-12);
        CHECK(std::abs(computed[1] - prescribed[1]) < 2e-12);
        const State &material = sign < 0 ? prescribed : inner;
        CHECK((computed.tail<3>() - material.tail<3>()).norm() < 2e-12);

        constexpr DNDS::real eps = 1e-5;
        const State plus = MMS::CharacteristicTrace(inside + eps * increment, normal, exponential * std::exp(eps * dPhase));
        const State minus = MMS::CharacteristicTrace(inside - eps * increment, normal, exponential * std::exp(-eps * dPhase));
        CHECK((differential - (plus - minus) / (2 * eps)).norm() < 1e-8);
    }
}

TEST_CASE("NCFV two-dimensional reservoir trace switches all data at supersonic inflow and outflow")
{
    using State = Eigen::Vector<DNDS::real, 4>;
    constexpr DNDS::real gamma = 1.4;
    const std::vector<DNDS::real> reservoir{1.2, 2.1, .3, .9};
    const State prescribed{1.2, 1.2 * 2.1, 1.2 * .3, .9 / (gamma - 1) + .6 * (2.1 * 2.1 + .3 * .3)};
    const State inside{1, 3, .2, 1 / (gamma - 1) + .5 * (9 + .04)};
    const State delta{.01, -.2, .08, .3};
    for (int sign : {-1, 1})
    {
        const Eigen::Vector2d normal{static_cast<DNDS::real>(sign), 0};
        const auto [trace, derivative] = DNDS::NCFV::ReservoirCharacteristicTrace<2>(inside, delta, normal, gamma, reservoir);
        const State expectedDerivative = sign < 0 ? State::Zero().eval() : delta;
        CHECK((trace - (sign < 0 ? prescribed : inside)).norm() < 1e-12);
        CHECK((derivative - expectedDerivative).norm() < 1e-12);
    }
}

TEST_CASE("NCFV variational reconstruction accepts both integration modes")
{
    DNDS::NCFV::Configuration configuration;
    configuration.mesh.meshFile = "unused.cgns";
    configuration.algorithm.mode = DNDS::NCFV::IntegrationMode::EfficientDifferential;
    configuration.reconstruction.method = DNDS::NCFV::ReconstructionMethod::Variational;
    CHECK_NOTHROW(configuration.Validate());
    configuration.algorithm.mode = DNDS::NCFV::IntegrationMode::TraditionalQuadrature;
    CHECK_NOTHROW(configuration.Validate());
    configuration.algorithm.mode = DNDS::NCFV::IntegrationMode::EfficientDifferential;
    configuration.reconstruction.method = DNDS::NCFV::ReconstructionMethod::SVDLeastSquares;
    CHECK_NOTHROW(configuration.Validate());
}

TEST_CASE("NCFV third-order surface quadrature integrates quadratic traces exactly")
{
    const DNDS::NCFV::AlgorithmSettings settings;
    CHECK(settings.surfaceQuadratureOrder == 3);
    CHECK(settings.SurfaceQuadraturePolynomialDegree() == 2);

    const DNDS::Geom::Elem::Quadrature lineRule(
        DNDS::Geom::Elem::Element{DNDS::Geom::Elem::Line2},
        settings.SurfaceQuadraturePolynomialDegree());
    CHECK(lineRule.GetNumPoints() == 2);
    real lineQuadraticIntegral = 0;
    for (int iG = 0; iG < lineRule.GetNumPoints(); iG++)
    {
        const auto [point, weight] = lineRule.GetQuadraturePointInfo(iG);
        lineQuadraticIntegral += weight * point[0] * point[0];
    }
    CHECK(lineQuadraticIntegral == doctest::Approx(2.0 / 3.0).epsilon(2e-14));

    const DNDS::Geom::Elem::Quadrature triangleRule(
        DNDS::Geom::Elem::Element{DNDS::Geom::Elem::Tri3},
        settings.SurfaceQuadraturePolynomialDegree());
    CHECK(triangleRule.GetNumPoints() == 3);
    real triangleX2Integral = 0;
    real triangleXYIntegral = 0;
    for (int iG = 0; iG < triangleRule.GetNumPoints(); iG++)
    {
        const auto [point, weight] = triangleRule.GetQuadraturePointInfo(iG);
        triangleX2Integral += weight * point[0] * point[0];
        triangleXYIntegral += weight * point[0] * point[1];
    }
    CHECK(triangleX2Integral == doctest::Approx(1.0 / 12.0).epsilon(2e-14));
    CHECK(triangleXYIntegral == doctest::Approx(1.0 / 24.0).epsilon(2e-14));
}

TEST_CASE("NCFV quadratic basis gradient is the exact physical derivative")
{
    using DNDS::NCFV::Reconstruction;

    const Vector3 referenceLengths{0.37, 1.37, 2.19};
    const Vector3 displacement{0.31, -0.22, 0.47};
    constexpr DNDS::real epsilon = 1e-7;
    for (int dimension : {2, 3})
    {
        const Eigen::MatrixXd analytical =
            Reconstruction::EvaluateBasisGradient(
                displacement, referenceLengths, dimension);
        for (int direction = 0; direction < dimension; direction++)
        {
            Vector3 perturbation = Vector3::Zero();
            perturbation(direction) = epsilon;
            const Eigen::VectorXd numerical =
                (Reconstruction::EvaluateBasis(
                     displacement + perturbation, referenceLengths, dimension) -
                 Reconstruction::EvaluateBasis(
                     displacement - perturbation, referenceLengths, dimension)) /
                (2.0 * epsilon);
            CHECK((analytical.row(direction).transpose() - numerical).norm() < 2e-9);
        }
    }
}

TEST_CASE("NCFV analytic vortex conservative gradient and periodic translation")
{
    DNDS::NCFV::Configuration configuration;
    configuration.dimension = 3;
    configuration.physics.initialPrimitive = {1, 1, 1, 0, 1};
    configuration.mesh.periodicLengths = {10, 10, 4};
    const Vector3 point{4.3, 5.7, 1.6};
    const auto [state, gradient] = DNDS::NCFV::IsentropicVortex<3>(configuration, point, 0);
    constexpr DNDS::real epsilon = 1e-6;
    for (int d = 0; d < 3; d++)
    {
        const Vector3 perturbation = epsilon * Vector3::Unit(d);
        const auto plus = DNDS::NCFV::IsentropicVortex<3>(configuration, point + perturbation, 0).first;
        const auto minus = DNDS::NCFV::IsentropicVortex<3>(configuration, point - perturbation, 0).first;
        CHECK(((plus - minus) / (2 * epsilon) - gradient.row(d).transpose()).norm() < 2e-9);
    }
    const auto translated = DNDS::NCFV::IsentropicVortex<3>(configuration, point + Vector3{12, 2, 4}, 2).first;
    CHECK((translated - state).norm() < 2e-13);
}

TEST_CASE("NCFV thesis reference lengths are directional half spans")
{
    const Vector3 lower{-0.02, -2, -30}, upper{0.04, 6, 10};
    const Vector3 lengths = DualGeometry::ReferenceLengthsFromBounds(lower, upper, 3);
    CHECK((lengths - Vector3{0.03, 4, 20}).norm() < 1e-14);
    const Vector3 flat = DualGeometry::ReferenceLengthsFromBounds(
        Vector3{-1, -2, 0}, Vector3{3, 6, 0}, 2);
    CHECK((flat - Vector3{2, 4, 1}).norm() < 1e-14);
    CHECK_THROWS(DualGeometry::ReferenceLengthsFromBounds(
        Vector3{-1, -2, 0}, Vector3{3, 6, 0}, 3));
}

TEST_CASE("NCFV anisotropic zero mean bases match independent simplex integration")
{
    using DNDS::NCFV::Reconstruction;
    const Vector3 lengths{0.03, 2, 7};
    const Vector3 anchor{0.02, -0.7, 1.3};
    for (int dimension : {2, 3})
    {
        std::vector<Vector3> points{anchor};
        for (int d = 0; d < dimension; d++)
            points.push_back(anchor + 2 * lengths(d) * Vector3::Unit(d));
        const auto [measure, jacobian] = DualGeometry::SimplexMeasure(points, dimension);
        const RawMoments moments = DualGeometry::ExactSimplexMoments(points, measure);
        const Eigen::VectorXd mean = Reconstruction::MeanBasis(moments, anchor, lengths, dimension);
        Eigen::VectorXd integrated = Eigen::VectorXd::Zero(mean.size());
        if (dimension == 2)
        {
            for (int d = 0; d < 3; d++)
                integrated += Reconstruction::EvaluateBasis(
                                  0.5 * (points[d] + points[(d + 1) % 3]) - anchor, lengths, dimension) /
                              3;
        }
        else
        {
            const DNDS::real a = (5 + 3 * std::sqrt(5.0)) / 20;
            const DNDS::real b = (5 - std::sqrt(5.0)) / 20;
            for (int q = 0; q < 4; q++)
            {
                Vector3 point = Vector3::Zero();
                for (int d = 0; d < 4; d++)
                    point += (d == q ? a : b) * points[d];
                integrated += Reconstruction::EvaluateBasis(point - anchor, lengths, dimension) / 4;
            }
        }
        CHECK((integrated - mean).norm() < 2e-13);
        CHECK((mean.head(dimension) - Eigen::VectorXd::Constant(dimension, 2.0 / (dimension + 1))).norm() < 2e-13);

        const Vector3 normalized{0.2, -0.3, 0.4};
        const auto values = Reconstruction::EvaluateBasis(lengths.cwiseProduct(normalized), lengths, dimension);
        CHECK((values.head(dimension) - normalized.head(dimension)).norm() < 2e-14);
        const auto derivative = Reconstruction::EvaluateBasisGradient(Vector3::Zero(), lengths, dimension);
        CHECK((derivative.leftCols(dimension) -
               lengths.head(dimension).cwiseInverse().asDiagonal().toDenseMatrix())
                  .norm() < 2e-14);
    }
}
