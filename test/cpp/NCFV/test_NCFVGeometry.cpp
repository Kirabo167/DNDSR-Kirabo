/**
 * @file test_NCFVGeometry.cpp
 * @brief Polynomial-exactness tests for the median-dual differential rules.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "NCFV/NCFVReconstruction.hpp"
#include "NCFV/NCFVAnalytic.hpp"
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
    CHECK(nlohmann::json("最小二乘重构").get<ReconstructionMethod>() ==
          ReconstructionMethod::LeastSquares);
    CHECK(nlohmann::json("SVD最小二乘重构").get<ReconstructionMethod>() ==
          ReconstructionMethod::SVDLeastSquares);
    CHECK(nlohmann::json("invalid").get<ReconstructionMethod>() ==
          ReconstructionMethod::Unknown);
    DNDS::NCFV::ReconstructionSettings reconstruction;
    CHECK(reconstruction.method == ReconstructionMethod::SVDLeastSquares);
    nlohmann::ordered_json reconstructionJson = reconstruction;
    CHECK(reconstructionJson.at("method") == "SVDLeastSquares");
    reconstructionJson["method"] = "LeastSquares";
    CHECK(reconstructionJson.get<DNDS::NCFV::ReconstructionSettings>().method ==
          ReconstructionMethod::LeastSquares);
    CHECK(nlohmann::json("NoSlipAdiabaticWall").get<BoundaryMode>() ==
          BoundaryMode::NoSlipAdiabaticWall);
    CHECK(nlohmann::json("PressureOutlet").get<BoundaryMode>() ==
          BoundaryMode::PressureOutlet);
    CHECK(nlohmann::json("Sutherland").get<ViscosityModel>() ==
          ViscosityModel::Sutherland);
    CHECK(nlohmann::json("Conservative").get<InitialFieldVariables>() ==
          InitialFieldVariables::Conservative);
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
