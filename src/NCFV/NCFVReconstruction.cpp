#include "NCFVReconstruction.hpp"

#include "DNDS/Errors.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>

namespace DNDS::NCFV
{
    namespace
    {
        Vector3 InverseReferenceLengths(const Vector3 &lengths, int dimension)
        {
            DNDS_check_throw_info(dimension == 2 || dimension == 3,
                                  "NCFV quadratic basis requires dimension 2 or 3");
            DNDS_check_throw_info(lengths.head(dimension).allFinite() &&
                                      lengths.head(dimension).minCoeff() > verySmallReal,
                                  "NCFV reconstruction has invalid reference lengths");
            Vector3 inverse = Vector3::Ones();
            inverse.head(dimension) = lengths.head(dimension).cwiseInverse();
            return inverse;
        }

        Eigen::VectorXd CubicPointBasis(
            const Vector3 &displacement,
            const Vector3 &referenceLengths,
            int dimension)
        {
            const Vector3 x = displacement.cwiseProduct(
                InverseReferenceLengths(referenceLengths, dimension));
            if (dimension == 2)
            {
                Eigen::VectorXd basis(4);
                basis << x.x() * x.x() * x.x() / 6.0,
                    x.x() * x.x() * x.y() / 2.0,
                    x.x() * x.y() * x.y() / 2.0,
                    x.y() * x.y() * x.y() / 6.0;
                return basis;
            }
            Eigen::VectorXd basis(10);
            basis << x.x() * x.x() * x.x() / 6.0,
                x.x() * x.x() * x.y() / 2.0,
                x.x() * x.x() * x.z() / 2.0,
                x.x() * x.y() * x.y() / 2.0,
                x.x() * x.y() * x.z(),
                x.x() * x.z() * x.z() / 2.0,
                x.y() * x.y() * x.y() / 6.0,
                x.y() * x.y() * x.z() / 2.0,
                x.y() * x.z() * x.z() / 2.0,
                x.z() * x.z() * x.z() / 6.0;
            return basis;
        }

        Eigen::MatrixXd WeightedWeakDirections(
            const Eigen::MatrixXd &weighted,
            ReconstructionMethod method)
        {
            const Eigen::Index nBasis = weighted.cols();
            Eigen::VectorXd singularValues;
            Eigen::MatrixXd directions;
            if (method == ReconstructionMethod::SVDLeastSquares)
            {
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                    weighted, Eigen::ComputeFullV);
                singularValues = svd.singularValues();
                directions = svd.matrixV();
            }
            else
            {
                const Eigen::MatrixXd normalMatrix =
                    weighted.transpose() * weighted;
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(
                    normalMatrix);
                if (eigen.info() != Eigen::Success)
                    return Eigen::MatrixXd::Identity(nBasis, nBasis);
                singularValues = eigen.eigenvalues()
                                     .cwiseMax(0.0)
                                     .cwiseSqrt();
                directions = eigen.eigenvectors();
            }

            const real largest =
                singularValues.size() ? singularValues.maxCoeff() : 0;
            if (largest <= verySmallReal)
                return Eigen::MatrixXd::Identity(nBasis, nBasis);
            const real regularization = std::max(
                largest * 1e-6, verySmallReal);
            for (Eigen::Index i = 0; i < nBasis; i++)
            {
                const real singular =
                    i < singularValues.size() ? singularValues(i) : 0;
                directions.col(i) *=
                    largest / std::max(singular, regularization);
            }
            return directions;
        }

        struct WeightedLeastSquaresOperator
        {
            Eigen::MatrixXd inverse;
            int rank = 0;
            real condition = veryLargeReal;
        };

        WeightedLeastSquaresOperator BuildWeightedLeastSquaresOperator(
            const Eigen::MatrixXd &weighted,
            const Eigen::VectorXd &weights,
            ReconstructionMethod method,
            real relativeTolerance)
        {
            WeightedLeastSquaresOperator result;
            const Eigen::Index nBasis = weighted.cols();
            if (method == ReconstructionMethod::SVDLeastSquares)
            {
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                    weighted, Eigen::ComputeThinU | Eigen::ComputeThinV);
                const auto singularValues = svd.singularValues();
                const real largest = singularValues.size()
                                         ? singularValues(0)
                                         : 0;
                const real cutoff = relativeTolerance * largest;
                Eigen::VectorXd inverseSingular = singularValues;
                real smallestAccepted = veryLargeReal;
                for (Eigen::Index i = 0;
                     i < singularValues.size(); i++)
                {
                    if (singularValues(i) > cutoff)
                    {
                        inverseSingular(i) = 1.0 / singularValues(i);
                        result.rank++;
                        smallestAccepted = singularValues(i);
                    }
                    else
                        inverseSingular(i) = 0;
                }
                result.condition = result.rank == nBasis
                                       ? largest / smallestAccepted
                                       : veryLargeReal;
                result.inverse =
                    svd.matrixV() * inverseSingular.asDiagonal() *
                    svd.matrixU().transpose() * weights.asDiagonal();
                return result;
            }

            const Eigen::MatrixXd normalMatrix =
                weighted.transpose() * weighted;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(
                normalMatrix, Eigen::EigenvaluesOnly);
            if (eigen.info() != Eigen::Success ||
                eigen.eigenvalues().size() != nBasis)
                return result;

            const auto eigenvalues = eigen.eigenvalues();
            const real largestSquared = std::max<real>(
                eigenvalues(nBasis - 1), 0);
            const real cutoffSquared =
                std::max(relativeTolerance * relativeTolerance,
                         std::numeric_limits<real>::epsilon()) *
                largestSquared;
            real smallestAcceptedSquared = veryLargeReal;
            for (Eigen::Index i = 0; i < eigenvalues.size(); i++)
                if (eigenvalues(i) > cutoffSquared)
                {
                    result.rank++;
                    smallestAcceptedSquared = std::min(
                        smallestAcceptedSquared, eigenvalues(i));
                }
            result.condition = result.rank == nBasis
                                   ? std::sqrt(largestSquared /
                                               smallestAcceptedSquared)
                                   : veryLargeReal;
            if (result.rank != nBasis)
                return result;

            Eigen::LDLT<Eigen::MatrixXd> factorization(normalMatrix);
            if (factorization.info() != Eigen::Success)
            {
                result.rank = 0;
                result.condition = veryLargeReal;
                return result;
            }
            result.inverse = factorization.solve(weighted.transpose()) *
                             weights.asDiagonal();
            if (factorization.info() != Eigen::Success ||
                !result.inverse.allFinite())
            {
                result.inverse.resize(0, 0);
                result.rank = 0;
                result.condition = veryLargeReal;
            }
            return result;
        }
    }

    int Reconstruction::QuadraticBasisSize(int dimension)
    {
        DNDS_check_throw_info(dimension == 2 || dimension == 3,
                              "NCFV quadratic basis requires dimension 2 or 3");
        return dimension + dimension * (dimension + 1) / 2;
    }

    Eigen::VectorXd Reconstruction::EvaluateBasis(
        const Vector3 &displacement,
        const Vector3 &referenceLengths,
        int dimension)
    {
        const Vector3 coordinate = displacement.cwiseProduct(
            InverseReferenceLengths(referenceLengths, dimension));
        Eigen::VectorXd basis(QuadraticBasisSize(dimension));
        if (dimension == 2)
        {
            basis << coordinate.x(), coordinate.y(),
                0.5 * sqr(coordinate.x()),
                coordinate.x() * coordinate.y(),
                0.5 * sqr(coordinate.y());
        }
        else
        {
            basis << coordinate.x(), coordinate.y(), coordinate.z(),
                0.5 * sqr(coordinate.x()),
                coordinate.x() * coordinate.y(),
                coordinate.x() * coordinate.z(),
                0.5 * sqr(coordinate.y()),
                coordinate.y() * coordinate.z(),
                0.5 * sqr(coordinate.z());
        }
        return basis;
    }

    Eigen::MatrixXd Reconstruction::EvaluateBasisGradient(
        const Vector3 &displacement,
        const Vector3 &referenceLengths,
        int dimension)
    {
        const Vector3 inverseLengths = InverseReferenceLengths(referenceLengths, dimension);
        const Vector3 coordinate = displacement.cwiseProduct(inverseLengths);
        Eigen::MatrixXd gradient(
            dimension, QuadraticBasisSize(dimension));
        gradient.setZero();
        if (dimension == 2)
        {
            gradient.row(0) << 1.0, 0.0,
                coordinate.x(), coordinate.y(), 0.0;
            gradient.row(1) << 0.0, 1.0,
                0.0, coordinate.x(), coordinate.y();
        }
        else
        {
            gradient.row(0) << 1.0, 0.0, 0.0,
                coordinate.x(), coordinate.y(), coordinate.z(),
                0.0, 0.0, 0.0;
            gradient.row(1) << 0.0, 1.0, 0.0,
                0.0, coordinate.x(), 0.0,
                coordinate.y(), coordinate.z(), 0.0;
            gradient.row(2) << 0.0, 0.0, 1.0,
                0.0, 0.0, coordinate.x(),
                0.0, coordinate.y(), coordinate.z();
        }
        return inverseLengths.head(dimension).asDiagonal() * gradient;
    }

    Eigen::VectorXd Reconstruction::MeanBasis(
        const RawMoments &moments,
        const Vector3 &anchor,
        const Vector3 &referenceLengths,
        int dimension)
    {
        DNDS_check_throw_info(moments.measure > verySmallReal,
                              "NCFV reconstruction received a zero-volume moment set");
        const Vector3 mean = moments.first / moments.measure;
        Matrix3 centredSecond = moments.second / moments.measure;
        centredSecond -= anchor * mean.transpose();
        centredSecond -= mean * anchor.transpose();
        centredSecond += anchor * anchor.transpose();

        Eigen::VectorXd result(QuadraticBasisSize(dimension));
        const Vector3 inverseLengths = InverseReferenceLengths(referenceLengths, dimension);
        const Vector3 first = (mean - anchor).cwiseProduct(inverseLengths);
        centredSecond = (inverseLengths.asDiagonal() * centredSecond *
                         inverseLengths.asDiagonal())
                            .eval();
        if (dimension == 2)
        {
            result << first.x(), first.y(),
                0.5 * centredSecond(0, 0), centredSecond(0, 1),
                0.5 * centredSecond(1, 1);
        }
        else
        {
            result << first.x(), first.y(), first.z(),
                0.5 * centredSecond(0, 0), centredSecond(0, 1), centredSecond(0, 2),
                0.5 * centredSecond(1, 1), centredSecond(1, 2),
                0.5 * centredSecond(2, 2);
        }
        return result;
    }

    void Reconstruction::BuildNodeGraph()
    {
        _nodeGraph.assign(
            static_cast<std::size_t>(_nodeHalo.NumNodeProc()), {});
        _nodeGraphGlobals.assign(
            static_cast<std::size_t>(_mesh->NumNode()), {});
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const auto &rings = _nodeHalo.RingsGlobal(iNode);
            if (rings.empty())
                continue;
            _nodeGraphGlobals[static_cast<std::size_t>(iNode)] = rings.front();
            auto &neighbors = _nodeGraph[static_cast<std::size_t>(iNode)];
            neighbors.reserve(rings.front().size());
            for (index global : rings.front())
                neighbors.push_back(_nodeHalo.GlobalToLocal(global));
        }
    }

    ReconstructionOperator Reconstruction::BuildOperator(index iNode) const
    {
        const int dimension = _mesh->getDim();
        const int nBasis = QuadraticBasisSize(dimension);
        const int targetStencilSize = std::max(
            nBasis,
            static_cast<int>(std::ceil(_settings.stencilSizeFactor * nBasis)));
        const index graphNode = iNode;
        const Vector3 anchor = Vector3::Zero();
        const real lengthScale = LengthScale(iNode);
        const auto momentsAt = [&](index node)
        {
            return _nodeHalo.MomentsRelative(iNode, node);
        };

        ReconstructionOperator result;
        result.node = iNode;
        result.lengthScale = lengthScale;
        result.referenceLengths = ReferenceLengths(iNode);
        result.targetBasisMean = MeanBasis(
            momentsAt(graphNode), anchor, result.referenceLengths, dimension);

        struct Candidate
        {
            index node = UnInitIndex;
            index global = UnInitIndex;
            int ring = 0;
            std::array<long long, 4> geometricKey{};
        };

        std::set<index> visited{graphNode};
        std::vector<Candidate> candidates;
        const auto &rings = _nodeHalo.RingsGlobal(iNode);
        for (int ring = 1;
             ring <= _settings.maximumStencilRings &&
             ring <= static_cast<int>(rings.size());
             ring++)
            for (index global : rings[static_cast<std::size_t>(ring - 1)])
            {
                const index neighbor = _nodeHalo.GlobalToLocal(global);
                if (!visited.insert(neighbor).second)
                    continue;
                const Vector3 scaledDisplacement =
                    _nodeHalo.Displacement(iNode, neighbor) / lengthScale;
                candidates.push_back(
                    {neighbor, global, ring,
                     {std::llround(
                          scaledDisplacement.squaredNorm() * 1e12),
                      std::llround(scaledDisplacement(0) * 1e10),
                      std::llround(scaledDisplacement(1) * 1e10),
                      std::llround(scaledDisplacement(2) * 1e10)}});
            }

        // Prefer direct graph neighbours when completing the compact stencil.
        // Geometry, rather than rank-local numbering, orders equal-ring points.
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate &left, const Candidate &right)
                  {
                      const bool leftDirect = left.ring == 1;
                      const bool rightDirect = right.ring == 1;
                      if (leftDirect != rightDirect)
                          return leftDirect;
                      if (left.geometricKey != right.geometricKey)
                          return left.geometricKey < right.geometricKey;
                      if (left.ring != right.ring)
                          return left.ring < right.ring;
                      return left.global < right.global;
                  });

        DNDS_check_throw_info(
            static_cast<int>(candidates.size()) >= nBasis,
            fmt::format("NCFV node {} has only {} stencil candidates for {} quadratic unknowns",
                        _mesh->NodeIndexLocal2Global(iNode), candidates.size(), nBasis));

        Eigen::MatrixXd candidateMatrix(candidates.size(), nBasis);
        const int nCubic = dimension == 2 ? 4 : 10;
        Eigen::MatrixXd candidateCubic(candidates.size(), nCubic);
        Eigen::VectorXd candidateWeights(candidates.size());
        for (std::size_t row = 0; row < candidates.size(); row++)
        {
            const index neighbor = candidates[row].node;
            candidateMatrix.row(static_cast<Eigen::Index>(row)) =
                (MeanBasis(momentsAt(neighbor), anchor,
                           result.referenceLengths, dimension) -
                 result.targetBasisMean)
                    .transpose();
            candidateCubic.row(static_cast<Eigen::Index>(row)) =
                CubicPointBasis(
                    _nodeHalo.Displacement(iNode, neighbor),
                    result.referenceLengths, dimension)
                    .transpose();
            const real normalizedDistance = std::max(
                _nodeHalo.Displacement(iNode, neighbor).norm() /
                    lengthScale,
                _settings.distanceWeightFloor);
            candidateWeights(static_cast<Eigen::Index>(row)) =
                std::pow(normalizedDistance,
                         -_settings.distanceWeightPower);
        }

        std::vector<index> stencil;
        Eigen::MatrixXd acceptedInverse;
        Eigen::MatrixXd acceptedCubic;
        int acceptedRank = 0;
        real acceptedCondition = veryLargeReal;
        bool accepted = false;
        const int maximumCandidateRing = std::min(
            _settings.maximumStencilRings,
            static_cast<int>(rings.size()));
        for (int allowedRing = 1;
             allowedRing <= maximumCandidateRing && !accepted;
             allowedRing++)
        {
            std::vector<int> pool;
            for (std::size_t candidate = 0;
                 candidate < candidates.size(); candidate++)
                if (candidates[candidate].ring <= allowedRing)
                    pool.push_back(static_cast<int>(candidate));
            if (static_cast<int>(pool.size()) < nBasis)
                continue;
            if (static_cast<int>(pool.size()) < targetStencilSize &&
                allowedRing != maximumCandidateRing)
                continue;

            std::vector<int> selectionOrder;
            selectionOrder.reserve(pool.size());
            std::vector<bool> selected(candidates.size(), false);
            Vector3 displacementBalance = Vector3::Zero();
            Eigen::VectorXd cubicBalance = Eigen::VectorXd::Zero(nCubic);
            const auto select = [&](int candidate)
            {
                if (!selected[static_cast<std::size_t>(candidate)])
                {
                    selected[static_cast<std::size_t>(candidate)] = true;
                    selectionOrder.push_back(candidate);
                    displacementBalance +=
                        _nodeHalo.Displacement(
                            iNode,
                            candidates[static_cast<std::size_t>(candidate)].node)
                            .cwiseQuotient(result.referenceLengths);
                    cubicBalance += candidateCubic.row(candidate).transpose();
                }
            };

            // A first-ring node is topologically indispensable. Retain the
            // complete one-ring before geometrically compacting the stencil.
            for (int candidate : pool)
                if (candidates[static_cast<std::size_t>(candidate)].ring == 1)
                    select(candidate);

            const int initialStencilSize = std::min(
                std::max({nBasis, targetStencilSize,
                          static_cast<int>(selectionOrder.size())}),
                static_cast<int>(pool.size()));

            const auto quantize = [](real value)
            {
                return std::llround(
                    std::clamp(value, 0.0, 1e8) * 1e10);
            };
            const auto weakDirections = [&]() -> Eigen::MatrixXd
            {
                if (selectionOrder.empty())
                    return Eigen::MatrixXd::Identity(nBasis, nBasis).eval();
                Eigen::MatrixXd selectedMatrix(
                    static_cast<Eigen::Index>(selectionOrder.size()), nBasis);
                for (std::size_t row = 0; row < selectionOrder.size(); row++)
                {
                    const int candidate = selectionOrder[row];
                    selectedMatrix.row(static_cast<Eigen::Index>(row)) =
                        candidateWeights(candidate) *
                        candidateMatrix.row(candidate);
                }
                return WeightedWeakDirections(
                    selectedMatrix, _settings.method);
            };
            const auto coverageScore = [&](
                                           int candidate,
                                           const Eigen::MatrixXd &weak)
            {
                const Eigen::RowVectorXd row =
                    candidateWeights(candidate) *
                    candidateMatrix.row(candidate);
                return (row * weak).squaredNorm();
            };
            const auto singletonScore = [&](
                                             int candidate,
                                             const Eigen::MatrixXd &weak)
            {
                const Vector3 displacement =
                    _nodeHalo.Displacement(
                        iNode,
                        candidates[static_cast<std::size_t>(candidate)].node)
                        .cwiseQuotient(result.referenceLengths);
                const Eigen::VectorXd cubic =
                    cubicBalance + candidateCubic.row(candidate).transpose();
                const real coverage = coverageScore(candidate, weak);
                return std::array<long long, 5>{
                    quantize(1.0 / std::max(coverage, 1e-30)),
                    quantize(cubic.squaredNorm()),
                    quantize((displacementBalance + displacement).squaredNorm()),
                    quantize(displacement.squaredNorm()),
                    static_cast<long long>(candidate)};
            };
            const auto bestSingleton = [&]()
            {
                int best = -1;
                std::array<long long, 5> bestScore;
                bestScore.fill(std::numeric_limits<long long>::max());
                const Eigen::MatrixXd weak = weakDirections();
                for (int candidate : pool)
                    if (!selected[static_cast<std::size_t>(candidate)])
                    {
                        const auto score = singletonScore(candidate, weak);
                        if (score < bestScore)
                        {
                            bestScore = score;
                            best = candidate;
                        }
                    }
                return best;
            };

            // Add approximately opposite pairs. Weak spectral directions are
            // reinforced first; among equally useful choices the
            // normalized cubic imbalance, first-moment imbalance, and pair
            // opposition defect decide. This avoids traversal-order truncation
            // and favours cancellation of the leading reconstruction error.
            while (static_cast<int>(selectionOrder.size()) + 1 <
                   initialStencilSize)
            {
                int bestLeft = -1;
                int bestRight = -1;
                std::array<long long, 7> bestScore;
                bestScore.fill(std::numeric_limits<long long>::max());
                const Eigen::MatrixXd weak = weakDirections();
                for (std::size_t i = 0; i < pool.size(); i++)
                {
                    const int left = pool[i];
                    if (selected[static_cast<std::size_t>(left)])
                        continue;
                    const Vector3 leftDisplacement =
                        _nodeHalo.Displacement(
                            iNode, candidates[static_cast<std::size_t>(left)].node)
                            .cwiseQuotient(result.referenceLengths);
                    for (std::size_t j = i + 1; j < pool.size(); j++)
                    {
                        const int right = pool[j];
                        if (selected[static_cast<std::size_t>(right)])
                            continue;
                        const Vector3 rightDisplacement =
                            _nodeHalo.Displacement(
                                iNode, candidates[static_cast<std::size_t>(right)].node)
                                .cwiseQuotient(result.referenceLengths);
                        const Vector3 pairDisplacement =
                            leftDisplacement + rightDisplacement;
                        const Eigen::VectorXd cubic =
                            cubicBalance + candidateCubic.row(left).transpose() +
                            candidateCubic.row(right).transpose();
                        const real coverage =
                            coverageScore(left, weak) +
                            coverageScore(right, weak);
                        const auto score = std::array<long long, 7>{
                            quantize(1.0 / std::max(coverage, 1e-30)),
                            quantize(cubic.squaredNorm()),
                            quantize((displacementBalance + pairDisplacement).squaredNorm()),
                            quantize(pairDisplacement.squaredNorm()),
                            quantize(std::max(leftDisplacement.squaredNorm(),
                                              rightDisplacement.squaredNorm())),
                            static_cast<long long>(left),
                            static_cast<long long>(right)};
                        if (score < bestScore)
                        {
                            bestScore = score;
                            bestLeft = left;
                            bestRight = right;
                        }
                    }
                }
                if (bestLeft < 0)
                    break;
                select(bestLeft);
                select(bestRight);
            }
            while (static_cast<int>(selectionOrder.size()) < initialStencilSize)
            {
                const int candidate = bestSingleton();
                DNDS_check_throw_info(candidate >= 0,
                                      "NCFV compact stencil selection exhausted its pool");
                select(candidate);
            }

            // Extra points are deterministic rank/condition fallbacks. They are appended
            // by the same cubic-balance criterion rather than traversal order.
            while (selectionOrder.size() < pool.size())
            {
                const int candidate = bestSingleton();
                DNDS_assert(candidate >= 0);
                select(candidate);
            }

            for (int count = initialStencilSize;
                 count <= static_cast<int>(selectionOrder.size()); count++)
            {
                stencil.clear();
                stencil.reserve(static_cast<std::size_t>(count));
                result.rings = 0;
                for (int i = 0; i < count; i++)
                {
                    const Candidate &candidate =
                        candidates[static_cast<std::size_t>(
                            selectionOrder[static_cast<std::size_t>(i)])];
                    stencil.push_back(candidate.node);
                    result.rings = std::max(
                        result.rings, candidate.ring);
                }

                Eigen::MatrixXd matrix(count, nBasis);
                Eigen::VectorXd weights(count);
                for (int row = 0; row < count; row++)
                {
                    const int candidate =
                        selectionOrder[static_cast<std::size_t>(row)];
                    matrix.row(row) = candidateMatrix.row(candidate);
                    weights(row) = candidateWeights(candidate);
                }

                const Eigen::MatrixXd weighted =
                    weights.asDiagonal() * matrix;
                const WeightedLeastSquaresOperator leastSquares =
                    BuildWeightedLeastSquaresOperator(
                        weighted, weights, _settings.method,
                        _settings.svdTolerance);
                acceptedRank = leastSquares.rank;
                acceptedCondition = leastSquares.condition;
                acceptedInverse = leastSquares.inverse;
                acceptedCubic.resize(count, nCubic);
                for (int row = 0; row < count; row++)
                    acceptedCubic.row(row) = candidateCubic.row(
                        selectionOrder[static_cast<std::size_t>(row)]);
                accepted = acceptedRank == nBasis &&
                           acceptedCondition <=
                               _settings.maximumConditionNumber;
                if (accepted)
                    break;
            }
        }

        DNDS_check_throw_info(
            acceptedRank == nBasis,
            fmt::format("NCFV node {} quadratic stencil rank is {}/{}; increase maximumStencilRings",
                        _mesh->NodeIndexLocal2Global(iNode), acceptedRank, nBasis));
        DNDS_check_throw_info(
            acceptedCondition <= _settings.maximumConditionNumber,
            fmt::format("NCFV node {} quadratic stencil condition number {} exceeds {}",
                        _mesh->NodeIndexLocal2Global(iNode), acceptedCondition,
                        _settings.maximumConditionNumber));

        result.numericalRank = acceptedRank;
        result.conditionNumber = acceptedCondition;
        result.cubicErrorIndicator =
            (acceptedInverse.topRows(dimension) * acceptedCubic).norm();
        result.stencil = std::move(stencil);
        result.directNeighborCount = static_cast<int>(std::count_if(
            result.stencil.begin(), result.stencil.end(),
            [&](index node)
            {
                const auto &direct =
                    _nodeGraph[static_cast<std::size_t>(iNode)];
                return std::find(direct.begin(), direct.end(), node) !=
                       direct.end();
            }));
        DNDS_check_throw_info(
            result.directNeighborCount ==
                static_cast<int>(
                    _nodeGraph[static_cast<std::size_t>(iNode)].size()),
            "NCFV compact stencil failed to retain every first-ring neighbor");
        result.stencilGlobals.reserve(result.stencil.size());
        for (index local : result.stencil)
            result.stencilGlobals.push_back(
                _nodeHalo.LocalToGlobal(local));
        result.inverseRows = _mode == IntegrationMode::EfficientDifferential
                                 ? acceptedInverse.topRows(dimension)
                                 : acceptedInverse;
        return result;
    }

    void Reconstruction::Build()
    {
        DNDS_check_throw_info(
            _settings.method == ReconstructionMethod::LeastSquares ||
                _settings.method == ReconstructionMethod::SVDLeastSquares,
            "NCFV reconstruction selected an unknown least-squares method");
        BuildNodeGraph();
        _operators.clear();
        _operators.resize(static_cast<std::size_t>(_mesh->NumNode()));
        real localMaximumCondition = 0;
        real localMaximumCubicError = 0;
        int localMaximumRings = 0;
        int localMinimumDirectNeighbors = std::numeric_limits<int>::max();
        int localMaximumDirectNeighbors = 0;
        int localMinimumStencil = std::numeric_limits<int>::max();
        int localMaximumStencil = 0;
        index localStencilSum = 0;
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            _operators[static_cast<std::size_t>(iNode)] = BuildOperator(iNode);
            localMaximumCondition = std::max(
                localMaximumCondition,
                _operators[static_cast<std::size_t>(iNode)].conditionNumber);
            localMaximumCubicError = std::max(
                localMaximumCubicError,
                _operators[static_cast<std::size_t>(iNode)].cubicErrorIndicator);
            localMaximumRings = std::max(
                localMaximumRings,
                _operators[static_cast<std::size_t>(iNode)].rings);
            const int stencilSize = static_cast<int>(
                _operators[static_cast<std::size_t>(iNode)].stencil.size());
            localMinimumStencil = std::min(localMinimumStencil, stencilSize);
            localMaximumStencil = std::max(localMaximumStencil, stencilSize);
            localStencilSum += stencilSize;
            const int directNeighbors =
                _operators[static_cast<std::size_t>(iNode)].directNeighborCount;
            localMinimumDirectNeighbors = std::min(
                localMinimumDirectNeighbors, directNeighbors);
            localMaximumDirectNeighbors = std::max(
                localMaximumDirectNeighbors, directNeighbors);
        }
        real globalMaximumCondition = 0;
        real globalMaximumCubicError = 0;
        int globalMaximumRings = 0;
        int globalMinimumDirectNeighbors = 0;
        int globalMaximumDirectNeighbors = 0;
        int globalMinimumStencil = 0;
        int globalMaximumStencil = 0;
        index globalStencilSum = 0;
        MPI_Allreduce(&localMaximumCondition, &globalMaximumCondition, 1,
                      DNDS_MPI_REAL, MPI_MAX, _mpi.comm);
        MPI_Allreduce(&localMaximumCubicError, &globalMaximumCubicError, 1,
                      DNDS_MPI_REAL, MPI_MAX, _mpi.comm);
        MPI_Allreduce(&localMaximumRings, &globalMaximumRings, 1,
                      MPI_INT, MPI_MAX, _mpi.comm);
        MPI_Allreduce(&localMinimumDirectNeighbors,
                      &globalMinimumDirectNeighbors, 1,
                      MPI_INT, MPI_MIN, _mpi.comm);
        MPI_Allreduce(&localMaximumDirectNeighbors,
                      &globalMaximumDirectNeighbors, 1,
                      MPI_INT, MPI_MAX, _mpi.comm);
        MPI_Allreduce(&localMinimumStencil, &globalMinimumStencil, 1,
                      MPI_INT, MPI_MIN, _mpi.comm);
        MPI_Allreduce(&localMaximumStencil, &globalMaximumStencil, 1,
                      MPI_INT, MPI_MAX, _mpi.comm);
        MPI_Allreduce(&localStencilSum, &globalStencilSum, 1,
                      DNDS_MPI_INDEX, MPI_SUM, _mpi.comm);
        if (_mpi.rank == 0)
            log() << "NCFV quadratic reconstruction: stored rows="
                  << (_mode == IntegrationMode::EfficientDifferential
                          ? _mesh->getDim()
                          : BasisSize())
                  << ", method="
                  << (_settings.method == ReconstructionMethod::LeastSquares
                          ? "LeastSquares"
                          : "SVDLeastSquares")
                  << ", max rings=" << globalMaximumRings
                  << ", max condition=" << globalMaximumCondition
                  << ", max cubic-gradient indicator="
                  << globalMaximumCubicError
                  << ", retained first-ring[min,max]=["
                  << globalMinimumDirectNeighbors << ","
                  << globalMaximumDirectNeighbors << "]"
                  << ", stencil[min,mean,max]=["
                  << globalMinimumStencil << ","
                  << static_cast<real>(globalStencilSum) /
                         _mesh->NumNodeGlobal()
                  << "," << globalMaximumStencil << "]"
                  << ", normalization=dual-bounds-half-span (thesis 3-34)" << std::endl;
    }

    std::vector<index> Reconstruction::CollectNodeDependencies() const
    {
        std::vector<index> globals;
        for (const auto &op : _operators)
            globals.insert(
                globals.end(), op.stencilGlobals.begin(),
                op.stencilGlobals.end());
        if (_settings.enableLimiter)
            for (const auto &neighbors : _nodeGraphGlobals)
                globals.insert(globals.end(), neighbors.begin(), neighbors.end());
        std::sort(globals.begin(), globals.end());
        globals.erase(std::unique(globals.begin(), globals.end()), globals.end());
        return globals;
    }

    void Reconstruction::RemapNodeIndices()
    {
        DNDS_check_throw_info(
            _nodeHalo.IsFinalized(),
            "NCFV reconstruction requires a finalized exact node halo before remapping");
        for (auto &op : _operators)
        {
            DNDS_check_throw_info(
                op.stencilGlobals.size() == op.stencil.size(),
                "NCFV reconstruction lost stable stencil IDs before halo remapping");
            for (std::size_t i = 0; i < op.stencil.size(); i++)
                op.stencil[i] =
                    _nodeHalo.GlobalToLocal(op.stencilGlobals[i]);
            op.stencilGlobals.clear();
            op.stencilGlobals.shrink_to_fit();
        }
        if (_settings.enableLimiter)
        {
            _nodeGraph.assign(
                static_cast<std::size_t>(_nodeHalo.NumNodeProc()), {});
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            {
                auto &neighbors = _nodeGraph[static_cast<std::size_t>(iNode)];
                const auto &globals =
                    _nodeGraphGlobals[static_cast<std::size_t>(iNode)];
                neighbors.reserve(globals.size());
                for (index global : globals)
                    neighbors.push_back(_nodeHalo.GlobalToLocal(global));
            }
        }
        else
        {
            _nodeGraph.clear();
            _nodeGraph.shrink_to_fit();
        }
        _nodeGraphGlobals.clear();
        _nodeGraphGlobals.shrink_to_fit();
    }

    void Reconstruction::ComputeCoefficients(
        const NodeStatePair &means,
        NodeMatrixPair &gradients,
        NodeMatrixPair &coefficients) const
    {
        const int nVars = means.father->MatRowSize();
        const int dimension = _mesh->getDim();
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const ReconstructionOperator &op = _operators[static_cast<std::size_t>(iNode)];
            Eigen::MatrixXd differences(op.stencil.size(), nVars);
            for (std::size_t row = 0; row < op.stencil.size(); row++)
            {
                differences.row(static_cast<Eigen::Index>(row)) =
                    (means[op.stencil[row]] - means[iNode]).transpose();
            }
            const Eigen::MatrixXd reconstructed = op.inverseRows * differences;
            if (_mode == IntegrationMode::EfficientDifferential)
                gradients[iNode] = op.referenceLengths.head(dimension).cwiseInverse().asDiagonal() *
                                   reconstructed.topRows(dimension);
            else
                coefficients[iNode] = reconstructed;
        }
    }

    void Reconstruction::RecoverPointValues(
        const NodeStatePair &means,
        const NodeMatrixPair &gradients,
        const NodeMatrixPair &coefficients,
        NodeStatePair &pointValues) const
    {
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            pointValues[iNode] = means[iNode];
            if (_mode == IntegrationMode::EfficientDifferential)
            {
                for (const auto &entry : _geometry.NodeVolume(iNode).pointRecoveryStencil)
                    pointValues[iNode] -= gradients[entry.node].transpose() *
                                          entry.gradientWeight.head(_mesh->getDim());
            }
            else
            {
                const ReconstructionOperator &op = _operators[static_cast<std::size_t>(iNode)];
                pointValues[iNode] -= coefficients[iNode].transpose() * op.targetBasisMean;
            }
        }
    }

    std::vector<real> Reconstruction::ComputeLimiterFactors(
        const NodeStatePair &means,
        const NodeStatePair &pointValues,
        const NodeMatrixPair &gradients,
        const NodeMatrixPair &coefficients) const
    {
        const int nVars = means.father->MatRowSize();
        std::vector<real> factors(static_cast<std::size_t>(_mesh->NumNode()), 1.0);

        if (_mode == IntegrationMode::EfficientDifferential)
        {
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            {
                real factor = 1.0;
                for (const NodeEdgeIncidence &incidence : _topology.Node2Edge(iNode))
                {
                    const EdgeControlSurface &surface =
                        _geometry.EdgeSurface(incidence.edge);
                    const bool isLeft = surface.nodes[0] == iNode;
                    DNDS_check_throw_info(
                        isLeft || surface.nodes[1] == iNode,
                        "NCFV limiter encountered an edge not incident to its node");
                    const index neighbor = surface.nodes[isLeft ? 1 : 0];
                    const std::size_t side = isLeft ? 0 : 1;

                    Eigen::VectorXd candidate =
                        surface.measure * pointValues[iNode];
                    for (const EfficientSurfaceNode &entry : surface.efficientStencil)
                        candidate += gradients[entry.node].transpose() *
                                     entry.stateGradientWeights[side].head(_mesh->getDim());
                    candidate /= surface.measure;

                    const Eigen::VectorXd minimum =
                        pointValues[iNode].cwiseMin(pointValues[neighbor]);
                    const Eigen::VectorXd maximum =
                        pointValues[iNode].cwiseMax(pointValues[neighbor]);
                    for (int iVar = 0; iVar < nVars; iVar++)
                    {
                        const real increment =
                            candidate(iVar) - pointValues[iNode](iVar);
                        if (increment > verySmallReal)
                            factor = std::min(
                                factor,
                                (maximum(iVar) - pointValues[iNode](iVar)) /
                                    increment);
                        else if (increment < -verySmallReal)
                            factor = std::min(
                                factor,
                                (minimum(iVar) - pointValues[iNode](iVar)) /
                                    increment);
                    }
                }
                factors[static_cast<std::size_t>(iNode)] =
                    std::clamp(factor, 0.0, 1.0);
            }
            return factors;
        }

        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            Eigen::VectorXd minimum = means[iNode];
            Eigen::VectorXd maximum = means[iNode];
            for (index neighbor : _nodeGraph[static_cast<std::size_t>(iNode)])
            {
                minimum = minimum.cwiseMin(means[neighbor]);
                maximum = maximum.cwiseMax(means[neighbor]);
            }

            real factor = 1.0;
            for (index neighbor : _nodeGraph[static_cast<std::size_t>(iNode)])
            {
                const Vector3 displacement =
                    0.5 * _nodeHalo.Displacement(iNode, neighbor);
                Eigen::VectorXd candidate = pointValues[iNode];
                const auto &op = _operators[static_cast<std::size_t>(iNode)];
                candidate += coefficients[iNode].transpose() *
                             EvaluateBasis(displacement,
                                           op.referenceLengths, _mesh->getDim());

                for (int iVar = 0; iVar < nVars; iVar++)
                {
                    const real increment = candidate(iVar) - means[iNode](iVar);
                    if (increment > verySmallReal)
                        factor = std::min(factor,
                                          (maximum(iVar) - means[iNode](iVar)) / increment);
                    else if (increment < -verySmallReal)
                        factor = std::min(factor,
                                          (minimum(iVar) - means[iNode](iVar)) / increment);
                }
            }
            factors[static_cast<std::size_t>(iNode)] = std::clamp(factor, 0.0, 1.0);
        }
        return factors;
    }

    void Reconstruction::ApplyLimiter(
        const std::vector<real> &factors,
        NodeMatrixPair &gradients,
        NodeMatrixPair &coefficients) const
    {
        DNDS_check_throw_info(factors.size() == static_cast<std::size_t>(_mesh->NumNode()),
                              "NCFV limiter-factor array has the wrong size");
        DNDS_check_throw_info(
            _mode == IntegrationMode::TraditionalQuadrature,
            "Efficient NCFV limiter factors are side-anchored and cannot be applied to the shared gradient field");
        static_cast<void>(gradients);
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
            coefficients[iNode] *= factors[static_cast<std::size_t>(iNode)];
    }
}
