#include "NCFVReconstruction.hpp"

#include "DNDS/Errors.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
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

    RawMoments Reconstruction::GetMoments(index iNode) const
    {
        const auto &stored = _geometry.NodeMoments();
        RawMoments moments;
        moments.measure = stored(iNode, 0);
        moments.first << stored(iNode, 1), stored(iNode, 2), stored(iNode, 3);
        moments.second.setZero();
        moments.second(0, 0) = stored(iNode, 4);
        moments.second(1, 1) = stored(iNode, 5);
        moments.second(2, 2) = stored(iNode, 6);
        moments.second(0, 1) = moments.second(1, 0) = stored(iNode, 7);
        moments.second(1, 2) = moments.second(2, 1) = stored(iNode, 8);
        moments.second(2, 0) = moments.second(0, 2) = stored(iNode, 9);
        return moments;
    }

    void Reconstruction::BuildNodeGraph()
    {
        if (_periodic)
        {
            _nodeGraph = _periodic->Graph();
            return;
        }
        const index nNodes = _mesh->NumNodeProc();
        std::vector<std::set<index>> graphSets(static_cast<std::size_t>(nNodes));
        for (index iCell = 0; iCell < _mesh->NumCellProc(); iCell++)
        {
            auto cell = _mesh->GetCellElement(iCell);
            const int nEdges = _mesh->getDim() == 2
                                   ? cell.GetNumFaces()
                                   : cell.GetNumEdges();
            for (int iEdge = 0; iEdge < nEdges; iEdge++)
            {
                const auto edge = _mesh->getDim() == 2
                                      ? cell.ObtainFace(iEdge)
                                      : cell.ObtainEdge(iEdge);
                std::vector<index> edgeNodes(static_cast<std::size_t>(edge.GetNumNodes()));
                if (_mesh->getDim() == 2)
                    cell.ExtractFaceNodes(iEdge, _mesh->cell2node[iCell], edgeNodes);
                else
                    cell.ExtractEdgeNodes(iEdge, _mesh->cell2node[iCell], edgeNodes);
                DNDS_check_throw_info(edgeNodes.size() >= 2,
                                      "NCFV graph encountered an invalid primal edge");
                graphSets[static_cast<std::size_t>(edgeNodes[0])].insert(edgeNodes[1]);
                graphSets[static_cast<std::size_t>(edgeNodes[1])].insert(edgeNodes[0]);
            }
        }

        _nodeGraph.resize(static_cast<std::size_t>(nNodes));
        for (index iNode = 0; iNode < nNodes; iNode++)
        {
            auto &neighbors = _nodeGraph[static_cast<std::size_t>(iNode)];
            neighbors.assign(graphSets[static_cast<std::size_t>(iNode)].begin(),
                             graphSets[static_cast<std::size_t>(iNode)].end());
            std::sort(neighbors.begin(), neighbors.end(), [this](index left, index right)
                      { return _mesh->NodeIndexLocal2Global(left) <
                               _mesh->NodeIndexLocal2Global(right); });
        }
    }

    ReconstructionOperator Reconstruction::BuildOperator(index iNode) const
    {
        const int dimension = _mesh->getDim();
        const int nBasis = QuadraticBasisSize(dimension);
        const int targetStencilSize = std::max(
            nBasis,
            static_cast<int>(std::ceil(_settings.stencilSizeFactor * nBasis)));
        const index graphNode = _periodic ? _periodic->Representative(iNode) : iNode;
        const Vector3 anchor = _periodic ? Vector3::Zero().eval() : Vector3(_mesh->coords[iNode]);
        const real lengthScale = LengthScale(iNode);
        const auto momentsAt = [&](index node)
        {
            return _periodic ? _periodic->Moments(node, _periodic->Displacement(graphNode, node)) : GetMoments(node);
        };

        ReconstructionOperator result;
        result.node = iNode;
        result.lengthScale = lengthScale;
        result.referenceLengths = ReferenceLengths(iNode);
        result.targetBasisMean = MeanBasis(
            momentsAt(graphNode), anchor, result.referenceLengths, dimension);

        std::set<index> visited{graphNode};
        std::vector<index> frontier{graphNode};
        std::vector<index> stencil;
        Eigen::MatrixXd acceptedInverse;
        int acceptedRank = 0;
        real acceptedCondition = veryLargeReal;

        for (int ring = 1; ring <= _settings.maximumStencilRings; ring++)
        {
            std::vector<index> next;
            for (index current : frontier)
                for (index neighbor : _nodeGraph[static_cast<std::size_t>(current)])
                    if (visited.insert(neighbor).second)
                        next.push_back(neighbor);
            std::sort(next.begin(), next.end(), [this](index left, index right)
                      { return _periodic ? left < right : _mesh->NodeIndexLocal2Global(left) < _mesh->NodeIndexLocal2Global(right); });
            stencil.insert(stencil.end(), next.begin(), next.end());
            frontier = std::move(next);
            const bool graphExhausted = frontier.empty();
            const bool lastAllowedRing = ring == _settings.maximumStencilRings;
            if (static_cast<int>(stencil.size()) < nBasis)
            {
                if (graphExhausted)
                    break;
                continue;
            }
            if (static_cast<int>(stencil.size()) < targetStencilSize &&
                !graphExhausted && !lastAllowedRing)
                continue;

            Eigen::MatrixXd matrix(stencil.size(), nBasis);
            Eigen::VectorXd weights(stencil.size());
            for (std::size_t row = 0; row < stencil.size(); row++)
            {
                const index neighbor = stencil[row];
                matrix.row(static_cast<Eigen::Index>(row)) =
                    (MeanBasis(momentsAt(neighbor), anchor, result.referenceLengths, dimension) -
                     result.targetBasisMean)
                        .transpose();
                const real normalizedDistance = std::max(
                    (_periodic ? _periodic->Displacement(graphNode, neighbor).norm() : (_mesh->coords[neighbor] - anchor).norm()) / lengthScale,
                    _settings.distanceWeightFloor);
                weights(static_cast<Eigen::Index>(row)) =
                    std::pow(normalizedDistance, -_settings.distanceWeightPower);
            }

            const Eigen::MatrixXd weighted = weights.asDiagonal() * matrix;
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                weighted, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const auto singularValues = svd.singularValues();
            const real largest = singularValues.size() ? singularValues(0) : 0;
            const real cutoff = _settings.svdTolerance * largest;
            Eigen::VectorXd inverseSingular = singularValues;
            acceptedRank = 0;
            real smallestAccepted = veryLargeReal;
            for (Eigen::Index i = 0; i < singularValues.size(); i++)
            {
                if (singularValues(i) > cutoff)
                {
                    inverseSingular(i) = 1.0 / singularValues(i);
                    acceptedRank++;
                    smallestAccepted = singularValues(i);
                }
                else
                    inverseSingular(i) = 0;
            }
            acceptedCondition = acceptedRank == nBasis
                                    ? largest / smallestAccepted
                                    : veryLargeReal;
            acceptedInverse = svd.matrixV() * inverseSingular.asDiagonal() *
                              svd.matrixU().transpose() * weights.asDiagonal();
            result.rings = ring;
            if (acceptedRank == nBasis &&
                acceptedCondition <= _settings.maximumConditionNumber)
                break;
            if (graphExhausted)
                break;
        }

        DNDS_check_throw_info(
            static_cast<int>(stencil.size()) >= nBasis,
            fmt::format("NCFV node {} has only {} stencil nodes for {} quadratic unknowns",
                        _mesh->NodeIndexLocal2Global(iNode), stencil.size(), nBasis));
        DNDS_check_throw_info(
            acceptedRank == nBasis,
            fmt::format("NCFV node {} quadratic stencil rank is {}/{}; increase ghostLayers or stencil rings",
                        _mesh->NodeIndexLocal2Global(iNode), acceptedRank, nBasis));
        DNDS_check_throw_info(
            acceptedCondition <= _settings.maximumConditionNumber,
            fmt::format("NCFV node {} quadratic stencil condition number {} exceeds {}",
                        _mesh->NodeIndexLocal2Global(iNode), acceptedCondition,
                        _settings.maximumConditionNumber));

        result.numericalRank = acceptedRank;
        result.conditionNumber = acceptedCondition;
        result.stencil = std::move(stencil);
        result.inverseRows = _mode == IntegrationMode::EfficientDifferential
                                 ? acceptedInverse.topRows(dimension)
                                 : acceptedInverse;
        return result;
    }

    void Reconstruction::Build()
    {
        BuildNodeGraph();
        _operators.clear();
        _operators.resize(static_cast<std::size_t>(_mesh->NumNode()));
        real localMaximumCondition = 0;
        int localMaximumRings = 0;
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            _operators[static_cast<std::size_t>(iNode)] = BuildOperator(iNode);
            localMaximumCondition = std::max(
                localMaximumCondition,
                _operators[static_cast<std::size_t>(iNode)].conditionNumber);
            localMaximumRings = std::max(
                localMaximumRings,
                _operators[static_cast<std::size_t>(iNode)].rings);
        }
        real globalMaximumCondition = 0;
        int globalMaximumRings = 0;
        MPI_Allreduce(&localMaximumCondition, &globalMaximumCondition, 1,
                      DNDS_MPI_REAL, MPI_MAX, _mpi.comm);
        MPI_Allreduce(&localMaximumRings, &globalMaximumRings, 1,
                      MPI_INT, MPI_MAX, _mpi.comm);
        if (_mpi.rank == 0)
            log() << "NCFV quadratic reconstruction: stored rows="
                  << (_mode == IntegrationMode::EfficientDifferential
                          ? _mesh->getDim()
                          : BasisSize())
                  << ", max rings=" << globalMaximumRings
                  << ", max condition=" << globalMaximumCondition
                  << ", normalization=dual-bounds-half-span (thesis 3-34)" << std::endl;
    }

    void Reconstruction::ComputeCoefficients(
        const NodeStatePair &means,
        NodeMatrixPair &gradients,
        NodeMatrixPair &coefficients) const
    {
        const int nVars = means.father->MatRowSize();
        const int dimension = _mesh->getDim();
        const Eigen::MatrixXd globalMeans = _periodic ? _periodic->Gather(means) : Eigen::MatrixXd{};
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const ReconstructionOperator &op = _operators[static_cast<std::size_t>(iNode)];
            Eigen::MatrixXd differences(op.stencil.size(), nVars);
            for (std::size_t row = 0; row < op.stencil.size(); row++)
            {
                if (_periodic)
                    differences.row(static_cast<Eigen::Index>(row)) =
                        (globalMeans.col(op.stencil[row]) - means[iNode]).transpose();
                else
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
                for (const auto &weight : _geometry.NodeVolume(iNode).pointRecoveryWeights)
                    pointValues[iNode] -= gradients[weight.node].transpose() *
                                          weight.value.head(_mesh->getDim());
            }
            else
            {
                const ReconstructionOperator &op = _operators[static_cast<std::size_t>(iNode)];
                pointValues[iNode] -= coefficients[iNode].transpose() * op.targetBasisMean;
            }
        }
        if (_periodic && _mode == IntegrationMode::EfficientDifferential)
            _periodic->Average(pointValues);
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
                    const auto &weights = isLeft
                                              ? surface.leftStateWeights
                                              : surface.rightStateWeights;

                    Eigen::VectorXd candidate = pointValues[iNode];
                    for (const SparseVectorWeight &weight : weights)
                        candidate += gradients[weight.node].transpose() *
                                     weight.value.head(_mesh->getDim()) /
                                     surface.measure;

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
                const Vector3 point = 0.5 * (_mesh->coords[iNode] + _mesh->coords[neighbor]);
                Eigen::VectorXd candidate = pointValues[iNode];
                const auto &op = _operators[static_cast<std::size_t>(iNode)];
                candidate += coefficients[iNode].transpose() *
                             EvaluateBasis(point - _mesh->coords[iNode],
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
