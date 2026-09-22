/**
 * @file NCFVReconstruction.hpp
 * @brief Third-order node reconstruction shared by both integration modes.
 */
#pragma once

#include "NCFVDualGeometry.hpp"
#include "NCFVNodeHalo.hpp"

#include "DNDS/ArrayDOF.hpp"

#include <vector>

namespace DNDS::NCFV
{
    using NodeStatePair = ArrayDof<DynamicSize, 1>;
    using NodeMatrixPair = ArrayDof<DynamicSize, DynamicSize>;

    struct ReconstructionOperator
    {
        index node = UnInitIndex;
        int rings = 0;
        int numericalRank = 0;
        real lengthScale = 0;                       // Volume-based scale for distance weights only.
        Vector3 referenceLengths = Vector3::Ones(); // Thesis (3-34), basis normalization.
        real conditionNumber = veryLargeReal;
        /** Dimensionless linear-mode response to normalized cubic point data. */
        real cubicErrorIndicator = veryLargeReal;
        /** Number of first-ring graph neighbours retained in stencil. */
        int directNeighborCount = 0;
        std::vector<index> stencil;
        /** Stable global IDs retained while the final exact halo is pruned. */
        std::vector<index> stencilGlobals;
        Eigen::MatrixXd inverseRows;
        Eigen::VectorXd targetBasisMean;
    };

    /**
     * @brief Weighted quadratic least-squares operators on dual-volume means.
     *
     * EfficientDifferential solves the same complete quadratic problem as the
     * traditional method, then stores only the first `dimension` rows.  Thus its
     * gradient is not a lower-order linear fit.
     */
    class Reconstruction
    {
        const MPIInfo &_mpi;
        ssp<Geom::UnstructuredMesh> _mesh;
        const Topology &_topology;
        const DualGeometry &_geometry;
        IntegrationMode _mode;
        ReconstructionSettings _settings;
        const NodeHalo &_nodeHalo;

        std::vector<std::vector<index>> _nodeGraph;
        std::vector<std::vector<index>> _nodeGraphGlobals;
        std::vector<ReconstructionOperator> _operators;

        void BuildNodeGraph();
        ReconstructionOperator BuildOperator(index iNode) const;

    public:
        Reconstruction(
            const MPIInfo &mpi,
            const ssp<Geom::UnstructuredMesh> &mesh,
            const Topology &topology,
            const DualGeometry &geometry,
            IntegrationMode mode,
            const ReconstructionSettings &settings,
            const NodeHalo &nodeHalo)
            : _mpi(mpi), _mesh(mesh), _topology(topology), _geometry(geometry),
              _mode(mode), _settings(settings), _nodeHalo(nodeHalo)
        {
        }

        void Build();
        [[nodiscard]] std::vector<index> CollectNodeDependencies() const;
        void RemapNodeIndices();
        /** @brief Volume scale for distance weights; not the polynomial reference lengths. */
        [[nodiscard]] real LengthScale(index local) const
        {
            return std::pow(_nodeHalo.Volume(local),
                            1.0 / _mesh->getDim());
        }
        [[nodiscard]] Vector3 ReferenceLengths(index local) const
        {
            return _nodeHalo.ReferenceLengths(local);
        }

        [[nodiscard]] int BasisSize() const { return QuadraticBasisSize(_mesh->getDim()); }
        [[nodiscard]] const ReconstructionOperator &Operator(index iNode) const
        {
            return _operators.at(static_cast<std::size_t>(iNode));
        }
        [[nodiscard]] const std::vector<index> &DirectNeighbors(index iNode) const
        {
            return _nodeGraph.at(static_cast<std::size_t>(iNode));
        }

        static int QuadraticBasisSize(int dimension);
        static Eigen::VectorXd EvaluateBasis(
            const Vector3 &displacement,
            const Vector3 &referenceLengths,
            int dimension);
        /** @brief Physical derivatives of the dimensionless quadratic basis. */
        static Eigen::MatrixXd EvaluateBasisGradient(
            const Vector3 &displacement,
            const Vector3 &referenceLengths,
            int dimension);
        static Eigen::VectorXd MeanBasis(
            const RawMoments &moments,
            const Vector3 &anchor,
            const Vector3 &referenceLengths,
            int dimension);

        /** @brief Compute gradients or complete quadratic coefficient fields. */
        void ComputeCoefficients(
            const NodeStatePair &means,
            NodeMatrixPair &gradients,
            NodeMatrixPair &coefficients) const;

        /** @brief Recover third-order primal-node point values from dual means. */
        void RecoverPointValues(
            const NodeStatePair &means,
            const NodeMatrixPair &gradients,
            const NodeMatrixPair &coefficients,
            NodeStatePair &pointValues) const;

        /**
         * @brief One scalar limiter coefficient per owned node.
         *
         * Efficient mode follows thesis (3-94)--(3-100): the constrained
         * quantity is the exact differential-weight macro-surface mean, and
         * its admissible interval is formed by the two endpoint point values.
         */
        std::vector<real> ComputeLimiterFactors(
            const NodeStatePair &means,
            const NodeStatePair &pointValues,
            const NodeMatrixPair &gradients,
            const NodeMatrixPair &coefficients) const;

        void ApplyLimiter(
            const std::vector<real> &factors,
            NodeMatrixPair &gradients,
            NodeMatrixPair &coefficients) const;
    };
}
