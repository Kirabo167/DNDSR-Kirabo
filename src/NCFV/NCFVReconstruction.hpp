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
     * EfficientDifferential with least squares solves the complete quadratic
     * problem, then stores only the first `dimension` rows. Variational mode
     * retains the full quadratic coefficients for neighbour sweeps, then
     * supplies their first derivatives to efficient flux integration.
     */
    class Reconstruction
    {
        struct VariationalNeighborOperator
        {
            index node = UnInitIndex;
            Eigen::MatrixXd coupling;
            Eigen::VectorXd meanJump;
        };
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
        std::vector<std::vector<VariationalNeighborOperator>> _variationalOperators;

        /**
         * @brief Copy complete first-ring graph adjacency into local/global lists for owned-node stencils and
         *     bounds.
         */
        void BuildNodeGraph();
        /**
         * @brief Select a sufficiently accurate stencil and form its weighted reconstruction inverse.
         * @param iNode Owned anchor-node index in the preliminary halo.
         * @return Operator with stencil IDs, polynomial normalization, inverse rows, rank/condition metrics, and
         *     cubic-error indicator.
         */
        ReconstructionOperator BuildOperator(index iNode) const;
        /**
         * @brief Assemble local self/neighbor coupling blocks for iterative quadratic variational reconstruction.
         */
        void BuildVariationalOperators();

    public:
        /**
         * @brief Bind node-centered quadratic reconstruction to geometry, topology, and its communication layout.
         * @param mpi MPI communicator metadata; collective operations must be entered by every participating rank.
         * @param mesh Distributed O1 primal mesh with owned/ghost connectivity prepared for NCFV.
         * @param topology Built primal-edge/face topology.
         * @param geometry Built dual-volume moments and integration data.
         * @param mode Traditional quadrature or efficient differential integration; determines retained operator
         *     rows.
         * @param settings Stencil selection, least-squares/variational solve, and limiter settings.
         * @param nodeHalo Preliminary node geometry/rings for Build(), later finalized for runtime evaluation.
         * @note Referenced topology, geometry, halo, and MPI metadata must outlive this object.
         */
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

        /**
         * @brief Build owned-node reconstruction stencils and report global stencil quality diagnostics.
         * @pre NodeHalo::BuildPreliminary() has completed; call before pruning the runtime halo.
         * @note Collective because stencil quality diagnostics use MPI reductions.
         */
        void Build();
        /**
         * @brief Collect the final reconstruction and limiter node dependencies.
         * @return Sorted unique global node IDs referenced by selected stencils and any enabled first-ring limiter
         *     bounds.
         */
        [[nodiscard]] std::vector<index> CollectNodeDependencies() const;
        /**
         * @brief Convert retained global stencil/graph IDs to finalized runtime-halo indices and build variational
         *     coupling.
         * @pre NodeHalo::Finalize() has completed; reconstruction operators have already been built.
         */
        void RemapNodeIndices();
        /**
         *  @brief Volume scale for distance weights; not the polynomial reference lengths.
         * @param local Node index in the active NCFV halo layout, covering owned and retained ghost nodes.
         * @return pow(dual volume, 1/dimension), used for distance weights rather than polynomial normalization.
         */
        [[nodiscard]] real LengthScale(index local) const
        {
            return std::pow(_nodeHalo.Volume(local),
                            1.0 / _mesh->getDim());
        }
        /**
         * @brief Read the anisotropic normalization lengths for a node-centered polynomial.
         * @param local Node index in the active NCFV halo layout, covering owned and retained ghost nodes.
         * @return Dual-volume coordinate half-spans from the active halo.
         */
        [[nodiscard]] Vector3 ReferenceLengths(index local) const
        {
            return _nodeHalo.ReferenceLengths(local);
        }

        /**
         * @brief Return the number of nonconstant quadratic polynomial modes.
         * @return Five in 2-D or nine in 3-D.
         */
        [[nodiscard]] int BasisSize() const { return QuadraticBasisSize(_mesh->getDim()); }
        /**
         * @brief Access one owned node's precomputed reconstruction operator.
         * @param iNode Owned anchor-node index.
         * @return Read-only stencil and inverse-matrix metadata.
         */
        [[nodiscard]] const ReconstructionOperator &Operator(index iNode) const
        {
            return _operators.at(static_cast<std::size_t>(iNode));
        }
        /**
         * @brief Access the complete first-ring neighbors used for local reconstruction bounds.
         * @param iNode Owned anchor-node index.
         * @return Read-only neighbor IDs in the current reconstruction node layout.
         */
        [[nodiscard]] const std::vector<index> &DirectNeighbors(index iNode) const
        {
            return _nodeGraph.at(static_cast<std::size_t>(iNode));
        }

        /**
         * @brief Count linear and symmetric quadratic modes without a constant mode.
         * @param dimension Active spatial dimension, either 2 or 3; the third coordinate is ignored in 2-D.
         * @return dimension + dimension*(dimension+1)/2.
         */
        static int QuadraticBasisSize(int dimension);
        /**
         * @brief Evaluate the dimensionless nonconstant quadratic Taylor basis.
         * @param displacement Physical point-minus-anchor offset.
         * @param referenceLengths Positive reference half-spans of the anchor dual volume in each active
         *     coordinate direction.
         * @param dimension Active spatial dimension, either 2 or 3; the third coordinate is ignored in 2-D.
         * @return Basis vector [x,y,x*x/2,x*y,y*y/2] in 2-D, or its nine-mode 3-D counterpart, using normalized
         *     coordinates.
         */
        static Eigen::VectorXd EvaluateBasis(
            const Vector3 &displacement,
            const Vector3 &referenceLengths,
            int dimension);
        /**
         *  @brief Physical derivatives of the dimensionless quadratic basis.
         * @param displacement Physical point-minus-anchor offset.
         * @param referenceLengths Positive reference half-spans of the anchor dual volume in each active
         *     coordinate direction.
         * @param dimension Active spatial dimension, either 2 or 3; the third coordinate is ignored in 2-D.
         * @return dimension by BasisSize() matrix of physical basis derivatives; reference-length factors are
         *     included.
         */
        static Eigen::MatrixXd EvaluateBasisGradient(
            const Vector3 &displacement,
            const Vector3 &referenceLengths,
            int dimension);
        /**
         * @brief Integrate the normalized quadratic basis using supplied dual-volume raw moments.
         * @param moments Positive volume and first/second raw moments in the same coordinate frame as anchor.
         * @param anchor Polynomial origin in the moments' coordinate frame; use zero for already anchor-relative
         *     moments.
         * @param referenceLengths Positive reference half-spans of the anchor dual volume in each active
         *     coordinate direction.
         * @param dimension Active spatial dimension, either 2 or 3; the third coordinate is ignored in 2-D.
         * @return Volume-averaged nonconstant basis vector about anchor.
         */
        static Eigen::VectorXd MeanBasis(
            const RawMoments &moments,
            const Vector3 &anchor,
            const Vector3 &referenceLengths,
            int dimension);

        /**
         *  @brief Compute gradients or complete quadratic coefficient fields.
         * @param means Conservative dual-volume means [rho,rho*u...,rho*E] in the NCFV node-halo layout. Required
         *     ghost entries must already be synchronized.
         * @param [out] gradients Physical conservative-state derivatives, with derivative directions as rows and
         *     conserved components as columns. Filled on owned nodes in efficient mode; traditional mode uses
         *     coefficients instead.
         * @param [out] coefficients Allocated polynomial-mode by conserved-component matrices, filled in
         *     traditional/variational mode; efficient nonvariational mode writes gradients instead.
         */
        void ComputeCoefficients(
            const NodeStatePair &means,
            NodeMatrixPair &gradients,
            NodeMatrixPair &coefficients) const;

        /**
         *  @brief Recover third-order primal-node point values from dual means.
         * @param means Conservative dual-volume means [rho,rho*u...,rho*E] in the NCFV node-halo layout.
         * @param gradients Physical conservative-state derivatives, with derivative directions as rows and
         *     conserved components as columns. Required integration-support ghosts must be available.
         * @param coefficients Quadratic coefficient matrices needed by traditional/variational point recovery.
         * @param [out] pointValues Allocated owned-node conservative point states recovered from volume means;
         *     caller synchronizes output ghosts.
         */
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
         * With limitPointRecovery, the anchor and bounds instead use conservative
         * means and the same factor also limits the mean-to-point increment.
         * @param means Conservative dual-volume means [rho,rho*u...,rho*E] in the NCFV node-halo layout.
         * @param pointValues Recovered conservative primal-node point values, including required ghosts.
         * @param gradients Physical conservative-state derivatives, with derivative directions as rows and
         *     conserved components as columns.
         * @param coefficients Current quadratic coefficient matrices used by the selected integration mode.
         * @return One factor in [0,1] per owned node, common to all conserved components of that node's
         *     correction.
         */
        std::vector<real> ComputeLimiterFactors(
            const NodeStatePair &means,
            const NodeStatePair &pointValues,
            const NodeMatrixPair &gradients,
            const NodeMatrixPair &coefficients) const;

        /**
         * @brief Scale owned reconstruction gradients and stored polynomial coefficients by one node factor.
         * @param factors Limiter coefficients in owned-node order, one per owned node.
         * @param [in,out] gradients Owned physical-gradient matrices to scale.
         * @param [in,out] coefficients Owned stored polynomial coefficient matrices to scale using the same
         *     factors.
         * @note Does not exchange ghosts; the caller synchronizes limited fields before interface evaluation.
         */
        void ApplyLimiter(
            const std::vector<real> &factors,
            NodeMatrixPair &gradients,
            NodeMatrixPair &coefficients) const;
    };
}
