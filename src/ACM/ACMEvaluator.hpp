/**
 * @file ACMEvaluator.hpp
 * @brief Mesh-based high-order spatial evaluator for constant-density ACM equations.
 *
 * @details This module reuses DNDSR's geometry and CFV variational-reconstruction
 * infrastructure without including or modifying the Euler equation evaluator. It owns
 * the reconstruction buffers, performs MPI ghost exchange, reconstructs left/right
 * face states at quadrature points, evaluates ACM Riemann/viscous fluxes, and gathers
 * conservative face contributions into cell residuals.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMBC.hpp"
#include "ACMConfig.hpp"
#include "ACMFlux.hpp"

#include "CFV/VariationalReconstruction.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include "Solver/Linear.hpp"

#include <functional>
#include <cstdint>

namespace DNDS::ACM
{
    /**
     * @brief High-order unstructured finite-volume evaluator for one geometric dimension.
     * @tparam gDim Mesh dimension, either 2 or 3. The state remains `[u,v,w,p]` in both cases.
     * @note Modifier: Runzhi Ma.
     */
    template <int gDim>
    class ACMEvaluator
    {
    public:
        static_assert(gDim == 2 || gDim == 3, "ACMEvaluator supports only 2-D and 3-D meshes");
        static constexpr int nVarsFixed = 4;

        using TDof = CFV::tUDof<nVarsFixed>;               ///< Distributed cell-mean state array.
        using TRec = CFV::tURec<nVarsFixed>;               ///< Distributed variational coefficients.
        using TGrad = CFV::tUGrad<nVarsFixed, gDim>;       ///< Distributed direct gradients.
        using TScalar = CFV::tUDof<1>;                     ///< Distributed scalar limiter array.
        using TScalarPair = CFV::tScalarPair;              ///< CFV smooth-indicator pair.
        using TVFV = CFV::VariationalReconstruction<gDim>; ///< Existing DNDS CFV reconstruction engine.
        using TpVFV = ssp<TVFV>;                           ///< Shared reconstruction engine pointer.
        using TBoundaryFunction = typename TVFV::template TFBoundary<nVarsFixed>;
        using TBoundaryDiffFunction = typename TVFV::template TFBoundaryDiff<nVarsFixed>;
        struct ReconstructionReport
        {
            int iterations = 0;
            real equationDefect = 0;
            bool converged = false; ///< True only after checking the VR equation defect.
        };
        const ReconstructionReport &GetReconstructionReport() const { return _reconstructionReport; }
        std::uint64_t GetTotalReconstructionSweeps() const { return _totalReconstructionSweeps; }
        using TurbulencePrepareFunction =
            std::function<void(TDof &, real)>; ///< Refresh segregated turbulence data for a flow state.
        using TurbulentViscosityFunction =
            std::function<real(index, int)>; ///< Return frozen `mu_t` for `(face,quadrature)`.

        /**
         * @brief Derivatives of one integrated numerical face flux with respect to both cells.
         * @note On a physical boundary `right` is zero and `left` includes the ghost-state chain rule.
         * @note Modifier: Runzhi Ma.
         */
        struct FaceJacobianBlocks
        {
            Matrix4 left = Matrix4::Zero();  ///< `d integral(F_face) / d U_left`.
            Matrix4 right = Matrix4::Zero(); ///< `d integral(F_face) / d U_right`.
        };

        using FaceJacobianField = std::vector<FaceJacobianBlocks>; ///< One block pair per process face.

        /**
         * @brief Construct an ACM evaluator and allocate its reconstruction work arrays.
         * @param mesh Distributed, solver-ready unstructured mesh.
         * @param vfv Initialized CFV variational-reconstruction object on `mesh`.
         * @param settings ACM physical and numerical-flux settings.
         * @param reconstructionSettings High-order reconstruction and limiter controls.
         * @param boundaryHandler Per-zone ACM boundary mapping shared with the mesh reader.
         */
        ACMEvaluator(
            ssp<Geom::UnstructuredMesh> mesh,
            TpVFV vfv,
            const Settings &settings,
            const ReconstructionSettings &reconstructionSettings,
            ssp<BoundaryHandler> boundaryHandler);

        /**
         * @brief Initialize common CFV metrics and reconstruction matrices as Euler does.
         * @param mesh Distributed mesh whose periodic transformations are reused.
         * @param vfv Reconstruction object to initialize.
         * @param boundaryHandler Per-zone boundary mapping used for VR boundary weights.
         */
        static void InitializeFV(
            const ssp<Geom::UnstructuredMesh> &mesh,
            const TpVFV &vfv,
            const ssp<BoundaryHandler> &boundaryHandler);

        /**
         * @brief Attach optional segregated turbulence callbacks without changing the ACM state layout.
         * @param prepareFunction Callback that reconstructs turbulence fields and freezes face `mu_t`.
         * @param viscosityFunction Callback returning frozen turbulent dynamic viscosity.
         * @details Both callbacks must be supplied together. Empty callbacks restore laminar behavior.
         * @note Modifier: Runzhi Ma.
         */
        void SetTurbulenceCoupling(
            TurbulencePrepareFunction prepareFunction,
            TurbulentViscosityFunction viscosityFunction);

        /**
         * @brief Reconstruct the current distributed cell means.
         * @param u Cell means; owned values are pulled to ghost cells before reconstruction.
         */
        void Reconstruct(TDof &u, real time = 0);

        /**
         * @brief Evaluate the raw conservative spatial residual on all locally owned cells.
         * @param rhs Output residual `-div(F_inviscid-F_viscous)`.
         * @param u Distributed cell-mean ACM state.
         * @param time Current pseudo/physical time, reserved for time-dependent boundaries.
         */
        void EvaluateRHS(TDof &rhs, TDof &u, real time = 0);

        /**
         * @brief Assemble a frozen-reconstruction approximation to each residual diagonal block.
         * @param u Current distributed cell means.
         * @param diagonal Output `dR_i/dU_i` blocks for locally owned cells.
         * @param time Current time, reserved for time-dependent boundaries.
         * @details The face flux is finite-differenced while reconstruction coefficients are frozen.
         * This matches the usual inexpensive block-Jacobi linearization strategy.
         */
        void EvaluateDiagonalJacobian(
            TDof &u,
            std::vector<Matrix4> &diagonal,
            real time = 0);

        /**
         * @brief Compute Euler-style spectral-radius local pseudo-time steps.
         * @param pseudoTimeStep Output time step for every locally owned cell.
         * @param u Current distributed ACM cell means.
         * @param cfl Positive CFL multiplier.
         * @param maximumTimeStep Positive upper bound applied to every local step.
         * @param useLocalTimeStep Keep cell-local values when true; otherwise use the MPI minimum.
         * @param time Current boundary-evaluation time.
         * @return Global MPI minimum of the uncloned local time-step field.
         */
        real EvaluateTimeStep(
            ScalarField &pseudoTimeStep,
            TDof &u,
            real cfl,
            real maximumTimeStep,
            bool useLocalTimeStep,
            real time = 0);

        /**
         * @brief Assemble the ACM backward-Euler linearization used by LU-SGS and GMRES.
         * @param u Current distributed state.
         * @param pseudoTimeStep Positive local pseudo-time steps for owned cells.
         * @param diagonal Output diagonal blocks of `Gamma/dTau - dR/dU`.
         * @param faceJacobians Output integrated first-order numerical-flux Jacobian blocks.
         * @param time Current boundary-evaluation time.
         * @details Face derivatives are finite-differenced from the ACM Riemann/viscous flux and
         * include boundary ghost-state derivatives. No Euler Jacobian is called.
         */
        void AssembleImplicitLinearization(
            TDof &u,
            const ScalarField &pseudoTimeStep,
            MatrixField &diagonal,
            FaceJacobianField &faceJacobians,
            real time = 0);

        /**
         * @brief Apply the assembled distributed ACM implicit operator.
         * @param increment Input distributed correction vector; ghost values are refreshed.
         * @param diagonal Cell diagonal blocks.
         * @param faceJacobians Integrated face coupling blocks.
         * @param result Output operator product on owned cells.
         */
        void ApplyImplicitLinearization(
            TDof &increment,
            const MatrixField &diagonal,
            const FaceJacobianField &faceJacobians,
            TDof &result) const;

        /**
         * @brief Apply an independent 4x4 block-Jacobi inverse.
         * @param rhs Input distributed right-hand side.
         * @param diagonal Cell diagonal blocks.
         * @param result Output preconditioned vector.
         */
        void ApplyBlockJacobi(
            const TDof &rhs,
            const MatrixField &diagonal,
            TDof &result) const;

        /**
         * @brief Solve approximately with parallel block LU-SGS sweeps.
         * @param rhs Input distributed right-hand side.
         * @param diagonal Cell diagonal blocks.
         * @param faceJacobians Integrated off-diagonal face blocks.
         * @param result Output correction; reset to zero on entry.
         * @param nSweeps Number of fixed SGS residual-correction applications.
         * @details The first sweep starts from zero. Every later sweep applies a residual
         * correction `x += P_SGS^{-1}(rhs-A*x)`. The operator residual synchronizes off-rank
         * values, while each triangular correction solve is local to one MPI partition.
         */
        void SolveLUSGS(
            const TDof &rhs,
            const MatrixField &diagonal,
            const FaceJacobianField &faceJacobians,
            TDof &result,
            int nSweeps);

        /**
         * @brief Return the distributed mesh used by this evaluator.
         * @return Shared mesh pointer.
         */
        const ssp<Geom::UnstructuredMesh> &GetMesh() const { return _mesh; }

        /**
         * @brief Return the initialized CFV reconstruction engine.
         * @return Shared VFV pointer.
         */
        const TpVFV &GetVFV() const { return _vfv; }

    private:
        ssp<Geom::UnstructuredMesh> _mesh;              ///< Distributed mesh; modifier: Runzhi Ma.
        TpVFV _vfv;                                     ///< Existing CFV reconstruction object.
        Settings _settings;                             ///< ACM flux/preconditioning settings.
        ReconstructionSettings _reconstructionSettings; ///< Reconstruction controls.
        ssp<BoundaryHandler> _boundaryHandler;          ///< Per-zone boundary data and name mapping.
        TRec _uRec;                                     ///< Current VR coefficients.
        TRec _uRecWork;                                 ///< VR fixed-point workspace.
        TRec _uRecCorrection;                           ///< Consistent VR defect correction.
        ReconstructionReport _reconstructionReport;
        std::uint64_t _totalReconstructionSweeps = 0;
        TRec _uRecLimited;                              ///< WBAP/CWBAP output coefficients.
        TGrad _uGrad;                                   ///< Direct Green-Gauss gradients.
        TScalar _limiter;                               ///< One limiter factor per cell.
        TScalarPair _smoothIndicator;                   ///< CFV troubled-cell indicator.
        TDof _lusgsOperatorProduct;                     ///< Work array for `A*x` between LU-SGS sweeps.
        TDof _lusgsCorrection;                          ///< Work array for one LU-SGS residual correction.
        TurbulencePrepareFunction _prepareTurbulence;   ///< Optional segregated preparation hook.
        TurbulentViscosityFunction _turbulentViscosity; ///< Optional frozen face `mu_t` accessor.

        /**
         * @brief Return frozen turbulent viscosity or zero in laminar mode.
         * @param iFace Process-local face index.
         * @param iG Face quadrature index; `-1` requests the face average.
         * @return Non-negative turbulent dynamic viscosity.
         */
        real GetTurbulentViscosity(index iFace, int iG) const;

        /**
         * @brief Build the CFV boundary callback used by gradient and VR reconstruction.
         * @param time Current boundary-evaluation time.
         * @return Callback mapping an interior reconstruction to its ghost state.
         */
        TBoundaryFunction GetBoundaryFunction(real time) const;

        /** @brief Build the finite-increment boundary response used by reconstruction GMRES. */
        TBoundaryDiffFunction GetBoundaryDiffFunction(real time) const;

        /**
         * @brief Generate the exterior state for one mesh boundary face.
         * @param faceZone Mesh face-zone ID.
         * @param interior Interior state.
         * @param normal Outward unit normal.
         * @param point Physical face point.
         * @param time Current boundary-evaluation time.
         * @return Exterior Riemann state.
         */
        State GenerateBoundaryForFace(
            Geom::t_index faceZone,
            const State &interior,
            const Vector3 &normal,
            const Vector3 &point,
            real time) const;

        /**
         * @brief Compute and communicate local-extrema limiter factors.
         * @param u Current cell means including refreshed ghost values.
         * @param time Current boundary-evaluation time.
         */
        void EvaluateLimiter(TDof &u, real time);

        /**
         * @brief Apply the existing CFV WBAP or CWBAP kernel in ACM characteristic variables.
         * @param u Current distributed ACM cell means.
         * @details The generic CFV limiter is reused; only the four-variable ACM characteristic
         * transforms and pressure/velocity smooth indicator are defined here.
         * @note Modifier: Runzhi Ma.
         */
        void ApplyCharacteristicLimiter(TDof &u);

        /**
         * @brief Reconstruct one state at a cell-side face quadrature point.
         * @param u Distributed cell means.
         * @param iCell Local owned/ghost cell index.
         * @param iFace Local face index.
         * @param if2c Face side (`0` for back/left, `1` for front/right).
         * @param iG Face quadrature-point index.
         * @return Limited face state `[u,v,w,p]`.
         */
        State ReconstructFaceState(
            const TDof &u,
            index iCell,
            index iFace,
            rowsize if2c,
            int iG) const;

        /**
         * @brief Reconstruct the state-gradient matrix at a face quadrature point.
         * @param iCell Local owned/ghost cell index.
         * @param iFace Local face index.
         * @param if2c Face side (`0` or `1`).
         * @param iG Face quadrature-point index.
         * @return Three-by-four gradient with zero unused rows in 2-D.
         */
        Eigen::Matrix<real, 3, 4> ReconstructFaceGradient(
            index iCell,
            index iFace,
            rowsize if2c,
            int iG) const;

        /**
         * @brief Convert a geometric face normal to the three-component ACM convention.
         * @param normal DNDS geometry normal.
         * @return Three-component normal with zero z component in 2-D.
         */
        static Vector3 ToVector3(const Geom::tPoint &normal);
    };

    extern template class ACMEvaluator<2>;
    extern template class ACMEvaluator<3>;
}
