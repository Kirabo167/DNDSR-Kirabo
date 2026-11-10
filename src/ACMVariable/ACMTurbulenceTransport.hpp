/**
 * @file ACMTurbulenceTransport.hpp
 * @brief Segregated finite-volume transport interface for ACM RANS variables.
 *
 * @details Turbulence variables are stored in a separate distributed two-entry field;
 * only the entries required by the selected runtime model are active.  This preserves
 * the existing ACM `[rho,mx,my,mz,p]` state, characteristic reconstruction, and 5x5 implicit
 * operators.  MPI ghost exchange and Geom/CFV metrics are reused without changing Euler.
 *
 * @author Runzhi Ma
 * @date 2026-09-03
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMBC.hpp"
#include "ACMTime.hpp"
#include "ACMBDF2.hpp"
#include "ACMTurbulence.hpp"

#include "CFV/VariationalReconstruction.hpp"
#include "Geom/Mesh/Mesh.hpp"

#include <vector>

namespace DNDS::ACMVariable
{
    /**
     * @brief Runtime turbulence transport coupled segregatedly to an ACM flow evaluator.
     * @tparam gDim Geometric dimension, either two or three.
     * @note Modifier: Runzhi Ma.
     */
    template <int gDim>
    class ACMTurbulenceTransport
    {
    public:
        static_assert(gDim == 2 || gDim == 3,
                      "ACM turbulence transport supports only 2-D and 3-D meshes");
        static constexpr int nFlowVariables = 5;
        static constexpr int nStoredVariables = 2;

        using TFlowDof = CFV::tUDof<nFlowVariables>;         ///< Distributed ACM `[rho,mx,my,mz,p]` field.
        using TTurbulenceDof = CFV::tUDof<nStoredVariables>; ///< Separate primitive RANS-variable field.
        using TTurbulenceGradient =
            CFV::tUGrad<nStoredVariables, gDim>;              ///< Cell turbulence gradients.
        using TFlowGradient = CFV::tUGrad<nFlowVariables, gDim>; ///< Cell flow gradients.
        using TLimiter = CFV::tUDof<nStoredVariables>;        ///< Per-variable linear limiter factors.
        using TFlowLimiter = CFV::tUDof<1>;                   ///< Rotation-invariant scalar flow limiter.
        using TVFV = CFV::VariationalReconstruction<gDim>;    ///< Existing CFV metric provider.
        using TpVFV = ssp<TVFV>;                              ///< Shared CFV metric pointer.

        /**
         * @brief Construct and allocate a segregated turbulence transport object.
         * @param mesh Prepared distributed mesh.
         * @param vfv Initialized ACM CFV reconstruction/metric object.
         * @param flowSettings Variable-density ACM physical settings.
         * @param turbulenceSettings Runtime turbulence model and transport controls.
         * @param boundaryHandler Per-zone ACM boundary mapping.
         */
        ACMTurbulenceTransport(
            ssp<Geom::UnstructuredMesh> mesh,
            TpVFV vfv,
            const Settings &flowSettings,
            const TurbulenceSettings &turbulenceSettings,
            ssp<BoundaryHandler> boundaryHandler);

        /**
         * @brief Build wall distances and initialize the distributed turbulence state.
         * @details This method must be called once after mesh preparation and before transport.
         */
        void Initialize();

        /**
         * @brief Reconstruct gradients and freeze face eddy viscosities for a flow state.
         * @param flow Current distributed ACM flow field; ghost values are refreshed.
         * @param time Current pseudo/physical time reserved for boundary data.
         */
        void Prepare(TFlowDof &flow, real time = 0);

        /**
         * @brief Evaluate the segregated turbulence residual.
         * @param rhs Output primitive-variable residual `-div(u q-D grad(q))+S`.
         * @param flow Current distributed ACM flow field.
         * @param time Current pseudo/physical time.
         */
        void EvaluateRHS(TTurbulenceDof &rhs, TFlowDof &flow, real time = 0);

        /**
         * @brief Advance turbulence variables with positivity-preserving substepped SSPRK3.
         * @param flow Flow field frozen over this segregated turbulence update.
         * @param flowPseudoTimeStep Positive owned-cell flow pseudo-time steps.
         * @param time Current pseudo/physical time.
         * @return MPI-global RMS turbulence residual from the last SSPRK stage.
         */
        real Advance(
            TFlowDof &flow,
            const ScalarField &flowPseudoTimeStep,
            real time = 0);

        /**
         * @brief Return frozen turbulent dynamic viscosity on a face quadrature point.
         * @param iFace Process-local face index.
         * @param iG Face quadrature index; `-1` returns the arithmetic quadrature mean.
         * @return Non-negative turbulent dynamic viscosity.
         */
        real FaceEddyViscosity(index iFace, int iG) const;

        /** @brief Return mutable distributed primitive turbulence variables. */
        TTurbulenceDof &GetState() { return _state; }

        /** @brief Return read-only distributed primitive turbulence variables. */
        const TTurbulenceDof &GetState() const { return _state; }

        /** @brief Return the selected runtime turbulence model. */
        /** @brief Left, right and boundary-target isotropic pressures for the flow Riemann solver.
         * Increments are conservative rho*k cell-mean perturbations; reconstructed slopes stay frozen.
         */
        std::array<real, 3> FaceTurbulentPressure(index face, int point, const State &left,
            const State &right, real leftIncrement = 0, real rightIncrement = 0) const;

        using FlowBoundaryFunction = std::function<State(index, const State &, const Vector3 &, const Vector3 &, real)>;
        /** @brief Share the pressure-consistent flow boundary with closure gradients. */
        void SetFlowBoundaryProvider(FlowBoundaryFunction boundary) { _flowBoundary = std::move(boundary); }

        TurbulenceModel GetModel() const { return _turbulenceSettings.model; }

        /** @brief Return the number of active transport equations. */
        int ActiveVariableCount() const
        {
            return TurbulenceVariableCount(_turbulenceSettings.model);
        }

        /**
         * @brief Attach the flow evaluator's conservative mass-flux cache.
         * @param provider Callable (face,quadrature) returning last flow mass flux.
         * @author Runzhi Ma
         */
        void SetMassFluxProvider(std::function<real(index, int)> provider)
        { _massFlux = std::move(provider); }

        /**
         * @brief Initialize or commit physical history of rho*phi, not phi.
         * @param flow Current positive-density flow field; call only for accepted physical states.
         * @param initialize True initializes both history levels; false shifts history.
         * @author Runzhi Ma
         */
        void CommitPhysicalHistory(const TFlowDof &flow, bool initialize);

        /**
         * @brief Evaluate or relax the conservative RANS BDF physical defect.
         * @param flow Current flow iterate; its flux cache must match this iterate.
         * @param coefficients BDF1-startup/BDF2 coefficients shared with flow.
         * @param dt Physical step.
         * @param pseudoSteps Cell-local pseudo steps; nullptr evaluates without updating.
         * @param time Boundary time.
         * @return Global RMS of the conservative turbulence defect BEFORE relaxation.
         * @author Runzhi Ma
         */
        real PhysicalDefect(TFlowDof &flow, const BDF2Coefficients &coefficients,
                            real dt, const ScalarField *pseudoSteps, real time);

    private:
        FlowBoundaryFunction _flowBoundary;
        State FlowBoundaryState(index face, const State &left, const Vector3 &normal,
            const Settings &settings, const Vector3 &point, real time) const;

        const TFlowDof *_flow = nullptr; ///< Current density/velocity data, refreshed by Prepare.
        std::function<real(index, int)> _massFlux; ///< Shared oriented mass flux from flow.
        std::vector<TurbulenceState> _previous, _previousPrevious; ///< Accepted rho*phi histories.
        ssp<Geom::UnstructuredMesh> _mesh;          ///< Prepared distributed mesh.
        TpVFV _vfv;                                 ///< Existing CFV metrics and periodic maps.
        Settings _flowSettings;                     ///< Constant density and molecular viscosity.
        TurbulenceSettings _turbulenceSettings;     ///< Runtime closure/transport configuration.
        ssp<BoundaryHandler> _boundaryHandler;      ///< Per-zone boundary mapping.
        TTurbulenceDof _state;                       ///< Current primitive RANS variables.
        TTurbulenceDof _rhs;                         ///< Internal SSPRK residual buffer.
        TTurbulenceGradient _gradient;               ///< Current turbulence gradients.
        TFlowGradient _flowGradient;                 ///< Current flow gradients for production/mu_t.
        TLimiter _limiter;                           ///< Componentwise face-reconstruction limiter.
        TFlowLimiter _flowLimiter;                   ///< Shared limiter for all reconstructed flow entries.
        std::vector<real> _cellWallDistance;         ///< One positive value per owned/ghost cell.
        std::vector<real> _faceWallDistance;         ///< One positive value per process face.
        std::vector<std::vector<real>> _faceEddyViscosity; ///< Frozen values at face quadrature points.
        bool _initialized = false;                   ///< Wall-distance/state initialization guard.
        bool _prepared = false;                      ///< Frozen-viscosity cache validity guard.

        /**
         * @brief Compute cell/face wall distances through the existing Geom implementation.
         * @note Modifier: Runzhi Ma.
         */
        void BuildWallDistance();

        /**
         * @brief Compute Green--Gauss gradients for flow and turbulence fields.
         * @param flow Current flow state with valid ghost values.
         * @param time Current boundary-evaluation time.
         */
        void EvaluateGradients(TFlowDof &flow, real time);

        /**
         * @brief Compute per-variable local-extrema factors for turbulence reconstruction.
         * @param flow Current flow field used to bound the auxiliary face velocity.
         * @param time Current boundary-evaluation time.
         */
        void EvaluateLimiter(const TFlowDof &flow, real time);

        /**
         * @brief Reconstruct one turbulence state from a cell to a face quadrature point.
         * @param iCell Owned/ghost cell index.
         * @param iFace Face index.
         * @param if2c Face side (`0` left/back, `1` right/front).
         * @param iG Face quadrature index.
         * @return Positivity-clamped primitive turbulence face state.
         */
        TurbulenceState ReconstructTurbulenceState(
            index iCell,
            index iFace,
            rowsize if2c,
            int iG) const;

        /**
         * @brief Reconstruct one ACM flow state for turbulence advection/production.
         * @param flow Distributed flow state.
         * @param iCell Owned/ghost cell index.
         * @param iFace Face index.
         * @param if2c Face side.
         * @param iG Face quadrature index.
         * @return Linearly reconstructed `[rho,mx,my,mz,p]` state.
         */
        State ReconstructFlowState(
            const TFlowDof &flow,
            index iCell,
            index iFace,
            rowsize if2c,
            int iG) const;

        /**
         * @brief Convert a stored `gDim x 2` gradient to the closure's `3 x 2` convention.
         * @param iCell Cell index.
         * @return Zero-padded global turbulence gradient.
         */
        TurbulenceGradient CellTurbulenceGradient(index iCell) const;

        /**
         * @brief Convert a stored flow gradient to `du_i/dx_j` closure convention.
         * @param iCell Cell index.
         * @return Zero-padded three-dimensional velocity-gradient tensor.
         */
        VelocityGradient CellVelocityGradient(index iCell) const;

        /**
         * @brief Generate an exterior turbulence state for one boundary face.
         * @param iFace Face index used to obtain the configured boundary family.
         * @param interior Interior turbulence state.
         * @param ownerCell Owner-cell index supplying wall distance.
         * @return Boundary ghost state.
         */
        TurbulenceState GenerateBoundary(
            index iFace,
            const TurbulenceState &interior,
            index ownerCell) const;

        /**
         * @brief Convert a Geom point/vector to ACM's fixed three-component vector.
         * @param value Geometric point/vector.
         * @return Three-component value with zero unused entries in 2-D.
         */
        static Vector3 ToVector3(const Geom::tPoint &value);
    };

    extern template class ACMTurbulenceTransport<2>;
    extern template class ACMTurbulenceTransport<3>;
}
