#pragma once

#include "DNDS/Defines.hpp"

namespace DNDS::NCFV
{
    template <int dimension, int nTurbulence>
    struct StateLayout
    {
        static constexpr int FlowSize = dimension + 2;
        static constexpr int Size = FlowSize + nTurbulence;
        using State = Eigen::Vector<real, Size>;

        static real Pressure(const State &state, real gamma)
        {
            return (gamma - 1) * (state[dimension + 1] -
                                  .5 * state.template segment<dimension>(1).squaredNorm() / state[0]);
        }

        static bool TurbulenceAdmissible(const State &state)
        {
            if constexpr (nTurbulence == 2)
                return (state.template tail<2>().array() > 0).all();
#ifndef USE_NS_SA_NEGATIVE_MODEL
            if constexpr (nTurbulence == 1)
                return state[FlowSize] >= 0;
#endif
            return true;
        }

        static bool IsPhysical(const State &state, real gamma)
        {
            return state.allFinite() && state[0] > 1e-12 && Pressure(state, gamma) > 1e-12 &&
                   TurbulenceAdmissible(state);
        }

        static State ToConservative(const State &primitive, real gamma, real viscosityReference)
        {
            State state = primitive;
            state.template segment<dimension>(1) *= primitive[0];
            state[dimension + 1] = primitive[dimension + 1] / (gamma - 1) +
                                   .5 * primitive[0] * primitive.template segment<dimension>(1).squaredNorm();
            if constexpr (nTurbulence > 0)
                state.template tail<nTurbulence>() *= primitive[0];
            if constexpr (nTurbulence == 1)
                state[FlowSize] /= viscosityReference;
            return state;
        }

        static State ToPrimitive(const State &state, real gamma, real viscosityReference)
        {
            State primitive = state;
            primitive.template segment<dimension>(1) /= state[0];
            primitive[dimension + 1] = Pressure(state, gamma);
            if constexpr (nTurbulence > 0)
                primitive.template tail<nTurbulence>() /= state[0];
            if constexpr (nTurbulence == 1)
                primitive[FlowSize] *= viscosityReference;
            return primitive;
        }
    };
}
