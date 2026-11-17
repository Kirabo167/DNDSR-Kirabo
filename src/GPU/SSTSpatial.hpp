#pragma once

namespace DNDS::GPU
{
#if defined(__CUDACC__)
#define DNDS_SST_SPATIAL_INLINE __host__ __device__ inline
#else
#define DNDS_SST_SPATIAL_INLINE inline
#endif
    /** Dirichlet turbulence face data, distinct from the reflected diffusion ghost. */
    DNDS_SST_SPATIAL_INLINE bool SSTPrescribedInflow(int boundary)
    {
        return boundary == 1 || boundary == 7 || boundary == 8 || boundary == 10;
    }

    /** Minimum/maximum flags use the complete cell stencil, including boundary ghosts. */
    DNDS_SST_SPATIAL_INLINE int SSTExtremaFlags(double value, double lower, double upper)
    {
        return (value <= lower ? 1 : 0) | (value >= upper ? 2 : 0);
    }

    /** Prevent diffusion from deepening a stencil extremum; retain the full correction elsewhere.
     * The face derivative points from left to right. One shared face flux remains conservative.
     * This constrains diffusion only, not sources or the complete time-discrete update.
     */
    DNDS_SST_SPATIAL_INLINE double SSTBoundNormalDerivative(double derivative, int left, int right)
    {
        if (((left & 1) || (right & 2)) && derivative < 0) derivative = 0;
        if (((left & 2) || (right & 1)) && derivative > 0) derivative = 0;
        return derivative;
    }

    /** Frozen non-negative face conductance for limited nonorthogonal diffusion.
     * Bound the correction by the magnitude of the two-point contribution.
     * The resulting shared face flux is jump*conductance, with conductance in
     * [0, 2*twoPoint]. Equal states have zero flux independently of their gradients.
     * This replaces exact-extremum switches in the transport operator. Keeping
     * the same coefficient in the implicit operator gives a frozen-coefficient
     * linearization, not the full derivative of this nonlinear limiter.
     */
    DNDS_SST_SPATIAL_INLINE double SSTNormalConductance(
        double derivative, double jump, double projection, double normalSquared)
    {
        const double distance = projection < 0 ? -projection : projection;
        if (distance <= 1e-14 || normalSquared <= 0) return 0;
        const double twoPoint = normalSquared / distance;
        if (jump == 0) return twoPoint;
        const double candidate = derivative / jump;
        return candidate < 0 ? 0 : candidate > 2 * twoPoint ? 2 * twoPoint : candidate;
    }
#undef DNDS_SST_SPATIAL_INLINE
}
