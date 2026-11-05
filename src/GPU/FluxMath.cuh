#pragma once
#include "FluxBatch.hpp"
#include <cmath>

namespace DNDS::GPU::Detail
{
    __device__ inline double FixedAbs(double value, double delta)
    {
        const double magnitude = fabs(value);
        return magnitude < delta ? .5 * (value * value / delta + delta) : magnitude;
    }

    __device__ inline double FixedDerivative(double value, double delta)
    {
        return fabs(value) < delta ? value / delta : (value > 0 ? 1 : value < 0 ? -1
                                                                                : 0);
    }

    __device__ inline void ApplyB(const double *x, double *result, const double *mean,
                                  const double *normal, double un, const FluxInput &in, double shift)
    {
        double nx = 0;
        for (int i = 0; i < 3; ++i)
            nx += normal[i] * x[i];
        for (int i = 0; i < 3; ++i)
            result[i] = (un - shift) * x[i] - in.alpha * mean[i] * nx + normal[i] * x[3] / in.rho0;
        result[3] = in.beta2 * nx - shift * x[3];
    }

    __device__ inline void ACMFlux(const FluxInput &in, FluxOutput &out)
    {
        double normal[3], mean[4], jump[4], un = 0, unL = 0, unR = 0;
        const double norm = sqrt(in.normal[0] * in.normal[0] + in.normal[1] * in.normal[1] + in.normal[2] * in.normal[2]);
        for (int i = 0; i < 4; ++i)
        {
            mean[i] = .5 * (in.left[i] + in.right[i]);
            jump[i] = in.right[i] - in.left[i];
        }
        for (int i = 0; i < 3; ++i)
        {
            normal[i] = in.normal[i] / norm;
            un += normal[i] * mean[i];
            unL += normal[i] * in.left[i];
            unR += normal[i] * in.right[i];
        }
        const double centered = (1 - in.alpha) * un;
        const double root = hypot(centered, 2 * sqrt(in.beta2 / in.rho0));
        const double minus = .5 * (centered - root), plus = .5 * (centered + root);
        const double radius = fmax(fabs(un), fmax(fabs(minus), fabs(plus)));
        double transformed[4]{};
        if (in.solver == 1)
            for (int i = 0; i < 4; ++i)
                transformed[i] = radius * jump[i];
        else
        {
            // Hermite interpolation of |B|, including repeated/defective eigenvalues.
            const double delta = in.entropyRatio * fmax(radius, sqrt(in.beta2 / in.rho0));
            const double separated = fabs(un - minus) >= fabs(un - plus) ? minus : plus;
            const double clustered = separated == minus ? plus : minus;
            const double c0 = FixedAbs(separated, delta);
            const double c1 = (FixedAbs(un, delta) - c0) / (un - separated);
            const double c2 = (FixedDerivative(un, delta) - c1) / (un - separated);
            const double gap = clustered - un;
            const double scale = fmax(1., fmax(fabs(un), fmax(fabs(clustered), fabs(delta))));
            double repeated;
            if (fabs(gap) <= 1e-8 * scale)
                repeated = fabs(un) < delta ? .5 / delta : 0;
            else if (fabs(un) < delta && fabs(clustered) < delta)
                repeated = .5 / delta;
            else if ((un > delta && clustered > delta) || (un < -delta && clustered < -delta))
                repeated = 0;
            else
                repeated = ((FixedAbs(clustered, delta) - FixedAbs(un, delta)) / gap - FixedDerivative(un, delta)) / gap;
            const double c3 = (repeated - c2) / (clustered - separated);
            double x1[4], x2[4], x3[4];
            ApplyB(jump, x1, mean, normal, un, in, separated);
            ApplyB(x1, x2, mean, normal, un, in, un);
            ApplyB(x2, x3, mean, normal, un, in, un);
            for (int i = 0; i < 4; ++i)
                transformed[i] = c0 * jump[i] + c1 * x1[i] + c2 * x2[i] + c3 * x3[i];
        }
        // Transform by Gamma after applying |Gamma^-1 A|.
        for (int i = 0; i < 3; ++i)
        {
            const double dissipation = transformed[i] + (1 + in.alpha) * mean[i] * transformed[3] / in.beta2;
            out.value[i] = .5 * (unL * in.left[i] + unR * in.right[i] + normal[i] * (in.left[3] + in.right[3]) / in.rho0 - dissipation);
        }
        out.value[3] = .5 * (unL + unR - transformed[3] / in.beta2);
    }

    __device__ inline void EulerFlux(const FluxInput &in, FluxOutput &out)
    {
        const int d = in.dimension;
        double vl[3]{}, vr[3]{}, v[3]{}, vl2 = 0, vr2 = 0, v2 = 0;
        double unl = 0, unr = 0, un = 0, dmN = 0;
        const double rl = in.left[0], rr = in.right[0], ratio = sqrt(rr / rl);
        for (int i = 0; i < d; ++i)
        {
            vl[i] = in.left[i + 1] / rl;
            vr[i] = in.right[i + 1] / rr;
            v[i] = (vl[i] + ratio * vr[i]) / (1 + ratio);
            vl2 += vl[i] * vl[i];
            vr2 += vr[i] * vr[i];
            v2 += v[i] * v[i];
            unl += vl[i] * in.normal[i];
            unr += vr[i] * in.normal[i];
            un += v[i] * in.normal[i];
            dmN += (in.right[i + 1] - in.left[i + 1]) * in.normal[i];
        }
        const double pl = (in.gamma - 1) * (in.left[d + 1] - .5 * rl * vl2);
        const double pr = (in.gamma - 1) * (in.right[d + 1] - .5 * rr * vr2);
        const double hl = (in.left[d + 1] + pl) / rl, hr = (in.right[d + 1] + pr) / rr;
        const double h = (hl + ratio * hr) / (1 + ratio);
        const double a = sqrt((in.gamma - 1) * (h - .5 * v2));
        const double al = sqrt(in.gamma * pl / rl), ar = sqrt(in.gamma * pr / rr);
        double speeds[3];
        for (int k = 0; k < 3; ++k)
        {
            const double sign = k - 1;
            const double lambda = un + sign * a;
            const double delta = in.referenceHarten ? .15 * a : .05 * fmax(0., fmax(lambda - (unl + sign * al), unr + sign * ar - lambda));
            speeds[k] = FixedAbs(lambda, delta);
        }
        double jump[5], shear[3]{}, projection = 0, shearDot = 0;
        for (int i = 0; i < d + 2; ++i)
            jump[i] = in.right[i] - in.left[i];
        for (int i = 0; i < d; ++i)
            projection += (jump[i + 1] - jump[0] * v[i]) * in.normal[i];
        for (int i = 0; i < d; ++i)
        {
            shear[i] = jump[i + 1] - jump[0] * v[i] - in.normal[i] * projection;
            shearDot += shear[i] * v[i];
        }
        double b1 = (jump[0] * (h - un * un) + un * dmN - (jump[d + 1] - shearDot)) / (h - .5 * v2);
        double b0 = (jump[0] * (un + a) - dmN - a * b1) / (2 * a);
        double b4 = jump[0] - b0 - b1;
        b0 *= speeds[0];
        b1 *= speeds[1];
        b4 *= speeds[2];
        double diss[5]{};
        diss[0] = b0 + b1 + b4;
        for (int i = 0; i < d; ++i)
            diss[i + 1] = (v[i] - a * in.normal[i]) * b0 + v[i] * b1 + (v[i] + a * in.normal[i]) * b4 + speeds[1] * shear[i];
        diss[d + 1] = (h - un * a) * b0 + .5 * v2 * b1 + (h + un * a) * b4 + speeds[1] * shearDot;
        if (in.solver == 1)
            for (int i = 0; i < d + 2; ++i)
                diss[i] = fmax(fabs(unl) + al, fabs(unr) + ar) * jump[i];
        out.value[0] = .5 * (rl * unl + rr * unr - diss[0]);
        for (int i = 0; i < d; ++i)
            out.value[i + 1] = .5 * (in.left[i + 1] * unl + in.right[i + 1] * unr + (pl + pr) * in.normal[i] - diss[i + 1]);
        out.value[d + 1] = .5 * ((in.left[d + 1] + pl) * unl + (in.right[d + 1] + pr) * unr - diss[d + 1]);
    }

}
