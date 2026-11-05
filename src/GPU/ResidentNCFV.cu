#include "ResidentNCFV.hpp"
#include "ResidentCommon.cuh"
#include "FluxMath.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace DNDS::GPU
{
    namespace
    {
        using namespace ResidentDetail;
        struct View
        {
            ResidentNCFVSettings s;
            I owned = 0, ownedEdges = 0, total = 0, boundaryCount = 0;
            const ResidentNode *nodes;
            const ResidentRecEntry *rec;
            const ResidentVariationalEntry *variational;
            const I *variationalNodes;
            const ResidentRecoveryEntry *recovery;
            const ResidentSurfaceEntry *surface;
            const ResidentEdge *edges;
            const ResidentIncidence *incidence;
            const ResidentBoundary *boundaries;
            const ResidentBoundaryEntry *boundarySupport;
            const ResidentQuadrature *quadrature;
            double *u, *base, *gradient, *point, *rawPoint, *factor, *coefficient;
            double *fluxGradient, *edgeFlux, *edgeRadius, *rhs, *steps, *bounds;
            double *coefficientPrevious;
            const ResidentSharedTrace *sharedTraces;
            const ResidentSharedFace *sharedFaces;
            const ResidentRecoveryEntry *sharedWeights;
            const I *sharedSupport, *sharedFacesByGroup;
            double *sharedGradient, *sharedScales, *sharedRemote, *sharedAnchor;
            int *invalid;
        };
        __device__ double Pressure(const View &v, const double *q)
        {
            double kinetic = 0;
            for (int d = 0; d < v.s.dimension; ++d)
                kinetic += q[d + 1] * q[d + 1];
            return (v.s.gamma - 1) * (q[v.s.dimension + 1] - .5 * kinetic / q[0]);
        }
        __device__ bool Physical(const View &v, const double *q)
        {
            for (int k = 0; k < v.s.dimension + 2; ++k)
                if (!isfinite(q[k]))
                    return false;
            return q[0] > 1e-12 && Pressure(v, q) > 1e-12;
        }
        __device__ void Protect(const View &v, double *q, const double *anchor)
        {
            if (Physical(v, q))
                return;
            if (!Physical(v, anchor))
            {
                atomicExch(v.invalid, 1);
                return;
            }
            double low = 0, high = 1, test[5];
            for (int it = 0; it < 40; ++it)
            {
                const double mid = .5 * (low + high);
                for (int k = 0; k < 5; ++k)
                    test[k] = anchor[k] + mid * (q[k] - anchor[k]);
                if (Physical(v, test))
                    low = mid;
                else
                    high = mid;
            }
            for (int k = 0; k < 5; ++k)
                q[k] = anchor[k] + .95 * low * (q[k] - anchor[k]);
        }
        __device__ void Flux(const View &v, const double *q, const double *n, double *f)
        {
            double un = 0;
            for (int d = 0; d < v.s.dimension; ++d)
                un += q[d + 1] * n[d] / q[0];
            const double p = Pressure(v, q);
            f[0] = q[0] * un;
            for (int d = 0; d < v.s.dimension; ++d)
                f[d + 1] = q[d + 1] * un + p * n[d];
            f[v.s.dimension + 1] = (q[v.s.dimension + 1] + p) * un;
        }
        __device__ void Roe(const View &v, const double *left, const double *right, const double *normal, double *flux)
        {
            FluxInput in;
            in.dimension = v.s.dimension;
            in.gamma = v.s.gamma;
            for (int k = 0; k < 5; ++k)
            {
                in.left[k] = left[k];
                in.right[k] = right[k];
            }
            for (int d = 0; d < 3; ++d)
                in.normal[d] = normal[d];
            FluxOutput out;
            Detail::EulerFlux(in, out);
            for (int k = 0; k < 5; ++k)
                flux[k] = out.value[k];
        }
        __device__ double Speed(const View &v, const double *q, const double *n)
        {
            double un = 0;
            for (int d = 0; d < v.s.dimension; ++d)
                un += q[d + 1] * n[d] / q[0];
            return fabs(un) + sqrt(v.s.gamma * Pressure(v, q) / q[0]);
        }
        __device__ double Viscosity(const View &v, const double *q)
        {
            if (v.s.viscosityModel == 2)
                return v.s.viscosity * q[0];
            if (v.s.viscosityModel == 0)
                return v.s.viscosity;
            const double temperature = Pressure(v, q) / (q[0] * v.s.gasConstant);
            const double ratio = temperature / v.s.referenceTemperature;
            return v.s.viscosity * ratio * sqrt(ratio) * (v.s.referenceTemperature + v.s.sutherlandConstant) / (temperature + v.s.sutherlandConstant);
        }
        __device__ double Spectral(const View &v, const double *q, const double *n, double measure, double length)
        {
            double result = measure * Speed(v, q, n);
            if (v.s.viscous)
            {
                const double nu = Viscosity(v, q) / q[0];
                result += v.s.viscousSpectralFactor * fmax((4. / 3) * nu, nu / v.s.prandtl) * measure / fmax(length, 1e-100);
            }
            return result;
        }
        __device__ void Primitive(const View &v, const double *q, double *primitive)
        {
            for (int k = 0; k < 5; ++k)
                primitive[k] = q[k];
            for (int d = 0; d < v.s.dimension; ++d)
                primitive[d + 1] /= q[0];
            primitive[v.s.dimension + 1] = Pressure(v, q);
        }
        __device__ void Conservative(const View &v, const double *primitive, double *q)
        {
            double kinetic = 0;
            q[0] = primitive[0];
            for (int d = 0; d < v.s.dimension; ++d)
            {
                q[d + 1] = q[0] * primitive[d + 1];
                kinetic += .5 * q[0] * primitive[d + 1] * primitive[d + 1];
            }
            q[v.s.dimension + 1] = primitive[v.s.dimension + 1] / (v.s.gamma - 1) + kinetic;
        }
        __device__ void PrimitiveGradient(const View &v, const double *q, const double *gradient, double *result)
        {
            double v2 = 0;
            for (int d = 0; d < v.s.dimension; ++d)
                v2 += q[d + 1] * q[d + 1] / (q[0] * q[0]);
            for (int derivative = 0; derivative < v.s.dimension; ++derivative)
            {
                result[derivative] = gradient[derivative];
                double dp = gradient[(v.s.dimension + 1) * 3 + derivative] + .5 * v2 * gradient[derivative];
                for (int d = 0; d < v.s.dimension; ++d)
                {
                    const double velocity = q[d + 1] / q[0];
                    result[(d + 1) * 3 + derivative] = (gradient[(d + 1) * 3 + derivative] - velocity * gradient[derivative]) / q[0];
                    dp -= velocity * gradient[(d + 1) * 3 + derivative];
                }
                result[(v.s.dimension + 1) * 3 + derivative] = (v.s.gamma - 1) * dp;
            }
        }
        __device__ void ViscousFlux(const View &v, const double *q, const double *primitiveGradient, const double *normal, bool adiabatic, double *flux)
        {
            const int dim = v.s.dimension;
            const double mu = Viscosity(v, q), pressure = Pressure(v, q);
            double trace = 0, temperatureGradient[3]{}, normalTemperature = 0;
            for (int d = 0; d < dim; ++d)
            {
                trace += primitiveGradient[(d + 1) * 3 + d];
                temperatureGradient[d] = (q[0] * primitiveGradient[(dim + 1) * 3 + d] - pressure * primitiveGradient[d]) / (v.s.gasConstant * q[0] * q[0]);
                normalTemperature += temperatureGradient[d] * normal[d];
            }
            if (adiabatic)
                normalTemperature = 0;
            for (int component = 0; component < dim; ++component)
                for (int d = 0; d < dim; ++d)
                    flux[component + 1] += mu * (primitiveGradient[(component + 1) * 3 + d] + primitiveGradient[(d + 1) * 3 + component] - (component == d ? (2. / 3) * trace : 0)) * normal[d];
            const double cp = v.s.gasConstant * v.s.gamma / (v.s.gamma - 1);
            flux[dim + 1] = mu * cp / v.s.prandtl * normalTemperature;
            for (int d = 0; d < dim; ++d)
                flux[dim + 1] += q[d + 1] / q[0] * flux[d + 1];
        }
        __device__ void InternalViscous(const View &v, const double *l, const double *r, const double *gl, const double *gr,
                                        const double *normal, double distance, double *flux)
        {
            double pl[5]{}, pr[5]{}, primitive[5]{}, state[5]{}, gradient[15]{}, pg[15]{};
            Primitive(v, l, pl);
            Primitive(v, r, pr);
            for (int k = 0; k < 5; ++k)
                primitive[k] = .5 * (pl[k] + pr[k]);
            Conservative(v, primitive, state);
            for (int k = 0; k < v.s.dimension + 2; ++k)
                for (int d = 0; d < v.s.dimension; ++d)
                    gradient[k * 3 + d] = .5 * (gl[k * 3 + d] + gr[k * 3 + d]) + normal[d] * (r[k] - l[k]) / (2 * fmax(distance, 1e-100));
            PrimitiveGradient(v, state, gradient, pg);
            ViscousFlux(v, state, pg, normal, false, flux);
        }
        __device__ void BoundaryViscous(const View &v, const ResidentBoundary &b, const double *q, const double *gradient, double length, double *flux)
        {
            if (b.mode == 1 || b.mode == 2)
                return;
            double pg[15]{}, state[5], primitive[5];
            PrimitiveGradient(v, q, gradient, pg);
            for (int k = 0; k < 5; ++k)
                state[k] = q[k];
            if (b.mode == 3 || b.mode == 4)
            {
                Primitive(v, q, primitive);
                double wall[5];
                for (int k = 0; k < 5; ++k)
                    wall[k] = primitive[k];
                for (int d = 0; d < v.s.dimension; ++d)
                    wall[d + 1] = b.wallVelocity[d];
                if (b.mode == 4)
                    wall[v.s.dimension + 1] = wall[0] * v.s.gasConstant * b.wallTemperature;
                for (int k = 0; k < v.s.dimension + 2; ++k)
                    for (int d = 0; d < v.s.dimension; ++d)
                        pg[k * 3 + d] += b.normal[d] * (wall[k] - primitive[k]) / fmax(.5 * length, 1e-100);
                Conservative(v, wall, state);
            }
            ViscousFlux(v, state, pg, b.normal, b.mode == 3, flux);
        }
        __device__ void Polynomial(const View &v, I node, const double *point, double *state, double *gradient = nullptr)
        {
            double r[3], b[9]{}, db[27]{};
            for (int d = 0; d < 3; ++d)
            {
                double displacement = point[d] - v.nodes[node].coordinate[d];
                if (v.s.periodic[d] > 0)
                    displacement -= nearbyint(displacement / v.s.periodic[d]) * v.s.periodic[d];
                r[d] = displacement / v.nodes[node].reference[d];
            }
            const int dim = v.s.dimension, modes = dim == 2 ? 5 : 9;
            for (int d = 0; d < dim; ++d)
            {
                b[d] = r[d];
                db[d * 3 + d] = 1;
            }
            int mode = dim;
            for (int a = 0; a < dim; ++a)
                for (int bdim = a; bdim < dim; ++bdim, ++mode)
                {
                    b[mode] = (a == bdim ? .5 : 1.) * r[a] * r[bdim];
                    db[mode * 3 + a] = r[bdim];
                    if (a != bdim)
                        db[mode * 3 + bdim] = r[a];
                }
            for (int k = 0; k < 5; ++k)
            {
                state[k] = v.point[node * 5 + k];
                for (int m = 0; m < modes; ++m)
                    state[k] += v.coefficient[node * 45 + k * 9 + m] * b[m];
                if (gradient)
                    for (int d = 0; d < dim; ++d)
                    {
                        gradient[k * 3 + d] = 0;
                        for (int m = 0; m < modes; ++m)
                            gradient[k * 3 + d] += v.coefficient[node * 45 + k * 9 + m] * db[m * 3 + d] / v.nodes[node].reference[d];
                    }
            }
        }
        __device__ double SegmentPressure(const View &v, const double *mean, const double *candidate, double theta)
        {
            double q[5];
            for (int k = 0; k < 5; ++k)
                q[k] = mean[k] + theta * (candidate[k] - mean[k]);
            return Pressure(v, q);
        }
        __device__ void PressureFactor(const View &v, const double *mean, const double *candidate, double pmin, double pmax, double &factor)
        {
            if (factor == 0)
                return;
            double jump[5], m2 = 0, mdm = 0;
            for (int k = 0; k < 5; ++k)
                jump[k] = candidate[k] - mean[k];
            for (int d = 0; d < v.s.dimension; ++d)
            {
                m2 += jump[d + 1] * jump[d + 1];
                mdm += mean[d + 1] * jump[d + 1];
            }
            const double tolerance = 1e-10 * fmax(1., fmax(fabs(pmin), fabs(pmax)));
            const double lower = pmin - tolerance, upper = pmax + tolerance;
            const double quadratic = jump[0] * jump[v.s.dimension + 1] - .5 * m2;
            const double linear = mean[0] * jump[v.s.dimension + 1] + jump[0] * (mean[v.s.dimension + 1] - upper / (v.s.gamma - 1)) - mdm;
            double probe = factor;
            if (quadratic < 0 && linear > 0 && linear < -2 * quadratic * factor)
                probe = -.5 * linear / quadratic;
            if (SegmentPressure(v, mean, candidate, probe) > upper)
            {
                double lo = 0, hi = probe;
                for (int it = 0; it < 48; ++it)
                {
                    const double mid = .5 * (lo + hi);
                    if (SegmentPressure(v, mean, candidate, mid) <= upper)
                        lo = mid;
                    else
                        hi = mid;
                }
                factor = lo;
            }
            if (SegmentPressure(v, mean, candidate, factor) < lower)
            {
                double lo = 0, hi = factor;
                for (int it = 0; it < 48; ++it)
                {
                    const double mid = .5 * (lo + hi);
                    if (SegmentPressure(v, mean, candidate, mid) >= lower)
                        lo = mid;
                    else
                        hi = mid;
                }
                factor = lo;
            }
        }
        __device__ void Exterior(const View &v, const ResidentBoundary &b, const double *q, double *outside)
        {
            for (int k = 0; k < 5; ++k)
                outside[k] = q[k];
            if (b.mode == 0 || b.mode == 5)
                for (int k = 0; k < 5; ++k)
                    outside[k] = b.prescribed[k];
            else if (b.mode == 1 || b.mode == 2)
            {
                double mn = 0;
                for (int d = 0; d < v.s.dimension; ++d)
                    mn += q[d + 1] * b.normal[d];
                for (int d = 0; d < v.s.dimension; ++d)
                    outside[d + 1] -= 2 * mn * b.normal[d];
            }
            else if (b.mode == 3 || b.mode == 4)
            {
                double kinetic = 0;
                for (int d = 0; d < v.s.dimension; ++d)
                {
                    outside[d + 1] = 2 * q[0] * b.wallVelocity[d] - q[d + 1];
                    kinetic += outside[d + 1] * outside[d + 1] / (2 * q[0]);
                }
                outside[v.s.dimension + 1] = Pressure(v, q) / (v.s.gamma - 1) + kinetic;
            }
            else if (b.mode == 7)
            {
                outside[v.s.dimension + 1] += (b.pressure - Pressure(v, q)) / (v.s.gamma - 1);
            }
        }
        __device__ void LLF(const View &v, const double *l, const double *r, const double *n, double *f)
        {
            double fl[5]{}, fr[5]{};
            Flux(v, l, n, fl);
            Flux(v, r, n, fr);
            const double speed = fmax(Speed(v, l, n), Speed(v, r, n));
            for (int k = 0; k < 5; ++k)
                f[k] = .5 * (fl[k] + fr[k] - speed * (r[k] - l[k]));
        }
        __device__ void Side(const View &v, const ResidentEdge &e, int side, double *q)
        {
            const I node = e.nodes[side];
            for (int k = 0; k < 5; ++k)
            {
                double correction = 0;
                for (I j = e.begin; j < e.end; ++j)
                    for (int d = 0; d < v.s.dimension; ++d)
                        correction += v.gradient[v.surface[j].node * 15 + k * 3 + d] * v.surface[j].stateWeight[side][d];
                q[k] = v.point[node * 5 + k] + v.factor[node] * correction / e.measure;
            }
            Protect(v, q, v.point + node * 5);
        }
        __device__ void IncludeBounds(const View &v, I node, const double *q)
        {
            if (!Physical(v, q))
            {
                atomicExch(v.invalid, 1);
                return;
            }
            double *bounds = v.bounds + node * 14;
            for (int k = 0; k < v.s.dimension + 2; ++k)
            {
                bounds[k * 2] = fmin(bounds[k * 2], q[k]);
                bounds[k * 2 + 1] = fmax(bounds[k * 2 + 1], q[k]);
            }
            const double pressure = Pressure(v, q);
            bounds[10] = fmin(bounds[10], pressure);
            bounds[11] = fmin(bounds[11], pressure / pow(q[0], v.s.gamma));
        }
        __device__ void IncludeBar(const View &v, I node, const double *other, const double *normal)
        {
            const double *self = v.u + node * 5;
            double fs[5]{}, fo[5]{}, bar[5];
            const double speed = fmax(Speed(v, self, normal), Speed(v, other, normal));
            Flux(v, self, normal, fs);
            Flux(v, other, normal, fo);
            for (int k = 0; k < 5; ++k)
                bar[k] = .5 * (self[k] + other[k]) - .5 * (fo[k] - fs[k]) / speed;
            IncludeBounds(v, node, other);
            IncludeBounds(v, node, bar);
        }
        __global__ void FluxBounds(View v)
        {
            const I i = Thread();
            if (i >= v.owned)
                return;
            const double *q = v.u + i * 5;
            double *bounds = v.bounds + i * 14;
            for (int k = 0; k < 5; ++k)
                bounds[2 * k] = bounds[2 * k + 1] = q[k];
            bounds[10] = Pressure(v, q);
            bounds[11] = bounds[10] / pow(q[0], v.s.gamma);
            for (I j = v.nodes[i].edgeBegin; j < v.nodes[i].edgeEnd; ++j)
            {
                const auto &inc = v.incidence[j];
                const auto &edge = v.edges[inc.edge];
                double normal[3];
                for (int d = 0; d < 3; ++d)
                    normal[d] = inc.sign * edge.vectorMeasure[d] / edge.measure;
                IncludeBar(v, i, v.u + inc.neighbor * 5, normal);
            }
            for (I b = v.nodes[i].boundaryBegin; b < v.nodes[i].boundaryEnd; ++b)
            {
                const auto &boundary = v.boundaries[b];
                if (boundary.mode == 8)
                    continue;
                double outside[5];
                Exterior(v, boundary, q, outside);
                IncludeBar(v, i, outside, boundary.normal);
            }
        }
        __device__ bool FluxAdmissible(const View &v, const double *bar, const double *increment, double factor,
                                       double pmin, double smin, double padding)
        {
            double candidate[5];
            for (int k = 0; k < 5; ++k)
                candidate[k] = bar[k] + factor * increment[k];
            return Physical(v, candidate) && Pressure(v, candidate) >= pmin - padding &&
                   (!v.s.entropyFluxCorrection || Pressure(v, candidate) >= smin * pow(candidate[0], v.s.gamma) - padding);
        }
        __global__ void CorrectFlux(View v)
        {
            const I i = Thread();
            if (i >= v.ownedEdges)
                return;
            const auto &edge = v.edges[i];
            const double *left = v.u + edge.nodes[0] * 5, *right = v.u + edge.nodes[1] * 5;
            double normal[3], low[5]{}, fl[5]{}, fr[5]{}, correction[5], bar[5];
            for (int d = 0; d < 3; ++d)
                normal[d] = edge.vectorMeasure[d] / edge.measure;
            const double speed = fmax(Speed(v, left, normal), Speed(v, right, normal)), dissipation = speed * edge.measure;
            LLF(v, left, right, normal, low);
            Flux(v, left, normal, fl);
            Flux(v, right, normal, fr);
            for (int k = 0; k < 5; ++k)
            {
                low[k] *= edge.measure;
                correction[k] = v.edgeFlux[i * 5 + k] - low[k];
                bar[k] = .5 * (left[k] + right[k]) - .5 * (fr[k] - fl[k]) / speed;
            }
            double factor = v.s.maximumFluxCorrection;
            for (int side = 0; side < 2; ++side)
            {
                const double *bounds = v.bounds + edge.nodes[side] * 14;
                double increment[5];
                for (int k = 0; k < 5; ++k)
                    increment[k] = (side == 0 ? -1. : 1.) * correction[k] / dissipation;
                for (int k = 0; k < v.s.dimension + 2; ++k)
                {
                    const double minimum = v.s.pairFluxBounds ? fmin(fmin(left[k], right[k]), bar[k]) : bounds[k * 2];
                    const double maximum = v.s.pairFluxBounds ? fmax(fmax(left[k], right[k]), bar[k]) : bounds[k * 2 + 1];
                    const double padding = 64 * DBL_EPSILON * fmax(1., fmax(fabs(minimum), fabs(maximum)));
                    if (increment[k] > 0)
                        factor = fmin(factor, (maximum + padding - bar[k]) / increment[k]);
                    else if (increment[k] < 0)
                        factor = fmin(factor, (minimum - padding - bar[k]) / increment[k]);
                }
                factor = fmax(0., fmin(v.s.maximumFluxCorrection, factor));
                const double pmin = v.s.pairFluxBounds ? fmin(Pressure(v, bar), fmin(Pressure(v, left), Pressure(v, right))) : bounds[10];
                const double smin = v.s.pairFluxBounds ? fmin(Pressure(v, bar) / pow(bar[0], v.s.gamma),
                                                              fmin(Pressure(v, left) / pow(left[0], v.s.gamma), Pressure(v, right) / pow(right[0], v.s.gamma)))
                                                       : bounds[11];
                const double padding = 64 * DBL_EPSILON * fmax(1., fabs(pmin));
                if (!FluxAdmissible(v, bar, increment, factor, pmin, smin, padding))
                {
                    double lower = 0, upper = factor;
                    for (int it = 0; it < 48; ++it)
                    {
                        const double middle = .5 * (lower + upper);
                        if (FluxAdmissible(v, bar, increment, middle, pmin, smin, padding))
                            lower = middle;
                        else
                            upper = middle;
                    }
                    factor = lower;
                }
            }
            if (v.s.binaryFluxCorrection && factor < v.s.maximumFluxCorrection)
                factor = 0;
            for (int k = 0; k < 5; ++k)
                v.edgeFlux[i * 5 + k] = low[k] + factor * correction[k];
            v.edgeRadius[i] = dissipation;
        }
        __device__ void Bounds(const View &v, I i, double *lo, double *hi);
        __global__ void SharedAnchorBounds(View v)
        {
            const I i = Thread();
            if (i >= v.owned)
                return;
            double lo[5], hi[5];
            Bounds(v, i, lo, hi);
            double *data = v.sharedAnchor + i * 20;
            for (int k = 0; k < 5; ++k)
            {
                data[k] = v.u[i * 5 + k];
                data[5 + k] = lo[k];
                data[10 + k] = hi[k];
            }
            data[15] = data[16] = Pressure(v, v.u + i * 5);
            for (I j = v.nodes[i].edgeBegin; j < v.nodes[i].edgeEnd; ++j)
            {
                const double pressure = Pressure(v, v.u + v.incidence[j].neighbor * 5);
                data[15] = fmin(data[15], pressure);
                data[16] = fmax(data[16], pressure);
            }
        }
        __device__ double SharedTraceFactor(const View &v, const ResidentSharedTrace &trace)
        {
            double increment[5]{}, candidate[5];
            const double *data = v.sharedAnchor + trace.anchor * 20;
            for (I j = trace.begin; j < trace.end; ++j)
            {
                const auto &entry = v.sharedWeights[j];
                for (int k = 0; k < 5; ++k)
                {
                    double contribution = 0;
                    for (int d = 0; d < v.s.dimension; ++d)
                        contribution += v.sharedGradient[entry.node * 15 + k * 3 + d] * entry.weight[d];
                    increment[k] += v.sharedScales[entry.node] * contribution;
                }
            }
            double factor = 1;
            for (int k = 0; k < 5; ++k)
                candidate[k] = data[k] + increment[k];
            for (int k = 0; k < v.s.dimension + 2; ++k)
            {
                if (!isfinite(increment[k]))
                    atomicExch(v.invalid, 1);
                if (candidate[k] > data[10 + k] + 1e-10 && increment[k] > 1e-10)
                    factor = fmin(factor, (data[10 + k] - data[k]) / increment[k]);
                if (candidate[k] < data[5 + k] - 1e-10 && increment[k] < -1e-10)
                    factor = fmin(factor, (data[5 + k] - data[k]) / increment[k]);
            }
            factor = fmax(0., fmin(1., factor));
            double q[5];
            for (int k = 0; k < 5; ++k)
                q[k] = data[k] + factor * increment[k];
            if (!Physical(v, q))
            {
                double lo = 0, hi = factor;
                for (int it = 0; it < 48; ++it)
                {
                    const double middle = .5 * (lo + hi);
                    for (int k = 0; k < 5; ++k)
                        q[k] = data[k] + middle * increment[k];
                    if (Physical(v, q))
                        lo = middle;
                    else
                        hi = middle;
                }
                factor = .95 * lo;
            }
            if (v.s.limitPressure)
                PressureFactor(v, data, candidate, data[15], data[16], factor);
            return factor;
        }
        __global__ void SharedFaces(View v, I begin, I end)
        {
            const I position = begin + Thread();
            if (position >= end)
                return;
            const auto &face = v.sharedFaces[v.sharedFacesByGroup[position]];
            const double factor = fmin(SharedTraceFactor(v, v.sharedTraces[face.traces[0]]), SharedTraceFactor(v, v.sharedTraces[face.traces[1]]));
            if (factor >= 1)
                return;
            // Complete-support coloring makes these writes disjoint globally.
            for (I j = face.begin; j < face.end; ++j)
            {
                const I node = v.sharedSupport[j];
                v.sharedScales[node] *= factor;
                if (node >= v.owned)
                    v.sharedRemote[node] *= factor;
            }
        }
        __device__ bool PhysicalBudget(const View &v, const double *mean, const double *positive, const double *negative, double factor)
        {
            const double rho = mean[0] - factor * negative[0];
            if (rho <= 1e-12)
                return false;
            double momentum = 0;
            for (int d = 1; d <= v.s.dimension; ++d)
            {
                const double lo = mean[d] - factor * negative[d], hi = mean[d] + factor * positive[d];
                momentum += fmax(lo * lo, hi * hi);
            }
            return (v.s.gamma - 1) * (mean[v.s.dimension + 1] - factor * negative[v.s.dimension + 1] - .5 * momentum / rho) > 1e-12;
        }
        __global__ void SharedBudgets(View v, I count)
        {
            const I i = Thread();
            if (i >= count)
                return;
            const auto &trace = v.sharedTraces[i];
            const double *data = v.sharedAnchor + trace.anchor * 20;
            double positive[5]{}, negative[5]{};
            for (I j = trace.begin; j < trace.end; ++j)
            {
                const auto &entry = v.sharedWeights[j];
                for (int k = 0; k < v.s.dimension + 2; ++k)
                {
                    double contribution = 0;
                    for (int d = 0; d < v.s.dimension; ++d)
                        contribution += v.sharedGradient[entry.node * 15 + k * 3 + d] * entry.weight[d];
                    if (!isfinite(contribution))
                        atomicExch(v.invalid, 1);
                    positive[k] += fmax(0., contribution);
                    negative[k] += fmax(0., -contribution);
                }
            }
            double factor = 1;
            for (int k = 0; k < v.s.dimension + 2; ++k)
            {
                if (positive[k] > 1e-10)
                    factor = fmin(factor, fmax(0., data[k + 10] - data[k]) / positive[k]);
                if (negative[k] > 1e-10)
                    factor = fmin(factor, fmax(0., data[k] - data[k + 5]) / negative[k]);
            }
            if (!PhysicalBudget(v, data, positive, negative, factor))
            {
                double lo = 0, hi = factor;
                for (int it = 0; it < 48; ++it)
                {
                    const double middle = .5 * (lo + hi);
                    if (PhysicalBudget(v, data, positive, negative, middle))
                        lo = middle;
                    else
                        hi = middle;
                }
                factor = .99 * lo;
            }
            for (I j = trace.begin; j < trace.end; ++j)
                Minimum(v.sharedScales + v.sharedWeights[j].node, factor);
        }
        __global__ void ApplySharedGradient(View v)
        {
            const I i = Thread();
            if (i >= v.owned)
                return;
            for (int k = 0; k < 15; ++k)
                v.gradient[i * 15 + k] = v.sharedGradient[i * 15 + k] * v.sharedScales[i];
        }
        __global__ void ProtectSharedPoints(View v)
        {
            const I i = Thread();
            if (i < v.owned)
                Protect(v, v.point + i * 5, v.u + i * 5);
        }
        __global__ void Reconstruct(View v)
        {
            const I i = Thread();
            if (i >= v.owned)
                return;
            const auto &node = v.nodes[i];
            for (int k = 0; k < 5; ++k)
                for (int mode = 0; mode < ((v.s.traditional || v.s.variational) ? (v.s.dimension == 2 ? 5 : 9) : v.s.dimension); ++mode)
                {
                    double value = 0;
                    for (I j = node.recBegin; j < node.recEnd; ++j)
                        value += v.rec[j].weights[mode] * (v.u[v.rec[j].node * 5 + k] - v.u[i * 5 + k]);
                    if (v.s.traditional || v.s.variational)
                        v.coefficient[i * 45 + k * 9 + mode] = value;
                    else
                        v.gradient[i * 15 + k * 3 + mode] = value / node.reference[mode];
                }
            v.factor[i] = 1;
        }
        __global__ void VariationalSweep(View v, I begin, I end)
        {
            const I position = begin + Thread();
            if (position >= end)
                return;
            const I i = v.variationalNodes[position];
            const auto &node = v.nodes[i];
            const int modes = v.s.dimension == 2 ? 5 : 9;
            for (int k = 0; k < 5; ++k)
                for (int row = 0; row < modes; ++row)
                {
                    double value = 0;
                    for (I p = node.variationalBegin; p < node.variationalEnd; ++p)
                    {
                        const auto &entry = v.variational[p];
                        const double *source = entry.node < i ? v.coefficient : v.coefficientPrevious;
                        for (int col = 0; col < modes; ++col)
                            value += entry.coupling[row * 9 + col] * source[entry.node * 45 + k * 9 + col];
                        value += entry.jump[row] * (v.u[entry.node * 5 + k] - v.u[i * 5 + k]);
                    }
                    value = (1 - v.s.variationalRelaxation) * v.coefficientPrevious[i * 45 + k * 9 + row] + v.s.variationalRelaxation * value;
                    v.coefficient[i * 45 + k * 9 + row] = value;
                    if (!isfinite(value))
                        atomicExch(v.invalid, 1);
                }
        }
        __global__ void CoefficientsToGradient(View v)
        {
            const I i = Thread();
            if (i >= v.owned)
                return;
            for (int k = 0; k < 5; ++k)
                for (int d = 0; d < v.s.dimension; ++d)
                    v.gradient[i * 15 + k * 3 + d] = v.coefficient[i * 45 + k * 9 + d] / v.nodes[i].reference[d];
        }
        __global__ void Recover(View v)
        {
            const I i = Thread();
            if (i >= v.owned)
                return;
            const auto &node = v.nodes[i];
            for (int k = 0; k < 5; ++k)
            {
                double point = v.u[i * 5 + k];
                if (v.s.traditional)
                    for (int m = 0; m < (v.s.dimension == 2 ? 5 : 9); ++m)
                        point -= v.coefficient[i * 45 + k * 9 + m] * node.basisMean[m];
                else
                    for (I j = node.recoveryBegin; j < node.recoveryEnd; ++j)
                        for (int d = 0; d < v.s.dimension; ++d)
                            point -= v.gradient[v.recovery[j].node * 15 + k * 3 + d] * v.recovery[j].weight[d];
                v.rawPoint[i * 5 + k] = v.point[i * 5 + k] = point;
            }
            if (!v.s.traditional && ((v.s.limiter && !v.s.limitRecovery) || (v.s.fluxCorrection && !v.s.limiter)))
                Protect(v, v.point + i * 5, v.u + i * 5);
        }
        __device__ void Bounds(const View &v, I i, double *lo, double *hi)
        {
            for (int k = 0; k < 5; ++k)
                lo[k] = hi[k] = v.u[i * 5 + k];
            for (I j = v.nodes[i].edgeBegin; j < v.nodes[i].edgeEnd; ++j)
                for (int k = 0; k < 5; ++k)
                {
                    const double other = v.u[v.incidence[j].neighbor * 5 + k];
                    lo[k] = fmin(lo[k], other);
                    hi[k] = fmax(hi[k], other);
                }
        }
        __device__ void LimitCandidate(const View &v, const double *q, const double *anchor, const double *lo, const double *hi, double &factor)
        {
            for (int k = 0; k < v.s.dimension + 2; ++k)
            {
                if (v.s.binaryBarth)
                {
                    if (!isfinite(q[k]) || q[k] < lo[k] - 1e-10 || q[k] > hi[k] + 1e-10)
                        factor = 0;
                }
                else
                {
                    const double delta = q[k] - anchor[k];
                    if (delta > 1e-10)
                        factor = fmin(factor, (hi[k] - anchor[k]) / delta);
                    else if (delta < -1e-10)
                        factor = fmin(factor, (lo[k] - anchor[k]) / delta);
                }
            }
        }
        __device__ void LimitPhysical(const View &v, const double *q, const double *anchor, double &factor)
        {
            if (v.s.binaryBarth)
            {
                if (!Physical(v, q))
                    factor = 0;
                return;
            }
            double test[5];
            for (int k = 0; k < 5; ++k)
                test[k] = anchor[k] + factor * (q[k] - anchor[k]);
            if (Physical(v, test))
                return;
            double lo = 0, hi = factor;
            for (int it = 0; it < 40; ++it)
            {
                const double mid = .5 * (lo + hi);
                for (int k = 0; k < 5; ++k)
                    test[k] = anchor[k] + mid * (q[k] - anchor[k]);
                if (Physical(v, test))
                    lo = mid;
                else
                    hi = mid;
            }
            factor = .95 * lo;
        }
        __global__ void Limit(View v)
        {
            const I i = Thread();
            if (i >= v.owned || !v.s.limiter)
                return;
            double lo[5], hi[5], candidate[5], anchor[5];
            Bounds(v, i, lo, hi);
            for (int k = 0; k < 5; ++k)
                anchor[k] = (v.s.limitRecovery ? v.u : v.point)[i * 5 + k];
            double factor = 1;
            if (v.s.traditional)
            {
                for (I j = v.nodes[i].edgeBegin; j < v.nodes[i].edgeEnd; ++j)
                {
                    const I neighbor = v.incidence[j].neighbor;
                    double midpoint[3];
                    for (int d = 0; d < 3; ++d)
                    {
                        double dx = v.nodes[neighbor].coordinate[d] - v.nodes[i].coordinate[d];
                        if (v.s.periodic[d] > 0)
                            dx -= nearbyint(dx / v.s.periodic[d]) * v.s.periodic[d];
                        midpoint[d] = v.nodes[i].coordinate[d] + .5 * dx;
                    }
                    Polynomial(v, i, midpoint, candidate);
                    for (int k = 0; k < v.s.dimension + 2; ++k)
                    {
                        const double delta = candidate[k] - v.u[i * 5 + k];
                        if (delta > 1e-100)
                            factor = fmin(factor, (hi[k] - v.u[i * 5 + k]) / delta);
                        else if (delta < -1e-100)
                            factor = fmin(factor, (lo[k] - v.u[i * 5 + k]) / delta);
                    }
                }
                v.factor[i] = fmax(0., fmin(1., factor));
                return;
            }
            if (v.s.limitRecovery)
                LimitCandidate(v, v.rawPoint + i * 5, anchor, lo, hi, factor);
            for (I j = v.nodes[i].edgeBegin; j < v.nodes[i].edgeEnd; ++j)
            {
                const auto incidence = v.incidence[j];
                const auto &edge = v.edges[incidence.edge];
                for (int k = 0; k < 5; ++k)
                {
                    double correction = 0;
                    for (I p = edge.begin; p < edge.end; ++p)
                        for (int d = 0; d < v.s.dimension; ++d)
                            correction += v.gradient[v.surface[p].node * 15 + k * 3 + d] * v.surface[p].stateWeight[incidence.side][d];
                    candidate[k] = v.point[i * 5 + k] + correction / edge.measure;
                }
                double lower[5], upper[5];
                for (int k = 0; k < 5; ++k)
                {
                    const double self = (v.s.limitRecovery ? v.u : v.point)[i * 5 + k];
                    const double other = (v.s.limitRecovery ? v.u : v.point)[incidence.neighbor * 5 + k];
                    lower[k] = v.s.limitRecovery && !v.s.directionalBounds ? lo[k] : fmin(self, other);
                    upper[k] = v.s.limitRecovery && !v.s.directionalBounds ? hi[k] : fmax(self, other);
                }
                LimitCandidate(v, candidate, anchor, lower, upper, factor);
            }
            if (v.s.binaryBarth)
                for (I b = v.nodes[i].boundaryBegin; b < v.nodes[i].boundaryEnd; ++b)
                {
                    const auto &piece = v.boundaries[b];
                    if (piece.mode == 8)
                        continue;
                    for (int k = 0; k < 5; ++k)
                    {
                        candidate[k] = 0;
                        for (I p = piece.begin; p < piece.end; ++p)
                            candidate[k] += v.boundarySupport[p].weight * v.point[v.boundarySupport[p].node * 5 + k] / piece.measure;
                    }
                    LimitCandidate(v, candidate, anchor, lo, hi, factor);
                }
            factor = fmax(0., fmin(1., factor));
            if (v.s.limitRecovery)
            {
                double pmin = Pressure(v, anchor), pmax = pmin;
                if (v.s.limitPressure)
                    for (I j = v.nodes[i].edgeBegin; j < v.nodes[i].edgeEnd; ++j)
                    {
                        const double pressure = Pressure(v, v.u + v.incidence[j].neighbor * 5);
                        pmin = fmin(pmin, pressure);
                        pmax = fmax(pmax, pressure);
                    }
                LimitPhysical(v, v.rawPoint + i * 5, anchor, factor);
                if (v.s.limitPressure)
                    PressureFactor(v, anchor, v.rawPoint + i * 5, pmin, pmax, factor);
                for (I j = v.nodes[i].edgeBegin; j < v.nodes[i].edgeEnd; ++j)
                {
                    const auto incidence = v.incidence[j];
                    const auto &edge = v.edges[incidence.edge];
                    for (int k = 0; k < 5; ++k)
                    {
                        candidate[k] = v.rawPoint[i * 5 + k];
                        for (I p = edge.begin; p < edge.end; ++p)
                            for (int d = 0; d < v.s.dimension; ++d)
                                candidate[k] += v.gradient[v.surface[p].node * 15 + k * 3 + d] * v.surface[p].stateWeight[incidence.side][d] / edge.measure;
                    }
                    LimitPhysical(v, candidate, anchor, factor);
                    if (v.s.limitPressure)
                        PressureFactor(v, anchor, candidate, pmin, pmax, factor);
                }
                if (v.s.binaryBarth)
                    factor = factor < 1 ? 0 : 1;
            }
            v.factor[i] = factor;
        }
        __global__ void ApplyRecoveryLimit(View v)
        {
            const I i = Thread();
            if (i >= v.owned || !v.s.limiter)
                return;
            if (v.s.traditional)
            {
                for (int k = 0; k < 5; ++k)
                    for (int m = 0; m < (v.s.dimension == 2 ? 5 : 9); ++m)
                        v.coefficient[i * 45 + k * 9 + m] *= v.factor[i];
                return;
            }
            if (!v.s.limitRecovery)
                return;
            for (int k = 0; k < 5; ++k)
                v.point[i * 5 + k] = v.u[i * 5 + k] + v.factor[i] * (v.rawPoint[i * 5 + k] - v.u[i * 5 + k]);
        }
        __global__ void FluxGradients(View v)
        {
            const I i = Thread();
            if (i >= v.owned)
                return;
            const double *q = v.point + i * 5, *g = v.gradient + i * 15;
            double vel[3]{}, v2 = 0;
            for (int d = 0; d < v.s.dimension; ++d)
            {
                vel[d] = q[d + 1] / q[0];
                v2 += vel[d] * vel[d];
            }
            for (int derivative = 0; derivative < v.s.dimension; ++derivative)
            {
                double dp = g[(v.s.dimension + 1) * 3 + derivative] + .5 * v2 * g[derivative], dv[3]{};
                for (int d = 0; d < v.s.dimension; ++d)
                {
                    dp -= vel[d] * g[(d + 1) * 3 + derivative];
                    dv[d] = (g[(d + 1) * 3 + derivative] - vel[d] * g[derivative]) / q[0];
                }
                dp *= v.s.gamma - 1;
                for (int direction = 0; direction < v.s.dimension; ++direction)
                {
                    v.fluxGradient[i * 45 + derivative * 3 + direction] = g[(direction + 1) * 3 + derivative];
                    for (int component = 0; component < v.s.dimension; ++component)
                        v.fluxGradient[i * 45 + (component + 1) * 9 + derivative * 3 + direction] =
                            g[(component + 1) * 3 + derivative] * vel[direction] + q[component + 1] * dv[direction] + (component == direction ? dp : 0);
                    v.fluxGradient[i * 45 + (v.s.dimension + 1) * 9 + derivative * 3 + direction] =
                        (g[(v.s.dimension + 1) * 3 + derivative] + dp) * vel[direction] + (q[v.s.dimension + 1] + Pressure(v, q)) * dv[direction];
                }
            }
        }
        __global__ void EdgeFlux(View v)
        {
            const I i = Thread();
            if (i >= v.ownedEdges)
                return;
            const auto &e = v.edges[i];
            double n[3], norm = 0;
            for (int d = 0; d < 3; ++d)
                norm += e.vectorMeasure[d] * e.vectorMeasure[d];
            norm = sqrt(norm);
            for (int d = 0; d < 3; ++d)
                n[d] = e.vectorMeasure[d] / norm;
            double l[5], r[5], fl[5]{}, fr[5]{}, rl[5]{}, rr[5]{}, numerical[5]{};
            const double length = fmin(v.nodes[e.nodes[0]].volume, v.nodes[e.nodes[1]].volume) / e.measure;
            if (v.s.traditional)
            {
                double total[5]{};
                for (I p = e.quadBegin; p < e.quadEnd; ++p)
                {
                    const auto &q = v.quadrature[p];
                    double gl[15]{}, gr[15]{}, flux[5]{}, viscous[5]{};
                    Polynomial(v, e.nodes[0], q.coordinate, l, gl);
                    Polynomial(v, e.nodes[1], q.coordinate, r, gr);
                    Protect(v, l, v.point + e.nodes[0] * 5);
                    Protect(v, r, v.point + e.nodes[1] * 5);
                    Roe(v, l, r, q.normal, flux);
                    if (v.s.viscous)
                        InternalViscous(v, l, r, gl, gr, q.normal, length, viscous);
                    for (int k = 0; k < 5; ++k)
                        total[k] += q.weight * (flux[k] - viscous[k]);
                }
                for (int k = 0; k < 5; ++k)
                    v.edgeFlux[i * 5 + k] = total[k];
            }
            else
            {
                Side(v, e, 0, l);
                Side(v, e, 1, r);
                Flux(v, v.point + e.nodes[0] * 5, e.vectorMeasure, fl);
                Flux(v, v.point + e.nodes[1] * 5, e.vectorMeasure, fr);
                for (I p = e.begin; p < e.end; ++p)
                    for (int k = 0; k < 5; ++k)
                        for (int derivative = 0; derivative < v.s.dimension; ++derivative)
                            for (int direction = 0; direction < v.s.dimension; ++direction)
                            {
                                const double value = v.fluxGradient[v.surface[p].node * 45 + k * 9 + derivative * 3 + direction];
                                fl[k] += v.factor[e.nodes[0]] * value * v.surface[p].fluxWeight[0][derivative * 3 + direction];
                                fr[k] += v.factor[e.nodes[1]] * value * v.surface[p].fluxWeight[1][derivative * 3 + direction];
                            }
                Roe(v, l, r, n, numerical);
                Flux(v, l, n, rl);
                Flux(v, r, n, rr);
                for (int k = 0; k < 5; ++k)
                    v.edgeFlux[i * 5 + k] = .5 * (fl[k] + fr[k]) + e.measure * (numerical[k] - .5 * (rl[k] + rr[k]));
                if (v.s.viscous)
                {
                    double gl[15]{}, gr[15]{}, viscous[5]{};
                    for (I p = e.begin; p < e.end; ++p)
                        for (int k = 0; k < 5; ++k)
                            for (int d = 0; d < v.s.dimension; ++d)
                            {
                                const double value = v.surface[p].gradientWeight * v.gradient[v.surface[p].node * 15 + k * 3 + d] / e.measure;
                                gl[k * 3 + d] += v.factor[e.nodes[0]] * value;
                                gr[k * 3 + d] += v.factor[e.nodes[1]] * value;
                            }
                    InternalViscous(v, l, r, gl, gr, n, length, viscous);
                    for (int k = 0; k < 5; ++k)
                        v.edgeFlux[i * 5 + k] -= e.measure * viscous[k];
                }
            }
            double mean[5];
            for (int k = 0; k < 5; ++k)
                mean[k] = .5 * (v.point[e.nodes[0] * 5 + k] + v.point[e.nodes[1] * 5 + k]);
            Protect(v, mean, v.point + e.nodes[0] * 5);
            const double viscousLength = fmin(pow(v.nodes[e.nodes[0]].volume, 1. / v.s.dimension), pow(v.nodes[e.nodes[1]].volume, 1. / v.s.dimension));
            v.edgeRadius[i] = Spectral(v, mean, n, e.measure, viscousLength);
        }
        __device__ void BoundaryFlux(const View &v, const ResidentBoundary &b, double *f)
        {
            if (b.mode == 8)
                return;
            if (v.s.fluxCorrection)
            {
                double outside[5], flux[5]{};
                Exterior(v, b, v.u + b.anchor * 5, outside);
                LLF(v, v.u + b.anchor * 5, outside, b.normal, flux);
                for (int k = 0; k < 5; ++k)
                    f[k] += b.measure * flux[k];
                return;
            }
            if (v.s.traditional)
            {
                for (I p = b.quadBegin; p < b.quadEnd; ++p)
                {
                    const auto &q = v.quadrature[p];
                    double inside[5], outside[5], gradient[15]{}, flux[5]{}, viscous[5]{};
                    Polynomial(v, b.anchor, q.coordinate, inside, gradient);
                    Protect(v, inside, v.point + b.anchor * 5);
                    auto local = b;
                    for (int d = 0; d < 3; ++d)
                        local.normal[d] = q.normal[d];
                    Exterior(v, local, inside, outside);
                    Roe(v, inside, outside, q.normal, flux);
                    if (v.s.viscous)
                        BoundaryViscous(v, local, inside, gradient, pow(v.nodes[b.anchor].volume, 1. / v.s.dimension), viscous);
                    for (int k = 0; k < 5; ++k)
                        f[k] += q.weight * (flux[k] - viscous[k]);
                }
                return;
            }
            double outside[5], candidate[5]{}, lo[5], hi[5];
            Bounds(v, b.anchor, lo, hi);
            bool constant = v.s.binaryBarth && v.factor[b.anchor] == 0;
            for (I p = b.begin; p < b.end; ++p)
                for (int k = 0; k < 5; ++k)
                    candidate[k] += v.boundarySupport[p].weight * v.point[v.boundarySupport[p].node * 5 + k] / b.measure;
            if (v.s.binaryBarth)
            {
                if (!Physical(v, candidate))
                    constant = true;
                for (int k = 0; k < v.s.dimension + 2; ++k)
                    if (candidate[k] < lo[k] - 1e-10 || candidate[k] > hi[k] + 1e-10)
                        constant = true;
            }
            if (constant)
            {
                Exterior(v, b, v.u + b.anchor * 5, outside);
                double flux[5]{};
                Roe(v, v.u + b.anchor * 5, outside, b.normal, flux);
                for (int k = 0; k < 5; ++k)
                    f[k] += b.measure * flux[k];
            }
            else
            {
                for (I p = b.begin; p < b.end; ++p)
                {
                    const double *inside = v.point + v.boundarySupport[p].node * 5;
                    Exterior(v, b, inside, outside);
                    double flux[5]{};
                    Roe(v, inside, outside, b.normal, flux);
                    for (int k = 0; k < 5; ++k)
                        f[k] += v.boundarySupport[p].weight * flux[k];
                }
                if (v.s.viscous)
                {
                    Protect(v, candidate, v.point + b.anchor * 5);
                    double gradient[15]{}, viscous[5]{};
                    for (I p = b.begin; p < b.end; ++p)
                        for (int k = 0; k < 5; ++k)
                            for (int d = 0; d < v.s.dimension; ++d)
                                gradient[k * 3 + d] += v.factor[b.anchor] * v.boundarySupport[p].weight * v.gradient[v.boundarySupport[p].node * 15 + k * 3 + d] / b.measure;
                    BoundaryViscous(v, b, candidate, gradient, pow(v.nodes[b.anchor].volume, 1. / v.s.dimension), viscous);
                    for (int k = 0; k < 5; ++k)
                        f[k] -= b.measure * viscous[k];
                }
            }
        }
        __global__ void Assemble(View v, bool updateStep, double maximumStep)
        {
            const I i = Thread();
            if (i >= v.owned)
                return;
            double residual[5]{}, radius = 0;
            for (I b = v.nodes[i].boundaryBegin; b < v.nodes[i].boundaryEnd; ++b)
            {
                const auto &boundary = v.boundaries[b];
                BoundaryFlux(v, boundary, residual);
                if (boundary.mode != 8)
                {
                    if (v.s.fluxCorrection)
                    {
                        double outside[5];
                        Exterior(v, boundary, v.u + i * 5, outside);
                        radius += boundary.measure * fmax(Speed(v, v.u + i * 5, boundary.normal), Speed(v, outside, boundary.normal));
                    }
                    else
                        radius += Spectral(v, v.point + i * 5, boundary.normal, boundary.measure, pow(v.nodes[i].volume, 1. / v.s.dimension));
                }
            }
            for (I j = v.nodes[i].edgeBegin; j < v.nodes[i].edgeEnd; ++j)
            {
                const auto inc = v.incidence[j];
                radius += v.edgeRadius[inc.edge];
                for (int k = 0; k < 5; ++k)
                    residual[k] += inc.sign * v.edgeFlux[inc.edge * 5 + k];
            }
            for (int k = 0; k < 5; ++k)
                v.rhs[i * 5 + k] = -residual[k] / v.nodes[i].volume;
            if (updateStep)
            {
                double dt = v.s.cflStep ? v.s.cfl * v.nodes[i].volume / fmax(radius, 1e-100) : v.s.fixedStep;
                if (v.s.cflStep)
                    dt = fmin(v.s.maximumStep, fmax(v.s.minimumStep, dt));
                v.steps[i] = fmin(dt, maximumStep);
            }
            if (v.s.fluxCorrection)
                v.bounds[i * 14 + 12] = radius;
        }
        __global__ void CheckCFL(View v)
        {
            const I i = Thread();
            if (i >= v.owned || !v.s.fluxCorrection)
                return;
            if (v.steps[i] * v.bounds[i * 14 + 12] / v.nodes[i].volume > 1 + 1e-12)
                atomicExch(v.invalid, 2);
        }
        __global__ void SetStep(View v, double dt)
        {
            const I i = Thread();
            if (i < v.owned)
                v.steps[i] = dt;
        }
        __global__ void Update(View v, int stage)
        {
            const I i = Thread();
            if (i >= v.owned)
                return;
            for (int k = 0; k < 5; ++k)
            {
                const I p = i * 5 + k;
                const double next = v.u[p] + v.steps[i] * v.rhs[p];
                v.u[p] = stage == 0 ? next : stage == 1 ? .75 * v.base[p] + .25 * next
                                                        : (1. / 3) * v.base[p] + (2. / 3) * next;
            }
            int selected = -1, priority = 0;
            for (I b = v.nodes[i].boundaryBegin; b < v.nodes[i].boundaryEnd; ++b)
            {
                const auto &boundary = v.boundaries[b];
                if (!boundary.strong)
                    continue;
                const int candidate = boundary.mode == 5 ? 2 : boundary.mode == 3 || boundary.mode == 4 ? 1
                                                                                                        : 0;
                if (candidate > priority || (candidate && candidate == priority && (selected < 0 || boundary.zone < v.boundaries[selected].zone)))
                {
                    priority = candidate;
                    selected = int(b);
                }
            }
            if (selected >= 0)
            {
                const auto &b = v.boundaries[selected];
                if (b.mode == 5)
                    for (int k = 0; k < 5; ++k)
                        v.u[i * 5 + k] = b.prescribed[k];
                else
                {
                    double *q = v.u + i * 5;
                    const double pressure = b.mode == 4 ? q[0] * v.s.gasConstant * b.wallTemperature : Pressure(v, q);
                    double kinetic = 0;
                    for (int d = 0; d < v.s.dimension; ++d)
                    {
                        q[d + 1] = q[0] * b.wallVelocity[d];
                        kinetic += .5 * q[0] * b.wallVelocity[d] * b.wallVelocity[d];
                    }
                    q[v.s.dimension + 1] = pressure / (v.s.gamma - 1) + kinetic;
                }
            }
            if (!Physical(v, v.u + i * 5))
                atomicExch(v.invalid, 1);
        }
        __global__ void Reduce(View v, double *scalars)
        {
            __shared__ double minimum[128], maximum[128], sums[128];
            double low = DBL_MAX, high = 0, norm = 0;
            for (I i = Thread(); i < v.owned; i += I(gridDim.x) * blockDim.x)
            {
                low = fmin(low, v.steps[i]);
                high = fmax(high, v.steps[i]);
                for (int k = 0; k < v.s.dimension + 2; ++k)
                    norm += v.nodes[i].volume * v.rhs[i * 5 + k] * v.rhs[i * 5 + k];
            }
            const int lane = threadIdx.x;
            minimum[lane] = low;
            maximum[lane] = high;
            sums[lane] = norm;
            __syncthreads();
            for (int offset = 64; offset > 0; offset /= 2)
            {
                if (lane < offset)
                {
                    minimum[lane] = fmin(minimum[lane], minimum[lane + offset]);
                    maximum[lane] = fmax(maximum[lane], maximum[lane + offset]);
                    sums[lane] += sums[lane + offset];
                }
                __syncthreads();
            }
            if (lane == 0)
            {
                Minimum(scalars, minimum[0]);
                Maximum(scalars + 1, maximum[0]);
                atomicAdd(scalars + 2, sums[0]);
                Maximum(scalars + 3, double(*v.invalid));
            }
        }
    }

    struct ResidentNCFV::Impl
    {
        View v{};
        MPI_Comm comm;
        Buffer<ResidentNode> nodes;
        Buffer<ResidentRecEntry> rec;
        Buffer<ResidentRecoveryEntry> recovery;
        Buffer<ResidentVariationalEntry> variational;
        Buffer<I> variationalNodes;
        std::vector<I> variationalLevels;
        Buffer<ResidentSharedTrace> sharedTraces;
        Buffer<ResidentSharedFace> sharedFaces;
        Buffer<ResidentRecoveryEntry> sharedWeights;
        Buffer<I> sharedSupport, sharedFacesByGroup;
        std::vector<I> sharedGroups;
        Buffer<double> sharedGradient, sharedScales, sharedRemote, sharedAnchor;
        Buffer<ResidentSurfaceEntry> surface;
        Buffer<ResidentEdge> edges;
        Buffer<ResidentIncidence> incidence;
        Buffer<ResidentBoundary> boundaries;
        Buffer<ResidentBoundaryEntry> boundarySupport;
        Buffer<ResidentQuadrature> quadrature;
        Buffer<double> u, base, gradient, point, rawPoint, factor, coefficient, fluxGradient, edgeFlux, edgeRadius, rhs, steps, bounds, scalars;
        Buffer<int> invalid;
        Buffer<double> coefficientPrevious;
        std::unique_ptr<Halo> nodeHalo, edgeHalo;
        std::unique_ptr<Halo> sharedHalo;
        std::uint64_t advances = 0, residuals = 0, outputBytes = 0, scalarBytes = 0;
        bool inStep = false;
        double globalVolume = 0;

        Impl(const ResidentNCFVData &data, MPI_Comm communicator) : comm(communicator)
        {
            v.s = data.settings;
            v.owned = data.ownedNodes;
            v.ownedEdges = data.ownedEdges;
            v.total = data.nodes.size();
            v.boundaryCount = data.boundaries.size();
            double volume = 0;
            for (I i = 0; i < data.ownedNodes; ++i)
                volume += data.nodes[i].volume;
            MPIcheck(MPI_Allreduce(&volume, &globalVolume, 1, MPI_DOUBLE, MPI_SUM, comm));
            nodes.Upload(data.nodes);
            rec.Upload(data.reconstruction);
            recovery.Upload(data.recovery);
            surface.Upload(data.surface);
            edges.Upload(data.edges);
            variational.Upload(data.variational);
            variationalNodes.Upload(data.variationalNodes);
            variationalLevels = data.variationalLevels;
            incidence.Upload(data.incidence);
            boundaries.Upload(data.boundaries);
            boundarySupport.Upload(data.boundarySupport);
            quadrature.Upload(data.quadrature);
            u.Upload(data.initialState);
            base.Resize(v.total * 5);
            gradient.Resize(v.total * 15);
            point.Resize(v.total * 5);
            rawPoint.Resize(v.total * 5);
            factor.Resize(v.total);
            coefficient.Resize(v.total * 45);
            fluxGradient.Resize(v.total * 45);
            coefficientPrevious.Resize(v.s.variational ? v.total * 45 : 0);
            edgeFlux.Resize(data.edges.size() * 5);
            edgeRadius.Resize(data.edges.size());
            rhs.Resize(v.total * 5);
            steps.Resize(v.total);
            bounds.Resize(v.total * 14);
            scalars.Resize(4);
            invalid.Resize(1);
            Check(cudaMemset(gradient.data, 0, gradient.count * sizeof(double)), "initialize device gradients");
            Check(cudaMemset(coefficient.data, 0, coefficient.count * sizeof(double)), "initialize device coefficients");
            Check(cudaMemset(fluxGradient.data, 0, fluxGradient.count * sizeof(double)), "initialize device flux derivatives");
            Check(cudaMemset(rhs.data, 0, rhs.count * sizeof(double)), "initialize device residual");
            Check(cudaMemset(invalid.data, 0, sizeof(int)), "initialize device status");
            v.nodes = nodes.data;
            v.rec = rec.data;
            v.recovery = recovery.data;
            v.surface = surface.data;
            v.edges = edges.data;
            v.incidence = incidence.data;
            v.variational = variational.data;
            v.variationalNodes = variationalNodes.data;
            v.coefficientPrevious = coefficientPrevious.data;
            v.boundaries = boundaries.data;
            v.boundarySupport = boundarySupport.data;
            v.quadrature = quadrature.data;
            v.u = u.data;
            v.base = base.data;
            v.gradient = gradient.data;
            v.point = point.data;
            v.rawPoint = rawPoint.data;
            v.factor = factor.data;
            v.coefficient = coefficient.data;
            v.fluxGradient = fluxGradient.data;
            v.edgeFlux = edgeFlux.data;
            v.edgeRadius = edgeRadius.data;
            v.rhs = rhs.data;
            v.steps = steps.data;
            v.bounds = bounds.data;
            v.invalid = invalid.data;
            nodeHalo = std::make_unique<Halo>(data.nodeGlobals, data.ownedNodes, comm);
            edgeHalo = std::make_unique<Halo>(data.edgeGlobals, data.ownedEdges, comm);
            if (v.s.sharedGradient)
            {
                sharedTraces.Upload(data.sharedTraces);
                sharedFaces.Upload(data.sharedFaces);
                sharedWeights.Upload(data.sharedWeights);
                sharedSupport.Upload(data.sharedSupport);
                sharedFacesByGroup.Upload(data.sharedFacesByGroup);
                sharedGroups = data.sharedGroups;
                sharedGradient.Resize(data.sharedGlobals.size() * 15);
                sharedScales.Resize(data.sharedGlobals.size());
                sharedRemote.Resize(data.sharedGlobals.size());
                sharedAnchor.Resize(data.sharedGlobals.size() * 20);
                Check(cudaMemset(sharedAnchor.data, 0, sharedAnchor.count * sizeof(double)), "initialize shared device bounds");
                v.sharedTraces = sharedTraces.data;
                v.sharedFaces = sharedFaces.data;
                v.sharedWeights = sharedWeights.data;
                v.sharedSupport = sharedSupport.data;
                v.sharedFacesByGroup = sharedFacesByGroup.data;
                v.sharedGradient = sharedGradient.data;
                v.sharedScales = sharedScales.data;
                v.sharedRemote = sharedRemote.data;
                v.sharedAnchor = sharedAnchor.data;
                sharedHalo = std::make_unique<Halo>(data.sharedGlobals, data.ownedNodes, comm);
            }
            Check(cudaDeviceSynchronize(), "finish NCFV resident initialization");
        }
        void SharedLimit()
        {
            if (v.owned)
            {
                Check(cudaMemcpy(sharedGradient.data, gradient.data, v.owned * 15 * sizeof(double), cudaMemcpyDeviceToDevice), "freeze device shared gradients");
                SharedAnchorBounds<<<(v.owned + 127) / 128, 128>>>(v);
            }
            sharedHalo->Pull(sharedGradient.data, 15);
            sharedHalo->Pull(sharedAnchor.data, 20);
            if (sharedScales.count)
                Ones<<<(sharedScales.count + 127) / 128, 128>>>(sharedScales.data, sharedScales.count);
            if (v.s.sharedBudget)
            {
                if (sharedTraces.count)
                    SharedBudgets<<<(sharedTraces.count + 127) / 128, 128>>>(v, sharedTraces.count);
                sharedHalo->PushFactors(sharedScales.data, sharedScales.data, true);
            }
            else if (v.s.sharedParallel)
            {
                for (std::size_t color = 1; color < sharedGroups.size(); ++color)
                {
                    if (sharedRemote.count)
                        Ones<<<(sharedRemote.count + 127) / 128, 128>>>(sharedRemote.data, sharedRemote.count);
                    const I begin = sharedGroups[color - 1], end = sharedGroups[color];
                    if (end > begin)
                        SharedFaces<<<(end - begin + 127) / 128, 128>>>(v, begin, end);
                    sharedHalo->PushFactors(sharedScales.data, sharedRemote.data, false);
                    sharedHalo->Pull(sharedScales.data, 1);
                }
            }
            else
            {
                int ranks = 0, rank = 0;
                MPI_Comm_size(comm, &ranks);
                MPI_Comm_rank(comm, &rank);
                for (int activeRank = 0; activeRank < ranks; ++activeRank)
                {
                    if (sharedRemote.count)
                        Ones<<<(sharedRemote.count + 127) / 128, 128>>>(sharedRemote.data, sharedRemote.count);
                    if (rank == activeRank)
                        for (std::size_t level = 1; level < sharedGroups.size(); ++level)
                        {
                            const I begin = sharedGroups[level - 1], end = sharedGroups[level];
                            if (end > begin)
                                SharedFaces<<<(end - begin + 127) / 128, 128>>>(v, begin, end);
                        }
                    sharedHalo->PushFactors(sharedScales.data, sharedRemote.data, false);
                    sharedHalo->Pull(sharedScales.data, 1);
                }
            }
            sharedHalo->Pull(sharedScales.data, 1);
            if (v.owned)
                ApplySharedGradient<<<(v.owned + 127) / 128, 128>>>(v);
            nodeHalo->Pull(gradient.data, 15);
        }
        void RHS(bool updateStep, double maximumStep)
        {
            const auto blocks = (v.owned + 127) / 128;
            nodeHalo->Pull(u.data, 5);
            if (v.owned)
                Reconstruct<<<blocks, 128>>>(v);
            if (v.s.variational)
            {
                for (int iteration = 0; iteration < v.s.variationalIterations; ++iteration)
                {
                    nodeHalo->Pull(coefficient.data, 45);
                    Check(cudaMemcpy(coefficientPrevious.data, coefficient.data, coefficient.count * sizeof(double), cudaMemcpyDeviceToDevice), "save device variational sweep");
                    for (std::size_t level = 1; level < variationalLevels.size(); ++level)
                    {
                        const I begin = variationalLevels[level - 1], end = variationalLevels[level];
                        if (end > begin)
                            VariationalSweep<<<(end - begin + 127) / 128, 128>>>(v, begin, end);
                    }
                }
                if (!v.s.traditional && v.owned)
                    CoefficientsToGradient<<<blocks, 128>>>(v);
            }
            if (v.s.traditional)
                nodeHalo->Pull(coefficient.data, 45);
            else
                nodeHalo->Pull(gradient.data, 15);
            if (v.s.sharedGradient)
                SharedLimit();
            if (v.owned)
                Recover<<<blocks, 128>>>(v);
            nodeHalo->Pull(point.data, 5);
            if (v.s.sharedGradient)
            {
                if (v.owned)
                    ProtectSharedPoints<<<blocks, 128>>>(v);
            }
            else
            {
                if (v.owned)
                    Limit<<<blocks, 128>>>(v);
                if (v.owned)
                    ApplyRecoveryLimit<<<blocks, 128>>>(v);
            }
            if (v.s.traditional)
            {
                nodeHalo->Pull(coefficient.data, 45);
                if (v.owned)
                    Recover<<<blocks, 128>>>(v);
            }
            nodeHalo->Pull(point.data, 5);
            nodeHalo->Pull(factor.data, 1);
            if (v.s.fluxCorrection)
            {
                if (v.owned)
                    FluxBounds<<<blocks, 128>>>(v);
                nodeHalo->Pull(bounds.data, 14);
            }
            if (!v.s.traditional)
            {
                if (v.owned)
                    FluxGradients<<<blocks, 128>>>(v);
                nodeHalo->Pull(fluxGradient.data, 45);
            }
            if (v.ownedEdges)
                EdgeFlux<<<(v.ownedEdges + 127) / 128, 128>>>(v);
            if (v.s.fluxCorrection && v.ownedEdges)
                CorrectFlux<<<(v.ownedEdges + 127) / 128, 128>>>(v);
            edgeHalo->Pull(edgeFlux.data, 5);
            edgeHalo->Pull(edgeRadius.data, 1);
            if (v.owned)
                Assemble<<<blocks, 128>>>(v, updateStep, maximumStep);
            if (updateStep && v.s.cflStep && !v.s.localStep)
            {
                const auto report = Reduction();
                if (v.owned)
                    SetStep<<<blocks, 128>>>(v, report.minimumStep);
            }
            if (v.owned)
                CheckCFL<<<blocks, 128>>>(v);
            Check(cudaGetLastError(), "resident NCFV RHS kernels");
            ++residuals;
        }
        ResidentStepReport Reduction()
        {
            const std::array<double, 4> initial{DBL_MAX, 0, 0, 0};
            Check(cudaMemcpy(scalars.data, initial.data(), sizeof(initial), cudaMemcpyHostToDevice), "initialize scalar reduction");
            Reduce<<<std::max<I>(1, std::min<I>(1024, (v.owned + 127) / 128)), 128>>>(v, scalars.data);
            std::array<double, 4> local, global;
            Check(cudaMemcpy(local.data(), scalars.data, sizeof(local), cudaMemcpyDeviceToHost), "read device control scalars");
            scalarBytes += sizeof(local);
            MPIcheck(MPI_Allreduce(local.data(), global.data(), 1, MPI_DOUBLE, MPI_MIN, comm));
            MPIcheck(MPI_Allreduce(local.data() + 1, global.data() + 1, 1, MPI_DOUBLE, MPI_MAX, comm));
            MPIcheck(MPI_Allreduce(local.data() + 2, global.data() + 2, 1, MPI_DOUBLE, MPI_SUM, comm));
            MPIcheck(MPI_Allreduce(local.data() + 3, global.data() + 3, 1, MPI_DOUBLE, MPI_MAX, comm));
            if (global[3] >= 2)
                throw std::runtime_error("Resident NCFV flux correction violates the global convex CFL condition");
            if (global[3] != 0)
                throw std::runtime_error("Resident NCFV encountered a nonphysical device state");
            return {sqrt(global[2] / fmax(1e-100, globalVolume)), global[0], global[1], 0};
        }
    };
    ResidentNCFV::ResidentNCFV(const ResidentNCFVData &data, MPI_Comm communicator) : _impl(std::make_unique<Impl>(data, communicator)) {}
    ResidentNCFV::~ResidentNCFV() = default;
    ResidentStepReport ResidentNCFV::Advance(double maximumStep)
    {
        Range range("NCFV numerical step");
        auto &p = *_impl;
        p.inStep = true;
        struct Guard
        {
            bool &flag;
            ~Guard() { flag = false; }
        } guard{p.inStep};
        Check(cudaMemcpy(p.base.data, p.u.data, p.u.count * sizeof(double), cudaMemcpyDeviceToDevice), "save resident RK history");
        ResidentStepReport report;
        for (int retry = 0;; ++retry)
        {
            Check(cudaMemset(p.invalid.data, 0, sizeof(int)), "reset resident device status");
            if (retry)
                Check(cudaMemcpy(p.u.data, p.base.data, p.u.count * sizeof(double), cudaMemcpyDeviceToDevice), "restore rejected resident step");
            bool healthy = true;
            for (int stage = 0; stage < 3; ++stage)
            {
                p.RHS(stage == 0, maximumStep);
                if (stage == 0)
                    report = p.Reduction();
                if (stage == 2)
                    report.residual = p.Reduction().residual;
                if (p.v.owned)
                    Update<<<(p.v.owned + 127) / 128, 128>>>(p.v, stage);
                int localStatus = 0, globalStatus = 0;
                Check(cudaMemcpy(&localStatus, p.invalid.data, sizeof(int), cudaMemcpyDeviceToHost), "read resident stage status");
                p.scalarBytes += sizeof(int);
                MPIcheck(MPI_Allreduce(&localStatus, &globalStatus, 1, MPI_INT, MPI_MAX, p.comm));
                if (globalStatus >= 2)
                    throw std::runtime_error("Resident NCFV flux correction violates the global convex CFL condition");
                if (globalStatus != 0)
                {
                    healthy = false;
                    break;
                }
            }
            if (healthy)
            {
                report.retries = retry;
                break;
            }
            if (!p.v.s.rejectNonphysical || retry >= p.v.s.maximumRetries)
                throw std::runtime_error("Resident NCFV encountered a nonphysical stage and exhausted allowed retries");
            maximumStep = .5 * report.minimumStep;
            if (maximumStep < p.v.s.minimumStep)
                throw std::runtime_error("Resident NCFV positivity retry fell below minimumStep");
        }
        ++p.advances;
        return report;
    }
    double ResidentNCFV::EvaluateResidual()
    {
        auto &p = *_impl;
        p.RHS(true, p.v.s.maximumStep);
        return p.Reduction().residual;
    }
    void ResidentNCFV::Download(std::vector<double> &state, std::vector<double> *steps)
    {
        Range range("NCFV output snapshot");
        auto &p = *_impl;
        if (p.inStep)
            throw std::runtime_error("Full-field transfer during a resident time step is forbidden");
        state.resize(p.v.owned * 5);
        if (!state.empty())
            Check(cudaMemcpy(state.data(), p.u.data, state.size() * sizeof(double), cudaMemcpyDeviceToHost), "download output snapshot");
        p.outputBytes += state.size() * sizeof(double);
        if (steps)
        {
            steps->resize(p.v.owned);
            if (!steps->empty())
                Check(cudaMemcpy(steps->data(), p.steps.data, steps->size() * sizeof(double), cudaMemcpyDeviceToHost), "download output step snapshot");
            p.outputBytes += steps->size() * sizeof(double);
        }
    }
    void ResidentNCFV::Report(int rank) const
    {
        const auto &p = *_impl;
        std::cout << "Resident CUDA rank " << rank << ": steps=" << p.advances << ", RHS=" << p.residuals
                  << ", scalar_D2H_bytes=" << p.scalarBytes
                  << ", packed_MPI_staging_bytes=" << p.nodeHalo->bytes + p.edgeHalo->bytes + (p.sharedHalo ? p.sharedHalo->bytes : 0)
                  << ", GPU_halo_bytes=" << p.nodeHalo->deviceBytes + p.edgeHalo->deviceBytes + (p.sharedHalo ? p.sharedHalo->deviceBytes : 0)
                  << ", GPU_halo_backend=" << p.nodeHalo->BackendName()
                  << ", output_D2H_bytes=" << p.outputBytes << std::endl;
    }
}
