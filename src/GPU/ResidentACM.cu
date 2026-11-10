#include "ResidentACM.hpp"
#include "ResidentCommon.cuh"
#include "FluxMath.cuh"
#include <iostream>
#include <fstream>

namespace DNDS::GPU
{
    namespace ACMResidentDetail
    {
        using namespace ResidentDetail;
        constexpr int M = ACMMaximumModes, C = 4 * M;
        struct View
        {
            ResidentACMSettings s;
            I owned, count, faceCount;
            const ResidentACMCell *cells;
            const ResidentACMLink *links;
            const ResidentACMBoundarySample *boundaries;
            const ResidentACMFace *faces;
            const ResidentACMPoint *points;
            const double *coupling;
            double *u, *rec, *oldRec, *gradient, *theta, *flux, *rhs, *dt;
            double *q = nullptr, *qGradient = nullptr, *flowGradient = nullptr, *qLimiter = nullptr;
            double *eddy = nullptr, *volumeFlux = nullptr, *qFlux = nullptr, *qRate = nullptr, *qRhs = nullptr;
        };
        __device__ void Rotate(const double *matrix, const double *source, double *destination)
        {
            double value[4]{};
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    value[r] += matrix[r * 3 + c] * source[c];
            value[3] = source[3];
            for (int k = 0; k < 4; ++k)
                destination[k] = value[k];
        }
        __device__ void Boundary(const View &v, int type, const double *prescribed, const double *normal,
                                 const double *interior, double *ghost)
        {
            for (int k = 0; k < 4; ++k)
                ghost[k] = interior[k];
            if (type == 1)
            {
                double un = 0, deltaUn = 0;
                for (int d = 0; d < 3; ++d)
                {
                    un += interior[d] * normal[d];
                    deltaUn += (prescribed[d] - interior[d]) * normal[d];
                }
                const double center = (1 - v.s.flow.alpha) * un;
                const double root = hypot(center, 2 * sqrt(v.s.flow.beta2 / v.s.flow.rho0));
                const double minus = .5 * (center - root), plus = .5 * (center + root);
                const double dp = prescribed[3] - interior[3];
                const double lambda = un >= 0 ? minus : plus;
                const double amplitude = un >= 0 ? (plus * dp - v.s.flow.beta2 * deltaUn) / root : (v.s.flow.beta2 * deltaUn - minus * dp) / root;
                const double sign = un >= 0 ? 1 : -1;
                for (int d = 0; d < 3; ++d)
                {
                    const double mode = lambda / v.s.flow.beta2 * normal[d] +
                                        v.s.flow.alpha * lambda / (v.s.flow.beta2 * (un - lambda)) * (interior[d] - un * normal[d]);
                    ghost[d] = (un >= 0 ? interior[d] : prescribed[d]) + sign * amplitude * mode;
                }
                ghost[3] = (un >= 0 ? interior[3] : prescribed[3]) + sign * amplitude;
            }
            else if (type == 2 || type == 4 || type == 8)
                for (int d = 0; d < 3; ++d)
                    ghost[d] = 2 * prescribed[d] - interior[d];
            else if (type == 3 || type == 9)
            {
                double vn = 0;
                for (int d = 0; d < 3; ++d)
                    vn += (interior[d] - (type == 3 ? prescribed[d] : 0)) * normal[d];
                for (int d = 0; d < 3; ++d)
                    ghost[d] -= 2 * vn * normal[d];
            }
            else if (type == 6)
                ghost[3] = 2 * prescribed[3] - interior[3];
            else if (type == 7 || type == 10)
                for (int k = 0; k < 4; ++k)
                    ghost[k] = prescribed[k];
        }
        __device__ void Neighbor(const View &v, I cell, const ResidentACMLink &link, double *value)
        {
            if (link.neighbor >= 0)
                Rotate(link.rotation, v.u + link.neighbor * 4, value);
            else
            {
                const auto &face = v.faces[link.face];
                Boundary(v, face.type, face.prescribed, face.normal, v.u + cell * 4, value);
            }
        }
        __global__ void GreenGauss(View v)
        {
            const I cell = Thread();
            if (cell >= v.owned)
                return;
            double gradient[12]{};
            for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
            {
                const auto &link = v.links[j];
                double other[4];
                Neighbor(v, cell, link, other);
                for (int k = 0; k < 4; ++k)
                    for (int d = 0; d < v.s.dimension; ++d)
                        gradient[k * 3 + d] += (other[k] - v.u[cell * 4 + k]) * link.gradientWeight[d];
            }
            for (int k = 0; k < 12; ++k)
                v.gradient[cell * 12 + k] = gradient[k];
        }
        __global__ void Variational(View v, const I *nodes, I begin, I end, bool jacobi, double relaxation)
        {
            const I position = begin + Thread();
            if (position >= end)
                return;
            const I cell = nodes ? nodes[position] : position;
            const int modes = v.cells[cell].modes;
            double work[C]{};
            const double relax = relaxation >= 0 ? relaxation : v.cells[cell].relaxation;
            for (int k = 0; k < 4; ++k)
                for (int m = 0; m < modes; ++m)
                    work[k * M + m] = (1 - relax) * v.oldRec[cell * C + k * M + m];
            for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
            {
                const auto &link = v.links[j];
                double addition[C]{};
                if (link.neighbor >= 0)
                {
                    double other[4];
                    Neighbor(v, cell, link, other);
                    const int otherModes = v.cells[link.neighbor].modes;
                    for (int row = 0; row < modes; ++row)
                    {
                        double native[4]{};
                        for (int col = 0; col < otherModes; ++col)
                            for (int k = 0; k < 4; ++k)
                            {
                                const double coefficient = link.neighbor == cell && !jacobi ? work[k * M + col] : (jacobi || link.neighbor >= cell ? v.oldRec : v.rec)[link.neighbor * C + k * M + col];
                                native[k] += v.coupling[link.coupling + row * otherModes + col] * coefficient;
                            }
                        double rotated[4];
                        Rotate(link.rotation, native, rotated);
                        for (int k = 0; k < 4; ++k)
                            addition[k * M + row] = rotated[k] + link.meanWeight[row] * (other[k] - v.u[cell * 4 + k]);
                    }
                }
                else
                    for (I p = link.boundaryBegin; p < link.boundaryEnd; ++p)
                    {
                        const auto &sample = v.boundaries[p];
                        double state[4], ghost[4];
                        for (int k = 0; k < 4; ++k)
                        {
                            state[k] = v.u[cell * 4 + k];
                            for (int m = 0; m < modes; ++m)
                                state[k] += sample.basis[m] * (jacobi ? v.oldRec[cell * C + k * M + m] : work[k * M + m]);
                        }
                        Boundary(v, sample.type, sample.prescribed, sample.normal, state, ghost);
                        for (int k = 0; k < 4; ++k)
                            for (int m = 0; m < modes; ++m)
                                addition[k * M + m] += sample.response[m] * (ghost[k] - v.u[cell * 4 + k]);
                    }
                for (int k = 0; k < 4; ++k)
                    for (int m = 0; m < modes; ++m)
                        work[k * M + m] += relax * addition[k * M + m];
            }
            for (int k = 0; k < C; ++k)
                v.rec[cell * C + k] = work[k];
        }
        __device__ void Evaluate(const View &v, I cell, const ResidentACMPoint &point, int side, double *state, double *gradient, bool limited)
        {
            const double theta = limited ? v.theta[cell] : 1;
            for (int k = 0; k < 4; ++k)
            {
                state[k] = v.u[cell * 4 + k];
                for (int d = 0; d < 3; ++d)
                    gradient[k * 3 + d] = 0;
                if (v.s.reconstruction == 1)
                    for (int d = 0; d < v.s.dimension; ++d)
                    {
                        gradient[k * 3 + d] = theta * v.gradient[cell * 12 + k * 3 + d];
                        state[k] += gradient[k * 3 + d] * point.delta[side][d];
                    }
                if (v.s.reconstruction == 2)
                    for (int m = 0; m < v.cells[cell].modes; ++m)
                    {
                        const double coefficient = theta * v.rec[cell * C + k * M + m];
                        state[k] += coefficient * point.basis[side][m];
                        for (int d = 0; d < v.s.dimension; ++d)
                            gradient[k * 3 + d] += coefficient * point.gradient[side][m * 3 + d];
                    }
            }
        }
        __global__ void Limit(View v)
        {
            const I cell = Thread();
            if (cell >= v.owned)
                return;
            double minimum[4], maximum[4];
            for (int k = 0; k < 4; ++k)
                minimum[k] = maximum[k] = v.u[cell * 4 + k];
            for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
            {
                double other[4];
                Neighbor(v, cell, v.links[j], other);
                for (int k = 0; k < 4; ++k)
                {
                    minimum[k] = fmin(minimum[k], other[k]);
                    maximum[k] = fmax(maximum[k], other[k]);
                }
            }
            double theta = 1;
            for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
            {
                const auto &link = v.links[j];
                const auto &face = v.faces[link.face];
                for (I p = face.begin; p < face.end; ++p)
                {
                    double state[4], gradient[12];
                    Evaluate(v, cell, v.points[p], link.side, state, gradient, false);
                    for (int k = 0; k < 4; ++k)
                    {
                        const double increment = state[k] - v.u[cell * 4 + k];
                        if (increment > 1e-14)
                            theta = fmin(theta, (maximum[k] - v.u[cell * 4 + k]) / increment);
                        else if (increment < -1e-14)
                            theta = fmin(theta, (minimum[k] - v.u[cell * 4 + k]) / increment);
                    }
                }
            }
            v.theta[cell] = fmax(0., fmin(1., theta));
        }
        __device__ void AddViscous(const View &v, const double *normal, const double *displacement,
                                   const double *left, const double *right, const double *gl, const double *gr, double *flux, double viscosity = -1)
        {
            double gradient[12]{}, dn = 0;
            for (int d = 0; d < 3; ++d)
                dn += displacement[d] * normal[d];
            for (int k = 0; k < 4; ++k)
            {
                double mismatch = right[k] - left[k];
                for (int d = 0; d < 3; ++d)
                {
                    gradient[k * 3 + d] = .5 * (gl[k * 3 + d] + gr[k * 3 + d]);
                    mismatch -= displacement[d] * gradient[k * 3 + d];
                }
                for (int d = 0; d < 3; ++d)
                    gradient[k * 3 + d] += normal[d] * mismatch / dn;
            }
            const double divergence = gradient[0] + gradient[4] + gradient[8];
            for (int k = 0; k < 3; ++k)
                for (int d = 0; d < 3; ++d)
                    flux[k] -= (viscosity >= 0 ? viscosity : v.s.viscosity) / v.s.flow.rho0 * normal[d] *
                               (gradient[k * 3 + d] + gradient[d * 3 + k] - (k == d ? 2. / 3 * divergence : 0));
        }
        __global__ void FaceFlux(View v)
        {
            const I id = Thread();
            if (id >= v.faceCount)
                return;
            const auto &face = v.faces[id];
            double total[4]{};
            for (I p = face.begin; p < face.end; ++p)
            {
                const auto &point = v.points[p];
                FluxInput input = v.s.flow;
                double gl[12], gr[12], rightNative[4];
                Evaluate(v, face.left, point, 0, input.left, gl, true);
                if (face.right >= 0)
                {
                    double nativeGradient[12];
                    Evaluate(v, face.right, point, 1, rightNative, nativeGradient, true);
                    Rotate(face.rotation, rightNative, input.right);
                    for (int k = 0; k < 4; ++k)
                        for (int d = 0; d < 3; ++d)
                        {
                            gr[k * 3 + d] = 0;
                            for (int a = 0; a < 4; ++a)
                                for (int b = 0; b < 3; ++b)
                                    gr[k * 3 + d] += (k == 3 ? (a == 3 ? 1. : 0.) : (a == 3 ? 0. : face.rotation[k * 3 + a])) *
                                                     nativeGradient[a * 3 + b] * face.rotation[d * 3 + b];
                        }
                }
                else
                {
                    Boundary(v, face.type, face.prescribed, point.normal, input.left, input.right);
                    for (int k = 0; k < 12; ++k)
                        gr[k] = gl[k];
                }
                for (int d = 0; d < 3; ++d)
                    input.normal[d] = point.normal[d];
                FluxOutput output;
                Detail::ACMFlux(input, output);
                if (v.s.sst)
                    v.volumeFlux[p] = output.value[3];
                if (v.s.viscous)
                {
                    double meanRight[4];
                    if (face.right >= 0)
                        Rotate(face.rotation, v.u + face.right * 4, meanRight);
                    else
                        Boundary(v, face.type, face.prescribed, point.normal, v.u + face.left * 4, meanRight);
                    AddViscous(v, point.normal, point.displacement, v.u + face.left * 4, meanRight, gl, gr, output.value,
                               v.s.viscosity + (v.s.sst ? v.eddy[p] : 0));
                }
                for (int k = 0; k < 4; ++k)
                    total[k] -= point.measure * output.value[k];
            }
            for (int k = 0; k < 4; ++k)
                v.flux[id * 4 + k] = total[k];
        }
        __global__ void Assemble(View v, bool inverseGamma)
        {
            const I cell = Thread();
            if (cell >= v.owned)
                return;
            double residual[4]{};
            for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
            {
                const auto &link = v.links[j];
                const auto &face = v.faces[link.face];
                if (face.left == face.right)
                    continue;
                double flux[4];
                for (int k = 0; k < 4; ++k)
                    flux[k] = v.flux[link.face * 4 + k];
                if (link.side)
                {
                    double rotated[4];
                    Rotate(link.rotation, flux, rotated);
                    for (int k = 0; k < 4; ++k)
                        flux[k] = -rotated[k];
                }
                for (int k = 0; k < 4; ++k)
                    residual[k] += flux[k] / v.cells[cell].volume;
            }
            if (inverseGamma)
            {
                for (int k = 0; k < 3; ++k)
                    residual[k] -= (1 + v.s.flow.alpha) * v.u[cell * 4 + k] * residual[3];
                residual[3] *= v.s.flow.beta2;
            }
            for (int k = 0; k < 4; ++k)
                v.rhs[cell * 4 + k] = residual[k];
        }
        __device__ double Radius(const View &v, const double *u, const double *normal)
        {
            double un = 0;
            for (int d = 0; d < 3; ++d)
                un += u[d] * normal[d];
            const double centered = (1 - v.s.flow.alpha) * un;
            return fmax(fabs(un), .5 * (fabs(centered) + hypot(centered, 2 * sqrt(v.s.flow.beta2 / v.s.flow.rho0))));
        }
        __global__ void TimeStep(View v, double uniform)
        {
            const I cell = Thread();
            if (cell >= v.owned)
                return;
            if (uniform > 0)
            {
                v.dt[cell] = uniform;
                return;
            }
            double sum = 0;
            for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
            {
                const auto &face = v.faces[v.links[j].face];
                double right[4];
                if (face.right >= 0)
                    Rotate(face.rotation, v.u + face.right * 4, right);
                else
                    Boundary(v, face.type, face.prescribed, face.normal, v.u + face.left * 4, right);
                double radius = fmax(Radius(v, v.u + face.left * 4, face.normal), Radius(v, right, face.normal));
                if (v.s.viscous && v.s.viscosity > 0)
                {
                    double viscosity = v.s.viscosity;
                    if (v.s.sst)
                    {
                        double sumEddy = 0;
                        for (I p = face.begin; p < face.end; ++p)
                            sumEddy += v.eddy[p];
                        viscosity += sumEddy / (face.end - face.begin);
                    }
                    radius += 4. / 3 * viscosity / v.s.flow.rho0 * face.area *
                              (1 / v.cells[face.left].volume + 1 / v.cells[face.right >= 0 ? face.right : face.left].volume);
                }
                sum += radius * face.area;
            }
            v.dt[cell] = fmin(v.s.maximumStep, v.s.cfl * v.cells[cell].volume * v.cells[cell].smoothScale / (sum + 1e-100));
        }
        __global__ void RKStage(View v, const double *history, double weight)
        {
            const I i = Thread();
            if (i >= v.owned * 4)
                return;
            v.u[i] = weight * history[i] + (1 - weight) * (v.u[i] + v.dt[i / 4] * v.rhs[i]);
        }
        __global__ void Reduce(const double *values, I count, double *scalar, bool minimum)
        {
            __shared__ double partial[128];
            const I i = Thread();
            const double value = i < count ? values[i] : (minimum ? 1e300 : 0.);
            partial[threadIdx.x] = minimum ? value : value * value;
            __syncthreads();
            for (int stride = 64; stride; stride /= 2)
            {
                if (threadIdx.x < stride)
                    partial[threadIdx.x] = minimum ? fmin(partial[threadIdx.x], partial[threadIdx.x + stride]) : partial[threadIdx.x] + partial[threadIdx.x + stride];
                __syncthreads();
            }
            if (!threadIdx.x)
            {
                if (minimum)
                    Minimum(scalar, partial[0]);
                else
                    atomicAdd(scalar, partial[0]);
            }
        }
        __global__ void ReconstructionDefect(View v, double *scalar)
        {
            const I cell = Thread();
            if (cell >= v.owned)
                return;
            double maximum = 0;
            for (int k = 0; k < 4; ++k)
                for (int m = 0; m < v.cells[cell].modes; ++m)
                    maximum = fmax(maximum, fabs(v.rec[cell * C + k * M + m] - v.oldRec[cell * C + k * M + m]) /
                                                (k == 3 ? v.s.flow.beta2 : sqrt(v.s.flow.beta2 / v.s.flow.rho0)));
            Maximum(scalar, maximum);
        }
        __global__ void RelaxReconstruction(View v, double relaxation)
        {
            const I i = Thread();
            if (i >= v.owned * C)
                return;
            v.rec[i] = v.oldRec[i] + relaxation * (v.rec[i] - v.oldRec[i]);
        }
    }

#include "ResidentACMSST.cuh"

    struct ResidentACM::Impl
    {
        using I = DeviceIndex;
        ACMResidentDetail::View v{};
        MPI_Comm comm;
        ResidentDetail::Buffer<ResidentACMCell> cells;
        ResidentDetail::Buffer<ResidentACMLink> links;
        ResidentDetail::Buffer<ResidentACMBoundarySample> boundaries;
        ResidentDetail::Buffer<ResidentACMFace> faces;
        ResidentDetail::Buffer<ResidentACMPoint> points;
        ResidentDetail::Buffer<double> coupling, u, rec, oldRec, gradient, theta, flux, rhs, dt, history, scalar;
        ResidentDetail::Buffer<I> nodes;
        std::vector<I> levels;
        ResidentDetail::Halo halo;
        std::unique_ptr<ACMSSTWorkspace> advanced;
        cudaGraphExec_t reconstructionGraph = nullptr;
        std::uint64_t scalarBytes = 0, outputBytes = 0, steps = 0, sweeps = 0;
        I globalOwned = 0;
        bool inStep = false;
        static unsigned Blocks(I n) { return unsigned((n + 127) / 128); }
        Impl(const ResidentACMData &data, MPI_Comm communicator) : comm(communicator), levels(data.reconstructionLevels), halo(data.globals, data.owned, comm)
        {
            using namespace ResidentDetail;
            if (data.settings.sst && data.settings.integrator != 4 && data.settings.integrator != 2)
                throw std::runtime_error("Resident SST requires ImplicitEulerLUSGS or BDF2DualTimeLUSGS; CPU fallback is disabled");
            if (!data.settings.sst && data.settings.integrator == 4)
                throw std::runtime_error("Resident BDF2 currently requires the conservative six-variable SST solver; CPU fallback is disabled");
            if (data.settings.integrator != 0 && data.settings.integrator != 4 && !(data.settings.sst && data.settings.integrator == 2))
                throw std::runtime_error("Resident ACM implicit integrators are still being ported; CPU fallback is disabled");
            if (data.settings.limiter && data.settings.limiterType != 0)
                throw std::runtime_error("Resident ACM characteristic limiter is still being ported; CPU fallback is disabled");
            if (data.settings.reconstructionGMRES)
                throw std::runtime_error("Resident ACM reconstruction GMRES is still being ported; CPU fallback is disabled");
            v.s = data.settings;
            v.owned = data.owned;
            v.count = data.cells.size();
            v.faceCount = data.faces.size();
            MPIcheck(MPI_Allreduce(&v.owned, &globalOwned, 1, MPI_INT64_T, MPI_SUM, comm));
            cells.Upload(data.cells);
            links.Upload(data.links);
            boundaries.Upload(data.boundaries);
            faces.Upload(data.faces);
            points.Upload(data.points);
            coupling.Upload(data.coupling);
            u.Upload(data.initialState);
            nodes.Upload(data.reconstructionNodes);
            rec.Resize(v.count * ACMResidentDetail::C);
            oldRec.Resize(rec.count);
            gradient.Resize(v.count * 12);
            theta.Resize(v.count);
            flux.Resize(v.faceCount * 4);
            rhs.Resize(v.count * 4);
            dt.Resize(v.owned);
            history.Resize(v.count * 4);
            scalar.Resize(1);
            Check(cudaMemset(rec.data, 0, rec.count * sizeof(double)), "initialize ACM coefficients");
            Check(cudaMemset(gradient.data, 0, gradient.count * sizeof(double)), "initialize ACM gradients");
            v.cells = cells.data;
            v.links = links.data;
            v.boundaries = boundaries.data;
            v.faces = faces.data;
            v.points = points.data;
            v.coupling = coupling.data;
            v.u = u.data;
            v.rec = rec.data;
            v.oldRec = oldRec.data;
            v.gradient = gradient.data;
            v.theta = theta.data;
            v.flux = flux.data;
            v.rhs = rhs.data;
            v.dt = dt.data;
            if (v.s.integrator == 4 || (v.s.sst && v.s.integrator == 2))
                advanced = std::make_unique<ACMSSTWorkspace>(v, data, halo, comm, globalOwned, nodes.data, levels);
            if (v.s.reconstruction == 2 && v.s.sor && !v.s.controlled)
            {
                cudaStream_t stream;
                cudaGraph_t graph;
                Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create reconstruction capture stream");
                Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin reconstruction capture");
                for (std::size_t level = 0; level + 1 < levels.size(); ++level)
                    if (levels[level + 1] > levels[level])
                        ACMResidentDetail::Variational<<<Blocks(levels[level + 1] - levels[level]), 128, 0, stream>>>(v, nodes.data, levels[level], levels[level + 1], false, -1);
                Check(cudaStreamEndCapture(stream, &graph), "finish reconstruction capture");
                Check(cudaGraphInstantiate(&reconstructionGraph, graph, nullptr, nullptr, 0), "instantiate reconstruction graph");
                Check(cudaGraphDestroy(graph), "release reconstruction capture graph");
                Check(cudaStreamDestroy(stream), "release reconstruction capture stream");
                Check(cudaGraphUpload(reconstructionGraph, nullptr), "upload reconstruction execution graph");
            }
            Check(cudaDeviceSynchronize(), "finish ACM resident initialization");
        }
        ~Impl()
        {
            if (reconstructionGraph)
                cudaGraphExecDestroy(reconstructionGraph);
        }
        double Reduction(const double *values, I count, bool minimum)
        {
            using namespace ResidentDetail;
            const double initial = minimum ? 1e300 : 0.;
            Check(cudaMemcpy(scalar.data, &initial, sizeof(double), cudaMemcpyHostToDevice), "initialize ACM scalar reduction");
            if (count)
                ACMResidentDetail::Reduce<<<Blocks(count), 128>>>(values, count, scalar.data, minimum);
            double local = 0, global = 0;
            Check(cudaMemcpy(&local, scalar.data, sizeof(double), cudaMemcpyDeviceToHost), "read ACM reduction");
            MPIcheck(MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, minimum ? MPI_MIN : MPI_SUM, comm));
            scalarBytes += 16;
            return minimum ? global : std::sqrt(global / std::max<I>(1, globalOwned * 4));
        }
        void Residual(bool inverseGamma = true)
        {
            using namespace ResidentDetail;
            using namespace ACMResidentDetail;
            halo.Pull(u.data, 4);
            if (advanced)
                advanced->Prepare();
            if (v.s.reconstruction == 1)
            {
                if (v.owned)
                    GreenGauss<<<Blocks(v.owned), 128>>>(v);
                halo.Pull(gradient.data, 12);
            }
            if (v.s.reconstruction == 2)
            {
                if (v.s.resetCoefficients)
                    Check(cudaMemset(rec.data, 0, rec.count * sizeof(double)), "reset ACM coefficients");
                bool converged = false;
                const int maximum = v.s.controlled ? v.s.maximumSweeps : v.s.sweeps;
                for (int sweep = 0; sweep < maximum + int(v.s.controlled); ++sweep)
                {
                    Check(cudaMemcpy(oldRec.data, rec.data, rec.count * sizeof(double), cudaMemcpyDeviceToDevice), "snapshot ACM reconstruction");
                    if (v.s.controlled)
                    {
                        if (v.owned)
                            Variational<<<Blocks(v.owned), 128>>>(v, nullptr, 0, v.owned, true, 1);
                        if (sweep == 0 || sweep == maximum || (sweep >= v.s.sweeps && sweep % v.s.checkInterval == 0))
                        {
                            Check(cudaMemset(scalar.data, 0, sizeof(double)), "reset reconstruction defect");
                            if (v.owned)
                                ReconstructionDefect<<<Blocks(v.owned), 128>>>(v, scalar.data);
                            double local = 0, global = 0;
                            Check(cudaMemcpy(&local, scalar.data, sizeof(double), cudaMemcpyDeviceToHost), "read reconstruction convergence scalar");
                            MPIcheck(MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm));
                            scalarBytes += 8;
                            converged = global <= v.s.reconstructionTolerance;
                            if (converged || sweep == maximum)
                            {
                                Check(cudaMemcpy(rec.data, oldRec.data, rec.count * sizeof(double), cudaMemcpyDeviceToDevice), "restore checked reconstruction");
                                break;
                            }
                        }
                        if (v.owned)
                            RelaxReconstruction<<<Blocks(v.owned * C), 128>>>(v, v.s.reconstructionRelaxation);
                    }
                    else if (v.s.sor)
                    {
                        Check(cudaGraphLaunch(reconstructionGraph, nullptr), "launch resident reconstruction graph");
                    }
                    else if (v.owned)
                        Variational<<<Blocks(v.owned), 128>>>(v, nullptr, 0, v.owned, true, -1);
                    halo.Pull(rec.data, C);
                    ++sweeps;
                }
                if (v.s.controlled && !converged)
                    throw std::runtime_error("Resident ACM reconstruction failed to converge");
            }
            if (v.s.limiter && v.s.reconstruction && v.owned)
                Limit<<<Blocks(v.owned), 128>>>(v);
            else if (v.count)
                Ones<<<Blocks(v.count), 128>>>(theta.data, v.count);
            halo.Pull(theta.data, 1);
            if (v.faceCount)
                FaceFlux<<<Blocks(v.faceCount), 128>>>(v);
            if (v.owned)
                Assemble<<<Blocks(v.owned), 128>>>(v, inverseGamma);
            if (advanced)
                advanced->Residual();
            Check(cudaGetLastError(), "ACM residual kernels");
        }
    };
    ResidentACM::ResidentACM(const ResidentACMData &data, MPI_Comm communicator) : _impl(std::make_unique<Impl>(data, communicator)) {}
    ResidentACM::~ResidentACM() = default;
    void ResidentACM::SetCFL(double cfl)
    {
        if (_impl->inStep || !std::isfinite(cfl) || cfl <= 0)
            throw std::runtime_error("Invalid resident CFL update");
        _impl->v.s.cfl = cfl;
    }
    void ResidentACM::WriteSSTAudit(const std::string &filename)
    {
        using namespace ResidentDetail;
        using namespace ACMResidentDetail;
        auto &p = *_impl;
        if (p.inStep || !p.advanced || !p.v.s.sst)
            throw std::runtime_error("SST audit requires an idle resident SST solver");
        p.Residual(false);
        if (p.v.owned)
        {
            TimeStep<<<Impl::Blocks(p.v.owned), 128>>>(p.v, p.v.s.cflStep ? -1 : p.v.s.fixedStep);
            FaceJacobian<<<Impl::Blocks(p.v.faceCount), 128>>>(p.v, p.advanced->jac.data);
            SSTImplicitFaces<<<Impl::Blocks(p.v.faceCount), 128>>>(p.v, p.advanced->qJac.data, p.advanced->advected.data);
            ImplicitDiagonal<<<Impl::Blocks(p.v.owned), 128>>>(p.v, p.advanced->jac.data, p.advanced->qJac.data,
                p.advanced->diagonal.data, p.advanced->qDiagonal.data,
                p.advanced->qPrevious.data, p.advanced->qPreviousPrevious.data, true);
        }
        // Only a small chunk is allocated in addition to the resident workspace.
        constexpr I chunk = 8192, stride = 49;
        Buffer<double> device;
        device.Resize(chunk * stride);
        std::vector<double> host(chunk * stride);
        std::ofstream stream(filename, std::ios::binary);
        if (!stream) throw std::runtime_error("Cannot open SST audit output");
        const std::uint64_t header[3]{1, static_cast<std::uint64_t>(p.v.owned), stride};
        stream.write(reinterpret_cast<const char *>(header), sizeof(header));
        for (I begin = 0; begin < p.v.owned; begin += chunk)
        {
            const I count = std::min(chunk, p.v.owned - begin);
            SSTAuditCells<<<Impl::Blocks(count), 128>>>(p.v, p.advanced->qDiagonal.data, begin, count, device.data);
            Check(cudaMemcpy(host.data(), device.data, count * stride * sizeof(double), cudaMemcpyDeviceToHost), "read SST audit chunk");
            stream.write(reinterpret_cast<const char *>(host.data()), count * stride * sizeof(double));
        }
        if (!stream) throw std::runtime_error("Failed to write SST audit output");
    }
    ResidentACMReport ResidentACM::Advance()
    {
        using namespace ResidentDetail;
        using namespace ACMResidentDetail;
        Range range("ACM numerical step");
        auto &p = *_impl;
        p.inStep = true;
        ResidentACMReport report;
        p.halo.Pull(p.u.data, 4);
        if (p.advanced)
            p.advanced->Prepare();
        if (p.v.owned)
            TimeStep<<<Impl::Blocks(p.v.owned), 128>>>(p.v, p.v.s.cflStep ? -1 : p.v.s.fixedStep);
        report.minimumStep = p.Reduction(p.dt.data, p.v.owned, true);
        if (p.v.s.cflStep && !p.v.s.localStep && p.v.owned)
            TimeStep<<<Impl::Blocks(p.v.owned), 128>>>(p.v, report.minimumStep);
        if (p.advanced)
        {
            const double minimum = report.minimumStep;
            report = p.advanced->Advance([&]() { p.Residual(false); });
            report.minimumStep = minimum;
            p.inStep = false;
            ++p.steps;
            return report;
        }
        Check(cudaMemcpy(p.history.data, p.u.data, p.u.count * sizeof(double), cudaMemcpyDeviceToDevice), "snapshot ACM RK history");
        p.Residual();
        report.initialNorm = p.Reduction(p.rhs.data, p.v.owned * 4, false);
        for (double weight : {0., .75, 1. / 3})
        {
            if (p.v.owned)
                RKStage<<<Impl::Blocks(p.v.owned * 4), 128>>>(p.v, p.history.data, weight);
            p.Residual();
        }
        report.finalNorm = p.Reduction(p.rhs.data, p.v.owned * 4, false);
        if (!std::isfinite(report.finalNorm))
            throw std::runtime_error("Resident ACM generated a non-finite residual");
        report.iterations = 3;
        report.converged = true;
        p.inStep = false;
        ++p.steps;
        return report;
    }
    void ResidentACM::Download(std::vector<double> &state)
    {
        ResidentDetail::Range range("ACM output snapshot");
        auto &p = *_impl;
        if (p.inStep)
            throw std::logic_error("ACM field download inside numerical step is prohibited");
        state.resize(p.v.owned * 4);
        ResidentDetail::Check(cudaMemcpy(state.data(), p.u.data, state.size() * sizeof(double), cudaMemcpyDeviceToHost), "download ACM output snapshot");
        p.outputBytes += state.size() * sizeof(double);
    }
    void ResidentACM::Report(int rank) const
    {
        const auto &p = *_impl;
        std::cout << "Resident CUDA rank " << rank << " ACM: steps=" << p.steps << ", reconstruction_sweeps=" << p.sweeps
                  << ", scalar_transfer_bytes=" << p.scalarBytes + (p.advanced ? p.advanced->scalarBytes : 0) << ", MPI_staging_bytes=" << p.halo.bytes
                  << ", GPU_halo_bytes=" << p.halo.deviceBytes
                  << ", GPU_halo_backend=" << p.halo.BackendName()
                  << ", output_D2H_bytes=" << p.outputBytes << std::endl;
    }
    void ResidentACM::DownloadTurbulence(std::vector<double> &state)
    {
        ResidentDetail::Range range("ACM output snapshot");
        auto &p = *_impl;
        if (p.inStep || !p.advanced || !p.v.s.sst)
            throw std::logic_error("SST output requires completed resident numerical step");
        state.resize(p.v.owned * 2);
        ResidentDetail::Check(cudaMemcpy(state.data(), p.v.q, state.size() * sizeof(double), cudaMemcpyDeviceToHost), "download SST output snapshot");
        p.outputBytes += state.size() * sizeof(double);
    }
}
