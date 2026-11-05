// Included inside DNDS::GPU after the common ACM reconstruction/flux kernels.
// All state-dependent SST closures, derivatives, linear solves and BDF histories
// below execute on the device. Host code only schedules kernels and reads norms.
namespace ACMResidentDetail
{
    __device__ double QClamp(const View &v, double value, int k)
    {
        return fmax(v.s.turbulenceMinimum[k], fmin(v.s.turbulenceMaximum[k], value));
    }
    __device__ double Positive(double value) { return fmax(1e-30, value); }
    __device__ void TurbulenceBoundary(const View &v, I owner, int type, const double *interior, double *ghost)
    {
        for (int k = 0; k < 2; ++k)
            ghost[k] = interior[k];
        if (type == 2 || type == 4)
        {
            const double d = fmax(v.cells[owner].wallDistance, v.s.minimumWallDistance);
            const double omega = fmin(v.s.turbulenceMaximum[1], v.s.wallOmegaCoefficient * v.s.viscosity / v.s.flow.rho0 / (d * d));
            ghost[0] = -interior[0];
            ghost[1] = 2 * omega - interior[1];
        }
        else if (type == 1 || type == 7 || type == 8 || type == 10)
            for (int k = 0; k < 2; ++k)
                ghost[k] = 2 * v.s.turbulenceFar[k] - interior[k];
    }
    __device__ void QNeighbor(const View &v, I owner, const ResidentACMLink &link, double *flow, double *q)
    {
        Neighbor(v, owner, link, flow);
        if (link.neighbor >= 0)
            for (int k = 0; k < 2; ++k)
                q[k] = v.q[link.neighbor * 2 + k];
        else
        {
            const auto &face = v.faces[link.face];
            TurbulenceBoundary(v, owner, face.type, v.q + owner * 2, q);
            double un = 0;
            for (int d = 0; d < 3; ++d)
                un += v.u[owner * 4 + d] * face.normal[d];
            if (face.type == 1 && un >= 0)
                for (int k = 0; k < 2; ++k)
                    q[k] = v.q[owner * 2 + k];
        }
    }
    __global__ void SSTGradients(View v)
    {
        const I cell = Thread();
        if (cell >= v.owned) return;
        double gf[12]{}, gq[6]{};
        for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
        {
            const auto &link = v.links[j];
            double other[4], q[2];
            QNeighbor(v, cell, link, other, q);
            for (int d = 0; d < v.s.dimension; ++d)
            {
                for (int k = 0; k < 4; ++k)
                    gf[k * 3 + d] += (other[k] - v.u[cell * 4 + k]) * link.turbulenceWeight[d];
                for (int k = 0; k < 2; ++k)
                    gq[k * 3 + d] += (q[k] - v.q[cell * 2 + k]) * link.turbulenceWeight[d];
            }
        }
        for (int k = 0; k < 12; ++k) v.flowGradient[cell * 12 + k] = gf[k];
        for (int k = 0; k < 6; ++k) v.qGradient[cell * 6 + k] = gq[k];
    }
    __global__ void SSTLimiter(View v)
    {
        const I cell = Thread();
        if (cell >= v.owned) return;
        double lo[6], hi[6], theta[3]{1, 1, 1};
        for (int k = 0; k < 6; ++k)
            lo[k] = hi[k] = k < 4 ? v.u[cell * 4 + k] : v.q[cell * 2 + k - 4];
        if (v.s.turbulenceSecondOrder)
        {
            for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
            {
                double f[4], q[2];
                QNeighbor(v, cell, v.links[j], f, q);
                for (int k = 0; k < 6; ++k)
                {
                    const double value = k < 4 ? f[k] : q[k - 4];
                    lo[k] = fmin(lo[k], value); hi[k] = fmax(hi[k], value);
                }
            }
            for (int k = 0; k < 2; ++k) lo[k + 4] = fmax(lo[k + 4], v.s.turbulenceMinimum[k]);
            for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
            {
                const auto &link = v.links[j];
                const auto &face = v.faces[link.face];
                for (I p = face.begin; p < face.end; ++p)
                    for (int k = 0; k < 6; ++k)
                    {
                        const double mean = k < 4 ? v.u[cell * 4 + k] : v.q[cell * 2 + k - 4];
                        const double *g = k < 4 ? v.flowGradient + cell * 12 + k * 3 : v.qGradient + cell * 6 + (k - 4) * 3;
                        double du = 0;
                        for (int d = 0; d < v.s.dimension; ++d) du += g[d] * v.points[p].delta[link.side][d];
                        const int entry = k < 4 ? 2 : k - 4;
                        if (du > 1e-14) theta[entry] = fmin(theta[entry], (hi[k] - mean) / du);
                        else if (du < -1e-14) theta[entry] = fmin(theta[entry], (lo[k] - mean) / du);
                    }
            }
        }
        for (int k = 0; k < 3; ++k) v.qLimiter[cell * 3 + k] = fmax(0., fmin(1., theta[k]));
    }
    __device__ void Blend(const View &v, const double *qInput, const double *g, double distance, double &f1, double &f2)
    {
        const double k = Positive(QClamp(v, qInput[0], 0)), w = Positive(QClamp(v, qInput[1], 1));
        const double d = Positive(distance), nu = v.s.viscosity / Positive(v.s.flow.rho0);
        double cross = 0;
        for (int j = 0; j < 3; ++j) cross += g[j] * g[j + 3];
        const double cd = fmax(2 * v.s.flow.rho0 * .856 * cross / w, 1e-10);
        const double near = sqrt(k) / (.09 * w * d), viscous = 500 * nu / (d * d * w);
        const double arg1 = fmax(0., fmin(10., fmin(fmax(near, viscous), 4 * v.s.flow.rho0 * .856 * k / (cd * d * d))));
        const double arg2 = fmax(0., fmin(10., fmax(2 * near, viscous)));
        f1 = tanh(pow(arg1, 4.)); f2 = tanh(arg2 * arg2);
    }
    __device__ double Eddy(const View &v, const double *q, const double *gf, const double *gq, double distance)
    {
        double f1, f2, rotation = 0;
        Blend(v, q, gq, distance, f1, f2);
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
            {
                const double r = .5 * (gf[a * 3 + b] - gf[b * 3 + a]);
                rotation += r * r;
            }
        const double viscosity = v.s.flow.rho0 * .31 * QClamp(v, q[0], 0) /
            Positive(fmax(.31 * QClamp(v, q[1], 1), sqrt(2 * rotation) * f2));
        return fmax(0., fmin(viscosity, v.s.eddyRatioMaximum * v.s.viscosity));
    }
    __device__ void Diffusivity(const View &v, const double *q, const double *gq, double distance, double eddy, double *diffusion)
    {
        double f1, f2;
        Blend(v, q, gq, distance, f1, f2);
        diffusion[0] = (v.s.viscosity + (.85 * f1 + 1. * (1 - f1)) * eddy) / v.s.flow.rho0;
        diffusion[1] = (v.s.viscosity + (.5 * f1 + .856 * (1 - f1)) * eddy) / v.s.flow.rho0;
    }
    __device__ void SSTSource(const View &v, const double *qInput, const double *gf, const double *gq, double distance, double *source)
    {
        source[0] = source[1] = 0;
        if (!v.s.turbulenceSource) return;
        const double k = QClamp(v, qInput[0], 0), w = QClamp(v, qInput[1], 1);
        const double divergence = gf[0] + gf[4] + gf[8];
        double strain = 0, cross = 0;
        for (int a = 0; a < 3; ++a)
        {
            cross += gq[a] * gq[a + 3];
            for (int b = 0; b < 3; ++b)
            {
                const double s = .5 * (gf[a * 3 + b] + gf[b * 3 + a]) - (a == b ? divergence / 3 : 0);
                strain += s * s;
            }
        }
        // Preserve the CPU closure's sqrt followed by square.
        const double magnitude = sqrt(2 * strain), invariant = magnitude * magnitude;
        const double mu = Eddy(v, qInput, gf, gq, fmax(distance, v.s.minimumWallDistance));
        const double production = fmax(0., mu * invariant / v.s.flow.rho0 - (2. / 3) * k * divergence);
        double f1, f2;
        Blend(v, qInput, gq, fmax(distance, v.s.minimumWallDistance), f1, f2);
        const double beta = .075 * f1 + .0828 * (1 - f1);
        const double gamma1 = .075 / .09 - .5 * .41 * .41 / sqrt(.09);
        const double gamma2 = .0828 / .09 - .856 * .41 * .41 / sqrt(.09);
        source[0] = fmin(production, 20 * .09 * k * w) - .09 * k * w;
        source[1] = (gamma1 * f1 + gamma2 * (1 - f1)) * invariant - beta * w * w + 2 * (1 - f1) * .856 * cross / Positive(w);
    }
    __device__ void CorrectGradient(const double *left, const double *right, const double *gl, const double *gr,
                                    const double *displacement, const double *normal, int variables, double *result)
    {
        double dn = 0;
        for (int d = 0; d < 3; ++d) dn += displacement[d] * normal[d];
        for (int k = 0; k < variables; ++k)
        {
            double mismatch = right[k] - left[k];
            for (int d = 0; d < 3; ++d)
            {
                result[k * 3 + d] = .5 * (gl[k * 3 + d] + gr[k * 3 + d]);
                mismatch -= displacement[d] * result[k * 3 + d];
            }
            if (fabs(dn) > 1e-14)
                for (int d = 0; d < 3; ++d) result[k * 3 + d] += normal[d] * mismatch / dn;
        }
    }
    __device__ void SSTPointData(const View &v, const ResidentACMFace &face, const ResidentACMPoint &point,
                                 double *ql, double *qr, double *qg, double *fg)
    {
        double fl[4], fr[4], qmr[2], fmr[4], gqr[6], gfr[12];
        for (int k = 0; k < 2; ++k)
        {
            ql[k] = v.q[face.left * 2 + k];
            if (v.s.turbulenceSecondOrder)
                for (int d = 0; d < v.s.dimension; ++d)
                    ql[k] += v.qLimiter[face.left * 3 + k] * v.qGradient[face.left * 6 + k * 3 + d] * point.delta[0][d];
            ql[k] = QClamp(v, ql[k], k);
        }
        for (int k = 0; k < 4; ++k)
        {
            fl[k] = v.u[face.left * 4 + k];
            if (v.s.turbulenceSecondOrder)
                for (int d = 0; d < v.s.dimension; ++d)
                    fl[k] += v.qLimiter[face.left * 3 + 2] * v.flowGradient[face.left * 12 + k * 3 + d] * point.delta[0][d];
        }
        if (face.right >= 0)
        {
            for (int k = 0; k < 2; ++k)
            {
                qr[k] = qmr[k] = v.q[face.right * 2 + k];
                if (v.s.turbulenceSecondOrder)
                    for (int d = 0; d < v.s.dimension; ++d)
                        qr[k] += v.qLimiter[face.right * 3 + k] * v.qGradient[face.right * 6 + k * 3 + d] * point.delta[1][d];
                qr[k] = QClamp(v, qr[k], k);
                for (int d = 0; d < 3; ++d)
                {
                    gqr[k * 3 + d] = 0;
                    for (int b = 0; b < 3; ++b) gqr[k * 3 + d] += face.rotation[d * 3 + b] * v.qGradient[face.right * 6 + k * 3 + b];
                }
            }
            Rotate(face.rotation, v.u + face.right * 4, fmr);
            for (int k = 0; k < 4; ++k)
                for (int d = 0; d < 3; ++d)
                {
                    gfr[k * 3 + d] = 0;
                    for (int a = 0; a < 4; ++a)
                        for (int b = 0; b < 3; ++b)
                            gfr[k * 3 + d] += (k == 3 ? (a == 3 ? 1. : 0.) : (a == 3 ? 0. : face.rotation[k * 3 + a])) *
                                v.flowGradient[face.right * 12 + a * 3 + b] * face.rotation[d * 3 + b];
                }
        }
        else
        {
            TurbulenceBoundary(v, face.left, face.type, ql, qr);
            TurbulenceBoundary(v, face.left, face.type, v.q + face.left * 2, qmr);
            Boundary(v, face.type, face.prescribed, point.normal, v.u + face.left * 4, fmr);
            double un = 0;
            for (int d = 0; d < 3; ++d) un += fl[d] * point.normal[d];
            if (face.type == 1 && un >= 0)
                for (int k = 0; k < 2; ++k) { qr[k] = ql[k]; qmr[k] = v.q[face.left * 2 + k]; }
            for (int k = 0; k < 6; ++k) gqr[k] = v.qGradient[face.left * 6 + k];
            for (int k = 0; k < 12; ++k) gfr[k] = v.flowGradient[face.left * 12 + k];
        }
        CorrectGradient(v.q + face.left * 2, qmr, v.qGradient + face.left * 6, gqr, point.displacement, point.normal, 2, qg);
        CorrectGradient(v.u + face.left * 4, fmr, v.flowGradient + face.left * 12, gfr, point.displacement, point.normal, 4, fg);
    }
    __global__ void SSTEddy(View v)
    {
        const I faceId = Thread();
        if (faceId >= v.faceCount) return;
        const auto &face = v.faces[faceId];
        for (I p = face.begin; p < face.end; ++p)
        {
            double ql[2], qr[2], qg[6], fg[12], mean[2];
            SSTPointData(v, face, v.points[p], ql, qr, qg, fg);
            for (int k = 0; k < 2; ++k) mean[k] = QClamp(v, .5 * (ql[k] + qr[k]), k);
            v.eddy[p] = Eddy(v, mean, fg, qg, face.wallDistance);
        }
    }
    __global__ void SSTFlux(View v)
    {
        const I faceId = Thread();
        if (faceId >= v.faceCount) return;
        const auto &face = v.faces[faceId];
        double total[2]{}, rate[2]{};
        for (I p = face.begin; p < face.end; ++p)
        {
            const auto &point = v.points[p];
            double ql[2], qr[2], qg[6], fg[12], mean[2], diffusion[2], rateDiff[2];
            SSTPointData(v, face, point, ql, qr, qg, fg);
            for (int k = 0; k < 2; ++k) mean[k] = QClamp(v, .5 * (ql[k] + qr[k]), k);
            Diffusivity(v, mean, qg, face.wallDistance, v.eddy[p], diffusion);
            double dn = 0;
            for (int d = 0; d < 3; ++d) dn += point.displacement[d] * point.normal[d];
            const double distance = fmax(fabs(dn), v.s.minimumWallDistance);
            for (int k = 0; k < 2; ++k)
            {
                double derivative = 0;
                for (int d = 0; d < 3; ++d) derivative += point.normal[d] * qg[k * 3 + d];
                double advective = v.volumeFlux[p] * (v.volumeFlux[p] >= 0 ? ql[k] : qr[k]);
                if (face.right < 0 && (face.type == 2 || face.type == 4 || face.type == 3 || face.type == 9)) advective = 0;
                total[k] -= point.measure * (advective - diffusion[k] * derivative);
                double unitGradient[6]{};
                for (int d = 0; d < 3; ++d) unitGradient[k * 3 + d] = point.normal[d];
                Diffusivity(v, v.q + face.left * 2, unitGradient, face.wallDistance, v.eddy[p], rateDiff);
                double normalSquared = 0;
                for (int d = 0; d < 3; ++d) normalSquared += point.normal[d] * point.normal[d];
                rate[k] += point.measure * (fabs(v.volumeFlux[p]) + 2 * fabs(rateDiff[k] * normalSquared) / distance);
            }
        }
        for (int k = 0; k < 2; ++k) { v.qFlux[faceId * 2 + k] = total[k]; v.qRate[faceId * 2 + k] = rate[k]; }
    }
    __global__ void SSTAssemble(View v)
    {
        const I cell = Thread();
        if (cell >= v.owned) return;
        double source[2];
        SSTSource(v, v.q + cell * 2, v.flowGradient + cell * 12, v.qGradient + cell * 6, v.cells[cell].wallDistance, source);
        for (int k = 0; k < 2; ++k)
        {
            double sum = 0;
            for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
            {
                const auto &link = v.links[j];
                if (v.faces[link.face].left != v.faces[link.face].right)
                    sum += (link.side ? -1 : 1) * v.qFlux[link.face * 2 + k] / v.cells[cell].volume;
            }
            v.qRhs[cell * 2 + k] = sum + source[k];
        }
    }
    __device__ void MeanFlux(const View &v, const ResidentACMFace &face, const ResidentACMPoint &point,
                             const double *left, const double *rightNative, double eddy, double *result)
    {
        FluxInput input = v.s.flow;
        for (int k = 0; k < 4; ++k) input.left[k] = left[k];
        if (face.right >= 0) Rotate(face.rotation, rightNative, input.right);
        else Boundary(v, face.type, face.prescribed, point.normal, left, input.right);
        for (int d = 0; d < 3; ++d) input.normal[d] = point.normal[d];
        FluxOutput output;
        Detail::ACMFlux(input, output);
        if (v.s.viscous)
        {
            const double zero[12]{};
            AddViscous(v, point.normal, point.displacement, left, input.right, zero, zero, output.value, v.s.viscosity + eddy);
        }
        for (int k = 0; k < 4; ++k) result[k] = output.value[k];
    }
    __global__ void FaceJacobian(View v, double *blocks)
    {
        const I id = Thread();
        if (id >= v.faceCount) return;
        const auto &face = v.faces[id];
        double total[32]{};
        for (I p = face.begin; p < face.end; ++p)
            for (int side = 0; side < 2; ++side)
            {
                if (side && face.right < 0) continue;
                for (int variable = 0; variable < 4; ++variable)
                {
                    double left[4], right[4]{}, plus[4], minus[4];
                    for (int k = 0; k < 4; ++k)
                    {
                        left[k] = v.u[face.left * 4 + k];
                        if (face.right >= 0) right[k] = v.u[face.right * 4 + k];
                    }
                    double *value = side ? right : left;
                    const double epsilon = 1e-7 * fmax(1., fabs(value[variable]));
                    const double original = value[variable];
                    value[variable] = original + epsilon;
                    MeanFlux(v, face, v.points[p], left, right, v.s.sst ? v.eddy[p] : 0, plus);
                    value[variable] = original - epsilon;
                    MeanFlux(v, face, v.points[p], left, right, v.s.sst ? v.eddy[p] : 0, minus);
                    for (int row = 0; row < 4; ++row)
                        total[side * 16 + row * 4 + variable] += v.points[p].measure * (plus[row] - minus[row]) / (2 * epsilon);
                }
            }
        for (int k = 0; k < 32; ++k) blocks[id * 32 + k] = total[k];
    }
    __device__ double BDFCurrent(const View &v, const double *previous, const double *previousPrevious, I cell, int k, bool startup)
    {
        if (startup) return 1;
        if (k >= 4)
        {
            const I i = cell * 2 + k - 4;
            // GetPositiveTurbulenceBDFCoefficients: BE when the BDF2 forcing is below the floor.
            if ((2 * previous[i] - .5 * previousPrevious[i]) / 1.5 < v.s.turbulenceMinimum[k - 4]) return 1;
        }
        return 1.5;
    }
    __global__ void BDFDefect(View v, const double *previous, const double *previousPrevious,
                              const double *qPrevious, const double *qPreviousPrevious, bool startup)
    {
        const I cell = Thread();
        if (cell >= v.owned) return;
        const double a0 = startup ? 1 : 1.5, a1 = startup ? -1 : -2, a2 = startup ? 0 : .5;
        for (int k = 0; k < 3; ++k)
            v.rhs[cell * 4 + k] -= (a0 * v.u[cell * 4 + k] + a1 * previous[cell * 4 + k] + a2 * previousPrevious[cell * 4 + k]) / v.s.physicalStep;
        if (v.s.sst)
            for (int k = 0; k < 2; ++k)
            {
                const I i = cell * 2 + k;
                const double current = BDFCurrent(v, qPrevious, qPreviousPrevious, cell, k + 4, startup);
                v.qRhs[i] -= (current * v.q[i] + (current == 1 ? -1 : -2) * qPrevious[i] + (current == 1 ? 0 : .5) * qPreviousPrevious[i]) / v.s.physicalStep;
            }
    }
    __device__ void SolveBlock(double *matrix, double *rhs, int n)
    {
        for (int col = 0; col < n; ++col)
        {
            int pivot = col;
            for (int row = col + 1; row < n; ++row)
                if (fabs(matrix[row * n + col]) > fabs(matrix[pivot * n + col])) pivot = row;
            if (pivot != col)
            {
                for (int j = 0; j < n; ++j) { const double x = matrix[col * n + j]; matrix[col * n + j] = matrix[pivot * n + j]; matrix[pivot * n + j] = x; }
                const double x = rhs[col]; rhs[col] = rhs[pivot]; rhs[pivot] = x;
            }
            for (int row = col + 1; row < n; ++row)
            {
                const double factor = matrix[row * n + col] / matrix[col * n + col];
                for (int j = col + 1; j < n; ++j) matrix[row * n + j] -= factor * matrix[col * n + j];
                rhs[row] -= factor * rhs[col];
            }
        }
        for (int row = n - 1; row >= 0; --row)
        {
            for (int j = row + 1; j < n; ++j) rhs[row] -= matrix[row * n + j] * rhs[j];
            rhs[row] /= matrix[row * n + row];
        }
    }
    __global__ void ImplicitDiagonal(View v, const double *jac, double *diagonal, double *qDiagonal,
                                     const double *qPrevious, const double *qPreviousPrevious, bool startup)
    {
        const I cell = Thread();
        if (cell >= v.owned) return;
        double block[16]{}, qb[4]{};
        for (int k = 0; k < 3; ++k)
        {
            block[k * 4 + k] = 1 / v.dt[cell] + (startup ? 1 : 1.5) / v.s.physicalStep;
            block[k * 4 + 3] = (1 + v.s.flow.alpha) * v.u[cell * 4 + k] / (v.s.flow.beta2 * v.dt[cell]);
        }
        block[15] = 1 / (v.s.flow.beta2 * v.dt[cell]);
        for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
        {
            const auto &link = v.links[j];
            const double scale = (link.side ? -1 : 1) / v.cells[cell].volume;
            const double *own = jac + link.face * 32 + (link.side ? 16 : 0);
            const double *other = jac + link.face * 32 + (link.side ? 0 : 16);
            for (int k = 0; k < 16; ++k) block[k] += scale * (own[k] + (link.neighbor == cell ? other[k] : 0));
        }
        for (int k = 0; k < 16; ++k) diagonal[cell * 16 + k] = block[k];
        if (!v.s.sst) return;
        for (int variable = 0; variable < 2; ++variable)
        {
            const double original = v.q[cell * 2 + variable];
            const double epsilon = 1.4901161193847656e-8 * fmax(fabs(original), v.s.turbulenceMinimum[variable]);
            double plus[2]{v.q[cell * 2], v.q[cell * 2 + 1]}, minus[2]{plus[0], plus[1]}, sp[2], sm[2];
            plus[variable] += epsilon;
            minus[variable] = fmax(original - epsilon, v.s.turbulenceMinimum[variable]);
            SSTSource(v, plus, v.flowGradient + cell * 12, v.qGradient + cell * 6, v.cells[cell].wallDistance, sp);
            SSTSource(v, minus, v.flowGradient + cell * 12, v.qGradient + cell * 6, v.cells[cell].wallDistance, sm);
            for (int row = 0; row < 2; ++row) qb[row * 2 + variable] = -(sp[row] - sm[row]) / (plus[variable] - minus[variable]);
            qb[variable * 2 + variable] += 1 / (v.dt[cell] * v.s.turbulenceTimeScale) +
                BDFCurrent(v, qPrevious, qPreviousPrevious, cell, variable + 4, startup) / v.s.physicalStep;
        }
        for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
            for (int k = 0; k < 2; ++k) qb[k * 2 + k] += v.qRate[v.links[j].face * 2 + k] / v.cells[cell].volume;
        for (int k = 0; k < 4; ++k) qDiagonal[cell * 4 + k] = qb[k];
    }
    __global__ void SGSSweep(View v, const I *nodes, I begin, I end, const double *jac, const double *diagonal,
                             const double *linearRhs, double *work, bool backward)
    {
        const I position = begin + Thread();
        if (position >= end) return;
        const I cell = nodes[position];
        double value[4]{}, block[16];
        if (!backward)
            for (int k = 0; k < 4; ++k) value[k] = linearRhs[cell * 4 + k];
        for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
        {
            const auto &link = v.links[j];
            const I other = link.neighbor;
            if (other < 0 || other >= v.owned || (backward ? other <= cell : other >= cell)) continue;
            const double *coupling = jac + link.face * 32 + (link.side ? 0 : 16);
            const double scale = (link.side ? -1 : 1) / v.cells[cell].volume;
            for (int row = 0; row < 4; ++row)
                for (int col = 0; col < 4; ++col) value[row] -= scale * coupling[row * 4 + col] * work[other * 4 + col];
        }
        for (int k = 0; k < 16; ++k) block[k] = diagonal[cell * 16 + k];
        SolveBlock(block, value, 4);
        for (int k = 0; k < 4; ++k)
            work[cell * 4 + k] = backward ? work[cell * 4 + k] + value[k] : value[k];
    }
    __global__ void LinearProduct(View v, const double *jac, const double *diagonal, const double *qDiagonal,
                                  const double *increment, const double *qIncrement, double *residual, double *qResidual)
    {
        const I cell = Thread();
        if (cell >= v.owned) return;
        for (int row = 0; row < 4; ++row)
        {
            double value = 0;
            for (int col = 0; col < 4; ++col) value += diagonal[cell * 16 + row * 4 + col] * increment[cell * 4 + col];
            for (I j = v.cells[cell].begin; j < v.cells[cell].end; ++j)
            {
                const auto &link = v.links[j];
                if (link.neighbor < 0 || link.neighbor == cell) continue;
                const double scale = (link.side ? -1 : 1) / v.cells[cell].volume;
                const double *coupling = jac + link.face * 32 + (link.side ? 0 : 16);
                for (int col = 0; col < 4; ++col) value += scale * coupling[row * 4 + col] * increment[link.neighbor * 4 + col];
            }
            residual[cell * 4 + row] = v.rhs[cell * 4 + row] - value;
        }
        if (v.s.sst)
            for (int row = 0; row < 2; ++row)
            {
                double value = 0;
                for (int col = 0; col < 2; ++col) value += qDiagonal[cell * 4 + row * 2 + col] * qIncrement[cell * 2 + col];
                qResidual[cell * 2 + row] = v.qRhs[cell * 2 + row] - value;
            }
    }
    __global__ void AddLinearCorrection(View v, const double *work, const double *qDiagonal, const double *qResidual,
                                       double *increment, double *qIncrement)
    {
        const I cell = Thread();
        if (cell >= v.owned) return;
        for (int k = 0; k < 4; ++k) increment[cell * 4 + k] += work[cell * 4 + k];
        if (v.s.sst)
        {
            double matrix[4], value[2]{qResidual[cell * 2], qResidual[cell * 2 + 1]};
            for (int k = 0; k < 4; ++k) matrix[k] = qDiagonal[cell * 4 + k];
            SolveBlock(matrix, value, 2);
            for (int k = 0; k < 2; ++k) qIncrement[cell * 2 + k] += value[k];
        }
    }
    __global__ void ApplyNonlinearCorrection(View v, const double *increment, const double *qIncrement)
    {
        const I cell = Thread();
        if (cell >= v.owned) return;
        for (int k = 0; k < 4; ++k) v.u[cell * 4 + k] += v.s.implicitRelaxation * increment[cell * 4 + k];
        if (v.s.sst)
            for (int k = 0; k < 2; ++k)
            {
                const I i = cell * 2 + k;
                double du = v.s.implicitRelaxation * qIncrement[i];
                du = du < 0 ? fmax(du, -.9 * fmax(v.q[i] - v.s.turbulenceMinimum[k], 0.)) :
                    fmin(du, .9 * fmax(v.s.turbulenceMaximum[k] - v.q[i], 0.));
                v.q[i] += du;
            }
    }
    __global__ void EquationNorms(View v, const double *flow, const double *q, double *sums)
    {
        __shared__ double partial[128];
        const I cell = Thread();
        for (int k = 0; k < (v.s.sst ? 6 : 4); ++k)
        {
            const double value = cell >= v.owned ? 0 : k < 4 ? flow[cell * 4 + k] : v.s.flow.rho0 * q[cell * 2 + k - 4];
            partial[threadIdx.x] = value * value;
            __syncthreads();
            for (int stride = 64; stride; stride /= 2)
            {
                if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
                __syncthreads();
            }
            if (!threadIdx.x) atomicAdd(sums + k, partial[0]);
            __syncthreads();
        }
    }
}

struct ACMSSTWorkspace
{
    using I = DeviceIndex;
    ACMResidentDetail::View &v;
    ResidentDetail::Halo &halo;
    MPI_Comm comm;
    I globalOwned;
    ResidentDetail::Buffer<double> q, qGradient, flowGradient, qLimiter, eddy, volumeFlux, qFlux, qRate, qRhs;
    ResidentDetail::Buffer<double> jac, diagonal, qDiagonal, increment, qIncrement, linearResidual, qLinearResidual, work;
    ResidentDetail::Buffer<double> previous, previousPrevious, qPrevious, qPreviousPrevious, norms;
    const I *nodes;
    const std::vector<I> &levels;
    std::uint64_t scalarBytes = 0;
    std::size_t completed = 0;
    cudaGraphExec_t sgsGraph = nullptr;
    static unsigned Blocks(I n) { return unsigned((n + 127) / 128); }
    ACMSSTWorkspace(ACMResidentDetail::View &view, const ResidentACMData &data, ResidentDetail::Halo &transport,
                    MPI_Comm communicator, I global, const I *orderedNodes, const std::vector<I> &orderedLevels)
        : v(view), halo(transport), comm(communicator), globalOwned(global), nodes(orderedNodes), levels(orderedLevels)
    {
        using namespace ResidentDetail;
        if (v.s.integrator != 4)
            throw std::runtime_error("Resident SST currently requires BDF2DualTimeLUSGS; CPU fallback is disabled");
        previous.Upload(data.initialState); previousPrevious.Upload(data.initialState);
        if (v.s.sst)
        {
            q.Upload(data.initialTurbulence);
            qPrevious.Upload(data.initialTurbulence); qPreviousPrevious.Upload(data.initialTurbulence);
            qGradient.Resize(v.count * 6); flowGradient.Resize(v.count * 12); qLimiter.Resize(v.count * 3);
            eddy.Resize(data.points.size()); volumeFlux.Resize(data.points.size());
            qFlux.Resize(v.faceCount * 2); qRate.Resize(v.faceCount * 2); qRhs.Resize(v.count * 2);
            qDiagonal.Resize(v.owned * 4); qIncrement.Resize(v.count * 2); qLinearResidual.Resize(v.count * 2);
            v.q = q.data; v.qGradient = qGradient.data; v.flowGradient = flowGradient.data; v.qLimiter = qLimiter.data;
            v.eddy = eddy.data; v.volumeFlux = volumeFlux.data; v.qFlux = qFlux.data; v.qRate = qRate.data; v.qRhs = qRhs.data;
        }
        jac.Resize(v.faceCount * 32); diagonal.Resize(v.owned * 16);
        increment.Resize(v.count * 4); linearResidual.Resize(v.count * 4); work.Resize(v.count * 4); norms.Resize(6);
        // Capture the fixed local dependency schedule once. Replaying this graph
        // preserves the CPU LU-SGS ordering without one host launch per level.
        cudaStream_t stream;
        cudaGraph_t graph;
        Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create SGS capture stream");
        Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin SGS capture");
        for (std::size_t l = 0; l + 1 < levels.size(); ++l)
            if (levels[l + 1] > levels[l]) ACMResidentDetail::SGSSweep<<<Blocks(levels[l + 1] - levels[l]), 128, 0, stream>>>(v, nodes, levels[l], levels[l + 1], jac.data, diagonal.data, linearResidual.data, work.data, false);
        for (std::size_t l = levels.size() - 1; l > 0; --l)
            if (levels[l] > levels[l - 1]) ACMResidentDetail::SGSSweep<<<Blocks(levels[l] - levels[l - 1]), 128, 0, stream>>>(v, nodes, levels[l - 1], levels[l], jac.data, diagonal.data, linearResidual.data, work.data, true);
        Check(cudaStreamEndCapture(stream, &graph), "finish SGS capture");
        Check(cudaGraphInstantiate(&sgsGraph, graph, nullptr, nullptr, 0), "instantiate SGS graph");
        Check(cudaGraphDestroy(graph), "release SGS capture graph");
        Check(cudaStreamDestroy(stream), "release SGS capture stream");
        Check(cudaGraphUpload(sgsGraph, nullptr), "upload SGS execution graph");
    }
    ~ACMSSTWorkspace()
    {
        if (sgsGraph) cudaGraphExecDestroy(sgsGraph);
    }
    void Prepare()
    {
        using namespace ACMResidentDetail;
        using namespace ResidentDetail;
        if (!v.s.sst) return;
        halo.Pull(v.q, 2);
        if (v.owned) SSTGradients<<<Blocks(v.owned), 128>>>(v);
        halo.Pull(v.qGradient, 6); halo.Pull(v.flowGradient, 12);
        if (v.owned) SSTLimiter<<<Blocks(v.owned), 128>>>(v);
        halo.Pull(v.qLimiter, 3);
        if (v.faceCount) SSTEddy<<<Blocks(v.faceCount), 128>>>(v);
        Check(cudaGetLastError(), "SST prepare kernels");
    }
    void Residual()
    {
        using namespace ACMResidentDetail;
        if (!v.s.sst) return;
        if (v.faceCount) SSTFlux<<<Blocks(v.faceCount), 128>>>(v);
        if (v.owned) SSTAssemble<<<Blocks(v.owned), 128>>>(v);
    }
    std::array<double, 6> Norm(const double *flow, const double *turbulence)
    {
        using namespace ResidentDetail;
        Check(cudaMemset(norms.data, 0, 6 * sizeof(double)), "reset SST equation reductions");
        if (v.owned) ACMResidentDetail::EquationNorms<<<Blocks(v.owned), 128>>>(v, flow, turbulence, norms.data);
        std::array<double, 6> local{}, global{};
        Check(cudaMemcpy(local.data(), norms.data, 6 * sizeof(double), cudaMemcpyDeviceToHost), "read SST equation reductions");
        MPIcheck(MPI_Allreduce(local.data(), global.data(), 6, MPI_DOUBLE, MPI_SUM, comm));
        scalarBytes += 48;
        for (auto &x : global) x = std::sqrt(x / std::max<I>(1, globalOwned));
        for (double x : global)
            if (!std::isfinite(x)) throw std::runtime_error("Non-finite resident BDF/SST equation defect");
        return global;
    }
    static double Group(const std::array<double, 6> &x, int first, int last)
    {
        double sum = 0;
        for (int k = first; k < last; ++k) sum += x[k] * x[k];
        return std::sqrt(sum / (last - first));
    }
    bool Converged(const ResidentACMReport &report) const
    {
        const auto target = [&](double absolute, double relative)
        {
            return v.s.convergenceCriterion == 0 ? absolute : v.s.convergenceCriterion == 1 ? relative : std::max(absolute, relative);
        };
        for (int k = 0; k < (v.s.sst ? 6 : 4); ++k)
            if ((k < 4 ? v.s.flowRelativeTolerance : v.s.turbulenceRelativeTolerance) == 0 &&
                report.equationNorms[k] > target(v.s.implicitTolerance * v.s.residualScales[k],
                    v.s.relativeTolerances[k] * report.initialEquationNorms[k])) return false;
        if (v.s.flowRelativeTolerance > 0 && report.finalNorm > target(v.s.implicitTolerance, v.s.flowRelativeTolerance * report.initialNorm)) return false;
        if (v.s.sst && v.s.turbulenceRelativeTolerance > 0 && report.turbulenceNorm > target(v.s.implicitTolerance, v.s.turbulenceRelativeTolerance * report.initialTurbulenceNorm)) return false;
        return true;
    }
    template <class EvaluateResidual>
    ResidentACMReport Advance(EvaluateResidual &&evaluate)
    {
        using namespace ACMResidentDetail;
        using namespace ResidentDetail;
        const auto defect = [&]()
        {
            evaluate();
            if (v.owned) BDFDefect<<<Blocks(v.owned), 128>>>(v, previous.data, previousPrevious.data, qPrevious.data, qPreviousPrevious.data, completed == 0);
        };
        defect();
        ResidentACMReport report;
        report.equationNorms = report.initialEquationNorms = Norm(v.rhs, v.qRhs);
        report.initialNorm = report.finalNorm = Group(report.equationNorms, 0, 4);
        report.initialTurbulenceNorm = report.turbulenceNorm = v.s.sst ? Group(report.equationNorms, 4, 6) : 0;
        for (int iteration = 0; iteration < v.s.implicitIterations && !Converged(report); ++iteration)
        {
            if (v.faceCount) FaceJacobian<<<Blocks(v.faceCount), 128>>>(v, jac.data);
            if (v.owned) ImplicitDiagonal<<<Blocks(v.owned), 128>>>(v, jac.data, diagonal.data, qDiagonal.data, qPrevious.data, qPreviousPrevious.data, completed == 0);
            Check(cudaMemset(increment.data, 0, increment.count * sizeof(double)), "reset BDF correction");
            if (v.s.sst) Check(cudaMemset(qIncrement.data, 0, qIncrement.count * sizeof(double)), "reset SST correction");
            Check(cudaMemcpy(linearResidual.data, v.rhs, v.count * 4 * sizeof(double), cudaMemcpyDeviceToDevice), "copy BDF linear RHS");
            if (v.s.sst) Check(cudaMemcpy(qLinearResidual.data, v.qRhs, v.count * 2 * sizeof(double), cudaMemcpyDeviceToDevice), "copy SST linear RHS");
            const auto initial = Norm(linearResidual.data, qLinearResidual.data);
            const double linearInitial = Group(initial, 0, v.s.sst ? 6 : 4);
            const double linearTarget = std::max(v.s.linearAbsoluteTolerance, v.s.linearRelativeTolerance * linearInitial);
            double linearNorm = linearInitial;
            for (int sweep = 0; sweep < v.s.lusgsSweeps && linearNorm > linearTarget; ++sweep)
            {
                Check(cudaGraphLaunch(sgsGraph, nullptr), "launch resident SGS graph");
                if (v.owned) AddLinearCorrection<<<Blocks(v.owned), 128>>>(v, work.data, qDiagonal.data, qLinearResidual.data, increment.data, qIncrement.data);
                halo.Pull(increment.data, 4);
                if (v.owned) LinearProduct<<<Blocks(v.owned), 128>>>(v, jac.data, diagonal.data, qDiagonal.data, increment.data, qIncrement.data, linearResidual.data, qLinearResidual.data);
                linearNorm = Group(Norm(linearResidual.data, qLinearResidual.data), 0, v.s.sst ? 6 : 4);
            }
            if (v.owned) ApplyNonlinearCorrection<<<Blocks(v.owned), 128>>>(v, increment.data, qIncrement.data);
            defect();
            report.equationNorms = Norm(v.rhs, v.qRhs);
            report.finalNorm = Group(report.equationNorms, 0, 4);
            report.turbulenceNorm = v.s.sst ? Group(report.equationNorms, 4, 6) : 0;
            report.iterations = iteration + 1;
        }
        report.converged = Converged(report);
        if (!report.converged && v.s.requireConvergence)
            throw std::runtime_error("Resident BDF physical step did not converge; histories were not committed");
        const auto shift = [&](Buffer<double> &old, Buffer<double> &older, const double *current)
        {
            Check(cudaMemcpy(older.data, old.data, old.count * sizeof(double), cudaMemcpyDeviceToDevice), "shift BDF previous history");
            Check(cudaMemcpy(old.data, current, old.count * sizeof(double), cudaMemcpyDeviceToDevice), "commit BDF completed history");
        };
        shift(previous, previousPrevious, v.u);
        if (v.s.sst) shift(qPrevious, qPreviousPrevious, v.q);
        ++completed;
        return report;
    }
};
