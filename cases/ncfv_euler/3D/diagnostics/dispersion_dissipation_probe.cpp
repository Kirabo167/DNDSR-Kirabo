/** Semi-discrete Fourier response of the production NCFV Euler operator. */
#include "NCFV/NCFVSolver.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iostream>

using DNDS::real;
using Index = DNDS::index;
using State = Eigen::Matrix<real, 5, 1>;
using Vector = Eigen::Matrix<real, 3, 1>;
using Complex = std::complex<real>;
using namespace DNDS::NCFV;

namespace
{
    void Allocate(NodeStatePair &field, const char *name,
                  const DNDS::MPIInfo &mpi, Index owned, const NodeHalo &halo)
    {
        field.InitPair(name, mpi);
        field.father->Resize(owned, 5, 1);
        field.son->Resize(halo.NumNodeGhost(), 5, 1);
        field.BorrowSetup(halo.Layout());
        field.trans.initPersistentPull();
        for (Index i = 0; i < field.Size(); i++)
            field[i].setZero();
    }

    // For barycentric lambda on a tetrahedron, E(exp(sum lambda_j z_j))
    // = 6 sum_n h_n(z_0,...,z_3)/(n+3)!, where h_n is the complete
    // homogeneous polynomial. Centering the real phases keeps this series
    // stable at repeated vertices and at small wave numbers.
    Complex TetrahedronMean(const MicroVolume &micro, const Vector &k,
                            real &maximumTail)
    {
        constexpr int Order = 40;
        std::array<real, 4> phases{};
        DNDS_check_throw_info(micro.nPoints == 4,
                              "Fourier probe requires tetrahedral micro volumes");
        real center = 0;
        for (int j = 0; j < 4; j++)
        {
            phases[j] = k.dot(micro.points[j]);
            center += phases[j] / 4;
        }
        std::array<real, Order + 1> homogeneous{};
        homogeneous[0] = 1;
        for (real phase : phases)
        {
            const real offset = phase - center;
            DNDS_check_throw_info(std::abs(offset) < 8,
                                  "Wave exceeds the integration series range");
            for (int n = 1; n <= Order; n++)
                homogeneous[n] += offset * homogeneous[n - 1];
        }
        Complex sum = 1;
        real coefficient = 1, tail = 0;
        const std::array<Complex, 4> powers{
            Complex(1, 0), Complex(0, 1), Complex(-1, 0), Complex(0, -1)};
        for (int n = 1; n <= Order; n++)
        {
            coefficient /= n + 3;
            const real term = coefficient * homogeneous[n];
            sum += term * powers[n % 4];
            if (n > Order - 4)
                tail += std::abs(term);
        }
        maximumTail = std::max(maximumTail, tail);
        DNDS_check_throw_info(tail < 1e-12 && std::abs(sum) <= 1 + 1e-10,
                              "Fourier dual-volume integration failed to converge");
        return std::exp(Complex(0, center)) * sum;
    }

    real GlobalMaximum(real value, const DNDS::MPIInfo &mpi)
    {
        real result = 0;
        MPI_Allreduce(&value, &result, 1, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        return result;
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        try
        {
            DNDS_check_throw_info(argc >= 7 && argc <= 9,
                "usage: config output method Traditional|Efficient amplitude max-harmonic [mesh [xyz-max-harmonic]]");
            const std::filesystem::path output(argv[2]);
            DNDS_check_throw_info(!std::filesystem::exists(output),
                                  "Refusing to overwrite a Fourier result");
            std::ifstream input(argv[1]);
            DNDS_check_throw_info(input.good(), "Cannot read configuration");
            nlohmann::ordered_json resolved = Configuration{};
            resolved.merge_patch(nlohmann::ordered_json::parse(input));
            for (auto &boundary : resolved["physics"]["boundaryZones"])
            {
                nlohmann::ordered_json normalized = BoundaryZoneSettings{};
                normalized.merge_patch(boundary);
                boundary = std::move(normalized);
            }
            auto cfg = resolved.get<Configuration>();
            const std::string method(argv[3]), integration(argv[4]);
            DNDS_check_throw_info(method == "LeastSquares" || method == "SVDLeastSquares",
                                  "Select LS or SVDLS reconstruction");
            DNDS_check_throw_info(integration == "Traditional" || integration == "Efficient",
                                  "Select Traditional or Efficient integration");
            const real amplitude = std::stod(argv[5]);
            const int maxHarmonic = std::stoi(argv[6]);
            const int xyzMaxHarmonic = argc == 9 ? std::stoi(argv[8]) : maxHarmonic;
            DNDS_check_throw_info(amplitude > 0 && amplitude <= 1e-3 &&
                                  maxHarmonic > 0 && xyzMaxHarmonic > 0 &&
                                  xyzMaxHarmonic <= maxHarmonic,
                                  "Invalid perturbation or harmonic range");
            if (argc >= 8)
                cfg.mesh.meshFile = argv[7];
            cfg.reconstruction.method = method == "LeastSquares" ?
                ReconstructionMethod::LeastSquares : ReconstructionMethod::SVDLeastSquares;
            // Match the default 1.7 quadratic stencil factor and weight power.
            cfg.reconstruction.stencilSizeFactor = 1.7;
            cfg.reconstruction.distanceWeightPower = 1;
            cfg.reconstruction.enableLimiter = false;
            cfg.algorithm.mode = integration == "Traditional" ?
                IntegrationMode::TraditionalQuadrature : IntegrationMode::EfficientDifferential;
            cfg.algorithm.retainMicroGeometry = true;
            cfg.algorithm.quadratureOrder = 4;
            cfg.algorithm.surfaceQuadratureOrder = 3;
            cfg.physics.gamma = 1.4;
            cfg.physics.riemannSolver = DNDS::Euler::Gas::Roe;
            cfg.physics.viscous.enabled = false;
            cfg.physics.initialPrimitive = {1, 1, 1, 0, 1};
            cfg.physics.farFieldPrimitive = cfg.physics.initialPrimitive;
            cfg.initialField.isentropicVortex = false;
            cfg.initialField.nodeFile.clear();
            cfg.io.writeVTK = false;
            cfg.io.writeInitial = false;
            cfg.io.writeFinal = false;
            cfg.io.writeFinalRestart = false;
            cfg.io.writeResolvedConfiguration = false;
            cfg.io.outputInterval = 0;
            cfg.io.restartInterval = 0;
            for (real length : cfg.mesh.periodicLengths)
                DNDS_check_throw_info(length > 0, "Three periodic lengths are required");

            Solver<3> solver(mpi, cfg);
            solver.Initialize();
            auto &mesh = *solver.Mesh();
            const auto &geometry = solver.Geometry();
            const auto &halo = solver.NodeCommunication();
            SpatialOperator<3> spatial(mpi, solver.Mesh(), solver.EdgeTopology(), geometry,
                                       solver.ReconstructionData(), solver.Boundaries(),
                                       cfg.algorithm.mode, cfg.reconstruction,
                                       cfg.physics, cfg.time, halo);
            spatial.Initialize();
            NodeStatePair state, rhs, derivativeReal, derivativeImag, plus;
            Allocate(state, "fourier.state", mpi, mesh.NumNode(), halo);
            Allocate(rhs, "fourier.rhs", mpi, mesh.NumNode(), halo);
            Allocate(derivativeReal, "fourier.real", mpi, mesh.NumNode(), halo);
            Allocate(derivativeImag, "fourier.imag", mpi, mesh.NumNode(), halo);
            Allocate(plus, "fourier.plus", mpi, mesh.NumNode(), halo);
            const Vector velocity{1, 1, 0};
            const real gamma = cfg.physics.gamma, sound = std::sqrt(gamma);
            State background;
            background << 1, velocity, 1 / (gamma - 1) + 0.5 * velocity.squaredNorm();
            real localVolume = 0, localUniform = 0;
            Index localUnknowns = 0;
            for (Index i = 0; i < mesh.NumNode(); i++)
            {
                const auto &volume = geometry.NodeVolume(i);
                DNDS_check_throw_info(volume.boundaryPieces.empty(),
                                      "Fourier probe requires fully stitched periodic geometry");
                localVolume += volume.moments.measure;
                localUnknowns += volume.moments.measure > 0;
                state[i] = background;
            }
            spatial.EvaluateRHS(state, rhs, false, false);
            for (Index i = 0; i < mesh.NumNode(); i++)
                if (geometry.NodeVolume(i).moments.measure > 0)
                    localUniform = std::max(localUniform, rhs[i].cwiseAbs().maxCoeff());
            real globalVolume = 0;
            Index globalUnknowns = 0;
            MPI_Allreduce(&localVolume, &globalVolume, 1, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(&localUnknowns, &globalUnknowns, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            const real uniformMaximum = GlobalMaximum(localUniform, mpi);
            DNDS_check_throw_info(uniformMaximum < 1e-10,
                                  "Constant state residual is too large for a wave measurement");

            auto rows = nlohmann::ordered_json::array();
            const std::array<std::string, 4> directions{"x", "y", "xy", "xyz"};
            for (const auto &direction : directions)
                for (int harmonic = 1;
                     harmonic <= (direction == "xyz" ? xyzMaxHarmonic : maxHarmonic);
                     harmonic++)
                {
                    Vector k = Vector::Zero();
                    if (direction != "y")
                        k.x() = 2 * DNDS::pi * harmonic / cfg.mesh.periodicLengths[0];
                    if (direction != "x")
                        k.y() = 2 * DNDS::pi * harmonic / cfg.mesh.periodicLengths[1];
                    if (direction == "xyz")
                        k.z() = 2 * DNDS::pi * harmonic / cfg.mesh.periodicLengths[2];
                    const Vector normal = k.normalized();
                    std::vector<Complex> phi(static_cast<std::size_t>(mesh.NumNode()));
                    real localTail = 0, localWeightMismatch = 0;
                    for (Index i = 0; i < mesh.NumNode(); i++)
                    {
                        const auto &volume = geometry.NodeVolume(i);
                        if (volume.moments.measure <= 0)
                            continue;
                        real measure = 0;
                        for (const auto &micro : volume.microVolumes)
                        {
                            phi[i] += micro.measure * TetrahedronMean(micro, k, localTail);
                            measure += micro.measure;
                        }
                        localWeightMismatch = std::max(localWeightMismatch,
                            std::abs(measure / volume.moments.measure - 1));
                        phi[i] /= volume.moments.measure;
                    }
                    const real tail = GlobalMaximum(localTail, mpi);
                    const real weightMismatch = GlobalMaximum(localWeightMismatch, mpi);
                    DNDS_check_throw_info(weightMismatch < 1e-10,
                                          "Retained micro volumes do not cover the dual volume");

                    for (const std::string wave : {"entropy", "acoustic_plus"})
                    {
                        State right, left;
                        right << 1, velocity, 0.5 * velocity.squaredNorm();
                        left << 1 - 0.5 * (gamma - 1) * velocity.squaredNorm() / (sound * sound),
                            (gamma - 1) * velocity / (sound * sound),
                            -(gamma - 1) / (sound * sound);
                        real omega = k.dot(velocity);
                        if (wave == "acoustic_plus")
                        {
                            const real enthalpy = gamma / (gamma - 1) + 0.5 * velocity.squaredNorm();
                            right << 1, velocity + sound * normal,
                                enthalpy + sound * normal.dot(velocity);
                            left << 0.5 * (0.5 * (gamma - 1) * velocity.squaredNorm() /
                                         (sound * sound) - normal.dot(velocity) / sound),
                                0.5 * (normal / sound - (gamma - 1) * velocity / (sound * sound)),
                                0.5 * (gamma - 1) / (sound * sound);
                            omega += sound * k.norm();
                        }
                        DNDS_check_throw_info(std::abs(left.dot(right) - 1) < 1e-12,
                                              "Characteristic projection is not normalized");
                        const double started = MPI_Wtime();
                        real localAsymmetrySquared = 0, localDerivativeSquared = 0;
                        for (int component = 0; component < 2; component++)
                        {
                            auto &derivative = component == 0 ? derivativeReal : derivativeImag;
                            for (int sign : {1, -1})
                            {
                                for (Index i = 0; i < mesh.NumNode(); i++)
                                {
                                    const real phase = component == 0 ? phi[i].real() : phi[i].imag();
                                    state[i] = background + sign * amplitude * phase * right;
                                }
                                spatial.EvaluateRHS(state, rhs, false, false);
                                for (Index i = 0; i < mesh.NumNode(); i++)
                                    if (sign == 1)
                                        plus[i] = rhs[i];
                                    else
                                    {
                                        derivative[i] = (plus[i] - rhs[i]) / (2 * amplitude);
                                        const real volume = geometry.NodeVolume(i).moments.measure;
                                        localAsymmetrySquared += volume *
                                            ((plus[i] + rhs[i]) / (2 * amplitude)).squaredNorm();
                                        localDerivativeSquared += volume * derivative[i].squaredNorm();
                                    }
                            }
                        }
                        real sums[7]{}, global[7]{};
                        for (Index i = 0; i < mesh.NumNode(); i++)
                        {
                            const real volume = geometry.NodeVolume(i).moments.measure;
                            const Complex response(left.dot(State(derivativeReal[i])),
                                                   left.dot(State(derivativeImag[i])));
                            const Complex projection = volume * std::conj(phi[i]) * response;
                            sums[0] += projection.real();
                            sums[1] += projection.imag();
                            sums[2] += volume * std::norm(phi[i]);
                            sums[3] += volume * std::norm(response);
                            sums[4] += volume * response.real();
                            sums[5] += volume * response.imag();
                        }
                        MPI_Allreduce(sums, global, 7, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                        DNDS_check_throw_info(global[2] > globalVolume * 1e-10,
                                              "Wave has a vanishing discrete norm (aliasing)");
                        const Complex lambda(global[0] / global[2], global[1] / global[2]);
                        real errors[4]{}, globalErrors[4]{};
                        for (Index i = 0; i < mesh.NumNode(); i++)
                        {
                            const real volume = geometry.NodeVolume(i).moments.measure;
                            const State realError = State(derivativeReal[i]) - right * (lambda * phi[i]).real();
                            const State imagError = State(derivativeImag[i]) - right * (lambda * phi[i]).imag();
                            errors[0] += volume * (realError.squaredNorm() + imagError.squaredNorm());
                            const Complex scalar(left.dot(State(derivativeReal[i])),
                                                 left.dot(State(derivativeImag[i])));
                            errors[1] += volume * std::norm(scalar - lambda * phi[i]);
                        }
                        errors[2] = localAsymmetrySquared;
                        errors[3] = localDerivativeSquared;
                        MPI_Allreduce(errors, globalErrors, 4, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                        const real omegaNumerical = -lambda.imag();
                        const real damping = -lambda.real();
                        const real norm = std::sqrt(global[2]);
                        const real leakage = std::sqrt(globalErrors[0]) /
                            (std::abs(omega) * right.norm() * norm);
                        const real modeLeakage = std::sqrt(globalErrors[1]) / (std::abs(omega) * norm);
                        const real constantLeakage = std::hypot(global[4], global[5]) /
                            (std::abs(omega) * std::sqrt(globalVolume) * norm);
                        rows.push_back({
                            {"mode", wave}, {"direction", direction}, {"harmonic", harmonic},
                            {"k", std::vector<real>(k.data(), k.data() + 3)},
                            {"omega_exact", omega}, {"lambda_real", lambda.real()},
                            {"lambda_imag", lambda.imag()}, {"phase_speed_ratio", omegaNumerical / omega},
                            {"damping_ratio", damping / std::abs(omega)},
                            {"attenuation_per_period", std::exp(-2 * DNDS::pi * damping / std::abs(omega))},
                            {"leakage_ratio", leakage}, {"mode_leakage_ratio", modeLeakage},
                            {"constant_leakage", constantLeakage},
                            {"central_difference_asymmetry", std::sqrt(globalErrors[2] /
                                std::max(globalErrors[3], real(1e-300)))},
                            {"phi_norm", norm / std::sqrt(globalVolume)},
                            {"phi_integration_tail", tail}, {"micro_volume_weight_mismatch", weightMismatch},
                            {"rhs_seconds", GlobalMaximum(MPI_Wtime() - started, mpi)}});
                        if (mpi.rank == 0)
                            std::cout << direction << ' ' << harmonic << ' ' << wave
                                      << " c/c0=" << omegaNumerical / omega
                                      << " D/omega=" << damping / std::abs(omega)
                                      << " leakage=" << leakage << std::endl;
                    }
                }
            const Index globalNodes = mesh.NumNodeGlobal();
            const Index globalCells = mesh.NumCellGlobal();
            if (mpi.rank == 0)
            {
                const nlohmann::ordered_json result{
                    {"configuration", cfg}, {"method", method}, {"integration", integration},
                    {"mpi_ranks", mpi.size}, {"nodes", globalNodes},
                    {"cells", globalCells}, {"periodic_unknowns", globalUnknowns},
                    {"volume", globalVolume}, {"amplitude", amplitude},
                    {"uniform_rhs_max", uniformMaximum}, {"rows", rows}};
                std::ofstream stream(output);
                DNDS_check_throw_info(stream.good(), "Cannot write Fourier output");
                stream << result.dump(2) << '\n';
            }
        }
        catch (const std::exception &error)
        {
            std::cerr << "Fourier probe rank " << mpi.rank << ": " << error.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
