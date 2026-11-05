/** Compare the configured production reconstruction and an explicit normal equation. */
#include "NCFV/NCFVAnalytic.hpp"
#include "NCFV/NCFVSolver.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>

using DNDS::real;
using Index = DNDS::index;
using namespace DNDS::NCFV;
using State = Eigen::Matrix<real, 5, 1>;
using Gradient = Eigen::Matrix<real, 3, 5>;

namespace
{
    struct ErrorAccumulator
    {
        real absolute = 0;
        real squared = 0;
        real maximum = 0;

        void Add(real error, real volume)
        {
            absolute += volume * std::abs(error);
            squared += volume * error * error;
            maximum = std::max(maximum, std::abs(error));
        }
    };

    void AllocateGradient(
        NodeMatrixPair &field,
        const std::string &name,
        const DNDS::MPIInfo &mpi,
        const DNDS::ssp<DNDS::Geom::UnstructuredMesh> &mesh,
        const NodeHalo &halo)
    {
        field.InitPair(name, mpi);
        field.father->Resize(mesh->NumNode(), 3, 5);
        field.son->Resize(halo.NumNodeGhost(), 3, 5);
        field.BorrowSetup(halo.Layout());
        field.trans.initPersistentPull();
    }

    void AllocateState(
        NodeStatePair &field,
        const std::string &name,
        const DNDS::MPIInfo &mpi,
        const DNDS::ssp<DNDS::Geom::UnstructuredMesh> &mesh,
        const NodeHalo &halo)
    {
        field.InitPair(name, mpi);
        field.father->Resize(mesh->NumNode(), 5, 1);
        field.son->Resize(halo.NumNodeGhost(), 5, 1);
        field.BorrowSetup(halo.Layout());
        field.trans.initPersistentPull();
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
            DNDS_check_throw_info(
                argc == 3 || argc == 4 || argc == 5,
                "Usage: normal_equation_reconstruction_probe config.json output-directory [target-stencil-size [reconstruction-method]]");
            auto cfg = LoadConfiguration(argv[1], {}, {}).configuration;
            DNDS_check_throw_info(
                cfg.dimension == 3 && cfg.initialField.isentropicVortex &&
                    cfg.algorithm.mode == IntegrationMode::EfficientDifferential &&
                    !cfg.reconstruction.enableLimiter && cfg.io.restartInput.empty(),
                "Use a fresh, unlimited, three-dimensional efficient vortex configuration");

            constexpr int nBasis = 9;
            const int targetStencil = argc >= 4 ? std::stoi(argv[3]) : 15;
            if (argc == 5)
                cfg.reconstruction.method =
                    nlohmann::ordered_json(argv[4])
                        .get<ReconstructionMethod>();
            DNDS_check_throw_info(targetStencil >= nBasis,
                                  "Target stencil is smaller than the quadratic basis");
            cfg.reconstruction.stencilSizeFactor = std::nextafter(
                static_cast<real>(targetStencil) / nBasis, 0.0);
            DNDS_check_throw_info(
                static_cast<int>(std::ceil(
                    cfg.reconstruction.stencilSizeFactor * nBasis)) == targetStencil,
                "Failed to encode the requested stencil size");
            cfg.time.iterations = 0;
            cfg.time.endTime = 0;
            cfg.io.writeVTK = false;
            cfg.io.writeInitial = false;
            cfg.io.writeFinal = false;
            cfg.io.writeFinalRestart = false;
            cfg.io.restartInterval = 0;
            cfg.io.writeResolvedConfiguration = false;

            const std::filesystem::path output = argv[2];
            if (mpi.rank == 0)
            {
                DNDS_check_throw_info(
                    !std::filesystem::exists(output),
                    "Refusing to overwrite a diagnostic output directory");
                std::filesystem::create_directories(output);
            }
            MPI_Barrier(mpi.comm);

            Solver<3> solver(mpi, cfg);
            solver.Initialize();
            const auto mesh = solver.Mesh();
            const auto &geometry = solver.Geometry();
            const auto &reconstruction = solver.ReconstructionData();
            const auto &halo = solver.NodeCommunication();
            const auto &means = solver.StateField();

            NodeMatrixPair configuredGradients, normalGradients, unusedCoefficients;
            NodeStatePair configuredPoints, normalPoints;
            AllocateGradient(configuredGradients, "NCFV.normalProbe.configuredGradients", mpi, mesh, halo);
            AllocateGradient(normalGradients, "NCFV.normalProbe.normalGradients", mpi, mesh, halo);
            AllocateState(configuredPoints, "NCFV.normalProbe.configuredPoints", mpi, mesh, halo);
            AllocateState(normalPoints, "NCFV.normalProbe.normalPoints", mpi, mesh, halo);

            reconstruction.ComputeCoefficients(
                means, configuredGradients, unusedCoefficients);
            configuredGradients.trans.startPersistentPull();
            configuredGradients.trans.waitPersistentPull();
            reconstruction.RecoverPointValues(
                means, configuredGradients, unusedCoefficients, configuredPoints);

            real localNormalProjectorDefect = 0;
            real localNormalCondition = 0;
            real localGradientDifference = 0;
            for (Index iNode = 0; iNode < mesh->NumNode(); iNode++)
            {
                const auto &op = reconstruction.Operator(iNode);
                const int count = static_cast<int>(op.stencil.size());
                Eigen::MatrixXd matrix(count, nBasis);
                Eigen::VectorXd weights(count);
                Eigen::MatrixXd differences(count, 5);
                for (int row = 0; row < count; row++)
                {
                    const Index neighbor = op.stencil[static_cast<std::size_t>(row)];
                    matrix.row(row) =
                        (Reconstruction::MeanBasis(
                             halo.MomentsRelative(iNode, neighbor),
                             Vector3::Zero(), op.referenceLengths, 3) -
                         op.targetBasisMean)
                            .transpose();
                    const real normalizedDistance = std::max(
                        halo.Displacement(iNode, neighbor).norm() / op.lengthScale,
                        cfg.reconstruction.distanceWeightFloor);
                    weights(row) = std::pow(
                        normalizedDistance,
                        -cfg.reconstruction.distanceWeightPower);
                    differences.row(row) =
                        (means[neighbor] - means[iNode]).transpose();
                }

                const Eigen::MatrixXd weighted = weights.asDiagonal() * matrix;
                const Eigen::MatrixXd normalMatrix = weighted.transpose() * weighted;
                const Eigen::MatrixXd normalInverseRows =
                    normalMatrix.inverse() * weighted.transpose() *
                    weights.asDiagonal();
                DNDS_check_throw_info(
                    normalInverseRows.allFinite(),
                    "Normal-equation inverse produced a nonfinite reconstruction operator");
                const Eigen::MatrixXd reconstructed =
                    normalInverseRows.topRows(3) * differences;
                normalGradients[iNode] =
                    op.referenceLengths.head<3>().cwiseInverse().asDiagonal() *
                    reconstructed;

                localNormalProjectorDefect = std::max(
                    localNormalProjectorDefect,
                    (normalInverseRows * matrix -
                     Eigen::MatrixXd::Identity(nBasis, nBasis))
                        .cwiseAbs()
                        .maxCoeff());
                localNormalCondition = std::max(
                    localNormalCondition, op.conditionNumber * op.conditionNumber);
                localGradientDifference = std::max(
                    localGradientDifference,
                    (Gradient(normalGradients[iNode]) -
                     Gradient(configuredGradients[iNode]))
                        .cwiseAbs()
                        .maxCoeff());
            }
            normalGradients.trans.startPersistentPull();
            normalGradients.trans.waitPersistentPull();
            reconstruction.RecoverPointValues(
                means, normalGradients, unusedCoefficients, normalPoints);
            configuredPoints.trans.startPersistentPull();
            normalPoints.trans.startPersistentPull();
            configuredPoints.trans.waitPersistentPull();
            normalPoints.trans.waitPersistentPull();

            std::map<std::string, ErrorAccumulator> errors;
            real localVolume = 0;
            real localPointDifference = 0;
            int localStencilMinimum = std::numeric_limits<int>::max();
            int localStencilMaximum = 0;
            Index localStencilSum = 0;
            for (Index iNode = 0; iNode < mesh->NumNode(); iNode++)
            {
                const real volume = geometry.NodeVolume(iNode).moments.measure;
                const auto [exact, exactGradient] =
                    IsentropicVortex<3>(cfg, mesh->coords[iNode], 0);
                for (int iVar = 0; iVar < 5; iVar++)
                {
                    errors[fmt::format("configured.u{}.value", iVar)].Add(
                        configuredPoints[iNode](iVar) - exact(iVar), volume);
                    errors[fmt::format("normal.u{}.value", iVar)].Add(
                        normalPoints[iNode](iVar) - exact(iVar), volume);
                    errors[fmt::format("configured.u{}.gradient", iVar)].Add(
                        (Gradient(configuredGradients[iNode]).col(iVar) -
                         exactGradient.col(iVar))
                            .norm(),
                        volume);
                    errors[fmt::format("normal.u{}.gradient", iVar)].Add(
                        (Gradient(normalGradients[iNode]).col(iVar) -
                         exactGradient.col(iVar))
                            .norm(),
                        volume);
                }
                localPointDifference = std::max(
                    localPointDifference,
                    (State(normalPoints[iNode]) - State(configuredPoints[iNode]))
                        .cwiseAbs()
                        .maxCoeff());
                localVolume += volume;
                const int stencilSize = static_cast<int>(
                    reconstruction.Operator(iNode).stencil.size());
                localStencilMinimum = std::min(
                    localStencilMinimum, stencilSize);
                localStencilMaximum = std::max(
                    localStencilMaximum, stencilSize);
                localStencilSum += stencilSize;
            }

            real volume = 0;
            MPI_Allreduce(&localVolume, &volume, 1,
                          DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            real localMaxima[]{localNormalProjectorDefect,
                               localNormalCondition,
                               localGradientDifference,
                               localPointDifference};
            real maxima[4]{};
            MPI_Allreduce(localMaxima, maxima, 4,
                          DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            int stencilMinimum = 0;
            int stencilMaximum = 0;
            Index stencilSum = 0;
            MPI_Allreduce(&localStencilMinimum, &stencilMinimum, 1,
                          MPI_INT, MPI_MIN, mpi.comm);
            MPI_Allreduce(&localStencilMaximum, &stencilMaximum, 1,
                          MPI_INT, MPI_MAX, mpi.comm);
            MPI_Allreduce(&localStencilSum, &stencilSum, 1,
                          DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);

            nlohmann::ordered_json reducedErrors;
            for (const auto &[name, local] : errors)
            {
                real localSums[]{local.absolute, local.squared};
                real sums[2]{};
                real maximum = 0;
                MPI_Allreduce(localSums, sums, 2,
                              DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                MPI_Allreduce(&local.maximum, &maximum, 1,
                              DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
                reducedErrors[name] = {
                    {"L1", sums[0] / volume},
                    {"L2", std::sqrt(sums[1] / volume)},
                    {"Linf", maximum}};
            }

            if (mpi.rank == 0)
            {
                nlohmann::ordered_json result = {
                    {"configuration", cfg},
                    {"configured_reconstruction_method", cfg.reconstruction.method},
                    {"mpi_ranks", mpi.size},
                    {"target_stencil", targetStencil},
                    {"stencil_min", stencilMinimum},
                    {"stencil_mean", static_cast<real>(stencilSum) /
                                         mesh->NumNodeGlobal()},
                    {"stencil_max", stencilMaximum},
                    {"normal_equation_condition_estimate_max", maxima[1]},
                    {"normal_equation_projector_defect_max", maxima[0]},
                    {"normal_minus_configured_gradient_Linf", maxima[2]},
                    {"normal_minus_configured_point_Linf", maxima[3]},
                    {"errors", reducedErrors}};
                std::ofstream metadata(output / "metrics.json");
                metadata << std::setprecision(17) << result.dump(2) << '\n';
                DNDS_check_throw_info(
                    metadata.good(),
                    "Failed to write normal-equation reconstruction metrics");
                DNDS::log()
                    << "Normal-equation reconstruction: rho point L2="
                    << reducedErrors["normal.u0.value"]["L2"]
                    << ", rho gradient L2="
                    << reducedErrors["normal.u0.gradient"]["L2"]
                    << ", normal-configured point Linf=" << maxima[3]
                    << ", normal-configured gradient Linf=" << maxima[2]
                    << std::endl;
            }
        }
        catch (const std::exception &exception)
        {
            std::cerr << "NCFV normal-equation reconstruction probe: "
                      << exception.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
