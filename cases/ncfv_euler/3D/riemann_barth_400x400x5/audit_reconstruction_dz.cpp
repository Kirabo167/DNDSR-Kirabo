/** Pure reconstruction audit: independent scalar dual means, without fluxes. */
#include "NCFV/NCFVSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using DNDS::DNDS_MPI_INDEX;
using DNDS::DNDS_MPI_REAL;

namespace
{
    using Values = Eigen::Matrix<DNDS::real, 5, 1>;
    using Derivatives = Eigen::Matrix<DNDS::real, 3, 5>;
    constexpr DNDS::real pi = 3.141592653589793238462643383279502884;

    struct Interval
    {
        DNDS::real lower, upper, mean, second;
    };

    Interval DualInterval(int node, int cells, DNDS::real length)
    {
        const DNDS::real lower = std::max(DNDS::real(0), node - .5) * length / cells;
        const DNDS::real upper = std::min(DNDS::real(cells), node + .5) * length / cells;
        return {lower, upper, .5 * (lower + upper),
                (lower * lower + lower * upper + upper * upper) / 3};
    }

    Values Means(int ix, int iy, int iz, int cells, int cellsZ, DNDS::real depth, bool flat)
    {
        const auto x = DualInterval(ix, cells, 1);
        const auto y = DualInterval(iy, cells, 1);
        const auto z = DualInterval(iz, cellsZ, depth);
        Values result;
        if (flat)
        {
            const DNDS::real sx = std::sin(pi * (x.upper - x.lower)) / (pi * (x.upper - x.lower));
            const DNDS::real sy = std::sin(pi * (y.upper - y.lower)) / (pi * (y.upper - y.lower));
            const DNDS::real left = std::clamp((.5 - x.lower) / (x.upper - x.lower), DNDS::real(0), DNDS::real(1));
            const DNDS::real bottom = std::clamp((.5 - y.lower) / (y.upper - y.lower), DNDS::real(0), DNDS::real(1));
            result << 2, 1 + .2 * x.mean - .3 * y.mean,
                1 + .2 * x.mean - .3 * y.mean + .4 * x.second + .5 * x.mean * y.mean + .6 * y.second,
                sx * sy * std::sin(2 * pi * x.mean) * std::cos(2 * pi * y.mean),
                .138 * left * bottom + .5323 * (1 - left) * bottom +
                    .5323 * left * (1 - bottom) + 1.5 * (1 - left) * (1 - bottom);
        }
        else
        {
            result << z.mean, x.mean * z.mean, y.mean * z.mean, z.second,
                1 + .2 * x.mean - .3 * y.mean + .4 * z.mean + .5 * x.mean * y.mean +
                    .6 * x.mean * z.mean + .7 * y.mean * z.mean +
                    .8 * x.second + .9 * y.second + 1.1 * z.second;
        }
        return result;
    }

    Derivatives Exact(DNDS::real x, DNDS::real y, DNDS::real z, bool flat)
    {
        Derivatives result = Derivatives::Zero();
        if (flat)
        {
            result(0, 1) = .2;
            result(1, 1) = -.3;
            result(0, 2) = .2 + .8 * x + .5 * y;
            result(1, 2) = -.3 + .5 * x + 1.2 * y;
            result(0, 3) = 2 * pi * std::cos(2 * pi * x) * std::cos(2 * pi * y);
            result(1, 3) = -2 * pi * std::sin(2 * pi * x) * std::sin(2 * pi * y);
            // RP1 is discontinuous: its x/y derivative is not compared.
        }
        else
        {
            result(2, 0) = 1;
            result(0, 1) = z;
            result(2, 1) = x;
            result(1, 2) = z;
            result(2, 2) = y;
            result(2, 3) = 2 * z;
            result(0, 4) = .2 + .5 * y + .6 * z + 1.6 * x;
            result(1, 4) = -.3 + .5 * x + .7 * z + 1.8 * y;
            result(2, 4) = .4 + .6 * x + .7 * y + 2.2 * z;
        }
        return result;
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    try
    {
        DNDS_check_throw_info(argc == 9 || argc == 10,
                              "usage: config out.json mesh Nx Nz depth complete(0|1) flat|3d [initial.csv|-]");
        const int cells = std::stoi(argv[4]), cellsZ = std::stoi(argv[5]);
        const DNDS::real depth = std::stod(argv[6]);
        const bool complete = std::string(argv[7]) == "1";
        const bool flat = std::string(argv[8]) == "flat";
        DNDS_check_throw_info(cells >= 4 && cellsZ >= 3 && depth > 0 &&
                                  (std::string(argv[7]) == "0" || complete) &&
                                  (flat || std::string(argv[8]) == "3d"),
                              "Invalid geometry or audit group");
        auto cfg = DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
        cfg.mesh.meshFile = argv[3];
        cfg.mesh.periodicLengths = {0, 0, flat ? depth : 0};
        cfg.mesh.periodicBoundaryPairs = flat ? std::vector<std::string>{"ZMIN", "ZMAX"} : std::vector<std::string>{};
        for (auto &zone : cfg.physics.boundaryZones)
            zone.mode = flat && (zone.name == "ZMIN" || zone.name == "ZMAX") ?
                            DNDS::NCFV::BoundaryMode::Periodic : DNDS::NCFV::BoundaryMode::SupersonicOutlet;
        cfg.initialField.nodeFile.clear();
        cfg.initialField.expressions.clear();
        cfg.algorithm.mode = DNDS::NCFV::IntegrationMode::EfficientDifferential;
        cfg.reconstruction.method = DNDS::NCFV::ReconstructionMethod::SVDLeastSquares;
        cfg.reconstruction.completeStencilRings = complete;
        cfg.time.iterations = 0;
        cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
        cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
        cfg.io.restartInterval = 0;
        DNDS::NCFV::Solver<3> solver(mpi, cfg);
        solver.Initialize();
        const auto &mesh = solver.Mesh();
        const auto &halo = solver.NodeCommunication();
        DNDS::NCFV::NodeStatePair means;
        means.InitPair("pureReconstruction.means", mpi);
        means.father->Resize(mesh->NumNode(), 5, 1);
        means.son->Resize(halo.NumNodeGhost(), 5, 1);
        means.BorrowSetup(halo.Layout());
        means.trans.initPersistentPull();
        DNDS::NCFV::NodeMatrixPair gradients, coefficients;
        for (auto *field : {&gradients, &coefficients})
        {
            const int rows = field == &gradients ? 3 : 9;
            field->InitPair("pureReconstruction.matrix", mpi);
            field->father->Resize(mesh->NumNode(), rows, 5);
            field->son->Resize(halo.NumNodeGhost(), rows, 5);
            field->BorrowSetup(halo.Layout());
            field->trans.initPersistentPull();
            for (DNDS::index i = 0; i < field->Size(); i++)
                (*field)[i].setZero();
        }
        std::unordered_map<DNDS::index, DNDS::index> originalToLocal;
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            const auto coordinate = halo.Coordinate(i);
            const int ix = std::llround(coordinate(0) * cells), iy = std::llround(coordinate(1) * cells);
            const int iz = std::llround(coordinate(2) * cellsZ / depth);
            DNDS_check_throw_info(ix >= 0 && ix <= cells && iy >= 0 && iy <= cells && iz >= 0 && iz <= cellsZ,
                                  "Node lies outside the supplied Cartesian dimensions");
            DNDS_check_throw_info(std::abs(coordinate(0) - DNDS::real(ix) / cells) < 1e-11 &&
                                      std::abs(coordinate(1) - DNDS::real(iy) / cells) < 1e-11 &&
                                      std::abs(coordinate(2) - depth * iz / cellsZ) < 1e-11,
                                  "Mesh spacing differs from the supplied dimensions");
            means[i] = Means(ix, iy, iz, cells, cellsZ, depth, flat);
            originalToLocal.emplace(mesh->node2nodeOrig(i, 0), i);
        }
        if (argc == 10 && std::string(argv[9]) != "-")
        {
            DNDS_check_throw_info(flat, "Optional density CSV is only used by the flat RP1 field");
            std::ifstream input(argv[9]);
            DNDS_check_throw_info(input.good(), "Cannot open optional initial density CSV");
            std::vector<bool> loaded(static_cast<std::size_t>(mesh->NumNode()), false);
            for (std::string line; std::getline(input, line);)
            {
                std::replace(line.begin(), line.end(), ',', ' ');
                std::istringstream parser(line);
                DNDS::index original = -1;
                DNDS::real density = 0;
                if (!(parser >> original >> density))
                    continue;
                const auto found = originalToLocal.find(original);
                if (found != originalToLocal.end())
                {
                    means[found->second](4) = density;
                    loaded[static_cast<std::size_t>(found->second)] = true;
                }
            }
            DNDS_check_throw_info(std::all_of(loaded.begin(), loaded.end(), [](bool value) { return value; }),
                                  "Optional density CSV is incomplete");
        }
        means.trans.startPersistentPull();
        means.trans.waitPersistentPull();
        // This is the only numerical operation under test; no SpatialOperator.
        solver.ReconstructionData().ComputeCoefficients(means, gradients, coefficients);
        std::array<DNDS::real, 15> maximum{}, error{};
        std::array<DNDS::real, 5> boundaryZ{}, interiorZ{};
        std::array<DNDS::index, 5> nonzeroZ{};
        std::array<DNDS::real, 60> examples{};
        std::array<DNDS::real, 5> exampleMaximum{-1, -1, -1, -1, -1};
        const int columns = (cells + 1) * (cells + 1);
        std::vector<DNDS::real> meanMinimum(columns * 5, 1e300), meanMaximum(columns * 5, -1e300);
        std::vector<DNDS::real> gradientMinimum(columns * 15, 1e300), gradientMaximum(columns * 15, -1e300);
        DNDS::index ownedNodes = mesh->NumNode(), globalNodes = 0;
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            const auto coordinate = halo.Coordinate(i);
            const int ix = std::llround(coordinate(0) * cells), iy = std::llround(coordinate(1) * cells);
            const int column = iy * (cells + 1) + ix;
            const auto exact = Exact(coordinate(0), coordinate(1), coordinate(2), flat);
            const bool boundary = ix == 0 || iy == 0 || ix == cells || iy == cells;
            const auto &op = solver.ReconstructionData().Operator(i);
            for (int field = 0; field < 5; field++)
            {
                meanMinimum[5 * column + field] = std::min(meanMinimum[5 * column + field], means[i](field));
                meanMaximum[5 * column + field] = std::max(meanMaximum[5 * column + field], means[i](field));
                const DNDS::real dz = std::abs(gradients[i](2, field));
                nonzeroZ[field] += dz > 1e-12;
                auto &regional = boundary ? boundaryZ[field] : interiorZ[field];
                regional = std::max(regional, dz);
                if (dz > exampleMaximum[field])
                {
                    exampleMaximum[field] = dz;
                    const std::array<DNDS::real, 12> record{
                        dz, gradients[i](2, field), coordinate(0), coordinate(1), coordinate(2),
                        static_cast<DNDS::real>(mesh->node2nodeOrig(i, 0)),
                        gradients[i](0, field), gradients[i](1, field), means[i](field),
                        op.conditionNumber, static_cast<DNDS::real>(op.stencil.size()), static_cast<DNDS::real>(op.rings)};
                    std::copy(record.begin(), record.end(), examples.begin() + 12 * field);
                }
                for (int direction = 0; direction < 3; direction++)
                {
                    const int packed = direction * 5 + field;
                    const DNDS::real derivative = gradients[i](direction, field);
                    maximum[packed] = std::max(maximum[packed], std::abs(derivative));
                    if (!(flat && field == 4 && direction < 2))
                        error[packed] = std::max(error[packed], std::abs(derivative - exact(direction, field)));
                    const int entry = 15 * column + packed;
                    gradientMinimum[entry] = std::min(gradientMinimum[entry], derivative);
                    gradientMaximum[entry] = std::max(gradientMaximum[entry], derivative);
                }
            }
        }
        std::array<DNDS::real, 15> globalMaximum{}, globalError{};
        std::array<DNDS::real, 5> globalBoundaryZ{}, globalInteriorZ{};
        std::array<DNDS::index, 5> globalNonzeroZ{};
        std::vector<DNDS::real> allMeanMinimum(meanMinimum.size()), allMeanMaximum(meanMaximum.size());
        std::vector<DNDS::real> allGradientMinimum(gradientMinimum.size()), allGradientMaximum(gradientMaximum.size());
        MPI_Reduce(&ownedNodes, &globalNodes, 1, DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(maximum.data(), globalMaximum.data(), 15, DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        MPI_Reduce(error.data(), globalError.data(), 15, DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        MPI_Reduce(boundaryZ.data(), globalBoundaryZ.data(), 5, DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        MPI_Reduce(interiorZ.data(), globalInteriorZ.data(), 5, DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        MPI_Reduce(nonzeroZ.data(), globalNonzeroZ.data(), 5, DNDS_MPI_INDEX, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(meanMinimum.data(), allMeanMinimum.data(), meanMinimum.size(), DNDS_MPI_REAL, MPI_MIN, 0, mpi.comm);
        MPI_Reduce(meanMaximum.data(), allMeanMaximum.data(), meanMaximum.size(), DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        MPI_Reduce(gradientMinimum.data(), allGradientMinimum.data(), gradientMinimum.size(), DNDS_MPI_REAL, MPI_MIN, 0, mpi.comm);
        MPI_Reduce(gradientMaximum.data(), allGradientMaximum.data(), gradientMaximum.size(), DNDS_MPI_REAL, MPI_MAX, 0, mpi.comm);
        std::vector<DNDS::real> gathered(static_cast<std::size_t>(mpi.size) * examples.size());
        MPI_Gather(examples.data(), 60, DNDS_MPI_REAL, gathered.data(), 60, DNDS_MPI_REAL, 0, mpi.comm);
        if (mpi.rank == 0)
        {
            std::array<DNDS::real, 5> meanSpan{};
            std::array<DNDS::real, 15> gradientSpan{};
            for (int column = 0; column < columns; column++)
            {
                for (int field = 0; field < 5; field++)
                    meanSpan[field] = std::max(meanSpan[field], allMeanMaximum[5 * column + field] - allMeanMinimum[5 * column + field]);
                for (int packed = 0; packed < 15; packed++)
                    gradientSpan[packed] = std::max(gradientSpan[packed], allGradientMaximum[15 * column + packed] - allGradientMinimum[15 * column + packed]);
            }
            nlohmann::ordered_json worst = nlohmann::ordered_json::array();
            for (int field = 0; field < 5; field++)
            {
                int rank = 0;
                for (int candidate = 1; candidate < mpi.size; candidate++)
                    if (gathered[60 * candidate + 12 * field] > gathered[60 * rank + 12 * field])
                        rank = candidate;
                const auto first = gathered.begin() + 60 * rank + 12 * field;
                worst.push_back({{"rank", rank}, {"values", std::vector<DNDS::real>(first, first + 12)}});
            }
            nlohmann::ordered_json result{
                {"mpi_ranks", mpi.size}, {"nodes", globalNodes}, {"cells_xy", cells}, {"cells_z", cellsZ},
                {"depth", depth}, {"mesh", cfg.mesh.meshFile}, {"group", flat ? "flat" : "3d"},
                {"completeStencilRings", complete}, {"periodic_z", flat},
                {"field_names", flat ? std::vector<std::string>{"constant", "linear_xy", "quadratic_xy", "smooth_sin_cos", "RP1_density"} :
                                      std::vector<std::string>{"z", "x_z", "y_z", "z_squared", "mixed_quadratic_3d"}},
                {"packed_order", "x five fields, y five fields, z five fields"},
                {"raw_gradient_abs_max", globalMaximum}, {"gradient_error_abs_max", globalError},
                {"mean_maximum_column_span", meanSpan}, {"gradient_maximum_column_span", gradientSpan},
                {"dz_max_xy_boundary", globalBoundaryZ}, {"dz_max_xy_interior", globalInteriorZ},
                {"nodes_abs_dz_above_1e_minus_12", globalNonzeroZ},
                {"worst_order", {"abs_dz", "dz", "x", "y", "z", "original_node", "dx", "dy", "mean", "condition", "stencil_size", "rings"}},
                {"worst_per_field", worst},
                {"initial_density_source", argc == 10 ? argv[9] : "analytic RP1 quadrant dual mean"},
                {"scalar_fields_not_Euler_states", true},
                {"scope", "Direct ComputeCoefficients only; no Riemann solver, limiter, pressure or time integration"},
                {"mean_definition", "Exact clipped Cartesian median-dual averages; sine/cosine use analytic sinc factors"},
                {"RP1_xy_error_note", "Packed x/y error entries for discontinuous RP1 density are intentionally omitted (stored zero)"},
                {"true3d_periodicity_note", "z/xz/yz/z^2 are not periodic: the 3d test uses nonperiodic z to avoid a false seam error"}};
            std::ofstream output(argv[2]);
            DNDS_check_throw_info(output.good(), "Cannot open pure reconstruction JSON");
            output << result.dump(2) << std::endl;
            std::cout << result.dump(2) << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "pure reconstruction audit rank " << mpi.rank << ": " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    MPI_Finalize();
}
