#include "GPU/FluxBatch.hpp"
#include "ACM/ACMFlux.hpp"
#include "Euler/Gas.hpp"
#include <mpi.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

namespace
{
    template <int dimension>
    DNDS::GPU::FluxOutput EulerReference(const DNDS::GPU::FluxInput &input)
    {
        Eigen::Vector<double, dimension + 2> left, right, flux;
        Eigen::Vector<double, dimension> normal, grid = Eigen::Vector<double, dimension>::Zero();
        for (int i = 0; i < dimension + 2; ++i)
        {
            left(i) = input.left[i]; right(i) = input.right[i];
        }
        for (int i = 0; i < dimension; ++i)
            normal(i) = input.normal[i];
        double minus, contact, plus;
        DNDS::Euler::Gas::InviscidFlux_IdealGas_Dispatcher<dimension>(
            input.solver ? DNDS::Euler::Gas::Roe_M2 : DNDS::Euler::Gas::Roe,
            left, right, left, right, grid, normal, input.gamma, input.gamma,
            flux, 0., 1., 1., []() {}, minus, contact, plus);
        DNDS::GPU::FluxOutput result;
        for (int i = 0; i < dimension + 2; ++i)
            result.value[i] = flux(i);
        return result;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int result = 0;
    try
    {
        DNDS::GPU::Configure(true, argc > 1 ? std::stoi(argv[1]) : 0, 0);
        std::vector<DNDS::GPU::FluxInput> inputs;
        std::vector<DNDS::GPU::FluxOutput> expected;
        std::mt19937_64 generator(89124);
        std::uniform_real_distribution<double> random(-1, 1);
        for (int model = 0; model < 2; ++model)
            for (int dimension : {2, 3})
                for (int solver = 0; solver < 2; ++solver)
                    for (int sample = 0; sample < 256; ++sample)
                    {
                        DNDS::GPU::FluxInput input;
                        input.model = model; input.dimension = dimension; input.solver = solver;
                        Eigen::Vector3d normal(random(generator), random(generator), dimension == 3 ? random(generator) : 0);
                        normal.normalize();
                        for (int i = 0; i < 3; ++i)
                            input.normal[i] = normal(i);
                        DNDS::GPU::FluxOutput reference;
                        if (model == 0)
                        {
                            input.gamma = 1.1 + .4 * (random(generator) + 1);
                            for (double *state : {input.left, input.right})
                            {
                                state[0] = 1.1 + random(generator);
                                double kinetic = 0;
                                for (int i = 0; i < dimension; ++i)
                                {
                                    state[i + 1] = 2 * random(generator);
                                    kinetic += .5 * state[i + 1] * state[i + 1] / state[0];
                                }
                                state[dimension + 1] = (1.1 + random(generator)) / (input.gamma - 1) + kinetic;
                            }
                            reference = dimension == 2 ? EulerReference<2>(input) : EulerReference<3>(input);
                        }
                        else
                        {
                            DNDS::ACM::Settings settings;
                            settings.rho0 = input.rho0 = 1.1 + random(generator);
                            settings.beta2 = input.beta2 = 1.1 + random(generator);
                            settings.alpha = input.alpha = 2 * random(generator);
                            settings.entropyFixRatio = input.entropyRatio = sample % 3 ? .05 : 0;
                            DNDS::ACM::State left, right;
                            for (int i = 0; i < 4; ++i)
                            {
                                left(i) = 2 * random(generator); right(i) = 2 * random(generator);
                            }
                            if (sample < 16)
                            {
                                settings.alpha = input.alpha = 1;
                                const double sign = sample % 2 ? 1 : -1;
                                const double speed = sign * std::sqrt(input.beta2 / input.rho0) * (1 + (sample / 2) * 1e-12);
                                const double old = .5 * (left.head<3>() + right.head<3>()).dot(normal);
                                left.head<3>() += (speed - old) * normal;
                                right.head<3>() += (speed - old) * normal;
                            }
                            for (int i = 0; i < 4; ++i)
                            {
                                input.left[i] = left(i); input.right[i] = right(i);
                            }
                            const auto flux = DNDS::ACM::InviscidFlux(
                                solver ? DNDS::ACM::RiemannSolverType::Rusanov : DNDS::ACM::RiemannSolverType::Roe,
                                left, right, normal, settings).flux;
                            for (int i = 0; i < 4; ++i)
                                reference.value[i] = flux(i);
                        }
                        inputs.push_back(input); expected.push_back(reference);
                    }
        std::vector<DNDS::GPU::FluxOutput> actual;
        DNDS::GPU::Evaluate({}, actual);
        if (!actual.empty())
            throw std::runtime_error("Empty batch returned values");
        DNDS::GPU::Evaluate(inputs, actual);
        double maximum = 0;
        for (std::size_t i = 0; i < actual.size(); ++i)
            for (int component = 0; component < 5; ++component)
            {
                const double error = std::abs(actual[i].value[component] - expected[i].value[component]) /
                    std::max(1., std::abs(expected[i].value[component]));
                maximum = std::max(maximum, error);
                if (!std::isfinite(error) || error > 2e-9)
                    throw std::runtime_error("CPU/CUDA flux mismatch at packet " + std::to_string(i) +
                        ", component " + std::to_string(component) + ", error=" + std::to_string(error));
            }
        // Reuse the persistent buffers and test a smaller dispatch.
        inputs.resize(7);
        DNDS::GPU::Evaluate(inputs, actual);
        if (actual.size() != 7)
            throw std::runtime_error("Resized batch returned wrong size");
        std::cout << "CPU/CUDA flux comparison: " << expected.size() << " packets passed; maximum scaled error=" << maximum << std::endl;
        DNDS::GPU::Report(0);
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        result = 1;
    }
    MPI_Finalize();
    return result;
}
