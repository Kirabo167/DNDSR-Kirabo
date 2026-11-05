#pragma once

#include <cstddef>
#include <vector>

namespace DNDS::GPU
{
    // Plain double-precision packets shared by C++ callers and CUDA kernels.
    struct FluxInput
    {
        double left[5]{}, right[5]{}, normal[3]{};
        double gamma = 1.4, rho0 = 1, beta2 = 1, alpha = 0, entropyRatio = 0.05;
        int dimension = 3, model = 0, solver = 0;
        bool referenceHarten = false;
    };

    struct FluxOutput
    {
        double value[5]{};
    };

    bool Enabled();
    void Configure(bool enabled, int requestedDevice, int localRank);
    void Evaluate(const std::vector<FluxInput> &input, std::vector<FluxOutput> &output);
    void Report(int rank);
}
