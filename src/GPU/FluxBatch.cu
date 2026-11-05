#include "FluxBatch.hpp"
#include "FluxMath.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace DNDS::GPU
{
    namespace
    {
        void Check(cudaError_t error, const char *operation)
        {
            if (error != cudaSuccess)
                throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
        }

        struct Context
        {
            bool enabled = true;
            int device = -1;
            FluxInput *input = nullptr;
            FluxOutput *output = nullptr;
            std::size_t capacity = 0, batches = 0, points = 0;
            ~Context()
            {
                // A host-only or --check-config process must never initialize
                // the CUDA runtime for the first time during static teardown.
                if (device >= 0)
                {
                    cudaSetDevice(device);
                    if (input)
                        cudaFree(input);
                    if (output)
                        cudaFree(output);
                }
            }
        } context;

        using namespace Detail;

        __global__ void FluxKernel(const FluxInput *input, FluxOutput *output, std::size_t count)
        {
            const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < count)
            {
                output[i] = FluxOutput{};
                if (input[i].model == 1)
                    ACMFlux(input[i], output[i]);
                else
                    EulerFlux(input[i], output[i]);
            }
        }
    }

    bool Enabled() { return context.enabled; }

    void Configure(bool enabled, int requestedDevice, int localRank)
    {
        context.enabled = enabled;
        if (!enabled)
            return;
        int count = 0;
        Check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (!count || requestedDevice >= count)
            throw std::runtime_error("Requested CUDA device is unavailable");
        const int device = requestedDevice < 0 ? localRank % count : requestedDevice;
        if (context.device >= 0 && device != context.device)
            throw std::runtime_error("Cannot change CUDA device after initialization");
        Check(cudaSetDevice(device), "cudaSetDevice");
        context.device = device;
        cudaDeviceProp properties;
        Check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        std::cout << "GPU backend: CUDA device " << device << " (" << properties.name << "), local rank " << localRank << std::endl;
    }

    void Evaluate(const std::vector<FluxInput> &input, std::vector<FluxOutput> &output)
    {
        output.resize(input.size());
        if (input.empty())
            return;
        if (!context.enabled)
            throw std::runtime_error("CUDA flux batch requested with host backend");
        if (context.device < 0)
            Configure(true, -1, 0);
        if (input.size() > context.capacity)
        {
            Check(cudaFree(context.input), "cudaFree input");
            Check(cudaFree(context.output), "cudaFree output");
            context.input = nullptr; context.output = nullptr; context.capacity = 0;
            Check(cudaMalloc(reinterpret_cast<void **>(&context.input), input.size() * sizeof(FluxInput)), "cudaMalloc input");
            Check(cudaMalloc(reinterpret_cast<void **>(&context.output), input.size() * sizeof(FluxOutput)), "cudaMalloc output");
            context.capacity = input.size();
        }
        Check(cudaMemcpy(context.input, input.data(), input.size() * sizeof(FluxInput), cudaMemcpyHostToDevice), "copy flux inputs");
        FluxKernel<<<(input.size() + 255) / 256, 256>>>(context.input, context.output, input.size());
        Check(cudaGetLastError(), "launch flux kernel");
        Check(cudaMemcpy(output.data(), context.output, input.size() * sizeof(FluxOutput), cudaMemcpyDeviceToHost), "copy flux outputs");
        ++context.batches; context.points += input.size();
        for (const auto &value : output)
            for (double component : value.value)
                if (!std::isfinite(component))
                    throw std::runtime_error("CUDA numerical flux is non-finite");
    }

    void Report(int rank)
    {
        std::cout << "GPU statistics rank " << rank << ": backend=" << (Enabled() ? "cuda" : "host")
                  << ", batches=" << context.batches << ", flux points=" << context.points << std::endl;
    }
}
