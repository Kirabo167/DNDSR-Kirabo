#include "GPU/ResidentCommon.cuh"
#include <iostream>

using namespace DNDS::GPU::ResidentDetail;
__global__ void Initialize(double *values, int stride, int rank, int iteration)
{
    const int i = int(Thread());
    if (i < 3 * stride)
        values[i] = iteration * 10000. + (rank * 3 + i / stride) * 100. + i % stride;
}
__global__ void Verify(const double *values, int stride, int rank, int iteration, int *invalid)
{
    const int i = int(Thread());
    if (i >= 3 * stride)
        return;
    const double expected = iteration * 10000. + ((1 - rank) * 3 + i / stride) * 100. + i % stride;
    if (values[3 * stride + i] != expected)
        atomicExch(invalid, i + 1);
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try
    {
        if (size != 2)
            throw std::runtime_error("Halo verification requires two ranks");
        Check(cudaSetDevice(rank), "select halo verification device");
        int peerSupported = 0;
        Check(cudaDeviceCanAccessPeer(&peerSupported, rank, 1 - rank), "query GPU peer access");
        std::cout << "Rank " << rank << " peer access capability=" << peerSupported << std::endl;
        std::vector<I> globals;
        for (int i = 0; i < 3; ++i)
            globals.push_back(rank * 3 + i);
        for (int i = 0; i < 3; ++i)
            globals.push_back((1 - rank) * 3 + i);
        {
            Halo halo(globals, 3, MPI_COMM_WORLD), secondHalo(globals, 3, MPI_COMM_WORLD);
            Buffer<double> field;
            field.Resize(6 * 76);
            Buffer<int> invalid;
            invalid.Resize(1);
            for (int iteration = 0; iteration < 1000; ++iteration)
            {
                for (int phase = 0; phase < 2; ++phase)
                {
                    const int seed = iteration * 2 + phase;
                    const int stride = iteration % 3 == 0 ? 76 : iteration % 3 == 1 ? 5
                                                                                    : 1;
                    Check(cudaMemset(field.data, 0, field.count * sizeof(double)), "clear verification field");
                    Initialize<<<2, 128>>>(field.data, stride, rank, seed);
                    (phase ? secondHalo : halo).Pull(field.data, stride);
                    Check(cudaMemset(invalid.data, 0, sizeof(int)), "clear halo verification status");
                    Verify<<<2, 128>>>(field.data, stride, rank, seed, invalid.data);
                    int local = 0, global = 0;
                    Check(cudaMemcpy(&local, invalid.data, sizeof(int), cudaMemcpyDeviceToHost), "read halo verification status");
                    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                    if (global)
                    {
                        std::vector<double> received(3 * stride);
                        Check(cudaMemcpy(received.data(), field.data + 3 * stride, received.size() * sizeof(double), cudaMemcpyDeviceToHost), "inspect failed halo");
                        std::cerr << "Rank " << rank << " received: ";
                        for (int i = 0; i < 8 && i < 3 * stride; ++i)
                            std::cerr << received[i] << ' ';
                        std::cerr << "; last=" << received.back() << std::endl;
                        throw std::runtime_error("GPU halo mismatch at iteration " + std::to_string(iteration) + " component " + std::to_string(global));
                    }
                }
            }
            if (!rank)
                std::cout << "GPU halo: 2000 alternating-buffer/stride exchanges passed; host halo staging bytes=" << halo.bytes + secondHalo.bytes << std::endl;
        }
        MPI_Finalize();
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Rank " << rank << ": " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
