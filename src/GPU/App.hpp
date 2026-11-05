#pragma once

#include "FluxBatch.hpp"
#include <mpi.h>
#include <stdexcept>
#include <string>

namespace DNDS::GPU
{
    template <class Parser>
    void AddArguments(Parser &parser)
    {
        parser.add_argument("--backend").default_value(std::string("cuda"));
        parser.add_argument("--gpu-device").default_value(std::string("auto"));
    }

    template <class Parser>
    void Initialize(const Parser &parser)
    {
        const auto backend = parser.template get<std::string>("--backend");
        if (backend != "cuda" && backend != "host")
            throw std::runtime_error("--backend must be cuda or host");
        const auto device = parser.template get<std::string>("--gpu-device");
        int selected = -1;
        if (device != "auto")
        {
            std::size_t end = 0;
            selected = std::stoi(device, &end);
            if (end != device.size() || selected < 0)
                throw std::runtime_error("--gpu-device must be auto or a nonnegative integer");
        }
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
        int localRank = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        Configure(backend == "cuda", selected, localRank);
    }
}
