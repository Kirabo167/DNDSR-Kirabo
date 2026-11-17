#pragma once

#include "NativeGeom/IO/NativeMeshCGNS.hpp"
#include "NativeGeom/Unstructured/NativeMeshPreparation.hpp"
#include "NativeGeom/Connectivity/NativeMeshTopology.hpp"
#include "NativeGeom/Transition/NativeMeshAssembler.hpp"
#include <filesystem>

namespace NativeConnectivityLegacyConditions
{
    enum class Placement
    {
        LegacyReaderMetis,
        NativeEven
    };

    /// Reproduce the frozen legacy real-grid regression conditions, using
    /// native storage: reader rank zero, AutoAppendName2ID, serial KWAY,
    /// ufactor=30, seed=42, ncuts=1, primary redistribution and one cell ring.
    /// NativeEven is an ADDITIONAL fixture, never its replacement.
    inline std::shared_ptr<const DNDS::NativeGeom::UnstructuredMeshState> UniformSquare10(
        const DNDS::MPIInfo &mpi, Placement placement, bool faces = false)
    {
        namespace G = DNDS::NativeGeom;
        const auto path = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
                          "data/mesh/UniformSquare_10.cgns";
        G::AutoAppendName2ID names;
        G::NativePeriodicImportOptions periodic;
        periodic.tolerance = 1e-8;
        auto owned = G::ReadNativeCGNSOwned(mpi, path.string(), 2, [&](const std::string &name)
                                            { return names(name); }, 0, {1, true}, periodic, placement == Placement::LegacyReaderMetis ? G::NativeSerialDistribution::ReaderOwned : G::NativeSerialDistribution::Even);
        CHECK(owned.spaces.at(G::EntityKind::Cell).NumGlobal() == 100);
        CHECK(owned.spaces.at(G::EntityKind::Node).NumGlobal() == 121);
        CHECK(owned.spaces.at(G::EntityKind::Bnd).NumGlobal() == 40);
        if (placement == Placement::LegacyReaderMetis)
        {
            CHECK(owned.spaces.at(G::EntityKind::Cell).NumOwned() == (mpi.rank == 0 ? 100 : 0));
            G::NativePartitionOptions options;
            options.metisType = "KWAY";
            options.metisUfactor = 30;
            options.metisSeed = 42;
            options.metisNcuts = 1;
            owned = G::NativeMeshPreparation::PartitionOwned(owned, mpi, options).owned;
        }
        if (faces)
            owned = G::NativeMeshTopology::Build(std::move(owned), mpi);
        const auto recipe = G::DefaultNativeMeshGhostSpec(owned, 1, true);
        return G::NativeMeshAssembler::Build(mpi, std::move(owned), recipe);
    }
}
