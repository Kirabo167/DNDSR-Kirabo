#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/ArrayDOF.hpp"
#include "DNDS/Compat/LegacyDofBridge.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Runtime/Communication/MPIProgress.hpp"

namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;

TEST_CASE("Uncompressed CSR row owners survive replacement and copies stay independent")
{
    std::shared_ptr<const void> retained;
    {
        DNDS::Array<double, DNDS::NonUniformSize> rows;
        rows.Resize(1, 2);
        rows(0, 0) = 3.;
        rows(0, 1) = 7.;
        retained = rows.rowLease(0);
        rows.ResizeRow(0, 4);
        rows(0, 0) = 11.;
        const auto copy = rows;
        rows(0, 0) = 19.;
        CHECK(copy(0, 0) == 11.);
        CHECK(static_cast<const double *>(retained.get())[0] == 3.);
        CHECK(static_cast<const double *>(retained.get())[1] == 7.);
    }
    CHECK(static_cast<const double *>(retained.get())[1] == 7.);
}

TEST_CASE("Legacy solver fields share native arithmetic and native MPI without value copies")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    DNDS::ArrayDof<DNDS::DynamicSize, 1> legacy;
    legacy.InitPair("legacy/state", mpi);
    legacy.father->Resize(3, 5, 1);
    legacy.son->Resize(1, 5, 1);
    legacy.TransAttach();
    legacy.setConstant(mpi.rank + 1.);
    auto context = std::make_shared<R::CommunicationContext>(mpi.comm);
    A::DofSpace space(context, 3, 1, "legacy/node-order");
    auto native = R::ShareLegacyDofField(space, legacy);
    CHECK(native.Owned().Identity() == legacy.father->HostDataIdentity());
    CHECK(native.Ghost().Identity() == legacy.son->HostDataIdentity());
    R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
    const auto compute = engine.CreateCPUStream();
    A::Scale(engine, compute, native, 2.);
    engine.Synchronize();
    CHECK(legacy[1](4, 0) == 2. * (mpi.rank + 1));
    CHECK(A::DotGlobal(native, native) == doctest::Approx(60. * mpi.size * (mpi.size + 1.) * (2. * mpi.size + 1.) / 6.));
    R::RowTransferPlan route(context, 3, 1, {{(mpi.rank + 1) % mpi.size, 2, 0}});
    auto binding = R::PayloadPlan(route, native.Owned().Layout(), native.Ghost().Layout(), R::ElementSchema::For<double>())
                       .Bind(native.Owned(), native.Ghost());
    auto transfer = binding.Begin();
    transfer.Wait();
    CHECK(legacy.son->operator[](0)(3, 0) == 2. * ((mpi.rank + 1) % mpi.size + 1));
    const auto snapshot = native.Owned();
    legacy.father->Resize(4, 5, 1);
    CHECK(native.Owned().Identity() != legacy.father->HostDataIdentity());
    CHECK(snapshot.Read().View()(2, 4) == 2. * (mpi.rank + 1));
    A::DofSpace wrong(2, 1);
    CHECK_THROWS(R::ShareLegacyDofField(wrong, legacy));
}

int main(int argc, char **argv)
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
