"""Completed Python setup uses the owned construction sequence and read leases."""
from pathlib import Path

import pytest
from mpi4py import MPI

from DNDSR import DNDS, NativeGeom
from DNDSR.NativeGeom.utils import read_mesh, prepare_mesh, prepare_native_mesh


def world():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    return mpi


def mesh_file():
    return str(Path(__file__).resolve().parents[2] / "data/mesh/UniformSquare_10.cgns")


def ids(mesh, kind):
    local = [mesh.original_entity_id(kind, row) for row in range(mesh.entities(kind).owned)]
    return {value for rows in MPI.COMM_WORLD.allgather(local) for value in rows}


@pytest.mark.mpi
def test_python_setup_partition_precedes_refinement_and_retains_original_labels(monkeypatch):
    events, original = [], {}
    partition = NativeGeom.partition_native_mesh
    elevate = NativeGeom.elevate_native_mesh

    def record_partition(*args, **kwargs):
        events.append("partition")
        return partition(*args, **kwargs)

    def record_elevation(mesh, *args, **kwargs):
        events.append("elevation")
        for kind in (NativeGeom.EntityKind.Node, NativeGeom.EntityKind.Cell, NativeGeom.EntityKind.Boundary):
            original[kind] = ids(mesh, kind)
        return elevate(mesh, *args, **kwargs)

    monkeypatch.setattr(NativeGeom, "partition_native_mesh", record_partition)
    monkeypatch.setattr(NativeGeom, "elevate_native_mesh", record_elevation)
    result = read_mesh(mesh_file(), mpi=world(), dim=2, elevation="O2", complete_node_star=True)
    assert events == ["partition", "elevation"]
    assert result.mesh.IsO2()
    assert ids(result.mesh, NativeGeom.EntityKind.Cell) == original[NativeGeom.EntityKind.Cell]
    assert ids(result.mesh, NativeGeom.EntityKind.Boundary) == original[NativeGeom.EntityKind.Boundary]
    assert original[NativeGeom.EntityKind.Node] < ids(result.mesh, NativeGeom.EntityKind.Node)


@pytest.mark.mpi
def test_owned_preparation_detaches_or_rejects_in_place_active_reads():
    mpi = world()
    source = NativeGeom.read_native_cgns(mpi, mesh_file(), 2, complete_node_star=True)
    original_cells = ids(source, NativeGeom.EntityKind.Cell)
    assert not NativeGeom.native_mesh_requirements(source)["faces"]
    active = source.acquire_read(inverse=True)
    detached = prepare_native_mesh(source, reorder_parts=2)
    assert not NativeGeom.native_mesh_requirements(source)["faces"]
    assert ids(detached, NativeGeom.EntityKind.Cell) == original_cells
    assert sum(MPI.COMM_WORLD.allgather(detached.NumFace())) == 220
    assert ids(detached, NativeGeom.EntityKind.Cell) == ids(source, NativeGeom.EntityKind.Cell)
    with pytest.raises(RuntimeError):
        prepare_mesh(source, None, reorder_parts=2, build_serial_out=False)
    assert not NativeGeom.native_mesh_requirements(source)["faces"]
    active.close()
    prepare_mesh(source, None, reorder_parts=2, build_serial_out=False)
    assert sum(MPI.COMM_WORLD.allgather(source.NumFace())) == 220
    assert ids(source, NativeGeom.EntityKind.Cell) == original_cells


@pytest.mark.mpi
def test_combined_setup_rejects_rank_mismatches_before_publication():
    mpi = world()
    source = NativeGeom.read_native_cgns(mpi, mesh_file(), 2)
    original_cells = ids(source, NativeGeom.EntityKind.Cell)
    with pytest.raises(RuntimeError):
        NativeGeom.prepare_native_mesh_setup(source, reorder_parts=-1)
    if mpi.size > 1:
        with pytest.raises(RuntimeError):
            NativeGeom.prepare_native_mesh_setup(source, reorder_parts=1, in_place=mpi.rank == 0)
    assert not NativeGeom.native_mesh_requirements(source)["faces"]
    assert ids(source, NativeGeom.EntityKind.Cell) == original_cells
