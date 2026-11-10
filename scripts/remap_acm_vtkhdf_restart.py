#!/usr/bin/env python3
"""Reorder a VTKHDF restart onto an identical mesh with different MPI partitioning.

The template must be written by the target MPI layout with the same mesh and
thread/reordering settings. Every vertex and cell is verified; no interpolation
or change to any cell value is permitted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


def remap(source, template, output):
    if output.exists() or output.with_suffix('.partial.vtkhdf').exists():
        raise FileExistsError(output)
    with h5py.File(source) as old, h5py.File(template) as target:
        a, b = old['VTKHDF'], target['VTKHDF']
        pa, pb = a['Points'][()], b['Points'][()]
        if pa.shape != pb.shape or not np.isfinite(pa).all() or not np.isfinite(pb).all():
            raise ValueError('Mesh point shapes or coordinates are invalid')
        distances, point_map = cKDTree(pb).query(pa, workers=8)
        if (np.any(distances != 0) or np.unique(point_map).size != len(pb)
                or not np.array_equal(pa, pb[point_map])):
            raise ValueError('Meshes do not have exactly the same unique vertex coordinates')
        ca, cb = a['Connectivity'][()], b['Connectivity'][()]
        oa, ob = a['Offsets'][()], b['Offsets'][()]
        ta, tb = a['Types'][()], b['Types'][()]
        if len(ta) != len(tb) or len(oa) != len(ta) + 1 or len(ob) != len(tb) + 1:
            raise ValueError('Mesh cell counts or offset shapes differ')
        na, nb = np.diff(oa), np.diff(ob)
        if np.any(na <= 0) or np.any(nb <= 0):
            raise ValueError('Invalid cell connectivity')
        centers_a = np.add.reduceat(pa[ca], oa[:-1], axis=0) / na[:, None]
        centers_b = np.add.reduceat(pb[cb], ob[:-1], axis=0) / nb[:, None]
        distances, cell_map = cKDTree(centers_a).query(centers_b, workers=8)
        # Centroids identify candidates only; full vertex sets below establish
        # identity, so centroid rounding cannot silently alter cell assignment.
        if np.unique(cell_map).size != len(ta) or not np.array_equal(ta[cell_map], tb):
            raise ValueError('Cannot establish a bijective cell/type mapping')
        if not np.array_equal(na[cell_map], nb):
            raise ValueError('Mapped cells have different vertex counts')
        for count in np.unique(nb):
            selected = np.flatnonzero(nb == count)
            old_vertices = ca[oa[cell_map[selected], None] + np.arange(count)]
            new_vertices = cb[ob[selected, None] + np.arange(count)]
            if not np.array_equal(np.sort(point_map[old_vertices], axis=1),
                                  np.sort(new_vertices, axis=1)):
                raise ValueError('Mapped cell vertex sets differ')
        print(f'Exact vertex/type/connectivity mapping verified for {len(tb)} cells.', flush=True)
        partial = output.with_suffix('.partial.vtkhdf')
        fields = {}
        with h5py.File(partial, 'w') as result:
            for key, value in target.attrs.items():
                result.attrs[key] = value
            for key in target:
                if key != 'VTKHDF':
                    target.copy(key, result)
            group = result.create_group('VTKHDF')
            for key, value in b.attrs.items():
                group.attrs[key] = value
            for key in b:
                if key != 'CellData':
                    owner = a if key == 'FieldData' and key in a else b
                    owner.copy(key, group)
            data = group.create_group('CellData')
            for key, value in a['CellData'].attrs.items():
                data.attrs[key] = value
            for name, dataset in a['CellData'].items():
                values = dataset[()]
                if values.shape[0] != len(ta) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid source cell field: {name}')
                if 'Turbulence' in name and np.any(values <= 0):
                    raise ValueError(f'Nonpositive turbulence field: {name}')
                mapped = values[cell_map]
                saved = data.create_dataset(name, data=mapped)
                for key, value in dataset.attrs.items():
                    saved.attrs[key] = value
                if not np.array_equal(saved[()], mapped):
                    raise ValueError(f'Reordered field write mismatch: {name}')
                fields[name] = dict(shape=list(mapped.shape), minimum=float(mapped.min()),
                                    maximum=float(mapped.max()), exact=True)
        partial.replace(output)
    mapping = output.with_suffix('.target_to_source_cells.npy')
    np.save(mapping, cell_map)
    audit = dict(source=str(source), template=str(template), output=str(output), cells=len(cell_map),
                 points=len(pa), exact_vertex_coordinates=True, bijective_cell_mapping=True,
                 every_cell_vertex_set_verified=True, field_interpolation=False, fields=fields,
                 target_to_source_mapping=str(mapping),
                 cell_mapping_sha256=hashlib.sha256(cell_map.tobytes()).hexdigest(), passed=True)
    output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    remap(args.source.resolve(), args.template.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
