#!/usr/bin/env python3
"""Generate the constrained surface mesh for the SUBOFF prism-front mesher.

Run with netgen-mesher 6.2.2608 and NumPy. The input domain STL contains an
outward-facing box and an inward-facing submarine, in metres. A mesh-size
file controls surface sizing. Volume refinement is a separate step.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from netgen.stl import STLGeometry
from pyngcore import TaskManager, SetNumThreads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--min-size", type=float)
    parser.add_argument("--curvature-factor", type=float)
    args = parser.parse_args()
    directory = args.directory.resolve()
    SetNumThreads(args.threads)
    geometry = STLGeometry(str(directory / "fluid_domain.stl"))
    options = dict(
        maxh=1.5,
        grading=0.3,
        meshsizefilename=str(directory / "local_sizes.txt"),
        yangle=35,
        contyangle=20,
        closeedgefac=1.0,
        optsteps2d=3,
        optsteps3d=3,
    )
    if args.min_size is not None:
        options["minh"] = args.min_size
    if args.curvature_factor is not None:
        options["surfmeshcurvfac"] = args.curvature_factor
    with TaskManager():
        mesh = geometry.GenerateMesh(perfstepsend=4, **options)
    print("SURFACE_DONE", len(mesh.Points()), len(mesh.Elements2D()), flush=True)
    points = np.array([tuple(p.p) for p in mesh.Points()])
    wall_indices = set()
    for element in mesh.Elements2D():
        vertices = np.array([v.nr - 1 for v in element.vertices])
        center = points[vertices].mean(axis=0)
        if -0.01 < center[0] < 4.37 and np.linalg.norm(center[1:]) < 0.7:
            wall_indices.add(element.index)
    if not wall_indices:
        raise RuntimeError("No submarine wall surfaces found")
    for index, descriptor in enumerate(mesh.FaceDescriptors(), 1):
        name = "WALL" if index in wall_indices else "OUTER"
        mesh.SetBCName(descriptor.bc - 1, name)
    mesh.SetMaterial(1, "fluid")
    mesh.Save(str(directory / "surface.vol"))
    surface_counts = Counter(element.index for element in mesh.Elements2D())
    summary = {
        "surface_points": len(points),
        "surface_triangles": len(mesh.Elements2D()),
        "wall_triangles": sum(surface_counts[i] for i in wall_indices),
        "wall_descriptors": sorted(wall_indices),
    }
    (directory / "surface_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SURFACE_SUMMARY", json.dumps(summary), flush=True)
    surfaces = mesh.Elements2D().NumPy()
    np.savez(directory / "surface_arrays.npz", points=points,
             triangles=surfaces["nodes"][:, :3] - 1, patch=surfaces["index"])


if __name__ == "__main__":
    main()
