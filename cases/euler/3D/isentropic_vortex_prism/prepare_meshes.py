"""Copy the existing NCFV Prism6 IV meshes with CFV periodic boundary names.

Only CGNS section and BC names change; coordinates and connectivity are unchanged.
Run from any directory with the project's Python environment (h5py required).
"""

from pathlib import Path
import shutil

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
NAMES = {
    "bc-2": "PERIODIC_1",
    "bc-2-1": "PERIODIC_1_DONOR",
    "bc-3": "PERIODIC_2",
    "bc-3-1": "PERIODIC_2_DONOR",
    "bc-4": "PERIODIC_3",
    "bc-4-1": "PERIODIC_3_DONOR",
}


def main():
    for n in (10, 20, 40, 80):
        source = ROOT / f"cases/ncfv_euler/3D/iv{n}_3d_1_dnds.cgns"
        destination = HERE / f"iv{n}_cfv_periodic.cgns"
        if destination.exists():
            with h5py.File(destination) as mesh:
                zone = mesh["Base/blk-1"]
                for renamed in NAMES.values():
                    for group in (zone, zone["ZoneBC"]):
                        if renamed not in group or group[renamed].attrs.get_id("name").dtype.str != "|S33":
                            raise ValueError(f"Invalid existing CGNS boundary name in {destination}: {renamed}")
            print(f"iv{n}: already exists: {destination}")
            continue
        shutil.copyfile(source, destination)
        try:
            with h5py.File(destination, "r+") as mesh:
                zone = mesh["Base/blk-1"]
                for original, renamed in NAMES.items():
                    for group in (zone, zone["ZoneBC"]):
                        group.move(original, renamed)
                        # CGNS/HDF5 stores node names as fixed-width S33. Replacing
                        # the attribute with a variable-length string corrupts
                        # cg_boco_info's name buffer on some CGNS builds.
                        del group[renamed].attrs["name"]
                        group[renamed].attrs.create("name", np.bytes_(renamed), dtype="S33")
            print(f"iv{n}: {destination}")
        except Exception:
            destination.unlink()
            raise


if __name__ == "__main__":
    main()
