# Third-Party Binary Dependencies

DNDSR is licensed under GPL-3.0-only. Its local Python wheel workflow also
copies several shared libraries built by `external/cfd_externals` into
`DNDSR/_lib/dndsr_external/`. This inventory documents those copies; it is not
a substitute for the complete license texts or a legal review.

| Component | Bundled library | Current source version | Upstream license text |
|---|---|---:|---|
| CGNS | `libcgns.so*` | 4.5.0 | `external/cfd_externals/repos/cgns/license.txt` (Zlib) |
| HDF5 | `libhdf5.so*` | 1.14.3 | `external/cfd_externals/repos/hdf5/COPYING` (HDF5) and `external/cfd_externals/repos/hdf5/COPYING_LBNL_HDF5` (BSD-3-Clause-LBNL) |
| METIS | `libmetis.so` | 5.1.0 | `external/cfd_externals/repos/parmetis_fix/metis/LICENSE.txt` (Apache-2.0 notice; a distribution also needs the complete Apache-2.0 text) |
| ParMETIS | `libparmetis.so` | 4.0.3 | `external/cfd_externals/repos/parmetis_fix/LICENSE.txt` (custom restrictive terms) |
| zlib | `libz.so*` | 1.3.1 | `external/cfd_externals/repos/zlib/LICENSE` (Zlib) |

Complete redistributable copies of these notices, plus the Apache-2.0
text required by METIS, live under `licenses/`. The Python build declares the
inventory and those texts as PEP 639 license files, so a wheel stages them in
its standard `.dist-info/licenses/` directory. Their inclusion records terms;
it does not change those terms or authorize ParMETIS redistribution.

MPI and the C++ standard library are linked from the host and are deliberately
not bundled. Consequently, a locally built wheel is tied to the MPI ABI used
to build it even though that ABI is not represented in the wheel filename.

Cauchy's operational logger uses header-only spdlog 1.15.3 with the project's
external fmt configuration. `scripts/install_spdlog_deps.sh` pins its source
archive and SHA256. The spdlog MIT notice is retained in
`licenses/spdlog-MIT.txt` and included by the existing wheel license-file glob.

## Distribution Status

The current wheel is a development/internal artifact, not an approved public
binary distribution. In particular, the ParMETIS 4.0.3 terms restrict use and
prohibit sale or redistribution without prior approval. Copying its license
into a wheel does not grant that approval.

Before publishing or redistributing a wheel, the distributor must at least:

1. obtain and retain any permission required for ParMETIS, or redesign the
   package so ParMETIS is not redistributed;
2. include the complete applicable license and notice texts for every bundled
   component, including both HDF5 texts and the full Apache-2.0 license;
3. review compatibility between the project license and every bundled binary;
4. build and test against the target system's MPI implementation/ABI; and
5. inspect every packaged ELF dependency and RPATH/RUNPATH before release.

The generated sdist is also not a standalone installation artifact: the
header bundle and compiled `cfd_externals` tree are prepared separately. Do
not publish it as a source-install release until that build model is made
self-contained and verified in an isolated environment.
