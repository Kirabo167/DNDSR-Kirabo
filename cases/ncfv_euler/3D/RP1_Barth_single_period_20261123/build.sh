#!/usr/bin/env bash
# Build the single-layer experimental operator with its original framework API.
set -euo pipefail
case_dir=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$case_dir/../../../.." && pwd)
work=${1:-"$root/build/single_layer_barth"}
mkdir -p "$work"
work=$(cd "$work" && pwd)
revision=bee9b412873e090ed0e3fed38a281116f97dbdd3
# Keep the source materialization private to this build, even on failure or retry.
exec 9>"$work/build.lock"
flock 9
build_id=$(cat "$case_dir/build.sh" "$case_dir/operator.patch" | sha256sum | cut -c1-16)
variant="$work/$build_id"
mkdir -p "$variant"
src="$variant/framework"
if [[ ! -d "$src" ]]; then
    stage=$(mktemp -d "$work/source.XXXXXXXX")
    trap 'rm -rf -- "$stage"' EXIT
    git -C "$root" archive "$revision" CMakeLists.txt cmakeCommonUtils.cmake VERSION cmake \
        src app examples pyproject.toml cases/ncfv_euler/3D/diagnostics docs/doxygen/Doxyfile \
        | tar -xf - -C "$stage"
    (cd "$stage" && patch --batch --fuzz=0 -p1 < "$case_dir/operator.patch")
    for header in NCFVPlanarCompression.hpp NCFVDifferential.hxx NCFVLocalLimiterMath.hxx PeriodicLayer.hpp SingleLayerRun.hxx; do
        ln -s "$case_dir/$header" "$stage/src/NCFV/$header"
    done
    ln -s "$root/external" "$stage/external"
    mkdir -p "$stage/cases/sensor"
    ln -s "$case_dir/test_planar_compression.cpp" "$stage/cases/sensor/test.cpp"
    cat >> "$stage/CMakeLists.txt" <<'CMAKE'

add_executable(planar_compression_test cases/sensor/test.cpp)
target_include_directories(planar_compression_test PRIVATE src/NCFV external
    ${DNDS_INCLUDES} ${DNDS_EXTERNAL_INCLUDES})
target_link_libraries(planar_compression_test PRIVATE MPI::MPI_CXX)
CMAKE
    mv "$stage" "$src"
    trap - EXIT
fi

export CC=${CC:-mpicc} CXX=${CXX:-mpicxx}
cmake -S "$src" -B "$variant/build" -DCMAKE_BUILD_TYPE=Release \
    -DDNDS_USE_CANTERA=OFF -DDNDS_BUILD_TESTS=OFF -DDNDS_USE_PRECOMPILED_HEADER=OFF
cmake --build "$variant/build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-8}" \
    --target ncfv_euler3D planar_compression_test
"$variant/build/planar_compression_test"
ln -sfn "$variant/build/app/ncfv_euler3D.exe" "$work/ncfv_single_period_3D.exe"
echo "$work/ncfv_single_period_3D.exe"
