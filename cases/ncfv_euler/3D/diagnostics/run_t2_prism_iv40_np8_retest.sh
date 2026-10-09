#!/usr/bin/env bash

set -euo pipefail

probe=${1:-/tmp/ncfv_t2_serial_probe}
output_root=${2:-/tmp/ncfv-t2-prism-iv40-np8-retest-20260914}
project_root=$(cd "$(dirname "$0")/../../../.." && pwd)

mkdir -p "$output_root/results" "$output_root/logs"

export OMP_NUM_THREADS=1
export DNDS_DIST_OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$project_root/build"

for mode in efficient traditional
do
    name="original_iv40_${mode}_np8"
    result="$output_root/results/${name}.json"
    log="$output_root/logs/${name}.log"
    config="../cases/ncfv_euler/3D/NCFV_iv40.json"

    if [[ -e "$result" ]]
    then
        echo "Refusing to reuse existing result: $result" >&2
        exit 1
    fi

    echo "START $(date --iso-8601=seconds) $name"
    run_start=$SECONDS
    mpirun --mca pml ob1 --mca btl self,vader \
        --bind-to core --map-by core -np 8 \
        "$probe" "$config" "$mode" "$result" 2 '' '' >"$log" 2>&1
    echo "DONE  $(date --iso-8601=seconds) $name wall_seconds=$((SECONDS - run_start))"
done

echo "ALL_COMPLETE $(date --iso-8601=seconds)"
