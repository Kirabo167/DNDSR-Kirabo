#!/usr/bin/env bash

set -euo pipefail

probe=${1:-/tmp/ncfv_t2_serial_probe_after_init_opt}
output_root=${2:-/tmp/ncfv-t2-fresh-mixed-optimized-20260914}
project_root=$(cd "$(dirname "$0")/../../../.." && pwd)

if [[ -e "$output_root/complete.marker" ]]
then
    echo "Refusing to reuse a completed result directory: $output_root" >&2
    exit 1
fi

mkdir -p "$output_root/results" "$output_root/logs"

export OMP_NUM_THREADS=1
export DNDS_DIST_OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$project_root/build"

for iv in 10 20 40 80
do
    case "$iv" in
        10|20) ranks=1 ;;
        40) ranks=8 ;;
        80) ranks=32 ;;
    esac

    for family in prism hex tet
    do
        case "$family" in
            prism) config="../cases/ncfv_euler/3D/NCFV_iv${iv}.json" ;;
            hex|tet) config="../cases/ncfv_euler/3D/NCFV_periodic_${family}_iv${iv}.json" ;;
        esac

        for mode in efficient traditional
        do
            name="${family}_iv${iv}_${mode}_np${ranks}"
            result="$output_root/results/${name}.json"
            log="$output_root/logs/${name}.log"
            if [[ -e "$result" ]]
            then
                echo "Refusing to reuse existing result: $result" >&2
                exit 1
            fi

            echo "START $(date --iso-8601=seconds) $name"
            run_start=$SECONDS
            mpirun --mca pml ob1 --mca btl self,vader \
                --bind-to core --map-by core -np "$ranks" \
                "$probe" "$config" "$mode" "$result" 2 '' '' \
                >"$log" 2>&1
            echo "DONE  $(date --iso-8601=seconds) $name wall_seconds=$((SECONDS - run_start))"
        done
    done
done

date --iso-8601=seconds >"$output_root/complete.marker"
echo "ALL_COMPLETE $(date --iso-8601=seconds)"
