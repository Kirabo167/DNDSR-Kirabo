#!/usr/bin/env bash
# Regenerate JSON Schema files for all solver variants.
#
# Usage:
#   cd <project_root>
#   bash cases/update_schemas.sh [build_dir]
#
# Euler schemas require a Cantera-enabled build (default: build-reactive).
# ACM/NCFV executables may also come from a CPU build (second argument: build).
# Set DNDS_SCHEMA_CHECK_ONLY=1 to generate and validate without replacing files.

set -euo pipefail

BUILD_DIR="${1:-build-reactive}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CPU_BUILD_DIR="${2:-build}"
[[ "$BUILD_DIR" = /* ]] || BUILD_DIR="${PROJECT_DIR}/${BUILD_DIR}"
[[ "$CPU_BUILD_DIR" = /* ]] || CPU_BUILD_DIR="${PROJECT_DIR}/${CPU_BUILD_DIR}"

VARIANTS=(euler euler2D euler3D eulerSA eulerSA3D euler2EQ euler2EQ3D eulerEX eulerEX3D
    acm2D acm3D acmVariable2D acmVariable3D ncfv_euler2D ncfv_euler3D)
declare -A CASE_DIRECTORIES=(
    [euler]=euler/2D [euler2D]=euler/2D [euler3D]=euler/3D
    [eulerSA]=euler/2D/SA [eulerSA3D]=euler/3D/SA
    [euler2EQ]=euler/2D/2EQ [euler2EQ3D]=euler/3D/2EQ
    [eulerEX]=euler/2D/EX [eulerEX3D]=euler/3D/EX
    [acm2D]=acm/2D [acm3D]=acm/3D
    [acmVariable2D]=acm/2D/variable_density [acmVariable3D]=acm/3D/variable_density
    [ncfv_euler2D]=ncfv_euler/2D [ncfv_euler3D]=ncfv_euler/3D)

solver_executable()
{
    local variant="$1"
    if [[ "$variant" != euler* && ! -x "${BUILD_DIR}/app/${variant}.exe" ]]; then
        printf '%s\n' "${CPU_BUILD_DIR}/app/${variant}.exe"
    else
        printf '%s\n' "${BUILD_DIR}/app/${variant}.exe"
    fi
}

cache="${BUILD_DIR}/CMakeCache.txt"
if [[ ! -f "$cache" ]] || ! grep -q '^DNDS_USE_CANTERA:BOOL=ON$' "$cache"; then
    echo "Error: schema regeneration requires a configured DNDS_USE_CANTERA=ON build." >&2
    echo "Run: cmake --preset reactive-test && cmake --build --preset schemas -j8" >&2
    exit 1
fi

# Refuse a partial refresh: leaving old schemas in place while reporting
# success makes configuration compatibility failures very difficult to spot.
missing=()
for v in "${VARIANTS[@]}"; do
    exe="$(solver_executable "$v")"
    if [[ ! -x "$exe" ]]; then
        missing+=("$exe")
    fi
done
if (( ${#missing[@]} > 0 )); then
    echo "Error: all schema-producing solver executables must be built first:" >&2
    printf '  missing %s\n' "${missing[@]}" >&2
    exit 1
fi

# Generate and validate every schema before replacing any committed output.
# Staging beside the destination keeps the final rename on the same filesystem.
staging_dir="$(mktemp -d "${SCRIPT_DIR}/.schema-update.XXXXXX")"
cleanup()
{
    if [[ -d "$staging_dir" ]]; then
        rm -rf -- "$staging_dir"
    fi
}
trap cleanup EXIT

for v in "${VARIANTS[@]}"; do
    exe="$(solver_executable "$v")"
    staged="${staging_dir}/${v}_schema.json"
    stderr_log="${staging_dir}/${v}.stderr.log"
    if mpirun -np 1 "$exe" --emit-schema 2>"$stderr_log" \
        | grep -v '^JSON:' > "$staged"; then
        if ! python3 -m json.tool "$staged" >/dev/null; then
            echo "FAIL $v: generated output is not valid JSON" >&2
            exit 1
        fi
        echo "  generated $v"
    else
        if [[ -s "$stderr_log" ]]; then
            cat "$stderr_log" >&2
        fi
        echo "FAIL $v: mpirun pipeline failed" >&2
        exit 1
    fi
done

# Each rename is atomic, and this phase is reached only after all staged
# schemas have passed generation and JSON parsing.
for v in "${VARIANTS[@]}"; do
    out="${SCRIPT_DIR}/${CASE_DIRECTORIES[$v]}/${v}_schema.json"
    if [[ "${DNDS_SCHEMA_CHECK_ONLY:-0}" != 1 ]]; then
        mkdir -p -- "$(dirname "$out")"
        mv -- "${staging_dir}/${v}_schema.json" "$out"
    fi
    echo "  $v -> ${CASE_DIRECTORIES[$v]}/${v}_schema.json"
done

echo "Done."
