#!/usr/bin/env bash
# Install the pinned header-only dependency bundle used by DNDSR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTERNAL_DIR="${PROJECT_ROOT}/external"
ARCHIVE="${EXTERNAL_DIR}/external_headeronlys.tar.gz"

HEADERONLYS_VERSION="v0.1.0"
HEADERONLYS_SHA256="4550723d321fb31d3a1e52e5bb40b8e23fc8ae15f233d620f16b8139b9f38e4a"
HEADERONLYS_URL="https://github.com/harryzhou2000/cfd_externals_headeronlys/releases/download/${HEADERONLYS_VERSION}/external_headeronlys.tar.gz"

EXPECTED_DIRS=(
    CGAL
    argparse
    boost
    cppcodec
    cpptrace
    doctest
    doxygen-awesome-css
    eigen
    exprtk
    fmt
    nanoflann
    nlohmann
    pybind11
    pybind11_json
)

ARCHIVE_TMP=""
STAGING_ROOT=""
BACKUP_ROOT=""
REPLACEMENT_STARTED=0
REPLACEMENT_COMPLETE=0
declare -A HAD_ORIGINAL=()
declare -A INSTALLED=()

cleanup()
{
    local status=$?
    local dependency target backup failed
    local rollback_failed=0

    # A failure during replacement restores every directory already moved.
    # Staging and backups live under external/, so all moves stay on one
    # filesystem and each individual directory replacement is atomic.
    if (( REPLACEMENT_STARTED && ! REPLACEMENT_COMPLETE )); then
        set +e
        echo "Header-only dependency installation failed; restoring previous directories." >&2
        for dependency in "${EXPECTED_DIRS[@]}"; do
            target="${EXTERNAL_DIR}/${dependency}"
            backup="${BACKUP_ROOT}/${dependency}"
            failed="${STAGING_ROOT}/.failed-${dependency}"

            if [[ "${INSTALLED[$dependency]:-0}" == 1 ]] &&
                [[ -e "$target" || -L "$target" ]]; then
                mv -- "$target" "$failed"
            fi
            if [[ "${HAD_ORIGINAL[$dependency]:-0}" == 1 ]] &&
                [[ -e "$backup" || -L "$backup" ]]; then
                if ! mv -- "$backup" "$target"; then
                    rollback_failed=1
                    echo "Error: could not restore ${target}; backup retained at ${backup}" >&2
                fi
            fi
        done
    fi

    [[ -z "$ARCHIVE_TMP" || ! -e "$ARCHIVE_TMP" ]] || rm -f -- "$ARCHIVE_TMP"
    [[ -z "$STAGING_ROOT" || ! -d "$STAGING_ROOT" ]] || rm -rf -- "$STAGING_ROOT"
    if (( ! rollback_failed )); then
        [[ -z "$BACKUP_ROOT" || ! -d "$BACKUP_ROOT" ]] || rm -rf -- "$BACKUP_ROOT"
    fi

    trap - EXIT
    exit "$status"
}

trap cleanup EXIT

archive_is_valid()
{
    [[ -f "$ARCHIVE" ]] &&
        printf '%s  %s\n' "$HEADERONLYS_SHA256" "$ARCHIVE" | sha256sum --check --status
}

if ! archive_is_valid; then
    ARCHIVE_TMP="$(mktemp "${EXTERNAL_DIR}/.external_headeronlys.tar.gz.XXXXXX")"
    curl --fail --location --retry 3 --output "$ARCHIVE_TMP" "$HEADERONLYS_URL"
    printf '%s  %s\n' "$HEADERONLYS_SHA256" "$ARCHIVE_TMP" | sha256sum --check -
    mv -- "$ARCHIVE_TMP" "$ARCHIVE"
    ARCHIVE_TMP=""
fi

STAGING_ROOT="$(mktemp -d "${EXTERNAL_DIR}/.headeronlys-stage.XXXXXX")"
tar -xzf "$ARCHIVE" -C "$STAGING_ROOT"

mapfile -t actual_entries < <(
    find "$STAGING_ROOT" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort
)
mapfile -t expected_entries < <(
    printf '%s\n' "${EXPECTED_DIRS[@]}" | LC_ALL=C sort
)

if [[ "${actual_entries[*]}" != "${expected_entries[*]}" ]]; then
    echo "Error: header-only archive has an unexpected top-level layout" >&2
    echo "  expected: ${expected_entries[*]}" >&2
    echo "  actual:   ${actual_entries[*]}" >&2
    exit 1
fi

for dependency in "${EXPECTED_DIRS[@]}"; do
    if [[ ! -d "${STAGING_ROOT}/${dependency}" ||
        -L "${STAGING_ROOT}/${dependency}" ]]; then
        echo "Error: expected a real directory in archive: ${dependency}" >&2
        exit 1
    fi
done

BACKUP_ROOT="$(mktemp -d "${EXTERNAL_DIR}/.headeronlys-backup.XXXXXX")"
REPLACEMENT_STARTED=1
for dependency in "${EXPECTED_DIRS[@]}"; do
    target="${EXTERNAL_DIR}/${dependency}"
    backup="${BACKUP_ROOT}/${dependency}"

    if [[ -e "$target" || -L "$target" ]]; then
        HAD_ORIGINAL["$dependency"]=1
        mv -- "$target" "$backup"
    else
        HAD_ORIGINAL["$dependency"]=0
    fi

    # Mark before mv so the EXIT rollback also covers a signal arriving
    # immediately after the directory becomes visible at its final path.
    INSTALLED["$dependency"]=1
    mv -- "${STAGING_ROOT}/${dependency}" "$target"
done
REPLACEMENT_COMPLETE=1

bash "${SCRIPT_DIR}/install_spdlog_deps.sh"
echo "Installed header-only dependencies ${HEADERONLYS_VERSION} (${HEADERONLYS_SHA256})."
