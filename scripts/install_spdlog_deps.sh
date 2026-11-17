#!/usr/bin/env bash
# Additional pinned header-only dependency for the native Cauchy frontend.
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
destination="${project_root}/external/spdlog"
if [[ -f "${destination}/include/spdlog/spdlog.h" ]]; then
    echo "spdlog already available at ${destination}"
    exit 0
fi
if [[ -e "$destination" ]]; then
    echo "Refusing to replace incomplete ${destination}; inspect it first." >&2
    exit 1
fi
stage="$(mktemp -d "${project_root}/external/.spdlog-stage.XXXXXX")"
trap 'rm -rf -- "$stage"' EXIT
archive="${1:-${stage}/spdlog-v1.15.3.tar.gz}"
if [[ $# -eq 0 ]]; then
    curl --fail --location --retry 3 --output "$archive" \
        https://codeload.github.com/gabime/spdlog/tar.gz/refs/tags/v1.15.3
fi
printf '%s  %s\n' '15a04e69c222eb6c01094b5c7ff8a249b36bb22788d72519646fb85feb267e67' "$archive" | sha256sum --check -
tar -xzf "$archive" -C "$stage"
mv -- "${stage}/spdlog-1.15.3" "$destination"
