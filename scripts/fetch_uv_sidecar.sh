#!/usr/bin/env bash
# Fetch a pinned uv binary as Tauri sidecar (bundle.externalBin "binaries/uv").
#
#   bash scripts/fetch_uv_sidecar.sh <rust-target-triple> [dest_dir]
#
# Result: <dest_dir>/uv-<triple>[.exe]  (Tauri strips the triple when bundling;
# installer.rs passes the bundled path to installer.sh/ps1 as TB_UV_BIN).
# uv release assets are named by the same Rust target triples.
set -euo pipefail

TRIPLE="${1:?usage: fetch_uv_sidecar.sh <target-triple> [dest_dir]}"
DEST="${2:-toolboxv2/simple-core/src-tauri/binaries}"
UV_VERSION="${UV_VERSION:-0.13.7}"
BASE="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}"

case "$TRIPLE" in
  *windows*) ARCHIVE="uv-${TRIPLE}.zip";    EXE=".exe" ;;
  *)         ARCHIVE="uv-${TRIPLE}.tar.gz"; EXE="" ;;
esac

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$DEST"

curl -fsSL "${BASE}/${ARCHIVE}" -o "${TMP}/${ARCHIVE}"
curl -fsSL "${BASE}/${ARCHIVE}.sha256" -o "${TMP}/${ARCHIVE}.sha256"

# Verify checksum (file format: "<sha256>  <name>")
expected="$(awk '{print $1}' "${TMP}/${ARCHIVE}.sha256")"
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "${TMP}/${ARCHIVE}" | awk '{print $1}')"
else
  actual="$(shasum -a 256 "${TMP}/${ARCHIVE}" | awk '{print $1}')"
fi
[ "$expected" = "$actual" ] || { echo "uv checksum mismatch for ${ARCHIVE}" >&2; exit 1; }

case "$ARCHIVE" in
  *.zip)    (cd "$TMP" && unzip -q "$ARCHIVE") ;;
  *.tar.gz) tar -xzf "${TMP}/${ARCHIVE}" -C "$TMP" ;;
esac

bin="$(find "$TMP" -type f -name "uv${EXE}" | head -1)"
[ -n "$bin" ] || { echo "uv${EXE} not found in ${ARCHIVE}" >&2; exit 1; }
install -m 0755 "$bin" "${DEST}/uv-${TRIPLE}${EXE}"
echo "uv ${UV_VERSION} -> ${DEST}/uv-${TRIPLE}${EXE}"
