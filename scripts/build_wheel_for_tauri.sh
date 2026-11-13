#!/usr/bin/env bash
# Build the core wheel (with the web UI inside) and put it into the Tauri bundle.
#
#   bash scripts/build_wheel_for_tauri.sh
#
# Needs: node/npm, and `tb` on PATH (pip install -e . build).
# Result: toolboxv2/simple-core/src-tauri/wheels/toolboxv2-<version>-py3-none-any.whl
#         toolboxv2/simple-core/src-tauri/wheels/BUILD_ID   (sha256 of that wheel)
# The Tauri app hands the wheel to installer.sh/ps1 as TB_WHEEL and the BUILD_ID
# as TB_CORE_BUILD (installer.rs); the core update compares that id (lib.rs).
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) web UI (the web feature zip packs toolboxv2/dist)
(
  cd toolboxv2
  npm install
  npm install --prefix ./web
  npm run build
)

# 2) guard against the CI stub page
[ -f toolboxv2/dist/index.html ] || { echo "ERROR: toolboxv2/dist/index.html missing" >&2; exit 1; }
[ "$(wc -c < toolboxv2/dist/index.html)" -ge 500 ] || { echo "ERROR: toolboxv2/dist/index.html is a stub (<500 B)" >&2; exit 1; }

# 3) old artifacts
rm -rf dist toolboxv2/features_packed

# 4) feature zips + wheel
tb build build

# 5) the web zip must contain the UI
zipfile="$(ls toolboxv2/features_packed/tbv2-feature-web-*.zip 2>/dev/null | head -1 || true)"
[ -n "$zipfile" ] || { echo "ERROR: no tbv2-feature-web-*.zip" >&2; exit 1; }
listing="$(python -m zipfile -l "$zipfile")"   # no pipe into grep -q: pipefail + early exit
case "$listing" in *dist/index.html*) ;; *) echo "ERROR: web zip without dist/index.html" >&2; exit 1 ;; esac

# 6) wheel + build id into the bundle
dest=toolboxv2/simple-core/src-tauri/wheels
mkdir -p "$dest"
rm -f "$dest"/*.whl "$dest"/BUILD_ID
cp dist/toolboxv2-*.whl "$dest"/
wheel="$(ls "$dest"/toolboxv2-*.whl | head -1)"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$wheel" | awk '{print $1}' > "$dest/BUILD_ID"
else
  shasum -a 256 "$wheel" | awk '{print $1}' > "$dest/BUILD_ID"
fi
echo "OK: $(basename "$wheel") build=$(cut -c1-12 "$dest/BUILD_ID")"
