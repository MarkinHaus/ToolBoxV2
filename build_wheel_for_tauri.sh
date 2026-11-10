#!/usr/bin/env bash
# T6/F4: Kern-Wheel fuer das Tauri-Bundle bauen (Git Bash / CI).
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) Web-UI bauen
cd toolboxv2 && npm install && npm install --prefix ./web && npm run build && cd ..

# 2) Schutz vor CI-Stub
[ -f toolboxv2/dist/index.html ] || { echo "ERROR: toolboxv2/dist/index.html fehlt (CI-Stub?)"; exit 1; }
[ "$(wc -c < toolboxv2/dist/index.html)" -ge 500 ] || { echo "ERROR: index.html zu klein (<500 B)"; exit 1; }

# 3) alte Artefakte
rm -rf dist toolboxv2/features_packed

# 4) Wheel bauen
tb build build

# 5) Web-ZIP muss die UI enthalten
zipfile=$(ls toolboxv2/features_packed/tbv2-feature-web-*.zip 2>/dev/null | head -1 || true)
[ -n "$zipfile" ] || { echo "ERROR: kein tbv2-feature-web-*.zip"; exit 1; }
python -m zipfile -l "$zipfile" | grep -q "dist/index.html" || { echo "ERROR: ZIP ohne dist/index.html"; exit 1; }

# 6) Wheel ins Bundle
mkdir -p toolboxv2/simple-core/src-tauri/wheels
rm -f toolboxv2/simple-core/src-tauri/wheels/*.whl
cp dist/toolboxv2-*.whl toolboxv2/simple-core/src-tauri/wheels/
echo "OK: $(ls toolboxv2/simple-core/src-tauri/wheels/)"
