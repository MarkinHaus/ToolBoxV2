#!/usr/bin/env bash
# tests/test_installer_integration.sh
# Voraussetzung: docker verfügbar
# Ausführen: bash tests/test_installer_integration.sh

set -euo pipefail

IMAGE="ubuntu:24.04"
INSTALLER="$(dirname "$0")/../installer.sh"
PASS=0; FAIL=0

run_test() {
    local name="$1"; shift
    local cmd="$*"
    echo -n "  TEST [$name] ... "
    if eval "$cmd" &>/dev/null; then
        echo "✅ PASS"; ((PASS++))
    else
        echo "❌ FAIL"; ((FAIL++))
    fi
}

echo ""
echo "=== Integration: Fresh Install (Docker) ==="
echo ""

# --- Test 2.1: Installer läuft durch ohne Error-Exit ---
run_test "fresh_install_exit_0" \
  "docker run --rm -v '$INSTALLER:/installer.sh:ro' '$IMAGE' bash -c \
    'apt-get update -q && apt-get install -yq python3 python3-pip python3-venv git curl && \
     INSTALL_MODE_OVERRIDE=enduser bash /installer.sh 2>&1 | tail -5 | grep -q \"Installation Complete\"'"

# --- Test 2.2: 'tb' binary existiert nach Installation ---
run_test "tb_binary_exists" \
  "docker run --rm -v '$INSTALLER:/installer.sh:ro' '$IMAGE' bash -c \
    'apt-get update -q && apt-get install -yq python3 python3-pip python3-venv git && \
     bash /installer.sh && \
     test -f \$HOME/.local/bin/tb'"

# --- Test 2.3: Zweiter Durchlauf (Update) findet bestehende Installation ---
run_test "update_detects_existing" \
  "docker run --rm -v '$INSTALLER:/installer.sh:ro' '$IMAGE' bash -c \
    'apt-get update -q && apt-get install -yq python3 python3-pip python3-venv git && \
     bash /installer.sh && \
     bash /installer.sh 2>&1 | grep -q \"Found existing install\"'"

# --- Test 2.4: 'tb manifest' subcommand erreichbar ---
run_test "tb_manifest_reachable" \
  "docker run --rm -v '$INSTALLER:/installer.sh:ro' '$IMAGE' bash -c \
    'apt-get update -q && apt-get install -yq python3 python3-pip python3-venv git && \
     bash /installer.sh && \
     \$HOME/.local/bin/tb manifest --help 2>&1 | grep -q \"manifest\"'"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
