"""T2/F15 Regression-Guard: .gitignore deckt Runtime-State ab, ohne getrackte Dateien zu ignorieren.

Laeuft stdlib-only (subprocess + git check-ignore), pytest-entdeckbar.
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

MUST_IGNORE = [
    ".info/",                                    # Runtime-State root
    "logs/",                                     # Logs root
    "toolboxv2/.info/",                          # Runtime-State package
    "toolboxv2/features_packed/",                # Build-Output Feature-Packer
    "toolboxv2/features/isaa/.installed",        # Feature-Install-Marker
    "toolboxv2/features/cli/.installed",
    # R8a: Rueckstaende des Wheel-Skripts / eines Worker-Starts
    "toolboxv2/node_modules/",
    "toolboxv2/package-lock.json",
    "toolboxv2/web/package-lock.json",
    "toolboxv2/minio_cluster_config.json",
]

# Getrackte Dateien, die trotz aehnlicher Namen NICHT ignoriert werden duerfen:
MUST_TRACK = [
    "toolboxv2/mods/TruthSeeker/.info/device.key",   # Kollisions-Guard
    "init.config",
    "toolboxv2/tbjs/dist/tbjs.css",
]

def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)

def test_runtime_state_ignored():
    for p in MUST_IGNORE:
        r = _git("check-ignore", "-q", p)
        assert r.returncode == 0, f"NICHT ignoriert: {p} (gitignore fehlt?"

def test_tracked_files_still_tracked():
    tracked = set(_git("ls-files").stdout.splitlines())
    for p in MUST_TRACK:
        assert p in tracked, f"Guard-Baseline ungueltig, nicht getrackt: {p}"
        r = _git("check-ignore", "-q", p)
        assert r.returncode != 0, f"getrackte Datei wuerde neu ignoriert: {p}"

def test_tracked_ignored_baseline_frozen():
    # Bekannt ignoriert-getrackt: genau diese Altlasten. Mehr = Regression.
    # wheels/.gitkeep seit R7 (dir-Marker unter dem ignorierten wheels/).
    ci = set(_git("ls-files", "-ci", "--exclude-standard").stdout.splitlines())
    expected = {
        "init.config",
        "llm-gateway/data/config.json",
        "toolboxv2/simple-core/src-tauri/wheels/.gitkeep",
        "toolboxv2/tbjs/dist/tbjs.css",
    }
    assert ci == expected, f"Baseline verschoben: neu={ci - expected}, weg={expected - ci}"
