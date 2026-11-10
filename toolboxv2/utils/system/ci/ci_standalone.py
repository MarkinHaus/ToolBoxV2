#!/usr/bin/env python3
"""
ToolBoxV2 Nuitka Standalone Builder.

Baut ein onefile-Binary aus dem Mini-Core plus optional weiteren Features.
Nutzt das Feature-System (features/<name>/feature.yaml) zur Auflösung der
einzuschliessenden Module und Dependencies.

Usage:
    tb build standalone                          # mini core, current platform
    tb build standalone --features mini,cli      # mini + cli feature
    tb build standalone --target linux-x64       # explizites Ziel
    tb build standalone --upload                 # -> Registry Artifact
    tb build standalone --all-targets            # alle Targets (braucht Runner)

Platform/Architecture Werte entsprechen den Registry-Enums
(tb-registry/registry/models/package.py):
    Platform:     all | windows | linux | macos | android | ios
    Architecture: all | x64 | x86 | arm64 | arm32
"""
from __future__ import annotations

import os
import platform as _platform
import subprocess
import sys
from pathlib import Path

from toolboxv2.utils.system.ci.ci_build import (
    ROOT,
    TOOLBOXV2_DIR,
    _p,
    console,
    discover_features,
)

# ── Konstanten ────────────────────────────────────────────────────────────────

STANDALONE_DIR = ROOT / "dist_standalone"

#: Registry-Artifact-Name (ArtifactType.CLI_EXECUTABLE)
DEFAULT_ARTIFACT_NAME = "tb-standalone"
ARTIFACT_TYPE = "cli_executable"

#: Default entry point: toolboxv2/__main__.py
DEFAULT_ENTRY = TOOLBOXV2_DIR / "__main__.py"

#: Feature das immer eingeschlossen wird (Core-Runtime).
BASE_FEATURE = "mini"

#: Build-Targets. Keys sind CLI-Namen, Werte matchen die Registry-Enums.
TARGETS: dict[str, dict[str, str]] = {
    "windows-x64": {"platform": "windows", "architecture": "x64", "ext": ".exe"},
    "windows-x86": {"platform": "windows", "architecture": "x86", "ext": ".exe"},
    "windows-arm64": {"platform": "windows", "architecture": "arm64", "ext": ".exe"},
    "linux-x64": {"platform": "linux", "architecture": "x64", "ext": ""},
    "linux-arm64": {"platform": "linux", "architecture": "arm64", "ext": ""},
    "macos-x64": {"platform": "macos", "architecture": "x64", "ext": ""},
    "macos-arm64": {"platform": "macos", "architecture": "arm64", "ext": ""},
}

#: platform.machine() -> Registry Architecture Enum
_MACHINE_MAP = {
    "x86_64": "x64",
    "amd64": "x64",
    "x64": "x64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "armv8": "arm64",
    "i386": "x86",
    "i686": "x86",
    "x86": "x86",
    "armv7l": "arm32",
}

#: platform.system() -> Registry Platform Enum
_SYSTEM_MAP = {"windows": "windows", "linux": "linux", "darwin": "macos"}

#: Schwere Pakete die nie in ein Standalone-Binary gehören.
HEAVY_EXCLUDES = (
    "numpy",
    "scipy",
    "torch",
    "tensorflow",
    "matplotlib",
    "pandas",
    "tkinter",
    "PyQt5",
    "PyQt6",
    "PySide6",
    "customtkinter",
    "IPython",
    "notebook",
    "pytest",
    "sphinx",
)


# ── Target-Erkennung ─────────────────────────────────────────────────────────


def detect_current_target() -> str:
    """Ermittle den Target-Key der laufenden Plattform.

    Returns:
        Key aus TARGETS, z.B. "windows-x64". Fällt auf "linux-x64" zurück
        wenn die Plattform unbekannt ist.
    """
    system = _SYSTEM_MAP.get(_platform.system().lower())
    arch = _MACHINE_MAP.get(_platform.machine().lower())
    if not system or not arch:
        return "linux-x64"
    key = f"{system}-{arch}"
    return key if key in TARGETS else "linux-x64"


def _binary_name(features: list[str], target: str) -> str:
    """Baue den Dateinamen für ein Standalone-Binary."""
    ext = TARGETS[target]["ext"]
    tag = "-".join(sorted(set(features)))
    return f"tb-{tag}-{target}{ext}"


# ── Feature-Auflösung ────────────────────────────────────────────────────────


def resolve_feature_files(
    features: list[str],
) -> tuple[set[str], set[str], list[str], list[str]]:
    """Löse feature.yaml Einträge zu Nuitka-Argumenten auf.

    Liest `files`, `dependencies` und `imports` der angeforderten Features.
    BASE_FEATURE ("mini") wird immer eingeschlossen.

    Args:
        features: Angeforderte Feature-Namen.

    Returns:
        (include_modules, include_packages, data_dirs, pip_deps)
        - include_modules: dotted module names für --include-module
        - include_packages: dotted package names für --include-package
        - data_dirs: absolute Verzeichnisse für --include-data-dir
        - pip_deps: Dependency-Strings aus feature.yaml (Info/Preflight)
    """
    all_features = discover_features()
    requested = {f.strip() for f in features if f.strip()}
    requested.add(BASE_FEATURE)

    include_modules: set[str] = set()
    include_packages: set[str] = set()
    data_dirs: list[str] = []
    pip_deps: list[str] = []
    missing: list[str] = []

    for name in sorted(requested):
        config = all_features.get(name)
        if config is None:
            missing.append(name)
            continue

        pip_deps.extend(config.get("dependencies", []) or [])

        # imports: bereits dotted -> direkt als Package einschliessen
        for imp in config.get("imports", []) or []:
            if imp:
                include_packages.add(imp)

        for pattern in config.get("files", []) or []:
            if not pattern:
                continue
            if pattern.endswith("/*"):
                # Directory: als Package einschliessen + Datenfiles mitnehmen
                rel = pattern[:-2]
                src_dir = TOOLBOXV2_DIR / rel
                if not src_dir.is_dir():
                    continue
                include_packages.add(
                    "toolboxv2." + rel.replace("/", ".").replace(os.sep, ".")
                )
                data_dirs.append(str(src_dir))
            else:
                src_file = TOOLBOXV2_DIR / pattern
                if not src_file.is_file():
                    continue
                mod = "toolboxv2." + pattern.replace("/", ".").replace(os.sep, ".")
                mod = mod.removesuffix(".py").removesuffix(".__init__")
                include_modules.add(mod)

    if missing:
        _p(f"  ! Unbekannte Features ignoriert: {', '.join(missing)}", style="yellow")

    # Nicht angeforderte Features explizit nicht folgen
    return include_modules, include_packages, data_dirs, pip_deps


def _excluded_feature_modules(features: list[str]) -> list[str]:
    """Ermittle Module nicht angeforderter Features für --nofollow-import-to."""
    all_features = discover_features()
    requested = {f.strip() for f in features if f.strip()}
    requested.add(BASE_FEATURE)

    excluded: set[str] = set()
    for name, config in all_features.items():
        if name in requested:
            continue
        for pattern in config.get("files", []) or []:
            if not pattern:
                continue
            rel = pattern[:-2] if pattern.endswith("/*") else pattern
            mod = "toolboxv2." + rel.replace("/", ".").replace(os.sep, ".")
            mod = mod.removesuffix(".py").removesuffix(".__init__")
            excluded.add(mod)

    # Module die auch von einem angeforderten Feature gebraucht werden: behalten
    keep_mods, keep_pkgs, _, _ = resolve_feature_files(features)
    keep = keep_mods | keep_pkgs
    return sorted(m for m in excluded if m not in keep)


# ── Nuitka Build ─────────────────────────────────────────────────────────────


def _nuitka_available() -> bool:
    """Prüfe ob Nuitka importierbar ist."""
    try:
        subprocess.run(
            [sys.executable, "-m", "nuitka", "--version"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def build_nuitka_onefile(
    features: list[str],
    target: str | None = None,
    entry: Path | None = None,
    output_dir: Path | None = None,
    binary_name: str | None = None,
    extra_args: list[str] | None = None,
) -> Path | None:
    """Baue ein Nuitka onefile-Binary.

    Nuitka kann NICHT cross-compilen. `target` dient nur der Benennung und
    den Registry-Metadaten; es muss dem laufenden System entsprechen.

    Args:
        features: Feature-Namen (mini wird immer ergänzt).
        target: Target-Key aus TARGETS. None = aktuelles System.
        entry: Entry-Point Python-Datei. None = toolboxv2/__main__.py
        output_dir: Zielverzeichnis. None = dist_standalone/
        binary_name: Dateiname. None = automatisch.
        extra_args: Zusätzliche Nuitka-Argumente.

    Returns:
        Pfad zum Binary oder None bei Fehler.
    """
    target = target or detect_current_target()
    if target not in TARGETS:
        _p(f"  x Unbekanntes Target: {target}", style="red")
        return None

    current = detect_current_target()
    if target != current:
        _p(
            f"  ! Nuitka kann nicht cross-compilen "
            f"(laufend: {current}, angefordert: {target})",
            style="yellow",
        )
        return None

    if not _nuitka_available():
        _p("  x Nuitka nicht installiert -> pip install nuitka", style="red")
        return None

    entry = entry or DEFAULT_ENTRY
    if not entry.is_file():
        _p(f"  x Entry-Point fehlt: {entry}", style="red")
        return None

    output_dir = output_dir or STANDALONE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_name = binary_name or _binary_name(features, target)

    inc_mods, inc_pkgs, data_dirs, pip_deps = resolve_feature_files(features)
    excluded = _excluded_feature_modules(features)

    _p(f"  Target:   {target}")
    _p(f"  Features: {', '.join(sorted(set(features) | {BASE_FEATURE}))}")
    _p(f"  Entry:    {entry.relative_to(ROOT) if entry.is_relative_to(ROOT) else entry}")
    _p(f"  Packages: {len(inc_pkgs)}  Modules: {len(inc_mods)}")
    if pip_deps:
        _p(f"  Deps:     {', '.join(pip_deps)}")

    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--onefile",
        "--standalone",
        "--assume-yes-for-downloads",
        "--nofollow-imports",
        f"--output-dir={output_dir}",
        f"--output-filename={binary_name}",
    ]
    cmd += [f"--include-package={p}" for p in sorted(inc_pkgs)]
    cmd += [f"--include-module={m}" for m in sorted(inc_mods)]
    cmd += [f"--include-data-dir={d}={Path(d).name}" for d in data_dirs]
    cmd += [f"--nofollow-import-to={m}" for m in HEAVY_EXCLUDES]
    if extra_args:
        cmd += extra_args
    cmd.append(str(entry))

    _p(f"\n  Nuitka: {len(cmd)} Argumente, Build startet ...")
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
    except subprocess.CalledProcessError as e:
        _p(f"  x Nuitka Build fehlgeschlagen (exit {e.returncode})", style="red")
        return None
    except KeyboardInterrupt:
        _p("  x Build abgebrochen", style="red")
        return None

    result = output_dir / binary_name
    if not result.is_file():
        _p(f"  x Binary nicht gefunden: {result}", style="red")
        return None

    size_mb = result.stat().st_size / (1024 * 1024)
    _p(f"  + {result.name} ({size_mb:.1f} MB)")
    return result


# ── Registry Upload ──────────────────────────────────────────────────────────


def upload_standalone(
    binary: Path,
    version: str,
    target: str,
    artifact_name: str = DEFAULT_ARTIFACT_NAME,
    changelog: str = "",
    create_if_missing: bool = True,
) -> bool:
    """Lade ein Standalone-Binary als Registry-Artifact-Build hoch.

    Nutzt RegistryClient.upload_artifact_build().

    Args:
        binary: Pfad zum Binary.
        version: Versions-String.
        target: Target-Key aus TARGETS (liefert platform + architecture).
        artifact_name: Registry-Artifact-Name.
        changelog: Changelog-Text.
        create_if_missing: Artifact anlegen wenn 404.

    Returns:
        True bei Erfolg.
    """
    import asyncio

    meta = TARGETS.get(target)
    if not meta:
        _p(f"  x Unbekanntes Target: {target}", style="red")
        return False

    try:
        from toolboxv2.utils.extras.registry_client import RegistryClient
    except ImportError as e:
        _p(f"  x registry_client nicht verfuegbar: {e}", style="red")
        return False

    try:
        from toolboxv2.feature_loader_registry import (
            _get_registry_url,
            _load_auth_token,
        )
    except ImportError as e:
        _p(f"  x feature_loader_registry nicht verfuegbar: {e}", style="red")
        return False

    token = _load_auth_token()
    if not token:
        _p("  x Kein Auth-Token -> tb registry login", style="red")
        return False

    async def _do_upload() -> bool:
        async with RegistryClient(registry_url=_get_registry_url()) as client:
            await client.login(token)
            if create_if_missing:
                existing = await client.get_artifact(artifact_name)
                if existing is None:
                    _p(f"  Artifact '{artifact_name}' anlegen ...")
                    await client.create_artifact(
                        name=artifact_name,
                        artifact_type=ARTIFACT_TYPE,
                        description="ToolBoxV2 standalone CLI binary (Nuitka onefile)",
                    )
            return await client.upload_artifact_build(
                name=artifact_name,
                version=version,
                platform=meta["platform"],
                architecture=meta["architecture"],
                file_path=binary,
                changelog=changelog,
            )

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_do_upload())
        # Bereits in einem Loop -> in eigenem Thread ausfuehren
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _do_upload()).result(timeout=600)
    except Exception as e:
        _p(f"  x Upload fehlgeschlagen: {e}", style="red")
        return False


# ── CLI Command ──────────────────────────────────────────────────────────────


def _base_version() -> str:
    """Version aus features/mini/feature.yaml lesen."""
    config = discover_features().get(BASE_FEATURE, {})
    return str(config.get("version", "0.0.0"))


def cmd_standalone(
    features: list[str] | None = None,
    target: str | None = None,
    all_targets: bool = False,
    upload: bool = False,
    artifact_name: str = DEFAULT_ARTIFACT_NAME,
    changelog: str = "",
    entry: Path | None = None,
) -> list[dict]:
    """Entry point für `tb build standalone`.

    Args:
        features: Feature-Namen. None = ["mini"].
        target: Explizites Target. None = aktuelles System.
        all_targets: Alle Targets versuchen (nur der lokale wird gebaut,
            Nuitka kann nicht cross-compilen).
        upload: Nach dem Build zur Registry hochladen.
        artifact_name: Registry-Artifact-Name.
        changelog: Changelog für den Upload.
        entry: Alternativer Entry-Point.

    Returns:
        Liste von Result-Dicts pro Target.
    """
    console.rule("STANDALONE")

    features = features or [BASE_FEATURE]
    version = _base_version()

    if all_targets:
        targets = list(TARGETS)
    else:
        targets = [target or detect_current_target()]

    results: list[dict] = []
    for tgt in targets:
        _p(f"\n  -- {tgt} --")
        binary = build_nuitka_onefile(
            features=features, target=tgt, entry=entry
        )
        entry_result: dict = {
            "target": tgt,
            "features": sorted(set(features) | {BASE_FEATURE}),
            "version": version,
            "binary": str(binary) if binary else None,
            "size_mb": round(binary.stat().st_size / (1024 * 1024), 1)
            if binary
            else 0,
            "uploaded": False,
        }
        if binary and upload:
            entry_result["uploaded"] = upload_standalone(
                binary=binary,
                version=version,
                target=tgt,
                artifact_name=artifact_name,
                changelog=changelog,
            )
            _p(
                f"  {'+' if entry_result['uploaded'] else 'x'} "
                f"Registry {artifact_name}@{version} [{tgt}]"
            )
        results.append(entry_result)

    ok = sum(1 for r in results if r["binary"])
    _p(f"\n  {ok}/{len(results)} Builds erfolgreich")
    return results


if __name__ == "__main__":  # pragma: no cover - manueller Aufruf
    cmd_standalone()
