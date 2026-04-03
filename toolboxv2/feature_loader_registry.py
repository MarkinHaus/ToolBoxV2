"""
feature_loader_registry.py
===========================
Registry-Integration für den Feature-Loader.

Drop-in Erweiterung zu feature_loader.py.
Stellt bereit:
  - download_feature_from_registry()  ← ersetzt den urllib-Stub
  - upload_feature_to_registry()      ← neu
  - check_for_updates()               ← neu
  - update_feature()                  ← neu

Naming-Konvention Registry ↔ Feature:
  Feature "web"  ←→ Registry-Package "tbv2-feature-web"
  Feature "isaa" ←→ Registry-Package "tbv2-feature-isaa"

Verwendung (in feature_loader.py):
    from .feature_loader_registry import (
        download_feature_from_registry,
        upload_feature_to_registry,
        check_for_updates,
        update_feature,
    )

    # Ersetze den urllib-Stub:
    zip_path = download_feature_from_registry(feature_name)
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Lazy imports – RegistryClient braucht httpx (nur im web-Feature verfügbar)
# Daher: alle Registry-Calls als async + mit try/except-Wrapping


# =============================================================================
# Hilfsfunktionen
# =============================================================================

REGISTRY_PACKAGE_PREFIX = "tbv2-feature-"


def feature_to_registry_name(feature_name: str) -> str:
    """Konvertiere Feature-Namen in Registry-Package-Namen."""
    return f"{REGISTRY_PACKAGE_PREFIX}{feature_name}"


def registry_name_to_feature(registry_name: str) -> Optional[str]:
    """Konvertiere Registry-Package-Namen zurück in Feature-Namen."""
    if registry_name.startswith(REGISTRY_PACKAGE_PREFIX):
        return registry_name[len(REGISTRY_PACKAGE_PREFIX):]
    return None


def _get_registry_url() -> str:
    """Registry-URL aus Manifest oder Umgebungsvariable."""
    # 1. Umgebungsvariable
    if url := os.environ.get("TB_REGISTRY_URL"):
        return url

    # 2. Manifest
    try:
        from toolboxv2.utils.manifest.loader import ManifestLoader
        from toolboxv2 import tb_root_dir
        loader = ManifestLoader(tb_root_dir)
        if loader.exists():
            manifest = loader.load(resolve_env=True)
            return manifest.registry.url
    except Exception:
        pass

    return "https://registry.simplecore.app"


def _get_cache_dir() -> Path:
    """Cache-Verzeichnis aus Manifest oder Default."""
    try:
        from toolboxv2.utils.manifest.loader import ManifestLoader
        from toolboxv2 import tb_root_dir
        loader = ManifestLoader(tb_root_dir)
        if loader.exists():
            manifest = loader.load(resolve_env=True)
            return Path(manifest.paths.registry_cache_dir)
    except Exception:
        pass
    return Path.home() / ".tb-registry" / "cache"


def _load_auth_token() -> Optional[str]:
    """Auth-Token aus gespeicherter Datei oder CloudM LogIn."""
    # 1. Datei-basierter Token
    token_file = Path.home() / ".tb-registry" / "auth_token.txt"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token

    # 2. CloudM LoginSystem
    try:
        from toolboxv2.mods.CloudM.LogInSystem import _load_cli_token
        return _load_cli_token()
    except Exception:
        pass

    return None


def _get_installed_version(feature_name: str) -> Optional[str]:
    """Ermittle installierte Version eines Features."""
    try:
        import yaml
        from toolboxv2.feature_loader import get_features_dir
        feature_yaml = get_features_dir() / feature_name / "feature.yaml"
        if feature_yaml.exists():
            with open(feature_yaml, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("version")
    except Exception:
        pass
    return None


def _get_features_packed_dir() -> Path:
    """Verzeichnis für gepackte Feature-ZIPs."""
    try:
        from toolboxv2.feature_loader import get_features_packed_dir
        return get_features_packed_dir()
    except Exception:
        return Path(__file__).parent / "features_packed"


# =============================================================================
# DOWNLOAD
# =============================================================================

async def _download_feature_async(
    feature_name: str,
    version: Optional[str] = None,
    force: bool = False,
) -> Optional[Path]:
    """
    Lade Feature-ZIP aus dem Registry herunter.

    Verwendet RegistryClient mit SHA256-Verifikation und Cache.

    Args:
        feature_name: Feature-Name (z.B. "web")
        version: Gewünschte Version (None = latest)
        force: Auch wenn bereits gecacht

    Returns:
        Pfad zur ZIP-Datei oder None bei Fehler
    """
    try:
        from toolboxv2.utils.extras.registry_client import (
            RegistryClient,
            RegistryError,
            PackageNotFoundError,
            VersionNotFoundError,
        )
    except ImportError:
        print(
            f"Warning: httpx not available, falling back to urllib for feature '{feature_name}'",
            file=sys.stderr,
        )
        return _download_feature_urllib_fallback(feature_name)

    registry_name = feature_to_registry_name(feature_name)
    registry_url = _get_registry_url()
    dest_dir = _get_features_packed_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    async with RegistryClient(
        registry_url=registry_url,
        cache_dir=_get_cache_dir(),
    ) as client:
        try:
            # Version auflösen
            if version is None:
                version = await client.get_latest_version(registry_name)
                if not version:
                    print(
                        f"Warning: Feature '{feature_name}' not found in registry ({registry_url})",
                        file=sys.stderr,
                    )
                    return None

            # Ziel-Dateiname prüfen (Cache-Hit)
            zip_name = f"tbv2-feature-{feature_name}-{version}.zip"
            dest_path = dest_dir / zip_name

            if dest_path.exists() and not force:
                print(
                    f"[feature-loader] Using cached: {zip_name}",
                    file=sys.stderr,
                )
                return dest_path

            # Download mit Checksum-Verifikation
            print(
                f"[feature-loader] Downloading '{feature_name}' v{version} from registry...",
                file=sys.stderr,
            )
            downloaded = await client.download(registry_name, version, dest_dir)

            # Datei umbenennen auf Standard-Format
            if downloaded.name != zip_name:
                downloaded.rename(dest_path)

            print(
                f"[feature-loader] Downloaded: {dest_path.name} "
                f"({dest_path.stat().st_size // 1024} KB)",
                file=sys.stderr,
            )
            return dest_path

        except PackageNotFoundError:
            print(
                f"Warning: Feature '{feature_name}' ('{registry_name}') not found in registry",
                file=sys.stderr,
            )
            return None
        except VersionNotFoundError:
            print(
                f"Warning: Version '{version}' not found for feature '{feature_name}'",
                file=sys.stderr,
            )
            return None
        except RegistryError as e:
            print(
                f"Warning: Registry error for feature '{feature_name}': {e}",
                file=sys.stderr,
            )
            return None
        except Exception as e:
            print(
                f"Warning: Unexpected error downloading feature '{feature_name}': {e}",
                file=sys.stderr,
            )
            return None


def _download_feature_urllib_fallback(feature_name: str) -> Optional[Path]:
    """Fallback ohne httpx – rudimentäres urllib.request."""
    import urllib.request
    import json

    registry_url = _get_registry_url().rstrip("/")
    registry_name = feature_to_registry_name(feature_name)
    dest_dir = _get_features_packed_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Versuche Metadaten-Endpoint
    zip_path = dest_dir / f"tbv2-feature-{feature_name}-latest.zip"
    try:
        meta_url = f"{registry_url}/api/v1/features/{registry_name}/latest"
        with urllib.request.urlopen(meta_url, timeout=8) as r:
            meta = json.loads(r.read())
            download_url = meta.get(
                "download_url",
                f"{registry_url}/features/{registry_name}-latest.zip",
            )
    except Exception:
        download_url = f"{registry_url}/features/{registry_name}-latest.zip"

    try:
        print(
            f"[feature-loader] Downloading '{feature_name}' via urllib (fallback)...",
            file=sys.stderr,
        )
        urllib.request.urlretrieve(download_url, zip_path)
        return zip_path
    except Exception as e:
        print(
            f"Warning: urllib download failed for '{feature_name}': {e}",
            file=sys.stderr,
        )
        if zip_path.exists():
            zip_path.unlink()
        return None


def download_feature_from_registry(
    feature_name: str,
    version: Optional[str] = None,
    force: bool = False,
) -> Optional[Path]:
    """
    Synchroner Wrapper für _download_feature_async.

    Ersetzt den urllib-Stub in feature_loader.py.

    Args:
        feature_name: Feature-Name (z.B. "web")
        version: Gewünschte Version (None = latest)
        force: Auch wenn bereits gecacht, neu herunterladen

    Returns:
        Pfad zur heruntergeladenen ZIP-Datei oder None

    Beispiel:
        # In feature_loader.unpack_feature():
        zip_path = get_packed_feature_path(feature_name)
        if not zip_path:
            zip_path = download_feature_from_registry(feature_name)
        if not zip_path:
            return False
    """
    try:
        # Bestehenden Event-Loop nutzen oder neuen erstellen
        try:
            loop = asyncio.get_running_loop()
            # Wir sind bereits in einem async-Kontext (z.B. App-Startup)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    _download_feature_async(feature_name, version, force),
                )
                return future.result(timeout=60)
        except RuntimeError:
            # Kein laufender Event-Loop
            return asyncio.run(_download_feature_async(feature_name, version, force))
    except Exception as e:
        print(
            f"Warning: Download failed for feature '{feature_name}': {e}",
            file=sys.stderr,
        )
        return _download_feature_urllib_fallback(feature_name)


# =============================================================================
# UPLOAD
# =============================================================================

async def _upload_feature_async(
    feature_name: str,
    zip_path: Path,
    version: str,
    changelog: Optional[str] = None,
    create_if_missing: bool = True,
    description: str = "",
    visibility: str = "public",
) -> bool:
    """
    Lade Feature-ZIP in den Registry hoch.

    Benötigt: auth_token mit is_verified=True.

    Args:
        feature_name: Feature-Name
        zip_path: Pfad zur Feature-ZIP-Datei
        version: Version des Features
        changelog: Optionaler Changelog-Text
        create_if_missing: Package anlegen wenn nicht existiert
        description: Beschreibung (für create)
        visibility: public / private / unlisted

    Returns:
        True bei Erfolg
    """
    try:
        from toolboxv2.utils.extras.registry_client import (
            RegistryClient,
            RegistryError,
            RegistryAuthError,
            PublishPermissionError,
        )
    except ImportError:
        print("Error: httpx not installed. Cannot upload features.", file=sys.stderr)
        return False

    token = _load_auth_token()
    if not token:
        print(
            "Error: Not authenticated. Run 'tb registry login' first.",
            file=sys.stderr,
        )
        return False

    registry_name = feature_to_registry_name(feature_name)

    async with RegistryClient(
        registry_url=_get_registry_url(),
        auth_token=token,
        cache_dir=_get_cache_dir(),
    ) as client:
        try:
            # Auth prüfen
            user = await client.get_current_user()
            if not user:
                print("Error: Invalid auth token.", file=sys.stderr)
                return False
            if not user.is_verified:
                print(
                    f"Error: Publisher not verified. Contact registry admin.",
                    file=sys.stderr,
                )
                return False

            # Prüfen ob Package existiert
            existing = await client.get_package(registry_name)

            if existing is None and create_if_missing:
                print(
                    f"[feature-upload] Creating new registry package '{registry_name}'...",
                    file=sys.stderr,
                )

                # Feature-YAML für Metadaten lesen
                try:
                    import yaml
                    from toolboxv2.feature_loader import get_features_dir
                    feature_yaml_path = get_features_dir() / feature_name / "feature.yaml"
                    with open(feature_yaml_path, encoding="utf-8") as f:
                        feature_data = yaml.safe_load(f) or {}
                    description = description or feature_data.get("description", "")
                except Exception:
                    pass

                await client.create_package(
                    name=registry_name,
                    display_name=f"TB Feature: {feature_name}",
                    package_type="library",
                    visibility=visibility,
                    description=description or f"ToolBoxV2 feature: {feature_name}",
                )
                print(
                    f"[feature-upload] Package '{registry_name}' created.",
                    file=sys.stderr,
                )

            elif existing is None:
                print(
                    f"Error: Package '{registry_name}' not found and create_if_missing=False",
                    file=sys.stderr,
                )
                return False

            # Version hochladen
            print(
                f"[feature-upload] Uploading '{feature_name}' v{version}...",
                file=sys.stderr,
            )
            success = await client.upload_version(
                name=registry_name,
                version=version,
                file_path=zip_path,
                changelog=changelog,
            )

            if success:
                print(
                    f"[feature-upload] Successfully uploaded '{feature_name}' v{version} "
                    f"to {_get_registry_url()}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Error: Upload failed for '{feature_name}' v{version}",
                    file=sys.stderr,
                )

            return success

        except RegistryAuthError as e:
            print(f"Auth error: {e}", file=sys.stderr)
            return False
        except PublishPermissionError as e:
            print(f"Permission error: {e}", file=sys.stderr)
            return False
        except RegistryError as e:
            print(f"Registry error: {e}", file=sys.stderr)
            return False


def upload_feature_to_registry(
    feature_name: str,
    version: Optional[str] = None,
    changelog: Optional[str] = None,
    zip_path: Optional[Path] = None,
    create_if_missing: bool = True,
    visibility: str = "public",
) -> bool:
    """
    Packe Feature und lade es in den Registry hoch.

    Führt automatisch pack_feature() aus wenn kein zip_path angegeben.

    Args:
        feature_name: Feature-Name (z.B. "web")
        version: Version-Override (None = aus feature.yaml)
        changelog: Optionaler Changelog-Text
        zip_path: Bereits gepacktes ZIP (None = wird erstellt)
        create_if_missing: Package anlegen wenn nicht existiert
        visibility: public / private / unlisted

    Returns:
        True bei Erfolg

    Beispiel:
        # Feature packen und hochladen:
        upload_feature_to_registry("web", changelog="Added websocket auth")

        # Bereits vorhandenes ZIP hochladen:
        upload_feature_to_registry("web", zip_path=Path("my-web.zip"))

    CLI-Verwendung (nach Integration in feature_loader.main()):
        tb fl upload web
        tb fl upload web --changelog "Fixed bug in workers"
    """
    # ZIP erstellen wenn nicht vorhanden
    if zip_path is None:
        try:
            from toolboxv2.utils.system.feature_manager import FeatureManager
            from toolboxv2.feature_loader import get_features_dir
            features_dir = get_features_dir()

            # FeatureManager Singleton nutzen (falls vorhanden) oder temp erstellen
            try:
                fm = FeatureManager.__instance__
                if fm is None:
                    raise AttributeError
            except AttributeError:
                fm = FeatureManager(features_dir=str(features_dir))

            if feature_name not in fm.features:
                print(f"Error: Feature '{feature_name}' not found.", file=sys.stderr)
                return False

            # Version aus feature.yaml
            if version is None:
                version = fm.features[feature_name].version

            # Packen
            output_dir = features_dir.parent / "features_sto"
            print(
                f"[feature-upload] Packing '{feature_name}' v{version}...",
                file=sys.stderr,
            )
            zip_path_str = fm.pack_feature(feature_name, output_path=str(output_dir))
            if not zip_path_str:
                print(f"Error: Failed to pack feature '{feature_name}'", file=sys.stderr)
                return False
            zip_path = Path(zip_path_str)

        except Exception as e:
            print(f"Error packing feature '{feature_name}': {e}", file=sys.stderr)
            return False

    # Version bestimmen
    if version is None:
        version = _get_installed_version(feature_name) or "0.0.0"

    # Upload
    try:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    _upload_feature_async(
                        feature_name, zip_path, version,
                        changelog, create_if_missing, "", visibility,
                    ),
                )
                return future.result(timeout=120)
        except RuntimeError:
            return asyncio.run(
                _upload_feature_async(
                    feature_name, zip_path, version,
                    changelog, create_if_missing, "", visibility,
                )
            )
    except Exception as e:
        print(f"Error uploading feature '{feature_name}': {e}", file=sys.stderr)
        return False


# =============================================================================
# CHECK FOR UPDATES
# =============================================================================

async def _check_updates_async(
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Tuple[str, str]]:
    """
    Prüfe ob Updates verfügbar sind.

    Returns:
        Dict von feature_name → (installed_version, latest_version)
        Nur Features mit verfügbaren Updates.
    """
    try:
        from toolboxv2.utils.extras.registry_client import RegistryClient, RegistryError
    except ImportError:
        return {}

    updates: Dict[str, Tuple[str, str]] = {}

    # Installierte Features bestimmen
    if feature_names is None:
        try:
            from toolboxv2.feature_loader import list_available_features, is_feature_installed
            feature_names = [f for f in list_available_features() if is_feature_installed(f)]
        except Exception:
            return {}

    async with RegistryClient(
        registry_url=_get_registry_url(),
        cache_dir=_get_cache_dir(),
    ) as client:
        for feature_name in feature_names:
            installed_version = _get_installed_version(feature_name)
            if not installed_version:
                continue

            registry_name = feature_to_registry_name(feature_name)
            try:
                latest_version = await client.get_latest_version(registry_name)
                if latest_version and latest_version != installed_version:
                    # Einfacher String-Vergleich – für semver packaging.version nutzen
                    try:
                        from packaging.version import Version
                        if Version(latest_version) > Version(installed_version):
                            updates[feature_name] = (installed_version, latest_version)
                    except Exception:
                        # Fallback: rein lexikografisch
                        if latest_version > installed_version:
                            updates[feature_name] = (installed_version, latest_version)
            except RegistryError:
                continue
            except Exception:
                continue

    return updates


def check_for_updates(
    feature_names: Optional[List[str]] = None,
    print_results: bool = False,
) -> Dict[str, Tuple[str, str]]:
    """
    Prüfe ob Feature-Updates im Registry verfügbar sind.

    Args:
        feature_names: Features prüfen (None = alle installierten)
        print_results: Ergebnis auf stdout ausgeben

    Returns:
        Dict von feature_name → (installed_version, latest_version)

    Beispiel:
        updates = check_for_updates()
        for name, (installed, latest) in updates.items():
            print(f"  {name}: {installed} → {latest}")

    CLI-Verwendung:
        tb fl check-updates
    """
    try:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    _check_updates_async(feature_names),
                )
                updates = future.result(timeout=30)
        except RuntimeError:
            updates = asyncio.run(_check_updates_async(feature_names))
    except Exception as e:
        print(f"Warning: Update check failed: {e}", file=sys.stderr)
        return {}

    if print_results:
        if updates:
            print("\n=== Available Feature Updates ===\n")
            for name, (installed, latest) in updates.items():
                print(f"  {name:<12} {installed} → {latest}")
            print()
        else:
            print("All features are up to date.")

    return updates


# =============================================================================
# UPDATE
# =============================================================================

async def _update_feature_async(
    feature_name: str,
    target_version: Optional[str] = None,
    use_diff: bool = True,
) -> bool:
    """
    Update ein Feature auf neue Version.

    Nutzt diff-Download wenn verfügbar und kleiner als 50% des Full-Downloads.
    """
    try:
        from toolboxv2.utils.extras.registry_client import RegistryClient, RegistryError
        from toolboxv2.feature_loader import unpack_feature
    except ImportError:
        return False

    installed_version = _get_installed_version(feature_name)
    if not installed_version:
        print(f"Error: Feature '{feature_name}' is not installed.", file=sys.stderr)
        return False

    registry_name = feature_to_registry_name(feature_name)
    dest_dir = _get_features_packed_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    async with RegistryClient(
        registry_url=_get_registry_url(),
        cache_dir=_get_cache_dir(),
    ) as client:
        try:
            # Zielversion bestimmen
            if target_version is None:
                target_version = await client.get_latest_version(registry_name)
                if not target_version:
                    print(
                        f"Warning: Feature '{feature_name}' not found in registry.",
                        file=sys.stderr,
                    )
                    return False

            if target_version == installed_version:
                print(
                    f"Feature '{feature_name}' is already at v{installed_version}.",
                    file=sys.stderr,
                )
                return True

            print(
                f"[feature-update] Updating '{feature_name}': "
                f"v{installed_version} → v{target_version}",
                file=sys.stderr,
            )

            # Download (mit Diff wenn verfügbar)
            if use_diff:
                zip_path = await client.download_with_diff(
                    name=registry_name,
                    from_version=installed_version,
                    to_version=target_version,
                    dest_dir=dest_dir,
                    max_diff_size_ratio=0.5,
                )
            else:
                zip_path = await client.download(registry_name, target_version, dest_dir)

            if not zip_path or not zip_path.exists():
                print(
                    f"Error: Download failed for '{feature_name}' v{target_version}",
                    file=sys.stderr,
                )
                return False

            # Entpacken (force=True für Update)
            print(
                f"[feature-update] Unpacking '{feature_name}' v{target_version}...",
                file=sys.stderr,
            )
            success = unpack_feature(feature_name, force=True)

            if success:
                print(
                    f"[feature-update] Successfully updated '{feature_name}' "
                    f"to v{target_version}",
                    file=sys.stderr,
                )
                # Feature-Manager-Cache invalidieren
                try:
                    from toolboxv2.utils.system.feature_manager import FeatureManager
                    if hasattr(FeatureManager, "__instance__") and FeatureManager.__instance__:
                        FeatureManager.__instance__._load_single_feature(feature_name)
                except Exception:
                    pass
            else:
                print(
                    f"Error: Failed to unpack update for '{feature_name}'",
                    file=sys.stderr,
                )

            return success

        except RegistryError as e:
            print(f"Registry error updating '{feature_name}': {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Error updating '{feature_name}': {e}", file=sys.stderr)
            return False


def update_feature(
    feature_name: str,
    target_version: Optional[str] = None,
    use_diff: bool = True,
) -> bool:
    """
    Update ein Feature auf die neueste (oder angegebene) Version.

    Nutzt diff-Download wenn verfügbar (< 50% des Full-Download-Größe).

    Args:
        feature_name: Feature-Name (z.B. "web")
        target_version: Zielversion (None = latest)
        use_diff: Diff-Download versuchen (Standard: True)

    Returns:
        True bei Erfolg

    Beispiel:
        update_feature("web")              # Update auf latest
        update_feature("web", "0.2.0")    # Update auf spezifische Version

    CLI-Verwendung:
        tb fl update web
        tb fl update web --version 0.2.0
        tb fl update --all
    """
    try:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    _update_feature_async(feature_name, target_version, use_diff),
                )
                return future.result(timeout=120)
        except RuntimeError:
            return asyncio.run(
                _update_feature_async(feature_name, target_version, use_diff)
            )
    except Exception as e:
        print(f"Error updating feature '{feature_name}': {e}", file=sys.stderr)
        return False


def update_all_features(use_diff: bool = True) -> Dict[str, bool]:
    """
    Update alle installierten Features auf die neueste Version.

    Args:
        use_diff: Diff-Download versuchen

    Returns:
        Dict von feature_name → success
    """
    # Updates prüfen
    updates = check_for_updates()

    if not updates:
        print("All features are up to date.")
        return {}

    results: Dict[str, bool] = {}
    for feature_name, (installed, latest) in updates.items():
        results[feature_name] = update_feature(feature_name, latest, use_diff)

    return results


# =============================================================================
# CLI-Integration (für feature_loader.main())
# =============================================================================

def extend_feature_loader_cli(subparsers) -> None:
    """
    Füge Registry-Subcommands zur feature_loader CLI hinzu.

    Aufruf in feature_loader.main():
        from .feature_loader_registry import extend_feature_loader_cli
        extend_feature_loader_cli(subparsers)
    """
    # upload
    p_upload = subparsers.add_parser("upload", help="Upload feature to registry")
    p_upload.add_argument("feature", help="Feature name")
    p_upload.add_argument("--version", help="Version override")
    p_upload.add_argument("--changelog", help="Changelog text")
    p_upload.add_argument("--zip", help="Path to pre-built ZIP")
    p_upload.add_argument(
        "--private",
        action="store_true",
        help="Upload as private package",
    )

    # check-updates
    subparsers.add_parser("check-updates", help="Check for available feature updates")

    # update
    p_update = subparsers.add_parser("update", help="Update feature(s)")
    p_update.add_argument(
        "feature",
        nargs="?",
        help="Feature name (omit for --all)",
    )
    p_update.add_argument("--version", help="Target version")
    p_update.add_argument("--all", action="store_true", help="Update all features")
    p_update.add_argument(
        "--no-diff",
        action="store_true",
        help="Always do full download (no diff)",
    )


def handle_registry_cli_command(args) -> int:
    """
    Verarbeite Registry-Subcommands in feature_loader.main().

    Rückgabe: Exit-Code (0 = OK, 1 = Fehler)

    Beispiel in feature_loader.main():
        from .feature_loader_registry import (
            extend_feature_loader_cli,
            handle_registry_cli_command,
        )
        args = parser.parse_args()
        if args.command in ("upload", "check-updates", "update"):
            sys.exit(handle_registry_cli_command(args))
    """
    if args.command == "upload":
        zip_path = Path(args.zip) if args.zip else None
        visibility = "private" if args.private else "public"
        success = upload_feature_to_registry(
            feature_name=args.feature,
            version=args.version,
            changelog=args.changelog,
            zip_path=zip_path,
            visibility=visibility,
        )
        return 0 if success else 1

    elif args.command == "check-updates":
        check_for_updates(print_results=True)
        return 0

    elif args.command == "update":
        use_diff = not getattr(args, "no_diff", False)

        if getattr(args, "all", False):
            results = update_all_features(use_diff=use_diff)
            failed = [n for n, ok in results.items() if not ok]
            if failed:
                print(f"\nFailed updates: {', '.join(failed)}", file=sys.stderr)
                return 1
            return 0

        if not args.feature:
            print("Error: Specify a feature name or --all", file=sys.stderr)
            return 1

        success = update_feature(
            feature_name=args.feature,
            target_version=args.version,
            use_diff=use_diff,
        )
        return 0 if success else 1

    return 1
