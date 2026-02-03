"""
Feature Manager - Singleton for Feature Management

Verwaltet Feature-Flags, Lazy Loading, Pack/Unpack und State Sync.
"""
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Any
import subprocess
import sys

from toolboxv2.utils.manifest.schema import FeatureSpec
from toolboxv2.utils.singelton_class import Singleton

if TYPE_CHECKING:
    from toolboxv2.utils.toolbox import App
    from toolboxv2.utils.system.state_system import TbState, FeatureStateElement


class FeatureManager(metaclass=Singleton):
    """
    Singleton Feature Manager

    Wird in App initialisiert und verwaltet alle Features.
    Erbt von Singleton metaclass wie App.
    """

    def __init__(self, app: Optional["App"] = None, features_dir: Optional[str] = None):
        self.app = app
        self.features_dir = Path(features_dir or "features") if features_dir and isinstance(features_dir, str) else features_dir or Path("features")
        self.features: Dict[str, FeatureSpec] = {}
        self.loaded_features: Set[str] = set()
        self._feature_files_cache: Dict[str, List[str]] = {}
        self._initialized = False

        # Lade alle Feature Metadaten
        self._load_feature_metadata()
        self._initialized = True

    def _load_feature_metadata(self):
        """Lade alle features/*/feature.yaml Dateien"""
        # Versuche zuerst aus YAML-Dateien zu laden
        if self.features_dir.exists():
            self._load_from_yaml_files()

        # Fallback: Lade aus tb-manifest.yaml wenn keine YAML-Dateien gefunden
        if not self.features:
            self._load_from_manifest()

    def _load_from_yaml_files(self):
        """Lade Features aus features/*/feature.yaml"""
        try:
            import yaml
        except ImportError:
            if self.app:
                self.app.logger.warning("PyYAML not installed, skipping YAML feature loading")
            return

        for feature_yaml in self.features_dir.glob("*/feature.yaml"):

            feature_name = feature_yaml.parent.name

            try:
                with open(feature_yaml, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data:
                    self.features[feature_name] = FeatureSpec(
                        **data
                    )

                    # Cache der Dateien für schnellen Lookup
                    self._feature_files_cache[feature_name] = data.get("files", [])

                    if self.app:
                        self.app.logger.debug(f"Loaded feature metadata: {feature_name}")
            except Exception as e:
                print(f"Failed to load feature {feature_name}: {e}")
                import traceback
                traceback.print_exc()
                if self.app:
                    self.app.logger.error(f"Failed to load feature {feature_name}: {e}")

    def _load_from_manifest(self):
        """Lade Features aus tb-manifest.yaml"""
        try:
            import yaml
            from pathlib import Path

            # Suche tb-manifest.yaml im Root-Verzeichnis
            manifest_paths = [
                Path("tb-manifest.yaml"),
                Path.cwd() / "tb-manifest.yaml",
            ]

            if self.app:
                manifest_paths.insert(0, Path(self.app.start_dir) / "tb-manifest.yaml")

            manifest_path = None
            for p in manifest_paths:
                if p.exists():
                    manifest_path = p
                    break

            if not manifest_path:
                if self.app:
                    self.app.logger.debug("No tb-manifest.yaml found for feature loading")
                return

            with open(manifest_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            features_data = data.get("features", {})
            for feature_name, feature_config in features_data.items():
                if isinstance(feature_config, dict):
                    self.features[feature_name] = FeatureSpec(
                        name=feature_name,
                        **feature_config
                    )
                    self._feature_files_cache[feature_name] = feature_config.get("files", [])

                    if self.app:
                        self.app.logger.debug(f"Loaded feature from manifest: {feature_name}")

        except Exception as e:
            if self.app:
                self.app.logger.error(f"Failed to load features from manifest: {e}")

    def enabled(self, feature_name: str) -> bool:
        """
        Prüfe ob Feature aktiviert ist (einfache Methode)

        Verwendung:
            if feature_manager.enabled("cli"):
                # Code wird nur ausgeführt wenn CLI aktiviert
                import toolboxv2.__main__
        """
        return self.is_enabled(feature_name)

    def is_enabled(self, feature_name: str) -> bool:
        """Prüfe ob Feature aktiviert ist"""
        feature = self.features.get(feature_name)
        if not feature:
            return False
        return feature.enabled

    def enable(self, feature_name: str) -> bool:
        """
        Aktiviere Feature mit auto-Installation der Dependencies
        """
        feature = self.features.get(feature_name)
        if not feature:
            if self.app:
                self.app.logger.error(f"Feature not found: {feature_name}")
            return False

        if feature.immutable:
            if self.app:
                self.app.logger.warning(
                    f"Warning: Feature '{feature_name}' is marked as immutable. "
                    f"Enabling it may have unexpected effects."
                )
            # Warnung aber erlaubt

        # Prüfe Dependencies
        if feature.requires:
            for req in feature.requires:
                if not self.is_enabled(req):
                    if self.app:
                        self.app.logger.error(
                            f"Cannot enable {feature_name}: required feature {req} is not enabled"
                        )
                    return False

        # Installiere Dependencies mit pip oder uv
        if feature.dependencies:
            if not self._install_dependencies(feature.dependencies):
                if self.app:
                    self.app.logger.error(f"Failed to install dependencies for {feature_name}")
                return False

        feature.enabled = True
        if self.app:
            self.app.logger.info(f"Feature enabled: {feature_name}")
        return True

    def _install_dependencies(self, dependencies: List[str]) -> bool:
        """
        Installiere Dependencies mit pip oder uv (automatische Erkennung)
        Verwendet immer sys.executable
        """
        # Prüfe ob uv verfügbar ist
        try:
            result = subprocess.run(
                [sys.executable, "-m", "uv", "--version"],
                capture_output=True,
                text=True
            )
            use_uv = result.returncode == 0
        except Exception:
            use_uv = False

        # Erstelle Kommando
        if use_uv:
            cmd = [sys.executable, "-m", "uv", "pip", "install"]
        else:
            cmd = [sys.executable, "-m", "pip", "install"]

        cmd.extend(dependencies)

        if self.app:
            self.app.logger.info(f"Installing dependencies: {' '.join(dependencies)}")
            self.app.logger.info(f"Using: {'uv' if use_uv else 'pip'}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            if self.app:
                self.app.logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            if self.app:
                self.app.logger.error(f"Failed to install dependencies: {e}")
                if e.stderr:
                    self.app.logger.error(f"Error output: {e.stderr}")
            return False

    def disable(self, feature_name: str) -> bool:
        """Deaktiviere Feature"""
        feature = self.features.get(feature_name)
        if not feature:
            return False

        if feature_name == "core":
            if self.app:
                self.app.logger.warning(
                    "Warning: Disabling 'core' feature is not recommended. "
                    "This may break essential functionality."
                )
            # Warnung aber erlaubt (wie gewünscht)

        if feature_name in self.loaded_features:
            if self.app:
                self.app.logger.warning(
                    f"Cannot disable loaded feature: {feature_name} (already imported)"
                )
            return False

        feature.enabled = False
        if self.app:
            self.app.logger.info(f"Feature disabled: {feature_name}")
        return True

    def get_files_for_feature(self, feature_name: str) -> List[str]:
        """Gebe alle Dateien zurück die zum Feature gehören"""
        return self._feature_files_cache.get(feature_name, [])

    def get_features_for_file(self, file_path: str) -> List[str]:
        """
        Gebe alle Features zurück die eine Datei verwenden

        Eine Datei kann zu mehreren Features gehören!
        """
        matching_features = []

        for feature_name, files in self._feature_files_cache.items():
            for file_pattern in files:
                # Wildcard matching
                if file_pattern.endswith("/*"):
                    # Ordner-Matching
                    dir_pattern = file_pattern[:-2]
                    if file_path.startswith(dir_pattern):
                        matching_features.append(feature_name)
                        break
                else:
                    # Exaktes Matching
                    if file_path == file_pattern:
                        matching_features.append(feature_name)
                        break

        return matching_features

    def should_import(self, file_path: str) -> bool:
        """
        Prüfe ob Datei importiert werden sollte

        Wird in __main__.py verwendet um Imports zu steuern
        """
        features_for_file = self.get_features_for_file(file_path)

        # Wenn kein Feature zugeordnet, immer importieren (Core/Shared)
        if not features_for_file:
            return True

        # Wenn mindestens ein Feature aktiviert ist, importieren
        return any(self.is_enabled(f) for f in features_for_file)

    def list_features(self) -> Dict[str, dict]:
        """Liste alle Features auf"""
        return {
            name: {
                "enabled": f.enabled,
                "loaded": name in self.loaded_features,
                "version": f.version,
                "immutable": f.immutable,
                "description": f.description
            }
            for name, f in self.features.items()
        }

    def mark_as_loaded(self, feature_name: str):
        """Markiere Feature als geladen"""
        self.loaded_features.add(feature_name)

    def save_to_manifest(self, feature_name: str):
        """Speichere Feature-Status in tb-manifest.yaml"""
        try:
            import yaml
            from pathlib import Path

            manifest_paths = [
                Path("tb-manifest.yaml"),
                Path.cwd() / "tb-manifest.yaml",
            ]

            if self.app:
                manifest_paths.insert(0, Path(self.app.start_dir) / "tb-manifest.yaml")

            manifest_path = None
            for p in manifest_paths:
                if p.exists():
                    manifest_path = p
                    break

            if not manifest_path:
                return False

            with open(manifest_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            if "features" not in data:
                data["features"] = {}

            if feature_name not in data["features"]:
                data["features"][feature_name] = {}

            feature = self.features.get(feature_name)
            if feature:
                data["features"][feature_name]["enabled"] = feature.enabled

            with open(manifest_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

            return True
        except Exception as e:
            if self.app:
                self.app.logger.error(f"Failed to save feature to manifest: {e}")
            return False

    # =================== State Sync ===================

    def sync_with_state(self, state: "TbState") -> None:
        """
        Synchronisiere Feature Manager mit State System.

        Wird aufgerufen wenn State neu gebaut wird.
        Übernimmt enabled-Status aus dem State.

        Args:
            state: TbState Objekt mit features Dict
        """
        if not state.features:
            return

        for feature_name, feature_state in state.features.items():
            if feature_name in self.features:
                # Update local feature mit State Info
                self.features[feature_name].enabled = feature_state.enabled
                if self.app:
                    self.app.logger.debug(
                        f"Synced feature '{feature_name}' from state: enabled={feature_state.enabled}"
                    )
            else:
                # Feature aus State übernehmen (z.B. nach State-Restore)
                self.features[feature_name] = FeatureSpec(
                    name=feature_name,
                    version=feature_state.version,
                    enabled=feature_state.enabled,
                    dependencies=feature_state.dependencies,
                    requires=feature_state.requires,
                )
                if self.app:
                    self.app.logger.info(f"Restored feature '{feature_name}' from state")

    def export_to_state(self) -> Dict[str, dict]:
        """
        Exportiere Feature Status für State System.

        Returns:
            Dict mit Feature State Daten für Serialisierung
        """
        return {
            name: {
                "name": f.name,
                "version": f.version,
                "enabled": f.enabled,
                "dependencies": f.dependencies,
                "requires": f.requires,
                "source": "local",
            }
            for name, f in self.features.items()
        }

    # =================== Pack/Unpack System ===================

    def pack_feature(self, feature_name: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Packe ein Feature in eine ZIP-Datei.

        Format: tbv2-feature-{name}-{version}.zip
        Inhalt:
            - feature.yaml
            - requirements.txt (falls vorhanden)
            - Alle files aus feature.yaml

        Args:
            feature_name: Name des Features
            output_path: Optionaler Ausgabepfad (Default: ./features_sto/)

        Returns:
            Pfad zur erstellten ZIP-Datei oder None bei Fehler
        """
        import yaml

        feature = self.features.get(feature_name)
        if not feature:
            if self.app:
                self.app.logger.error(f"Feature not found: {feature_name}")
            return None

        feature_dir = self.features_dir / feature_name
        if not feature_dir.exists():
            if self.app:
                self.app.logger.error(f"Feature directory not found: {feature_dir}")
            return None

        # Output Verzeichnis
        if output_path:
            out_dir = Path(output_path)
        else:
            out_dir = Path("features_sto")
        out_dir.mkdir(parents=True, exist_ok=True)

        # ZIP Dateiname
        timestamp = datetime.now().strftime("%Y%m%d")
        zip_name = f"tbv2-feature-{feature_name}-{feature.version}-{timestamp}.zip"
        zip_path = out_dir / zip_name

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 1. feature.yaml
                feature_yaml = feature_dir / "feature.yaml"
                if feature_yaml.exists():
                    zf.write(feature_yaml, "feature.yaml")
                else:
                    # Generiere feature.yaml aus FeatureSpec
                    feature_data = {
                        "version": feature.version,
                        "enabled": feature.enabled,
                        "description": feature.description,
                        "immutable": feature.immutable,
                        "dependencies": feature.dependencies,
                        "requires": feature.requires,
                        "files": self._feature_files_cache.get(feature_name, []),
                    }
                    zf.writestr("feature.yaml", yaml.dump(feature_data, allow_unicode=True))

                # 2. requirements.txt (falls Dependencies)
                if feature.dependencies:
                    req_content = "\n".join(feature.dependencies)
                    zf.writestr("requirements.txt", req_content)

                # 3. Alle Feature-spezifischen Dateien
                files = self._feature_files_cache.get(feature_name, [])
                tb_root = self._get_tb_root()

                for file_pattern in files:
                    self._add_files_to_zip(zf, tb_root, file_pattern, feature_name)

                # 4. Metadata
                metadata = {
                    "packed_at": datetime.now().isoformat(),
                    "feature_name": feature_name,
                    "version": feature.version,
                    "files_count": len(zf.namelist()),
                }
                zf.writestr("_metadata.yaml", yaml.dump(metadata, allow_unicode=True))

            if self.app:
                self.app.logger.info(f"Feature packed: {zip_path}")

            return str(zip_path)

        except Exception as e:
            if self.app:
                self.app.logger.error(f"Failed to pack feature: {e}")
            # Cleanup
            if zip_path.exists():
                zip_path.unlink()
            return None

    def unpack_feature(self, zip_path: str, target_dir: Optional[str] = None) -> Optional[str]:
        """
        Entpacke ein Feature aus einer ZIP-Datei.

        Args:
            zip_path: Pfad zur ZIP-Datei
            target_dir: Optionales Zielverzeichnis (Default: self.features_dir)

        Returns:
            Name des entpackten Features oder None bei Fehler
        """
        import yaml

        zip_path = Path(zip_path)
        if not zip_path.exists():
            if self.app:
                self.app.logger.error(f"ZIP file not found: {zip_path}")
            return None

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # 1. Lese feature.yaml um Namen zu ermitteln
                if "feature.yaml" not in zf.namelist():
                    if self.app:
                        self.app.logger.error("Invalid feature package: missing feature.yaml")
                    return None

                feature_yaml_content = zf.read("feature.yaml").decode("utf-8")
                feature_data = yaml.safe_load(feature_yaml_content)

                # Feature Name aus Metadaten oder ZIP-Dateiname
                if "_metadata.yaml" in zf.namelist():
                    metadata = yaml.safe_load(zf.read("_metadata.yaml").decode("utf-8"))
                    feature_name = metadata.get("feature_name")
                else:
                    # Extrahiere aus Dateiname: tbv2-feature-{name}-{version}.zip
                    stem = zip_path.stem
                    if stem.startswith("tbv2-feature-"):
                        parts = stem.replace("tbv2-feature-", "").split("-")
                        feature_name = parts[0] if parts else None
                    else:
                        feature_name = stem

                if not feature_name:
                    if self.app:
                        self.app.logger.error("Could not determine feature name")
                    return None

                # 2. Zielverzeichnis
                if target_dir:
                    features_target = Path(target_dir)
                else:
                    features_target = self.features_dir

                feature_target = features_target / feature_name

                # Backup falls existiert
                if feature_target.exists():
                    backup_path = feature_target.with_suffix(".backup")
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.move(str(feature_target), str(backup_path))
                    if self.app:
                        self.app.logger.info(f"Created backup: {backup_path}")

                feature_target.mkdir(parents=True, exist_ok=True)

                # 3. Entpacke feature.yaml und requirements.txt
                zf.extract("feature.yaml", feature_target)

                if "requirements.txt" in zf.namelist():
                    zf.extract("requirements.txt", feature_target)

                # 4. Entpacke Feature-Dateien (files/ Ordner)
                tb_root = self._get_tb_root()
                for name in zf.namelist():
                    if name.startswith("files/") and not name.endswith("/"):
                        # Relativer Pfad ohne "files/" Prefix
                        rel_path = name[6:]  # Remove "files/"
                        target_file = tb_root / rel_path
                        target_file.parent.mkdir(parents=True, exist_ok=True)

                        with zf.open(name) as src, open(target_file, "wb") as dst:
                            dst.write(src.read())

                # 5. Reload Feature Metadata
                self._load_single_feature(feature_name)

                if self.app:
                    self.app.logger.info(f"Feature unpacked: {feature_name} -> {feature_target}")

                return feature_name

        except Exception as e:
            if self.app:
                self.app.logger.error(f"Failed to unpack feature: {e}")
            return None

    def _get_tb_root(self) -> Path:
        """Ermittle ToolBoxV2 Root Verzeichnis"""
        if self.app:
            return Path(self.app.start_dir) / "toolboxv2"
        # Fallback: Relativ zu features_dir
        return self.features_dir.parent

    def _add_files_to_zip(
        self,
        zf: zipfile.ZipFile,
        tb_root: Path,
        file_pattern: str,
        feature_name: str
    ) -> None:
        """
        Füge Dateien basierend auf Pattern zum ZIP hinzu.

        Args:
            zf: ZipFile Objekt
            tb_root: ToolBoxV2 Root Pfad
            file_pattern: Datei-Pattern (z.B. "mods/CloudM/*" oder "utils/file.py")
            feature_name: Feature Name für Logging
        """
        if file_pattern.endswith("/*"):
            # Ordner-Pattern
            dir_pattern = file_pattern[:-2]
            source_dir = tb_root / dir_pattern

            if source_dir.exists() and source_dir.is_dir():
                for file_path in source_dir.rglob("*"):
                    if file_path.is_file() and "__pycache__" not in str(file_path):
                        rel_path = file_path.relative_to(tb_root)
                        arc_name = f"files/{rel_path}"
                        zf.write(file_path, arc_name)
        else:
            # Einzelne Datei
            source_file = tb_root / file_pattern
            if source_file.exists() and source_file.is_file():
                arc_name = f"files/{file_pattern}"
                zf.write(source_file, arc_name)

    def _load_single_feature(self, feature_name: str) -> bool:
        """
        Lade ein einzelnes Feature neu.

        Args:
            feature_name: Name des Features

        Returns:
            True wenn erfolgreich
        """
        import yaml

        feature_yaml = self.features_dir / feature_name / "feature.yaml"
        if not feature_yaml.exists():
            return False

        try:
            with open(feature_yaml, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data:
                self.features[feature_name] = FeatureSpec(
                    name=feature_name,
                    **data
                )
                self._feature_files_cache[feature_name] = data.get("files", [])
                return True
        except Exception as e:
            if self.app:
                self.app.logger.error(f"Failed to reload feature {feature_name}: {e}")

        return False

    def list_packed_features(self, search_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Liste alle gepackten Feature-Archive auf.

        Args:
            search_dir: Suchverzeichnis (Default: ./features_sto/)

        Returns:
            Liste mit Infos zu gefundenen Packages
        """
        import yaml

        if search_dir:
            search_path = Path(search_dir)
        else:
            search_path = Path("features_sto")

        if not search_path.exists():
            return []

        packages = []
        for zip_file in search_path.glob("tbv2-feature-*.zip"):
            try:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    info = {
                        "path": str(zip_file),
                        "filename": zip_file.name,
                        "size_kb": zip_file.stat().st_size // 1024,
                    }

                    if "_metadata.yaml" in zf.namelist():
                        metadata = yaml.safe_load(
                            zf.read("_metadata.yaml").decode("utf-8")
                        )
                        info.update(metadata)
                    elif "feature.yaml" in zf.namelist():
                        feature_data = yaml.safe_load(
                            zf.read("feature.yaml").decode("utf-8")
                        )
                        info["version"] = feature_data.get("version", "unknown")

                    packages.append(info)
            except Exception:
                packages.append({
                    "path": str(zip_file),
                    "filename": zip_file.name,
                    "error": "Could not read package",
                })

        return packages
