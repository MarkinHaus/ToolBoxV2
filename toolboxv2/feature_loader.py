"""
Feature Loader - Automatisches Entpacken von Feature-ZIPs bei Installation

Dieses Modul wird beim Import von toolboxv2 ausgeführt und entpackt
die benötigten Features basierend auf den installierten optional-dependencies.

Flow:
1. pip install toolboxv2       → Alle ZIPs werden mitgeliefert, nur Core entpackt
2. pip install toolboxv2[web]  → Core + Web werden entpackt
3. pip install toolboxv2[all]  → Alle Features werden entpackt

Die Feature-ZIPs liegen in toolboxv2/features_packed/ und werden nach
toolboxv2/features/{name}/ und toolboxv2/{target_dirs}/ entpackt.
"""
import importlib.util
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set

# Mapping: pip extra name → feature names
EXTRA_TO_FEATURES: Dict[str, List[str]] = {
    "cli": ["cli"],
    "web": ["web"],
    "desktop": ["desktop"],
    "exotic": ["exotic"],
    "isaa": ["isaa"],
    "all": ["core", "cli", "web", "desktop", "exotic", "isaa"],
    "production": ["core", "cli", "web"],
    "dev": ["core", "cli", "web", "desktop", "exotic", "isaa"],
}

# Feature detection via importable packages
# Wenn eines dieser Packages installiert ist, wird das Feature benötigt
FEATURE_DETECTION: Dict[str, List[str]] = {
    "cli": ["prompt_toolkit", "rich", "readchar"],
    "web": ["starlette", "uvicorn", "httpx"],
    "desktop": ["PyQt6"],
    "exotic": ["scipy", "matplotlib", "pandas"],
    "isaa": ["litellm", "langchain_core", "groq"],
}

# Core ist immer dabei
CORE_FEATURE = "core"


def get_package_root() -> Path:
    """Ermittle das Root-Verzeichnis des toolboxv2 Packages."""
    return Path(__file__).parent


def get_features_packed_dir() -> Path:
    """Verzeichnis mit gepackten Features."""
    return get_package_root() / "features_packed"


def get_features_dir() -> Path:
    """Verzeichnis für entpackte Feature-Configs."""
    return get_package_root() / "features"


def is_feature_installed(feature_name: str) -> bool:
    """
    Prüfe ob ein Feature bereits entpackt/installiert ist.
    
    Ein Feature gilt als installiert wenn:
    1. Das Feature-Verzeichnis existiert UND
    2. Eine feature.yaml UND .installed Marker darin liegt
    """
    feature_dir = get_features_dir() / feature_name
    feature_yaml = feature_dir / "feature.yaml"
    installed_marker = feature_dir / ".installed"
    return feature_yaml.exists() and installed_marker.exists()


def is_feature_available(feature_name: str) -> bool:
    """
    Prüfe ob ein Feature als Source oder ZIP verfügbar ist.
    """
    # Als Source vorhanden?
    feature_yaml = get_features_dir() / feature_name / "feature.yaml"
    if feature_yaml.exists():
        return True
    
    # Als ZIP vorhanden?
    return is_feature_packed(feature_name)


def is_feature_packed(feature_name: str) -> bool:
    """Prüfe ob ein Feature als ZIP verfügbar ist."""
    packed_dir = get_features_packed_dir()
    if not packed_dir.exists():
        return False
    
    # Suche nach tbv2-feature-{name}-*.zip
    for _ in packed_dir.glob(f"tbv2-feature-{feature_name}-*.zip"):
        return True
    return False


def get_packed_feature_path(feature_name: str) -> Optional[Path]:
    """Finde den Pfad zum gepackten Feature ZIP."""
    packed_dir = get_features_packed_dir()
    if not packed_dir.exists():
        return None
    
    # Finde neueste Version
    candidates = list(packed_dir.glob(f"tbv2-feature-{feature_name}-*.zip"))
    if not candidates:
        return None
    
    # Sortiere nach Dateiname (enthält Version)
    candidates.sort(reverse=True)
    return candidates[0]


def detect_installed_extras() -> Set[str]:
    """
    Erkenne welche optional-dependencies installiert sind.
    
    Prüft ob die Marker-Packages für jedes Feature importierbar sind.
    """
    installed_extras = set()
    
    for extra_name, marker_packages in FEATURE_DETECTION.items():
        # Prüfe ob mindestens ein Marker-Package installiert ist
        for pkg in marker_packages:
            if importlib.util.find_spec(pkg) is not None:
                installed_extras.add(extra_name)
                break
    
    return installed_extras


def get_required_features() -> Set[str]:
    """
    Ermittle welche Features basierend auf installierten Extras benötigt werden.
    
    Returns:
        Set von Feature-Namen die entpackt werden sollen
    """
    required = {CORE_FEATURE}  # Core ist immer dabei
    
    installed_extras = detect_installed_extras()
    
    for extra in installed_extras:
        if extra in EXTRA_TO_FEATURES:
            required.update(EXTRA_TO_FEATURES[extra])
    
    return required


def unpack_feature(feature_name: str, force: bool = False) -> bool:
    """
    Entpacke ein Feature aus seinem ZIP.
    
    Args:
        feature_name: Name des Features
        force: Überschreibe existierendes Feature
        
    Returns:
        True wenn erfolgreich entpackt
    """
    if is_feature_installed(feature_name) and not force:
        return True  # Bereits installiert
    
    zip_path = get_packed_feature_path(feature_name)
    if not zip_path:
        # Kein ZIP vorhanden - Feature ist vielleicht schon im Source
        # Erstelle .installed Marker falls feature.yaml existiert
        feature_yaml = get_features_dir() / feature_name / "feature.yaml"
        if feature_yaml.exists():
            (get_features_dir() / feature_name / ".installed").touch()
            return True
        return False
    
    try:
        package_root = get_package_root()
        features_dir = get_features_dir()
        feature_target = features_dir / feature_name
        
        # Erstelle Features-Verzeichnis falls nicht vorhanden
        features_dir.mkdir(parents=True, exist_ok=True)
        feature_target.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Entpacke feature.yaml nach features/{name}/
            if "feature.yaml" in zf.namelist():
                with zf.open("feature.yaml") as src:
                    (feature_target / "feature.yaml").write_bytes(src.read())
                
                # Extrahiere requirements.txt falls vorhanden
                if "requirements.txt" in zf.namelist():
                    with zf.open("requirements.txt") as src:
                        (feature_target / "requirements.txt").write_bytes(src.read())
            
            # Entpacke files/ nach entsprechenden Verzeichnissen
            for name in zf.namelist():
                if name.startswith("files/") and not name.endswith("/"):
                    # Relativer Pfad ohne "files/" Prefix
                    rel_path = name[6:]  # Remove "files/"
                    target_file = package_root / rel_path
                    
                    # Erstelle Verzeichnis falls nötig
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Extrahiere Datei
                    with zf.open(name) as src:
                        target_file.write_bytes(src.read())
        
        # Erstelle .installed Marker
        (feature_target / ".installed").touch()
        
        return True
        
    except Exception as e:
        print(f"Warning: Failed to unpack feature '{feature_name}': {e}", file=sys.stderr)
        return False


def ensure_features_loaded() -> Dict[str, bool]:
    """
    Stelle sicher dass alle benötigten Features geladen sind.
    
    Wird beim Import von toolboxv2 aufgerufen.
    
    Returns:
        Dict mit {feature_name: success}
    """
    results = {}
    required = get_required_features()
    
    for feature_name in required:
        if is_feature_installed(feature_name):
            results[feature_name] = True
        elif is_feature_packed(feature_name):
            results[feature_name] = unpack_feature(feature_name)
        else:
            # Feature weder installiert noch gepackt
            # Prüfe ob Source vorhanden (Development Mode)
            feature_yaml = get_features_dir() / feature_name / "feature.yaml"
            if feature_yaml.exists():
                # Erstelle .installed Marker
                (get_features_dir() / feature_name / ".installed").touch()
                results[feature_name] = True
            else:
                results[feature_name] = False
    
    return results


def get_feature_status() -> Dict[str, dict]:
    """
    Zeige Status aller Features.
    
    Returns:
        Dict mit Feature-Infos
    """
    status = {}
    
    # Alle möglichen Features
    all_features = {CORE_FEATURE}
    for features in EXTRA_TO_FEATURES.values():
        all_features.update(features)
    
    required = get_required_features()
    
    for feature_name in sorted(all_features):
        status[feature_name] = {
            "required": feature_name in required,
            "installed": is_feature_installed(feature_name),
            "packed": is_feature_packed(feature_name),
            "available": is_feature_available(feature_name),
        }
    
    return status


def list_available_features() -> List[str]:
    """Liste alle verfügbaren Features (installiert oder gepackt)."""
    available = set()
    
    # Installierte Features
    features_dir = get_features_dir()
    if features_dir.exists():
        for feature_dir in features_dir.iterdir():
            if feature_dir.is_dir() and (feature_dir / "feature.yaml").exists():
                available.add(feature_dir.name)
    
    # Gepackte Features
    packed_dir = get_features_packed_dir()
    if packed_dir.exists():
        for zip_file in packed_dir.glob("tbv2-feature-*.zip"):
            # Extract name from tbv2-feature-{name}-{version}.zip
            name = zip_file.stem
            if name.startswith("tbv2-feature-"):
                parts = name[13:].rsplit("-", 2)  # Remove prefix, split by -
                if parts:
                    available.add(parts[0])
    
    return sorted(available)


def cleanup_unpacked_features(keep_features: Optional[Set[str]] = None):
    """
    Lösche entpackte Features die nicht mehr benötigt werden.
    
    Args:
        keep_features: Features die behalten werden sollen (Default: required features)
    """
    if keep_features is None:
        keep_features = get_required_features()
    
    features_dir = get_features_dir()
    package_root = get_package_root()
    
    if not features_dir.exists():
        return
    
    for feature_dir in features_dir.iterdir():
        if feature_dir.is_dir() and feature_dir.name not in keep_features:
            # Prüfe ob es ein gepacktes Backup gibt
            if is_feature_packed(feature_dir.name):
                # Lösche entpackte Dateien
                # Lade feature.yaml um files patterns zu bekommen
                feature_yaml = feature_dir / "feature.yaml"
                if feature_yaml.exists():
                    try:
                        import yaml
                        with open(feature_yaml) as f:
                            config = yaml.safe_load(f) or {}
                        
                        # Lösche Feature-Dateien
                        for pattern in config.get("files", []):
                            if pattern.endswith("/*"):
                                dir_path = package_root / pattern[:-2]
                                if dir_path.exists():
                                    shutil.rmtree(dir_path)
                            else:
                                file_path = package_root / pattern
                                if file_path.exists():
                                    file_path.unlink()
                    except Exception:
                        pass
                
                # Lösche Feature-Verzeichnis
                shutil.rmtree(feature_dir)


# CLI für manuelle Feature-Verwaltung
def main():
    """CLI für Feature-Loader."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ToolBoxV2 Feature Loader",
        prog="python -m toolboxv2.feature_loader"
    )
    
    subparsers = parser.add_subparsers(dest="command")
    
    # status
    subparsers.add_parser("status", help="Show feature status")
    
    # list
    subparsers.add_parser("list", help="List available features")
    
    # load
    p_load = subparsers.add_parser("load", help="Load/unpack required features")
    p_load.add_argument("--all", action="store_true", help="Load all features")
    p_load.add_argument("--force", action="store_true", help="Force reload")
    
    # unpack
    p_unpack = subparsers.add_parser("unpack", help="Unpack specific feature")
    p_unpack.add_argument("feature", help="Feature name")
    p_unpack.add_argument("--force", action="store_true", help="Force unpack")
    
    # cleanup
    p_cleanup = subparsers.add_parser("cleanup", help="Remove non-required features")
    
    args = parser.parse_args()
    
    if args.command == "status":
        print("\n=== Feature Status ===\n")
        status = get_feature_status()
        required = get_required_features()
        installed_extras = detect_installed_extras()
        
        print(f"Detected extras: {installed_extras or 'none'}")
        print(f"Required features: {required}")
        print()
        
        for name, info in status.items():
            req = "✓" if info["required"] else " "
            inst = "✓" if info["installed"] else "✗"
            pack = "✓" if info["packed"] else " "
            avail = "✓" if info["available"] else "✗"
            print(f"  [{req}] {name:12} installed={inst} packed={pack} available={avail}")
    
    elif args.command == "list":
        print("\n=== Available Features ===\n")
        for feature in list_available_features():
            installed = "✓" if is_feature_installed(feature) else " "
            packed = "(packed)" if is_feature_packed(feature) else "(source)"
            print(f"  [{installed}] {feature} {packed}")
        
    elif args.command == "load":
        if args.all:
            features = set(list_available_features())
        else:
            features = get_required_features()
        
        print(f"\nLoading features: {features}\n")
        for feature in features:
            success = unpack_feature(feature, force=args.force)
            status = "✓" if success else "✗"
            print(f"  {status} {feature}")
            
    elif args.command == "unpack":
        if not is_feature_available(args.feature):
            print(f"✗ Feature not available: {args.feature}")
            print(f"  Available: {list_available_features()}")
            sys.exit(1)
        
        success = unpack_feature(args.feature, force=args.force)
        if success:
            print(f"✓ Unpacked: {args.feature}")
        else:
            print(f"✗ Failed to unpack: {args.feature}")
            sys.exit(1)
    
    elif args.command == "cleanup":
        print("\nCleaning up non-required features...\n")
        required = get_required_features()
        cleanup_unpacked_features(required)
        print("Done.")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
