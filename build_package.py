"""
Build Script für ToolBoxV2 Package Distribution

Dieses Script bereitet das Package für pip upload vor:
1. Packt alle Features (außer core) in ZIP-Dateien
2. Kopiert sie nach toolboxv2/features_packed/
3. Optional: Entfernt die Feature-Source-Dateien (für Production)

Flow:
    pip install toolboxv2       → Alle ZIPs werden mitgeliefert, nur Core ist entpackt
    pip install toolboxv2[web]  → feature_loader.py entpackt Web-ZIP automatisch
    pip install toolboxv2[all]  → Alle ZIPs werden entpackt

Verwendung:
    python build_package.py          # Nur Features packen (Development)
    python build_package.py --clean  # Features packen + Source entfernen (Production)
    python build_package.py --verify # Prüfe ob alles korrekt ist
"""
import argparse
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

# Pfade
TOOLBOX_ROOT = Path(__file__).parent
TOOLBOXV2_DIR = TOOLBOX_ROOT / "toolboxv2"
FEATURES_DIR = TOOLBOXV2_DIR / "features"
FEATURES_PACKED_DIR = TOOLBOXV2_DIR / "features_packed"

# Features die NICHT gepackt werden (immer im Source)
ALWAYS_UNPACKED = {"core"}


def load_feature_config(feature_name: str) -> dict:
    """Lade feature.yaml Konfiguration."""
    feature_yaml = FEATURES_DIR / feature_name / "feature.yaml"
    if not feature_yaml.exists():
        return {}
    
    with open(feature_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_all_features() -> list:
    """Liste alle verfügbaren Features auf."""
    if not FEATURES_DIR.exists():
        return []
    
    return [
        d.name for d in FEATURES_DIR.iterdir()
        if d.is_dir() and (d / "feature.yaml").exists()
    ]


def pack_feature(feature_name: str, output_dir: Path) -> Path:
    """
    Packe ein Feature in eine ZIP-Datei.
    
    Args:
        feature_name: Name des Features
        output_dir: Zielverzeichnis für ZIP
        
    Returns:
        Pfad zur erstellten ZIP-Datei
    """
    config = load_feature_config(feature_name)
    version = config.get("version", "0.0.0")
    files_patterns = config.get("files", [])
    dependencies = config.get("dependencies", [])
    
    # ZIP Dateiname (ohne Datum für konsistente Namen)
    zip_name = f"tbv2-feature-{feature_name}-{version}.zip"
    zip_path = output_dir / zip_name
    
    # Lösche alte Version falls vorhanden
    if zip_path.exists():
        zip_path.unlink()
    
    print(f"  Packing {feature_name} v{version}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. feature.yaml
        feature_yaml = FEATURES_DIR / feature_name / "feature.yaml"
        if feature_yaml.exists():
            zf.write(feature_yaml, "feature.yaml")
        
        # 2. requirements.txt
        if dependencies:
            req_content = "\n".join(dependencies)
            zf.writestr("requirements.txt", req_content)
        
        # 3. Feature-Dateien
        files_added = 0
        for pattern in files_patterns:
            files_added += add_files_to_zip(zf, pattern)
        
        # 4. Metadata
        metadata = {
            "feature_name": feature_name,
            "version": version,
            "packed_at": datetime.now().isoformat(),
            "files_count": len(zf.namelist()),
            "patterns": files_patterns,
        }
        zf.writestr("_metadata.yaml", yaml.dump(metadata, allow_unicode=True))
    
    size_kb = zip_path.stat().st_size // 1024
    print(f"    → {zip_name} ({size_kb} KB, {files_added} files)")
    
    return zip_path


def add_files_to_zip(zf: zipfile.ZipFile, pattern: str) -> int:
    """
    Füge Dateien basierend auf Pattern zum ZIP hinzu.
    
    Returns:
        Anzahl hinzugefügter Dateien
    """
    count = 0
    
    if pattern.endswith("/*"):
        # Ordner-Pattern
        dir_pattern = pattern[:-2]
        source_dir = TOOLBOXV2_DIR / dir_pattern
        
        if source_dir.exists() and source_dir.is_dir():
            for file_path in source_dir.rglob("*"):
                if file_path.is_file() and "__pycache__" not in str(file_path):
                    rel_path = file_path.relative_to(TOOLBOXV2_DIR)
                    arc_name = f"files/{rel_path}"
                    zf.write(file_path, arc_name)
                    count += 1
    else:
        # Einzelne Datei
        source_file = TOOLBOXV2_DIR / pattern
        if source_file.exists() and source_file.is_file():
            arc_name = f"files/{pattern}"
            zf.write(source_file, arc_name)
            count += 1
    
    return count


def clean_feature_sources(feature_name: str):
    """
    Entferne Source-Dateien eines Features (nach dem Packen).
    
    ACHTUNG: Dies entfernt die echten Source-Dateien!
    Nur für Distribution verwenden.
    """
    config = load_feature_config(feature_name)
    files_patterns = config.get("files", [])
    
    print(f"  Cleaning sources for {feature_name}...")
    
    for pattern in files_patterns:
        if pattern.endswith("/*"):
            dir_pattern = pattern[:-2]
            source_dir = TOOLBOXV2_DIR / dir_pattern
            
            if source_dir.exists():
                shutil.rmtree(source_dir)
                print(f"    Removed: {dir_pattern}/")
        else:
            source_file = TOOLBOXV2_DIR / pattern
            if source_file.exists():
                source_file.unlink()
                print(f"    Removed: {pattern}")
    
    # Entferne auch das Feature-Verzeichnis (außer feature.yaml bleibt)
    feature_dir = FEATURES_DIR / feature_name
    if feature_dir.exists():
        # Behalte nur feature.yaml
        for item in feature_dir.iterdir():
            if item.name != "feature.yaml":
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()


def verify_package():
    """Verifiziere dass das Package korrekt strukturiert ist."""
    print("\n=== Verifying Package Structure ===\n")
    
    errors = []
    warnings = []
    
    # 1. Prüfe features_packed Verzeichnis
    if not FEATURES_PACKED_DIR.exists():
        errors.append("features_packed/ directory missing")
    else:
        packed_features = list(FEATURES_PACKED_DIR.glob("tbv2-feature-*.zip"))
        print(f"Packed features: {len(packed_features)}")
        for zf in packed_features:
            print(f"  • {zf.name} ({zf.stat().st_size // 1024} KB)")
    
    # 2. Prüfe Core Feature (sollte immer entpackt sein)
    core_yaml = FEATURES_DIR / "core" / "feature.yaml"
    if not core_yaml.exists():
        errors.append("Core feature not found (features/core/feature.yaml)")
    else:
        print("✓ Core feature present (always unpacked)")
    
    # 3. Prüfe feature_loader.py
    loader = TOOLBOXV2_DIR / "feature_loader.py"
    if not loader.exists():
        errors.append("feature_loader.py missing")
    else:
        print("✓ feature_loader.py present")
    
    # 4. Prüfe __init__.py Integration
    init_file = TOOLBOXV2_DIR / "__init__.py"
    if init_file.exists():
        content = init_file.read_text()
        if "feature_loader" not in content:
            warnings.append("__init__.py should import feature_loader for auto-loading")
        else:
            print("✓ __init__.py imports feature_loader")
    
    # 5. Prüfe dass alle Features entweder gepackt oder Core sind
    all_features = get_all_features()
    for feature in all_features:
        if feature in ALWAYS_UNPACKED:
            continue
        
        zip_exists = any(
            FEATURES_PACKED_DIR.glob(f"tbv2-feature-{feature}-*.zip")
        )
        if not zip_exists:
            warnings.append(f"Feature '{feature}' is not packed (will be included as source)")
    
    # Ergebnisse
    print()
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
    
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ! {w}")
    
    if not errors and not warnings:
        print("✓ Package structure looks good!")
    
    return len(errors) == 0


def create_manifest_in():
    """
    Erstelle MANIFEST.in für setuptools.
    
    Stellt sicher dass features_packed/ in das Package aufgenommen wird.
    """
    manifest_content = """# Include feature packages
recursive-include toolboxv2/features_packed *.zip
recursive-include toolboxv2/features_packed *.md

# Include feature configs (only core is unpacked by default)
recursive-include toolboxv2/features/core *

# Exclude Python cache
global-exclude __pycache__
global-exclude *.py[cod]
global-exclude .DS_Store
"""
    
    manifest_path = TOOLBOX_ROOT / "MANIFEST.in"
    manifest_path.write_text(manifest_content)
    print(f"✓ Created {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Build ToolBoxV2 Package")
    parser.add_argument("--clean", action="store_true", 
                        help="Remove source files after packing (for distribution)")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify package structure")
    parser.add_argument("--features", nargs="*",
                        help="Specific features to pack (default: all except core)")
    parser.add_argument("--manifest", action="store_true",
                        help="Create MANIFEST.in file")
    
    args = parser.parse_args()
    
    if args.verify:
        success = verify_package()
        sys.exit(0 if success else 1)
    
    if args.manifest:
        create_manifest_in()
        return
    
    # Erstelle Output-Verzeichnis
    FEATURES_PACKED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Bestimme welche Features gepackt werden sollen
    if args.features:
        features_to_pack = [f for f in args.features if f not in ALWAYS_UNPACKED]
    else:
        features_to_pack = [f for f in get_all_features() if f not in ALWAYS_UNPACKED]
    
    print(f"\n=== Packing Features ===\n")
    print(f"Features to pack: {features_to_pack}")
    print(f"Always unpacked: {ALWAYS_UNPACKED}")
    print(f"Output: {FEATURES_PACKED_DIR}\n")
    
    # Packe Features
    packed = []
    for feature in features_to_pack:
        try:
            zip_path = pack_feature(feature, FEATURES_PACKED_DIR)
            packed.append((feature, zip_path))
        except Exception as e:
            print(f"  ERROR packing {feature}: {e}")
            continue
    
    # Optional: Clean sources
    if args.clean and packed:
        print(f"\n=== Cleaning Source Files ===\n")
        print("WARNING: This removes actual source files!")
        print("Only use this for creating distribution packages.")
        confirm = input("\nContinue? (yes/no): ").strip().lower()
        
        if confirm == "yes":
            for feature, _ in packed:
                clean_feature_sources(feature)
            print("\nSource files cleaned.")
        else:
            print("Skipped cleaning.")
    
    # Erstelle MANIFEST.in
    create_manifest_in()
    
    print(f"\n=== Done ===\n")
    verify_package()


if __name__ == "__main__":
    main()
