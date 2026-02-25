# Install Validator - Verwendung und Integration

## Übersicht

Der `install_validator.py` bietet umfassende Validierungsfunktionen für installierte Pakete. Er wird während der Paketinstallation und Updates verwendet, um sicherzustellen, dass:

1. Die Konfigurationsdatei valide ist
2. Alle Abhängigkeiten erfüllt sind
3. Die Module korrekt importiert werden können
4. Die Dateistruktur vollständig ist

## Status

**✅ Komponente erstellt**: `toolboxv2/utils/install_validator.py` (563 Zeilen)
**⚠️ Integration teilweise**: Nur grundlegende Validierung in `ModManager.py` aktiv
**📋 Todo:** Volle Integration mit `validate_installation()` Funktion

## Aktuelle Verwendung

### In Tests

Der Validator wird in `toolboxv2/tests/test_utils/test_install_validator.py` getestet:

```python
from toolboxv2.utils.extras.install.install_validator import (
    validate_tb_config,
    validate_dependencies,
    validate_module_import,
    validate_installation,
    ValidationResult,
)
```

### In ModManager.py (Aktuell - Vereinfacht)

**Datei**: `toolboxv2/mods/CloudM/ModManager.py`
**Funktion**: `update_from_registry()` (Zeilen 1667-1684)

Aktuell wird nur eine **vereinfachte Validierung** durchgeführt:

```python
# Step 3: Validate new installation
# Check that the module can be imported
try:
    # Basic validation: check if main files exist
    if current_path.exists():
        init_file = current_path / "__init__.py"
        main_file = current_path / f"{pkg_name}.py"

        if not (init_file.exists() or main_file.exists()):
            # Also check for v2 config
            config_file = current_path / f"{pkg_name}.yaml"
            if not config_file.exists():
                raise InstallationError(
                    f"Validation failed: No valid module files found"
                )
except Exception as validation_err:
    app.print(f"⚠ Validation failed: {validation_err}")
    raise InstallationError(f"Installation validation failed: {validation_err}")
```

## Vollständige Integration (Geplant)

### Schritt 1: Import hinzufügen

Am Anfang von `ModManager.py`:

```python
from toolboxv2.utils.extras.install.install_validator import (
    validate_installation,
    ValidationResult,
    InstallationError as ValidatorError,
)
```

### Schritt 2: Vollständige Validierung in `update_from_registry()`

Ersetzen der vereinfachten Validierung (Zeilen 1667-1684) durch:

```python
# Step 3: Validate new installation with full validator
try:
    app.print(f"🔍 Validating installation of {pkg_name}...")

    mods_dir = Path(app.start_dir) / "mods"
    validation_result = validate_installation(
        package_name=pkg_name,
        install_path=current_path,
        mods_dir=mods_dir,
        lock_manager=lock_manager,
    )

    if not validation_result.is_valid:
        # Validation failed - collect all errors
        error_details = []
        if validation_result.errors:
            error_details.append("Errors:")
            error_details.extend([f"  - {e}" for e in validation_result.errors])
        if validation_result.warnings:
            error_details.append("Warnings:")
            error_details.extend([f"  - {w}" for w in validation_result.warnings])

        error_msg = "\n".join(error_details)
        app.print(f"❌ Installation validation failed:\n{error_msg}")

        raise InstallationError(
            f"Package {pkg_name} failed validation:\n{error_msg}"
        )
    else:
        app.print(f"✓ Installation validated successfully")
        if validation_result.warnings:
            for warning in validation_result.warnings:
                app.print(f"⚠ {warning}")

except Exception as validation_err:
    app.print(f"⚠ Validation failed: {validation_err}")
    raise InstallationError(f"Installation validation failed: {validation_err}")
```

### Schritt 3: Validierung in `install_from_registry()` hinzufügen

Ähnliche Validierung auch in der `install_from_registry()` Funktion durchführen.

## Validator Funktionen

### `validate_installation()` - Hauptfunktion

Führt alle Validierungsprüfungen durch:

```python
def validate_installation(
    package_name: str,
    install_path: Path,
    mods_dir: Optional[Path] = None,
    lock_manager=None,
) -> ValidationResult:
```

**Prüfungen:**
1. Pfad-Existenz
2. Konfigurations-Validierung
3. Modul-Datei-Validierung
4. Import-Validierung
5. Abhängigkeits-Validierung (optional)

### `validate_tb_config()` - Konfiguration

```python
def validate_tb_config(config_path: Path) -> ValidationResult:
```

Validiert die `.yaml` Konfigurationsdatei gegen das Schema.

### `validate_dependencies()` - Abhängigkeiten

```python
def validate_dependencies(
    dependencies: List[str],
    mods_dir: Path,
) -> ValidationResult:
```

Prüft, ob alle Abhängigkeiten installiert sind.

### `validate_module_import()` - Import

```python
def validate_module_import(
    module_name: str,
    module_path: Path,
) -> ValidationResult:
```

Prüft, ob der Modul-Import funktioniert.

### `validate_module_files()` - Dateien

```python
def validate_module_files(module_path: Path) -> ValidationResult:
```

Prüft auf erforderliche Python-Dateien und verdächtige Dateien.

## ValidationResult Klasse

```python
@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
```

**Verwendung:**

```python
result = validate_installation("my-mod", Path("/path/to/mod"))

if result.is_valid:
    print("✓ Valid")
else:
    print("✗ Invalid")
    for error in result.errors:
        print(f"  ERROR: {error}")
    for warning in result.warnings:
        print(f"  WARNING: {warning}")

# Details abrufen
print(f"Package: {result.details['package_name']}")
print(f"Path: {result.details['install_path']}")
print(f"Errors: {result.details['error_count']}")
print(f"Warnings: {result.details['warning_count']}")
```

## Verwendung in der Registry CLI

Die Registry-CLI (`toolboxv2/utils/clis/cli_registry.py`) kann den Validator für `download`- und `publish`-Befehle verwenden:

```python
from toolboxv2.utils.extras.install.install_validator import validate_installation


# Nach dem Download validieren
def cmd_download(args):
    # ... download code ...

    # Validate downloaded package
    result = validate_installation(
        package_name=args.package,
        install_path=download_path,
    )

    if not result.is_valid:
        print_status(f"Download validation failed", "error")
        for error in result.errors:
            print(f"  - {error}")
        return 1
```

## Fehlerbehandlung

### ValidationError

Basis-Exception für Validierungsfehler:

```python
from toolboxv2.utils.extras.install.install_validator import ValidationError

try:
    result = validate_installation(...)
    if not result.is_valid:
        raise ValidationError("Installation failed")
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Spezielle Exceptions

- `ConfigValidationError` - Konfiguration ungültig
- `DependencyValidationError` - Abhängigkeiten fehlen
- `ImportValidationError` - Modul-Import fehlgeschlagen

## Integrations-Checklist

- [ ] Import von `install_validator` in `ModManager.py` hinzugefügt
- [ ] `validate_installation()` in `update_from_registry()` integriert
- [ ] `validate_installation()` in `install_from_registry()` integriert
- [ ] Validierung in Registry-CLI für `download` Befehl
- [ ] Validierung in Registry-CLI für `publish` Befehl (pre-upload check)
- [ ] Tests erweitert für alle Validierungs-Integrationen
- [ ] Dokumentation aktualisiert

## Rollback mit Validierung

Der Validator arbeitet zusammen mit dem `RollbackManager`:

1. **Backup erstellen** (davor)
2. **Installation durchführen**
3. **Validierung** → Wenn fehlschlägt:
4. **Rollback** durchführen

```python
backup_metadata = rollback_mgr.create_backup(...)

# Install new version
await install_from_registry(...)

# Validate
validation_result = validate_installation(...)
if not validation_result.is_valid:
    # Rollback
    rollback_mgr.restore_backup(backup_metadata.backup_id, ...)
    raise InstallationError("Validation failed, rolled back")
```

## Beispiele

### Beispiel 1: Einfache Validierung

```python
from pathlib import Path
from toolboxv2.utils.extras.install.install_validator import validate_installation

result = validate_installation(
    package_name="CloudM",
    install_path=Path("/mods/CloudM"),
)

if result.is_valid:
    print("✓ CloudM ist valide")
else:
    print("✗ Fehler:")
    for error in result.errors:
        print(f"  - {error}")
```

### Beispiel 2: Mit Abhängigkeiten

```python
mods_dir = Path("/mods")
result = validate_installation(
    package_name="my-mod",
    install_path=Path("/mods/my-mod"),
    mods_dir=mods_dir,
    lock_manager=lock_manager,
)

if not result.is_valid:
    missing = result.details.get("missing_dependencies", [])
    if missing:
        print(f"Fehlende Abhängigkeiten: {missing}")
```

### Beispiel 3: Nur Konfiguration prüfen

```python
from toolboxv2.utils.extras.install.install_validator import validate_tb_config

result = validate_tb_config(Path("/mods/my-mod/my-mod.yaml"))
if not result.is_valid:
    print("Konfiguration ungültig:")
    for error in result.errors:
        print(f"  - {error}")
```

## Timeline

- **Handoff Point 4**: `install_validator.py` erstellt
- **Handoff Point 5**: Vollständige Integration geplant (noch nicht durchgeführt)
- **Aktuell**: Nur vereinfachte Validierung in `ModManager.py`

## Nächste Schritte

1. **Integration in ModManager.py** durchführen
2. **Tests aktualisieren** für neue Integration
3. **CLI-Befehle** erweitern für Validierung
4. **Dokumentation** aktualisieren

---

**Status**: 📋 In Arbeit
**Zuletzt aktualisiert**: 2026-02-25
**Verantwortlich**: Handoff Point 5 - Final Validation & Rollout
