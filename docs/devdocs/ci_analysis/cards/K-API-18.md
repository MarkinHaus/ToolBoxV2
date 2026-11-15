# Karte K-API-18

```
Karte:       K-API-18
Paket:       P4-ENTSCHEIDUNG – Braucht Entscheidung des Owners
Status:      UNGETESTET (Richtung offen)
Fix-Typ:     B (Hat das Dockerfile noch Feature-Build-Argumente (`FEATURE_*`), oder ist es bewusst ein Vollabbild?)
Fix-Auftrag: FIX-23
```

**Ursache:** `test_docker_build.py` verlangt für jedes Feature in `FEATURE_MAP` ein `ARG` im Dockerfile; `FEATURE_CLI` fehlt dort.

**Betroffen:** 1 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-18

**Beleg:** CI: `AssertionError: 'FEATURE_CLI' not found in '# syntax=docker/dockerfile:1.4 … ToolBoxV2 - Full-Featured Docker Image (python:3.11-slim) …'`

**Hypothese:** Das Dockerfile wurde auf ein „Full-Featured“-Abbild umgestellt (keine Feature-Schalter mehr), oder der Schalter fehlt versehentlich.

**Repro:**

```
python -m pytest toolboxv2/tests/test_core/test_docker_build.py -q
```

**Vorher:** `1 failed, 41 passed`

**Probe-Fix:** –

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_core/test_docker_build.py` – 1

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_core/test_docker_build.py::TestDockerfileStructure::test_all_feature_build_args_declared | FAILED | AssertionError | 'FEATURE_CLI' not found in '# syntax=docker/dockerfile:1.4\n# ============================================================================\n# ToolBoxV2 - Full-Featured Docker Image (python:3.11-slim)\
```
