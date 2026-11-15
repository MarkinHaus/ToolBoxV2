# FIX-38 – aus Karte K-OS-03

Paket: **P6-PLATTFORM** (Plattform- und versionsabhängig) · Fix-Typ: **A** · Karte: UNGETESTET · betrifft 4 Test-IDs

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-38  (aus Karte K-OS-03)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_core/test_endpoint_dist_dir.py, toolboxv2/tests/test_isaa/test_codig/test_base_coder.py   (nur diese ändern)
Vorher lesen:  die 4 Tests und die Produktfunktion, die den Pfad liefert
```

**Schritte:**

1. Im Test den Temp-Pfad direkt nach `mkdtemp()` mit `os.path.realpath` auflösen und nur noch den aufgelösten verwenden.
2. Bei `test_base_coder` zusätzlich Trenner normalisieren (`Path(x).resolve()` auf beiden Seiten).

**Nicht tun:** Nicht auf Teilstring-Vergleich des letzten Ordnernamens abschwächen. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 4 IDs grün (macOS: 2, Windows: 4).
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_core/test_endpoint_dist_dir.py::TestDistDirManifestBase::test_dist_dir_relative_manifest_value_resolved_against_root
toolboxv2/tests/test_core/test_endpoint_dist_dir.py::TestDistDirManifestBase::test_dist_dir_uses_manifest_base_dir
toolboxv2/tests/test_isaa/test_codig/test_base_coder.py::TestCoderAgentBash::test_cwd_is_worktree
toolboxv2/tests/test_isaa/test_codig/test_base_coder.py::TestGitWorktreeReal::test_setup_creates_worktree
```
