# FIX-36 – aus Karte K-OS-02

Paket: **P6-PLATTFORM** (Plattform- und versionsabhängig) · Fix-Typ: **A** · Karte: UNGETESTET · betrifft 4 Test-IDs

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-36  (aus Karte K-OS-02)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/mods/isaa/module.py bzw. das Shell-Werkzeug (Sitzungs- und Marker-Logik)   (nur diese ändern)
Vorher lesen:  `shell_tool_function`, Marker-Erkennung, `TestMarkerPlumbing`
```

**Schritte:**

1. Unter Windows reproduzieren.
2. Prompt unterdrücken (`function prompt {''}` beim Sitzungsstart) oder Ausgabe bis zur Markierung sauber schneiden.
3. `test_fallback_when_marker_is_unsupported` einzeln prüfen (`'' != 'legacy'`).

**Nicht tun:** Test nicht auf `startswith` abschwächen. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 4 IDs grün unter Windows; Linux/macOS unverändert grün.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_shell_tool.py::TestCompletion::test_finished_status_and_exit_code
toolboxv2/tests/test_isaa/test_shell_tool.py::TestMarkerPlumbing::test_fallback_when_marker_is_unsupported
toolboxv2/tests/test_isaa/test_shell_tool.py::TestNoneCommand::test_none_for_optional_args
toolboxv2/tests/test_isaa/test_shell_tool.py::TestSessions::test_new_session_resets_state
```
