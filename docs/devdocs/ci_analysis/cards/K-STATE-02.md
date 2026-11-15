# Karte K-STATE-02

```
Karte:       K-STATE-02
Paket:       P5-ARCH – Architektur
Status:      UNGETESTET (Verschmutzer je ID nicht isoliert; belegt ist nur: einzeln grün, im Gesamtlauf rot, und die Schreibzugriffe auf `sys.modules`)
Fix-Typ:     C (-> ARCH-01)
Fix-Auftrag: – (ARCH-01)
```

**Ursache:** Testmodule ersetzen beim Import Einträge in `sys.modules` dauerhaft durch `MagicMock`; spätere Tests bekommen die Attrappen statt der echten Module.

**Betroffen:** 14 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-STATE-02

**Beleg:** `toolboxv2/tests/test_isaa/test_base/test_agent/conftest2.py:8-12`: `sys.modules['toolboxv2.mods.isaa.base.Agent.types'] = MagicMock(__path__=[])`, ebenso `rule_set`, `litellm`, `python_a2a`, `mcp.server.fastmcp`. 13 Testdateien schreiben in `sys.modules`. CI: `ImportError: cannot import name 'Result' from 'toolboxv2.utils.system.types' (unknown location)` (4, `test_types.py`); 9 Fehler in `test_narrator.py` (`IndexError`, `None is not an instance of dict`, …)

**Hypothese:** `test_narrator.py` (100 Tests) und `test_types.py` (9 Tests) sind einzeln grün und schlagen nur im Gesamtlauf fehl. Ein Modul wurde vorher durch eine Attrappe oder ein Namespace-Paket ersetzt (`unknown location` heißt: Modul ohne `__file__`). Welcher Verschmutzer genau welche der 13 IDs trifft, ist nicht einzeln zugeordnet.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_narrator.py -q   # allein: 100 passed
python -m pytest toolboxv2/tests/ -q                                                   # Gesamtlauf: 9–10 rot
```

**Vorher:** allein grün; im Gesamtlauf rot (CI, alle 12 Jobs)

**Probe-Fix:** kein Einzel-Fix (→ ARCH-01)

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_narrator.py` – 10
- `toolboxv2/tests/test_mini/test_system/test_types.py` – 4

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_narrator.py::TestNarratorRobustness::test_mixed_workload_priorities | FAILED | IndexError | list index out of range
```
