# Karte K-PY-02

```
Karte:       K-PY-02
Paket:       P6-PLATTFORM – Plattform- und versionsabhängig
Status:      UNGETESTET (Probe-Fix nicht angewendet; zweite ID ungeklärt)
Fix-Typ:     A
Fix-Auftrag: FIX-41
```

**Ursache:** Zwei Tests laufen nur unter 3.10 nicht: `asyncio.timeout` gibt es erst ab 3.11; `'NoneType' object has no attribute 'get_memory'`.

**Betroffen:** 2 Tests in 3 Jobs (macos-3.10, ubuntu-3.10, windows-3.10); Liste in catalog.csv, Spalte karte=K-PY-02

**Beleg:** CI nur 3.10: `test_performance_and_limiter.py::TestAgentPerformance::test_rate_limiter_blocks_correctly - AttributeError: module 'asyncio' has no attribute 'timeout'`; `test_agent_performance_stability.py::TestAgentPerformance::test_vfs_mount_blocking_analysis - AttributeError: 'NoneType' object has no attribute 'get_memory'`

**Hypothese:** (1) Der Test selbst nutzt `asyncio.timeout(...)`; im Produktcode kommt der Aufruf nicht vor (`grep` leer). `requires-python = ">=3.10"`. (2) Ungeklärt, vermutlich Folgefehler eines 3.10-Imports.

**Repro:**

```
python3.10 -m pytest toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py -q
```

**Vorher:** CI: 2 rot unter 3.10

**Probe-Fix:** `asyncio.wait_for(coro, timeout=…)` statt `async with asyncio.timeout(…)` im Test

**Nachher:** –

**Nachbarn:** –

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py` – 1
- `toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py` – 1

Beispiel aus `raw/macos-3.10.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py::TestAgentPerformance::test_vfs_mount_blocking_analysis | FAILED | AttributeError | 'NoneType' object has no attribute 'get_memory'
```
