# Karte K-API-02

```
Karte:       K-API-02
Paket:       P2-AUTO-MEDIUM – Mittlerer Fix, automatisch
Status:      GETESTET (vfs_view-Teil: 33 von 36; der vfs_shell-Teil `reason` – 5 IDs in `test_agent_session_tools.py`, 2–3 weitere – ist nach demselben Muster geplant, aber nicht angewendet)
Fix-Typ:     A
Fix-Auftrag: FIX-17
```

**Ursache:** `vfs_view` ist jetzt eine Koroutine und `vfs_shell` hat die Signatur `(reason, command)`; die Tests rufen beide synchron bzw. mit einem Argument auf.

**Betroffen:** 46 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-02

**Beleg:** `toolboxv2/mods/isaa/base/patch/vfs_shell_tool.py:445` `def vfs_shell(reason: str, command: str)`, `:1196` `async def vfs_view(`. CI: `TypeError: 'coroutine' object is not subscriptable` (29), `<coroutine object make_vfs_view.<locals>.vfs_view …> is not an instance of <class 'dict'>`, `'closed' != 'open'`, `vfs_shell() missing 1 required positional argument: 'command'` (5)

**Hypothese:** Schnittstelle geändert, Tests nicht. `'closed' != 'open'` ist derselbe Fehler: die Koroutine wurde nie ausgeführt, die Datei nie geöffnet. `test_self_agents_default_tool_set` und der vfs_shell-Gesundheitscheck in `test_agent_fetures` melden `missing … 'reason'`.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shared_store_edits.py -q
```

**Vorher:** `36 failed` in den vier Dateien

**Probe-Fix:** Hilfsfunktion `_sync_tool(fn)` im Testmodul (führt eine Koroutinen-Funktion auf einer eigenen Schleife aus) und `make_vfs_view(...)` → `_sync_tool(make_vfs_view(...))` (9 Stellen)

**Nachher:** `3 failed, 372 passed` – Rest: 2× `KeyError: '/system_context.md'` (K-API-09), 1× Tabs (K-API-22)

**Nachbarn:** s. Nachher; 2 IDs wechseln zur Ursache K-API-09

## Test-IDs

- `toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py` – 1
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py` – 8
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_self_agents_default_tool_set.py` – 2
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shared_store_edits.py` – 1
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py` – 7
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py` – 8
- `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py` – 19

Beispiel aus `raw/macos-3.11.txt`:

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestSessionToolsLive::test_07_vfs_shell_health_check | FAILED | AssertionError | 'FAILED' not found in ('HEALTHY', 'GUARANTEED', 'SKIPPED') : vfs_shell health: FAILED — make_vfs_shell.<locals>.vfs_shell() missing 1 required positional argument: 'reason'
```
