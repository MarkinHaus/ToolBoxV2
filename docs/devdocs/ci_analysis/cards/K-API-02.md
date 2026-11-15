# Karte K-API-02

```
Karte:       K-API-02
Paket:       P2-AUTO-MEDIUM – Mittlerer Fix, automatisch
Status:      GETESTET
Fix-Typ:     A
Fix-Auftrag: FIX-17
```

**Ursache:** `vfs_view` ist jetzt eine Koroutine und `vfs_shell` hat die Signatur `(reason, command)`. Die Tests rufen beide noch synchron bzw. ohne `reason` auf – und der eingebaute Gesundheitscheck des Produkts ebenfalls.

**Betroffen:** 46 Tests in 12 Jobs (alle 12); Liste in catalog.csv, Spalte karte=K-API-02

**Beleg:** `toolboxv2/mods/isaa/base/patch/vfs_shell_tool.py:445` `def vfs_shell(reason: str, command: str)`, `:1196` `async def vfs_view(`. `toolboxv2/mods/isaa/base/Agent/flow_agent.py:3613` `"live_test_inputs": [{"command": "ls /"}]` (ohne `reason`). CI: `TypeError: 'coroutine' object is not subscriptable` (29), `<coroutine object make_vfs_view.<locals>.vfs_view …> is not an instance of <class 'dict'>`, `'closed' != 'open'`, `vfs_shell() missing 1 required positional argument: 'command'` (5), `vfs_shell health: FAILED — … missing 1 required positional argument: 'reason'`

**Hypothese:** Schnittstelle bewusst geändert (Docstring), Tests nicht nachgezogen. `'closed' != 'open'` ist derselbe Fehler: die Koroutine wurde nie ausgeführt, die Datei nie geöffnet. Der Gesundheitscheck (`tool_manager.py:1116` ruft `entry.function(**live_test_inputs[0])`) ist ein **echter Produktfehler**: `vfs_shell` meldet im laufenden Agenten bei jedem Health-Check `FAILED`.

**Repro:**

```
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shared_store_edits.py toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py toolboxv2/tests/test_isaa/test_base/test_agent/test_self_agents_default_tool_set.py -q
python -m pytest toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py -q -k "test_03_per_tool_subtests or test_07_vfs_shell_health_check"
```

**Vorher:** erste Zeile: 44 der 46 IDs rot; zweite Zeile: `2 failed, 1 passed, 225 deselected`

**Probe-Fix:** (1) `_sync_tool`-Wrapper in den 4 vfs-Testdateien; (2) `test_agent_session_tools.py`: `self.sh = lambda command: _sh("test", command)` und `await self.view(...)`; (3) Produkt `flow_agent.py:3613`: `{"reason": "health check", "command": "ls /"}`

**Nachher:** erste Zeile: `3 failed, 430 passed, 20 skipped` – Rest: 2× `KeyError: '/system_context.md'` (wechseln zu K-API-09), 1× Tabs (K-API-22); zweite Zeile: `2 passed, 225 deselected`

**Nachbarn:** sechs Dateien vollständig gelaufen, keine neue rote ID; Probe auf `master` @ `ed32daf`

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
