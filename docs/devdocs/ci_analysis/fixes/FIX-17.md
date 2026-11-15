# FIX-17 – aus Karte K-API-02

Paket: **P2-AUTO-MEDIUM** (Mittlerer Fix, automatisch) · Fix-Typ: **A** · Karte: GETESTET · betrifft 46 Test-IDs

```
Fix:           FIX-17  (aus Karte K-API-02)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeile 3613); toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py, test_vfs_shell_tool.py, test_vfs_shell_bullet_proof.py, test_vfs_shared_store_edits.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py   (nur diese ändern)
Vorher lesen:  `vfs_shell_tool.py:437-450` und `:1188-1200`; `tool_manager.py:1080-1130` (`health_check_single`); `flow_agent.py:3608-3640` (`_TOOL_HEALTH_EXTENSIONS`); in jeder Testdatei, wie `self.view`/`self.sh` gebaut werden
```

**Schritte:**

1. `flow_agent.py:3613`: `"live_test_inputs": [{"command": "ls /"}],` → `"live_test_inputs": [{"reason": "health check", "command": "ls /"}],`. Sonst nichts in der Datei ändern (die Zeile `vfs_view` mit `/system_context.md` gehört zu einer anderen Karte).
2. In die vier vfs-Testdateien vor der ersten `class`-Zeile einfügen:

```python
def _sync_tool(fn):
    """vfs_view is a coroutine function since the async VFS refactor; these sync
    tests call it directly, so run it to completion on a private loop."""
    import asyncio, functools, inspect
    if not inspect.iscoroutinefunction(fn):
        return fn
    @functools.wraps(fn)
    def _run(*a, **k):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(fn(*a, **k))
        finally:
            loop.close()
    return _run
```
3. Dort jedes `make_vfs_view(<arg>)` – nicht die Import-Zeile – durch `_sync_tool(make_vfs_view(<arg>))` ersetzen (test_vfs_v2.py 5, test_vfs_shell_tool.py 1, test_vfs_shell_bullet_proof.py 2, test_vfs_shared_store_edits.py 1).
4. `test_agent_session_tools.py`, Klasse `TestVFSToolContracts.setUp`: die zwei Zeilen

```python
        self.sh = make_vfs_shell(self.session)
        self.view = make_vfs_view(self.session)
```

ersetzen durch

```python
        _sh = make_vfs_shell(self.session)
        # vfs_shell(reason, command): reason is required since the VFS tool refactor
        self.sh = lambda command: _sh("test", command)
        self.view = make_vfs_view(self.session)  # coroutine function: await its result
```

und in derselben Klasse die drei Aufrufe `self.view(` zu `await self.view(` machen (`self._check(self.view(` → `self._check(await self.view(`, zweimal `r = self.view(` → `r = await self.view(`).

**Nicht tun:** `vfs_shell_tool.py` nicht ändern (die neue Signatur ist gewollt). `test_self_agents_default_tool_set.py` und `test_agent_fetures.py` nicht ändern – sie werden durch Schritt 1 grün. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- Repro-Befehle der Karte: erste Zeile genau 3 rot (`test_close_others_does_not_close_system_files`, `test_close_others_preserves_system_files`, `test_write_with_tabs`), zweite Zeile `2 passed`.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py::TestSessionToolsLive::test_07_vfs_shell_health_check
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_26_vfs_shell_ls_contract
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_27_vfs_shell_result_has_four_keys
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_28_vfs_shell_fail_has_error_info
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_29_vfs_view_returns_dict
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_30_vfs_view_success_has_content_and_showing
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_31_vfs_view_fail_has_error_key
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_32_vfs_shell_unknown_cmd_stderr_has_hint
toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py::TestVFSToolContracts::test_33_vfs_shell_write_then_cleanup
toolboxv2/tests/test_isaa/test_base/test_agent/test_self_agents_default_tool_set.py::TestSelfAgentLiveTools::test_04_no_failed_tools
toolboxv2/tests/test_isaa/test_base/test_agent/test_self_agents_default_tool_set.py::TestSelfAgentLiveTools::test_05_skipped_tools_need_attention
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shared_store_edits.py::TestAgentRoundtripSharedMount::test_view_after_edit_shows_new_content
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestAgentRoundtrip::test_grep_find_edit_verify_single_line
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestVfsViewRealWorld::test_close_others_workflow
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestVfsViewRealWorld::test_line_range_precise
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestVfsViewRealWorld::test_scroll_to_class_shows_context
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestVfsViewRealWorld::test_scroll_to_method
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestVfsViewRealWorld::test_scroll_to_nonexistent_returns_hint
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_bullet_proof.py::TestVfsViewRealWorld::test_view_returns_total_lines
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_basic_open
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_close_others
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_close_others_preserves_system_files
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_line_range
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_missing_file_fails
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_result_has_required_keys
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_scroll_to
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_shell_tool.py::TestVfsView::test_scroll_to_not_found_hint
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewCloseOthers::test_close_others_closes_open_files
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewCloseOthers::test_close_others_does_not_close_system_files
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewCloseOthers::test_close_others_note_in_result
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewLineRange::test_open_full_file
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewLineRange::test_open_range
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewLineRange::test_open_range_clamps_end
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewLineRange::test_open_range_clamps_start
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewLineRange::test_open_range_updates_vfs_state
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewReturnSchema::test_basic_open
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewReturnSchema::test_nonexistent
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewReturnSchema::test_scroll_to
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_case_insensitive
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_context_lines
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_finds_line
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_missing_pattern
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_overrides_line_range
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_regex
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewScrollTo::test_scroll_to_sets_state_open
toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py::TestVfsViewWorkflow::test_full_xy_workflow
```
