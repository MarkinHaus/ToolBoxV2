# FIX-17 – aus Karte K-API-02

Paket: **P2-AUTO-MEDIUM** (Mittlerer Fix, automatisch) · Fix-Typ: **A** · Karte: GETESTET · betrifft 46 Test-IDs

```
Fix:           FIX-17  (aus Karte K-API-02)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py, test_vfs_shell_tool.py, test_vfs_shell_bullet_proof.py, test_vfs_shared_store_edits.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_self_agents_default_tool_set.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_fetures.py (nur vfs_shell-Aufrufe)   (nur diese ändern)
Vorher lesen:  `vfs_shell_tool.py:437-450` und `:1188-1200`; in jeder Testdatei, wie `self.view`/`self.sh` gebaut werden
```

**Schritte:**

1. In die vier vfs-Testdateien vor der ersten Klasse einfügen:

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
2. Dort jedes `make_vfs_view(<arg>)` (nicht den Import) durch `_sync_tool(make_vfs_view(<arg>))` ersetzen.
3. `test_agent_session_tools.py` (async Tests): `self.view(...)`-Ergebnisse mit `await` holen; `self.sh("ls /")` → `self.sh("test", "ls /")` bzw. beim Bau von `self.sh` einen Wrapper `lambda cmd: sh("test", cmd)`.
4. `test_self_agents_default_tool_set.py` und der vfs_shell-Gesundheitscheck: Aufruf mit `reason="health"` ergänzen.

**Nicht tun:** Produktcode nicht ändern (Signatur ist gewollt, s. Docstring). Keine Assertion abschwächen. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 46 IDs der Karte grün, außer den 2, die zu K-API-09 wechseln (`…close_others_does_not_close_system_files`, `…close_others_preserves_system_files`).
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
