# FIX-28 – aus Karte K-API-08

Paket: **P3-VERALTET** (Test veraltet (Schnittstelle absichtlich geändert)) · Fix-Typ: **B** · Karte: UNGETESTET · betrifft 20 Test-IDs

> **Erst nach Antwort des Owners umsetzen.** Frage: Ist die neue Overlay-Struktur der Soll-Zustand? `'thought A' not found in ['thought Athought B']` könnte ein echter Fehler sein (Gedanken werden zusammengeklebt) – gewollt?

> Die Hypothese der Karte ist **ungetestet**. Schritt 1 ist immer: Repro aus der Karte ausführen und den Befund bestätigen. Stimmt er nicht: stoppen und melden.

```
Fix:           FIX-28  (aus Karte K-API-08)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/tests/test_isaa/test_icli_overlay.py, toolboxv2/tests/test_isaa/test_icli_logic.py   (nur diese ändern)
Vorher lesen:  `TaskOverlay`, `IterView` in `toolboxv2/flows/isaa/`
```

**Schritte:**

1. Tests auf die heutigen Felder/Methoden umschreiben.
2. Die 4 `icli_logic`-Fälle einzeln bewerten: Ausgabeformat geändert oder Fehler?

**Nicht tun:** – Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- 20 IDs grün; Bericht zu den 4 Logik-Fällen.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_isaa/test_icli_logic.py::TestIngestChunkTypeRouting::test_reasoning_appended_to_iter
toolboxv2/tests/test_isaa/test_icli_logic.py::TestRenderFooterToolbar::test_failed_task_shows_fail
toolboxv2/tests/test_isaa/test_icli_logic.py::TestRenderFooterToolbar::test_overflow_shown
toolboxv2/tests/test_isaa/test_icli_logic.py::TestToolResultInfo::test_non_json_string
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestEffectiveView::test_effective_view_returns_selected_task
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestEffectiveView::test_effective_view_sub_agent_own_view
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestEffectiveView::test_effective_view_sub_no_own_view_falls_back_to_parent
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestIngestChunkIter::test_tool_result_clears_pending
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestIngestChunkIter::test_tool_start_sets_pending
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestIterDrillDown::test_direct_iter_jump
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestIterDrillDown::test_drill_down_sets_selected_iter
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestIterDrillDown::test_jump_to_nonexistent_iter_no_change
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestOverlayInitState::test_init_defaults
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestOverlayInitState::test_left_items_multiple_tasks
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestOverlayInitState::test_left_items_task_only
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestOverlayInitState::test_left_items_with_sub_agents
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestOverlayNavigation::test_focus_toggle
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestOverlayNavigation::test_navigate_down_resets_iter_and_scroll
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestOverlayNavigation::test_navigate_down_wraps
toolboxv2/tests/test_isaa/test_icli_overlay.py::TestOverlayNavigation::test_navigate_up_wraps
```
