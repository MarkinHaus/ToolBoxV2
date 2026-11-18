# Übergabe icli2 / isaa – Stand 18.11.2026

## Regeln (gelten immer)
- **`CLAUDE.md` im Repo-Wurzelordner:**
  - vorhandenen Code finden und verwenden
  - vorher Kaputtes mit beheben, fragen nur bei großem Eingriff, fehlender Info oder Widerspruch
  - Tests prüfen den echten Code, ohne Nachbau
  - Logging nur über `get_logger()`
- **Antworten und Kommentare auf Deutsch**, Code-Kommentare ohne Umlaute.
- **Nach jedem Schritt** gibt es einen Plain-Diff-`.patch`.
  - Er baut auf dem vorherigen auf.
  - Mit `git apply --check` gegen `origin/master` plus die vorherigen Patches prüfen.
  - Er liegt im Scratchpad, nicht im Repo.
  - Dazu ein Mini-Prompt zum Fortfahren.
- **Alles wird validiert**, wo möglich mit einem echten Durchlauf (e2e) gegen den isaa-Agenten mit dem Ersatz-LLM.
- **Subagenten** nur auf ausdrücklichen Wunsch, dann günstige mit genauen Anweisungen.
- **Branch:** `claude/gracious-rubin-a8256g`. Markin wendet die Patches selbst auf `master` an.

## Stand
- **Patches auf `master`, in dieser Reihenfolge:**
  - 16 Unterbefehle und Wachhund
  - 17 Hintergrund-Befehle, Job-Sessions und `/runs`
  - 18 Warteschlange, Debug-Hinweise und VFS-Index
  - 19 `icli2-19-irl-retry-persona.patch`: ersetzt die alten 19 und 20
- **Ob Markin 16–19 schon angewendet hat**, ist offen. Vor dem nächsten Patch `git fetch` und `origin/master` prüfen.
- **`PLAN.md` dokumentiert** die Schritte 1–19 und die Liste „Danach“.

## Offene Punkte (nur Befund und Code-Stellen – Ursache und Lösung erst nach Lesen klären)
1. **History-Kompaktierung:** Bei Markin 215k Eingabe-Tokens, `History (Work)` 206k, 917 Nachrichten, trotz Kompaktierung. Nachlesen:
   - `toolboxv2/mods/isaa/base/Agent/execution_engine.py:5575` `_maybe_compact_history` (Grenze, `_COMPACT_KEEP` in Zeile 5573)
   - Aufruf in `execution_engine.py:3540`
   - `execution_engine.py:1121` `HistoryCompressor.compress_partial`
   - `execution_engine.py:999` `HistoryCompressor` (Semantic Ledger ab Zeile 1009/1081)
   - Vor einer Änderung Markin fragen, was beim Verdichten verloren gehen darf.
2. **Persona im Lauf:** Narrator meldete „Persona-Injektion ignoriert — kein Moderationskontext“. Nachlesen:
   - `execution_engine.py:669` `PersonaRouter.route`, besonders ab Zeile 696 (gelernte Personas)
   - `execution_engine.py:758` `load_learned_personas` (Quelle: VFS `/global/.memory/dreamer/personas.json` auf Markins Rechner)
   - `execution_engine.py:5883` (Einspielen von `prompt_modifier`)
   - `narrator.py:773` `_set_thought`
3. **Blockierte Engine:** Bei Markin stand die Engine 294 s. Der Wachhund meldet weitere Stellen im Debug-Modus (`ICLI2_DEBUG=1`), der Stack steht im ToolBox-Log. Code: `icli2/app.py` `_watchdog`, `icli2/engine.py` `stack`/`culprit`.
4. **Stream-Abbruch nach dem ersten Chunk** wird nicht wiederholt: `toolboxv2/mods/isaa/base/llm_router/router.py:224`.
5. **Nicht gewählte Optionen:** `PLAN.md`, Schritt 17, „Offen“.
6. **Geplant, nicht begonnen:** `PLAN.md`, Abschnitt „Danach“.
7. **Rot wegen Umgebung (kein Netz):** `toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_view.py`, Klasse `TestVfsViewVisionSystem`.

## Werkzeuge
- **icli2-Prüfskripte:** `toolboxv2/mods/isaa/extras/icli2/tests/check_*.py`, darunter `check_ui.py` (~3 min), `check_watchdog.py`, `check_final.py`, `check_errors.py` und `check_termguard.py`.
- **pytest:**
  - `toolboxv2/tests/test_isaa/test_icli2_*.py`
  - `test_shell_background.py`
  - `test_vfs_path_index.py`
  - `test_base/test_llm_router_retry.py`
  - `test_base/test_agent/test_vfs`
- **Echter Durchlauf:** `tests/e2e/`. `fake_llm.py` startet auf Port 8765, die Dateien unter `/tmp/e2e` ablegen.
  - Hier vorher `local_embeddings.local_embed` durch Nullen ersetzen, sonst hängt der Lauf am Laden des Embedding-Modells (siehe `e2e_bg.py`).
- **Lange Läufe:** in den Hintergrund legen und mit einer Schleife auf eine Endmarke warten; nicht immer wieder abfragen.
