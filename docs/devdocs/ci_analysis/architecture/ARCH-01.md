# ARCH-01 — Globaler Zustand in den Tests: Event-Loop, `sys.modules`, Umgebungsvariablen

Karten: K-ASYNC-01 (160 IDs, Symptom per Fixture behebbar), K-STATE-02 (14), K-API-11 (8); verwandt K-STATE-01 (20), K-STATE-03 (1).
Nicht reparieren, sondern entscheiden. Der Typ-A-Teil ist unten je Weg genannt.

## 1. Befund

Der Lauf ist **ein Prozess für ~4850 Tests**. Drei Arten von Zustand überleben einen Test und treffen spätere:

| Zustand | Wer schreibt | Wer leidet | Beleg |
|---|---|---|---|
| aktuelle asyncio-Schleife | `tests/a_util.py:10-22` (`async_test` schließt seine Schleife), `tests/test_isaa/test_isaa_module.py:54-65` (`new_event_loop`/`set_event_loop`/`close`), jede `IsolatedAsyncioTestCase`, pytest-asyncio | 16 Testdateien mit `asyncio.get_event_loop().run_until_complete(...)` (Liste unten) | K-ASYNC-01: Paar-Lauf `test_isaa_module.py` + `test_job_manager.py` → 40× `Event loop is closed`; allein 107 passed |
| `sys.modules` | 13 Testdateien, am stärksten `tests/test_isaa/test_base/test_agent/conftest2.py:8-12` (ersetzt `toolboxv2.mods.isaa.base.Agent.types`, `…rule_set`, `litellm`, `python_a2a`, `mcp.server.fastmcp` beim **Import** durch `MagicMock`, nie zurückgesetzt) | `test_narrator.py` (9), `test_types.py` (4: `cannot import name 'Result' … (unknown location)`), alle Tests, die danach `litellm` oder `AgentModelData` echt brauchen | K-STATE-02: einzeln 100 passed bzw. 9 passed, im Gesamtlauf rot |
| `os.environ` | `test_session.py` (`TB_DATA_DIR`), Tests mit `del os.environ[...]` | `test_toolboxv2.py` (13 ERROR) | K-STATE-01, reproduziert |

Es gibt **drei verschiedene async-Hilfen** nebeneinander: `a_util.async_test` (eigene Schleife, schließt sie), lokale `async_run`/`_run`-Funktionen mit `get_event_loop()` (16 Dateien), `unittest.IsolatedAsyncioTestCase`, dazu pytest-asyncio ohne Konfiguration (Warnung `asyncio_default_fixture_loop_scope is unset` in jedem Lauf). Keine davon räumt für die anderen auf.

Dateien mit `asyncio.get_event_loop()`: `test_isaa/test_codig/test_coder_apply_back_hypotheses.py`, `test_isaa/test_jobs/{test_jobs_offline_validation,test_jobs_extended,test_job_manager}.py`, `test_isaa/test_icli_logic.py`, `test_isaa/test_base/test_agent/test_dreamer/test_dreamer_aggregator.py`, `…/test_honesty.py`, `…/test_vfs/test_vfs_session_tools.py`, `…/test_agent_fetures.py`, `test_isaa/test_base/test_audio/test_omni_p0_fixes.py`, `test_clis/test_minicli_macros.py`, `conftest.py`, `test_mini/test_cloudm/{test_extras,test_LiveSync/test_LiveSync_index,test_LiveSync/test_LiveSync_client,test_LiveSync/test_LiveSync_server}.py`, `test_mini/test_toolboxv2.py`.

Dateien mit Schreibzugriff auf `sys.modules`: `test_isaa/test_base/test_agent/{test_ruleset,test_exec_code_harness,conftest2,test_execution_engine}.py`, `test_isaa/test_base/test_without_litellm.py`, `test_isaa/test_base/test_audio/{test_omni_p1_fixes,test_omni_p0_fixes,test_omni_p2_fixes,test_omni_reconnect_audio}.py`, `test_isaa/test_mcp/test_flow_manager.py`, `test_clis/test_config_wizard.py`, `test_web/test_workers/test_fast/test_stubs.py`, `test_mini/test_cloudm/test_auth.py`.

Produktverhalten hängt nur indirekt daran: Bibliothekscode, der `asyncio.get_event_loop()` aufruft, hat dasselbe Problem in einer Anwendung, die vorher `asyncio.run()` benutzt hat (O8, doppelter Async-Helper in `SocketManager`/`EventManager`, ist dieselbe Familie).

## 2. Lösungswege

### Weg A — Wächter in `conftest.py` (klein, sofort)

Autouse-Fixtures stellen vor jedem Test den Grundzustand her: offene Schleife (FIX zu K-ASYNC-01), Schnappschuss von `sys.modules` und `os.environ` vor dem Test, Rücksetzen danach.

- Aufwand **S**. Betroffen: nur `tests/conftest.py`.
- Risiko: Rücksetzen von `sys.modules` nach jedem Test kann Tests brechen, die sich auf einen Import-Stub aus dem **Modulkopf** verlassen (`conftest2.py`): der Stub wäre nach dem ersten Test weg. Darum zuerst nur melden (Schritt 1), dann rücksetzen (Schritt 2).
- Migration: (1) Schleifen-Fixture (Typ A, getestet). (2) Fixture, die nach jedem Test neue/ersetzte `sys.modules`-Einträge **protokolliert**; ein Lauf zeigt die Verschmutzer je Test. (3) Rücksetzen einschalten, Verschmutzer einzeln auf `patch.dict(sys.modules, …)` umstellen.
- Sofort als Typ A: Schritt 1 (FIX zu K-ASYNC-01) und der `os.environ`-Schnappschuss.

### Weg B — Eine async-Hilfe, keine Import-Stubs (mittel)

Ein Helfer `a_util.run_async(coro)` (eigene Schleife je Aufruf, setzt die vorherige wieder), alle 16 Dateien nutzen ihn; `get_event_loop()` kommt in Tests nicht mehr vor. `conftest2.py` verliert die fünf `sys.modules`-Zeilen; Tests, die wirklich ohne `litellm` laufen sollen, nutzen `patch.dict(sys.modules, {"litellm": None})` im Test (so macht es `test_without_litellm.py` schon fast).

- Aufwand **M** (16 + 13 Dateien, mechanisch; `conftest2.py` hat 3 Nutzer).
- Risiko: ohne die Stubs laufen echte Importe; fehlen optionale Pakete, werden neue Fehler sichtbar (dann ARCH-03). `MockAgentModelData` muss der echten `AgentModelData` folgen (K-API-11: `web_config`).
- Migration: Datei für Datei; eine Prüfung im Gate (`grep` auf `get_event_loop()` und `sys.modules[` in `tests/`) verhindert Rückfall.
- Sofort als Typ A: die Hilfsfunktion anlegen; die 3 Nutzer von `conftest2.py` umstellen.

### Weg C — Prozess-Isolation (groß)

Verschmutzende Gruppen laufen in eigenen Prozessen: `pytest-xdist --dist loadfile` oder getrennte CI-Schritte je Ordner (`test_isaa`, `test_mini`, `test_core`, …). Die `serial`-Marke und der xdist-Hook in `conftest.py` sind schon vorbereitet.

- Aufwand **L** für sauberes Verhalten (Ports, gemeinsame `.data`, `TB_DATA_DIR` je Worker), **S** für „je Ordner ein Schritt“.
- Risiko: verdeckt Verschmutzung, statt sie zu beheben; innerhalb einer Datei bleibt sie. Nutzen: Laufzeit (heute 19–48 Minuten je Job).
- Sofort als Typ A: nichts, was die Karten löst.

## 3. Experimente

| Weg | Experiment | Ergebnis |
|---|---|---|
| A | Schleifen-Fixture im Worktree, Paar- und Ordnerläufe | 160 IDs: Repro rot → grün (Karte K-ASYNC-01); Lauf über die 77 betroffenen Dateien s. README Abschnitt 6 |
| A | `/fake` → Temp-Verzeichnis + `addCleanup` | 20 IDs grün (Karte K-STATE-01) |
| B | `_sync_tool`-Wrapper (eigene Schleife je Aufruf) in 4 vfs-Testdateien | 33 von 36 grün (Karte K-API-02) – zeigt, dass das Muster „eigene Schleife je Aufruf“ trägt |
| B | `conftest2.py` ohne Stubs | **nicht ausgeführt**: braucht zuerst die Antwort zu ARCH-03 (ist `litellm` in der Testumgebung installiert?) |
| C | – | kein Experiment: xdist ist in der CI nicht installiert, und der Weg behebt keine Karte |

## 4. Fragen an den Owner

1. Reicht dir Weg A als Dauerlösung für die Schleife, oder soll `get_event_loop()` aus den Tests verschwinden (Weg B)?
2. Dürfen Tests Module beim Import global ersetzen? Vorschlag: nein, nur `patch.dict(sys.modules, …)` innerhalb eines Tests; Gate prüft es.
3. `conftest2.py`: der Kommentar sagt „Da toolboxv2 evtl. nicht installiert ist, mocken wir die Importe vorab“. Gilt das noch? In der CI ist `toolboxv2` immer installiert.
4. Soll die CI je Ordner getrennt laufen (kürzere Jobs, klarere Zuordnung), unabhängig von A/B?
