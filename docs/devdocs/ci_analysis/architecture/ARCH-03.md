# ARCH-03 — Optionale Abhängigkeiten und was die CI installiert

Karten: K-ENV-05 (26 IDs), verwandt K-ENV-02 (45, ungepinnt), K-ENV-03 (8, `waitress`), K-API-22 (teils Folge), K-OS-05 (echte Paketmanager).

## 1. Befund

`pyproject.toml` hat 20 Basis-Abhängigkeiten und die Extras `core`, `cli`, `web`, `desktop`, `isaa`, … Die CI installiert `.` **ohne Extras** plus das Paket `isaa` (eigene Abhängigkeiten) plus `pytest pytest-asyncio Cython`. Die Tests setzen aber voraus:

| Paket | steht in | Tests | CI-Meldung |
|---|---|---|---|
| `pyautogui` | Extra `desktop` | 12 (`test_agent_fetures.py`, `extras/destop_auto.py:18` importiert auf Modulebene) | `No module named 'pyautogui'` |
| `torch` | nirgends | 3 (`test_omni_p2_fixes.py`) | `No module named 'torch'` |
| `fastapi` | nirgends (weder `pyproject.toml` noch `mods/isaa/pyproject.toml`) | 5 (`test_icli_web.py`; `mods/icli_web/server.py:29`) | `No module named 'fastapi'` + Folgefehler |
| `litellm` | `mods/isaa/requirements.txt` (`litellm~=1.76.0`), nicht in `pyproject` | 6 (`test_vfs_view.py`, `test_agent_performance_stability.py`) | `<MagicMock …> does not have the attribute 'completion'/'acompletion'` (Stub aus `conftest2.py`) |
| `waitress` | Extra `web` | 8 (K-ENV-03) | `No module named 'waitress'` |
| `mcp` | Extra `isaa`, `>=1.6.0` ohne Obergrenze | 45 (K-ENV-02) | API von 2.x |

Es gibt keine Regel, was ein Test tun soll, wenn ein optionales Paket fehlt. Einige überspringen sich (`7 skipped` in `test_icli_web.py`), einige stubben das Paket global (ARCH-01), die meisten schlagen fehl. Außerdem ist nichts gepinnt: derselbe Commit kann morgen andere Fehler zeigen (mcp 2.x ist so entstanden).

## 2. Lösungswege

### Weg A — CI installiert die Extras, die getestet werden (klein bis mittel)

`uv pip install -e ".[web,isaa,cli]"` und was auf einem Runner ohne Bildschirm installierbar ist. Für `pyautogui` (braucht Display), `torch` (groß) bleibt Weg B.

- Aufwand **S** für `web`, `cli`, `isaa`; Risiko: längere Installation (faiss, playwright), neue Fehler durch echte statt gestubbte Pakete.
- Sofort als Typ A: `waitress` (FIX-04).

### Weg B — Feature-Marken mit ehrlichem Überspringen (mittel)

Jeder Test, der ein optionales Paket braucht, sagt das: `pytest.importorskip("pyautogui")` am Dateikopf oder Marke `@pytest.mark.needs("desktop")`. Die CI hat **je Profil einen Job**, in dem das Extra installiert ist und nichts übersprungen werden darf (`-p no:skipping`-Prüfung bzw. Zählung der Skips = 0 für diese Marke). Dann ist ein Skip kein Wegsehen: irgendwo läuft der Test sicher.

- Aufwand **M**: ~10 Testdateien markieren, Matrix um Profile erweitern (z. B. `core`, `isaa`, `desktop` nur unter Windows mit Display).
- Risiko: Regel 1 von WP2 verbietet `skip` als Fix. Dieser Weg ist nur zulässig, wenn der Owner ihn ausdrücklich als Regel beschließt **und** der Profil-Job existiert.
- Sofort als Typ A: nichts ohne Entscheidung.

### Weg C — Optionale Importe im Produkt träge machen (mittel, ergänzend)

`destop_auto.py` importiert `pyautogui` auf Modulebene; schon das Registrieren des Features scheitert. Wird der Import in die Funktion verschoben, laufen Registrierungs- und Schema-Tests ohne das Paket, nur die Ausführung braucht es. Dasselbe für `icli_web/server.py` (`fastapi`).

- Aufwand **M**, Produktcode in 2–4 Modulen. Risiko: Fehler erst zur Laufzeit statt beim Laden; dafür klare Meldung „Feature X braucht `toolboxv2[desktop]`“ (so macht es `litellm_fallback.py:25-28` schon).
- Sofort als Typ A: `destop_auto.py` nach Entscheidung.

### Unabhängig von A/B/C: Versionen festhalten

Eine Lock-Datei (`uv lock` / `uv pip compile`) für die CI, wöchentlich bewusst erneuert. Sofort als Typ A: `mcp<2` (FIX-03).

## 3. Experimente

| Weg | Experiment | Ergebnis |
|---|---|---|
| A | `waitress` nachinstalliert | 8 IDs grün (K-ENV-03) |
| A | `mcp<2` | 45 IDs grün (K-ENV-02) |
| A | `pyautogui`/`torch`/`fastapi` nachinstallieren | **nicht ausgeführt**: `pyautogui` braucht ein Display, `torch` ist ~2 GB; Aussagekraft für die CI gering, solange nicht entschieden ist, ob die CI sie haben soll |
| B, C | – | kein Experiment: hängt an der Regelentscheidung |

## 4. Fragen an den Owner

1. Welche Extras gehören zum „geprüften Kern“ (bei jedem Push, alle 12 Jobs)? Vorschlag: Basis + `web` + `cli` + `isaa`.
2. Darf es eine Regel „Test braucht Extra X → übersprungen, wenn X fehlt, aber ein Profil-Job mit X ist Pflicht“ geben (Weg B)? Ohne diese Regel bleibt nur Weg A.
3. Desktop-Automatisierung (`pyautogui`) und Audio (`torch`): in der CI testen (Windows-Job mit Display) oder nur lokal?
4. `litellm`: Abhängigkeit von ISAA (dann in `pyproject`), oder nur Fallback (dann dürfen Tests es nicht voraussetzen)?
5. Lock-Datei für die CI: ja/nein?
