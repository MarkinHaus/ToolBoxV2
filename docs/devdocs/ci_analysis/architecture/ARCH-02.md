# ARCH-02 — Zwei Stände von `toolboxv2` und ISAA als eigenes Paket mit Top-Level-Namen

Karten: K-ENV-04 (1 ID, O5), K-MISS-02 (9 IDs, Folge), K-ENV-03 (8 IDs, verwandt). Betrifft WP1 (was wird beim Nutzer installiert?).

## 1. Befund

Die CI installiert so (`build.yml:143-147`):

```
uv pip install -e toolboxv2/mods/isaa --system     # Paket "isaa", editierbar
uv pip install . --system                           # Paket "toolboxv2", NICHT editierbar
```

Daraus entstehen drei Dinge, alle in dieser Analyse gemessen:

1. **Zwei Stände von `toolboxv2`.** Aus dem Checkout gestartet (`python -m pytest`, cwd zuerst im Suchpfad) wird `./toolboxv2` importiert. Jeder Prozess mit anderem Startverzeichnis oder Skriptordner importiert `site-packages/toolboxv2`:
   `cd /tmp && python -c "import toolboxv2; print(toolboxv2.__file__)"` → `…/site-packages/toolboxv2/__init__.py`.
2. **Die installierte Kopie hat kein ISAA.** `pyproject.toml:89` schließt `toolboxv2.mods.[!CDW]*` aus. `from toolboxv2.mods.isaa.module import …` in einem Unterprozess → `ModuleNotFoundError: No module named 'toolboxv2.mods.isaa'` (K-ENV-04, der O5-Test).
3. **Das Paket `isaa` legt fremde Top-Level-Namen an.** `toolboxv2/mods/isaa/pyproject.toml:46-48`: `include = ["base*", "CodingAgent*", "extras*"]`. Die editierbare Installation trägt in die Umgebung ein: `base`, `extras`, `CodingAgent` (gemessen in `__editable___isaa_0_2_0_finder.py`, `MAPPING = {'CodingAgent': …, 'base': …, 'extras': …}`). ISAA ist also als `base.…` und `extras.…` importierbar, **nicht** als `toolboxv2.mods.isaa.…`. Die Namen `base` und `extras` sind so allgemein, dass sie mit anderen Paketen kollidieren können. Die Tests mit `patch("llm_router.…")` (K-MISS-02) stammen aus derselben Denkweise.

Nebenbefunde:
- Tests schreiben in den Checkout und in die Umgebung: nach einem Lauf liegt `docs/.docs_index.json` neu im Repo; in `site-packages/toolboxv2/` bleibt nach Deinstallation ein Ordner `dist` zurück, der danach eine editierbare Installation als leeres Namespace-Paket verdeckt (`toolboxv2.__file__ == None`, `cannot import name 'FileHandler' from 'toolboxv2' (unknown location)`). Das ist dasselbe Fehlerbild wie in `test_types.py` (K-STATE-02).
- `fast_tb.serve()` braucht `waitress`, das nur im Extra `web` steht (K-ENV-03).

Aufrufkette O5: `test_shell_tool.py:405-422` → `subprocess [python -c launcher]` → `python -c wrapper` → `python /tmp/…/harness.py` → `import toolboxv2` (site-packages) → `toolboxv2.mods.isaa` fehlt → Exit 1 → `launcher chain was killed`.

## 2. Lösungswege

### Weg A — CI installiert editierbar (klein, sofort)

`uv pip install -e . --system`. Dann gibt es genau einen Stand, der Checkout.

- Aufwand **S**, eine Zeile (FIX-02). Risiko: die CI prüft dann nicht mehr, ob das **gebaute Paket** vollständig ist (fehlende `package-data`, ausgeschlossene Module fallen nicht auf). Das prüft heute aber auch niemand gezielt.
- Sofort als Typ A: ja, vollständig.

### Weg B — Zwei getrennte Prüfungen (mittel)

(1) Entwicklungs-Tests editierbar (Weg A). (2) Ein eigener, kleiner Job „Paket-Rauchtest“: Wheel bauen, in leere venv installieren, **außerhalb** des Checkouts `python -c "import toolboxv2; …"`, `tb --version`, lokale Oberfläche starten, `/live/health` abfragen. Das ist die Prüfung, die dem Nutzer-Stand (SimpleCore, WP1) entspricht, und sie hätte `waitress` (K-ENV-03) gefunden.

- Aufwand **M** (ein Job, ein Skript; `scripts/verify_install_windows.ps1` macht Ähnliches für Windows).
- Risiko: gering. Betroffen: `build.yml`, ein neues Skript unter `scripts/`.
- Sofort als Typ A: Teil (1).

### Weg C — ISAA als echtes Unterpaket von `toolboxv2` (groß)

`toolboxv2/mods/isaa/pyproject.toml` entfällt oder liefert das Paket unter dem Namen `toolboxv2.mods.isaa` (kein `base`, `extras` auf oberster Ebene). ISAA-Abhängigkeiten stehen nur im Extra `toolboxv2[isaa]`. Der Ausschluss `toolboxv2.mods.[!CDW]*` wird für `isaa` aufgehoben oder ISAA wird ein Namespace-Teilpaket.

- Aufwand **L**: Paketgrenzen, Installer (`ToolBoxV2[cli,web,isaa]`), Feature-Pakete (`features_packed/*.zip`), Nuitka-Build.
- Risiko: hoch für WP1 (Installation/Update), darum nicht parallel dazu.
- Sofort als Typ A: nur die Prüfung, ob irgendwo `import base.` / `import extras.` benutzt wird.

## 3. Experimente

| Weg | Experiment | Ergebnis |
|---|---|---|
| A | frische venv, `-e toolboxv2/mods/isaa` + `-e .`, `test_shell_tool.py` | `44 passed` (vorher 1 failed); `toolboxv2.__file__` zeigt aus jedem Verzeichnis auf den Checkout |
| A | dieselbe Umgebung **nach** einer nicht editierbaren Installation (Rest-Ordner `site-packages/toolboxv2/dist`) | editierbare Installation defekt (`__file__ None`) → in der CI unkritisch (frischer Runner), auf Entwicklerrechnern eine Falle |
| B | – | nicht ausgeführt (neuer Job, Workflow-Änderung) |
| C | `grep` im editierbaren Finder | belegt die Top-Level-Namen; Umbau nicht versucht |

## 4. Fragen an den Owner

1. Wie installiert der SimpleCore-Installer: Wheel mit Extras (`ToolBoxV2[cli,web,isaa] @ file://…` laut `test_installer_protocol`)? Ist ISAA dort über das Extra oder über Feature-Pakete dabei? Davon hängt ab, ob Befund 2 nur die CI betrifft.
2. Ist die Installation von `isaa` als eigenes Paket mit `base`/`extras` auf oberster Ebene gewollt (z. B. für eigenständige Nutzung), oder ein Überbleibsel?
3. Weg B: Soll der Paket-Rauchtest in `build.yml` (jeder Push) oder nur vor einer Veröffentlichung laufen?
