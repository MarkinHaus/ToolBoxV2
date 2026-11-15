# ISAA-Auftrag 06 – Eine Sitzungsdatei je Nutzer, mit Übernahme der alten Datei

> Komplexität: **nach Owner-Entscheidung – klein bis mittel, Produktpfad Login (WP1)** · Fix-Aufträge: FIX-19 · Test-IDs: 3
> Braucht keine Antwort des Owners. Unabhängig von den anderen Aufträgen (01–07); läuft parallel zu ihnen.
> Erstellt aus der WP2-CI-Analyse (Lauf 43742117802 auf `ebed4d6`, 12 Jobs; Stellen geprüft auf `master` @ `ed32daf`).

**Du bist eine ISAA-Instanz und arbeitest diesen Auftrag selbstständig ab.** Heute teilen sich alle Nutzer eines Geräts eine Sitzungsdatei (`user/_session.json`); startet ein anderer Nutzer, bekommt er die Anmeldung des vorigen. Der Owner hat entschieden: **eine Datei je Nutzer**. Bestehende Installationen werden beim ersten Start einmalig übernommen, damit nach dem Update niemand abgemeldet ist. Die Änderung ist auf `master` @ `ed32daf` getestet und liegt als Patch bei.

## 0. Rahmen – gilt für dich und für die anderen Instanzen

Bis zu sieben ISAA-Instanzen arbeiten gleichzeitig im Repository `MarkinHaus/ToolBoxV2`, alle direkt auf `master`, jede an **anderen Dateien**. Diese Datei ist dein vollständiger Auftrag; du brauchst keine weiteren Dokumente und keine Rückfragen. Die Analyse, aus der er stammt, liegt auf dem Branch `origin/claude/keen-goldberg-o59mt9` unter `docs/devdocs/ci_analysis/` (nur lesen, z. B. `git show origin/claude/keen-goldberg-o59mt9:docs/devdocs/ci_analysis/README.md`; nicht auschecken, nicht nach `master` übernehmen).

### 0.1 Eigene Arbeitskopie anlegen (einmal)

Jede Instanz hat ihren eigenen Ordner (git worktree) und ihre eigene venv. Den Haupt-Checkout des Owners benutzt du nicht.

```powershell
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2
git fetch origin
git worktree add -B isaa/06 C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa06 origin/master
cd C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa06
uv venv -p 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install pytest pytest-asyncio
uv pip install -e toolboxv2/mods/isaa
uv pip install -e .
$env:MOCK_EXTERNAL_SERVICES = "true"; $env:IS_OFFLINE_DB = "true"
python -c "import toolboxv2; print(toolboxv2.__file__)"   # muss in C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa06 liegen
```

`.venv` und `.venv310` stehen in `.gitignore` bzw. werden nie committet. In jeder neuen Shell: venv aktivieren und die zwei Umgebungsvariablen setzen.

### 0.2 Regeln

1. **Nur die Dateien aus Abschnitt 1 ändern.** Jede andere Datei gehört einer anderen Instanz oder ist gesperrt.
2. **Kein Grün durch Wegsehen:** kein `skip`, `skipIf`, `xfail`, `importorskip`, kein `continue-on-error`, kein `|| true`, keinen Test löschen, keine Assertion abschwächen, keinen Erwartungswert auf den Ist-Wert setzen, ohne dass der Auftrag es sagt.
3. **Ein Commit je Fix-Auftrag.** Dateien einzeln hinzufügen (`git add <pfad> <pfad>`), nie `git add -A`, `git add .` oder `git commit -a`. Vor jedem Commit `git status --short` ansehen: nur deine Dateien. Nie committen: `.venv*`, `docs/.docs_index.json`, Test-Artefakte, Logs.
4. **Kein `--force`, kein `--no-verify`, keine Merge-Commits auf `master`, kein Umschreiben fremder Commits.**
5. **Owner-Umgebung schützen:** Ports 8467 und 8468 nicht belegen, wenn die Owner-Instanz läuft; keine fremden Prozesse beenden; keine Daten unter dem Haupt-Checkout oder im Benutzerprofil löschen. Keine Geheimnisse (API-Schlüssel, Tokens) in Dateien, Commits oder den Bericht.
6. **Stoppen und melden**, statt weiterzumachen, wenn: die Abnahme rot bleibt; eine Datei außerhalb deiner Liste geändert werden müsste; `git rebase` einen Konflikt meldet; der Code an der genannten Stelle anders aussieht als im Auftrag beschrieben. Dann `git rebase --abort` bzw. Änderung zurücksetzen (`git checkout -- <pfad>`), den Auftrag als „gestoppt“ in den Bericht, weiter mit dem nächsten Auftrag.

### 0.3 Commit und Push nach `master`

Commit-Nachricht: erste Zeile `FIX-NN (Karte): kurze Beschreibung`, darunter `WP2-Analyse, Karte <Karte>.` und die letzte Zeile der Abnahme-Ausgabe.

```powershell
git fetch origin
git rebase origin/master          # holt die Commits der anderen Instanzen unter deinen Commit
# Gate-Check (0.4) erneut laufen lassen, wenn der Rebase fremde Commits gebracht hat
git push origin HEAD:master
```

Wird der Push abgelehnt (`non-fast-forward`): die drei Schritte wiederholen, höchstens dreimal. Danach melden.

### 0.4 Gate: keine neue rote Test-ID

Lege dieses Skript **außerhalb** des Repos ab, als `C:\Users\Markin\Workspace\ISAABOX\tb_gate.py` (einmal; alle Instanzen nutzen dieselbe Datei):

```python
# tb_gate.py - Baseline/Check fuer rote Test-IDs (WP2). Liegt AUSSERHALB des Repos.
import json, os, re, subprocess, sys
def run(paths):
    env = dict(os.environ, MOCK_EXTERNAL_SERVICES="true", IS_OFFLINE_DB="true")
    cmd = [sys.executable, "-m", "pytest", *paths, "-q", "-rfE", "--tb=no", "-p", "no:cacheprovider"]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    red = sorted({m.group(2).replace("\\", "/") for m in re.finditer(r"^(FAILED|ERROR) (\S+)", out, re.M)})
    summ = [l for l in out.splitlines() if re.search(r"\d+ (passed|failed|error)", l)]
    return red, (summ[-1] if summ else out[-400:])
mode, file, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
red, summ = run(paths)
print(summ)
if mode == "baseline":
    json.dump(red, open(file, "w", encoding="utf-8"), indent=0)
    print("Baseline gespeichert:", len(red), "rote IDs ->", file)
    sys.exit(0)
base = set(json.load(open(file, encoding="utf-8")))
new = [t for t in red if t not in base]
gone = [t for t in sorted(base) if t not in set(red)]
print(f"rot: {len(red)} | gruen geworden: {len(gone)} | NEU rot: {len(new)}")
for t in new:
    print("  NEU:", t)
sys.exit(1 if new else 0)
```

Dein Gate-Bereich:

```powershell
$GATE = @('toolboxv2/tests/test_mini/test_system', 'toolboxv2/tests/test_mini/test_cli_session.py', 'toolboxv2/tests/test_mini/test_toolboxv2.py')
```

**Vor der ersten Änderung** einmal die Baseline schreiben, **nach jedem Fix** den Check:

```powershell
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py baseline C:\Users\Markin\Workspace\ISAABOX\isaa06_baseline.json @GATE
python C:\Users\Markin\Workspace\ISAABOX\tb_gate.py check    C:\Users\Markin\Workspace\ISAABOX\isaa06_baseline.json @GATE   # Exit 0 = ok, 'NEU rot: 0'
```

`NEU rot` muss 0 sein. Erscheint nach einem `git rebase` eine neue rote ID, prüfe, ob sie von dir kommt:

1. `git switch --detach origin/master`, dann den Check erneut.
2. Auch dort rot → Ursache ist ein fremder Commit: ID im Bericht unter „fremd“ notieren, dort (auf `origin/master`) die Baseline neu schreiben, `git switch isaa/06`, weiterarbeiten.
3. Dort grün → Ursache bist du: `git switch isaa/06`, nicht pushen, Fehler suchen oder Fix zurücknehmen.

### 0.5 Bericht

Am Ende schreibst du `C:\Users\Markin\Workspace\ISAABOX\isaa06_bericht.md` (außerhalb des Repos) und gibst denselben Text als Antwort aus:

| Fix | Ergebnis (`erledigt` / `gestoppt` / `nur CI prüfbar`) | Commit | Abnahme (letzte Zeile, wörtlich) | grün gewordene IDs | Bemerkung |
|---|---|---|---|---|---|

Darunter: neue rote IDs (sollte leer sein), rote IDs aus fremden Commits, alles, was du gefunden, aber nicht geändert hast.

## Besonderheit dieses Auftrags

- `toolboxv2/tests/test_mini/test_system/test_session.py` ändert auch Auftrag 02 (`FIX-08`: Kopf der Datei und die `setUp`-/`tearDown`-Methoden). Du änderst **nur** die Zeilen 49, 59, 80 und 104. Git führt beides beim `git rebase` zusammen; meldet es einen Konflikt, stoppen (Regel 6).
- Vor FIX-08 sind in deinem Gate-Bereich zusätzlich bis zu 20 IDs rot (`/fake/dir`); das ist die Baseline, nicht dein Fehler.

## 1. Deine Dateien

- toolboxv2/utils/system/session.py; toolboxv2/tests/test_mini/test_system/test_session.py (nur Zeilen 49, 59, 80, 104); toolboxv2/tests/test_mini/test_cli_session.py (nur `test_long_username_truncated`)

**Gehören anderen Instanzen – nicht ändern:**

- Auftrag 01 (CI-Signal reparieren (Workflow-Dateien)): .github/workflows/build.yml (nur Job `py-check`); .github/workflows/release.yml (nur Jobs `build-windows-x64`, `build-macos-arm64`)
- Auftrag 02 (Minimale Fixes, alle vorab getestet): toolboxv2/tests/conftest.py; toolboxv2/tests/test_mini/test_system/test_session.py; toolboxv2/tests/test_clis/test_minicli_macros.py; toolboxv2/tests/test_clis/test_config_wizard.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_execution_engine.py; toolboxv2/tests/test_isaa/test_base/test_embeddings.py, toolboxv2/tests/test_isaa/test_base/test_llm_router.py, toolboxv2/tests/test_isaa/test_base/test_adapters.py; toolboxv2/utils/extras/mkdocs.py; toolboxv2/mods/isaa/base/ai_semantic_memory.py; toolboxv2/tests/test_core/test_port_unification.py; toolboxv2/tests/test_core/test_tray_daemon_integration.py
- Auftrag 03 (VFS-Werkzeuge: Tests und Health-Check an die neue Schnittstelle): toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeile 3613); toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_v2.py, test_vfs_shell_tool.py, test_vfs_shell_bullet_proof.py, test_vfs_shared_store_edits.py; toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_session_tools.py; toolboxv2/tests/test_mini/test_manifest/test_manifest_schema.py
- Auftrag 04 (Sporadische Tests stabilisieren): toolboxv2/tests/test_isaa/test_base/test_agent/test_vfs/test_vfs_pooling.py (nur die Klassen `TestPollIntegration` und `TestMultiAgentSync` bzw. die Tests mit festen Wartezeiten), toolboxv2/tests/test_mini/test_extras/test_analyser/test_profiler.py, toolboxv2/tests/test_mini/test_extras/test_install/test_rollback_manager.py, toolboxv2/tests/test_isaa/test_mem/test_hybrid_memory_realworld.py, toolboxv2/tests/test_isaa/test_base/test_audio/test_audio_core.py
- Auftrag 05 (Windows- und Python-3.10-Fehler: erst reproduzieren, dann beheben): installer.ps1 und toolboxv2/simple-core/src-tauri/installer.ps1 (müssen byte-gleich bleiben: nach jeder Änderung die eine Datei auf die andere kopieren, `fc.exe /b` muss „keine Unterschiede“ melden), scripts/verify_install_windows.ps1, toolboxv2/tests/test_core/test_installer_protocol.py, toolboxv2/tests/test_core/test_verify_install_windows.py; toolboxv2/mods/isaa/module.py (nur die Shell-Sitzung: Klasse um Zeilen 2470–2640 und `shell_tool_function` ab Zeile 2942); toolboxv2/mods/isaa/base/Agent/vfs_v2.py (`rmdir` Zeile 1473, `rm`, `mv`), toolboxv2/mods/isaa/base/patch/vfs_shell_tool.py (nur die Verzeichnisbefehle), toolboxv2/mods/CloudM/LiveSync/share_store.py; toolboxv2/tests/test_core/test_endpoint_dist_dir.py, toolboxv2/tests/test_isaa/test_codig/test_base_coder.py; toolboxv2/tests/test_isaa/test_providers/test_performance_and_limiter.py, toolboxv2/tests/test_isaa/test_base/test_agent/test_agent_performance_stability.py; toolboxv2/tests/test_core/test_job_execution_state.py; liegt die Ursache in ISAA-Produktcode unter Python 3.10 (Syntax/Typing): genau die Datei, die der Traceback nennt, sofern sie nicht in der Sperrliste von Abschnitt 1 steht – sonst stoppen und melden; toolboxv2/tests/test_mini/test_system/test_state_system.py und der Test, der die Variable hinterlässt (nur wenn er keiner anderen Instanz gehört)
- Auftrag 07 (Migration auf mcp 2 (Server, ISAA-Client, FlowAgent)): pyproject.toml (nur die mcp-Zeile im Extra `isaa`), toolboxv2/mods/isaa/pyproject.toml (nur die mcp-Zeile), toolboxv2/mcp_server/server.py, toolboxv2/mods/isaa/extras/mcp_session_manager.py, toolboxv2/mods/isaa/base/Agent/flow_agent.py (nur Zeilen 72, 429, 4518, 4529)
- Außerdem gesperrt (offene Owner-Fragen oder nicht Teil der Aufträge): `pyproject.toml` und `toolboxv2/mods/isaa/pyproject.toml` außer der mcp-Zeile (Auftrag 07), `toolboxv2/mcp_server/models.py`, `toolboxv2/tests/test_isaa/test_mcp/test_models.py`, `toolboxv2/utils/brodcast/`, `docs/`.

**Mit einer anderen Instanz geteilte Dateien (nur die genannten Zeilen):** `toolboxv2/tests/test_mini/test_system/test_session.py` – Auftrag 02 ändert Kopf und `setUp`/`tearDown`; du nur die Zeilen 49, 59, 80, 104.

## 2. Aufträge in dieser Reihenfolge

### FIX-19 – Karte K-API-13

**Ursache:** `Session._get_blob_name()` gibt immer `"user/_session.json"` zurück; der bereinigte Nutzername wird berechnet, aber nicht verwendet. Alle Nutzer eines Geräts teilen sich eine Sitzungsdatei.

**Beleg:** `toolboxv2/utils/system/session.py:119-125`: `safe_name = …` / `return f"user/_session.json"`, eingeführt mit `8dbf40b`. `BlobFile` verlangt einen Namen der Form `<blob_id>/<datei>` – `alice_session.json` (was die Tests erwarten) wirft `ValueError: Filename must include a path within the blob, e.g., 'blob_id/file.txt'`. CI: `'user/_session.json' != 'default_session.json'`, `!= 'user_evil_______name__session.json'`, `test_cli_session.py: 'default' not found in 'user/_session.json'`

**Warum:** Vermutlich wurde `{safe_name}_session.json` wegen dieses `ValueError` durch einen festen Namen ersetzt und der Nutzername ging dabei verloren. **Owner-Entscheidung 2026-11-15: eine Sitzungsdatei je Nutzer.** Weil `BlobFile` einen Ordner verlangt: `user/<name>_session.json`. Die App übergibt immer einen Namen (`toolbox.py:226` `Session(self.get_username())`, Vorgabe `"loot"`). Bestehende Installationen haben ihre Sitzung in `user/_session.json`; ohne Übernahme wären alle nach dem Update abgemeldet.

> In der Analyse getestet: Repro rot → Probe-Fix → grün, Nachbartests unverändert.

**Dateien (nur diese):** toolboxv2/utils/system/session.py; toolboxv2/tests/test_mini/test_system/test_session.py (nur Zeilen 49, 59, 80, 104); toolboxv2/tests/test_mini/test_cli_session.py (nur `test_long_username_truncated`)

**Vorher lesen:** `session.py:100-215`; `toolboxv2/utils/toolbox.py:220-230, 476-487`; `toolboxv2/utils/extras/blobs.py` (`BlobFile`, Namensregel)

**Repro (vor der Änderung, muss rot sein):**

```powershell
python -m pytest toolboxv2/tests/test_mini/test_system/test_session.py toolboxv2/tests/test_mini/test_cli_session.py toolboxv2/tests/test_mini/test_toolboxv2.py -q
```

Erwartet (Analyse): `8 failed, 28 passed` (3× K-API-13, 2× K-API-14 b/c, 3× K-API-14 a/d bzw. Netz). Szenario-Skript ohne Fix: `3 carol: {'access_token': 'B-TOKEN', 'username': 'bob'}` – **eine andere Nutzerin bekommt bobs Anmeldung**; `4 dave: None | user/_session.json`.

**Schritte:**

1. Patch aus diesem Auftrag anwenden (s. „Patch“). Gelingt `git apply --check` nicht, von Hand nach dem Patch.
2. Szenario-Prüfskript aus „Zusätzliche Abnahme“ ausführen.

**Nicht tun:** Die alte Datei nicht löschen, nur leeren (wie der Patch). Sitzung eines anderen Nutzers nie übernehmen. `test_login_valid_token_sets_session_valid` und `test_clear_session_token_deletes_blob_and_clears_attributes` nicht anpassen – offene Frage.

**Abnahme:** Repro → genau die 3 roten IDs aus „Nachher“; Szenario-Skript gibt die vier Zeilen aus „Nachher“ aus. Danach Gate-Check (0.4).

**Patch (getestet):** Speichere den Block unten als `C:\Users\Markin\Workspace\ISAABOX\isaa06.patch` (außerhalb des Repos, UTF-8; nicht mit `Out-File`/`>` aus Windows PowerShell 5.1, das schreibt UTF-16). Dann im Wurzelordner deiner Arbeitskopie:

```powershell
git fetch origin
git apply --3way --check C:\Users\Markin\Workspace\ISAABOX\isaa06.patch
git apply --3way C:\Users\Markin\Workspace\ISAABOX\isaa06.patch
git diff --cached --stat     # --3way legt die Änderung direkt in den Index
```

`--3way` nutzt die Blob-IDs im Patch und funktioniert auch mit CRLF-Zeilenenden in der Arbeitskopie. Meldet es einen Konflikt, hat sich eine Stelle seit `ed32daf` geändert: `git checkout -- .` und die Änderungen von Hand nach dem Patch übertragen; passt eine Stelle inhaltlich nicht mehr, stoppen (Regel 6).

<details><summary>Patch</summary>

```diff
diff --git a/toolboxv2/tests/test_mini/test_cli_session.py b/toolboxv2/tests/test_mini/test_cli_session.py
index dbecb36..c276a02 100644
--- a/toolboxv2/tests/test_mini/test_cli_session.py
+++ b/toolboxv2/tests/test_mini/test_cli_session.py
@@ -126,7 +126,8 @@ class TestSessionPathSafety(unittest.TestCase):
         sess = self._make_session("a" * 200)
         path = sess._get_blob_name()
         # Filename part (without extension) should be <= 32 + len("_session.json")
-        name_part = path.replace("_session.json", "")
+        # path is "user/<name>_session.json" (BlobFile needs "<blob_id>/<file>")
+        name_part = path.rsplit("/", 1)[-1].replace("_session.json", "")
         self.assertLessEqual(len(name_part), 32)
 
 
diff --git a/toolboxv2/tests/test_mini/test_system/test_session.py b/toolboxv2/tests/test_mini/test_system/test_session.py
index e5bc348..d63cf4d 100644
--- a/toolboxv2/tests/test_mini/test_system/test_session.py
+++ b/toolboxv2/tests/test_mini/test_system/test_session.py
@@ -46,7 +46,7 @@ class TestSessionStorage(unittest.TestCase):
         blob_name = self.sess._get_blob_name()
 
         # Assert
-        self.assertEqual(blob_name, "user_evil_______name__session.json")
+        self.assertEqual(blob_name, "user/user_evil_______name__session.json")
 
     def test_get_blob_name_empty_username_uses_default(self):
         # Arrange
@@ -56,7 +56,7 @@ class TestSessionStorage(unittest.TestCase):
         blob_name = self.sess._get_blob_name()
 
         # Assert
-        self.assertEqual(blob_name, "default_session.json")
+        self.assertEqual(blob_name, "user/default_session.json")
 
     @patch('toolboxv2.utils.system.session.BlobFile')
     @patch('toolboxv2.utils.system.session.Code')
@@ -77,7 +77,7 @@ class TestSessionStorage(unittest.TestCase):
         self.assertEqual(self.sess.access_token, "access_123")
 
         mock_blob_file.assert_called_with(
-            "testuser_session.json",
+            "user/testuser_session.json",
             mode="w",
             key="fake-device-key",
             storage=self.sess._storage
@@ -101,7 +101,7 @@ class TestSessionStorage(unittest.TestCase):
         }
 
         # Temporarily clear base to test restoration
-        del os.environ["TOOLBOXV2_REMOTE_BASE"]
+        os.environ.pop("TOOLBOXV2_REMOTE_BASE", None)
 
         # Act
         data = self.sess._load_session_token()
diff --git a/toolboxv2/utils/system/session.py b/toolboxv2/utils/system/session.py
index fa91905..1bae938 100644
--- a/toolboxv2/utils/system/session.py
+++ b/toolboxv2/utils/system/session.py
@@ -116,13 +116,41 @@ class Session(metaclass=Singleton):
             storage_directory=str(storage_path)
         )
 
+    # Until 2026-11 every user on a device shared this one file; read once and migrated.
+    _LEGACY_BLOB_NAME = "user/_session.json"
+
     def _get_blob_name(self) -> str:
-        """Determines the blob name (filename)."""
+        """Determines the blob name (filename): one session file per user."""
         # Default to 'default' if no username, allows basic session persistence without login
         safe_name = self.username or "default"
         # Sanitize filename
         safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe_name)[:32]
-        return f"user/_session.json"
+        # BlobFile needs "<blob_id>/<file>"; the blob is "user", one file per user inside it
+        return f"user/{safe_name}_session.json"
+
+    def _migrate_legacy_session(self) -> Optional[dict]:
+        """Move a session from the old device-wide file to this user's file (once).
+
+        Only when the old file belongs to this user (or has no owner recorded);
+        another user's old session is left untouched.
+        """
+        key = self._get_encryption_key()
+        try:
+            with BlobFile(self._LEGACY_BLOB_NAME, mode="r", key=key, storage=self._storage) as blob:
+                data = blob.read_json()
+        except Exception:
+            return None
+        if not data:
+            return None
+        owner = data.get("username") or ""
+        if self.username and owner and owner != self.username:
+            return None
+        with BlobFile(self._get_blob_name(), mode="w", key=key, storage=self._storage) as blob:
+            blob.write_json(data)
+        with BlobFile(self._LEGACY_BLOB_NAME, mode="w", key=key, storage=self._storage) as blob:
+            blob.write_json({})
+        get_logger().info(f"Session migrated from {self._LEGACY_BLOB_NAME} to {self._get_blob_name()}")
+        return data
 
     def _get_encryption_key(self) -> str:
         """
@@ -163,9 +191,14 @@ class Session(metaclass=Singleton):
         try:
             blob_name = self._get_blob_name()
 
-            with BlobFile(blob_name, mode="r", key=self._get_encryption_key(), storage=self._storage) as blob:
-                session_data = blob.read_json()
+            try:
+                with BlobFile(blob_name, mode="r", key=self._get_encryption_key(), storage=self._storage) as blob:
+                    session_data = blob.read_json()
+            except Exception:
+                session_data = None
 
+            if not session_data:
+                session_data = self._migrate_legacy_session()
             if not session_data:
                 return None
 
```
</details>

**Zusätzliche Abnahme (Prüfskript):** Speichere den Block als `C:\Users\Markin\Workspace\ISAABOX\isaa06_check_session.py` und führe ihn im Wurzelordner deiner Arbeitskopie aus (venv aktiv): `python C:\Users\Markin\Workspace\ISAABOX\isaa06_check_session.py`. Das Skript schreibt nur in ein Temp-Verzeichnis.

```python
"""ISAA-Auftrag 06, Abnahme: Sitzungsdatei je Nutzer und einmalige Übernahme der alten Datei.
Start im Repo-Wurzelordner:  python <pfad>\\isaa06_check_session.py
Schreibt nur in ein Temp-Verzeichnis (TB_DATA_DIR)."""
import os, sys, tempfile
os.environ["TB_DATA_DIR"] = tempfile.mkdtemp(prefix="isaa06_")
sys.path.insert(0, os.getcwd())
from toolboxv2.utils.singelton_class import Singleton
from toolboxv2.utils.system.session import Session
from toolboxv2.utils.extras.blobs import BlobFile
def fresh(name):
    Singleton._instances.pop(Session, None); Singleton._args.pop(Session, None)
    return Session(username=name, base="http://localhost:9")
def read(s, blob):
    try:
        with BlobFile(blob, mode="r", key=s._get_encryption_key(), storage=s._storage) as b: return b.read_json()
    except Exception as e: return {"access_token": f"<{type(e).__name__}>"}
def write(s, blob, data):
    with BlobFile(blob, mode="w", key=s._get_encryption_key(), storage=s._storage) as b: b.write_json(data)
s = fresh("alice"); write(s, "user/_session.json", {"access_token": "A-TOKEN", "username": "alice", "user_id": "u1"})
d = s._load_session_token()
print("1 alice:", (d or {}).get("access_token"), "| neue Datei:", read(s, "user/alice_session.json").get("access_token"), "| alte Datei:", read(s, "user/_session.json"))
s = fresh("alice"); print("2 alice erneut:", (s._load_session_token() or {}).get("access_token"))
s = fresh("carol"); write(s, "user/_session.json", {"access_token": "B-TOKEN", "username": "bob"})
print("3 carol:", s._load_session_token(), "| bobs alte Datei:", read(s, "user/_session.json").get("access_token"))
s = fresh("dave"); s._save_session_token("D-TOKEN", "r", "u4")
print("4 dave:", read(s, "user/dave_session.json").get("access_token"), "|", s._get_blob_name())
```

Erwartete Ausgabe nach dem Fix (aus der Analyse): `3 failed, 33 passed` – übrig: `test_clear_session_token_deletes_blob_and_clears_attributes`, `test_login_valid_token_sets_session_valid` (offene Frage K-API-14), `test_download_file_writes_chunks_to_disk` (braucht Netz). Szenario-Probe mit echter `BlobStorage`: (1) alte Datei von alice → alice lädt `A-TOKEN`, neue Datei angelegt, alte geleert; (2) zweiter Start lädt aus der neuen Datei; (3) alte Datei von bob, Nutzer carol → nichts übernommen, bobs Datei bleibt; (4) Speichern schreibt `user/dave_session.json`.

**Hinweis:** Betrifft WP1 (Anmeldung nach Update). Nach dem Rollout im Bericht festhalten, ob bei einem echten Update die Übernahme im Log erscheint (`Session migrated from user/_session.json to …`).

**Test-IDs (3):** in der CI unter Windows rot und damit lokal rot erwartet: 3; nur auf Linux/macOS rot: 0.

<details><summary>Liste</summary>

```
toolboxv2/tests/test_mini/test_cli_session.py::TestSessionPathSafety::test_empty_username_uses_default
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_get_blob_name_empty_username_uses_default
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_get_blob_name_sanitizes_special_characters
```
</details>

**Commit:** `FIX-19 (K-API-13): …`

## 3. Abschluss

1. Alle Commits gepusht (`git log origin/master --oneline -10` zeigt sie).
2. Bericht (0.5) geschrieben.
3. Arbeitskopie bleibt bestehen; der Owner entfernt sie mit `git worktree remove C:\Users\Markin\Workspace\ISAABOX\Toolboxv2-isaa06`.
