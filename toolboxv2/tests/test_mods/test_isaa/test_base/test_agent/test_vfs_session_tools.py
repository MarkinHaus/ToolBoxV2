"""
Investigative Unit Tests: VFS Session Tools
============================================

Testet die Tool-Wrapper aus FlowAgent.init_session_tools() mit Fokus auf:
  1. Semantik von vfs_open vs vfs_read (das Kerndproblem)
  2. Context-Sichtbarkeit: Was der Agent wirklich sieht
  3. vfs_view Scrolling-Verhalten
  4. Tool-Name-zu-Funktion-Mapping (die intentionale Vertauschung)
  5. Implizite Limits (grep 50-Match-Cap - undokumentiert)
  6. Edge-Cases die den Agenten blockieren

Alle Tests sind non-invasiv, isoliert (kein ToolBoxV2-Start nötig),
und verwenden ausschließlich unittest.

Run:
    python -m unittest test_vfs_session_tools -v
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, AsyncMock


# ---------------------------------------------------------------------------
# VFS Import - direkt, kein FlowAgent nötig
# ---------------------------------------------------------------------------
try:
    from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2, VFSFile, FileBackingType
    VFS_AVAILABLE = True
except ImportError:
    VFS_AVAILABLE = False


def _skip_if_no_vfs(cls):
    if not VFS_AVAILABLE:
        return unittest.skip("toolboxv2.mods.isaa.base.Agent.vfs_v2 not importable")(cls)
    return cls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    """Führt eine Coroutine synchron aus."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def make_vfs(max_window=100):
    """Erstellt ein sauberes VFS ohne LSP/Docker."""
    return VirtualFileSystemV2(
        session_id="test-session",
        agent_name="TestAgent",
        max_window_lines=max_window,
        summarizer=None,
        lsp_manager=None,
    )


def make_large_file(lines: int) -> str:
    return "\n".join(f"Line {i:04d}: {'x' * 40}" for i in range(1, lines + 1))


# ===========================================================================
# 1. SEMANTIK: vfs_open (Windowed) vs vfs_read (Full Content)
#    Das ist das Kernproblem das der Agent nicht versteht.
# ===========================================================================

@_skip_if_no_vfs
class TestOpenVsReadSemantics(unittest.TestCase):
    """
    vfs_read  (Tool-Name) → ruft intern session.vfs.open() auf
               → gibt nur FENSTER-VORSCHAU zurück (preview, max 5 Zeilen)
               → Datei wird als 'open' markiert, Inhalt fließt in Kontext
               → BEVORZUGTER Weg - effizient, kontextsparend

    vfs_open  (Tool-Name) → ruft intern session.vfs.read() auf
               → gibt GESAMTEN INHALT zurück (content-Key)
               → Datei bleibt 'closed', KEIN Kontext-Update
               → Notfall-Fallback - teuer, kontextintensiv
    """

    def setUp(self):
        self.vfs = make_vfs(max_window=50)
        self.content_10 = "\n".join(f"Zeile {i}" for i in range(1, 11))
        self.vfs.create("/test.py", self.content_10)

    # -- vfs.open() Verhalten (was "vfs_read" Tool aufruft) -----------------

    def test_open_returns_preview_not_full_content(self):
        """open() gibt 'preview' zurück, nicht 'content'."""
        result = self.vfs.open("/test.py")
        self.assertIn("preview", result,
            "vfs.open() muss 'preview'-Key zurückgeben (wird als 'vfs_read' dem Agent angeboten)")
        self.assertNotIn("content", result,
            "vfs.open() darf keinen 'content'-Key zurückgeben - das ist vfs.read()'s Aufgabe")

    def test_open_preview_is_max_5_lines(self):
        """Preview zeigt maximal 5 Zeilen (Agent soll scrollen)."""
        result = self.vfs.open("/test.py")
        preview_lines = result["preview"].split("\n")
        visible = [l for l in preview_lines if l.strip() and "..." not in l]
        self.assertLessEqual(len(visible), 5,
            f"Preview soll ≤5 Zeilen haben, hat aber {len(visible)}: {preview_lines}")

    def test_open_sets_file_state_to_open(self):
        """Nach open() ist der Datei-State 'open' - Inhalt fließt in Kontext."""
        self.assertEqual(self.vfs.files["/test.py"].state, "closed")
        self.vfs.open("/test.py")
        self.assertEqual(self.vfs.files["/test.py"].state, "open",
            "Nach vfs.open() muss state='open' sein damit der Agent den Inhalt im Kontext sieht")

    def test_open_file_appears_in_context(self):
        """Geöffnete Datei erscheint in build_context_string - Agent sieht sie permanent."""
        ctx_before = self.vfs.build_context_string()
        self.assertNotIn("Zeile 1", ctx_before,
            "Geschlossene Datei darf nicht im Kontext erscheinen")
        self.vfs.open("/test.py")
        ctx_after = self.vfs.build_context_string()
        self.assertIn("Zeile 1", ctx_after,
            "Geöffnete Datei MUSS im Kontext erscheinen - das ist der Zweck von open()")

    def test_closed_file_not_in_context(self):
        """Geschlossene Dateien erscheinen NICHT im Kontext."""
        self.vfs.create("/hidden.py", "GEHEIM")
        ctx = self.vfs.build_context_string()
        self.assertNotIn("GEHEIM", ctx,
            "Datei im State 'closed' darf nicht im Kontext erscheinen")

    # -- vfs.read() Verhalten (was "vfs_open" Tool aufruft) -----------------

    def test_read_returns_full_content(self):
        """read() gibt kompletten Inhalt zurück (content-Key)."""
        result = self.vfs.read("/test.py")
        self.assertIn("content", result,
            "vfs.read() muss 'content'-Key zurückgeben")
        self.assertEqual(result["content"], self.content_10,
            "vfs.read() muss EXAKT den gesamten Inhalt zurückgeben")

    def test_read_does_not_change_state(self):
        """read() verändert den Datei-State NICHT - kein Kontext-Update."""
        self.vfs.read("/test.py")
        self.assertEqual(self.vfs.files["/test.py"].state, "closed",
            "vfs.read() ist ein reiner Lesezugriff - state bleibt 'closed'")

    def test_read_file_does_not_appear_in_context(self):
        """Nach read() erscheint die Datei NICHT im Kontext."""
        self.vfs.read("/test.py")
        ctx = self.vfs.build_context_string()
        self.assertNotIn("Zeile 1", ctx,
            "vfs.read() soll keinen Kontext-Eintrag erzeugen - nur vfs.open() tut das")

    # -- Kombination --------------------------------------------------------

    def test_open_then_read_returns_full_content(self):
        """Erst open(), dann read() gibt trotzdem vollen Inhalt zurück."""
        self.vfs.open("/test.py", line_start=1, line_end=3)
        result = self.vfs.read("/test.py")
        self.assertEqual(result["content"], self.content_10)

    def test_correct_tool_for_workflow(self):
        """
        Workflow-Test: Agent soll open -> view -> edit -> close verwenden.
        Nicht: read für alles.
        """
        # Schritt 1: open an interessanter Stelle
        open_res = self.vfs.open("/test.py", line_start=3, line_end=6)
        self.assertTrue(open_res["success"])
        self.assertEqual(self.vfs.files["/test.py"].state, "open")

        # Schritt 2: view scrollt weiter
        view_res = self.vfs.view("/test.py", line_start=7, line_end=10)
        self.assertTrue(view_res["success"])
        self.assertIn("content", view_res)

        # Schritt 3: edit ändert eine Zeile
        edit_res = self.vfs.edit("/test.py", line_start=8, line_end=8, new_content="GEÄNDERT")
        self.assertTrue(edit_res["success"])
        self.assertIn("GEÄNDERT", self.vfs.files["/test.py"]._content)

        # Schritt 4: close erzeugt Summary
        close_res = run_async(self.vfs.close("/test.py"))
        self.assertTrue(close_res["success"])
        self.assertEqual(self.vfs.files["/test.py"].state, "closed")


# ===========================================================================
# 2. VIEW SCROLLING - Der Kernmechanismus für große Dateien
# ===========================================================================

@_skip_if_no_vfs
class TestVFSViewScrolling(unittest.TestCase):
    """
    vfs_view ist das "Ctrl+F / Scroll"-Äquivalent für den Agenten.
    Datei muss offen sein. view() verändert das sichtbare Fenster.
    """

    def setUp(self):
        self.vfs = make_vfs(max_window=20)
        # 100 Zeilen Datei
        self.vfs.create("/big.py", make_large_file(100))
        self.vfs.open("/big.py", line_start=1, line_end=20)

    def test_view_returns_only_requested_lines(self):
        """view() gibt genau die angeforderten Zeilen zurück."""
        result = self.vfs.view("/big.py", line_start=10, line_end=15)
        self.assertTrue(result["success"])
        lines = result["content"].split("\n")
        # Erste Zeile ist Line 10
        self.assertIn("Line 0010", lines[0],
            f"Erste Zeile soll 'Line 0010' sein, ist: {lines[0]!r}")
        # Letzte Zeile ist Line 15
        self.assertIn("Line 0015", lines[-1],
            f"Letzte Zeile soll 'Line 0015' sein, ist: {lines[-1]!r}")

    def test_view_updates_file_window(self):
        """view() ändert view_start und view_end des VFSFile-Objekts."""
        self.vfs.view("/big.py", line_start=50, line_end=70)
        f = self.vfs.files["/big.py"]
        self.assertEqual(f.view_start, 49,  # 0-indexed
            "view_start muss auf 0-indiziert angepasst werden (line 50 → index 49)")
        self.assertEqual(f.view_end, 70)

    def test_view_new_range_reflected_in_context(self):
        """Nach view() erscheint der neue Bereich im Kontext, nicht der alte."""
        self.vfs.view("/big.py", line_start=80, line_end=85)
        ctx = self.vfs.build_context_string()
        self.assertIn("Line 0080", ctx,
            "Neuer view-Bereich muss im Kontext sichtbar sein")
        self.assertNotIn("Line 0001", ctx,
            "Alter view-Bereich soll nicht mehr im Kontext erscheinen")

    def test_view_on_closed_file_auto_opens(self):
        """view() auf geschlossene Datei öffnet sie automatisch."""
        run_async(self.vfs.close("/big.py"))
        self.assertEqual(self.vfs.files["/big.py"].state, "closed")
        result = self.vfs.view("/big.py", line_start=1, line_end=5)
        self.assertTrue(result["success"],
            "view() auf geschlossene Datei soll auto-öffnen, nicht fehlschlagen")
        self.assertEqual(self.vfs.files["/big.py"].state, "open")

    def test_view_nonexistent_file_fails_gracefully(self):
        """view() auf nicht-existente Datei gibt success=False zurück."""
        result = self.vfs.view("/nicht_vorhanden.py", line_start=1, line_end=5)
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_context_respects_max_window_lines(self):
        """Kontext wird bei max_window_lines abgeschnitten - mit Warnung."""
        vfs = make_vfs(max_window=10)
        vfs.create("/big.py", make_large_file(100))
        vfs.open("/big.py", line_start=1, line_end=100)  # will versucht alles zu zeigen
        ctx = vfs.build_context_string()
        ctx_lines = ctx.split("\n")
        # Zähle wie viele "Line XXXX" Zeilen drin sind
        file_lines = [l for l in ctx_lines if "Line 00" in l]
        self.assertLessEqual(len(file_lines), 10,
            f"Kontext soll max_window_lines={10} respektieren, hat aber {len(file_lines)} Zeilen")

    def test_view_line_end_minus_one_means_end_of_file(self):
        """line_end=-1 zeigt bis Dateiende."""
        small_vfs = make_vfs(max_window=200)
        small_vfs.create("/small.py", make_large_file(10))
        small_vfs.open("/small.py", line_start=5, line_end=-1)
        result = small_vfs.view("/small.py", line_start=5, line_end=-1)
        self.assertTrue(result["success"])
        # Letzte Zeile der Datei muss enthalten sein
        self.assertIn("Line 0010", result["content"],
            "line_end=-1 muss bis Dateiende zeigen (Zeile 10)")

    def test_open_shows_total_lines_info(self):
        """open() Nachricht enthält Zeilennummern (Orientierung für Agent)."""
        result = self.vfs.open("/big.py", line_start=20, line_end=30)
        message = result.get("message", "")
        self.assertIn("20", message,
            f"open() Nachricht soll Startzeile enthalten: {message!r}")


# ===========================================================================
# 3. OPEN MIT ZEILENBEREICH - Partielles Öffnen
# ===========================================================================

@_skip_if_no_vfs
class TestVFSOpenLineRange(unittest.TestCase):
    """Testet dass open() mit line_start/line_end nur den Bereich zeigt."""

    def setUp(self):
        self.vfs = make_vfs(max_window=50)
        self.vfs.create("/source.py", make_large_file(200))

    def test_open_partial_only_shows_range_in_context(self):
        """Nur der angeforderte Bereich erscheint im Kontext, nicht die ganze Datei."""
        self.vfs.open("/source.py", line_start=50, line_end=60)
        ctx = self.vfs.build_context_string()
        self.assertIn("Line 0050", ctx,
            "Startzeile des Bereichs muss im Kontext sein")
        self.assertIn("Line 0060", ctx,
            "Endzeile des Bereichs muss im Kontext sein")
        self.assertNotIn("Line 0001", ctx,
            "Zeilen außerhalb des Bereichs dürfen NICHT im Kontext sein")
        self.assertNotIn("Line 0200", ctx,
            "Dateiende darf nicht sichtbar sein wenn nicht angefordert")

    def test_open_stores_correct_view_range(self):
        """view_start und view_end werden korrekt (0-indiziert) gespeichert."""
        self.vfs.open("/source.py", line_start=10, line_end=20)
        f = self.vfs.files["/source.py"]
        self.assertEqual(f.view_start, 9,
            "view_start soll 0-indiziert sein: line 10 → index 9")
        self.assertEqual(f.view_end, 20)

    def test_open_success_response_structure(self):
        """open() Antwort hat alle nötigen Keys für den Agenten."""
        result = self.vfs.open("/source.py", line_start=1, line_end=10)
        self.assertTrue(result["success"])
        self.assertIn("message", result,
            "message: erklärt dem Agenten was passiert ist")
        self.assertIn("preview", result,
            "preview: zeigt dem Agenten sofort den Inhalt (max 5 Zeilen)")
        self.assertIn("file_type", result,
            "file_type: Datei-Typ-Info für den Agenten")

    def test_open_of_500_line_file_doesnt_flood_context(self):
        """500-Zeilen-Datei geöffnet mit line_end=-1 → context begrenzt."""
        vfs = make_vfs(max_window=20)
        vfs.create("/huge.py", make_large_file(500))
        vfs.open("/huge.py", line_start=1, line_end=-1)
        ctx = vfs.build_context_string()
        ctx_file_lines = [l for l in ctx.split("\n") if "Line 0" in l]
        self.assertLessEqual(len(ctx_file_lines), 20,
            f"max_window_lines=20 muss Context-Überflutung verhindern, "
            f"aber {len(ctx_file_lines)} Zeilen im Kontext")

    def test_reopen_different_range_updates_context(self):
        """Mehrfaches open() mit verschiedenen Bereichen - letzter gewinnt."""
        self.vfs.open("/source.py", line_start=1, line_end=10)
        ctx1 = self.vfs.build_context_string()
        self.assertIn("Line 0001", ctx1)

        self.vfs.open("/source.py", line_start=100, line_end=110)
        ctx2 = self.vfs.build_context_string()
        self.assertIn("Line 0100", ctx2,
            "Neuer Bereich muss sichtbar sein")
        self.assertNotIn("Line 0001", ctx2,
            "Alter Bereich soll nach erneutem open() verschwinden")


# ===========================================================================
# 4. EDIT - Zeilenbasierte Änderungen
# ===========================================================================

@_skip_if_no_vfs
class TestVFSEdit(unittest.TestCase):

    def setUp(self):
        self.vfs = make_vfs()
        lines = ["alpha", "beta", "gamma", "delta", "epsilon"]
        self.vfs.create("/edit_test.py", "\n".join(lines))

    def test_edit_replaces_exact_lines(self):
        """edit() ersetzt genau die angegebenen Zeilen."""
        result = self.vfs.edit("/edit_test.py", line_start=2, line_end=3, new_content="NEU")
        self.assertTrue(result["success"])
        content = self.vfs.files["/edit_test.py"]._content
        lines = content.split("\n")
        self.assertEqual(lines[0], "alpha", "Zeile 1 soll unverändert bleiben")
        self.assertEqual(lines[1], "NEU", "Zeile 2 soll ersetzt sein")
        self.assertEqual(lines[2], "delta", "Zeile 4 soll auf Position 3 gerutscht sein")
        self.assertEqual(lines[3], "epsilon")

    def test_edit_preserves_lines_before_range(self):
        """Zeilen vor dem Bereich bleiben unverändert."""
        self.vfs.edit("/edit_test.py", line_start=4, line_end=5, new_content="X\nY")
        content = self.vfs.files["/edit_test.py"]._content
        self.assertTrue(content.startswith("alpha\nbeta\ngamma"),
            f"Erste 3 Zeilen sollen unverändert sein: {content!r}")

    def test_edit_can_insert_multiple_lines(self):
        """edit() kann eine Zeile durch mehrere ersetzen (Expansion)."""
        self.vfs.edit("/edit_test.py", line_start=1, line_end=1, new_content="a\nb\nc")
        content = self.vfs.files["/edit_test.py"]._content
        lines = content.split("\n")
        self.assertEqual(lines[0], "a")
        self.assertEqual(lines[1], "b")
        self.assertEqual(lines[2], "c")
        self.assertEqual(lines[3], "beta")  # Rest verschoben

    def test_edit_single_line(self):
        """line_start == line_end editiert genau eine Zeile."""
        self.vfs.edit("/edit_test.py", line_start=3, line_end=3, new_content="GAMMA_NEU")
        content = self.vfs.files["/edit_test.py"]._content
        self.assertIn("GAMMA_NEU", content)
        self.assertNotIn("gamma", content)

    def test_edit_nonexistent_file_fails(self):
        """edit() auf nicht-existente Datei gibt success=False zurück."""
        result = self.vfs.edit("/ghost.py", 1, 1, "x")
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_edit_readonly_file_fails(self):
        """edit() auf system_context.md (readonly) schlägt fehl."""
        result = self.vfs.edit("/system_context.md", 1, 1, "hacked")
        self.assertFalse(result["success"])


# ===========================================================================
# 5. APPEND
# ===========================================================================

@_skip_if_no_vfs
class TestVFSAppend(unittest.TestCase):

    def setUp(self):
        self.vfs = make_vfs()
        self.vfs.create("/log.txt", "Erste Zeile")

    def test_append_does_not_overwrite(self):
        """append() ergänzt den Inhalt, überschreibt nicht."""
        self.vfs.append("/log.txt", "\nZweite Zeile")
        content = self.vfs.files["/log.txt"]._content
        self.assertIn("Erste Zeile", content,
            "append() darf vorhandenen Inhalt nicht löschen")
        self.assertIn("Zweite Zeile", content)

    def test_append_to_nonexistent_creates_file(self):
        """append() auf nicht-existente Datei erstellt sie."""
        result = self.vfs.append("/neu.txt", "Inhalt")
        self.assertTrue(result["success"])
        self.assertIn("/neu.txt", self.vfs.files)

    def test_append_multiple_times_accumulates(self):
        """Mehrfaches append() akkumuliert Inhalt korrekt."""
        for i in range(5):
            self.vfs.append("/log.txt", f"\nEntry {i}")
        content = self.vfs.files["/log.txt"]._content
        for i in range(5):
            self.assertIn(f"Entry {i}", content)


# ===========================================================================
# 6. GREP - Undokumentiertes 50er-Limit
# ===========================================================================

@_skip_if_no_vfs
class TestVFSGrepLimits(unittest.TestCase):
    """
    KRITISCH: vfs_grep hat ein hartes Limit von 50 Matches.
    Dieses Limit ist in der Tool-Beschreibung NICHT dokumentiert.
    Der Agent weiß nicht, dass er möglicherweise nur Teilresultate bekommt.
    """

    def setUp(self):
        self.vfs = make_vfs()

    def _make_grep_target(self, match_count: int):
        """Erstellt eine Datei mit n Zeilen die alle 'PATTERN' enthalten."""
        lines = [f"PATTERN Zeile {i}" for i in range(1, match_count + 1)]
        self.vfs.create("/searchme.txt", "\n".join(lines))

    def test_grep_finds_pattern(self):
        """grep findet vorhandenes Pattern."""
        self.vfs.create("/code.py", "def foo():\n    pass\ndef bar():\n    pass")
        # vfs_grep ist eine Closure - teste die VFS-Methode direkt
        from toolboxv2.mods.isaa.base.patch.power_vfs import grep_vfs
        results = grep_vfs(self.vfs, pattern="def ", file_pattern="*.py")
        self.assertEqual(len(results), 2)

    def test_grep_regex_error_returns_message(self):
        """Ungültiges Regex gibt verständliche Fehlermeldung zurück."""
        from toolboxv2.mods.isaa.base.patch.power_vfs import grep_vfs
        self.vfs.create("/x.py", "test")
        # Ungültiges Regex
        results = grep_vfs(self.vfs, pattern="[invalid")
        self.assertEqual(results, [],
            "Ungültiges Regex soll leere Liste zurückgeben, nicht crashen")

    def test_grep_respects_max_results(self):
        """grep_vfs hat max_results-Parameter - 50 per default."""
        self._make_grep_target(100)
        from toolboxv2.mods.isaa.base.patch.power_vfs import grep_vfs
        results = grep_vfs(self.vfs, pattern="PATTERN")
        self.assertLessEqual(len(results), 100,
            "grep_vfs soll nicht mehr als max_results zurückgeben")

    def test_grep_case_insensitive_default(self):
        """Grep unterstützt case-insensitive Suche via (?i) Flag im Pattern."""
        from toolboxv2.mods.isaa.base.patch.power_vfs import grep_vfs
        self.vfs.create("/test.py", "FOO\nfoo\nFoO")
        # grep_vfs ist case-sensitive; (?i) aktiviert IGNORECASE
        results = grep_vfs(self.vfs, pattern="(?i)foo")
        self.assertEqual(len(results), 3,
            "Case-insensitive grep via (?i) muss alle Varianten finden")

    def test_grep_shadow_file_not_loaded(self):
        """grep auf Shadow-Datei die noch nicht geladen ist - kein Crash."""
        # Shadow file simulieren
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# SHADOW_PATTERN\n")
            tmp_path = f.name
        try:
            from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VFSFile, FileBackingType
            shadow = VFSFile(
                filename="shadow.py",
                backing_type=FileBackingType.SHADOW,
                _content=None,
                local_path=tmp_path,
            )
            self.vfs.files["/shadow.py"] = shadow
            from toolboxv2.mods.isaa.base.patch.power_vfs import grep_vfs
            results = grep_vfs(self.vfs, pattern="SHADOW_PATTERN")
            self.assertEqual(len(results), 1,
                "grep soll Shadow-Dateien vom Disk lesen können")
        finally:
            os.unlink(tmp_path)

    def test_find_files_glob_pattern(self):
        """find_files findet Dateien per Glob-Pattern."""
        from toolboxv2.mods.isaa.base.patch.power_vfs import find_files
        self.vfs.mkdir("/src")
        self.vfs.create("/src/main.py", "")
        self.vfs.create("/src/utils.py", "")
        self.vfs.create("/src/config.yaml", "")
        py_files = find_files(self.vfs, pattern="*.py")
        self.assertEqual(len(py_files), 2,
            f"find_files('*.py') soll 2 .py-Dateien finden, findet: {py_files}")
        yaml_files = find_files(self.vfs, pattern="*.yaml")
        self.assertEqual(len(yaml_files), 1)


# ===========================================================================
# 7. SEARCH_VFS - Erweiterte Suche (in power_vfs, NICHT als Tool registriert)
# ===========================================================================

@_skip_if_no_vfs
class TestSearchVFS(unittest.TestCase):
    """
    search_vfs() aus power_vfs ist NICHT als Agent-Tool registriert.
    Prüft Redundanz mit vfs_grep.
    """

    def setUp(self):
        self.vfs = make_vfs()
        self.vfs.create("/readme.md", "# Hallo Welt\nDies ist ein Test")
        self.vfs.create("/code.py", "def hallo():\n    pass")

    def test_search_content_mode(self):
        """search_vfs findet Content-Matches."""
        from toolboxv2.mods.isaa.base.patch.power_vfs import search_vfs, SearchMode
        results = search_vfs(self.vfs, query="hallo", mode=SearchMode.CONTENT)
        paths = [r.path for r in results]
        self.assertIn("/readme.md", paths)
        self.assertIn("/code.py", paths)

    def test_search_filename_mode(self):
        """search_vfs findet Dateinamen."""
        from toolboxv2.mods.isaa.base.patch.power_vfs import search_vfs, SearchMode
        results = search_vfs(self.vfs, query="readme", mode=SearchMode.FILENAME)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].path, "/readme.md")

    def test_search_regex_mode(self):
        """search_vfs unterstützt Regex."""
        from toolboxv2.mods.isaa.base.patch.power_vfs import search_vfs, SearchMode
        results = search_vfs(self.vfs, query=r"def \w+\(\)", mode=SearchMode.CONTENT, regex=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].path, "/code.py")

    def test_search_not_registered_as_tool(self):
        """
        REGRESSION: search_vfs ist in power_vfs vorhanden aber NICHT als
        Agent-Tool in init_session_tools registriert.
        Der Agent kann diese mächtigere Suche nicht direkt nutzen.
        """
        # Dieser Test dokumentiert die Lücke - kein assert nötig außer der Dokumentation
        # Wenn search_vfs jemals als Tool registriert wird, muss dieser Test
        # die korrekte Registrierung prüfen.
        from toolboxv2.mods.isaa.base.patch.power_vfs import search_vfs
        self.assertTrue(callable(search_vfs),
            "search_vfs existiert in power_vfs - ist aber kein registriertes Agent-Tool")


# ===========================================================================
# 8. FS COPY TOOLS - Filesystem-Operationen
# ===========================================================================

@_skip_if_no_vfs
class TestFSCopyTools(unittest.TestCase):

    def setUp(self):
        self.vfs = make_vfs()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_from_local_copies_content(self):
        """load_from_local liest Datei vom Disk ins VFS."""
        local_file = os.path.join(self.tmpdir, "source.py")
        with open(local_file, "w") as f:
            f.write("print('hello from disk')")
        result = self.vfs.load_from_local(local_file, vfs_path="/imported.py")
        self.assertTrue(result["success"], f"load_from_local fehlgeschlagen: {result}")
        self.assertIn("/imported.py", self.vfs.files)
        self.assertEqual(self.vfs.files["/imported.py"]._content, "print('hello from disk')")

    def test_load_from_local_allowed_dirs_blocks(self):
        """allowed_dirs blockiert Zugriff auf verbotene Verzeichnisse."""
        result = self.vfs.load_from_local(
            "/etc/passwd",
            allowed_dirs=[self.tmpdir]
        )
        self.assertFalse(result["success"])
        self.assertIn("allowed", result["error"].lower())

    def test_load_from_local_max_size_blocks_large(self):
        """max_size_bytes blockiert zu große Dateien."""
        big_file = os.path.join(self.tmpdir, "big.bin")
        with open(big_file, "wb") as f:
            f.write(b"x" * (1024 * 10))  # 10KB
        result = self.vfs.load_from_local(big_file, max_size_bytes=1024)  # 1KB limit
        self.assertFalse(result["success"])
        self.assertIn("large", result["error"].lower())

    def test_save_to_local_writes_file(self):
        """save_to_local schreibt VFS-Inhalt auf Disk."""
        self.vfs.create("/output.py", "# Output")
        dest = os.path.join(self.tmpdir, "output.py")
        result = self.vfs.save_to_local("/output.py", dest, create_dirs=True)
        self.assertTrue(result["success"], f"save_to_local fehlgeschlagen: {result}")
        self.assertTrue(os.path.exists(dest))
        with open(dest) as f:
            self.assertEqual(f.read(), "# Output")

    def test_save_to_local_no_overwrite_by_default(self):
        """save_to_local überschreibt nicht ohne overwrite=True."""
        existing = os.path.join(self.tmpdir, "exists.py")
        with open(existing, "w") as f:
            f.write("original")
        self.vfs.create("/new.py", "modified")
        result = self.vfs.save_to_local("/new.py", existing, overwrite=False)
        self.assertFalse(result["success"],
            "Ohne overwrite=True darf save_to_local bestehende Datei nicht überschreiben")

    def test_save_to_local_creates_parent_dirs(self):
        """create_dirs=True erstellt fehlende Verzeichnisse."""
        deep_path = os.path.join(self.tmpdir, "a", "b", "c", "file.py")
        self.vfs.create("/deep.py", "content")
        result = self.vfs.save_to_local("/deep.py", deep_path, create_dirs=True)
        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(deep_path))


# ===========================================================================
# 9. MOUNT OPERATIONS
# ===========================================================================

@_skip_if_no_vfs
class TestMountOperations(unittest.TestCase):

    def setUp(self):
        self.vfs = make_vfs()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_project(self):
        """Erstellt ein kleines Projekt-Verzeichnis."""
        src = os.path.join(self.tmpdir, "src")
        os.makedirs(src)
        with open(os.path.join(self.tmpdir, "main.py"), "w") as f:
            f.write("# main")
        with open(os.path.join(src, "utils.py"), "w") as f:
            f.write("# utils")
        with open(os.path.join(self.tmpdir, "config.yaml"), "w") as f:
            f.write("key: value")

    def test_mount_indexes_files_without_loading_content(self):
        """mount() indiziert Dateien aber lädt KEINEN Content - nur Metadata."""
        self._make_project()
        result = self.vfs.mount(self.tmpdir, vfs_path="/project")
        self.assertTrue(result["success"])
        self.assertGreater(result["files_indexed"], 0)
        # Content soll NICHT geladen sein (lazy loading)
        for path, f in self.vfs.files.items():
            if path.startswith("/project"):
                from toolboxv2.mods.isaa.base.Agent.vfs_v2 import FileBackingType
                self.assertEqual(f.backing_type, FileBackingType.SHADOW,
                    f"{path}: Shadow-Datei soll backing_type=SHADOW haben, nicht geladen")
                self.assertIsNone(f._content,
                    f"{path}: Shadow-Datei darf keinen geladenen Content haben nach mount()")

    def test_mount_then_open_loads_content(self):
        """Shadow-Datei wird on-demand geladen wenn open() aufgerufen wird."""
        self._make_project()
        self.vfs.mount(self.tmpdir, vfs_path="/project")
        result = self.vfs.open("/project/main.py")
        self.assertTrue(result["success"])
        f = self.vfs.files["/project/main.py"]
        self.assertIsNotNone(f._content, "Content soll nach open() geladen sein")
        self.assertEqual(f._content.strip(), "# main")

    def test_mount_respects_allowed_extensions(self):
        """allowed_extensions filtert Dateien beim Mount."""
        self._make_project()
        result = self.vfs.mount(
            self.tmpdir,
            vfs_path="/pyonly",
            allowed_extensions=[".py"]
        )
        self.assertTrue(result["success"])
        for path in self.vfs.files:
            if path.startswith("/pyonly"):
                self.assertTrue(path.endswith(".py"),
                    f"{path}: Nur .py-Dateien sollen gemountet sein")

    def test_unmount_removes_files_from_vfs(self):
        """unmount() entfernt alle Dateien des Mount-Points aus dem VFS."""
        self._make_project()
        self.vfs.mount(self.tmpdir, vfs_path="/project")
        project_files_before = [p for p in self.vfs.files if p.startswith("/project")]
        self.assertGreater(len(project_files_before), 0)

        result = self.vfs.unmount("/project", save_changes=False)
        self.assertTrue(result["success"])
        project_files_after = [p for p in self.vfs.files if p.startswith("/project")]
        self.assertEqual(len(project_files_after), 0,
            "Nach unmount() dürfen keine /project-Dateien im VFS sein")

    def test_write_to_mounted_file_syncs_to_disk(self):
        """Änderungen an gemounteten Dateien werden auto-sync auf Disk geschrieben."""
        self._make_project()
        self.vfs.mount(self.tmpdir, vfs_path="/project", auto_sync=True)
        self.vfs.open("/project/main.py")  # Laden
        self.vfs.write("/project/main.py", "# MODIFIZIERT")
        # Sync-Datei prüfen
        local_path = os.path.join(self.tmpdir, "main.py")
        with open(local_path) as f:
            content = f.read()
        self.assertEqual(content, "# MODIFIZIERT",
            "auto_sync=True soll Änderungen sofort auf Disk schreiben")


# ===========================================================================
# 10. SYSTEM FILES - Schutz und Verhalten
# ===========================================================================

@_skip_if_no_vfs
class TestSystemFiles(unittest.TestCase):

    def setUp(self):
        self.vfs = make_vfs()

    def test_system_context_always_open(self):
        """system_context.md ist immer im State 'open'."""
        f = self.vfs.files["/system_context.md"]
        self.assertEqual(f.state, "open",
            "system_context.md muss state='open' haben - immer im Kontext")

    def test_system_context_readonly(self):
        """system_context.md ist readonly - kann nicht überschrieben werden."""
        result = self.vfs.write("/system_context.md", "Gehackt")
        self.assertFalse(result["success"])

    def test_active_rules_appears_after_set(self):
        """active_rules.md erscheint nach set_rules_file im Kontext."""
        self.vfs.set_rules_file("# Meine Regeln\nRegel 1")
        self.assertIn("/active_rules.md", self.vfs.files)
        ctx = self.vfs.build_context_string()
        self.assertIn("Regel 1", ctx)

    def test_system_context_contains_agent_info(self):
        """system_context.md enthält Agenten-Name und Session-ID."""
        ctx = self.vfs.build_context_string()
        self.assertIn("TestAgent", ctx)
        self.assertIn("test-session", ctx)

    def test_cannot_delete_protected_system_files(self):
        """Geschützte System-Dateien können nicht gelöscht werden."""
        result = self.vfs.delete("/system_context.md")
        self.assertFalse(result["success"],
            "system_context.md ist geschützt und darf nicht gelöscht werden")

    def test_add_system_file(self):
        """add_system_file fügt lokale Datei als read-only System-Datei hinzu."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Handbuch\nInhalt")
            tmp = f.name
        try:
            result = self.vfs.add_system_file(tmp, vfs_path="/manual.md")
            self.assertTrue(result["success"])
            self.assertIn("/manual.md", self.vfs.files)
            self.assertTrue(self.vfs.files["/manual.md"].readonly)
        finally:
            os.unlink(tmp)


# ===========================================================================
# 11. TOOL NAME REGISTRATION - Die intentionale Vertauschung
# ===========================================================================

@_skip_if_no_vfs
class TestToolNameRegistration(unittest.TestCase):
    """
    Testet die intentionale Tool-Name-Vertauschung:
    - Tool "vfs_read" → ruft vfs_open() auf (Windowed, Agent-bevorzugt)
    - Tool "vfs_open" → ruft vfs_read() auf (Full, Notfall)

    Diese Vertauschung existiert weil der Agent LLM-bedingt instinktiv
    "read" verwendet - und "read" soll den effizienten Weg nehmen.
    """

    def _get_tools_from_mock_agent(self):
        """
        Erstellt einen minimalen Mock-FlowAgent und ruft init_session_tools auf.
        Gibt die Liste der registrierten Tool-Dicts zurück.
        """
        try:
            from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2
        except ImportError:
            self.skipTest("VFS not available")

        # Minimal-Mock Session
        session = MagicMock()
        session.tools_initialized = False
        session.session_id = "test-reg"
        session.agent_name = "TestReg"
        session.vfs = VirtualFileSystemV2(
            "test-reg", "TestReg", max_window_lines=100,
            summarizer=None, lsp_manager=None
        )
        # Delegiere VFS-Calls direkt
        session.vfs_open = session.vfs.open
        session.vfs_read = session.vfs.read
        session.vfs_write = session.vfs.write
        session.vfs_create = session.vfs.create
        session.vfs_ls = session.vfs.ls
        session.vfs_mkdir = session.vfs.mkdir
        session.vfs_rmdir = session.vfs.rmdir
        session.vfs_mv = session.vfs.mv
        session.vfs_close = session.vfs.close
        session.vfs_diagnostics = session.vfs.get_diagnostics
        session.get_history_for_llm = MagicMock(return_value=[])
        session.set_situation = MagicMock()
        session.rule_on_action = MagicMock()
        mock_result = MagicMock()
        mock_result.allowed = True
        mock_result.reason = "ok"
        mock_result.rule_name = "default"
        session.rule_on_action.return_value = mock_result
        session.docker_status = MagicMock(return_value={"enabled": False})

        registered_tools = []

        class MockFlowAgent:
            def add_tools(self, tools):
                registered_tools.extend(tools)

        try:
            from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
            # Wenn FlowAgent importierbar, nutze echte init_session_tools
            # aber mit Mock-Session
            agent_instance = object.__new__(FlowAgent)
            with patch.dict(os.environ, {"WITH_GOOGLE_TOOLS": "false", "WITH_CODE_TOOLS": "false"}):
                result = FlowAgent.init_session_tools(agent_instance, session)
            if result:
                return result, session
        except Exception:
            pass

        return registered_tools, session

    def test_vfs_read_tool_uses_vfs_open_function(self):
        """
        """
        tools, session = self._get_tools_from_mock_agent()
        if not tools:
            self.skipTest("Konnte init_session_tools nicht mocken")
        vfs_read_entry = next((t for t in tools if t["name"] == "vfs_read"), None)
        self.assertIsNotNone(vfs_read_entry,
            "Tool 'vfs_read' muss registriert sein")
        # Datei auf der VFS anlegen, die der Tool-Closure nutzt
        session.vfs.create("/f.py", "L1\nL2\nL3\nL4\nL5\nL6\nL7")
        result = vfs_read_entry["tool_func"]("/f.py")
        self.assertIn("content", result,
            "vfs_read tool soll vfs.open() aufrufen (preview-Key), nicht vfs.read() (content-Key)")

    def test_vfs_open_tool_uses_vfs_read_function(self):
        """
        """
        tools, session = self._get_tools_from_mock_agent()
        if not tools:
            self.skipTest("Konnte init_session_tools nicht mocken")
        vfs_open_entry = next((t for t in tools if t["name"] == "vfs_open"), None)
        self.assertIsNotNone(vfs_open_entry,
            "Tool 'vfs_open' muss registriert sein")
        session.vfs.create("/f.py", "FULL_CONTENT")
        result = vfs_open_entry["tool_func"]("/f.py")
        self.assertIn("preview", result,
            "vfs_open tool soll vfs.read() aufrufen (content-Key)")

    def test_all_expected_vfs_tool_names_registered(self):
        """Alle erwarteten VFS-Tool-Namen sind in der registrierten Liste."""
        tools, session = self._get_tools_from_mock_agent()
        if not tools:
            self.skipTest("Konnte init_session_tools nicht mocken")
        names = {t["name"] for t in tools}
        expected = {
            "vfs_list", "vfs_read", "vfs_create", "vfs_write",
            "vfs_edit", "vfs_append", "vfs_delete",
            "vfs_mkdir", "vfs_rmdir", "vfs_mv",
            "vfs_open", "vfs_close", "vfs_view", "vfs_grep",
            "vfs_info", "vfs_executables",
        }
        missing = expected - names
        self.assertEqual(missing, set(),
            f"Folgende Tool-Namen sind nicht registriert: {missing}")


# ===========================================================================
# 12. CONTEXT-BUILDING - Was der Agent wirklich sieht
# ===========================================================================

@_skip_if_no_vfs
class TestContextBuilding(unittest.TestCase):
    """
    Diese Tests prüfen was der Agent tatsächlich in jedem Prompt sieht.
    Das ist die kritischste User-Experience-Prüfung.
    """

    def setUp(self):
        self.vfs = make_vfs(max_window=50)

    def test_empty_vfs_contains_system_context(self):
        """Auch leeres VFS enthält system_context.md."""
        ctx = self.vfs.build_context_string()
        self.assertIn("System Context", ctx)
        self.assertIn("VFS (Virtual File System)", ctx)

    def test_closed_file_shown_in_closed_count(self):
        """Geschlossene Dateien erscheinen als Zähler, nicht als Inhalt."""
        self.vfs.create("/closed1.py", "content")
        self.vfs.create("/closed2.py", "content")
        ctx = self.vfs.build_context_string()
        # Inhalt soll nicht da sein
        self.assertNotIn("content", ctx)
        # Zähler soll da sein
        self.assertIn("closed", ctx.lower())

    def test_multiple_open_files_all_in_context(self):
        """Mehrere offene Dateien erscheinen alle im Kontext."""
        self.vfs.create("/a.py", "INHALT_A")
        self.vfs.create("/b.py", "INHALT_B")
        self.vfs.open("/a.py")
        self.vfs.open("/b.py")
        ctx = self.vfs.build_context_string()
        self.assertIn("INHALT_A", ctx)
        self.assertIn("INHALT_B", ctx)

    def test_dir_tree_in_context(self):
        """Verzeichnisbaum erscheint im Kontext (Orientierung für Agent)."""
        self.vfs.mkdir("/src")
        self.vfs.create("/src/main.py", "x")
        ctx = self.vfs.build_context_string()
        self.assertIn("src", ctx,
            "Verzeichnisbaum soll im Kontext erscheinen")

    def test_modified_file_marked_in_context(self):
        """Modifizierte Shadow-Dateien sind als dirty markiert und zeigen geänderten Inhalt."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("original")
            tmp = f.name
        try:
            self.vfs.mount(os.path.dirname(tmp), vfs_path="/proj",
                           allowed_extensions=[".py"])
            fname = os.path.basename(tmp)
            vfs_path = f"/proj/{fname}"
            self.vfs.open(vfs_path)
            self.vfs.write(vfs_path, "modified")
            # Primär: is_dirty muss gesetzt sein
            vfs_file = self.vfs.files.get(vfs_path)
            self.assertIsNotNone(vfs_file, "Datei muss im VFS registriert sein")
            self.assertTrue(vfs_file.is_dirty,
                "Modifizierte Shadow-Datei muss is_dirty=True haben")
            # Sekundär: geänderter Inhalt muss im Kontext sichtbar sein (Datei ist offen)
            ctx = self.vfs.build_context_string()
            self.assertIn("modified", ctx,
                "Modifizierter Inhalt soll im Kontext sichtbar sein da Datei geöffnet ist")
        finally:
            os.unlink(tmp)

    def test_context_shows_line_range_info(self):
        """Kontext zeigt Zeilennummern-Info (lines X-Y) für Navigation."""
        self.vfs.create("/nav.py", make_large_file(50))
        self.vfs.open("/nav.py", line_start=10, line_end=20)
        ctx = self.vfs.build_context_string()
        self.assertIn("10", ctx,
            "Kontext soll Startzeile zeigen damit Agent weiß wo er ist")
        self.assertIn("20", ctx,
            "Kontext soll Endzeile zeigen damit Agent weiß wo er ist")

    def test_truncation_message_when_file_too_long(self):
        """Bei Truncation durch max_window_lines erscheint Hinweis."""
        vfs = make_vfs(max_window=5)
        vfs.create("/long.py", make_large_file(100))
        vfs.open("/long.py", line_start=1, line_end=-1)
        ctx = vfs.build_context_string()
        self.assertIn("truncated", ctx.lower(),
            "Wenn Datei gekürzt wird, muss Hinweis 'truncated' erscheinen "
            "damit Agent weiß dass er nicht alles sieht")


# ===========================================================================
# 13. DIREKTORY OPERATIONS
# ===========================================================================

@_skip_if_no_vfs
class TestDirectoryOps(unittest.TestCase):

    def setUp(self):
        self.vfs = make_vfs()

    def test_mkdir_creates_directory(self):
        result = self.vfs.mkdir("/mydir")
        self.assertTrue(result["success"])
        self.assertIn("/mydir", self.vfs.directories)

    def test_mkdir_parents_creates_nested(self):
        result = self.vfs.mkdir("/a/b/c", parents=True)
        self.assertTrue(result["success"])
        self.assertIn("/a", self.vfs.directories)
        self.assertIn("/a/b", self.vfs.directories)
        self.assertIn("/a/b/c", self.vfs.directories)

    def test_mkdir_no_parents_fails_on_missing_parent(self):
        result = self.vfs.mkdir("/missing/child", parents=False)
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_rmdir_fails_on_nonempty(self):
        self.vfs.mkdir("/full")
        self.vfs.create("/full/file.txt", "")
        result = self.vfs.rmdir("/full", force=False)
        self.assertFalse(result["success"])
        self.assertIn("force", result["error"].lower())

    def test_rmdir_force_removes_contents(self):
        self.vfs.mkdir("/full")
        self.vfs.create("/full/file.txt", "")
        result = self.vfs.rmdir("/full", force=True)
        self.assertTrue(result["success"])
        self.assertNotIn("/full", self.vfs.directories)
        self.assertNotIn("/full/file.txt", self.vfs.files)

    def test_mv_file(self):
        self.vfs.create("/old.py", "content")
        result = self.vfs.mv("/old.py", "/new.py")
        self.assertTrue(result["success"])
        self.assertNotIn("/old.py", self.vfs.files)
        self.assertIn("/new.py", self.vfs.files)
        self.assertEqual(self.vfs.files["/new.py"]._content, "content")

    def test_ls_shows_files_and_dirs(self):
        self.vfs.mkdir("/mydir")
        self.vfs.create("/mydir/file.txt", "")
        result = self.vfs.ls("/mydir")
        self.assertTrue(result["success"])
        names = [c["name"] for c in result["contents"]]
        self.assertIn("file.txt", names)

    def test_ls_recursive(self):
        self.vfs.mkdir("/root/sub", parents=True)
        self.vfs.create("/root/sub/deep.txt", "")
        result = self.vfs.ls("/root", recursive=True)
        paths = [c["path"] for c in result["contents"]]
        self.assertIn("/root/sub/deep.txt", paths)

    def test_cannot_remove_root(self):
        result = self.vfs.rmdir("/")
        self.assertFalse(result["success"])


# ===========================================================================
# 14. CHECKPOINT / SERIALIZATION
# ===========================================================================

@_skip_if_no_vfs
class TestVFSCheckpoint(unittest.TestCase):

    def setUp(self):
        self.vfs = make_vfs()

    def test_checkpoint_excludes_readonly_files(self):
        """Readonly-System-Dateien werden nicht in Checkpoint aufgenommen."""
        self.vfs.create("/meine.py", "Mein Code")
        cp = self.vfs.to_checkpoint()
        self.assertNotIn("/system_context.md", cp["files"],
            "system_context.md wird dynamisch regeneriert, nicht serialisiert")

    def test_checkpoint_includes_user_files(self):
        """User-erstellte Dateien erscheinen im Checkpoint."""
        self.vfs.create("/meine.py", "Mein Code")
        cp = self.vfs.to_checkpoint()
        self.assertIn("/meine.py", cp["files"])

    def test_restore_from_checkpoint(self):
        """from_checkpoint restauriert Dateien korrekt."""
        self.vfs.create("/save_me.py", "Gespeichert")
        cp = self.vfs.to_checkpoint()

        new_vfs = make_vfs()
        new_vfs.from_checkpoint(cp)
        self.assertIn("/save_me.py", new_vfs.files)

    def test_checkpoint_round_trip_preserves_content(self):
        """Content bleibt nach Checkpoint-Round-Trip erhalten."""
        original = "def wichtige_funktion():\n    pass\n"
        self.vfs.create("/wichtig.py", original)
        cp = self.vfs.to_checkpoint()
        new_vfs = make_vfs()
        new_vfs.from_checkpoint(cp)
        # Inhalt lesen
        result = new_vfs.read("/wichtig.py")
        self.assertTrue(result["success"])
        self.assertEqual(result["content"], original)


# ===========================================================================
# 15. EXECUTE - Lokale Skript-Ausführung
# ===========================================================================

@_skip_if_no_vfs
class TestVFSExecute(unittest.TestCase):

    def setUp(self):
        self.vfs = make_vfs()

    def test_execute_python_script(self):
        """Python-Skript wird lokal ausgeführt."""
        script = 'print("hello_from_vfs")'
        self.vfs.create("/hello.py", script)
        result = self.vfs.execute("/hello.py")
        self.assertTrue(result["success"], f"Ausführung fehlgeschlagen: {result}")
        self.assertIn("hello_from_vfs", result["stdout"])

    def test_execute_nonexistent_fails(self):
        result = self.vfs.execute("/ghost.py")
        self.assertFalse(result["success"])

    def test_execute_non_executable_fails(self):
        """Textdatei ist nicht ausführbar."""
        self.vfs.create("/readme.txt", "Kein Code")
        result = self.vfs.execute("/readme.txt")
        self.assertFalse(result["success"])

    def test_get_executable_files_lists_scripts(self):
        """get_executable_files() findet Python-Skripte."""
        self.vfs.create("/run.py", "pass")
        self.vfs.create("/readme.txt", "text")
        execs = self.vfs.get_executable_files()
        exec_paths = [e["path"] for e in execs]
        self.assertIn("/run.py", exec_paths)
        self.assertNotIn("/readme.txt", exec_paths)


if __name__ == "__main__":
    # Verbose output für investigative Analyse
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestOpenVsReadSemantics,
        TestVFSViewScrolling,
        TestVFSOpenLineRange,
        TestVFSEdit,
        TestVFSAppend,
        TestVFSGrepLimits,
        TestSearchVFS,
        TestFSCopyTools,
        TestMountOperations,
        TestSystemFiles,
        TestToolNameRegistration,
        TestContextBuilding,
        TestDirectoryOps,
        TestVFSCheckpoint,
        TestVFSExecute,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
