# =============================================================================
# VFS_VIEW — VISION / MEDIA SYSTEM
# =============================================================================

import os
import unittest
import urllib.request
import tempfile
import base64
from unittest.mock import patch, MagicMock

from toolboxv2.mods.isaa.base.patch.vfs_shell_tool import make_vfs_view
from toolboxv2.tests.test_mods.test_isaa.test_base.test_agent.test_vfs.test_vfs_v2 import _make_session


class TestVfsViewVisionSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Lädt reale Dummy-Dateien aus dem Netz für die Tests herunter."""
        cls.temp_dir = tempfile.TemporaryDirectory()

        # 1. Online Bild laden
        cls.img_path = os.path.join(cls.temp_dir.name, "test_image.png")
        urllib.request.urlretrieve("https://httpbin.org/image/png", cls.img_path)

        # 2. Online PDF laden (W3C Dummy)
        cls.pdf_path = os.path.join(cls.temp_dir.name, "test_doc.pdf")
        urllib.request.urlretrieve(
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            cls.pdf_path
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        self.session = _make_session()
        self.view = make_vfs_view(self.session)
        self.vfs = self.session.vfs

        # Hänge die physischen Dateien als Shadow-Files ins VFS
        from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VFSFile, FileBackingType

        self.vfs.files["/test_image.png"] = VFSFile(
            filename="test_image.png",
            backing_type=FileBackingType.SHADOW,
            local_path=self.img_path,
            size_bytes=os.path.getsize(self.img_path),
            state="closed"
        )
        self.vfs.files["/test_doc.pdf"] = VFSFile(
            filename="test_doc.pdf",
            backing_type=FileBackingType.SHADOW,
            local_path=self.pdf_path,
            size_bytes=os.path.getsize(self.pdf_path),
            state="closed"
        )

    @patch("litellm.completion")
    def test_image_auto_detection_and_base64_conversion(self, mock_litellm):
        """Prüft ob Bilder automatisch erkannt, in Base64 gewandelt und analysiert werden."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Mocked Image Content"
        mock_litellm.return_value = mock_resp

        r = self.view("/test_image.png")

        self.assertTrue(r["success"])
        self.assertIn("--- MEDIA ANALYSIS RESULT", r["content"])
        self.assertIn("Mocked Image Content", r["content"])

        # Sicherstellen dass der LLM-Call die korrekte payload hatte
        mock_litellm.assert_called_once()
        call_kwargs = mock_litellm.call_args.kwargs
        messages = call_kwargs["messages"][0]["content"]

        self.assertEqual(messages[0]["type"], "text")
        self.assertEqual(messages[1]["type"], "image_url")
        self.assertTrue(messages[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    @patch("litellm.completion")
    def test_pdf_native_mode(self, mock_litellm):
        """Prüft ob PDFs zuerst als nativer base64-PDF Stream an das Modell gesendet werden."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Mocked Native PDF Content"
        mock_litellm.return_value = mock_resp

        r = self.view("/test_doc.pdf")

        self.assertTrue(r["success"])
        call_kwargs = mock_litellm.call_args.kwargs
        messages = call_kwargs["messages"][0]["content"]

        self.assertEqual(messages[1]["type"], "image_url")
        self.assertTrue(messages[1]["image_url"]["url"].startswith("data:application/pdf;base64,"))

    @patch("litellm.completion")
    def test_pdf_fallback_fitz_mode(self, mock_litellm):
        """Prüft den PyMuPDF (fitz) Fallback, wenn das Modell kein natives PDF unterstützt."""

        def side_effect(*args, **kwargs):
            # Erster Call: Natives PDF -> Wirft Fehler (z.B. OpenAI unterstützt kein PDF direkt)
            if kwargs["messages"][0]["content"][1]["image_url"]["url"].startswith("data:application/pdf"):
                raise Exception("Model does not support PDF")

            # Zweiter Call: Fallback (fitz PNGs) -> Erfolgreich
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = "Mocked Fitz PNG Content"
            return mock_resp

        mock_litellm.side_effect = side_effect

        r = self.view("/test_doc.pdf", line_start=1, line_end=1)  # Nur 1 Seite umwandeln

        self.assertTrue(r["success"])
        self.assertIn("Mocked Fitz PNG Content", r["content"])
        self.assertEqual(mock_litellm.call_count, 2)  # Nativ fehlgeschlagen, Fitz erfolgreich

    @patch("litellm.completion")
    def test_focus_on_media_section_forces_reanalysis(self, mock_litellm):
        """Prüft ob ein zweiter View-Call das Cached-Ergebnis nutzt, AUSSER focus_on_media_section ist gesetzt."""
        mock_resp1 = MagicMock()
        mock_resp1.choices[0].message.content = "First Cached View"
        mock_litellm.return_value = mock_resp1

        # 1. Erster Call: Erstellt Cache
        self.view("/test_image.png")
        self.assertEqual(mock_litellm.call_count, 1)

        # 2. Zweiter Call ohne Focus: Greift auf Cache zu (Call-Count bleibt 1)
        self.view("/test_image.png")
        self.assertEqual(mock_litellm.call_count, 1)

        # 3. Dritter Call mit Focus: Erzwingt Re-Analyse (Call-Count wird 2)
        mock_resp2 = MagicMock()
        mock_resp2.choices[0].message.content = "Second Focused View"
        mock_litellm.return_value = mock_resp2

        r = self.view("/test_image.png", focus_on_media_section="Find the red button")

        self.assertEqual(mock_litellm.call_count, 2)
        self.assertIn("Second Focused View", r["content"])
        self.assertIn("Find the red button", r["content"])  # Prompt sollte im VFS Content gespeichert sein
