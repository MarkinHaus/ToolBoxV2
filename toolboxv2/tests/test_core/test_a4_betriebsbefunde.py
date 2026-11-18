"""
Betriebstests WP3 Auftrag 4 - Paket und Sync (7 Tests, Tabelle Nr. 1-7).

Belegt die 7 Betrieberbefunde aus WP3_ordnung_und_betriebstests_2026-11.md
auf Basis c36c3dc. Erwartung: alle 7 ROT auf der Basis, GRUEN nach Fix.
KEIN Produktcode wird hier geaendert (Betriebstests-Regel).
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_T = Path(tempfile.mkdtemp(prefix="tb_a4_"))
# Env-Isolation NUR waehrend unserer Tests (kein Modul-Import-Pollution,
# sonst faellt test_device_account im selben pytest-Prozess mit).
_SAVED_ENV = None


def _env_enter():
    global _SAVED_ENV
    _SAVED_ENV = {k: os.environ.get(k)
                  for k in ("TB_DATA_DIR", "TB_MANIFEST_DIR")}
    os.environ["TB_DATA_DIR"] = str(_T)
    os.environ["TB_MANIFEST_DIR"] = str(_T)


def _env_exit():
    if _SAVED_ENV is None:
        return
    for k, v in _SAVED_ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def setUpModule():  # pytest/unittest-Hook vor allen Tests dieses Moduls
    _env_enter()


def tearDownModule():  # Hook nach allen Tests - Env sauber zurueckgeben
    _env_exit()

REPO = Path(__file__).resolve().parents[3]


class TestA4Row1_WheelImportsLiveSync(unittest.TestCase):
    """Nr. 1: Wheel bauen, entpacken, `python -I` aus fremdem Ordner:
    `import toolboxv2.mods.CloudM.LiveSync.data_api_sync` muss gelingen.
    Ist: ModuleNotFoundError - pyproject.toml schliesst LiveSync aus
    (exclude-package-data + packages.find.exclude)."""

    def test_wheel_import_livesync(self):
        whl_dir = _T / "whl"
        whl_dir.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", ".", "--no-deps",
             "--no-build-isolation", "-w", str(whl_dir)],
            cwd=str(REPO), capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, f"pip wheel failed:\n{r.stderr[-800:]}")
        whls = list(whl_dir.glob("*.whl"))
        self.assertTrue(whls, "kein Wheel gebaut")
        whl = whls[0]
        extract = _T / "whl_x"
        extract.mkdir(exist_ok=True)
        import zipfile
        with zipfile.ZipFile(whl) as z:
            names = z.namelist()
            z.extractall(extract)
        # Paketierungsnachweis: LiveSync-Module muessen im Wheel liegen.
        ls = [n for n in names if "mods/CloudM/LiveSync" in n]
        self.assertTrue(
            ls, f"LiveSync fehlt im Wheel ({whl.name}) - Packaging-Excludes")

        # Betriebsnachweis: Import NUR aus dem entpackten Wheel, fremdes
        # CWD, -I isoliert sys.path (kein Projekt-Leak). env: erben, weil
        # toolboxv2 beim Import HOME/APPDATA braucht (db_cli_manager).
        code = ("import sys; sys.path.insert(0, r'%s'); "
                "import toolboxv2.mods.CloudM.LiveSync.data_api_sync; "
                "print('IMPORT_OK')" % extract)
        run_dir = _T / "fremd"
        run_dir.mkdir(exist_ok=True)
        p = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=str(run_dir), capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace")
        self.assertIn(
            "IMPORT_OK", p.stdout or "",
            f"Import aus Wheelinstallation schlaegt fehl:\n{p.stderr[-500:]}\n"
            f"(Wheel: {whl.name}; LiveSync ausgeschlossen?)")


class TestA4Row2_MultipartUploadWorks(unittest.TestCase):
    """Nr. 2: Multipart-Upload an parse_request - Datei muss ankommen.
    Ist: MULTIPART_AVAILABLE=False - `from multipart import parse_form_data`
    existiert im deklarierten Paket `python-multipart` nicht."""

    def test_parse_request_delivers_uploaded_file(self):
        boundary = "----tbboundary42"
        payload = (f"--{boundary}\r\n"
                   'Content-Disposition: form-data; name="file"; '
                   'filename="hello.txt"\r\n'
                   "Content-Type: text/plain\r\n\r\n"
                   "INHALT-42\r\n"
                   f"--{boundary}--\r\n").encode()
        environ = {
            "REQUEST_METHOD": "POST",
            "PATH_INFO": "/upload",
            "QUERY_STRING": "",
            "CONTENT_TYPE": f"multipart/form-data; boundary={boundary}",
            "CONTENT_LENGTH": str(len(payload)),
            "wsgi.input": io.BytesIO(payload),
        }
        from toolboxv2.utils.workers.server_worker import parse_request
        req = parse_request(environ)
        self.assertIn(
            "file", req.files,
            f"Upload-Feld fehlt (files={dict(req.files)}; MULTIPART_AVAILABLE"
            "-Pfad kaputt: 'multipart'-Paket hat kein parse_form_data)")
        if "file" in req.files:
            up = req.files["file"]
            data = Path(up.temp_path).read_bytes()
            self.assertEqual(data, b"INHALT-42", "Datei-Inhalt kommt veraendert an")


class TestA4Row3_ConflictCopySurvivesReExport(unittest.TestCase):
    """Nr. 3: Konfliktkopie im Konto-Spiegel muss einen erneuten
    Export ueberleben (lokaler Wert bleibt erhalten).
    Ist: export_account loescht 'vanished keys' inkl. Konfliktkopien
    (account_bridge.py, delete-vanished-keys-Block)."""

    def test_conflict_copy_not_deleted_by_export(self):
        from toolboxv2.mods.CloudM.LiveSync.account_bridge import export_account
        from toolboxv2.mods.CloudM.LiveSync.conflict import save_conflict_copy

        mirror = _T / "mirror3"
        settings = mirror / "settings"
        settings.mkdir(parents=True, exist_ok=True)
        (settings / "key.json").write_text('{"v": 1}', encoding="utf-8")
        conflict_file = save_conflict_copy(
            settings, "key.json", b'{"v": "lokal"}')
        self.assertTrue(Path(conflict_file).is_file(),
                        "Vorbedingung: Konfliktkopie liegt im Spiegel")

        user = type("U", (), {"settings": {"key": {"v": 2}},
                              "mod_data": {}})()
        export_account(user, mirror)

        self.assertTrue(
            Path(conflict_file).is_file(),
            "Konfliktkopie wurde vom Re-Export geloescht - lokale Aenderung "
            "geht verloren (delete-vanished-keys fasst Konfliktkopien)")


class _FakeTransport:
    """Data-API-Fake: dict als Server-Speicher, zeichnet alle Writes auf."""

    def __init__(self):
        self.entries = {}          # path -> bytes (Ciphertext)
        self.puts = []
        self.gets = []
        self.deleted = []

    async def list(self, prefix):
        return {p: (len(v), 1.0) for p, v in self.entries.items()
                if p.startswith(prefix)}

    async def put(self, path, ciphertext):
        self.puts.append(path)
        self.entries[path] = ciphertext

    async def get(self, path):
        self.gets.append(path)
        return self.entries.get(path)

    async def delete(self, path):
        self.deleted.append(path)
        self.entries.pop(path, None)


class TestA4Row4_SymlinkStaysInside(unittest.TestCase):
    """Nr. 4: Ordner-Symlink (hier: Junction) im globalen ISAA-Ordner, der
    nach aussen zeigt: nichts ausserhalb darf gelesen/hochgeladen werden.
    Ist: Walker folgt dem Link und laedt externe Dateien hoch."""

    def test_junction_outside_is_not_uploaded(self):
        import asyncio

        from toolboxv2.mods.CloudM.LiveSync.data_api_sync import (
            PREFIX_GLOBAL, LocalIndex, sync_folder)

        outside = _T / "a4_outside"
        outside.mkdir(exist_ok=True)
        (outside / "secret.txt").write_text("GEHEIM", encoding="utf-8")

        root = _T / "a4_global" / "Agents" / "VFS" / "global"
        root.mkdir(parents=True, exist_ok=True)
        junction = root / "verweis"
        if not junction.exists():
            r = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
                capture_output=True, text=True)
            self.assertEqual(
                r.returncode, 0,
                f"Junction nicht erstellbar: {r.stderr} {r.stdout}")
        self.assertTrue(junction.exists(), "Junction fehlt")

        from toolboxv2.utils.security.cryp import Code
        key = Code.generate_symmetric_key()

        async def _run():
            tr = _FakeTransport()
            idx = LocalIndex(str(_T / "a4_idx4.db"))
            await idx.init()
            try:
                await sync_folder(tr, root, PREFIX_GLOBAL, idx, key, "dev-a4")
            finally:
                await idx.close()
            return tr

        tr = asyncio.new_event_loop().run_until_complete(_run())
        leaked = [p for p in tr.puts if "secret" in p.lower()
                  or "verweis" in p.lower()]
        self.assertEqual(
            leaked, [],
            f"Datei ausserhalb des Sync-Ordners wurde hochgeladen: {leaked}"
            " - Walker folgt Junction/Symlink ohne Beschraenkung")


class TestA4Row5_DeletePropagates(unittest.TestCase):
    """Nr. 5: Datei auf A loeschen, sync (A), sync (B): Eintrag muss auf
    der Gegenseite verschwinden. Ist: Delete wird nie propagiert - der
    Transport hat nicht mal ein delete-Interface, der Eintrag bleibt."""

    def test_local_delete_removes_remote_entry(self):
        import asyncio

        from toolboxv2.mods.CloudM.LiveSync.data_api_sync import (
            PREFIX_GLOBAL, LocalIndex, sync_folder)

        root = _T / "a5_global" / "Agents" / "VFS" / "global"
        f = root / "gone.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("alt", encoding="utf-8")

        from toolboxv2.utils.security.cryp import Code
        key = Code.generate_symmetric_key()
        db = str(_T / "a5_idx.db")

        async def _run():
            tr = _FakeTransport()
            idx = LocalIndex(db)
            await idx.init()
            try:
                await sync_folder(tr, root, PREFIX_GLOBAL, idx, key, "dev-a5")
            finally:
                await idx.close()
            return tr

        tr = asyncio.new_event_loop().run_until_complete(_run())
        self.assertTrue(tr.entries, "Vorbedingung: erster Sync hat hochgeladen")
        f.unlink()  # auf A geloescht

        async def _resync():
            idx = LocalIndex(db)
            await idx.init()
            try:
                await sync_folder(tr, root, PREFIX_GLOBAL, idx, key, "dev-a5")
            finally:
                await idx.close()

        asyncio.new_event_loop().run_until_complete(_resync())
        remaining = [p for p in tr.entries if "gone" in p]
        self.assertEqual(
            remaining, [],
            f"Geloeschte Datei bleibt auf der Gegenseite: {remaining} - "
            "Delete-Propagation fehlt")


class _FakeAppForTransport:
    """Liefert run_http-Ergebnisse steuerbar (Data-API-Antworten)."""

    def __init__(self, response):
        self._response = response

    async def run_http(self, *a, **k):
        return self._response


class TestA4Row6_401MeansReauthRequired(unittest.TestCase):
    """Nr. 6: Data-API antwortet 401 (gecachter Schluessel ungueltig) -
    Zustand muss 'reauth_required' sein, nicht 'offline'.
    Ist: DataApiTransport wirft TransferError('offline')."""

    def test_401_raises_reauth_required(self):
        import asyncio

        from toolboxv2.mods.CloudM.LiveSync.data_api_sync import DataApiTransport
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import TransferError

        app = _FakeAppForTransport({
            "error": 7,  # irgendein Fehlercode
            "result": {},
            "info": {"exec_code": 401, "help_text": "Session fetch failed"},
        })
        tr = DataApiTransport(app)

        async def _call():
            await tr.list("livesync/global/")

        with self.assertRaises(TransferError) as cm:
            asyncio.new_event_loop().run_until_complete(_call())
        self.assertEqual(
            str(cm.exception), "reauth_required",
            f"401 muss reauth_required melden, nicht {cm.exception!r} - "
            "Geraet muss Konto neu verknuepfen statt nur offline zu melden")


class TestA4Row7_SyncKeyNotPlaintextInUserRecord(unittest.TestCase):
    """Nr. 7: Konto-Sync - auf dem Server darf kein Kontoschluessel im
    Klartext im Nutzerdatensatz liegen.
    Ist: get_sync_key erzeugt Code.generate_symmetric_key() und speichert
    ihn Klartext via _save_user (auth/api_session.py:172-195)."""

    def test_saved_user_record_holds_no_plaintext_sync_key(self):
        import asyncio

        import toolboxv2.mods.CloudM.auth.api_session as apis

        class _User:
            user_pass_sync = None
            def to_public_dict(self):
                return {"user_pass_sync": self.user_pass_sync}

        user = _User()
        saved = []

        async def _fake_validate(app=None, request=None, **k):
            from toolboxv2.utils.system.types import Result
            return Result.ok({"user_id": "u-a4"})

        async def _fake_load(app, uid):
            return user

        async def _fake_save(app, u):
            saved.append(u)

        async def _run():
            with patch.object(apis, "validate_session", _fake_validate), \
                 patch.object(apis, "_load_user", _fake_load), \
                 patch.object(apis, "_save_user", _fake_save):
                return await apis.get_sync_key(
                    app=object(), request=None, data={})

        res = asyncio.new_event_loop().run_until_complete(_run())
        self.assertFalse(res.is_error(),
                         f"get_sync_key schlug fehl: {res.get_info()}")

        # Der Schluessel ging an den Client - aber im gespeicherten
        # Nutzerdatensatz (das, was auf dem Server liegt) darf er nicht
        # im Klartext stehen.
        key = res.get().get("key")
        self.assertTrue(saved, "Vorbedingung: Nutzer wurde gespeichert")
        stored = saved[0]
        self.assertNotEqual(
            getattr(stored, "user_pass_sync", None), key,
            "Sync-Schluessel liegt Klartext im Nutzerdatensatz auf dem "
            "Server (user_pass_sync) - nur verschluesselt erlaubt")


if __name__ == "__main__":
    unittest.main()
