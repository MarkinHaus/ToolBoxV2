"""
Betriebstests WP3 Auftrag 4 - Paket und Sync (7 Tests, Tabelle Nr. 1-7).

Belegt die 7 Betrieberbefunde aus WP3_ordnung_und_betriebstests_2026-11.md
auf Basis c36c3dc. Erwartung: alle 7 ROT auf der Basis, GRUEN nach Fix.
KEIN Produktcode wird hier geaendert (Betriebstests-Regel).
"""
import asyncio
import io
import json
import os
import shutil
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


class TestA4Row2_WheelHasMultipart(unittest.TestCase):
    """Nr. 2 gegen das gebaute Paket (Testleitfaden Regel 8, Fall 12): Wheel
    bauen, in eine FRISCHE venv installieren (`pip install <wheel>[web]`, nur
    was pyproject.toml deklariert), mit `python -I` aus fremdem Ordner
    importieren. Der Checkout-Test oben ist gruen, sobald `multipart` von Hand
    in der eigenen venv liegt; dieser Test misst die Deklaration.
    Braucht Netz zum Paketindex. Zeitlimits je Schritt."""

    def test_fresh_venv_install_enables_uploads(self):
        work = Path(tempfile.mkdtemp(prefix="tb_a09_whl_", dir=str(_T)))
        self.addCleanup(shutil.rmtree, str(work), True)
        whl_dir = work / "whl"
        env = dict(os.environ, HOME=str(work / "home"),
                   USERPROFILE=str(work / "home"), TB_DATA_DIR=str(work / "data"),
                   MOCK_EXTERNAL_SERVICES="true", IS_OFFLINE_DB="true")
        # Der Checkout darf nicht ueber PYTHONPATH in die frische venv lecken
        # (pip wuerde "toolboxv2" dort schon als installiert sehen).
        env.pop("PYTHONPATH", None)
        (work / "home").mkdir()
        (work / "data").mkdir()
        (work / "fremd").mkdir()

        def run(cmd, cwd, timeout):
            return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True,
                                  text=True, timeout=timeout, encoding="utf-8",
                                  errors="replace")

        r = run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps",
                 "--no-build-isolation", "-w", str(whl_dir)], REPO, 600)
        self.assertEqual(r.returncode, 0, f"pip wheel failed:\n{r.stderr[-800:]}")
        wheel = next(whl_dir.glob("*.whl"))

        r = run([sys.executable, "-m", "venv", str(work / "venv")], work, 300)
        self.assertEqual(r.returncode, 0, f"venv failed:\n{r.stderr[-500:]}")
        vpy = work / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

        r = run([str(vpy), "-m", "pip", "install", "--quiet", f"{wheel}[web]"],
                work, 900)
        self.assertEqual(r.returncode, 0, f"pip install failed:\n{r.stderr[-800:]}")

        r = run([str(vpy), "-I", "-c",
                 "import toolboxv2.utils.workers.server_worker as w;"
                 "print('MULTIPART_AVAILABLE', w.MULTIPART_AVAILABLE)"],
                work / "fremd", 300)
        self.assertIn("MULTIPART_AVAILABLE True", r.stdout,
                      f"Uploads im installierten Paket abgeschaltet:\n"
                      f"{r.stdout[-300:]}\n{r.stderr[-500:]}")


# ---------------------------------------------------------------------------
# Gemeinsame Bausteine fuer Nr. 3-6 (WP3-09)
# Einziges Double: App.run_http (Netz zu simplecore.app), siehe FakeUserDataApi.
# DataApiTransport, sync_folder, LocalIndex, export_account, import_account sind
# die echten Klassen/Funktionen.
# ---------------------------------------------------------------------------

def _wire(result) -> dict:
    """Result -> Antwortform wie sie ueber HTTP ankommt (JSON-Rundlauf)."""
    return json.loads(json.dumps(result.as_dict(), default=str))


class FakeUserDataApi:
    """Fake der Data-API hinter `app.run_http` (Prozessgrenze: Netz).

    Antwortformen sind die von CloudM.UserDataAPI (get_data/set_data/
    delete_data/list_data) und von App.run_http bei Netzfehler:
      ok            Result.ok(...).as_dict()
      404           Result.default_user_error(info=..., exec_code=404)
      kein Login    Result.default_user_error("Nicht authentifiziert", 401)
      kein Netz     Result.default_internal_error("Session fetch failed")
    Vertragstest gegen die echte UserDataAPI: TestA4_FakeContract.
    """

    def __init__(self):
        self.store = {}          # path -> (bytes, updated_at)
        self._clock = 1000.0
        self.mode = "ok"         # "ok" | "offline" | "expired"
        self.failing = set()     # Funktionsnamen, die "kein Netz" liefern
        self.calls = []          # (funktion, pfad/prefix)

    def _tick(self) -> float:
        self._clock += 1.0
        return self._clock

    def seed(self, path: str, data: dict) -> None:
        self.store[path] = (json.dumps(data).encode(), self._tick())

    async def run_http(self, mod_function_name, function_name=None, args_=None,
                       kwargs_=None, method="GET", *args, **kwargs):
        from toolboxv2 import Result
        assert mod_function_name == "CloudM.UserDataAPI", mod_function_name
        assert kwargs.get("scope") == "private", kwargs
        self.calls.append((function_name, kwargs.get("path") or kwargs.get("prefix")))
        if self.mode == "offline" or function_name in self.failing:
            return _wire(Result.default_internal_error(info="Session fetch failed"))
        if self.mode == "expired":
            return _wire(Result.default_user_error(
                info="Nicht authentifiziert", exec_code=401))
        return _wire(getattr(self, "_" + function_name)(**kwargs))

    def _list_data(self, prefix="", scope="private"):
        from toolboxv2 import Result
        return Result.ok(data=[
            {"path": p, "size": len(b), "updated_at": ts}
            for p, (b, ts) in sorted(self.store.items()) if p.startswith(prefix)])

    def _get_data(self, path, scope="private"):
        from toolboxv2 import Result
        if path not in self.store:
            return Result.default_user_error(info="Daten nicht gefunden", exec_code=404)
        return Result.ok(data=json.loads(self.store[path][0].decode()))

    def _set_data(self, path, data, scope="private"):
        from toolboxv2 import Result
        raw = json.dumps(data).encode()
        self.store[path] = (raw, self._tick())
        return Result.ok(data={"path": path, "size": len(raw),
                               "checksum": "x", "encrypted": False})

    def _delete_data(self, path, scope="private"):
        from toolboxv2 import Result
        if path not in self.store:
            return Result.default_user_error(info="Nicht gefunden", exec_code=404)
        del self.store[path]
        return Result.ok(data_info=f"Gelöscht: {path}")


def _tmpdir(case: unittest.TestCase) -> Path:
    d = Path(tempfile.mkdtemp(prefix="tb_a09_", dir=str(_T)))
    case.addCleanup(shutil.rmtree, str(d), True)
    return d


def _real_app():
    from toolboxv2 import get_app
    return get_app("a09-sync-test")


def _transport(case: unittest.TestCase, fake: FakeUserDataApi):
    """Echte App + echter DataApiTransport; nur run_http ist ersetzt."""
    from toolboxv2.mods.CloudM.LiveSync.data_api_sync import DataApiTransport
    app = _real_app()
    p1 = patch.object(app, "run_http", fake.run_http)   # Netzgrenze simplecore.app
    p2 = patch.object(app.session, "base", "https://simplecore.app")
    for p in (p1, p2):
        p.start()
        case.addCleanup(p.stop)
    return DataApiTransport(app)


class _Device:
    """Ein Geraet: eigener Ordner, eigener echter LocalIndex, gemeinsamer Key."""

    def __init__(self, case, name, key, fake):
        self.name = name
        self.root = _tmpdir(case)
        self.dir = self.root / "global"
        self.dir.mkdir()
        self.db = str(self.root / "index.db")
        self.key = key
        self.transport = _transport(case, fake)

    def sync(self, directory=None, prefix=None):
        from toolboxv2.mods.CloudM.LiveSync.data_api_sync import (
            PREFIX_GLOBAL, LocalIndex, sync_folder)

        async def _go():
            idx = LocalIndex(self.db)
            await idx.init()
            try:
                return await sync_folder(
                    self.transport, directory or self.dir,
                    prefix or PREFIX_GLOBAL, idx, self.key, self.name)
            finally:
                await idx.close()
        return asyncio.run(_go())

    def write(self, rel, text):
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p


def _link_dir(link: Path, target: Path) -> None:
    """Verzeichnis-Verweis nach aussen: NTFS-Junction unter Windows, sonst
    Symlink. Beide Faelle pruefen dasselbe Soll."""
    if os.name == "nt":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"Junction nicht erstellbar: {r.stderr} {r.stdout}"
    else:
        os.symlink(str(target), str(link), target_is_directory=True)


def _key():
    from toolboxv2.utils.security.cryp import Code
    return Code.generate_symmetric_key()


class TestA4Row3_ConflictCopySurvivesReExport(unittest.TestCase):
    """Nr. 3: Zwei Geraete aendern denselben Konto-Wert. Auf B entsteht beim
    Abgleich eine Konfliktkopie (lokaler Wert). Der naechste Lauf
    (export_account -> sync_folder -> import_account, wie sync_all) darf sie
    nicht loeschen: der lokale Wert bleibt erhalten.
    Ist: export_account loescht 'vanished keys' inkl. Konfliktkopien."""

    def test_conflict_copy_survives_next_cycle(self):
        from toolboxv2.mods.CloudM.LiveSync.account_bridge import (
            export_account, import_account)
        from toolboxv2.mods.CloudM.LiveSync.conflict import is_conflict_copy
        from toolboxv2.mods.CloudM.LiveSync.data_api_sync import PREFIX_ACCOUNT
        from toolboxv2.mods.CloudM.auth.models import UserData

        fake = FakeUserDataApi()
        key = _key()
        a = _Device(self, "dev-a", key, fake)
        b = _Device(self, "dev-b", key, fake)
        ua = UserData(user_id="usr_a09", username="u", email="u@example.test",
                      settings={"theme": "dark"})
        ub = UserData(user_id="usr_a09", username="u", email="u@example.test",
                      settings={"theme": "dark"})

        def cycle(dev, user):
            mirror = dev.root / "mirror"
            export_account(user, mirror)
            out = dev.sync(mirror, PREFIX_ACCOUNT)
            import_account(user, mirror)
            return mirror, out

        cycle(a, ua)                       # A legt theme=dark an
        cycle(b, ub)                       # B holt es (gleicher Inhalt)
        ua.settings["theme"] = "light"     # A aendert und synchronisiert
        cycle(a, ua)
        ub.settings["theme"] = "blue"      # B aendert dasselbe Feld parallel
        mirror_b, out = cycle(b, ub)       # Konflikt: remote (light) gewinnt
        self.assertEqual(out["conflicts"], ["settings/theme.json"], out)
        self.assertEqual(ub.settings["theme"], "light")

        copies = [p for p in (mirror_b / "settings").iterdir()
                  if is_conflict_copy(p.name)]
        self.assertEqual(len(copies), 1, "Vorbedingung: eine Konfliktkopie")
        self.assertEqual(json.loads(copies[0].read_text("utf-8")), "blue")

        cycle(b, ub)                       # naechster regulaerer Lauf auf B

        survivors = [p for p in (mirror_b / "settings").iterdir()
                     if is_conflict_copy(p.name)]
        self.assertEqual(
            [p.name for p in survivors], [copies[0].name],
            "Konfliktkopie wurde vom naechsten Export geloescht - der lokale "
            "Wert ('blue') ist verloren")
        self.assertEqual(json.loads(survivors[0].read_text("utf-8")), "blue")


    def test_mod_data_conflict_copy_survives_but_vanished_key_goes(self):
        """mod_data/: Konfliktkopie bleibt, ein wirklich entfallener Schluessel
        wird weiterhin entfernt (der Loesch-Block bleibt wirksam)."""
        from toolboxv2.mods.CloudM.LiveSync.account_bridge import export_account
        from toolboxv2.mods.CloudM.LiveSync.conflict import save_conflict_copy
        from toolboxv2.mods.CloudM.auth.models import UserData

        mirror = _tmpdir(self) / "mirror"
        user = UserData(user_id="u", username="u", email="u@example.test",
                        mod_data={"m": {"a": 1, "weg": 2}})
        export_account(user, mirror)
        conflict = save_conflict_copy(str(mirror / "mod_data"), "m/a.json", "dev-x")
        (mirror / "mod_data" / "m" / "a.json").write_text("1", encoding="utf-8")
        del user.mod_data["m"]["weg"]
        export_account(user, mirror)
        self.assertTrue(Path(conflict).is_file(), "Konfliktkopie in mod_data geloescht")
        self.assertFalse((mirror / "mod_data" / "m" / "weg.json").exists())


class TestA4Row4_SymlinkStaysInside(unittest.TestCase):
    """Nr. 4: Verweis (Junction/Symlink) im globalen ISAA-Ordner, der nach
    aussen zeigt: nichts ausserhalb wird gelesen/hochgeladen oder
    geschrieben. Unter Windows Junction (`mklink /J`), sonst Symlink
    (zusaetzlich ein Datei-Symlink; unter Windows braucht der Rechte, dort
    entfaellt nur dieser Teil).
    Ist: Walker folgt dem Verweis."""

    def setUp(self):
        self.fake = FakeUserDataApi()
        self.dev = _Device(self, "dev-a4", _key(), self.fake)
        self.outside = _tmpdir(self) / "aussen"
        self.outside.mkdir()
        (self.outside / "secret.txt").write_text("GEHEIM", encoding="utf-8")
        _link_dir(self.dev.dir / "verweis", self.outside)
        if os.name != "nt":
            os.symlink(str(self.outside / "secret.txt"),
                       str(self.dev.dir / "datei.txt"))
        self.dev.write("normal.txt", "ok")

    def test_link_outside_is_not_uploaded(self):
        out = self.dev.sync()
        leaked = [p for p in self.fake.store
                  if "secret" in p or "verweis" in p or "datei" in p]
        self.assertEqual(leaked, [], f"Datei ausserhalb hochgeladen: {leaked}")
        self.assertEqual(out["pushed"], ["normal.txt"])
        if os.name != "nt":
            self.assertIn("datei.txt", out["skipped"], "Verweis nicht gemeldet")

    def test_remote_entry_under_link_is_not_written_outside(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import encrypt_bytes
        from toolboxv2.mods.CloudM.LiveSync.data_api_sync import _payload
        self.fake.seed("livesync/global/verweis/neu.txt",
                       _payload(encrypt_bytes(b"VON-AUSSEN", self.dev.key)))
        out = self.dev.sync()
        self.assertEqual(sorted(p.name for p in self.outside.iterdir()),
                         ["secret.txt"],
                         "Remote-Eintrag wurde ueber den Verweis nach aussen geschrieben")
        self.assertIn("verweis/neu.txt", out["skipped"], "Verweis nicht gemeldet")


class TestA4Row5_DeletePropagates(unittest.TestCase):
    """Nr. 5: Datei auf A loeschen, sync (A), sync (B): sie verschwindet auf
    dem Server und auf B. Zwei Geraete, ein gemeinsamer Fake der Data-API.
    Ist: Delete wird nie propagiert (der Transport hat kein delete)."""

    def test_delete_on_a_reaches_b(self):
        fake = FakeUserDataApi()
        key = _key()
        a = _Device(self, "dev-a", key, fake)
        b = _Device(self, "dev-b", key, fake)
        for n in ("gone.txt", "k1.txt", "k2.txt", "k3.txt"):
            a.write(n, "inhalt " + n)
        a.sync()
        b.sync()
        self.assertTrue((b.dir / "gone.txt").is_file(), "Vorbedingung: B hat die Datei")

        (a.dir / "gone.txt").unlink()
        a.sync()
        self.assertNotIn("livesync/global/gone.txt", fake.store,
                         "Geloeschte Datei bleibt auf dem Server")
        b.sync()
        self.assertFalse((b.dir / "gone.txt").exists(),
                         "Geloeschte Datei bleibt auf B")
        self.assertEqual(sorted(p.name for p in b.dir.iterdir()
                                if p.is_file()), ["k1.txt", "k2.txt", "k3.txt"])


class TestA09_DeletionsTwoDevices(unittest.TestCase):
    """WP3-09 2.4: Loeschungen zwischen zwei Geraeten (zwei Datenordner, zwei
    LocalIndex, ein gemeinsamer FakeUserDataApi hinter run_http)."""

    def setUp(self):
        self.fake = FakeUserDataApi()
        key = _key()
        self.a = _Device(self, "dev-a", key, self.fake)
        self.b = _Device(self, "dev-b", key, self.fake)

    def _seed_both(self, names):
        for n in names:
            self.a.write(n, "inhalt " + n)
        self.a.sync()
        self.b.sync()
        for n in names:
            self.assertTrue((self.b.dir / n).is_file(), f"Vorbedingung: {n} auf B")

    @staticmethod
    def _files(dev):
        return sorted(p.name for p in dev.dir.iterdir() if p.is_file())

    def test_deleting_the_only_file_propagates(self):
        self._seed_both(["einzig.txt"])
        (self.a.dir / "einzig.txt").unlink()
        out = self.a.sync()
        self.assertEqual(out["deleted_remote"], ["einzig.txt"])
        self.assertEqual(self.fake.store, {})
        self.b.sync()
        self.assertEqual(self._files(self.b), [])

    def test_remote_deletion_moves_local_file_to_sync_trash(self):
        self._seed_both(["gone.txt", "k1.txt", "k2.txt"])
        (self.a.dir / "gone.txt").unlink()
        self.a.sync()
        out = self.b.sync()
        self.assertEqual(out["deleted_local"], ["gone.txt"])
        self.assertEqual(self._files(self.b), ["k1.txt", "k2.txt"])
        trash = list((self.b.dir / ".sync-trash").iterdir())
        self.assertEqual(len(trash), 1)
        self.assertEqual(trash[0].read_text("utf-8"), "inhalt gone.txt")

    def test_local_change_beats_remote_deletion(self):
        self._seed_both(["x.txt", "k1.txt", "k2.txt"])
        (self.a.dir / "x.txt").unlink()
        self.a.sync()                                    # Server: x.txt weg
        (self.b.dir / "x.txt").write_text("B hat geaendert", encoding="utf-8")
        out = self.b.sync()
        self.assertEqual(out["pushed"], ["x.txt"])
        self.assertEqual(out["deleted_local"], [])
        self.assertEqual((self.b.dir / "x.txt").read_text("utf-8"), "B hat geaendert")
        self.assertIn("livesync/global/x.txt", self.fake.store)
        self.a.sync()
        self.assertEqual((self.a.dir / "x.txt").read_text("utf-8"), "B hat geaendert")

    def test_remote_change_beats_local_deletion(self):
        self._seed_both(["x.txt", "k1.txt", "k2.txt"])
        (self.a.dir / "x.txt").unlink()                  # A loescht, noch kein Sync
        (self.b.dir / "x.txt").write_text("B hat geaendert", encoding="utf-8")
        self.b.sync()
        out = self.a.sync()
        self.assertEqual(out["pulled"], ["x.txt"])
        self.assertEqual(out["deleted_remote"], [])
        self.assertEqual((self.a.dir / "x.txt").read_text("utf-8"), "B hat geaendert")

    def test_failing_list_deletes_nothing(self):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import TransferError
        self._seed_both(["a.txt", "b.txt", "c.txt", "d.txt"])
        (self.a.dir / "a.txt").unlink()
        self.fake.failing = {"list_data"}
        with self.assertRaises(TransferError):
            self.a.sync()
        with self.assertRaises(TransferError):
            self.b.sync()
        self.assertEqual(len(self.fake.store), 4, "Server-Eintrag trotz Listenfehler geloescht")
        self.assertEqual(self._files(self.b), ["a.txt", "b.txt", "c.txt", "d.txt"])
        self.fake.failing = set()
        self.a.sync()                                    # danach wirkt die Loeschung
        self.assertNotIn("livesync/global/a.txt", self.fake.store)

    def test_mass_delete_is_blocked_locally(self):
        names = [f"f{i}.txt" for i in range(6)]
        self._seed_both(names)
        for n in names[:4]:
            (self.a.dir / n).unlink()
        out = self.a.sync()
        self.assertEqual(out["error"], "mass_delete_blocked")
        self.assertEqual(out["blocked"], names[:4])
        self.assertEqual(out["deleted_remote"], [])
        self.assertEqual(len(self.fake.store), 6)

    def test_empty_server_listing_does_not_wipe_the_device(self):
        names = [f"f{i}.txt" for i in range(6)]
        self._seed_both(names)
        self.fake.store.clear()                          # Server liefert leer
        out = self.b.sync()
        self.assertEqual(out["error"], "mass_delete_blocked")
        self.assertEqual(self._files(self.b), names)
        self.assertEqual(out["deleted_local"], [])

    def test_folder_replaced_by_outside_link_is_not_read_as_deleted(self):
        self._seed_both(["sub/a.txt", "k1.txt"])
        shutil.rmtree(str(self.a.dir / "sub"))
        outside = _tmpdir(self) / "aussen"
        outside.mkdir()
        (outside / "a.txt").write_text("von aussen", encoding="utf-8")
        _link_dir(self.a.dir / "sub", outside)
        out = self.a.sync()
        self.assertEqual(out["deleted_remote"], [])
        self.assertIn("livesync/global/sub/a.txt", self.fake.store)
        self.assertEqual((outside / "a.txt").read_text("utf-8"), "von aussen")


class TestA09_SkippedNamesAreVisible(unittest.TestCase):
    """WP3-09 2.5: Dateinamen, die nicht synchronisiert werden koennen (":"),
    stehen in out["skipped"] und im Log - nicht mehr still uebersprungen.
    Ein lokales ":"-File gibt es nur auf Dateisystemen, die den Namen
    erlauben (nicht unter Windows); der Remote-Fall laeuft ueberall."""

    LOGGER = "toolboxv2.mods.CloudM.LiveSync.data_api_sync"

    def setUp(self):
        self.fake = FakeUserDataApi()
        self.dev = _Device(self, "dev-a5", _key(), self.fake)

    def test_local_name_with_colon_is_reported(self):
        if os.name == "nt":
            self.dev.write("ok.txt", "ok")   # Windows erlaubt ':' im Namen nicht
            self.assertEqual(self.dev.sync()["skipped"], [])
            return
        self.dev.write("ok.txt", "ok")
        self.dev.write("a:b.txt", "kolon")
        with self.assertLogs(self.LOGGER, level="WARNING") as logs:
            out = self.dev.sync()
        self.assertEqual(out["skipped"], ["a:b.txt"])
        self.assertEqual(out["pushed"], ["ok.txt"])
        self.assertTrue(any("a:b.txt" in m for m in logs.output), logs.output)

    def test_remote_name_with_colon_is_reported(self):
        from toolboxv2.mods.CloudM.LiveSync.crypto import encrypt_bytes
        from toolboxv2.mods.CloudM.LiveSync.data_api_sync import _payload
        self.fake.seed("livesync/global/x:y.txt",
                       _payload(encrypt_bytes(b"x", self.dev.key)))
        with self.assertLogs(self.LOGGER, level="WARNING") as logs:
            out = self.dev.sync()
        self.assertEqual(out["skipped"], ["livesync/global/x:y.txt"])
        self.assertEqual(sorted(p.name for p in self.dev.dir.iterdir()), [])
        self.assertTrue(any("x:y.txt" in m for m in logs.output), logs.output)


class TestA4Row6_401MeansReauthRequired(unittest.TestCase):
    """Nr. 6: Data-API antwortet 401 (Sitzung abgelaufen) -> 'reauth_required';
    kein Netz ('Session fetch failed') -> 'offline'. Antwortformen wie sie
    App.run_http / CloudM.UserDataAPI wirklich liefern.
    Ist: beides wird zu TransferError('offline')."""

    def _list_error(self, mode):
        from toolboxv2.mods.CloudM.LiveSync.minio_helper import TransferError
        fake = FakeUserDataApi()
        fake.mode = mode
        dev = _Device(self, "dev-a6", _key(), fake)
        with self.assertRaises(TransferError) as cm:
            dev.sync()
        return str(cm.exception)

    def test_401_raises_reauth_required(self):
        self.assertEqual(self._list_error("expired"), "reauth_required")

    def test_no_network_raises_offline(self):
        self.assertEqual(self._list_error("offline"), "offline")


class TestA4_FakeContract(unittest.TestCase):
    """Regel 10: dieselben Faelle gegen FakeUserDataApi und gegen die echte
    CloudM.UserDataAPI (echte App, Temp-Datenbank, echter JWT). Verglichen
    wird die Antwortform (Result.as_dict()).

    Grenze (belegt, Bericht A09): Gegen die echte Funktion laufen nur Faelle
    OHNE gespeicherte Daten. Offline gibt es keinen Speicher, in dem Daten
    liegen koennten: `from mobile_db import MobileDB` (scoped_storage.py:38)
    ist nicht importierbar, MinIO fehlt, und `set_data` bricht in dieser
    Basis immer ab (ScopedBlobStorage.write() kennt kein `owner_id`,
    UserDataAPI.py set_data). Die Erfolgsformen von list/get/set/delete im
    Fake sind dem Quelltext der UserDataAPI entnommen, nicht gemessen."""

    @staticmethod
    def _shape(res: dict):
        r = res["result"]
        return (res["error"], res["info"]["exec_code"], res["info"]["help_text"],
                r["data_type"], r["data"], r["data_info"])

    def _fake_backend(self):
        fake = FakeUserDataApi()

        async def call(fn, **kw):
            return await fake.run_http("CloudM.UserDataAPI", fn, method="POST",
                                       scope="private", **kw)

        async def anon(fn, **kw):
            fake.mode = "expired"
            return await call(fn, **kw)
        return call, anon

    def _real_backend(self):
        import uuid
        from types import SimpleNamespace
        from toolboxv2.mods.CloudM import UserDataAPI as U
        from toolboxv2.mods.CloudM.auth.jwt_tokens import _generate_tokens
        from toolboxv2.mods.CloudM.auth.models import UserData
        app = _real_app()
        user = UserData(user_id="usr_a09_" + uuid.uuid4().hex[:8], username="a09",
                        email="a09@example.test")
        token = _generate_tokens(user)["access_token"]
        req = SimpleNamespace(
            request={"headers": {"authorization": "Bearer " + token}}, session={})
        anon_req = SimpleNamespace(request={"headers": {}}, session={})
        self.addCleanup(U._storage_providers.pop, user.user_id, None)

        async def call(fn, **kw):
            return _wire(await getattr(U, fn)(app, req, scope="private", **kw))

        async def anon(fn, **kw):
            return _wire(await getattr(U, fn)(app, anon_req, scope="private", **kw))
        return call, anon

    async def _cases(self, call, anon):
        return {
            "kein_login_list": await anon("list_data", prefix="livesync/"),
            "list_leer": await call("list_data", prefix="livesync/global/"),
            "get_fehlt": await call("get_data", path="livesync/global/nope.txt"),
            "delete_fehlt": await call("delete_data", path="livesync/global/nope.txt"),
        }

    def test_fake_has_the_shapes_of_the_real_api(self):
        fake_out = asyncio.run(self._cases(*self._fake_backend()))
        real_out = asyncio.run(self._cases(*self._real_backend()))
        self.assertEqual(fake_out.keys(), real_out.keys())
        for name in fake_out:
            with self.subTest(fall=name):
                self.assertEqual(
                    self._shape(fake_out[name]), self._shape(real_out[name]),
                    f"Fake weicht ab: fake={fake_out[name]} real={real_out[name]}")


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
