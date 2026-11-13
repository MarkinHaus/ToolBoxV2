"""R1: TB_COOKIE_SECRET auto-generation on first start (production).

Regelwerk 2026-11-11 Version 3, Task R1: load_config must never crash in
production without a secret; the secret lives in the ONE .env at
_state_dir.env_file() (packaged: TB_INSTALL_DIR/.env); empty
`TB_JWT_SECRET=` lines (installer .env) must be filled in place by the
single writer save_env_file.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _packaged(install_dir):
    import toolboxv2._state_dir as sd
    os.environ["TB_INSTALL_DIR"] = str(install_dir)
    return mock.patch.object(sd, "is_packaged_install", lambda: True)


class SecretAutogenTest(unittest.TestCase):
    KEYS = ("TB_ENV", "TAURI_ENV", "TB_JWT_SECRET", "TB_COOKIE_SECRET",
            "TB_CONFIG", "TB_DATA_DIR", "TB_INSTALL_DIR")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self.KEYS}
        for k in self.KEYS:
            os.environ.pop(k, None)
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.install = self.tmp / "install"          # TB_INSTALL_DIR (.env lives HERE)
        self.data = self.tmp / "data"                # TB_DATA_DIR (no .env allowed)
        os.environ["TB_DATA_DIR"] = str(self.data)
        (self.tmp / "empty.yaml").write_text("{}\n", encoding="utf-8")
        os.environ["TB_CONFIG"] = str(self.tmp / "empty.yaml")

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def test_production_config_autogenerates_stable_secret(self):
        from toolboxv2.utils.workers.config import load_config
        os.environ["TB_ENV"] = "production"
        os.environ["TB_JWT_SECRET"] = ""
        os.environ["TB_COOKIE_SECRET"] = ""
        with _packaged(self.install):
            cfg1 = load_config()
            secret1 = cfg1.session.cookie_secret
            self.assertTrue(secret1 and len(secret1) >= 64,
                            f"cookie_secret must be generated, got {secret1!r}")

            # simulate restart: fresh process, load_dotenv would set "" again
            os.environ["TB_JWT_SECRET"] = ""
            os.environ["TB_COOKIE_SECRET"] = ""
            cfg2 = load_config()
            self.assertEqual(secret1, cfg2.session.cookie_secret)

            env_file = self.install / ".env"
            self.assertTrue(env_file.exists(), "secret must be persisted to the one .env")
            content = env_file.read_text(encoding="utf-8", errors="surrogateescape")
            self.assertIn(f"TB_JWT_SECRET={secret1}", content)
            self.assertFalse((self.data / ".env").exists(),
                             "TB_DATA_DIR must never get a second .env")

    def test_empty_env_lines_do_not_block_persistence(self):
        from toolboxv2.init_onboarding import ensure_secret
        self.install.mkdir(parents=True, exist_ok=True)
        env_file = self.install / ".env"
        env_file.write_text('TB_ENV="production"\nTB_JWT_SECRET=\nTB_COOKIE_SECRET=\n',
                            encoding="utf-8")
        with _packaged(self.install):
            s1 = ensure_secret(persist=True)
            self.assertTrue(s1)
            content = env_file.read_text(encoding="utf-8", errors="surrogateescape")
            self.assertIn(f"TB_JWT_SECRET={s1}", content,
                          "empty line must be filled in place")
            self.assertIn(f"TB_COOKIE_SECRET={s1}", content)
            self.assertIn('TB_ENV="production"', content,
                          "unrelated lines must survive")

            # restart: env empty again, secret must come from the .env file
            os.environ.pop("TB_JWT_SECRET", None)
            os.environ.pop("TB_COOKIE_SECRET", None)
            s2 = ensure_secret(persist=True)
            self.assertEqual(s1, s2)

    def test_packaged_install_persists_outside_site_packages(self):
        import toolboxv2.init_onboarding as io_mod
        fake_sp = self.tmp / "site-packages" / "toolboxv2"
        fake_sp.mkdir(parents=True)
        user_dir = self.tmp / "userdir"
        with mock.patch.object(io_mod, "tb_root_dir", fake_sp), _packaged(user_dir):
            path = io_mod._writable_env_path()
            self.assertEqual(Path(user_dir) / ".env", Path(path),
                             "packaged installs must write to user dir")
            self.assertNotIn("site-packages", str(path))


if __name__ == "__main__":
    unittest.main()
