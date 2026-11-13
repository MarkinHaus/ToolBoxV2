import unittest, tempfile, sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, r"C:\Users\Markin\Workspace\ISAABOX\Toolboxv2")

from toolboxv2.utils.workers.fast import endpoint


class TestDistDirManifestBase(unittest.TestCase):
    def test_dist_dir_uses_manifest_base_dir(self):
        # R8b: Manifest liegt in manifest_base_dir() (nicht tb_root_dir);
        # absoluter dist_dir aus dem Manifest gewinnt.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dist = tmp / "from-manifest"
            dist.mkdir()
            mbase = tmp / "manifest-home"
            mbase.mkdir()
            want = str(dist).replace("\\", "/")
            (mbase / "tb-manifest.yaml").write_text(
                "paths:\n  dist_dir: " + want + "\n", encoding="utf-8")
            with mock.patch("toolboxv2._state_dir.manifest_base_dir", lambda: mbase):
                got = endpoint.dist_dir()
            self.assertEqual(dist, got)

    def test_dist_dir_relative_manifest_value_resolved_against_root(self):
        # relativer dist_dir wird gegen tb_root_dir aufgeloest, nicht gegen
        # die Manifest-Base; Kandidat muss existieren, sonst Fallback.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fake_root = tmp / "fake-root"
            rel = fake_root / "rel-dist"
            rel.mkdir(parents=True)
            mbase = tmp / "manifest-home"
            mbase.mkdir()
            (mbase / "tb-manifest.yaml").write_text(
                "paths:\n  dist_dir: rel-dist\n", encoding="utf-8")
            with mock.patch("toolboxv2._state_dir.manifest_base_dir", lambda: mbase), \
                 mock.patch("toolboxv2.tb_root_dir", fake_root):
                got = endpoint.dist_dir()
            self.assertEqual(rel, got)

    def test_dist_dir_fallback_without_manifest(self):
        # kein Manifest -> Fallback <tb_root_dir>/dist bleibt unberuehrt
        with tempfile.TemporaryDirectory() as tmp:
            mbase = Path(tmp) / "manifest-home"
            with mock.patch("toolboxv2._state_dir.manifest_base_dir", lambda: mbase):
                got = endpoint.dist_dir()
            from toolboxv2 import tb_root_dir
            self.assertEqual(Path(str(tb_root_dir)) / "dist", got)


if __name__ == "__main__":
    unittest.main()
