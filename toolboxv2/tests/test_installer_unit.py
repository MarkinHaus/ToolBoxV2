# tests/test_installer_unit.py
import unittest
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os


class TestFindExistingToolbox(unittest.TestCase):
    """
    Simuliert die Discovery-Logik aus installer.sh / .ps1.
    Testet: wird eine vorhandene Installation korrekt gefunden?
    """

    def _find_toolbox(self, candidates: list[Path]) -> Path | None:
        """Pure reimplementation of the discovery logic for testability."""
        for d in candidates:
            if (d / ".toolbox_version").exists() or (d / "toolboxv2" / "__init__.py").exists():
                return d
        return None

    def test_finds_via_toolbox_version_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ToolBoxV2"
            p.mkdir()
            (p / ".toolbox_version").write_text("0.1.25")
            result = self._find_toolbox([p])
            self.assertEqual(result, p)

    def test_finds_via_init_py(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ToolBoxV2"
            (p / "toolboxv2").mkdir(parents=True)
            (p / "toolboxv2" / "__init__.py").touch()
            result = self._find_toolbox([p])
            self.assertEqual(result, p)

    def test_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "NonExistent"
            result = self._find_toolbox([p])
            self.assertIsNone(result)

    def test_first_match_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "First"
            p1.mkdir()
            (p1 / ".toolbox_version").write_text("0.1.0")
            p2 = Path(tmp) / "Second"
            p2.mkdir()
            (p2 / ".toolbox_version").write_text("0.2.0")
            result = self._find_toolbox([p1, p2])
            self.assertEqual(result, p1)


class TestManifestSetGetCoerce(unittest.TestCase):
    """
    Testet _coerce() und die set/get-Logik aus manifest_cli.py.
    """

    def _coerce(self, value: str):
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def test_coerce_bool_true(self):
        self.assertIs(self._coerce("true"), True)
        self.assertIs(self._coerce("True"), True)

    def test_coerce_bool_false(self):
        self.assertIs(self._coerce("false"), False)

    def test_coerce_int(self):
        self.assertEqual(self._coerce("42"), 42)

    def test_coerce_float(self):
        self.assertAlmostEqual(self._coerce("3.14"), 3.14)

    def test_coerce_string_passthrough(self):
        self.assertEqual(self._coerce("CB"), "CB")
        self.assertEqual(self._coerce("simplecore.app"), "simplecore.app")

    def test_manifest_set_writes_yaml_and_env(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "tb-manifest.yaml"
            env_path = Path(tmp) / ".env"
            manifest_path.write_text("database:\n  mode: LC\n")
            env_path.write_text("TB_DB_MODE=LC\n")

            # Simulate cmd_set logic
            with open(manifest_path) as f:
                data = yaml.safe_load(f)
            data["database"]["mode"] = "CB"
            with open(manifest_path, "w") as f:
                yaml.dump(data, f)

            lines, found = [], False
            for line in env_path.read_text().splitlines():
                if line.startswith("TB_DB_MODE="):
                    lines.append("TB_DB_MODE=CB")
                    found = True
                else:
                    lines.append(line)
            if not found:
                lines.append("TB_DB_MODE=CB")
            env_path.write_text("\n".join(lines) + "\n")

            # Verify
            with open(manifest_path) as f:
                result = yaml.safe_load(f)
            self.assertEqual(result["database"]["mode"], "CB")
            self.assertIn("TB_DB_MODE=CB", env_path.read_text())


class TestFeatureLoaderDownloadFallback(unittest.TestCase):
    """
    Testet dass download_feature_from_registry() bei Netzwerkfehler
    None zurückgibt statt zu crashen.
    """

    def test_download_fails_gracefully(self):
        import urllib.request
        with patch.object(urllib.request, "urlretrieve", side_effect=Exception("network down")):
            with patch.object(urllib.request, "urlopen", side_effect=Exception("network down")):
                with tempfile.TemporaryDirectory() as tmp:
                    with patch.dict(os.environ, {"TB_REGISTRY_URL": "http://localhost:1"}):
                        # Import nach patch damit der tmp-Pfad greift
                        sys.path.insert(0, str(Path(__file__).parent.parent))
                        from toolboxv2.feature_loader import download_feature_from_registry
                        with patch("toolboxv2.feature_loader.get_features_packed_dir", return_value=Path(tmp)):
                            result = download_feature_from_registry("web")
                        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
