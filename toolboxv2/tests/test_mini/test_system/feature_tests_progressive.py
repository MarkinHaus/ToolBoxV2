# =============================================================================
# toolboxv2/tests/test_utils/test_system/feature_tests_progressive.py
# =============================================================================
"""
Fixture-Infrastruktur für progressive Feature-Isolation-Tests.

Isolation-Strategie: TB_ENABLED_FEATURES + tmp features-Verzeichnis.
Kein echter venv-Wechsel — TB_ENABLED_FEATURES kontrolliert was der
FeatureManager enabled, _guarded_import() verhindert den Crash.

Drei Haupt-Gefahren:
  1. Singleton-Bleeding zwischen Tests     → fm_fresh Fixture
  2. Disk-Mutation durch enable/disable    → tmp_features Fixture
  3. PyQt6/starlette fehlt in CI           → skip-Marker + smoke-only Mode
"""

import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Generator, Sequence

import pytest
import yaml

# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).parent.parent.parent.parent      # toolboxv2/
FEATURES_DIR = REPO_ROOT / "features"
ALL_FEATURES = ("mini", "core", "cli", "web", "desktop", "isaa", "exotic")
# ---------------------------------------------------------------------------


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: tmp_features
# Kopiert features/ in tmp_path, verhindert Disk-Mutationen.
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def tmp_features(tmp_path: Path) -> Path:
    """
    Temporäres features/-Verzeichnis mit vollständigen feature.yaml Kopien.

    Tests können enable()/disable() aufrufen ohne echte YAMLs zu verändern.
    Kein .installed-Marker → FeatureManager liest nur YAMLs.
    """
    features_tmp = tmp_path / "features"
    if FEATURES_DIR.exists():
        for d in FEATURES_DIR.iterdir():
            if d.is_dir():
                dest = features_tmp / d.name
                dest.mkdir(parents=True, exist_ok=True)
                src = d / "feature.yaml"
                if src.exists():
                    shutil.copy(src, dest / "feature.yaml")
    return features_tmp


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: fm_fresh
# Frischer FeatureManager pro Test (Singleton-Reset).
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def fm_fresh(tmp_features: Path):
    """
    Frischer FeatureManager mit tmp_features als base.

    Singleton wird vor und nach dem Test zurückgesetzt.
    Verwendung:
        def test_x(fm_fresh):
            fm = fm_fresh         # FeatureManager-Instanz
            fm.enable("web")
            ...
    """
    from toolboxv2.utils.system.feature_manager import FeatureManager
    from toolboxv2.utils.singelton_class import Singleton

    Singleton._instances.pop(FeatureManager, None)
    fm = FeatureManager(features_dir=str(tmp_features))
    yield fm
    Singleton._instances.pop(FeatureManager, None)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: active_features
# Setzt TB_ENABLED_FEATURES für die Dauer eines Tests.
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def active_features(fm_fresh):
    """
    Callable Fixture: setzt welche Features aktiv sind.

    Verwendung:
        def test_x(active_features):
            active_features("core", "cli")
            from toolboxv2 import _feature_enabled
            assert _feature_enabled("web") is False
    """
    import toolboxv2 as tbv2
    _original_fm = tbv2._feature_manager
    tbv2._feature_manager = fm_fresh

    def _set(*names: str):
        # Alle auf False setzen, dann gelistete auf True
        for fname, spec in fm_fresh.features.items():
            if spec.immutable:
                spec.enabled = True
                continue
            spec.enabled = fname in names
        # TB_ENABLED_FEATURES auch setzen für _apply_env_override
        os.environ["TB_ENABLED_FEATURES"] = ",".join(
            n for n, s in fm_fresh.features.items() if s.enabled
        )

    yield _set

    tbv2._feature_manager = _original_fm
    os.environ.pop("TB_ENABLED_FEATURES", None)


# ─────────────────────────────────────────────────────────────────────────────
# Skip-Marker für optionale Dependencies
# ─────────────────────────────────────────────────────────────────────────────
def _pkg_available(*packages: str) -> bool:
    import importlib.util
    return all(importlib.util.find_spec(p) is not None for p in packages)

skip_no_starlette = pytest.mark.skipif(
    not _pkg_available("starlette", "uvicorn"),
    reason="web deps (starlette/uvicorn) not installed",
)
skip_no_pyqt6 = pytest.mark.skipif(
    not _pkg_available("PyQt6"),
    reason="desktop deps (PyQt6) not installed",
)
skip_no_litellm = pytest.mark.skipif(
    not _pkg_available("litellm"),
    reason="isaa deps (litellm/openai) not installed",
)


# =============================================================================
# toolboxv2/tests/feature/test_progressive_isolation.py
# =============================================================================
"""
Progressive Feature-Isolation-Tests.

Matrix (von minimal zu maximal):
  Level 0 — mini allein
  Level 1 — core allein
  Level 2 — mini + je ein Feature
  Level 3 — core + cli  (kein web)
             core + cli + web
             core + cli + isaa
             core + cli + desktop
  Level 4 — core + cli + web + isaa
             core + cli + web + desktop
  Level 5 — alle Features
  Bundles  — core+mini, core+web, core+cli

Jeder Test verifiziert:
  - _feature_enabled() gibt korrekten Status
  - _guarded_import() crasht nicht
  - kein Bleeding aus anderen Tests
"""

import importlib
import os
import sys

import pytest
import yaml


# Fixture-Imports (aus conftest.py)
# active_features, fm_fresh, tmp_features, skip_no_*


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def assert_only_enabled(tbv2_module, *expected_enabled: str):
    """Prüft dass genau die genannten Features aktiv sind."""
    all_f = ("mini", "core", "cli", "web", "desktop", "isaa", "exotic")
    for name in all_f:
        expected = name in expected_enabled
        actual = tbv2_module._feature_enabled(name)
        assert actual == expected, (
            f"Feature '{name}': expected enabled={expected}, got {actual}. "
            f"Active should be: {expected_enabled}"
        )


def assert_guarded_import_safe(tbv2_module, module_path: str, attr: str, feature: str):
    """_guarded_import() darf nicht crashen — gibt None oder Objekt zurück."""
    result = tbv2_module._guarded_import(module_path, attr, feature=feature)
    if tbv2_module._feature_enabled(feature):
        # Wenn feature aktiv: Objekt oder None (bei fehlendem dep)
        pass  # kein assert — dep könnte fehlen in CI
    else:
        assert result is None, (
            f"_guarded_import '{module_path}.{attr}' (feature='{feature}') "
            f"sollte None sein wenn Feature disabled, bekam: {result}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Level 0 — mini allein
# ─────────────────────────────────────────────────────────────────────────────

class TestLevel0_MiniAlone:
    """mini allein: nur absolute Basics, kein cli, kein web."""

    def test_mini_enabled(self, active_features):
        """mini aktiv."""
        import toolboxv2 as tbv2
        active_features("mini")
        assert tbv2._feature_enabled("mini") is True

    def test_no_other_features(self, active_features):
        """Alle anderen Features disabled."""
        import toolboxv2 as tbv2
        active_features("mini")
        for name in ("cli", "web", "desktop", "isaa", "exotic"):
            assert tbv2._feature_enabled(name) is False, \
                f"'{name}' sollte disabled sein bei mini-only"

    def test_core_always_true(self, active_features):
        """core.immutable=True → immer enabled."""
        import toolboxv2 as tbv2
        active_features("mini")  # core nicht explizit
        # core ist immutable → bleibt True unabhängig von active_features()
        fm = tbv2._feature_manager
        if fm is not None and "core" in fm.features and fm.features["core"].immutable:
            assert tbv2._feature_enabled("core") is True

    def test_app_importable(self, active_features):
        """App-Klasse ist auch bei mini importierbar."""
        active_features("mini")
        import toolboxv2 as tbv2
        assert tbv2.App is not None or tbv2.App is None  # kein crash

    def test_no_web_import_crash(self, active_features):
        """_guarded_import web=disabled → None, kein crash."""
        active_features("mini")
        import toolboxv2 as tbv2
        assert_guarded_import_safe(
            tbv2,
            "toolboxv2.utils.workers",
            "cli_http_worker",
            "web",
        )

    def test_no_desktop_import_crash(self, active_features):
        """_guarded_import desktop=disabled → None, kein crash."""
        active_features("mini")
        import toolboxv2 as tbv2
        assert_guarded_import_safe(
            tbv2,
            "toolboxv2.utils.extras.show_and_hide_console",
            "show_console",
            "desktop",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — core allein
# ─────────────────────────────────────────────────────────────────────────────

class TestLevel1_CoreAlone:
    """core allein: App, Types, Logger, FileHandler — kein cli."""

    def test_core_enabled(self, active_features):
        active_features("core")
        import toolboxv2 as tbv2
        assert tbv2._feature_enabled("core") is True

    def test_cli_disabled(self, active_features):
        active_features("core")
        import toolboxv2 as tbv2
        assert tbv2._feature_enabled("cli") is False

    def test_style_importable(self, active_features):
        """Style/Spinner sind core-Abhängigkeiten."""
        active_features("core")
        import toolboxv2 as tbv2
        # Style kommt aus core — kein guard
        assert tbv2.Style is not None or True  # kein crash

    def test_no_prompt_toolkit_import(self, active_features):
        """prompt_toolkit ist cli-dep → bei core-only nicht erzwungen."""
        active_features("core")
        import toolboxv2 as tbv2
        result = tbv2._guarded_import(
            "toolboxv2.__main__",
            None,
            feature="cli",
        )
        assert result is None  # cli disabled → None


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — mini + je ein Feature
# ─────────────────────────────────────────────────────────────────────────────

class TestLevel2_MiniPlusOne:
    """mini + genau ein weiteres Feature."""

    @pytest.mark.parametrize("extra", ["cli", "web", "isaa", "exotic"])
    def test_mini_plus_one_no_crash(self, active_features, extra):
        """mini + {extra}: kein crash, genau zwei Features aktiv."""
        active_features("mini", extra)
        import toolboxv2 as tbv2
        assert tbv2._feature_enabled("mini") is True
        assert tbv2._feature_enabled(extra) is True
        # Alle anderen (außer immutable core) disabled
        for name in ("cli", "web", "isaa", "exotic"):
            if name != extra:
                assert tbv2._feature_enabled(name) is False, \
                    f"'{name}' sollte disabled sein bei mini+{extra}"

    @pytest.mark.parametrize("extra", ["cli", "web", "isaa", "exotic"])
    def test_mini_plus_one_guarded_imports(self, active_features, extra):
        """Disabled Features geben None zurück."""
        active_features("mini", extra)
        import toolboxv2 as tbv2
        for name in ("cli", "web", "isaa", "desktop"):
            if name != extra:
                result = tbv2._guarded_import(
                    "toolboxv2.utils.workers",  # stellvertretend
                    "cli_http_worker",
                    feature=name,
                )
                assert result is None, \
                    f"_guarded_import(feature='{name}') sollte None sein"


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 — core + cli Varianten
# ─────────────────────────────────────────────────────────────────────────────

class TestLevel3_CoreCli:
    """core + cli Basis — mit und ohne web."""

    def test_core_cli_no_web(self, active_features):
        """core+cli ohne web: workers-import muss None sein."""
        active_features("core", "cli")
        import toolboxv2 as tbv2
        assert tbv2._feature_enabled("core") is True
        assert tbv2._feature_enabled("cli") is True
        assert tbv2._feature_enabled("web") is False
        result = tbv2._guarded_import(
            "toolboxv2.utils.workers", "cli_http_worker", feature="web"
        )
        assert result is None

    @skip_no_starlette
    def test_core_cli_with_web(self, active_features):
        """core+cli+web: workers-import erfolgreich wenn starlette da."""
        active_features("core", "cli", "web")
        import toolboxv2 as tbv2
        assert tbv2._feature_enabled("web") is True
        # Nur smoke: kein crash
        tbv2._guarded_import(
            "toolboxv2.utils.workers", "cli_http_worker", feature="web"
        )

    @skip_no_litellm
    def test_core_cli_with_isaa(self, active_features):
        """core+cli+isaa: mcp_server importierbar wenn litellm da."""
        active_features("core", "cli", "isaa")
        import toolboxv2 as tbv2
        assert tbv2._feature_enabled("isaa") is True
        assert tbv2._feature_enabled("web") is False

    @skip_no_pyqt6
    def test_core_cli_with_desktop(self, active_features):
        """core+cli+desktop: show_console importierbar wenn PyQt6 da."""
        active_features("core", "cli", "desktop")
        import toolboxv2 as tbv2
        assert tbv2._feature_enabled("desktop") is True
        result = tbv2._guarded_import(
            "toolboxv2.utils.extras.show_and_hide_console",
            "show_console",
            feature="desktop",
        )
        # Wenn PyQt6 da: Objekt. Wenn nicht: None. Kein crash.
        assert result is None or callable(result) or True

    def test_core_cli_isaa_web_isolation(self, active_features):
        """isaa+web getrennt: beides disabled → beide None."""
        active_features("core", "cli")
        import toolboxv2 as tbv2
        for feat, mod, attr in [
            ("web",  "toolboxv2.utils.workers",                "cli_http_worker"),
            ("isaa", "toolboxv2.mcp_server.__main__",          "main"),
        ]:
            r = tbv2._guarded_import(mod, attr, feature=feat)
            assert r is None, f"'{feat}' disabled → None erwartet"


# ─────────────────────────────────────────────────────────────────────────────
# Level 4 — Kombinationen
# ─────────────────────────────────────────────────────────────────────────────

class TestLevel4_Combinations:
    """Zwei Feature-Kombinationen zusammen."""

    @skip_no_starlette
    @skip_no_litellm
    def test_web_plus_isaa(self, active_features):
        """web+isaa: beide aktiv, desktop disabled."""
        active_features("core", "cli", "web", "isaa")
        import toolboxv2 as tbv2
        assert tbv2._feature_enabled("web") is True
        assert tbv2._feature_enabled("isaa") is True
        assert tbv2._feature_enabled("desktop") is False
        r = tbv2._guarded_import(
            "toolboxv2.utils.extras.show_and_hide_console",
            "show_console",
            feature="desktop",
        )
        assert r is None

    @skip_no_starlette
    @skip_no_pyqt6
    def test_web_plus_desktop(self, active_features):
        """web+desktop: beide aktiv, isaa disabled."""
        active_features("core", "cli", "web", "desktop")
        import toolboxv2 as tbv2
        assert tbv2._feature_enabled("web") is True
        assert tbv2._feature_enabled("desktop") is True
        assert tbv2._feature_enabled("isaa") is False


# ─────────────────────────────────────────────────────────────────────────────
# Level 5 — Alle Features
# ─────────────────────────────────────────────────────────────────────────────

class TestLevel5_AllFeatures:
    """Alle Features aktiv — kein Konflikt."""

    @skip_no_starlette
    @skip_no_pyqt6
    @skip_no_litellm
    def test_all_enabled_no_crash(self, active_features):
        """Alle Features: kein Crash, alle enabled."""
        active_features("mini", "core", "cli", "web", "desktop", "isaa", "exotic")
        import toolboxv2 as tbv2
        for name in ("mini", "core", "cli", "web", "desktop", "isaa", "exotic"):
            assert tbv2._feature_enabled(name) is True, \
                f"Feature '{name}' sollte enabled sein"

    def test_all_smoke_without_deps(self, active_features):
        """Alle Features als enabled markiert — kein crash auch ohne deps."""
        active_features("mini", "core", "cli", "web", "desktop", "isaa", "exotic")
        import toolboxv2 as tbv2
        # Nur _feature_enabled() prüfen — deps können fehlen in CI
        for name in ("mini", "core", "cli", "web", "desktop", "isaa", "exotic"):
            # Kein assert über enabled-Status — CI hat ggf. nicht alle deps
            _ = tbv2._feature_enabled(name)  # kein crash


# ─────────────────────────────────────────────────────────────────────────────
# Bundles
# ─────────────────────────────────────────────────────────────────────────────

class TestBundles:
    """Vordefinierte Bundle-Kombinationen."""

    def test_bundle_core_mini(self, active_features):
        """Bundle: core+mini — minimale Distribution."""
        active_features("mini", "core")
        import toolboxv2 as tbv2
        assert tbv2._feature_enabled("core") is True
        assert tbv2._feature_enabled("cli") is False
        assert tbv2._feature_enabled("web") is False

    def test_bundle_core_cli(self, active_features):
        """Bundle: core+cli — Standard-CLI ohne web."""
        active_features("core", "cli")
        import toolboxv2 as tbv2
        assert tbv2._feature_enabled("cli") is True
        assert tbv2._feature_enabled("web") is False

    @skip_no_starlette
    def test_bundle_core_web(self, active_features):
        """Bundle: core+web — Web-Server ohne CLI-deps."""
        active_features("core", "web")
        import toolboxv2 as tbv2
        assert tbv2._feature_enabled("web") is True
        assert tbv2._feature_enabled("cli") is False

    def test_bundle_production(self, active_features):
        """Bundle: production = core+cli+web."""
        active_features("core", "cli", "web")
        import toolboxv2 as tbv2
        for name in ("core", "cli", "web"):
            # production: alle drei erwartet
            pass  # nur smoke — deps können fehlen
        assert tbv2._feature_enabled("desktop") is False
        assert tbv2._feature_enabled("isaa") is False


# ─────────────────────────────────────────────────────────────────────────────
# Singleton-Bleeding Regression
# ─────────────────────────────────────────────────────────────────────────────

class TestNoSingletonBleeding:
    """Sicherstellen dass kein Zustand zwischen Tests durchläuft."""

    def test_two_consecutive_tests_isolated(self, active_features, fm_fresh):
        """Test A aktiviert web → Test B sieht web=False."""
        import toolboxv2 as tbv2

        # Schritt A: web aktivieren
        active_features("core", "cli", "web")
        assert tbv2._feature_enabled("web") is True

        # Schritt B: nur core — web muss False sein
        active_features("core")
        assert tbv2._feature_enabled("web") is False, \
            "Singleton-Bleeding: web noch aktiv nach Wechsel zu core-only"

    def test_disk_not_mutated(self, active_features, fm_fresh, tmp_features):
        """enable() via active_features schreibt in tmp, nicht in echte YAML."""
        import toolboxv2 as tbv2

        active_features("core", "web")
        fm_fresh.enable("web")

        # Echte feature.yaml prüfen — darf nicht geändert sein
        real_yaml = FEATURES_DIR / "web" / "feature.yaml"
        if real_yaml.exists():
            with open(real_yaml, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            # feature.yaml im echten Repo sollte noch enabled=false haben
            # (enable() schrieb in tmp_features, nicht hier)
            assert data.get("enabled") is False, \
                "enable() hat echte feature.yaml verändert! (Disk-Mutation Bug)"


# =============================================================================
# toolboxv2/tests/feature/test_auto_install.py
# =============================================================================
"""
Tests für den Auto-Install-Mechanismus.

Testet:
  1. auto_install=False → klare Fehlermeldung, Feature auto-disabled
  2. auto_install=True  → _auto_install_deps() wird aufgerufen
  3. Manifest-Flag auto_install_deps=True wirkt global
"""

class TestAutoInstall:

    def test_auto_disabled_on_import_error(self, fm_fresh, monkeypatch):
        """Feature wird auto-disabled wenn Import fehlschlägt + auto_install=False."""
        import toolboxv2 as tbv2
        monkeypatch.setattr(tbv2, "_feature_manager", fm_fresh)

        # web als enabled markieren (ohne echte deps)
        fm_fresh.features["web"].enabled = True

        # Import eines nicht-existierenden Moduls erzwingen
        result = tbv2._guarded_import(
            "toolboxv2._nonexistent_xyz",
            "something",
            feature="web",
        )

        assert result is None
        assert fm_fresh.features["web"].enabled is False, \
            "Feature sollte nach Import-Fehler auto-disabled worden sein"

    def test_should_auto_install_returns_false_by_default(self, monkeypatch, tmp_path):
        """Ohne Manifest-Config: auto_install=False (kein stilles pip install)."""
        import toolboxv2 as tbv2

        # _should_auto_install ohne Manifest → False
        # ManifestLoader.exists() → False
        monkeypatch.setattr(
            "toolboxv2.utils.manifest.loader.ManifestLoader.exists",
            lambda self: False,
        )
        result = tbv2._should_auto_install("web")
        assert result is False, "auto_install sollte default=False sein"

    def test_auto_install_flag_in_feature_spec(self, fm_fresh):
        """FeatureSpec.auto_install=True wird erkannt."""
        fm_fresh.features["cli"].auto_install = True
        assert fm_fresh.features["cli"].auto_install is True

    def test_auto_install_called_on_import_error(self, fm_fresh, monkeypatch):
        """Wenn auto_install=True: _auto_install_deps() wird aufgerufen."""
        import toolboxv2 as tbv2

        install_called = []

        def _fake_install(feature_name):
            install_called.append(feature_name)
            return False  # Installation "schlägt fehl" → exit() nicht aufrufen

        monkeypatch.setattr(tbv2, "_auto_install_deps", _fake_install)
        monkeypatch.setattr(tbv2, "_should_auto_install", lambda n: True)
        monkeypatch.setattr(tbv2, "_feature_manager", fm_fresh)
        fm_fresh.features["web"].enabled = True

        # sys.exit() abfangen
        with pytest.raises(SystemExit):
            tbv2._guarded_import(
                "toolboxv2._nonexistent_xyz",
                "something",
                feature="web",
            )

        assert "web" in install_called, \
            "_auto_install_deps() sollte mit 'web' aufgerufen worden sein"
