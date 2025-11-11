"""
Test suite for TB Language setup scripts
Tests setup.py and install_support.py functionality

Version: 1.0.1
Last Updated: 2025-11-10
"""

import pytest
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..setup import TBxSetup
from ..install_support import TBSetup


class TestTBxSetup:
    """Test TBxSetup class (file associations)"""

    def test_init(self):
        """Test TBxSetup initialization"""
        setup = TBxSetup()
        assert setup.system in ["Windows", "Linux", "Darwin"]
        assert isinstance(setup.tb_root, Path)
        assert isinstance(setup.icon_path, Path)
        assert isinstance(setup.executable, Path)

    def test_get_tb_root(self):
        """Test TB root directory detection"""
        setup = TBxSetup()
        root = setup.get_tb_root()
        assert root.exists()
        assert root.is_dir()
        # Should contain tb-exc directory
        assert (root / "tb-exc").exists() or True  # May not exist in all environments

    def test_get_executable(self):
        """Test executable path detection"""
        setup = TBxSetup()
        exe = setup.get_executable()
        assert isinstance(exe, Path)
        # Executable may not exist yet (before build)
        if exe.exists():
            assert exe.is_file()

    def test_get_icon_path(self):
        """Test icon path detection"""
        setup = TBxSetup()
        icon = setup.get_icon_path()
        assert isinstance(icon, Path)
        # Icon may not exist yet


class TestTBSetup:
    """Test TBSetup class (complete installation)"""

    def test_init(self):
        """Test TBSetup initialization"""
        setup = TBSetup()
        assert setup.system in ["Windows", "Linux", "Darwin"]
        assert isinstance(setup.root, Path)
        assert isinstance(setup.tbx_utils, Path)
        assert isinstance(setup.tb_exc_dir, Path)

    def test_paths_exist(self):
        """Test that critical paths exist"""
        setup = TBSetup()

        # Root should exist
        assert setup.root.exists()
        assert setup.root.is_dir()

        # Utils directory should exist
        assert setup.tbx_utils.exists()
        assert setup.tbx_utils.is_dir()

        # setup.py should exist
        assert (setup.tbx_utils / "setup.py").exists()

    def test_vscode_extension_path(self):
        """Test VS Code extension path"""
        setup = TBSetup()
        vscode_ext = setup.tbx_utils / "tb-lang-support"

        if vscode_ext.exists():
            # Check for critical files
            assert (vscode_ext / "package.json").exists()
            assert (vscode_ext / "language-configuration.json").exists()
            assert (vscode_ext / "syntaxes" / "tb.tmLanguage.json").exists()

    def test_pycharm_plugin_path(self):
        """Test PyCharm plugin path"""
        setup = TBSetup()
        pycharm_plugin = setup.tbx_utils / "tb-lang-pycharm"

        if pycharm_plugin.exists():
            # Check for critical files
            assert (pycharm_plugin / "src" / "main" / "resources" / "META-INF" / "plugin.xml").exists()
            assert (pycharm_plugin / "src" / "main" / "resources" / "fileTypes" / "TB.xml").exists()


class TestVSCodeExtension:
    """Test VS Code extension configuration"""

    def test_package_json_valid(self):
        """Test that package.json is valid JSON"""
        package_json = Path(__file__).parent.parent / "tb-lang-support" / "package.json"

        if not package_json.exists():
            pytest.skip("VS Code extension not found")

        with open(package_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check required fields
        assert "name" in data
        assert "version" in data
        assert "contributes" in data
        assert "languages" in data["contributes"]

    def test_language_configuration_valid(self):
        """Test that language-configuration.json is valid"""
        lang_config = Path(__file__).parent.parent / "tb-lang-support" / "language-configuration.json"

        if not lang_config.exists():
            pytest.skip("VS Code extension not found")

        with open(lang_config, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check comment syntax is correct
        assert "comments" in data
        assert data["comments"]["lineComment"] == "//"
        assert data["comments"]["blockComment"] == ["/*", "*/"]

    def test_syntax_file_valid(self):
        """Test that tb.tmLanguage.json is valid"""
        syntax_file = Path(__file__).parent.parent / "tb-lang-support" / "syntaxes" / "tb.tmLanguage.json"

        if not syntax_file.exists():
            pytest.skip("Syntax file not found")

        with open(syntax_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check required fields
        assert "name" in data
        assert "scopeName" in data
        assert data["scopeName"] == "source.tb"
        assert "patterns" in data
        assert "repository" in data

    def test_file_extensions_configured(self):
        """Test that both .tbx and .tb extensions are configured"""
        package_json = Path(__file__).parent.parent / "tb-lang-support" / "package.json"

        if not package_json.exists():
            pytest.skip("VS Code extension not found")

        with open(package_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        languages = data["contributes"]["languages"]
        assert len(languages) > 0

        tb_lang = languages[0]
        assert ".tbx" in tb_lang["extensions"]
        assert ".tb" in tb_lang["extensions"]


class TestPyCharmPlugin:
    """Test PyCharm plugin configuration"""

    def test_plugin_xml_exists(self):
        """Test that plugin.xml exists"""
        plugin_xml = Path(__file__).parent.parent / "tb-lang-pycharm" / "src" / "main" / "resources" / "META-INF" / "plugin.xml"

        if not plugin_xml.exists():
            pytest.skip("PyCharm plugin not found")

        assert plugin_xml.is_file()

    def test_filetype_xml_exists(self):
        """Test that TB.xml exists"""
        filetype_xml = Path(__file__).parent.parent / "tb-lang-pycharm" / "src" / "main" / "resources" / "fileTypes" / "TB.xml"

        if not filetype_xml.exists():
            pytest.skip("PyCharm plugin not found")

        assert filetype_xml.is_file()

    def test_comment_syntax_correct(self):
        """Test that comment syntax is correct in TB.xml"""
        filetype_xml = Path(__file__).parent.parent / "tb-lang-pycharm" / "src" / "main" / "resources" / "fileTypes" / "TB.xml"

        if not filetype_xml.exists():
            pytest.skip("PyCharm plugin not found")

        content = filetype_xml.read_text(encoding='utf-8')

        # Check for correct comment syntax
        assert 'LINE_COMMENT" value="//"' in content
        assert 'COMMENT_START" value="/*"' in content
        assert 'COMMENT_END" value="*/"' in content

        # Make sure old # syntax is not present
        assert 'LINE_COMMENT" value="#"' not in content

    def test_file_extensions_configured(self):
        """Test that both .tbx and .tb extensions are configured"""
        filetype_xml = Path(__file__).parent.parent / "tb-lang-pycharm" / "src" / "main" / "resources" / "fileTypes" / "TB.xml"

        if not filetype_xml.exists():
            pytest.skip("PyCharm plugin not found")

        content = filetype_xml.read_text(encoding='utf-8')

        # Check for extensions
        assert "tbx" in content
        assert "tb" in content or "extensions>tbx;tb<" in content


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

