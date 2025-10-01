# file: tb-lang-pycharm/install_filetype.py
"""
Simplified PyCharm file type installation
Adds .tbx support without full plugin
"""

import platform
import xml.etree.ElementTree as ET
from pathlib import Path


def install_file_type():
    """Install TB file type to PyCharm"""
    system = platform.system()
    home = Path.home()

    # Find PyCharm config directories
    config_dirs = []

    if system == "Windows":
        base = home / "AppData" / "Roaming" / "JetBrains"
        if base.exists():
            config_dirs.extend(base.glob("PyCharm*"))

    elif system == "Linux":
        base = home / ".config" / "JetBrains"
        if base.exists():
            config_dirs.extend(base.glob("PyCharm*"))

    elif system == "Darwin":
        base = home / "Library" / "Application Support" / "JetBrains"
        if base.exists():
            config_dirs.extend(base.glob("PyCharm*"))

    if not config_dirs:
        print("❌ PyCharm installation not found")
        return False

    # File type XML
    filetype_xml = """<?xml version="1.0" encoding="UTF-8"?>
<application>
  <component name="FileTypeManager" version="18">
    <extensionMap>
      <mapping pattern="*.tbx" type="TB Language" />
    </extensionMap>
  </component>
</application>
"""

    installed = False
    for config_dir in config_dirs:
        options_dir = config_dir / "options"
        options_dir.mkdir(exist_ok=True)

        filetype_file = options_dir / "filetypes.xml"

        # Write or update file types
        filetype_file.write_text(filetype_xml)

        print(f"✓ Installed file type to: {config_dir.name}")
        installed = True

    if installed:
        print()
        print("✓ File type installed!")
        print("  Restart PyCharm to see .tbx files with custom icon")

    return installed


if __name__ == "__main__":
    import sys

    success = install_file_type()
    sys.exit(0 if success else 1)
