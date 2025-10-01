# file: tb-lang-pycharm/build_plugin.py
# !/usr/bin/env python3
"""
Simple PyCharm plugin builder
Creates a JAR file from plugin sources
"""

import zipfile
from pathlib import Path
import sys


def build_plugin():
    """Build PyCharm plugin JAR"""
    plugin_dir = Path(__file__).parent
    resources_dir = plugin_dir / "src" / "main" / "resources"
    output_jar = plugin_dir / "tb-language.jar"

    if not resources_dir.exists():
        print(f"❌ Resources directory not found: {resources_dir}")
        return False

    print(f"Building PyCharm plugin...")
    print(f"  Resources: {resources_dir}")
    print(f"  Output: {output_jar}")

    with zipfile.ZipFile(output_jar, 'w', zipfile.ZIP_DEFLATED) as jar:
        # Add all files from resources
        for file_path in resources_dir.rglob('*'):
            if file_path.is_file():
                # Get relative path from resources dir
                arcname = str(file_path.relative_to(resources_dir))
                jar.write(file_path, arcname)
                print(f"  + {arcname}")

    print(f"✓ Plugin built: {output_jar}")
    print(f"  Size: {output_jar.stat().st_size} bytes")
    return True


if __name__ == "__main__":
    success = build_plugin()
    sys.exit(0 if success else 1)
