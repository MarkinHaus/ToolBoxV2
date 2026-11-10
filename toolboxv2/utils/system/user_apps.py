"""User-facing app registry — single source for CLI (local_cli) and web (mainContent.html).

Apps are declared per feature in ``features/<name>/feature.yaml``::

    user_facing:
      - name: isaa              # tb module name
        function: ui            # CLI: tb_app.a_run_any((name, function))
        label: ISAA
        icon: smart_toy         # material symbol (web) / glyph (CLI)
        description: Agent chat & workspace
        mount:                  # web: mounted lazily into the local owner (8467)
          module: toolboxv2.mods.isaa.ui.app
          attr: app
          prefix: isaa

Only features with an ``.installed`` marker are listed.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _features_dir() -> Path:
    from toolboxv2 import tb_root_dir
    return tb_root_dir / "features"


def _entries(feature_yaml: Path) -> List[Dict[str, Any]]:
    try:
        data = yaml.safe_load(feature_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    out = []
    for entry in data.get("user_facing", []) or []:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        mount = entry.get("mount")
        out.append({
            "name": entry["name"],
            "function": entry.get("function") or entry.get("entry_point") or "start",
            "label": entry.get("label", entry["name"]),
            "icon": entry.get("icon", "▸"),
            "description": entry.get("description", ""),
            "feature": data.get("name", "unknown"),
            "mount": mount if isinstance(mount, dict) and mount.get("module") else None,
        })
    return out


def all_user_apps(features_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """All user-facing apps of installed features."""
    root = features_dir or _features_dir()
    if not root.exists():
        return []
    out = []
    for fyaml in sorted(root.glob("*/feature.yaml")):
        if (fyaml.parent / ".installed").exists():
            out.extend(_entries(fyaml))
    return out


def get_user_app(name: str, features_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    return next((a for a in all_user_apps(features_dir) if a["name"] == name), None)
