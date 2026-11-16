"""Account <-> mirror-folder bridge (WP1 Auftrag 08, Teil 10).

Settings and mod_data of a UserData account, key by key, as JSON files in
the mirror folder - synced by data_api_sync.sync_folder like any other
folder. Admin-only settings never leave the device.
"""
import json
import logging
from pathlib import Path
from urllib.parse import quote, unquote

from toolboxv2.mods.CloudM.LiveSync.conflict import is_conflict_copy

logger = logging.getLogger(__name__)

ADMIN_ONLY = ("anonymous", "local_admin")


def _dump(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")


def export_account(user, mirror: Path) -> int:
    """Write changed keys, delete vanished keys. Returns files written."""
    mirror = Path(mirror)
    written = 0

    def _put(sub: Path, key: str, value) -> bool:
        try:
            data = _dump(value)
        except (TypeError, ValueError):
            # log the key name only - never the value
            logger.warning("account_bridge: skip non-JSON key %r", key)
            return False
        target = sub / (quote(key, safe="") + ".json")
        if not target.exists() or target.read_bytes() != data:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return True
        return False

    # settings
    seen_settings = set()
    for key, value in (user.settings or {}).items():
        if key in ADMIN_ONLY:
            continue
        seen_settings.add(quote(key, safe="") + ".json")
        if _put(mirror / "settings", key, value):
            written += 1

    # mod_data
    seen_mods = set()
    for mod, blob in (user.mod_data or {}).items():
        base = mirror / "mod_data" / quote(mod, safe="")
        if isinstance(blob, dict):
            for key, value in blob.items():
                name = quote(key, safe="") + ".json"
                seen_mods.add(quote(mod, safe="") + "/" + name)
                if _put(base, key, value):
                    written += 1
        else:
            name = "__value__.json"
            seen_mods.add(quote(mod, safe="") + "/" + name)
            if _put(base, "__value__", blob):
                written += 1

    # delete vanished keys
    sub = mirror / "settings"
    if sub.is_dir():
        for f in sub.iterdir():
            if f.name not in seen_settings:
                f.unlink()
    sub = mirror / "mod_data"
    if sub.is_dir():
        for f in sub.rglob("*.json"):
            rel = f.relative_to(sub).as_posix()
            if rel not in seen_mods:
                f.unlink()
    return written


def import_account(user, mirror: Path) -> bool:
    """Read the mirror back into the account. Returns True if changed."""
    mirror = Path(mirror)
    changed = False

    sub = mirror / "settings"
    if sub.is_dir():
        for f in sub.iterdir():
            if not f.name.endswith(".json") or is_conflict_copy(f.name):
                continue
            try:
                key = unquote(f.name[:-5])
                value = json.loads(f.read_bytes())
            except (ValueError, UnicodeDecodeError):
                logger.warning("account_bridge: unreadable mirror file %r", f.name)
                continue
            if key in ADMIN_ONLY:
                continue
            if not isinstance(user.settings, dict):
                user.settings = {}
            if user.settings.get(key) != value:
                user.settings[key] = value
                changed = True

    sub = mirror / "mod_data"
    if sub.is_dir():
        for f in sub.rglob("*.json"):
            if not f.name.endswith(".json") or is_conflict_copy(f.name):
                continue
            rel = f.relative_to(sub).as_posix()
            mod_q, _, name = rel.partition("/")
            try:
                mod = unquote(mod_q)
                value = json.loads(f.read_bytes())
            except (ValueError, UnicodeDecodeError):
                logger.warning("account_bridge: unreadable mirror file %r", f.name)
                continue
            if not isinstance(user.mod_data, dict):
                user.mod_data = {}
            if name == "__value__.json":
                if user.mod_data.get(mod) != value:
                    user.mod_data[mod] = value
                    changed = True
            else:
                try:
                    key = unquote(name[:-5])
                except Exception:
                    continue
                if not isinstance(user.mod_data.get(mod), dict):
                    user.mod_data[mod] = {}
                if user.mod_data[mod].get(key) != value:
                    user.mod_data[mod][key] = value
                    changed = True
    return changed
