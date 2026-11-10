"""
wizard_api.py - Minimal JSON API for the one onboarding UI (installer.html,
served at 127.0.0.1:8467 "/" on first run and at "/setup").

Endpoints: GET state · POST profile · POST save. First-run state comes from
first_run.onboarding_state() (manifest app.profile) — same source as the CLI.

Route order: register before local_ui's spa_fallback catch-all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from toolboxv2 import tb_root_dir
from toolboxv2.utils.extras.blobs import BlobFile

_SESSION_FILE = "system/wizard_session.json"


def _load_session() -> Dict[str, Any]:
    try:
        with BlobFile(_SESSION_FILE, "r") as f:
            data = f.read_json()
            if isinstance(data, dict):
                return data
    except Exception as e:
        print(f"[wizard_api] session load failed: {e}")
    return {"draft": {}, "env": {}, "profile": None}


def _save_session(sess: Dict[str, Any]) -> None:
    try:
        with BlobFile(_SESSION_FILE, "w") as f:
            f.write_json(sess)
    except Exception as e:
        print(f"[wizard_api] session save failed: {e}")


def _manifest_data() -> Dict[str, Any]:
    """Manifest als editierbares dict (model_dump), default falls nicht vorhanden."""
    from toolboxv2.utils.manifest.loader import ManifestLoader
    from toolboxv2.utils.manifest.schema import TBManifest

    from toolboxv2._state_dir import manifest_base_dir
    loader = ManifestLoader(manifest_base_dir())
    if loader.exists():
        return loader.load().model_dump()
    return TBManifest().model_dump()


def _current_profile(draft: Dict[str, Any], sess: Dict[str, Any]) -> Optional[str]:
    if sess.get("profile"):
        return sess["profile"]
    app = draft.get("app") or {}
    p = app.get("profile")
    if p is None:
        return None
    if isinstance(p, dict) and "value" in p:
        p = p["value"]
    if hasattr(p, "value"):  # Enum aus model_dump() -> "developer", nicht "ProfileType.DEVELOPER"
        p = p.value
    return str(p) if p else None


def _set_nested(data: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = data
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def _get_nested(data: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = data
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


# =============================================================================
# Step-Definitionen (Felder abgeleitet aus config_wizard wizard_*-Funktionen)
# =============================================================================

def register_wizard_routes(app) -> None:
    from toolboxv2.utils.workers.server_worker import ParsedRequest

    def _json(request: ParsedRequest) -> Dict[str, Any]:
        return request.json_data if isinstance(request.json_data, dict) else {}

    @app.get("/api/wizard/state", auth=False)
    def wizard_state(request: ParsedRequest):
        sess = _load_session()
        draft = sess.get("draft") or _manifest_data()
        profile = _current_profile(draft, sess)
        from toolboxv2.utils.clis.first_run import onboarding_state
        inst = onboarding_state()  # same source as CLI: manifest app.profile
        return {"first_run": inst["first_run"], "installed": inst["installed"],
                "profile": profile or inst["profile"]}

    @app.post("/api/wizard/profile", auth=False)
    def wizard_set_profile(request: ParsedRequest):
        from toolboxv2.utils.clis.first_run import PROFILE_DEFAULTS, normalize_profile
        data = _json(request)
        try:  # accepts legacy names (consumer/homelab/business) like the manifest
            if not data.get("profile"):
                raise ValueError("missing")
            profile = normalize_profile(data.get("profile"))
        except ValueError:
            return 400, {"error": f"unknown profile '{data.get('profile')}'"}
        sess = _load_session()
        draft = sess.get("draft") or _manifest_data()
        sess["profile"] = profile
        _set_nested(draft, "app.profile", profile)
        for k, v in PROFILE_DEFAULTS.get(profile, {}).items():
            _set_nested(draft, k, v)
        sess["draft"] = draft
        _save_session(sess)
        return {"ok": True, "profile": profile}

    # --- Save ------------------------------------------------------------------
    @app.post("/api/wizard/save", auth=False)
    def wizard_save(request: ParsedRequest):
        from toolboxv2.utils.manifest.loader import ManifestLoader
        from toolboxv2.utils.manifest.schema import TBManifest
        from toolboxv2.utils.clis.config_wizard import save_env_file
        sess = _load_session()
        draft = sess.get("draft") or _manifest_data()
        try:
            manifest = TBManifest(**draft)
        except Exception as e:
            return 400, {"error": f"invalid manifest draft: {e}"}
        from toolboxv2._state_dir import manifest_base_dir
        loader = ManifestLoader(manifest_base_dir())
        loader.save(manifest)

        env_out = {}
        if sess.get("env"):
            from toolboxv2.utils.clis.config_wizard import load_existing_env
            env_out = load_existing_env(tb_root_dir / ".env")
            env_out.update(sess["env"])
            save_env_file(tb_root_dir / ".env", env_out)

        generated: List[str] = []
        data = _json(request)
        if data.get("apply", True):
            from toolboxv2.utils.manifest.converter import ConfigConverter
            generated = [str(p) for p in ConfigConverter(manifest, tb_root_dir).apply_all()]

        # first-run state = manifest app.profile (onboarding_state), nothing else
        profile = _current_profile(draft, sess)
        _save_session({"draft": draft, "env": {}, "profile": profile})
        return {"ok": True, "manifest": str(loader.manifest_path),
                "env_saved": bool(env_out), "generated": generated,
                "profile": profile, "first_run": False}
