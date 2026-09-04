"""
wizard_api.py - Schlanke JSON-API ueber dem Config-Wizard (kein Umbau von config_wizard/first_run).

Konsumenten: local_ui (/welcome) + Tauri Installer-Wizard.
Wiederverwendet aus config_wizard: validate_llm_key, LLM_PROVIDERS, get_profile_defaults, save_env_file.
Wiederverwendet aus first_run: PROFILES, PROFILE_DEFAULTS.
Wiederverwendet aus tauri_cli: download_file, fetch_latest_release_info, download_app (GUI-Install).

Route-Order: MUSS in local_ui.py VOR dem spa_fallback-Catch-all registriert werden
(register_wizard_routes(app) zwischen /install und spa_fallback aufrufen).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from toolboxv2 import tb_root_dir
from toolboxv2.utils.extras.blobs import BlobFile

# --- Profil-Bridge: first_run-Profile -> init_onboarding-Presets (decisions.md #2)
PROFILE_BRIDGE: Dict[str, str] = {
    "consumer": "desktop",
    "homelab": "desktop",
    "server": "server",
    "business": "server",
    "developer": "desktop",
}

_SESSION_FILE = "system/wizard_session.json"
_STATE_FILE = "system/installation_state.json"


def _mirror_state_file(payload: Dict[str, Any]) -> None:
    """FIX (bug-tauri-firstrun): plaintext Spiegel von system/installation_state.json.

    BlobFile ist modus-abhaengig (MOBILE/SQLite unter TAURI_ENV, sonst Server/dev) —
    der State kann je nach Prozess-Kontext in verschiedenen Backends liegen. Der
    Mirror liegt deterministisch unter tb_root_dir/system/ und bricht die
    First-Run-Schleife, wenn Backend-Zugriffe fehlschlagen oder Prozesse wechseln.
    """
    try:
        plain = Path(str(tb_root_dir)) / "system"
        plain.mkdir(parents=True, exist_ok=True)
        (plain / "installation_state.json").write_text(
            json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


def _load_state_with_mirror() -> Optional[Dict[str, Any]]:
    """State lesen: BlobFile zuerst, plaintext-Mirror als Fallback."""
    try:
        with BlobFile(_STATE_FILE, "r") as f:
            data = f.read_json()
            if data:
                return data
    except Exception:
        pass
    try:
        mirror = Path(str(tb_root_dir)) / "system" / "installation_state.json"
        if mirror.exists():
            data = json.loads(mirror.read_text(encoding="utf-8"))
            if data:
                return data
    except Exception:
        pass
    return None


def _looks_installed() -> bool:
    """Heuristik: existieren Daten einer echten TB-Installation?

    FIX (bug-tauri-firstrun): fehlender State-File bedeutet nicht zwangslaeufig
    First-Run — ein vorhandenes main-<HOST>-Datenverzeichnis oder user-data-enc
    zeigt eine Installation; der Wizard bleibt via /welcome erreichbar.
    """
    try:
        data_dirs = [
            Path(str(tb_root_dir)) / ".data",
            Path.home() / "AppData" / "Roaming" / "toolboxv2",
            Path.home() / ".local" / "share" / "toolboxv2",
            Path.home() / "Library" / "Application Support" / "toolboxv2",
        ]
        for d in data_dirs:
            try:
                if d.is_dir():
                    for child in d.iterdir():
                        if child.name.startswith("main-") or child.name in (
                                "user-data-enc", "tb-mods"):
                            return True
            except Exception:
                continue
    except Exception:
        pass
    return False

_gui_status: Dict[str, Any] = {"running": False, "error": None, "done": False}
_dist_status: Dict[str, Any] = {"running": False, "error": None, "done": False, "asset": None}


# =============================================================================
# State-Helpers (BlobFile-persistiert, single-user local-only)
# ponytail: multi-session-IDs sobald Remote-UI existiert
# =============================================================================

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

    loader = ManifestLoader(tb_root_dir)
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

def wizard_steps(draft: Dict[str, Any], profile: Optional[str]) -> List[Dict[str, Any]]:
    is_server = profile in ("server", "business")
    db_mode = str(_get_nested(draft, "database.mode", "LC") or "LC").upper()

    db_fields: List[Dict[str, Any]] = [{
        "name": "database.mode", "label": "Database mode", "type": "choice",
        "choices": ["LC", "LR", "RR", "CB"],
        "help": "LC=Local JSON, LR=Local Redis, RR=Remote Redis, CB=MinIO/S3",
        "value": db_mode,
    }]
    if db_mode == "LC":
        db_fields.append({"name": "database.local.path", "label": "Local DB path",
                          "type": "str", "value": _get_nested(draft, "database.local.path", ".data/MiniDictDB.json")})
    elif db_mode in ("LR", "RR"):
        db_fields += [
            {"name": "database.redis.url", "label": "Redis URL", "type": "str",
             "value": _get_nested(draft, "database.redis.url", "redis://localhost:6379")},
            {"name": "database.redis.db_index", "label": "Redis DB index", "type": "int",
             "value": _get_nested(draft, "database.redis.db_index", 0)},
        ]
        if db_mode == "RR":
            db_fields += [
                {"name": "database.redis.username", "label": "Redis username", "type": "str",
                 "value": _get_nested(draft, "database.redis.username", "")},
                {"name": "database.redis.password", "label": "Redis password", "type": "password",
                 "value": _get_nested(draft, "database.redis.password", "")},
            ]
    else:  # CB
        db_fields += [
            {"name": "database.minio.endpoint", "label": "MinIO endpoint", "type": "str",
             "value": _get_nested(draft, "database.minio.endpoint", "localhost:9000")},
            {"name": "database.minio.access_key", "label": "MinIO access key", "type": "str",
             "value": _get_nested(draft, "database.minio.access_key", "minioadmin")},
            {"name": "database.minio.secret_key", "label": "MinIO secret key", "type": "password",
             "value": _get_nested(draft, "database.minio.secret_key", "minioadmin")},
            {"name": "database.minio.bucket", "label": "MinIO bucket", "type": "str",
             "value": _get_nested(draft, "database.minio.bucket", "toolbox-data")},
            {"name": "database.minio.use_ssl", "label": "Use SSL/TLS", "type": "bool",
             "value": _get_nested(draft, "database.minio.use_ssl", False)},
        ]

    http0 = (_get_nested(draft, "workers.http") or [{}])[0] if _get_nested(draft, "workers.http") else {}

    steps: List[Dict[str, Any]] = [
        {
            "id": "app", "title": "Application",
            "fields": [
                {"name": "app.name", "label": "Application name", "type": "str",
                 "value": _get_nested(draft, "app.name", "ToolBoxV2")},
                {"name": "app.debug", "label": "Debug mode", "type": "bool",
                 "value": _get_nested(draft, "app.debug", False)},
                {"name": "app.log_level", "label": "Log level", "type": "choice",
                 "choices": ["DEBUG", "INFO", "WARNING", "ERROR"],
                 "value": _get_nested(draft, "app.log_level", "INFO")},
            ],
        },
        {"id": "database", "title": "Database", "fields": db_fields},
        {"id": "llm", "title": "LLM Providers", "type": "llm", "fields": []},
        {"id": "autostart", "title": "Autostart & Services",
         "fields": [
             {"name": "autostart.enabled", "label": "Enable autostart", "type": "bool",
              "value": _get_nested(draft, "autostart.enabled", True)},
             {"name": "autostart.services", "label": "Services", "type": "multi",
              "choices": ["daemon", "workers", "db"],
              "value": _get_nested(draft, "autostart.services", ["daemon"])},
         ]},
    ]
    if is_server:
        steps.insert(3, {
            "id": "workers", "title": "Worker Processes",
            "fields": [
                {"name": "workers.http_enabled", "label": "Enable HTTP worker", "type": "bool",
                 "value": http0.get("enabled", True)},
                {"name": "workers.http_instances", "label": "HTTP instances", "type": "int",
                 "value": http0.get("instances", 2)},
                {"name": "workers.http_port", "label": "HTTP port", "type": "int",
                 "value": http0.get("port", 5000)},
                {"name": "workers.ws.enabled", "label": "Enable WebSocket worker", "type": "bool",
                 "value": _get_nested(draft, "workers.ws.enabled", True)},
                {"name": "workers.ws.instances", "label": "WS instances", "type": "int",
                 "value": _get_nested(draft, "workers.ws.instances", 1)},
                {"name": "workers.ws.port", "label": "WS port", "type": "int",
                 "value": _get_nested(draft, "workers.ws.port", 6587)},
            ],
        })
        steps.insert(4, {
            "id": "services", "title": "External Services",
            "fields": [
                {"name": "services.redis.enabled", "label": "Enable Redis", "type": "bool",
                 "value": _get_nested(draft, "services.redis.enabled", False)},
                {"name": "services.redis.host", "label": "Redis host", "type": "str",
                 "value": _get_nested(draft, "services.redis.host", "localhost")},
                {"name": "services.redis.port", "label": "Redis port", "type": "int",
                 "value": _get_nested(draft, "services.redis.port", 6379)},
                {"name": "services.minio.enabled", "label": "Enable MinIO/S3", "type": "bool",
                 "value": _get_nested(draft, "services.minio.enabled", False)},
                {"name": "services.minio.endpoint", "label": "MinIO endpoint", "type": "str",
                 "value": _get_nested(draft, "services.minio.endpoint", "localhost:9000")},
            ],
        })
    steps.append({"id": "features", "title": "Features", "type": "features", "fields": []})
    return steps


def _check_field(field: Dict[str, Any], v: Any) -> Optional[str]:
    """Einzelnes Feld gegen seinen Deklarationstyp pruefen. None = ok."""
    t, name = field["type"], field["name"]
    if t == "bool":
        return None if isinstance(v, bool) else f"{name}: expected bool"
    if t == "int":
        if isinstance(v, bool):
            return f"{name}: expected int"
        try:
            int(v)
        except (TypeError, ValueError):
            return f"{name}: expected int"
        return None
    if t == "choice":
        return None if v in field.get("choices", []) else f"{name}: '{v}' not in {field['choices']}"
    if t == "multi":
        bad = not isinstance(v, list) or any(x not in field.get("choices", []) for x in v)
        return f"{name}: invalid selection {v}" if bad else None
    return None


def validate_values(fields: List[Dict[str, Any]], values: Dict[str, Any]) -> Optional[str]:
    """Type-/Choice-Check einer Antwort. Rueckgabe: Fehlermeldung oder None."""
    known = {f["name"] for f in fields}
    unknown = [k for k in values if k not in known]
    if unknown:
        return f"Unknown fields: {unknown}"
    for f in fields:
        if f["name"] in values:
            err = _check_field(f, values[f["name"]])
            if err:
                return err
    return None


def apply_values(draft: Dict[str, Any], fields: List[Dict[str, Any]], values: Dict[str, Any]) -> Dict[str, Any]:
    for f in fields:
        name = f["name"]
        if name not in values:
            continue
        v = values[name]
        if f["type"] == "int":
            v = int(v)
        if name == "workers.http_enabled":
            http = draft.get("workers", {}).get("http") or [{}]
            http[0]["enabled"] = v
            _set_nested(draft, "workers.http", http)
        elif name in ("workers.http_instances", "workers.http_port"):
            http = draft.get("workers", {}).get("http") or [{}]
            http[0][name.split(".")[-1].replace("http_", "")] = int(v)
            _set_nested(draft, "workers.http", http)
        else:
            _set_nested(draft, name, v)
    return draft


# =============================================================================
# Registration (in local_ui.py VOR spa_fallback aufrufen)
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
        inst = _load_state_with_mirror() or {}
        if not inst:
            installed = _looks_installed()
            inst = {"installed": installed, "first_run": not installed}
        return {"first_run": bool(inst.get("first_run", not inst.get("installed", False))),
                "installed": inst.get("installed", False), "profile": profile,
                "steps": wizard_steps(draft, profile)}

    @app.get("/api/wizard/profiles", auth=False)
    def wizard_profiles(request: ParsedRequest):
        from toolboxv2.utils.clis.first_run import PROFILES
        sess = _load_session()
        return {"profiles": [{"id": k, "label": v[0], "desc": v[1]} for k, v in PROFILES.items()],
                "current": sess.get("profile")}

    @app.post("/api/wizard/profile", auth=False)
    def wizard_set_profile(request: ParsedRequest):
        from toolboxv2.utils.clis.first_run import PROFILES, PROFILE_DEFAULTS
        data = _json(request)
        profile = data.get("profile")
        if profile not in PROFILES:
            return 400, {"error": f"unknown profile '{profile}'"}
        sess = _load_session()
        draft = sess.get("draft") or _manifest_data()
        sess["profile"] = profile
        _set_nested(draft, "app.profile", profile)
        # init_onboarding-Preset als Bruecke persistieren (decisions.md #2)
        _set_nested(draft, "app.init_preset", PROFILE_BRIDGE.get(profile, "desktop"))
        for k, v in PROFILE_DEFAULTS.get(profile, {}).items():
            _set_nested(draft, k, v)
        sess["draft"] = draft
        _save_session(sess)
        return {"ok": True, "profile": profile, "init_preset": PROFILE_BRIDGE.get(profile),
                "steps": wizard_steps(draft, profile)}

    @app.post("/api/wizard/step", auth=False)
    def wizard_step(request: ParsedRequest):
        data = _json(request)
        sess = _load_session()
        draft = sess.get("draft") or _manifest_data()
        profile = _current_profile(draft, sess)
        step_id = data.get("step")
        steps = wizard_steps(draft, profile)
        step = next((s for s in steps if s["id"] == step_id), None)
        if step is None:
            return 400, {"error": f"unknown step '{step_id}'",
                         "available": [s["id"] for s in steps]}
        values = data.get("values")
        if not isinstance(values, dict):
            return 400, {"error": "values must be an object"}
        if step.get("type") in ("llm", "features"):
            return 400, {"error": f"step '{step_id}' uses its own endpoint"}
        err = validate_values(step["fields"], values)
        if err:
            return 400, {"error": err}
        draft = apply_values(draft, step["fields"], values)
        sess["draft"] = draft
        _save_session(sess)
        return {"ok": True, "step": step_id, "steps": wizard_steps(draft, profile)}

    # --- LLM ---------------------------------------------------------------
    @app.get("/api/wizard/llm", auth=False)
    def wizard_llm(request: ParsedRequest):
        from toolboxv2.utils.clis.config_wizard import LLM_PROVIDERS
        sess = _load_session()
        env = sess.get("env") or {}
        out = []
        for pk, p in LLM_PROVIDERS.items():
            out.append({
                "id": pk, "name": p["name"], "default": p.get("default", False),
                "key_vars": p["key_vars"], "url_var": p.get("url_var"),
                "default_url": p.get("default_url", ""), "extra_vars": p.get("extra_vars", []),
                "configured": any(env.get(v) for v in p["key_vars"]),
            })
        return {"providers": out, "values": {k: v for k, v in env.items() if "PASSWORD" not in k}}

    @app.post("/api/wizard/llm", auth=False)
    def wizard_llm_save(request: ParsedRequest):
        from toolboxv2.utils.clis.config_wizard import LLM_PROVIDERS
        data = _json(request)
        provider = data.get("provider")
        values = data.get("values")
        if provider not in LLM_PROVIDERS or not isinstance(values, dict):
            return 400, {"error": "expected {provider, values}"}
        allowed = set(LLM_PROVIDERS[provider]["key_vars"] + LLM_PROVIDERS[provider].get("extra_vars", []))
        if LLM_PROVIDERS[provider].get("url_var"):
            allowed.add(LLM_PROVIDERS[provider]["url_var"])
        bad = [k for k in values if k not in allowed]
        if bad:
            return 400, {"error": f"unknown vars for {provider}: {bad}"}
        sess = _load_session()
        sess.setdefault("env", {}).update({k: str(v) for k, v in values.items()})
        _save_session(sess)
        return {"ok": True, "provider": provider, "configured": True}

    @app.post("/api/wizard/llm/validate", auth=False)
    def wizard_llm_validate(request: ParsedRequest):
        from toolboxv2.utils.clis.config_wizard import validate_llm_key
        data = _json(request)
        provider = data.get("provider", "")
        api_key = data.get("api_key", "")
        base_url = data.get("base_url", "")
        if not provider:
            return 400, {"error": "provider required"}
        ok, msg, models = validate_llm_key(provider, api_key, base_url)
        return {"valid": ok, "message": msg, "models": models}

    # --- Features ------------------------------------------------------------
    @app.get("/api/wizard/features", auth=False)
    def wizard_features(request: ParsedRequest):
        try:
            from toolboxv2.feature_loader import list_available_features, is_feature_installed
            feats = list_available_features()
            return {"features": [{"id": f, "installed": f == "core" or is_feature_installed(f)}
                                 for f in feats]}
        except Exception as e:
            return {"features": [], "error": str(e)}

    @app.post("/api/wizard/features", auth=False)
    def wizard_features_install(request: ParsedRequest):
        from toolboxv2.feature_loader import unpack_feature
        data = _json(request)
        feats = data.get("features")
        if not isinstance(feats, list):
            return 400, {"error": "features must be a list"}
        results = {}
        for f in feats:
            try:
                results[f] = bool(unpack_feature(f))
            except Exception as e:
                results[f] = f"error: {e}"
        return {"ok": True, "results": results}

    # --- GUI-Install (Tauri App) ---------------------------------------------
    @app.get("/api/wizard/gui", auth=False)
    def wizard_gui_status(request: ParsedRequest):
        from toolboxv2.utils.clis import tauri_cli
        path = tauri_cli.get_installed_app_path()
        st = dict(_gui_status)
        st["installed"] = path is not None
        st["app_path"] = str(path) if path else None
        st["version"] = tauri_cli.get_installed_version()
        return st

    @app.post("/api/wizard/gui", auth=False)
    def wizard_gui_install(request: ParsedRequest):
        if _gui_status.get("running"):
            return 409, {"error": "install already running"}
        data = _json(request)
        source = data.get("source", "auto")

        def _worker():
            _gui_status.update(running=True, error=None, done=False)
            try:
                from toolboxv2.utils.clis import tauri_cli
                ok = tauri_cli.download_app(source=source, show_progress=False) if "show_progress" in tauri_cli.download_app.__code__.co_varnames else tauri_cli.download_app(source=source)
                _gui_status.update(done=True, error=None if ok else "download_app returned False")
            except Exception as e:
                _gui_status.update(done=True, error=str(e))
            finally:
                _gui_status["running"] = False

        threading.Thread(target=_worker, daemon=True, name="tb-wizard-gui-install").start()
        return {"ok": True, "started": True}

    # --- dist/ Download (Server-Modus) ---------------------------------------
    @app.get("/api/wizard/dist", auth=False)
    def wizard_dist_assets(request: ParsedRequest):
        from toolboxv2.utils.workers.fast.endpoint import dist_dir
        from toolboxv2.utils.clis import tauri_cli
        d = dist_dir()
        release = tauri_cli.fetch_latest_release_info()
        if not release:
            return 502, {"error": "registry not reachable", "dist_dir": str(d) if d else None}
        assets = [a for a in release.get("assets", []) if "dist" in str(a.get("name", "")).lower()]
        return {"dist_dir": str(d) if d else None,
                "exists": bool(d and (d / "index.html").exists()),
                "assets": [{"name": a.get("name"), "size": a.get("size"),
                            "url": a.get("browser_download_url")} for a in assets],
                "all_asset_names": [a.get("name") for a in release.get("assets", [])]}

    @app.post("/api/wizard/dist", auth=False)
    def wizard_dist_download(request: ParsedRequest):
        import zipfile
        from toolboxv2.utils.workers.fast.endpoint import dist_dir
        from toolboxv2.utils.clis import tauri_cli
        if _dist_status.get("running"):
            return 409, {"error": "download already running"}
        data = _json(request)
        url = data.get("url")
        name = data.get("name") or (url or "").rsplit("/", 1)[-1]
        if not url or not name:
            return 400, {"error": "url and/or name required"}
        d = dist_dir()
        if not d:
            return 400, {"error": "no dist_dir configured (manifest paths.dist_dir / TB_DIST_DIR)"}

        def _worker():
            _dist_status.update(running=True, error=None, done=False, asset=name)
            try:
                d.mkdir(parents=True, exist_ok=True)
                dest = d / name
                if not tauri_cli.download_file(url, dest, show_progress=False):
                    raise RuntimeError("download_file failed")
                if name.lower().endswith(".zip"):
                    with zipfile.ZipFile(dest) as z:
                        z.extractall(d)
                    dest.unlink()
                _dist_status.update(done=True, error=None)
            except Exception as e:
                _dist_status.update(done=True, error=str(e))
            finally:
                _dist_status["running"] = False

        threading.Thread(target=_worker, daemon=True, name="tb-wizard-dist").start()
        return {"ok": True, "started": True, "target": str(d / name)}

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
        loader = ManifestLoader(tb_root_dir)
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

        profile = _current_profile(draft, sess)
        state_payload = {"installed": True, "first_run": False, "profile": profile,
                         "completed_at": datetime.now(timezone.utc).isoformat(),
                         "init_preset": PROFILE_BRIDGE.get(profile or "", "")}
        with BlobFile(_STATE_FILE, "w") as f:
            f.write_json(state_payload)
        # FIX (bug-tauri-firstrun): plaintext-Mirror mitschreiben (Backend-unabhaengig).
        _mirror_state_file(state_payload)
        _save_session({"draft": draft, "env": {}, "profile": profile})
        return {"ok": True, "manifest": str(loader.manifest_path),
                "env_saved": bool(env_out), "generated": generated,
                "profile": profile, "first_run": False}
