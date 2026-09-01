"""HabitTracker — Habit-Tracker PWA als ToolBoxV2 Mod.

App unter /api/HabitTracker/app, User-Daten (tasks/streaks) in CloudM MOD_DATA.
"""
import os

from toolboxv2 import Result, get_app

export = get_app(from_="HabitTracker.Export").tb
Name = "HabitTracker"
version = "0.1.0"

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Whitelist — kein freier Pfad-Parameter, keine Traversal-Flaeche
ASSETS = {
    "app":       ("index.html", "text/html; charset=utf-8"),
    "sw.js":     ("sw.js", "application/javascript"),
    "manifest.json": ("manifest.json", "application/manifest+json"),
    "icon-192.png": ("icons/icon-192.png", "image/png"),
    "icon-512.png": ("icons/icon-512.png", "image/png"),
}

VALID_KEYS = ("tasks", "streaks")


def _load_asset(asset: str) -> Result:
    if asset not in ASSETS:
        return Result.default_user_error(info="Unknown asset", exec_code=404)
    filename, ctype = ASSETS[asset]
    path = os.path.join(STATIC_DIR, filename)
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return Result.default_internal_error(info=f"asset missing: {filename}")
    # text as text (no download semantics), binary for icons
    if ctype.startswith(("text/", "application/javascript", "application/manifest")):
        return Result.text(data.decode("utf-8"), content_type=ctype)
    return Result.binary(data=data, content_type=ctype)

@export(mod_name=Name, name="init", initial=True)
def init(app=None):
    if app is None:
        app = get_app(Name+'.init')
    app.run_any(("CloudM", "add_ui"),
                name=Name,
                title="HabitTracker",
                path=f"/api/{Name}/app",
                description="HabitTracker"
                )

@export(name="app", mod_name=Name, version=version, api=True)
async def app(app):
    """App-UI ausliefern (einzelnes HTML mit inline JS/CSS)."""
    return _load_asset("app")


@export(name="asset", mod_name=Name, version=version, api=True)
async def asset(app, file: str = "sw.js"):
    """Statische Zusatz-Dateien (sw.js, manifest, icons) fuer Standalone-Betrieb."""
    return _load_asset(file)


@export(name="data_get", mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def data_get(app, request, key: str = "tasks"):
    """User-Daten aus CloudM MOD_DATA laden (tasks | streaks)."""
    if key not in VALID_KEYS:
        return Result.default_user_error(info="key must be tasks|streaks", exec_code=400)
    from toolboxv2.mods.CloudM.UserDataAPI import get_mod_data

    result = await get_mod_data(app, request, source_mod=Name, key=key)
    if result.is_error():
        return result  # 401 unauth -> Frontend faellt auf localStorage zurueck
    return Result.ok(data=result.get())


@export(name="data_set", mod_name=Name, version=version, api=True, request_as_kwarg=True)
async def data_set(app, request, data: dict, key: str = "tasks"):
    """User-Daten in CloudM MOD_DATA speichern (merge=False: ganze Datei ersetzen)."""
    if key not in VALID_KEYS:
        return Result.default_user_error(info="key must be tasks|streaks", exec_code=400)
    if not isinstance(data, dict):
        return Result.default_user_error(info="data must be an object", exec_code=400)
    from toolboxv2.mods.CloudM.UserDataAPI import set_mod_data

    result = await set_mod_data(app, request, source_mod=Name, data=data, key=key, merge=False)
    if result.is_error():
        return result
    return Result.ok(data_info="saved")
