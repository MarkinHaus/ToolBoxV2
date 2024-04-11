import json
import os
from typing import List, Dict, Optional

from toolboxv2 import get_app, App, Result, tbef, get_logger, Code
from .types import Widget, Sto
from toolboxv2.utils.system.types import ToolBoxInterfaces
from fastapi import Request

from ..CloudM import User
from ...utils.extras.blobs import BlobFile

Name = 'WidgetsProvider'
export = get_app("WidgetsProvider.Export").tb
default_export = export(mod_name=Name, test=False, api=True)
version = '0.0.1'
spec = ''

all_widgets = []


@export(mod_name=Name, version=version)
def get_all_widget_mods(app: App):
    global all_widgets
    if len(all_widgets) != 0:
        return all_widgets
    all_widget = [widget_mod for widget_mod in app.functions.keys() if 'widget' in widget_mod.lower()]
    valid_widgets = []
    for widget_mod in all_widget:
        _, error = app.get_function((widget_mod, "get_widget"))
        if 0 != error:
            continue
        valid_widgets.append(widget_mod)
    all_widgets = valid_widgets
    return all_widgets


@export(mod_name=Name, version=version)
def get_user_from_request(app, request):
    if request is None:
        return User()
    name = request.session.get('live_data', {}).get('user_name', "Cud be ur name")
    if name != "Cud be ur name":
        user = app.run_any(tbef.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=app.config_fh.decode_code(name))
    else:
        user = User()
    return user


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True)
async def save_user_sto(app, request, name: str = "Main-User-DBord"):
    if app is None:
        app = get_app(f"{Name}.open")
    user = get_user_from_request(app, request)
    b = await request.body()

    with BlobFile(f"users/{Code.one_way_hash(name, 'userWidgetSto', user.uid)}/{name}/bords", 'w') as f:
        f.clear()
        f.write(b)


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True)
async def get_user_sto(app, request, name: str = "Main-User-DBord"):
    if app is None:
        app = get_app(f"{Name}.open")
    if request is None:
        return
    user = get_user_from_request(app, request)
    with BlobFile(f"users/{Code.one_way_hash(name, 'userWidgetSto', user.uid)}/{name}/bords", 'r') as f:
        data = f.read()
    return data


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="open_widget")
def open_widget(app, request, name: str, **kwargs):
    if app is None:
        app = get_app(f"{Name}.open")
    if len(all_widgets) == 0:
        get_all_widget_mods(app)
    if name not in all_widgets:
        return "invalid widget name " + str(all_widgets)

    return app.run_any((name, "get_widget"), request=request, **kwargs)
