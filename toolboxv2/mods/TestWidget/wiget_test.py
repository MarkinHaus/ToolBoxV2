import os
import uuid
from typing import List, Dict, Optional

from toolboxv2 import get_app, App, Result, tbef, get_logger, Code
from toolboxv2.utils.system.types import ToolBoxInterfaces
from fastapi import Request

from ..CloudM import User

Name = 'TestWidget'
export = get_app("TestWidget.Export").tb
default_export = export(mod_name=Name)
version = '0.0.1'
spec = ''

counter = {}


def load_widget(app, display_name="Cud be ur name", cv=0, WidgetID=str(uuid.uuid4())[:4]):
    # vars : $providerurl $WidgetID $root $$username
    app.run_any(tbef.MINIMALHTML.ADD_GROUP, command=Name)
    # Usage
    # Sample data
    # /web/1/init0/titel.html -> <h1>test $test-name</h1>
    widget = {'name': "MainWidget",
              'group': [
                  {'name': 'main',
                   'file_path': f'./mods/{Name}/assets/template.html',
                   'kwargs': {
                       'username': display_name,
                       'root': f"/api/{Name}",
                       'WidgetID': WidgetID,
                       'value': cv,
                       'providerurl': f'/api/{Name}/setMVu'
                   }
                   },
              ]}
    app.run_any(tbef.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=widget)
    html_widget = app.run_any(tbef.MINIMALHTML.GENERATE_HTML, group_name=Name, collection_name="MainWidget")
    return html_widget[0]['html_element']


def get_s_id(request):
    if request is None:
        return Result.default_internal_error("No request specified")
    sID = request.session.get('ID', '')
    return Result.ok(sID)


def get_counter_value(id_):
    global counter
    if id_ not in counter:
        counter[id_] = 0
    return counter[id_]


def set_counter_value(id_, value):
    global counter
    counter[id_] = value


@export(mod_name=Name, version=version, level=1, row=True, name="Version", state=False)
def Version():
    return version


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True, name="add", state=False)
def add(request: Request or None = None):
    sid = get_s_id(request)
    if sid.is_error():
        return f"<h1>Error {sid.print(show=False)}</h1>"
    sid = sid.get()
    set_counter_value(sid, get_counter_value(sid) + 1)
    return f"<h2 > counter : {get_counter_value(sid)} </h2>"


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True, name="addM", state=False)
def addM(value: int = 1, request: Request or None = None):
    sid = get_s_id(request)
    if sid.is_error():
        return f"<h1>Error {sid.print(show=False)}</h1>"
    sid = sid.get()
    set_counter_value(sid, get_counter_value(sid) + int(value))
    return f"<h2 > counter : {get_counter_value(sid)} </h2>"


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True, name="reset", state=False)
def reset(request: Request or None = None):
    sid = get_s_id(request)
    if sid.is_error():
        return f"<h1>Error {sid.print(show=False)}</h1>"
    sid = sid.get()
    set_counter_value(sid, 0)
    return f"<h2 > counter : {get_counter_value(sid)} </h2>"


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="sync", state=False)
def sync(request: Request or None = None, counter=0, id="MainWidget"):
    pass
    # cumming soon withe Tauri and events
    # sid = get_s_id(request)
    # if sid.is_error():
    #    return f"<h1>Error {sid.print(show=False)}</h1>"
    # sid = sid.get()
    # set_counter_value(sid, counter)
    # return f"<h2 > counter : {get_counter_value(sid)} </h2>"


# get_wgiet
@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="get_widget")
def get_widget(app: App = None, request: Request or None = None):
    if app is None:
        app = get_app(from_=f"{Name}.get_widget")

    if request is None:
        return Result.default_internal_error("No request specified")

    username_c: str = request.session.get('live_data', {}).get('user_name', "Cud be ur name")
    if username_c != "Cud be ur name":
        username = app.config_fh.decode_code(username_c)
    else:
        username = username_c

    sid = get_s_id(request)
    if sid.is_error():
        cv = 0
    else:
        sid = sid.get()
        cv = get_counter_value(sid)

    widget = load_widget(app, username, cv)

    return widget


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="get_widget_r")
def get_widget_r(app: App = None, request: Request or None = None):
    if app is None:
        app = get_app(from_=f"{Name}.get_widget")

    if request is None:
        return Result.default_internal_error("No request specified")
    widget_renderer = app.run_any(tbef.WEBSOCKETMANAGER.CONSTRUCT_RENDER, content="<p> online</p>",
                                  element_id="widgetTest",
                                  externals=[f"/web/0/logic.js"],
                                  from_file=False, to_str=False)

    return widget_renderer
