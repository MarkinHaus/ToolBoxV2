import asyncio

from toolboxv2 import TBEF, App, Result, get_app, RequestData

from ..CloudM import User

Name = 'WidgetsProvider'
export = get_app("WidgetsProvider.Export").tb
default_export = export(mod_name=Name, test=False, api=True)
version = '0.0.1'
spec = ''

all_widgets = []


def get_s_id(request: RequestData):
    if request is None:
        return Result.default_internal_error("No request specified")
    sID = request.session.SiID
    return Result.ok(sID)


@export(mod_name=Name, version=version, row=True)
def get_all_widget_mods(app: App):
    global all_widgets
    if len(all_widgets) != 0:
        return all_widgets
    all_widget = [widget_mod for widget_mod in app.functions if 'widget' in widget_mod.lower()]
    valid_widgets = []
    for widget_mod in all_widget:
        _, error = app.get_function((widget_mod, "get_widget"))
        if error != 0:
            continue
        valid_widgets.append(widget_mod)
    all_widgets = valid_widgets
    return all_widgets


@export(mod_name=Name, version=version)
async def get_user_from_request(app, request: RequestData):
    if request is None:
        return User()
    name = request.session.user_name
    if name != "anonymous":
        user = await app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=name)
    else:
        user = User()
    return user


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="open_widget", row=True)
async def open_widget(app: App, request, name: str, **kwargs):
    if app is None:
        app = get_app(f"{Name}.open")
    if len(all_widgets) == 0:
        get_all_widget_mods(app)
    if name not in all_widgets:
        return "invalid widget name " +str(name)+" Valid ar :" + str(all_widgets)
    w = await app.a_run_any((name, "get_widget"), request=request, **kwargs)
    print("""WWWW""", w)
    if isinstance(w, asyncio.Task):
        w = await w
        w = w.as_result().get()
    app.print(f"opened widget {name} {w}")
    return w
