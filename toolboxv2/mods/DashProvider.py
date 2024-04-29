from toolboxv2 import get_app, App, Result, tbef
from toolboxv2.utils.system.types import ToolBoxInterfaces
from fastapi import Request

Name = 'DashProvider'
export = get_app("DashProvider.Export").tb
default_export = export(mod_name=Name)
version = '0.0.1'
spec = ''


@export(mod_name=Name, name='Version', version=version)
def get_version():
    return version


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="get_controller")
def get_controller(app: App = None, request: Request or None = None):
    if app is None:
        app = get_app(from_=f"{Name}.controller")
    if request is None:
        return Result.default_internal_error("No request specified")
    print(spec)

    print(request.session['live_data'].get('spec') == spec)

    # app.run_any(tbef.MINIMALHTML.GENERATE_HTML)

    return """<div>
    <p>Neue Steuerungselemente geladen!</p>
    <!-- Weitere Steuerelemente hier -->
</div>
"""


@export(mod_name=Name, version=version, level=1, api=True, name="getsMSG", interface=ToolBoxInterfaces.remote)
def getsMSG(app: App = None):
    if app is None:
        app = get_app(from_=f"{Name}.getTextWidget")
    # Sendeng system MSG message
    systemMSG_content = app.run_any(tbef.WEBSOCKETMANAGER.CONSTRUCT_RENDER, content="./web/systemMSG/text.html",
                                    element_id="SpeechBallonControls",
                                    externals=["/web/systemMSG/speech_balloon.js"],
                                    from_file=True, to_str=False)

    return systemMSG_content


@export(mod_name=Name, version=version, level=1, api=True, name="getsInsightsWidget",
        interface=ToolBoxInterfaces.remote)
def getsInsightsWidget(app: App = None):
    if app is None:
        app = get_app(from_=f"{Name}.getTextWidget")
    # Sendeng system MSG message
    insights_content = app.run_any(tbef.WEBSOCKETMANAGER.CONSTRUCT_RENDER,
                                   content="./web/1/insightsWidget/insights.html",
                                   element_id="widgetInsights",
                                   externals=[],
                                   from_file=True, to_str=False)

    return insights_content['render']['content']


@export(mod_name=Name, version=version, level=1, api=True, name="getTextWidget", interface=ToolBoxInterfaces.remote)
def getTextWidget(app: App = None):
    if app is None:
        app = get_app(from_=f"{Name}.getTextWidget")
    # Sendeng system MSG message
    widgetText_content = app.run_any(tbef.WEBSOCKETMANAGER.CONSTRUCT_RENDER, content="./web/1/textWidet/text.html",
                                     element_id="widgetText",
                                     externals=["/web/1/textWidet/testWiget.js"],
                                     from_file=True, to_str=False)

    return widgetText_content


@export(mod_name=Name, version=version, level=1, api=True, name="getPathWidget", interface=ToolBoxInterfaces.remote)
def getPathWidget(app: App = None):
    if app is None:
        app = get_app(from_=f"{Name}.getTextWidget")
    # Sendeng system MSG message
    widgetPath_content = app.run_any(tbef.WEBSOCKETMANAGER.CONSTRUCT_RENDER, content="./web/1/PathWidet/text.html",
                                     element_id="widgetPath",
                                     externals=["/web/1/PathWidet/pathWiget.js"],
                                     from_file=True, to_str=False)

    return widgetPath_content


@export(mod_name=Name, version=version, level=1, api=True, name="getWidgetNave", interface=ToolBoxInterfaces.remote)
# Sendeng system MSG message
def getWidgetNave(app: App = None):
    if app is None:
        app = get_app(from_=f"{Name}.getTextWidget")
    # Sendeng system MSG message
    widgetNav_content = app.run_any(tbef.WEBSOCKETMANAGER.CONSTRUCT_RENDER, content="./web/1/WigetNav/navDrow.html",
                                    element_id="controls",
                                    externals=["/web/1/WigetNav/module/navDrow.js"],
                                    from_file=True, to_str=False)

    return widgetNav_content


@export(mod_name=Name, version=version, level=1, api=True, name="getDrag", interface=ToolBoxInterfaces.remote)
def getDrag(app: App = None):
    if app is None:
        app = get_app(from_=f"{Name}.getTextWidget")
    # Sendeng system MSG message
    drag_content = app.run_any(tbef.WEBSOCKETMANAGER.CONSTRUCT_RENDER, content="./web/Drag/drag.html",
                               element_id="DragControls",
                               externals=["/web/Drag/module/drag.js"],
                               from_file=True, to_str=False)
    return drag_content


@export(mod_name=Name, version=version, level=1, api=True, name="getControls", interface=ToolBoxInterfaces.remote)
def getControls(app: App = None):
    if app is None:
        app = get_app(from_=f"{Name}.getTextWidget")
    # Sendeng system MSG message
    controller_content = app.run_any(tbef.WEBSOCKETMANAGER.CONSTRUCT_RENDER,
                                     content="""<li class="dropdown-item" id="ControlsWidget">Controls</li>""",
                                     element_id="editorWidget",
                                     externals=["/web/1/Controler/controller.js"], to_str=False)

    return controller_content


@export(mod_name=Name, version=version, level=1, api=True, name="serviceWorker", interface=ToolBoxInterfaces.remote)
def serviceWorker(app: App = None):
    if app is None:
        app = get_app(from_=f"{Name}.getTextWidget")
    # Sendeng system MSG message
    sw_content = app.run_any(tbef.WEBSOCKETMANAGER.CONSTRUCT_RENDER, content="",
                             element_id="control1",
                             externals=["/web/index.js", "/web/sw.js"], to_str=False)
    return sw_content
