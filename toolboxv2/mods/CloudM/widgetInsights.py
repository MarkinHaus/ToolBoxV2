from toolboxv2 import get_app, App, Result
from fastapi import Request

Name = 'CloudM.settings'
export = get_app(f"{Name}.Export").tb
default_export = export(mod_name=Name)
test_only = export(mod_name=Name, test_only=True, state=False)
version = '0.0.1'
spec = ''

# log in with jwt get instance id


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
