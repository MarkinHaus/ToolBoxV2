import base64
import datetime
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Union, List

import jwt
from pydantic import BaseModel

from toolboxv2.mods.DB.types import DatabaseModes
from toolboxv2.utils.types import ToolBoxInterfaces, ApiResult
from toolboxv2 import get_app, App, Result, tbef, ToolBox_over
from fastapi import Request
from toolboxv2.utils.cryp import Code

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
