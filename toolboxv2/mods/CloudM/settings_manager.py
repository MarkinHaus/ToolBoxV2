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

from toolboxv2.utils.cryp import Code

Name = 'CloudM.settings'
export = get_app(f"{Name}.Export").tb
default_export = export(mod_name=Name)
test_only = export(mod_name=Name, test_only=True, state=False)
version = '0.0.1'

# log in with jwt get instance id

