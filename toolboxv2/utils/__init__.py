from dataclasses import field
from typing import Any

from .all_functions_enums import *


class AppArgs:
    init = None
    init_file = 'init.config'
    get_version = False
    mm = False
    sm = False
    lm = False
    modi = 'cli'
    kill = False
    remote = False
    remote_direct_key = None
    background_application = False
    docker = False
    install = None
    remove = None
    update = None
    mod_version_name = 'mainTool'
    name = 'main'
    port = 8000
    host = '0.0.0.0'
    load_all_mod_in_files = False
    mods_folder = 'toolboxv2.mods.'
    debug = None
    test = None
    profiler = None

    def default(self):
        return self


class ApiOb:
    token = ""
    data = {}

    def __init__(self, data=None, token=""):
        if data is None:
            data = {}
        self.data = data
        self.token = token

    def default(self):
        return self


class Singleton(type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}
    _kwargs = {}
    _args = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            cls._args[cls] = args
            cls._kwargs[cls] = kwargs
        return cls._instances[cls]


class ToolBoxError(Enum):
    none = "none"
    input_error = "InputError"
    internal_error = "InternalError"
    custom_error = "CustomError"


class ToolBoxInterfaces(Enum):
    cli = "CLI"
    api = "API"
    remote = "REMOTE"
    native = "NATIVE"
    internal = "INTERNAL"


@dataclass
class ToolBoxResult:
    data_to: ToolBoxInterfaces or str = field(default=ToolBoxInterfaces.cli)
    data_info: Any or None = field(default=None)
    data: Any or None = field(default=None)
    data_type: str or None = field(default=None)


@dataclass
class ToolBoxInfo:
    exec_code: int
    help_text: str


class Result:
    def __init__(self,
                 error: ToolBoxError,
                 result: ToolBoxResult,
                 info: ToolBoxInfo,
                 ):
        self.error = error
        self.result = result
        self.info = info
        self.origin = None

    def set_origin(self, origin):
        if self.origin is not None:
            raise ValueError("You cannot Change the origin of a Result!")
        self.origin = origin
        return self

    @classmethod
    def default(cls, interface=ToolBoxInterfaces.native):
        error = ToolBoxError.none
        info = ToolBoxInfo(exec_code=-1, help_text="")
        result = ToolBoxResult(data_to=interface)
        return cls(error=error, info=info, result=result)

    @classmethod
    def ok(cls, data=None, data_info=None, info="OK", interface=ToolBoxInterfaces.native):
        error = ToolBoxError.none
        info = ToolBoxInfo(exec_code=0, help_text=info)
        result = ToolBoxResult(data_to=interface, data=data, data_info=data_info, data_type=type(data))
        return cls(error=error, info=info, result=result)

    @classmethod
    def custom_error(cls, data=None, data_info=None, info="", exec_code=-1, interface=ToolBoxInterfaces.native):
        error = ToolBoxError.custom_error
        info = ToolBoxInfo(exec_code=exec_code, help_text=info)
        result = ToolBoxResult(data_to=interface, data=data, data_info=data_info, data_type=type(data))
        return cls(error=error, info=info, result=result)

    @classmethod
    def default_user_error(cls, info="", exec_code=-3, interface=ToolBoxInterfaces.native):
        error = ToolBoxError.input_error
        info = ToolBoxInfo(exec_code, info)
        result = ToolBoxResult(data_to=interface)
        return cls(error=error, info=info, result=result)

    @classmethod
    def default_internal_error(cls, info="", exec_code=-2, interface=ToolBoxInterfaces.native):
        error = ToolBoxError.internal_error
        info = ToolBoxInfo(exec_code, info)
        result = ToolBoxResult(data_to=interface)
        return cls(error=error, info=info, result=result)

    def print(self, show=True, show_data=True, prifix=""):
        data = '\n'+f"{((prifix+'Data: '+str(self.result.data) if self.result.data is not None else 'NO Data') if not isinstance(self.result.data, Result) else self.print(show=False, show_data=show_data, prifix=prifix+'-')) if show_data else 'private'}"
        origin = '\n'+f"{prifix+'Origin: '+str(self.origin) if self.origin is not None else 'NO Origin'}"
        text = (f"Function Exec coed: {self.info.exec_code}"
                f"\n{prifix}Info's: {self.info.help_text}"
                f"{origin}{data if not data.endswith('NO Data') else ''}")
        if not show:
            return text
        print(text)

    def get(self):
        data = self.result.data
        if isinstance(data, Result):
            return data.get()
        return data


@dataclass
class CallingObject:
    module_name: str = field(default="")
    function_name: str = field(default="")
    kwargs: dict or None = field(default=None)

    @classmethod
    def empty(cls):
        return cls()

    def print(self):
        print(f"{self.module_name=};{self.function_name=};{self.kwargs=}")
