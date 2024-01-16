from dataclasses import field, asdict
from typing import Any, Optional
from pydantic import BaseModel
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


class ToolBoxError(str, Enum):
    none = "none"
    input_error = "InputError"
    internal_error = "InternalError"
    custom_error = "CustomError"


class ToolBoxInterfaces(str, Enum):
    cli = "CLI"
    api = "API"
    remote = "REMOTE"
    native = "NATIVE"
    internal = "INTERNAL"


@dataclass
class ToolBoxResult:
    data_to: ToolBoxInterfaces or str = field(default=ToolBoxInterfaces.cli)
    data_info: Optional[Any] = field(default=None)
    data: Optional[Any] = field(default=None)
    data_type: Optional[str] = field(default=None)


@dataclass
class ToolBoxInfo:
    exec_code: int
    help_text: str


class ToolBoxResultBM(BaseModel):
    data_to: str = ToolBoxInterfaces.cli.value
    data_info: Optional[str]
    data: Optional[Any]
    data_type: Optional[str]


class ToolBoxInfoBM(BaseModel):
    exec_code: int
    help_text: str


class ApiResult(BaseModel):
    error: Optional[str] = None
    origin: Optional[Any]
    result: Optional[ToolBoxResultBM] = None
    info: Optional[ToolBoxInfoBM]

    def as_result(self):
        return Result(
            error=self.error.value if isinstance(self.error, Enum) else self.error,
            result=ToolBoxResult(
                data_to=self.result.data_to.value if isinstance(self.result.data_to, Enum) else self.result.data_to,
                data_info=self.result.data_info,
                data=self.result.data,
                data_type=self.result.data_type
            ) if self.result else None,
            info=ToolBoxInfo(
                exec_code=self.info.exec_code,
                help_text=self.info.help_text
            ) if self.info else None,
            origin=self.origin
        )


class Result:
    def __init__(self,
                 error: ToolBoxError,
                 result: ToolBoxResult,
                 info: ToolBoxInfo,
                 origin: Optional[Any] = None,
                 ):
        self.error: ToolBoxError = error
        self.result: ToolBoxResult = result
        self.info: ToolBoxInfo = info
        self.origin = origin

    def set_origin(self, origin):
        if self.origin is not None:
            raise ValueError("You cannot Change the origin of a Result!")
        self.origin = origin
        return self

    def is_error(self):
        if test_is_result(self.result.data):
            return self.result.data.is_error()
        return self.info.exec_code != 0

    def is_data(self):
        return self.result.data is not None

    def to_api_result(self):
        print(f" error={self.error}, result= {self.result}, info= {self.info}, origin= {self.origin}")
        return ApiResult(
            error=self.error.value if isinstance(self.error, Enum) else self.error,
            result=ToolBoxResultBM(
                data_to=self.result.data_to.value if isinstance(self.result.data_to, Enum) else self.result.data_to,
                data_info=self.result.data_info,
                data=self.result.data,
                data_type=self.result.data_type
            ) if self.result else None,
            info=ToolBoxInfoBM(
                exec_code=self.info.exec_code,
                help_text=self.info.help_text
            ) if self.info else None,
            origin=self.origin
        )

    @classmethod
    def default(cls, interface=ToolBoxInterfaces.native):
        error = ToolBoxError.none
        info = ToolBoxInfo(exec_code=-1, help_text="")
        result = ToolBoxResult(data_to=interface)
        return cls(error=error, info=info, result=result)

    @classmethod
    def ok(cls, data=None, data_info="", info="OK", interface=ToolBoxInterfaces.native):
        error = ToolBoxError.none
        info = ToolBoxInfo(exec_code=0, help_text=info)
        result = ToolBoxResult(data_to=interface, data=data, data_info=data_info, data_type=type(data).__name__)
        return cls(error=error, info=info, result=result)

    @classmethod
    def custom_error(cls, data=None, data_info="", info="", exec_code=-1, interface=ToolBoxInterfaces.native):
        error = ToolBoxError.custom_error
        info = ToolBoxInfo(exec_code=exec_code, help_text=info)
        result = ToolBoxResult(data_to=interface, data=data, data_info=data_info, data_type=type(data).__name__)
        return cls(error=error, info=info, result=result)

    @classmethod
    def default_user_error(cls, info="", exec_code=-3, interface=ToolBoxInterfaces.native, data=None):
        error = ToolBoxError.input_error
        info = ToolBoxInfo(exec_code, info)
        result = ToolBoxResult(data_to=interface, data=data, data_type=type(data).__name__)
        return cls(error=error, info=info, result=result)

    @classmethod
    def default_internal_error(cls, info="", exec_code=-2, interface=ToolBoxInterfaces.native, data=None):
        error = ToolBoxError.internal_error
        info = ToolBoxInfo(exec_code, info)
        result = ToolBoxResult(data_to=interface, data=data, data_type=type(data).__name__)
        return cls(error=error, info=info, result=result)

    def print(self, show=True, show_data=True, prifix=""):
        data = '\n' + f"{((prifix + 'Data: ' + str(self.result.data) if self.result.data is not None else 'NO Data') if not isinstance(self.result.data, Result) else self.print(show=False, show_data=show_data, prifix=prifix + '-')) if show_data else 'Data: private'}"
        origin = '\n' + f"{prifix + 'Origin: ' + str(self.origin) if self.origin is not None else 'NO Origin'}"
        text = (f"Function Exec coed: {self.info.exec_code}"
                f"\n{prifix}Info's:"
                f" {self.info.help_text} {'<|> '+str(self.result.data_info) if self.result.data_info is not None else ''}"
                f"{origin}{data if not data.endswith('NO Data') else ''}")
        if not show:
            return text
        print(text)
        return self

    def log(self, show_data=True, prifix=""):
        from toolboxv2 import get_logger
        get_logger().debug(self.print(show=False, show_data=show_data, prifix=prifix).replace("\n", " - "))
        return self

    def __str__(self):
        return self.print(show=False, show_data=True)

    def get(self):
        data = self.result.data
        if isinstance(data, Result):
            return data.get()
        return data

    def lazy_return(self, _=0, data=None, **kwargs):
        flags = ['raise', 'logg', 'user', 'intern']
        if isinstance(_, int):
            flag = flags[_]
        else:
            flag = _
        if self.info.exec_code == 0:
            return self if data is None else data if test_is_result(data) else self.ok(data=data, **kwargs)
        if flag == 'raise':
            raise ValueError(self.print(show=False))
        if flag == 'logg':
            from .. import get_logger
            get_logger().error(self.print(show=False))

        if flag == 'user':
            return self if data is None else data if test_is_result(data) else self.default_user_error(data=data, **kwargs)
        if flag == 'intern':
            return self if data is None else data if test_is_result(data) else self.default_internal_error(data=data, **kwargs)

        return self if data is None else data if test_is_result(data) else self.custom_error(data=data, **kwargs)


def test_is_result(data: Result):
    return isinstance(data, Result)


@dataclass
class CallingObject:
    module_name: str = field(default="")
    function_name: str = field(default="")
    args: list or None = field(default=None)
    kwargs: dict or None = field(default=None)

    @classmethod
    def empty(cls):
        return cls()

    def print(self, show=True):
        s = f"{self.module_name=};{self.function_name=};{self.args=};{self.kwargs=}"
        if not show:
            return s
        print(s)
