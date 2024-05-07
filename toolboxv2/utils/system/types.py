import asyncio
import logging
import os
import subprocess
import threading
from dataclasses import field
from inspect import signature
from types import ModuleType
from typing import Any, Optional, List, Tuple, Dict, Callable
from pydantic import BaseModel

from .all_functions_enums import *
from ..extras import generate_test_cases
from .file_handler import FileHandler
from ..extras.Style import Spinner


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
    background_application_runner = False
    docker = False
    install = None
    remove = None
    update = None
    name = 'main'
    port = 8000
    host = '0.0.0.0'
    load_all_mod_in_files = False
    mods_folder = 'toolboxv2.mods.'
    debug = None
    test = None
    profiler = None
    hot_reload = False
    live_application = True
    sysPrint = False

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
    future = "FUTURE"


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

    def to_api_result(self):
        return self

    def print(self, *args, **kwargs):
        res = self.as_result().print(*args, **kwargs)
        if not isinstance(res, str):
            res = res.to_api_result()
        return res


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

    def as_result(self):
        return self

    def set_origin(self, origin):
        if self.origin is not None:
            raise ValueError("You cannot Change the origin of a Result!")
        self.origin = origin
        return self

    def set_dir_origin(self, name, extras="assets/"):
        if self.origin is not None:
            raise ValueError("You cannot Change the origin of a Result!")
        self.origin = f"mods/{name}/{extras}"
        return self

    def is_error(self):
        if _test_is_result(self.result.data):
            return self.result.data.is_error()
        return self.info.exec_code != 0

    def is_data(self):
        return self.result.data is not None

    def to_api_result(self):
        # print(f" error={self.error}, result= {self.result}, info= {self.info}, origin= {self.origin}")
        return ApiResult(
            error=self.error.value if isinstance(self.error, Enum) else self.error,
            result=ToolBoxResultBM(
                data_to=self.result.data_to.value if isinstance(self.result.data_to, Enum) else self.result.data_to,
                data_info=self.result.data_info,
                data=self.result.data,
                data_type=self.result.data_type
            ) if self.result else None,
            info=ToolBoxInfoBM(
                exec_code=self.info.exec_code,  # exec_code umwandel in http resposn codes
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
    def future(cls, data=None, data_info="", info="OK", interface=ToolBoxInterfaces.future):
        error = ToolBoxError.none
        info = ToolBoxInfo(exec_code=0, help_text=info)
        result = ToolBoxResult(data_to=interface, data=data, data_info=data_info, data_type="future")
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
        data = '\n' + f"{((prifix + 'Data: ' + str(self.result.data) if self.result.data is not None else 'NO Data') if not isinstance(self.result.data, Result) else self.result.data.print(show=False, show_data=show_data, prifix=prifix + '-')) if show_data else 'Data: private'}"
        origin = '\n' + f"{prifix + 'Origin: ' + str(self.origin) if self.origin is not None else 'NO Origin'}"
        text = (f"Function Exec coed: {self.info.exec_code}"
                f"\n{prifix}Info's:"
                f" {self.info.help_text} {'<|> ' + str(self.result.data_info) if self.result.data_info is not None else ''}"
                f"{origin}{data if not data.endswith('NO Data') else ''}")
        if not show:
            return text
        print("\n======== Result ========\n" + text + "\n------- EndOfD -------")
        return self

    def log(self, show_data=True, prifix=""):
        from toolboxv2 import get_logger
        get_logger().debug(self.print(show=False, show_data=show_data, prifix=prifix).replace("\n", " - "))
        return self

    def __str__(self):
        return self.print(show=False, show_data=True)

    def get(self, key=None, default=None):
        data = self.result.data
        if isinstance(data, Result):
            return data.get(key=key, default=default)
        if key is not None and isinstance(data, dict):
            return data.get(key, default)
        return data if data is not None else default

    async def aget(self, key=None, default=None):
        if self.result.data_to.name == ToolBoxInterfaces.future.name:
            data = await self.result.data
        else:
            data = self.get(key=None, default=None)
        if isinstance(data, Result):
            return data.get(key=key, default=default)
        if key is not None and isinstance(data, dict):
            return data.get(key, default)
        return data if data is not None else default

    def lazy_return(self, _=0, data=None, **kwargs):
        flags = ['raise', 'logg', 'user', 'intern']
        if isinstance(_, int):
            flag = flags[_]
        else:
            flag = _
        if self.info.exec_code == 0:
            return self if data is None else data if _test_is_result(data) else self.ok(data=data, **kwargs)
        if flag == 'raise':
            raise ValueError(self.print(show=False))
        if flag == 'logg':
            from .. import get_logger
            get_logger().error(self.print(show=False))

        if flag == 'user':
            return self if data is None else data if _test_is_result(data) else self.default_user_error(data=data,
                                                                                                        **kwargs)
        if flag == 'intern':
            return self if data is None else data if _test_is_result(data) else self.default_internal_error(data=data,
                                                                                                            **kwargs)

        return self if data is None else data if _test_is_result(data) else self.custom_error(data=data, **kwargs)


def _test_is_result(data: Result):
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

    def __str__(self):
        if self.args is not None and self.kwargs is not None:
            return (f"{self.module_name} {self.function_name} " + ' '.join(self.args) +
                    ' '.join([key + '-' + str(val) for key, val in self.kwargs.items()]))
        if self.args is not None:
            return f"{self.module_name} {self.function_name} " + ' '.join(self.args)
        return f"{self.module_name} {self.function_name}"

    def print(self, show=True):
        s = f"{self.module_name=};{self.function_name=};{self.args=};{self.kwargs=}"
        if not show:
            return s
        print(s)


def analyze_data(data):
    report = []
    for mod_name, mod_info in data.items():
        if mod_name in ['modular_run', 'modular_fatal_error', 'modular_sug']:
            continue  # Überspringen der allgemeinen Statistiken
        if mod_name in ['errors']:
            report.append(f"Total errors: {mod_info}")
            continue
        if mod_name == 'total_coverage':
            continue
        if mod_name == 'coverage':
            _ = '\t'.join(mod_info)
            report.append(f"Total coverage:\n {_}")
            continue
        report.append(f"Modul: {mod_name}")
        report.append(f"  Funktionen ausgeführt: {mod_info.get('functions_run', 0)}")
        report.append(f"  Funktionen mit Fatalen Fehler: {mod_info.get('functions_fatal_error', 0)}")
        report.append(f"  Funktionen mit Fehler: {mod_info.get('error', 0)}")
        report.append(f"  Funktionen erfolgreich: {mod_info.get('functions_sug', 0)}")
        if mod_info.get('coverage', [0])[0] == 0:
            c = 0
        else:
            c = mod_info.get('coverage', [0, 1])[1] / mod_info.get('coverage', [1])[0]
        report.append(f"  coverage: {c:.2f}")

        if 'callse' in mod_info and mod_info['callse']:
            report.append("  Fehler:")
            for func_name, errors in mod_info['callse'].items():
                for error in errors:
                    if isinstance(error, str):
                        error = error.replace('\n', ' - ')
                    report.append(f"    - {func_name}, Fehler: {error}")
    return "\n".join(report)


U = Any
A = Any


class MainToolType:
    toolID: str
    app: A
    interface: ToolBoxInterfaces
    spec: str

    version: str
    tools: dict  # TODO: own type
    name: str
    logger: logging
    color: str
    todo: Callable
    _on_exit: Callable
    stuf: bool
    config: dict
    user: Optional[U]
    description: str

    @staticmethod
    def return_result(error: ToolBoxError = ToolBoxError.none,
                      exec_code: int = 0,
                      help_text: str = "",
                      data_info=None,
                      data=None,
                      data_to=None) -> Result:
        """proxi attr"""

    def load(self):
        """proxi attr"""

    def print(self, message, end="\n", **kwargs):
        """proxi attr"""

    def add_str_to_config(self, command):
        if len(command) != 2:
            self.logger.error('Invalid command must be key value')
            return False
        self.config[command[0]] = command[1]

    def webInstall(self, user_instance, construct_render) -> str:
        """"Returns a web installer for the given user instance and construct render template"""

    def get_user(self, username: str) -> Result:
        return self.app.run_any(CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=username, get_results=True)


class AppType:
    prefix: str
    id: str
    globals: Dict[str, Any] = {"root": dict, }
    locals: Dict[str, Any] = {"user": {'app': "self"}, }

    data_dir: str
    config_dir: str
    info_dir: str

    logger: logging
    logging_filename: str

    version: str
    loop: asyncio.AbstractEventLoop

    keys: Dict[str, str] = {
        "MACRO": "macro~~~~:",
        "MACRO_C": "m_color~~:",
        "HELPER": "helper~~~:",
        "debug": "debug~~~~:",
        "id": "name-spa~:",
        "st-load": "mute~load:",
        "comm-his": "comm-his~:",
        "develop-mode": "dev~mode~:",
        "provider::": "provider::",
    }

    defaults: Dict[str, Optional[bool or Dict or Dict[str, Dict[str, str]] or str or List[str] or List[List]]] = {
        "MACRO": List[str],
        "MACRO_C": Dict,
        "HELPER": Dict,
        "debug": str,
        "id": str,
        "st-load": False,
        "comm-his": List[List],
        "develop-mode": bool,
    }

    config_fh: FileHandler
    _debug: bool
    runnable: Dict[str, Callable]
    dev_modi: bool
    functions: Dict[str, Any]

    interface_type: ToolBoxInterfaces
    REFIX: str

    alive: bool
    called_exit: Tuple[bool, float]
    args_sto: AppArgs
    system_flag = None
    exit_tasks = []

    def __init__(self, prefix: Optional[str] = None, args: Optional[AppArgs] = None):
        self.sto = None
        self.args_sto = args
        """proxi attr"""

    @staticmethod
    def exit_main(*args, **kwargs):
        """proxi attr"""

    @staticmethod
    def hide_console(*args, **kwargs):
        """proxi attr"""

    @staticmethod
    def show_console(*args, **kwargs):
        """proxi attr"""

    @staticmethod
    def disconnect(*args, **kwargs):
        """proxi attr"""

    def set_logger(self, debug=False):
        """proxi attr"""

    @property
    def debug(self):
        """proxi attr"""
        return self._debug

    def debug_rains(self, e):
        """proxi attr"""

    def set_runnable(self, r):
        """proxi attr"""

    def run_runnable(self, name, **kwargs):
        """proxi attr"""

    def rrun_runnable(self, name, **kwargs):
        """proxi attr"""

    @debug.setter
    def debug(self, value):
        """proxi attr"""

    def _coppy_mod(self, content, new_mod_dir, mod_name, file_type='py'):
        """proxi attr"""

    def _pre_lib_mod(self, mod_name, path_to="./runtime", file_type='py'):
        """proxi attr"""

    def _copy_load(self, mod_name, file_type='py', **kwargs):
        """proxi attr"""

    def inplace_load_instance(self, mod_name, loc="toolboxv2.mods.", spec='app', save=True):
        """proxi attr"""

    def save_instance(self, instance, modular_id, spec='app', instance_type="file/application", tools_class=None):
        """proxi attr"""

    def save_initialized_module(self, tools_class, spec):
        """proxi attr"""

    def mod_online(self, mod_name, installed=False):
        """proxi attr"""

    def _get_function(self,
                      name: Enum or None,
                      state: bool = True,
                      specification: str = "app",
                      metadata=False, as_str: tuple or None = None, r=0):
        """proxi attr"""

    def save_exit(self):
        """proxi attr"""

    def load_mod(self, mod_name: str, mlm='I', **kwargs):
        """proxi attr"""

    async def load_all_mods_in_file(self, working_dir="mods"):
        """proxi attr"""

    def get_all_mods(self, working_dir="mods", path_to="./runtime"):
        """proxi attr"""

    def remove_all_modules(self, delete=False):
        for mod in list(self.functions.keys()):
            self.logger.info(f"closing: {mod}")
            self.remove_mod(mod, delete=delete)

    async def a_remove_all_modules(self, delete=False):
        for mod in list(self.functions.keys()):
            self.logger.info(f"closing: {mod}")
            await self.a_remove_mod(mod, delete=delete)

    def print_ok(self):
        """proxi attr"""
        self.logger.info("OK")

    def remove_mod(self, mod_name, spec='app', delete=True):
        """proxi attr"""

    async def a_remove_mod(self, mod_name, spec='app', delete=True):
        """proxi attr"""

    def exit(self):
        """proxi attr"""

    async def a_exit(self):
        """proxi attr"""

    def save_load(self, modname, spec='app'):
        """proxi attr"""

    def get_function(self, name: Enum or tuple, **kwargs):
        """
        Kwargs for _get_function
            metadata:: return the registered function dictionary
                stateless: (function_data, None), 0
                stateful: (function_data, higher_order_function), 0
            state::boolean
                specification::str default app
        """

    def run_a_from_sync(self, function):
        """
        run a async fuction
        """

    def run_function(self, mod_function_name: Enum or tuple,
                     tb_run_function_with_state=True,
                     tb_run_with_specification='app',
                     args_=None,
                     kwargs_=None,
                     *args,
                     **kwargs) -> Result:

        """proxi attr"""

    def fuction_runner(self, function, function_data: dict, args: list, kwargs: dict):
        """
        parameters = function_data.get('params')
        modular_name = function_data.get('module_name')
        function_name = function_data.get('func_name')
        mod_function_name = f"{modular_name}.{function_name}"

        proxi attr
        """

    def run_any(self, mod_function_name: Enum or str or tuple, backwords_compability_variabel_string_holder=None,
                get_results=False, tb_run_function_with_state=True, tb_run_with_specification='app', args_=None,
                kwargs_=None,
                *args, **kwargs):
        """proxi attr"""

    def get_mod(self, name, spec='app') -> ModuleType or MainToolType:
        """proxi attr"""

    @staticmethod
    def print(text, *args, **kwargs):
        """proxi attr"""

    @staticmethod
    def sprint(text, *args, **kwargs):
        """proxi attr"""

    # ----------------------------------------------------------------
    # Decorators for the toolbox

    def _register_function(self, module_name, func_name, data):
        """proxi attr"""

    def _create_decorator(self, type_: str,
                          name: str = "",
                          mod_name: str = "",
                          level: int = -1,
                          restrict_in_virtual_mode: bool = False,
                          api: bool = False,
                          helper: str = "",
                          version: str or None = None,
                          initial=False,
                          exit_f=False,
                          test=True,
                          samples=None,
                          state=None,
                          pre_compute=None,
                          post_compute=None,
                          memory_cache=False,
                          file_cache=False,
                          row=False,
                          request_as_kwarg=False,
                          memory_cache_max_size=100,
                          memory_cache_ttl=300):
        """proxi attr"""

        # data = {
        #     "type": type_,
        #     "module_name": module_name,
        #     "func_name": func_name,
        #     "level": level,
        #     "restrict_in_virtual_mode": restrict_in_virtual_mode,
        #     "func": func,
        #     "api": api,
        #     "helper": helper,
        #     "version": version,
        #     "initial": initial,
        #     "exit_f": exit_f,
        #     "__module__": func.__module__,
        #     "signature": sig,
        #     "params": params,
        #     "state": (
        #         False if len(params) == 0 else params[0] in ['self', 'state', 'app']) if state is None else state,
        #     "do_test": test,
        #     "samples": samples,
        #     "request_as_kwarg": request_as_kwarg,

    def tb(self, name=None,
           mod_name: str = "",
           helper: str = "",
           version: str or None = None,
           test: bool = True,
           restrict_in_virtual_mode: bool = False,
           api: bool = False,
           initial: bool = False,
           exit_f: bool = False,
           test_only: bool = False,
           memory_cache: bool = False,
           file_cache: bool = False,
           row=False,
           request_as_kwarg: bool = False,
           state: bool or None = None,
           level: int = 0,
           memory_cache_max_size: int = 100,
           memory_cache_ttl: int = 300,
           samples: list or dict or None = None,
           interface: ToolBoxInterfaces or None or str = None,
           pre_compute=None,
           post_compute=None,
           ):
        """
    A decorator for registering and configuring functions within a module.

    This decorator is used to wrap functions with additional functionality such as caching, API conversion, and lifecycle management (initialization and exit). It also handles the registration of the function in the module's function registry.

    Args:
        name (str, optional): The name to register the function under. Defaults to the function's own name.
        mod_name (str, optional): The name of the module the function belongs to.
        helper (str, optional): A helper string providing additional information about the function.
        version (str or None, optional): The version of the function or module.
        test (bool, optional): Flag to indicate if the function is for testing purposes.
        restrict_in_virtual_mode (bool, optional): Flag to restrict the function in virtual mode.
        api (bool, optional): Flag to indicate if the function is part of an API.
        initial (bool, optional): Flag to indicate if the function should be executed at initialization.
        exit_f (bool, optional): Flag to indicate if the function should be executed at exit.
        test_only (bool, optional): Flag to indicate if the function should only be used for testing.
        memory_cache (bool, optional): Flag to enable memory caching for the function.
        request_as_kwarg (bool, optional): Flag to get request if the fuction is calld from api.
        file_cache (bool, optional): Flag to enable file caching for the function.
        row (bool, optional): rather to auto wrap the result in Result type default False means no row data aka result type
        state (bool or None, optional): Flag to indicate if the function maintains state.
        level (int, optional): The level of the function, used for prioritization or categorization.
        memory_cache_max_size (int, optional): Maximum size of the memory cache.
        memory_cache_ttl (int, optional): Time-to-live for the memory cache entries.
        samples (list or dict or None, optional): Samples or examples of function usage.
        interface (str, optional): The interface type for the function.
        pre_compute (callable, optional): A function to be called before the main function.
        post_compute (callable, optional): A function to be called after the main function.

    Returns:
        function: The decorated function with additional processing and registration capabilities.
    """
        if interface is None:
            interface = "tb"
        if test_only and 'test' not in self.id:
            return lambda *args, **kwargs: args
        return self._create_decorator(interface,
                                      name,
                                      mod_name,
                                      level=level,
                                      restrict_in_virtual_mode=restrict_in_virtual_mode,
                                      helper=helper,
                                      api=api,
                                      version=version,
                                      initial=initial,
                                      exit_f=exit_f,
                                      test=test,
                                      samples=samples,
                                      state=state,
                                      pre_compute=pre_compute,
                                      post_compute=post_compute,
                                      memory_cache=memory_cache,
                                      file_cache=file_cache,
                                      row=row,
                                      request_as_kwarg=request_as_kwarg,
                                      memory_cache_max_size=memory_cache_max_size,
                                      memory_cache_ttl=memory_cache_ttl)

    def print_functions(self, name=None):
        if not self.functions:
            print("Nothing to see")
            return

        def helper(_functions):
            for func_name, data in _functions.items():
                if not isinstance(data, dict):
                    continue

                func_type = data.get('type', 'Unknown')
                func_level = 'r' if data['level'] == -1 else data['level']
                api_status = 'Api' if data.get('api', False) else 'Non-Api'

                print(f"  Function: {func_name}{data.get('signature', '()')}; "
                      f"Type: {func_type}, Level: {func_level}, {api_status}")

        if name is not None:
            functions = self.functions.get(name)
            if functions is not None:
                print(f"\nModule: {name}; Type: {functions.get('app_instance_type', 'Unknown')}")
                helper(functions)
                return
        for module, functions in self.functions.items():
            print(f"\nModule: {module}; Type: {functions.get('app_instance_type', 'Unknown')}")
            helper(functions)

    def save_autocompletion_dict(self):
        """proxi attr"""

    def get_autocompletion_dict(self):
        """proxi attr"""

    def get_username(self, get_input=False):
        """proxi attr"""

    def save_registry_as_enums(self, directory: str, filename: str):
        """proxi attr"""

    async def execute_all_functions(self, m_query='', f_query=''):
        print("Executing all functions")
        all_data = {
            "modular_run": 0,
            "modular_fatal_error": 0,
            "errors": 0,
            "modular_sug": 0,
            "coverage": [],
            "total_coverage": {},
        }
        items = list(self.functions.items()).copy()
        for module_name, functions in items:
            infos = {
                "functions_run": 0,
                "functions_fatal_error": 0,
                "error": 0,
                "functions_sug": 0,
                'calls': {},
                'callse': {},
                "coverage": [0, 0],
            }
            all_data['modular_run'] += 1
            if not module_name.startswith(m_query):
                all_data['modular_sug'] += 1
                continue

            with Spinner(message=f"In {module_name}| "):
                f_items = list(functions.items()).copy()
                for function_name, function_data in f_items:
                    if not isinstance(function_data, dict):
                        continue
                    if not function_name.startswith(f_query):
                        continue
                    test: list = function_data.get('do_test')
                    # print(test, module_name, function_name, function_data)
                    infos["coverage"][0] += 1
                    if test is False:
                        continue

                    with Spinner(message=f"\t\t\t\t\t\tfuction {function_name}..."):
                        params: list = function_data.get('params')
                        sig: signature = function_data.get('signature')
                        state: bool = function_data.get('state')
                        samples: bool = function_data.get('samples')

                        test_kwargs_list = [{}]

                        if params is not None:
                            test_kwargs_list = samples if samples is not None else generate_test_cases(sig=sig)
                            # print(test_kwargs)
                            # print(test_kwargs[0])
                            # test_kwargs = test_kwargs_list[0]
                        # print(module_name, function_name, test_kwargs_list)
                        infos["coverage"][1] += 1
                        for test_kwargs in test_kwargs_list:
                            try:
                                # print(f"test Running {state=} |{module_name}.{function_name}")
                                result = self.run_function((module_name, function_name),
                                                           tb_run_function_with_state=state,
                                                           **test_kwargs)
                                if not isinstance(result, Result):
                                    result = Result.ok(result)
                                if result.info.exec_code == 0:
                                    infos['calls'][function_name] = [test_kwargs, str(result)]
                                    infos['functions_sug'] += 1
                                else:
                                    infos['functions_sug'] += 1
                                    infos['error'] += 1
                                    infos['callse'][function_name] = [test_kwargs, str(result)]
                            except Exception as e:
                                infos['functions_fatal_error'] += 1
                                infos['callse'][function_name] = [test_kwargs, str(e)]
                            finally:
                                infos['functions_run'] += 1

                if infos['functions_run'] == infos['functions_sug']:
                    all_data['modular_sug'] += 1
                else:
                    all_data['modular_fatal_error'] += 1
                if infos['error'] > 0:
                    all_data['errors'] += infos['error']

                all_data[module_name] = infos
                if infos['coverage'][0] == 0:
                    c = 0
                else:
                    c = infos['coverage'][1] / infos['coverage'][0]
                all_data["coverage"].append(f"{module_name}:{c:.2f}\n")
        total_coverage = sum([float(t.split(":")[-1]) for t in all_data["coverage"]]) / len(all_data["coverage"])
        print(
            f"\n{all_data['modular_run']=}\n{all_data['modular_sug']=}\n{all_data['modular_fatal_error']=}\n{total_coverage=}")
        d = analyze_data(all_data)
        return Result.ok(data=all_data, data_info=d)

    @staticmethod
    def calculate_complexity(filename_or_code):
        from radon.complexity import cc_rank, cc_visit
        if os.path.exists(filename_or_code):
            with open(filename_or_code, 'r') as file:
                code = file.read()
        else:
            code = filename_or_code

        # Calculate and print Cyclomatic Complexity
        complexity_results = cc_visit(code)
        i = -1
        avg_complexity = 0
        rang_complexity = 0
        for block in complexity_results:
            complexity = block.complexity
            i += 1
            print(f"block: {block.name} {i} Class/Fuction/Methode : {block.letter}")
            print(f"    fullname: {block.fullname}")
            print(f"    Cyclomatic Complexity: {complexity}")
            # Optional: Get complexity rank
            avg_complexity += complexity
            rank = cc_rank(complexity)
            print(f"    Complexity Rank: {rank}")
            # print(f"    lineno: {block.lineno}")
            print(f"    endline: {block.endline}")
            print(f"    col_offset: {block.col_offset}\n")
        if i <= 0:
            i += 2
        avg_complexity = avg_complexity / i
        print(f"\nAVG Complexity: {avg_complexity:.2f}")
        print(f"Total Rank: {cc_rank(int(avg_complexity + i // 10))}")
