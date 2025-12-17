"""Main module."""
import asyncio
import inspect
import json
import logging
import os
import pkgutil
import sys
import threading
import time
from asyncio import Task
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from functools import partial, wraps
from importlib import import_module, reload
from inspect import signature
from platform import node, system
from types import ModuleType
from typing import Any

from dotenv import load_dotenv

from .extras.mkdocs import add_to_app
from ..utils.system.main_tool import get_version_from_pyproject
from .extras.Style import Spinner, Style, stram_print
from .singelton_class import Singleton
from .system.cache import FileCache, MemoryCache
from .system.file_handler import FileHandler
from .system.getting_and_closing_app import get_app
from .system.tb_logger import get_logger, setup_logging, loggerNameOfToolboxv2
from .system.types import (
    ApiResult,
    AppArgs,
    AppType,
    MainToolType,
    RequestData,
    Result,
    ToolBoxInterfaces,
    WebSocketContext,
)

load_dotenv()


class App(AppType, metaclass=Singleton):

    def __init__(self, prefix: str = "", args=AppArgs().default()):
        if "test" not in prefix:
            self.logger_prefix = self.REFIX = prefix
            prefix = "main"
        super().__init__(prefix, args)
        self._web_context = None
        t0 = time.perf_counter()
        abspath = os.path.abspath(__file__)
        self.system_flag = system()  # Linux: Linux Mac: Darwin Windows: Windows

        self.appdata = os.getenv('APPDATA') if os.name == 'nt' else os.getenv('XDG_CONFIG_HOME') or os.path.expanduser(
                '~/.config') if os.name == 'posix' else None

        if self.system_flag == "Darwin" or self.system_flag == "Linux":
            dir_name = os.path.dirname(abspath).replace("/utils", "")
        else:
            dir_name = os.path.dirname(abspath).replace("\\utils", "")

        self.start_dir = str(dir_name)

        self.bg_tasks = []

        lapp = dir_name + '\\.data\\'

        if not prefix:
            if not os.path.exists(f"{lapp}last-app-prefix.txt"):
                os.makedirs(lapp, exist_ok=True)
                open(f"{lapp}last-app-prefix.txt", "a").close()
            with open(f"{lapp}last-app-prefix.txt") as prefix_file:
                cont = prefix_file.read()
                if cont:
                    prefix = cont.rstrip()
        else:
            if not os.path.exists(f"{lapp}last-app-prefix.txt"):
                os.makedirs(lapp, exist_ok=True)
                open(f"{lapp}last-app-prefix.txt", "a").close()
            with open(f"{lapp}last-app-prefix.txt", "w") as prefix_file:
                prefix_file.write(prefix)

        self.prefix = prefix

        node_ = node()

        if 'localhost' in node_ and (host := os.getenv('HOSTNAME', 'localhost')) != 'localhost':
            node_ = node_.replace('localhost', host)
        self.id = prefix + '-' + node_
        self.globals = {
            "root": {**globals()},
        }
        self.locals = {
            "user": {'app': self, **locals()},
        }

        identification = self.id
        collective_identification = self.id
        if "test" in prefix:
            if self.system_flag == "Darwin" or self.system_flag == "Linux":
                start_dir = self.start_dir.replace("ToolBoxV2/toolboxv2", "toolboxv2")
            else:
                start_dir = self.start_dir.replace("ToolBoxV2\\toolboxv2", "toolboxv2")
            self.data_dir = start_dir + '\\.data\\' + "test"
            self.config_dir = start_dir + '\\.config\\' + "test"
            self.info_dir = start_dir + '\\.info\\' + "test"
        elif identification.startswith('collective-'):
            collective_identification = identification.split('-')[1]
            self.data_dir = self.start_dir + '\\.data\\' + collective_identification
            self.config_dir = self.start_dir + '\\.config\\' + collective_identification
            self.info_dir = self.start_dir + '\\.info\\' + collective_identification
            self.id = collective_identification
        else:
            self.data_dir = self.start_dir + '\\.data\\' + identification
            self.config_dir = self.start_dir + '\\.config\\' + identification
            self.info_dir = self.start_dir + '\\.info\\' + identification

        if self.appdata is None:
            self.appdata = self.data_dir
        else:
            self.appdata += "/ToolBoxV2"

        if not os.path.exists(self.appdata):
            os.makedirs(self.appdata, exist_ok=True)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
        if not os.path.exists(self.info_dir):
            os.makedirs(self.info_dir, exist_ok=True)

        self.print(f"Starting ToolBox as {prefix} from :", Style.Bold(Style.CYAN(f"{os.getcwd()}")))

        pid_file = f"{self.start_dir}\\.info\\{args.modi}-{self.REFIX}.pid"
        app_pid = str(os.getpid())
        with open(pid_file, "w", encoding="utf8") as f:
            f.write(app_pid)

        logger_info_str, self.logger, self.logging_filename = self.set_logger(args.debug, self.logger_prefix)

        self.print("Logger " + logger_info_str)
        self.print("================================")
        self.logger.info("Logger initialized")
        get_logger().info(Style.GREEN("Starting Application instance"))
        if args.init and args.init is not None and self.start_dir not in sys.path:
            sys.path.append(self.start_dir)

        __version__ = get_version_from_pyproject()
        self.version = __version__

        self.keys = {
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

        defaults = {
            "MACRO": ['Exit'],
            "MACRO_C": {},
            "HELPER": {},
            "debug": args.debug,
            "id": self.id,
            "st-load": False,
            "comm-his": [[]],
            "develop-mode": False,
        }
        self.config_fh = FileHandler(collective_identification + ".config", keys=self.keys, defaults=defaults)
        self.config_fh.load_file_handler()
        self._debug = args.debug
        self.flows = {}
        self.dev_modi = self.config_fh.get_file_handler(self.keys["develop-mode"])
        if self.config_fh.get_file_handler("provider::") is None:
            self.config_fh.add_to_save_file_handler("provider::", "http://localhost:" + str(
                self.args_sto.port) if os.environ.get("HOSTNAME","localhost") == "localhost" else "https://simplecore.app")
        self.functions = {}
        self.modules = {}

        self.interface_type = ToolBoxInterfaces.native
        self.PREFIX = Style.CYAN(f"~{node()}@>")
        self.alive = True
        self.called_exit = False, time.time()

        self.print(f"Infos:\n  {'Name':<8} -> {node()}\n  {'ID':<8} -> {self.id}\n  {'Version':<8} -> {self.version}\n  {'PID':<8} -> {os.getpid()}\n")

        self.logger.info(
            Style.GREEN(
                f"Finish init up in {time.perf_counter() - t0:.2f}s"
            )
        )

        self.args_sto = args
        self.loop = None

        from .system.session import Session
        self.session: Session = Session(self.get_username())
        self.logger.info(f"Session created for {self.session.username}")
        if len(sys.argv) > 2 and sys.argv[1] == "db":
            return
        from .extras.blobs import create_server_storage, create_desktop_storage, create_offline_storage
        # TODO detect db status and (auto start)
        self.root_blob_storage = create_server_storage() if os.getenv("IS_OFFLINE_DB", "false")!="true" else create_offline_storage()
        self.desktop_blob_storage = create_desktop_storage() if os.getenv("IS_OFFLINE_DB", "false")!="true" else create_offline_storage()
        self.mkdocs = add_to_app(self)
        # self._start_event_loop()

    def _start_event_loop(self):
        """Starts the asyncio event loop in a separate thread."""
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
            self.loop_thread.start()

    def get_username(self, get_input=False, default="loot") -> str:
        user_name = self.config_fh.get_file_handler("ac_user:::")
        if get_input and user_name is None:
            user_name = input("Input your username: ")
            self.config_fh.add_to_save_file_handler("ac_user:::", user_name)
        if user_name is None:
            user_name = default
            self.config_fh.add_to_save_file_handler("ac_user:::", user_name)
        return user_name

    def set_username(self, username):
        return self.config_fh.add_to_save_file_handler("ac_user:::", username)

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

    def set_logger(self, debug=False, logger_prefix=None):
        # remove existing logger
        try:
            logging.getLogger(loggerNameOfToolboxv2).handlers.clear()
        except Exception as e:
            print("No logger to clear or potetial doubel logging")
        if debug is None and os.getenv("TOOLBOX_LOGGING_LEVEL") is not None:
            debug = True
        if logger_prefix is None:
            logger_prefix = self.logger_prefix
        if "test" in self.logger_prefix and not debug:
            logger, logging_filename = setup_logging(logging.NOTSET, name="toolbox-test", interminal=True,
                                                     file_level=logging.NOTSET, app_name=logger_prefix)
            logger_info_str = "in Test Mode"
        elif "live" in self.logger_prefix and not debug:
            logger, logging_filename = setup_logging(logging.DEBUG, name="toolbox-live", interminal=False,
                                                     file_level=logging.WARNING, app_name=logger_prefix)
            logger_info_str = "in Live Mode"
            # setup_logging(logging.WARNING, name="toolbox-live", is_online=True
            #              , online_level=logging.WARNING).info("Logger initialized")
        elif "debug" in self.logger_prefix or self.logger_prefix.endswith("D"):
            self.logger_prefix = self.logger_prefix.replace("-debug", '').replace("debug", '')
            logger, logging_filename = setup_logging(logging.DEBUG, name="toolbox-debug", interminal=True,
                                                     file_level=logging.WARNING, app_name=logger_prefix)
            logger_info_str = "in debug Mode"
            self.debug = True
        elif debug:
            if hasattr(logging, "getLevelNamesMapping"):
                level = logging.getLevelNamesMapping().get(os.getenv("TOOLBOX_LOGGING_LEVEL", "WARNING"))
            else:
                level = logging.WARNING
            logger, logging_filename = setup_logging(
                level=level, name=f"toolbox-{self.logger_prefix}-debug",
                interminal=True,
                file_level=level, app_name=logger_prefix)
            logger_info_str = "in args debug Mode"
        else:
            logger, logging_filename = setup_logging(logging.ERROR, name=f"toolbox-{self.logger_prefix}", app_name=logger_prefix)
            logger_info_str = "in Default"

        return logger_info_str, logger, logging_filename

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        if not isinstance(value, bool):
            self.logger.debug(f"Value must be an boolean. is : {value} type of {type(value)}")
            raise ValueError("Value must be an boolean.")

        # self.logger.info(f"Setting debug {value}")
        self._debug = value

    def debug_rains(self, e):
        if self.debug:
            import traceback
            x = "="*5
            x += " DEBUG "
            x += "="*5
            self.print(x)
            self.print(traceback.format_exc())
            self.print(x)
            raise e
        else:
            self.logger.error(f"Error: {e}")
            import traceback
            x = "="*5
            x += " DEBUG "
            x += "="*5
            self.print(x)
            self.print(traceback.format_exc())
            self.print(x)

    def set_flows(self, r):
        self.flows = r

    async def run_flows(self, name, **kwargs):
        from ..flows import flows_dict as flows_dict_func
        if name not in self.flows:
            self.flows = {**self.flows, **flows_dict_func(s=name, remote=True)}
        if name in self.flows:
            if asyncio.iscoroutinefunction(self.flows[name]):
                return await self.flows[name](get_app(from_="runner"), self.args_sto, **kwargs)
            else:
                return self.flows[name](get_app(from_="runner"), self.args_sto, **kwargs)
        else:
            print("Flow not found, active flows:", len(self.flows.keys()))

    def _coppy_mod(self, content, new_mod_dir, mod_name, file_type='py'):

        mode = 'xb'
        self.logger.info(f" coppy mod {mod_name} to {new_mod_dir} size : {sys.getsizeof(content) / 8388608:.3f} mb")

        if not os.path.exists(new_mod_dir):
            os.makedirs(new_mod_dir)
            with open(f"{new_mod_dir}/__init__.py", "w") as nmd:
                nmd.write(f"__version__ = '{self.version}'")

        if os.path.exists(f"{new_mod_dir}/{mod_name}.{file_type}"):
            mode = False

            with open(f"{new_mod_dir}/{mod_name}.{file_type}", 'rb') as d:
                runtime_mod = d.read()  # Testing version but not efficient

            if len(content) != len(runtime_mod):
                mode = 'wb'

        if mode:
            with open(f"{new_mod_dir}/{mod_name}.{file_type}", mode) as f:
                f.write(content)

    def _pre_lib_mod(self, mod_name, path_to="./runtime", file_type='py'):
        working_dir = self.id.replace(".", "_")
        lib_mod_dir = f"toolboxv2.runtime.{working_dir}.mod_lib."

        self.logger.info(f"pre_lib_mod {mod_name} from {lib_mod_dir}")

        postfix = "_dev" if self.dev_modi else ""
        mod_file_dir = f"./mods{postfix}/{mod_name}.{file_type}"
        new_mod_dir = f"{path_to}/{working_dir}/mod_lib"
        with open(mod_file_dir, "rb") as c:
            content = c.read()
        self._coppy_mod(content, new_mod_dir, mod_name, file_type=file_type)
        return lib_mod_dir

    def _copy_load(self, mod_name, file_type='py', **kwargs):
        loc = self._pre_lib_mod(mod_name, file_type)
        return self.inplace_load_instance(mod_name, loc=loc, **kwargs)

    def helper_install_pip_module(self, module_name):
        if 'main' in self.id:
            return
        self.print(f"Installing {module_name} GREEDY")
        os.system(f"{sys.executable} -m pip install {module_name}")

    def python_module_import_classifier(self, mod_name, error_message):

        if error_message.startswith("No module named 'toolboxv2.utils"):
            return Result.default_internal_error(f"404 {error_message.split('utils')[1]} not found")
        if error_message.startswith("No module named 'toolboxv2.mods"):
            if mod_name.startswith('.'):
                return
            return self.run_a_from_sync(self.a_run_any, ("CloudM", "install"), module_name=mod_name)
        if error_message.startswith("No module named '"):
            pip_requ = error_message.split("'")[1].replace("'", "").strip()
            # if 'y' in input(f"\t\t\tAuto install {pip_requ} Y/n").lower:
            return self.helper_install_pip_module(pip_requ)
            # return Result.default_internal_error(f"404 {pip_requ} not found")

    def inplace_load_instance(self, mod_name, loc="toolboxv2.mods.", spec='app', save=True, mfo=None):
        if self.dev_modi and loc == "toolboxv2.mods.":
            loc = "toolboxv2.mods_dev."
        if spec=='app' and self.mod_online(mod_name):
            self.logger.info(f"Reloading mod from : {loc + mod_name}")
            self.remove_mod(mod_name, spec=spec, delete=False)

        # Convert dotted module name to file path for existence check
        # e.g., "CloudM.AuthManager" -> "CloudM/AuthManager"
        mod_path = mod_name.replace('.', '/')

        if (os.path.exists(self.start_dir + '/mods/' + mod_path) or os.path.exists(
            self.start_dir + '/mods/' + mod_path + '.py')) and (
            os.path.isdir(self.start_dir + '/mods/' + mod_path) or os.path.isfile(
            self.start_dir + '/mods/' + mod_path + '.py')):
            try:
                if mfo is None:
                    modular_file_object = import_module(loc + mod_name)
                else:
                    modular_file_object = mfo
                self.modules[mod_name] = modular_file_object
            except ModuleNotFoundError as e:
                self.logger.error(Style.RED(f"module {loc + mod_name} not found is type sensitive {e}"))
                self.print(Style.RED(f"module {loc + mod_name} not found is type sensitive {e}"))
                if self.debug or self.args_sto.sysPrint:
                    self.python_module_import_classifier(mod_name, str(e))
                self.debug_rains(e)
                return None
        else:
            self.sprint(f"module {loc + mod_name} is not valid")
            return None
        if hasattr(modular_file_object, "Tools"):
            tools_class = modular_file_object.Tools
        else:
            if hasattr(modular_file_object, "name"):
                tools_class = modular_file_object
                modular_file_object = import_module(loc + mod_name)
            else:
                tools_class = None

        modular_id = None
        instance = modular_file_object
        app_instance_type = "file/application"

        if tools_class is None:
            modular_id = modular_file_object.Name if hasattr(modular_file_object, "Name") else mod_name

        if tools_class is None and modular_id is None:
            modular_id = str(modular_file_object.__name__)
            self.logger.warning(f"Unknown instance loaded {mod_name}")
            return modular_file_object

        if tools_class is not None:
            tools_class = self.save_initialized_module(tools_class, spec)
            modular_id = tools_class.name
            app_instance_type = "functions/class"
        else:
            instance.spec = spec
        # if private:
        #     self.functions[modular_id][f"{spec}_private"] = private

        if not save:
            return instance if tools_class is None else tools_class

        return self.save_instance(instance, modular_id, spec, app_instance_type, tools_class=tools_class)

    def save_instance(self, instance, modular_id, spec='app', instance_type="file/application", tools_class=None):

        if modular_id in self.functions and tools_class is None:
            if self.functions[modular_id].get(f"{spec}_instance", None) is None:
                self.functions[modular_id][f"{spec}_instance"] = instance
                self.functions[modular_id][f"{spec}_instance_type"] = instance_type
            else:
                self.print("Firest instance stays use new spec to get new instance")
                if modular_id in self.functions and self.functions[modular_id].get(f"{spec}_instance", None) is not None:
                    return self.functions[modular_id][f"{spec}_instance"]
                else:
                    raise ImportError(f"Module already known {modular_id} and not avalabel reload using other spec then {spec}")

        elif tools_class is not None:
            if modular_id not in self.functions:
                self.functions[modular_id] = {}
            self.functions[modular_id][f"{spec}_instance"] = tools_class
            self.functions[modular_id][f"{spec}_instance_type"] = instance_type

            try:
                if not hasattr(tools_class, 'tools'):
                    tools_class.tools = {"Version": tools_class.get_version, 'name': tools_class.name}
                for function_name in list(tools_class.tools.keys()):
                    t_function_name = function_name.lower()
                    if t_function_name != "all" and t_function_name != "name":
                        self.tb(function_name, mod_name=modular_id)(tools_class.tools.get(function_name))
                self.functions[modular_id][f"{spec}_instance_type"] += "/BC"
                if hasattr(tools_class, 'on_exit'):
                    if "on_exit" in self.functions[modular_id]:
                        self.functions[modular_id]["on_exit"].append(tools_class.on_exit)
                    else:
                        self.functions[modular_id]["on_exit"] = [tools_class.on_exit]
            except Exception as e:
                self.logger.error(f"Starting Module {modular_id} compatibility failed with : {e}")
                pass
        elif modular_id not in self.functions and tools_class is None:
            self.functions[modular_id] = {}
            self.functions[modular_id][f"{spec}_instance"] = instance
            self.functions[modular_id][f"{spec}_instance_type"] = instance_type

        else:
            raise ImportError(f"Modular {modular_id} is not a valid mod")
        on_start = self.functions[modular_id].get("on_start")
        if on_start is not None:
            i = 1
            for f in on_start:
                try:
                    f_, e = self.get_function((modular_id, f), state=True, specification=spec)
                    if e == 0:
                        self.logger.info(Style.GREY(f"Running On start {f} {i}/{len(on_start)}"))
                        if asyncio.iscoroutinefunction(f_):
                            self.print(f"Async on start is only in Tool claas supported for {modular_id}.{f}" if tools_class is None else f"initialization starting soon for {modular_id}.{f}")
                            self.run_bg_task_advanced(f_)
                        else:
                            o = f_()
                            if o is not None:
                                self.print(f"Function {modular_id} On start result: {o}")
                    else:
                        self.logger.warning(f"starting function not found {e}")
                except Exception as e:
                    self.logger.debug(Style.YELLOW(
                        Style.Bold(f"modular:{modular_id}.{f} on_start error {i}/{len(on_start)} -> {e}")))
                    self.debug_rains(e)
                finally:
                    i += 1
        return instance if tools_class is None else tools_class

    def save_initialized_module(self, tools_class, spec):
        tools_class.spec = spec
        live_tools_class = tools_class(app=self)
        return live_tools_class

    def mod_online(self, mod_name, installed=False):
        if installed and mod_name not in self.functions:
            self.save_load(mod_name)
        return mod_name in self.functions

    def _get_function(self,
                      name: Enum or None,
                      state: bool = True,
                      specification: str = "app",
                      metadata=False, as_str: tuple or None = None, r=0, **kwargs):

        if as_str is None and isinstance(name, Enum):
            modular_id = str(name.NAME.value)
            function_id = str(name.value)
        elif as_str is None and isinstance(name, list):
            modular_id, function_id = name[0], name[1]
        else:
            modular_id, function_id = as_str

        self.logger.info(f"getting function : {specification}.{modular_id}.{function_id}")

        if modular_id not in self.functions:
            if r == 0:
                self.save_load(modular_id, spec=specification)
                return self.get_function(name=(modular_id, function_id),
                                         state=state,
                                         specification=specification,
                                         metadata=metadata,
                                         r=1)
            self.logger.warning(f"function modular not found {modular_id} 404")
            return "404", 404

        if function_id not in self.functions[modular_id]:
            self.logger.warning(f"function data not found {modular_id}.{function_id} 404")
            return "404", 404

        function_data = self.functions[modular_id][function_id]

        if isinstance(function_data, list):
            print(f"functions {function_id} : {function_data}")
            function_data = self.functions[modular_id][function_data[kwargs.get('i', -1)]]
            print(f"functions {modular_id} : {function_data}")
        function = function_data.get("func")
        params = function_data.get("params")

        state_ = function_data.get("state")
        if state_ is not None and state != state_:
            state = state_

        if function is None:
            self.logger.warning("No function found")
            return "404", 404

        if params is None:
            self.logger.warning("No function (params) found")
            return "404", 301

        if metadata and not state:
            self.logger.info("returning metadata stateless")
            return (function_data, function), 0

        if not state:  # mens a stateless function
            self.logger.info("returning stateless function")
            return function, 0

        instance = self.functions[modular_id].get(f"{specification}_instance")

        # instance_type = self.functions[modular_id].get(f"{specification}_instance_type", "functions/class")

        if params[0] == 'app':
            instance = get_app(from_=f"fuction {specification}.{modular_id}.{function_id}")

        if instance is None and self.alive:
            self.inplace_load_instance(modular_id, spec=specification)
            instance = self.functions[modular_id].get(f"{specification}_instance")

        if instance is None:
            self.logger.warning("No live Instance found")
            return "404", 400

        # if instance_type.endswith("/BC"):  # for backwards compatibility  functions/class/BC old modules
        #     # returning as stateless
        #     # return "422", -1
        #     self.logger.info(
        #         f"returning stateless function, cant find tools class for state handling found {instance_type}")
        #     if metadata:
        #         self.logger.info(f"returning metadata stateless")
        #         return (function_data, function), 0
        #     return function, 0

        self.logger.info("wrapping in higher_order_function")

        self.logger.info(f"returned fuction {specification}.{modular_id}.{function_id}")
        higher_order_function = partial(function, instance)

        if metadata:
            self.logger.info("returning metadata stateful")
            return (function_data, higher_order_function), 0

        self.logger.info("returning stateful function")
        return higher_order_function, 0

    def save_exit(self):
        self.logger.info(f"save exiting saving data to {self.config_fh.file_handler_filename} states of {self.debug=}")
        self.config_fh.add_to_save_file_handler(self.keys["debug"], str(self.debug))

    def init_mod(self, mod_name, spec='app'):
        """
        Initializes a module in a thread-safe manner by submitting the
        asynchronous initialization to the running event loop.
        """
        if '.' in mod_name:
            mod_name = mod_name.split('.')[0]
        self.run_bg_task(self.a_init_mod, mod_name, spec)
        # loop = self.loop_gard()
        # if loop:
        #     # Create a future to get the result from the coroutine
        #     future: Future = asyncio.run_coroutine_threadsafe(
        #         self.a_init_mod(mod_name, spec), loop
        #     )
        #     # Block until the result is available
        #     return future.result()
        # else:
        #     raise ValueError("Event loop is not running")
        #     # return self.loop_gard().run_until_complete(self.a_init_mod(mod_name, spec))

    def run_bg_task(self, task: Callable, *args, **kwargs) -> asyncio.Task | None:
        """
        Runs a coroutine in the background without blocking the caller.

        This is the primary method for "fire-and-forget" async tasks. It schedules
        the coroutine to run on the application's main event loop.

        Args:
            task: The coroutine function to run.
            *args: Arguments to pass to the coroutine function.
            **kwargs: Keyword arguments to pass to the coroutine function.

        Returns:
            An asyncio.Task object representing the scheduled task, or None if
            the task could not be scheduled.
        """
        if not callable(task):
            self.logger.warning("Task passed to run_bg_task is not callable!")
            return None

        if not asyncio.iscoroutinefunction(task) and not asyncio.iscoroutine(task):
            self.logger.warning(f"Task '{getattr(task, '__name__', 'unknown')}' is not a coroutine. "
                                f"Use run_bg_task_advanced for synchronous functions.")
            # Fallback to advanced runner for convenience
            return self.run_bg_task_advanced(task, *args,  get_coro=True, **kwargs)

        try:
            loop = self.loop_gard()
            if not loop.is_running():
                # If the main loop isn't running, we can't create a task on it.
                # This scenario is handled by run_bg_task_advanced.
                self.logger.info("Main event loop not running. Delegating to advanced background runner.")
                return self.run_bg_task_advanced(task, *args, **kwargs)

            # Create the coroutine if it's a function
            coro = task(*args, **kwargs) if asyncio.iscoroutinefunction(task) else task

            # Create a task on the running event loop
            bg_task = loop.create_task(coro)

            # Add a callback to log exceptions from the background task
            def _log_exception(the_task: asyncio.Task):
                if not the_task.cancelled() and the_task.exception():
                    self.logger.error(f"Exception in background task '{the_task.get_name()}':",
                                      exc_info=the_task.exception())

            bg_task.add_done_callback(_log_exception)
            self.bg_tasks.append(bg_task)
            return bg_task

        except Exception as e:
            self.logger.error(f"Failed to schedule background task: {e}", exc_info=True)
            return None

    def run_bg_task_advanced(self, task: Callable, *args, get_coro=False, **kwargs) -> threading.Thread:
        """
        Runs a task in a separate, dedicated background thread with its own event loop.

        This is ideal for:
        1. Running an async task from a synchronous context.
        2. Launching a long-running, independent operation that should not
           interfere with the main application's event loop.

        Args:
            task: The function to run (can be sync or async).
            *args: Arguments for the task.
            **kwargs: Keyword arguments for the task.

        Returns:
            The threading.Thread object managing the background execution.
        """
        if not callable(task):
            self.logger.warning("Task for run_bg_task_advanced is not callable!")
            return None

        coro_0 = [None]
        def thread_target():
            # Each thread gets its own event loop.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Prepare the coroutine we need to run
                if asyncio.iscoroutinefunction(task):
                    coro = task(*args, **kwargs)
                elif asyncio.iscoroutine(task):
                    # It's already a coroutine object
                    coro = task
                else:
                    # It's a synchronous function, run it in an executor
                    # to avoid blocking the new event loop.
                    coro = loop.run_in_executor(None, lambda: task(*args, **kwargs))

                # Run the coroutine to completion
                coro_0[0] = coro
                result = loop.run_until_complete(coro)
                self.logger.debug(f"Advanced background task '{getattr(task, '__name__', 'unknown')}' completed.")
                if result is not None:
                    self.logger.debug(f"Task result: {str(result)[:100]}")

            except Exception as e:
                self.logger.error(f"Error in advanced background task '{getattr(task, '__name__', 'unknown')}':",
                                  exc_info=e)
            finally:
                # Cleanly shut down the event loop in this thread.
                try:
                    all_tasks = asyncio.all_tasks(loop=loop)
                    if all_tasks:
                        for t in all_tasks:
                            t.cancel()
                        loop.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)

        # Create, start, and return the thread.
        # It's a daemon thread so it won't prevent the main app from exiting.
        t = threading.Thread(target=thread_target, daemon=True, name=f"BGTask-{getattr(task, '__name__', 'unknown')}")
        self.bg_tasks.append(t)
        t.start()
        if get_coro:
            return coro_0[0]
        return t

    # Helper method to wait for background tasks to complete (optional)
    def wait_for_bg_tasks(self, timeout=None):
        """
        Wait for all background tasks to complete.

        Args:
            timeout: Maximum time to wait (in seconds) for all tasks to complete.
                     None means wait indefinitely.

        Returns:
            bool: True if all tasks completed, False if timeout occurred
        """
        active_tasks = [t for t in self.bg_tasks if t.is_alive()]

        for task in active_tasks:
            task.join(timeout=timeout)
            if task.is_alive():
                return False

        return True

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, *args, mod_function_name=None, request=None, running_function_coro=None, **kwargs):
        """
        Run a function with support for SSE streaming in both
        threaded and non-threaded contexts.
        """
        if mod_function_name is None:
            mod_function_name = args[0]
        if running_function_coro is None:
            mn, fn = mod_function_name
            if self.functions.get(mn, {}).get(fn, {}).get('request_as_kwarg', False):
                kwargs["request"] = RequestData.from_dict(request)
                if 'data' in kwargs and 'data' not in self.functions.get(mn, {}).get(fn, {}).get('params', []):
                    kwargs["request"].data = kwargs["request"].body = kwargs['data']
                    del kwargs['data']
                if 'form_data' in kwargs and 'form_data' not in self.functions.get(mn, {}).get(fn, {}).get('params',
                                                                                                           []):
                    kwargs["request"].form_data = kwargs["request"].body = kwargs['form_data']
                    del kwargs['form_data']
            else:
                params = self.functions.get(mn, {}).get(fn, {}).get('params', [])
                # auto pars data and form_data to kwargs by key
                do = False
                data = {}
                if 'data' in kwargs and 'data' not in params:
                    do = True
                    data = kwargs['data']
                    del kwargs['data']
                if 'form_data' in kwargs and 'form_data' not in params:
                    do = True
                    data = kwargs['form_data']
                    del kwargs['form_data']
                if do:
                    for k in params:
                        if k in data:
                            kwargs[k] = data[k]
                            del data[k]

            if 'spec' in kwargs and 'spec' not in self.functions.get(mn, {}).get(fn, {}).get('params',
                                                                                                   []):
                if "tb_run_with_specification" in kwargs:
                    kwargs.pop('spec')
                else:
                    kwargs['tb_run_with_specification'] = kwargs.pop('spec')

        # Create the coroutine
        coro = running_function_coro or self.a_run_any(*args,mod_function_name=mod_function_name, **kwargs)

        # Get or create an event loop
        try:
            loop = asyncio.get_event_loop()
            is_running = loop.is_running()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            is_running = False

        # If the loop is already running, run in a separate thread
        if is_running:
            # Create thread pool executor as needed
            if not hasattr(self.__class__, '_executor'):
                self.__class__._executor = ThreadPoolExecutor(max_workers=4)

            def run_in_new_thread():
                # Set up a new loop in this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)

                try:
                    # Run the coroutine
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            # Run in thread and get result
            thread_result = self.__class__._executor.submit(run_in_new_thread).result()

            # Handle streaming results from thread
            if isinstance(thread_result, dict) and thread_result.get("is_stream"):
                # Create a new SSE stream in the main thread
                async def stream_from_function():
                    # Re-run the function with direct async access
                    stream_result = await self.a_run_any(*args, **kwargs)

                    if (isinstance(stream_result, Result) and
                        getattr(stream_result.result, 'data_type', None) == "stream"):
                        # Get and forward data from the original generator
                        original_gen = stream_result.result.data.get("generator")
                        if inspect.isasyncgen(original_gen):
                            async for item in original_gen:
                                yield item

                # Return a new streaming Result
                return Result.stream(
                    stream_generator=stream_from_function(),
                    headers=thread_result.get("headers", {})
                )

            result = thread_result
        else:
            # Direct execution when loop is not running
            result = loop.run_until_complete(coro)

        # Process the final result
        if isinstance(result, Result):
            if 'debug' in self.id:
                result.print()
            if getattr(result.result, 'data_type', None) == "stream":
                return result
            return result.to_api_result().model_dump(mode='json')

        return result

    def loop_gard(self):
        if self.loop is None:
            self._start_event_loop()
            self.loop = asyncio.get_event_loop()
        if self.loop.is_closed():
            self.loop = asyncio.get_event_loop()
        return self.loop

    async def a_init_mod(self, mod_name, spec='app'):
        mod = self.save_load(mod_name, spec=spec)
        if hasattr(mod, "__initobj") and not mod.async_initialized:
            await mod
        return mod


    def load_mod(self, mod_name: str, mlm='I', **kwargs):
        from .. import __init__
        action_list_helper = ['I (inplace load dill on error python)',
                              # 'C (coppy py file to runtime dir)',
                              # 'S (save py file to dill)',
                              # 'CS (coppy and save py file)',
                              # 'D (development mode, inplace load py file)'
                              ]
        action_list = {"I": lambda: self.inplace_load_instance(mod_name, **kwargs),
                       "C": lambda: self._copy_load(mod_name, **kwargs)
                       }

        try:
            if mlm in action_list:

                return action_list.get(mlm)()
            else:
                self.logger.critical(
                    f"config mlm must be {' or '.join(action_list_helper)} is {mlm=}")
                raise ValueError(f"config mlm must be {' or '.join(action_list_helper)} is {mlm=}")
        except ValueError as e:
            self.logger.warning(Style.YELLOW(f"Error Loading Module '{mod_name}', with error :{e}"))
            self.debug_rains(e)
        except ImportError as e:
            self.logger.error(Style.YELLOW(f"Error Loading Module '{mod_name}', with error :{e}"))
            self.debug_rains(e)
        except Exception as e:
            self.logger.critical(Style.RED(f"Error Loading Module '{mod_name}', with critical error :{e}"))
            print(Style.RED(f"Error Loading Module '{mod_name}'"))
            self.debug_rains(e)

        return Result.default_internal_error(info="info's in logs.")

    async def load_external_mods(self):
        for mod_path in os.getenv("EXTERNAL_PATH_RUNNABLE", '').split(','):
            if mod_path:
                await self.load_all_mods_in_file(mod_path)

    async def load_all_mods_in_file(self, working_dir="mods"):
        print(f"LOADING ALL MODS FROM FOLDER : {working_dir}")
        t0 = time.perf_counter()
        # Get the list of all modules
        module_list = self.get_all_mods(working_dir)
        open_modules = self.functions.keys()
        start_len = len(open_modules)

        for om in open_modules:
            if om in module_list:
                module_list.remove(om)

        tasks: set[Task] = set()

        _ = {tasks.add(asyncio.create_task(asyncio.to_thread(self.save_load, mod, 'app'))) for mod in module_list}
        for t in asyncio.as_completed(tasks):
            try:
                result = await t
                if hasattr(result, 'Name'):
                    self.print('Opened :', result.Name)
                elif hasattr(result, 'name'):
                    if hasattr(result, 'async_initialized'):
                        if not result.async_initialized:
                            async def _():
                                try:
                                    if asyncio.iscoroutine(result):
                                        await result
                                    if hasattr(result, 'Name'):
                                        self.print('Opened :', result.Name)
                                    elif hasattr(result, 'name'):
                                        self.print('Opened :', result.name)
                                except Exception as e:
                                    self.debug_rains(e)
                                    if hasattr(result, 'Name'):
                                        self.print('Error opening :', result.Name)
                                    elif hasattr(result, 'name'):
                                        self.print('Error opening :', result.name)
                            asyncio.create_task(_())
                        else:
                            self.print('Opened :', result.name)
                else:
                    if result:
                        self.print('Opened :', result)
            except Exception as e:
                self.logger.error(Style.RED(f"An Error occurred while opening all modules error: {str(e)}"))
                self.debug_rains(e)
        opened = len(self.functions.keys()) - start_len

        self.logger.info(f"Opened {opened} modules in {time.perf_counter() - t0:.2f}s")
        return f"Opened {opened} modules in {time.perf_counter() - t0:.2f}s"

    def get_all_mods(self, working_dir="mods", path_to="./runtime", use_wd=True):
        self.logger.info(f"collating all mods in working directory {working_dir}")

        pr = "_dev" if self.dev_modi else ""
        if working_dir == "mods" and use_wd:
            working_dir = f"{self.start_dir}/mods{pr}"
        elif use_wd:
            pass
        else:
            w_dir = self.id.replace(".", "_")
            working_dir = f"{path_to}/{w_dir}/mod_lib{pr}/"
        res = os.listdir(working_dir)

        self.logger.info(f"found : {len(res)} files")

        def do_helper(_mod):
            if "mainTool" in _mod:
                return False
            # if not _mod.endswith(".py"):
            #     return False
            if _mod.startswith("__"):
                return False
            if _mod.startswith("."):
                return False
            return not _mod.startswith("test_")

        def r_endings(word: str):
            if word.endswith(".py"):
                return word[:-3]
            return word

        mods_list = list(map(r_endings, filter(do_helper, res)))

        self.logger.info(f"found : {len(mods_list)} Modules")
        return mods_list

    def remove_all_modules(self, delete=False):
        for mod in list(self.functions.keys()):
            self.logger.info(f"closing: {mod}")
            self.remove_mod(mod, delete=delete)

    def remove_mod(self, mod_name, spec='app', delete=True):
        if mod_name not in self.functions:
            self.logger.info(f"mod not active {mod_name}")
            return

        on_exit = self.functions[mod_name].get("on_exit")
        self.logger.info(f"closing: {on_exit}")
        def helper():
            if f"{spec}_instance" in self.functions[mod_name]:
                del self.functions[mod_name][f"{spec}_instance"]
            if f"{spec}_instance_type" in self.functions[mod_name]:
                del self.functions[mod_name][f"{spec}_instance_type"]

        if on_exit is None and self.functions[mod_name].get(f"{spec}_instance_type", "").endswith("/BC"):
            instance = self.functions[mod_name].get(f"{spec}_instance", None)
            if instance is not None and hasattr(instance, 'on_exit'):
                if asyncio.iscoroutinefunction(instance.on_exit):
                    self.exit_tasks.append(instance.on_exit)
                else:
                    instance.on_exit()

        if on_exit is None and delete:
            self.functions[mod_name] = {}
            del self.functions[mod_name]
            return
        if on_exit is None:
            helper()
            return

        i = 1

        for j, f in enumerate(on_exit):
            try:
                f_, e = self.get_function((mod_name, f), state=True, specification=spec, i=j)
                if e == 0:
                    self.logger.info(Style.GREY(f"Running On exit {f} {i}/{len(on_exit)}"))
                    if asyncio.iscoroutinefunction(f_):
                        self.exit_tasks.append(f_)
                        o = None
                    else:
                        o = f_()
                    if o is not None:
                        self.print(f"Function On Exit result: {o}")
                else:
                    self.logger.warning("closing function not found")
            except Exception as e:
                self.logger.debug(
                    Style.YELLOW(Style.Bold(f"modular:{mod_name}.{f} on_exit error {i}/{len(on_exit)} -> {e}")))

                self.debug_rains(e)
            finally:
                i += 1

        helper()

        if delete:
            self.functions[mod_name] = {}
            del self.functions[mod_name]

    async def a_remove_all_modules(self, delete=False):
        for mod in list(self.functions.keys()):
            self.logger.info(f"closing: {mod}")
            await self.a_remove_mod(mod, delete=delete)

    async def a_remove_mod(self, mod_name, spec='app', delete=True):
        if mod_name not in self.functions:
            self.logger.info(f"mod not active {mod_name}")
            return
        on_exit = self.functions[mod_name].get("on_exit")
        self.logger.info(f"closing: {on_exit}")
        def helper():
            if f"{spec}_instance" in self.functions[mod_name]:
                del self.functions[mod_name][f"{spec}_instance"]
            if f"{spec}_instance_type" in self.functions[mod_name]:
                del self.functions[mod_name][f"{spec}_instance_type"]

        if on_exit is None and self.functions[mod_name].get(f"{spec}_instance_type", "").endswith("/BC"):
            instance = self.functions[mod_name].get(f"{spec}_instance", None)
            if instance is not None and hasattr(instance, 'on_exit'):
                if asyncio.iscoroutinefunction(instance.on_exit):
                    await instance.on_exit()
                else:
                    instance.on_exit()

        if on_exit is None and delete:
            self.functions[mod_name] = {}
            del self.functions[mod_name]
            return
        if on_exit is None:
            helper()
            return

        i = 1
        for f in on_exit:
            try:
                e = 1
                if isinstance(f, str):
                    f_, e = self.get_function((mod_name, f), state=True, specification=spec)
                elif isinstance(f, Callable):
                    f_, e, f  = f, 0, f.__name__
                if e == 0:
                    self.logger.info(Style.GREY(f"Running On exit {f} {i}/{len(on_exit)}"))
                    if asyncio.iscoroutinefunction(f_):
                        o = await f_()
                    else:
                        o = f_()
                    if o is not None:
                        self.print(f"Function On Exit result: {o}")
                else:
                    self.logger.warning("closing function not found")
            except Exception as e:
                self.logger.debug(
                    Style.YELLOW(Style.Bold(f"modular:{mod_name}.{f} on_exit error {i}/{len(on_exit)} -> {e}")))
                self.debug_rains(e)
            finally:
                i += 1

        helper()

        if delete:
            self.functions[mod_name] = {}
            del self.functions[mod_name]

    def exit(self, remove_all=True):
        if not self.alive:
            return
        if self.args_sto.debug:
            self.hide_console()
        self.disconnect()
        if remove_all:
            self.remove_all_modules()
        self.logger.info("Exiting ToolBox interface")
        self.alive = False
        self.called_exit = True, time.time()
        self.save_exit()
        # if hasattr(self, 'root_blob_storage') and self.root_blob_storage:
        #     self.root_blob_storage.exit()
        try:
            self.config_fh.save_file_handler()
        except SystemExit:
            print("If u ar testing this is fine else ...")

        if hasattr(self, 'daemon_app'):
            import threading

            for thread in threading.enumerate()[::-1]:
                if thread.name == "MainThread":
                    continue
                try:
                    with Spinner(f"closing Thread {thread.name:^50}|", symbols="s", count_down=True,
                                 time_in_s=0.751 if not self.debug else 0.6):
                        thread.join(timeout=0.751 if not self.debug else 0.6)
                except TimeoutError as e:
                    self.logger.error(f"Timeout error on exit {thread.name} {str(e)}")
                    print(str(e), f"Timeout {thread.name}")
                except KeyboardInterrupt:
                    print("Unsave Exit")
                    break
        if hasattr(self, 'loop') and self.loop is not None:
            with Spinner("closing Event loop:", symbols="+"):
                self.loop.stop()

    async def a_exit(self):

        import inspect
        self.sprint(f"exit requested from: {inspect.stack()[1].filename}::{inspect.stack()[1].lineno} function: {inspect.stack()[1].function}")

        # Cleanup session before removing modules
        try:
            if hasattr(self, 'session') and self.session is not None:
                await self.session.cleanup()
        except Exception as e:
            self.logger.debug(f"Session cleanup error (ignored): {e}")

        await self.a_remove_all_modules(delete=True)
        results = await asyncio.gather(
            *[asyncio.create_task(f()) for f in self.exit_tasks if asyncio.iscoroutinefunction(f)])
        for result in results:
            self.print(f"Function On Exit result: {result}")
        self.exit(remove_all=False)

    def save_load(self, modname, spec='app'):
        self.logger.debug(f"Save load module {modname}")
        if not modname:
            self.logger.warning("no filename specified")
            return False
        try:
            return self.load_mod(modname, spec=spec)
        except ModuleNotFoundError as e:
            self.logger.error(Style.RED(f"Module {modname} not found"))
            self.debug_rains(e)

        return False

    def get_function(self, name: Enum or tuple, **kwargs):
        """
        Kwargs for _get_function
            metadata:: return the registered function dictionary
                stateless: (function_data, None), 0
                stateful: (function_data, higher_order_function), 0
            state::boolean
                specification::str default app
        """
        if isinstance(name, tuple):
            return self._get_function(None, as_str=name, **kwargs)
        else:
            return self._get_function(name, **kwargs)

    async def a_run_function(self, mod_function_name: Enum or tuple,
                             tb_run_function_with_state=True,
                             tb_run_with_specification='app',
                             args_=None,
                             kwargs_=None,
                             *args,
                             **kwargs) -> Result:

        if kwargs_ is not None and not kwargs:
            kwargs = kwargs_
        if args_ is not None and not args:
            args = args_
        if isinstance(mod_function_name, tuple):
            modular_name, function_name = mod_function_name
        elif isinstance(mod_function_name, list):
            modular_name, function_name = mod_function_name[0], mod_function_name[1]
        elif isinstance(mod_function_name, Enum):
            modular_name, function_name = mod_function_name.__class__.NAME.value, mod_function_name.value
        else:
            raise TypeError("Unknown function type")

        if tb_run_with_specification == 'ws_internal':
            modular_name = modular_name.split('/')[0]
            if not self.mod_online(modular_name, installed=True):
                self.get_mod(modular_name)
            handler_id, event_name = mod_function_name
            if handler_id in self.websocket_handlers and event_name in self.websocket_handlers[handler_id]:
                handler_func = self.websocket_handlers[handler_id][event_name]
                try:
                    # Führe den asynchronen Handler aus
                    res = None
                    if inspect.iscoroutinefunction(handler_func):
                        res = await handler_func(self, **kwargs)
                    else:
                        res = handler_func(self, **kwargs)  # Für synchrone Handler
                    if isinstance(res, Result) or isinstance(res, ApiResult):
                        return res
                    return Result.ok(info=f"WS handler '{event_name}' executed.", data=res)
                except Exception as e:
                    self.logger.error(f"Error in WebSocket handler '{handler_id}/{event_name}': {e}", exc_info=True)
                    return Result.default_internal_error(info=str(e))
            else:
                # Kein Handler registriert, aber das ist kein Fehler (z.B. on_connect ist optional)
                return Result.ok(info=f"No WS handler for '{event_name}'.")

        if not self.mod_online(modular_name, installed=True):
            self.get_mod(modular_name)

        function_data, error_code = self.get_function(mod_function_name, state=tb_run_function_with_state,
                                                      metadata=True, specification=tb_run_with_specification)
        self.logger.info(f"Received fuction : {mod_function_name}, with execode: {error_code}")
        if error_code == 404:
            mod = self.get_mod(modular_name)
            if hasattr(mod, "async_initialized") and not mod.async_initialized:
                await mod
            function_data, error_code = self.get_function(mod_function_name, state=tb_run_function_with_state,
                                                          metadata=True, specification=tb_run_with_specification)

        if error_code == 404:
            self.logger.warning(Style.RED("Function Not Found"))
            return (Result.default_user_error(interface=self.interface_type,
                                              exec_code=404,
                                              info="function not found function is not decorated").
                    set_origin(mod_function_name))

        if error_code == 300:
            return Result.default_internal_error(interface=self.interface_type,
                                                 info=f"module {modular_name}"
                                                      f" has no state (instance)").set_origin(mod_function_name)

        if error_code != 0:
            return Result.default_internal_error(interface=self.interface_type,
                                                 exec_code=error_code,
                                                 info=f"Internal error"
                                                      f" {modular_name}."
                                                      f"{function_name}").set_origin(mod_function_name)

        if not tb_run_function_with_state:
            function_data, _ = function_data
            function = function_data.get('func')
        else:
            function_data, function = function_data

        if not function:
            self.logger.warning(Style.RED(f"Function {function_name} not found"))
            return Result.default_internal_error(interface=self.interface_type,
                                                 exec_code=404,
                                                 info="function not found function").set_origin(mod_function_name)

        self.logger.info("Profiling function")
        t0 = time.perf_counter()
        if asyncio.iscoroutinefunction(function):
            return await self.a_fuction_runner(function, function_data, args, kwargs, t0)
        else:
            return self.fuction_runner(function, function_data, args, kwargs, t0)


    def run_function(self, mod_function_name: Enum or tuple,
                     tb_run_function_with_state=True,
                     tb_run_with_specification='app',
                     args_=None,
                     kwargs_=None,
                     *args,
                     **kwargs) -> Result:

        if kwargs_ is not None and not kwargs:
            kwargs = kwargs_
        if args_ is not None and not args:
            args = args_
        if isinstance(mod_function_name, tuple):
            modular_name, function_name = mod_function_name
        elif isinstance(mod_function_name, list):
            modular_name, function_name = mod_function_name[0], mod_function_name[1]
        elif isinstance(mod_function_name, Enum):
            modular_name, function_name = mod_function_name.__class__.NAME.value, mod_function_name.value
        else:
            raise TypeError("Unknown function type")

        if not self.mod_online(modular_name, installed=True):
            self.get_mod(modular_name)

        if tb_run_with_specification == 'ws_internal':
            handler_id, event_name = mod_function_name
            if handler_id in self.websocket_handlers and event_name in self.websocket_handlers[handler_id]:
                handler_func = self.websocket_handlers[handler_id][event_name]
                try:
                    # Führe den asynchronen Handler aus
                    res = None
                    if inspect.iscoroutinefunction(handler_func):
                        res = self.loop.run_until_complete(handler_func(self, **kwargs))
                    else:
                        res = handler_func(self, **kwargs)  # Für synchrone Handler
                    if isinstance(res, Result) or isinstance(res, ApiResult):
                        return res
                    return Result.ok(info=f"WS handler '{event_name}' executed.", data=res)
                except Exception as e:
                    self.logger.error(f"Error in WebSocket handler '{handler_id}/{event_name}': {e}", exc_info=True)
                    return Result.default_internal_error(info=str(e))
            else:
                # Kein Handler registriert, aber das ist kein Fehler (z.B. on_connect ist optional)
                return Result.ok(info=f"No WS handler for '{event_name}'.")

        function_data, error_code = self.get_function(mod_function_name, state=tb_run_function_with_state,
                                                      metadata=True, specification=tb_run_with_specification)
        self.logger.info(f"Received fuction : {mod_function_name}, with execode: {error_code}")
        if error_code == 1 or error_code == 3 or error_code == 400:
            self.get_mod(modular_name)
            function_data, error_code = self.get_function(mod_function_name, state=tb_run_function_with_state,
                                                          metadata=True, specification=tb_run_with_specification)

        if error_code == 2:
            self.logger.warning(Style.RED("Function Not Found"))
            return (Result.default_user_error(interface=self.interface_type,
                                              exec_code=404,
                                              info="function not found function is not decorated").
                    set_origin(mod_function_name))

        if error_code == -1:
            return Result.default_internal_error(interface=self.interface_type,
                                                 info=f"module {modular_name}"
                                                      f" has no state (instance)").set_origin(mod_function_name)

        if error_code != 0:
            return Result.default_internal_error(interface=self.interface_type,
                                                 exec_code=error_code,
                                                 info=f"Internal error"
                                                      f" {modular_name}."
                                                      f"{function_name}").set_origin(mod_function_name)

        if not tb_run_function_with_state:
            function_data, _ = function_data
            function = function_data.get('func')
        else:
            function_data, function = function_data

        if not function:
            self.logger.warning(Style.RED(f"Function {function_name} not found"))
            return Result.default_internal_error(interface=self.interface_type,
                                                 exec_code=404,
                                                 info="function not found function").set_origin(mod_function_name)

        self.logger.info("Profiling function")
        t0 = time.perf_counter()
        if asyncio.iscoroutinefunction(function):
            try:
                return asyncio.run(self.a_fuction_runner(function, function_data, args, kwargs, t0))
            except RuntimeError:
                try:
                    return self.loop.run_until_complete(self.a_fuction_runner(function, function_data, args, kwargs, t0))
                except RuntimeError:
                    pass
            raise ValueError(f"Fuction {function_name} is Async use a_run_any")
        else:
            return self.fuction_runner(function, function_data, args, kwargs, t0)

    def run_a_from_sync(self, function, *args, **kwargs):
        # Initialize self.loop if not already set.
        if self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = asyncio.new_event_loop()

        # If the loop is running, offload the coroutine to a new thread.
        if self.loop.is_running():
            result_future = Future()

            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(function(*args, **kwargs))
                    result_future.set_result(result)
                except Exception as e:
                    result_future.set_exception(e)
                finally:
                    new_loop.close()

            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join()  # Block until the thread completes.
            return result_future.result()
        else:
            # If the loop is not running, schedule and run the coroutine directly.
            future = self.loop.create_task(function(*args, **kwargs))
            return self.loop.run_until_complete(future)

    def fuction_runner(self, function, function_data: dict, args: list, kwargs: dict, t0=.0):

        parameters = function_data.get('params')
        modular_name = function_data.get('module_name')
        function_name = function_data.get('func_name')
        row = function_data.get('row')
        mod_function_name = f"{modular_name}.{function_name}"

        if_self_state = 1 if 'self' in parameters else 0

        try:
            if len(parameters) == 0:
                res = function()
            elif len(parameters) == len(args) + if_self_state:
                res = function(*args)
            elif len(parameters) == len(kwargs.keys()) + if_self_state:
                res = function(**kwargs)
            else:
                res = function(*args, **kwargs)
            self.logger.info(f"Execution done in {time.perf_counter()-t0:.4f}")
            if isinstance(res, Result):
                formatted_result = res
                if formatted_result.origin is None:
                    formatted_result.set_origin(mod_function_name)
            elif isinstance(res, ApiResult):
                formatted_result = res
                if formatted_result.origin is None:
                    formatted_result.as_result().set_origin(mod_function_name).to_api_result()
            elif row:
                formatted_result = res
            else:
                # Wrap the result in a Result object
                formatted_result = Result.ok(
                    interface=self.interface_type,
                    data_info="Auto generated result",
                    data=res,
                    info="Function executed successfully"
                ).set_origin(mod_function_name)
            if not row:
                self.logger.info(
                    f"Function Exec code: {formatted_result.info.exec_code} Info's: {formatted_result.info.help_text}")
            else:
                self.logger.info(
                    f"Function Exec data: {formatted_result}")
        except Exception as e:
            self.logger.error(
                Style.YELLOW(Style.Bold(
                    f"! Function ERROR: in {modular_name}.{function_name}")))
            # Wrap the exception in a Result object
            formatted_result = Result.default_internal_error(info=str(e)).set_origin(mod_function_name)
            # res = formatted_result
            self.logger.error(
                f"Function {modular_name}.{function_name}"
                f" executed wit an error {str(e)}, {type(e)}")
            self.debug_rains(e)
            self.print(f"! Function ERROR: in {modular_name}.{function_name} ")



        else:
            self.print_ok()

            self.logger.info(
                f"Function {modular_name}.{function_name}"
                f" executed successfully")

        return formatted_result

    async def a_fuction_runner(self, function, function_data: dict, args: list, kwargs: dict, t0=.0):

        parameters = function_data.get('params')
        modular_name = function_data.get('module_name')
        function_name = function_data.get('func_name')
        row = function_data.get('row')
        mod_function_name = f"{modular_name}.{function_name}"

        if_self_state = 1 if 'self' in parameters else 0

        try:
            if len(parameters) == 0:
                res = await function()
            elif len(parameters) == len(args) + if_self_state:
                res = await function(*args)
            elif len(parameters) == len(kwargs.keys()) + if_self_state:
                res = await function(**kwargs)
            else:
                res = await function(*args, **kwargs)
            self.logger.info(f"Execution done in {time.perf_counter()-t0:.4f}")
            if isinstance(res, Result):
                formatted_result = res
                if formatted_result.origin is None:
                    formatted_result.set_origin(mod_function_name)
            elif isinstance(res, ApiResult):
                formatted_result = res
                if formatted_result.origin is None:
                    formatted_result.as_result().set_origin(mod_function_name).to_api_result()
            elif row:
                formatted_result = res
            else:
                # Wrap the result in a Result object
                formatted_result = Result.ok(
                    interface=self.interface_type,
                    data_info="Auto generated result",
                    data=res,
                    info="Function executed successfully"
                ).set_origin(mod_function_name)
            if not row:
                self.logger.info(
                    f"Function Exec code: {formatted_result.info.exec_code} Info's: {formatted_result.info.help_text}")
            else:
                self.logger.info(
                    f"Function Exec data: {formatted_result}")
        except Exception as e:
            self.logger.error(
                Style.YELLOW(Style.Bold(
                    f"! Function ERROR: in {modular_name}.{function_name}")))
            # Wrap the exception in a Result object
            formatted_result = Result.default_internal_error(info=str(e)).set_origin(mod_function_name)
            # res = formatted_result
            self.logger.error(
                f"Function {modular_name}.{function_name}"
                f" executed wit an error {str(e)}, {type(e)}")
            self.debug_rains(e)

        else:
            self.print_ok()

            self.logger.info(
                f"Function {modular_name}.{function_name}"
                f" executed successfully")

        return formatted_result

    async def run_http(self, mod_function_name: Enum or str or tuple, function_name=None,
                       args_=None,
                       kwargs_=None, method="GET",
                       *args, **kwargs):
        if kwargs_ is not None and not kwargs:
            kwargs = kwargs_
        if args_ is not None and not args:
            args = args_

        modular_name = mod_function_name
        function_name = function_name

        if isinstance(mod_function_name, str) and isinstance(function_name, str):
            mod_function_name = (mod_function_name, function_name)

        if isinstance(mod_function_name, tuple):
            modular_name, function_name = mod_function_name
        elif isinstance(mod_function_name, list):
            modular_name, function_name = mod_function_name[0], mod_function_name[1]
        elif isinstance(mod_function_name, Enum):
            modular_name, function_name = mod_function_name.__class__.NAME.value, mod_function_name.value

        self.logger.info(f"getting function : {modular_name}.{function_name} from http {self.session.base}")
        r = await self.session.fetch(f"/api/{modular_name}/{function_name}{'?' + args_ if args_ is not None else ''}",
                                     data=kwargs, method=method)
        try:
            if not r:
                print("§ Session server Offline!", self.session.base)
                return Result.default_internal_error(info="Session fetch failed").as_dict()

            content_type = r.headers.get('Content-Type', '').lower()

            if 'application/json' in content_type:
                try:
                    return r.json()
                except Exception as e:
                    print(f"⚠ JSON decode error: {e}")
                    # Fallback to text if JSON decoding fails
                    text = r.text
            else:
                text = r.text

            if isinstance(text, Callable):
                if asyncio.iscoroutinefunction(text):
                    text = await text()
                else:
                    text = text()

            # Attempt YAML
            if 'yaml' in content_type or text.strip().startswith('---'):
                try:
                    import yaml
                    return yaml.safe_load(text)
                except Exception as e:
                    print(f"⚠ YAML decode error: {e}")

            # Attempt XML
            if 'xml' in content_type or text.strip().startswith('<?xml'):
                try:
                    import xmltodict
                    return xmltodict.parse(text)
                except Exception as e:
                    print(f"⚠ XML decode error: {e}")

            # Fallback: return plain text
            return Result.default_internal_error(data={'raw_text': text, 'content_type': content_type}).as_dict()

        except Exception as e:
            print("❌ Fatal error during API call:", e)
            self.debug_rains(e)
            return Result.default_internal_error(str(e)).as_dict()

    def run_local(self, *args, **kwargs):
        return self.run_any(*args, **kwargs)

    async def a_run_local(self, *args, **kwargs):
        return await self.a_run_any(*args, **kwargs)

    def run_any(self, mod_function_name: Enum or str or tuple, backwords_compability_variabel_string_holder=None,
                get_results=False, tb_run_function_with_state=True, tb_run_with_specification='app', args_=None,
                kwargs_=None,
                *args, **kwargs):

        # if self.debug:
        #     self.logger.info(f'Called from: {getouterframes(currentframe(), 2)}')

        if kwargs_ is not None and not kwargs:
            kwargs = kwargs_
        if args_ is not None and not args:
            args = args_

        if isinstance(mod_function_name, str) and backwords_compability_variabel_string_holder is None:
            backwords_compability_variabel_string_holder = mod_function_name.split('.')[-1]
            mod_function_name = mod_function_name.replace(f".{backwords_compability_variabel_string_holder}", "")

        if isinstance(mod_function_name, str) and isinstance(backwords_compability_variabel_string_holder, str):
            mod_function_name = (mod_function_name, backwords_compability_variabel_string_holder)

        res: Result = self.run_function(mod_function_name,
                                        tb_run_function_with_state=tb_run_function_with_state,
                                        tb_run_with_specification=tb_run_with_specification,
                                        args_=args, kwargs_=kwargs).as_result()
        if isinstance(res, ApiResult):
            res = res.as_result()

        if isinstance(res, Result) and res.bg_task is not None:
            self.run_bg_task(res.bg_task)

        if self.debug:
            res.log(show_data=False)

        if not get_results and isinstance(res, Result):
            return res.get()

        if get_results and not isinstance(res, Result):
            return Result.ok(data=res)

        return res

    async def a_run_any(self, mod_function_name: Enum or str or tuple,
                        backwords_compability_variabel_string_holder=None,
                        get_results=False, tb_run_function_with_state=True, tb_run_with_specification='app', args_=None,
                        kwargs_=None,
                        *args, **kwargs):

        # if self.debug:
        #     self.logger.info(f'Called from: {getouterframes(currentframe(), 2)}')

        if kwargs_ is not None and not kwargs:
            kwargs = kwargs_
        if args_ is not None and not args:
            args = args_

        if isinstance(mod_function_name, str) and backwords_compability_variabel_string_holder is None:
            backwords_compability_variabel_string_holder = mod_function_name.split('.')[-1]
            mod_function_name = mod_function_name.replace(f".{backwords_compability_variabel_string_holder}", "")

        if isinstance(mod_function_name, str) and isinstance(backwords_compability_variabel_string_holder, str):
            mod_function_name = (mod_function_name, backwords_compability_variabel_string_holder)

        res: Result = await self.a_run_function(mod_function_name,
                                                tb_run_function_with_state=tb_run_function_with_state,
                                                tb_run_with_specification=tb_run_with_specification,
                                                args_=args, kwargs_=kwargs)
        if isinstance(res, ApiResult):
            res = res.as_result()

        if isinstance(res, Result) and res.bg_task is not None:
            self.run_bg_task(res.bg_task)

        if self.debug:
            res.print()
            res.log(show_data=False) if isinstance(res, Result) else self.logger.debug(res)
        if not get_results and isinstance(res, Result):
            return res.get()

        if get_results and not isinstance(res, Result):
            return Result.ok(data=res)

        return res


    def web_context(self):
        if self._web_context is None:
            try:
                self._web_context = open("./dist/helper.html", encoding="utf-8").read()
            except Exception as e:
                self.logger.error(f"Could not load web context: {e}")
                self._web_context = "<div><h1>Web Context not found</h1></div>"
        return self._web_context

    def get_mod(self, name, spec='app') -> ModuleType or MainToolType:
        if spec != "app":
            self.print(f"Getting Module {name} spec: {spec}")
        if name not in self.functions:
            mod = self.save_load(name, spec=spec)
            if mod is False or (isinstance(mod, Result) and mod.is_error()):
                self.logger.warning(f"Could not find {name} in {list(self.functions.keys())}")
                raise ValueError(f"Could not find {name} in {list(self.functions.keys())} pleas install the module, or its posibly broken use --debug for infos")
        # private = self.functions[name].get(f"{spec}_private")
        # if private is not None:
        #     if private and spec != 'app':
        #         raise ValueError("Module is private")
        if name not in self.functions:
            self.logger.warning(f"Module '{name}' is not found")
            return None
        instance = self.functions[name].get(f"{spec}_instance")
        if instance is None:
            return self.load_mod(name, spec=spec)
        return self.functions[name].get(f"{spec}_instance")

    def print(self, text="", *args, **kwargs):
        # self.logger.info(f"Output : {text}")
        if 'live' in self.id:
            return

        flush = kwargs.pop('flush', True)
        if self.sprint(None):
            print(Style.CYAN(f"System${self.id}:"), end=" ", flush=flush)
        if 'color' in kwargs:
            text = Style.style_dic[kwargs.pop('color')] + text + Style.style_dic["END"]
        print(text, *args, **kwargs, flush=flush)

    def sprint(self, text="", show_system=True, *args, **kwargs):
        if text is None:
            return True
        if 'live' in self.id:
            return
        flush = kwargs.pop('flush', True)
        # self.logger.info(f"Output : {text}")
        if show_system:
            print(Style.CYAN(f"System${self.id}:"), end=" ", flush=flush)
        if isinstance(text, str) and kwargs == {} and text:
            stram_print(text + ' '.join(args))
            print()
        else:
            print(text, *args, **kwargs, flush=flush)

    # ----------------------------------------------------------------
    # Decorators for the toolbox

    def reload_mod(self, mod_name, spec='app', is_file=True, loc="toolboxv2.mods."):
        self.remove_mod(mod_name, delete=True)
        if mod_name not in self.modules:
            self.logger.warning(f"Module '{mod_name}' is not found")
            return
        if hasattr(self.modules[mod_name], 'reload_save') and self.modules[mod_name].reload_save:
            def reexecute_module_code(x):
                return x
        else:
            def reexecute_module_code(module_name):
                if isinstance(module_name, str):
                    module = import_module(module_name)
                else:
                    module = module_name
                # Get the source code of the module
                try:
                    source = inspect.getsource(module)
                except Exception:
                    # print(f"No source for {str(module_name).split('from')[0]}: {e}")
                    return module
                # Compile the source code
                try:
                    code = compile(source, module.__file__, 'exec')
                    # Execute the code in the module's namespace
                    exec(code, module.__dict__)
                except Exception:
                    # print(f"No source for {str(module_name).split('from')[0]}: {e}")
                    pass
                return module

        if not is_file:
            mods = self.get_all_mods("./mods/" + mod_name)
            def recursive_reload(package_name):
                package = import_module(package_name)

                # First, reload all submodules
                if hasattr(package, '__path__'):
                    for _finder, name, _ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                        try:
                            mod = import_module(name)
                            reexecute_module_code(mod)
                            reload(mod)
                        except Exception as e:
                            print(f"Error reloading module {name}: {e}")
                            break

                # Finally, reload the package itself
                reexecute_module_code(package)
                reload(package)

            for mod in mods:
                if mod.endswith(".txt") or mod.endswith(".yaml"):
                    continue
                try:
                    recursive_reload(loc + mod_name + '.' + mod)
                    self.print(f"Reloaded {mod_name}.{mod}")
                except ImportError:
                    self.print(f"Could not load {mod_name}.{mod}")
        reexecute_module_code(self.modules[mod_name])
        if mod_name in self.functions:
            if "on_exit" in self.functions[mod_name]:
                self.functions[mod_name]["on_exit"] = []
            if "on_start" in self.functions[mod_name]:
                self.functions[mod_name]["on_start"] = []
        self.inplace_load_instance(mod_name, spec=spec, mfo=reload(self.modules[mod_name]) if mod_name in self.modules else None)

    def watch_mod(self, mod_name, spec='app', loc="toolboxv2.mods.", use_thread=True, path_name=None, on_reload=None):
        if path_name is None:
            path_name = mod_name
        is_file = os.path.isfile(self.start_dir + '/mods/' + path_name + '.py')
        import watchfiles
        def helper():
            paths = f'mods/{path_name}' + ('.py' if is_file else '')
            self.logger.info(f'Watching Path: {paths}')
            try:
                for changes in watchfiles.watch(paths):
                    if not changes:
                        continue
                    self.reload_mod(mod_name, spec, is_file, loc)
                    if on_reload:
                        on_reload()
            except FileNotFoundError:
                self.logger.warning(f"Path {paths} not found")

        if not use_thread:
            helper()
        else:
            threading.Thread(target=helper, daemon=True).start()

    def _register_function(self, module_name, func_name, data):
        if module_name not in self.functions:
            self.functions[module_name] = {}
        if func_name in self.functions[module_name]:
            self.print(f"Overriding function {func_name} from {module_name}", end="\r")
            self.functions[module_name][func_name] = data
        else:
            self.functions[module_name][func_name] = data

    def _create_decorator(self, type_: str,
                          name: str = "",
                          mod_name: str = "",
                          level: int = -1,
                          restrict_in_virtual_mode: bool = False,
                          api: bool = False,
                          helper: str = "",
                          version: str or None = None,
                          initial: bool=False,
                          exit_f: bool=False,
                          test: bool=True,
                          samples:list[dict[str, Any]] | None=None,
                          state:bool | None=None,
                          pre_compute:Callable | None=None,
                          post_compute:Callable[[], Result] | None=None,
                          api_methods:list[str] | None=None,
                          memory_cache: bool=False,
                          file_cache: bool=False,
                          request_as_kwarg: bool=False,
                          row: bool=False,
                          memory_cache_max_size:int=100,
                          memory_cache_ttl:int=300,
                          websocket_handler: str | None = None,
                          websocket_context: bool=False,
                          ):

        if isinstance(type_, Enum):
            type_ = type_.value

        if memory_cache and file_cache:
            raise ValueError("Don't use both cash at the same time for the same fuction")

        use_cache = memory_cache or file_cache
        cache = {}
        if file_cache:
            cache = FileCache(folder=self.data_dir + f'\\cache\\{mod_name}\\',
                              filename=self.data_dir + f'\\cache\\{mod_name}\\{name}cache.db')
        if memory_cache:
            cache = MemoryCache(maxsize=memory_cache_max_size, ttl=memory_cache_ttl)

        version = self.version if version is None else self.version + ':' + version

        def _args_kwargs_helper(args_, kwargs_, parms, api=False):
            if websocket_context and "request" in kwargs_:
                # Prüfen ob es ein WebSocket-Request ist
                request_data = kwargs_.get("request", {})
                if isinstance(request_data, dict) and "websocket" in request_data:
                    # WebSocket-Kontext erstellen
                    ws_ctx = WebSocketContext.from_kwargs(kwargs_)
                    kwargs_["ws_context"] = ws_ctx
                    if "session" in parms and "session" not in kwargs_:
                        kwargs_["session"] = (
                            ws_ctx.user
                        )  # oder ws_ctx.session, je nach Implementierung

                    if "conn_id" in parms and "conn_id" not in kwargs_:
                        kwargs_["conn_id"] = ws_ctx.conn_id
                    # Wenn der Parameter erwartet wird, Request-Object erstellen
                    if "request" in parms or request_as_kwarg:
                        kwargs_["request"] = RequestData.from_dict(request_data)

            if request_as_kwarg and "request" in kwargs_:
                kwargs_["request"] = (
                    RequestData.from_dict(kwargs_["request"])
                    if isinstance(kwargs_["request"], dict)
                    else kwargs_["request"]
                )
                if "data" in kwargs_ and "data" not in parms:
                    kwargs_["request"].data = kwargs_["request"].body = kwargs_["data"]
                    del kwargs_["data"]
                if "form_data" in kwargs_ and "form_data" not in parms:
                    kwargs_["request"].form_data = kwargs_["request"].body = kwargs_[
                        "form_data"
                    ]
                    del kwargs_["form_data"]

            if not request_as_kwarg and "request" in kwargs_:
                del kwargs_["request"]

            if (
                api
                and "data" in kwargs_
                and "data" not in parms
            ):
                for k in kwargs_["data"]:
                    if k in parms:
                        kwargs_[k] = kwargs_["data"][k]
                del kwargs_["data"]

            if "app" not in parms and args_ and args_[0] is self and len(args_) == 1:
                args_ = ()

            args_ += (kwargs_.pop("args_"),) if "args_" in kwargs_ else ()
            args_ += (kwargs_.pop("args"),) if "args" in kwargs_ else ()
            return args_, kwargs_

        def a_additional_process(func):

            def args_kwargs_helper(args_, kwargs_):
                module_name = mod_name if mod_name else func.__module__.split('.')[-1]
                func_name = name if name else func.__name__
                parms = self.functions.get(module_name, {}).get(func_name, {}).get('params', [])
                return _args_kwargs_helper(args_, kwargs_, parms, api=self.functions.get(module_name, {}).get(func_name, {}).get('api', False))

            async def executor(*args, **kwargs):
                args, kwargs = args_kwargs_helper(args, kwargs)
                if pre_compute is not None:
                    args, kwargs = await pre_compute(*args, **kwargs)
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                if post_compute is not None:
                    result = await post_compute(result)
                if row:
                    return result
                if not isinstance(result, Result):
                    result = Result.ok(data=result)
                if result.origin is None:
                    result.set_origin((mod_name if mod_name else func.__module__.split('.')[-1]
                                       , name if name else func.__name__
                                       , type_))
                if result.result.data_to == ToolBoxInterfaces.native.name:
                    result.result.data_to = ToolBoxInterfaces.remote if api else ToolBoxInterfaces.native
                # Wenden Sie die to_api_result Methode auf das Ergebnis an, falls verfügbar
                if api and hasattr(result, 'to_api_result'):
                    return result.to_api_result()
                return result

            @wraps(func)
            async def wrapper(*args, **kwargs):

                if not use_cache:
                    return await executor(*args, **kwargs)

                try:
                    cache_key = (f"{mod_name if mod_name else func.__module__.split('.')[-1]}"
                                 f"-{func.__name__}-{str(args)},{str(kwargs.items())}")
                except ValueError:
                    cache_key = (f"{mod_name if mod_name else func.__module__.split('.')[-1]}"
                                 f"-{func.__name__}-{bytes(args)},{str(kwargs.items())}")

                result = cache.get(cache_key)
                if result is not None:
                    return result

                result = await executor(*args, **kwargs)

                cache.set(cache_key, result)

                return result

            return wrapper

        def additional_process(func):

            def args_kwargs_helper(args_, kwargs_):
                module_name = mod_name if mod_name else func.__module__.split('.')[-1]
                func_name = name if name else func.__name__
                parms = self.functions.get(module_name, {}).get(func_name, {}).get('params', [])
                return _args_kwargs_helper(args_, kwargs_, parms, api=self.functions.get(module_name, {}).get(func_name, {}).get('api', False))

            def executor(*args, **kwargs):

                args, kwargs = args_kwargs_helper(args, kwargs)

                if pre_compute is not None:
                    args, kwargs = pre_compute(*args, **kwargs)
                if asyncio.iscoroutinefunction(func):
                    result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                if post_compute is not None:
                    result = post_compute(result)
                if row:
                    return result
                if not isinstance(result, Result):
                    result = Result.ok(data=result)
                if result.origin is None:
                    result.set_origin((mod_name if mod_name else func.__module__.split('.')[-1]
                                       , name if name else func.__name__
                                       , type_))
                if result.result.data_to == ToolBoxInterfaces.native.name:
                    result.result.data_to = ToolBoxInterfaces.remote if api else ToolBoxInterfaces.native
                # Wenden Sie die to_api_result Methode auf das Ergebnis an, falls verfügbar
                if api and hasattr(result, 'to_api_result'):
                    return result.to_api_result()
                return result

            @wraps(func)
            def wrapper(*args, **kwargs):

                if not use_cache:
                    return executor(*args, **kwargs)

                try:
                    cache_key = (f"{mod_name if mod_name else func.__module__.split('.')[-1]}"
                                 f"-{func.__name__}-{str(args)},{str(kwargs.items())}")
                except ValueError:
                    cache_key = (f"{mod_name if mod_name else func.__module__.split('.')[-1]}"
                                 f"-{func.__name__}-{bytes(args)},{str(kwargs.items())}")

                result = cache.get(cache_key)
                if result is not None:
                    return result

                result = executor(*args, **kwargs)

                cache.set(cache_key, result)

                return result

            return wrapper

        def decorator(func):
            sig = signature(func)
            params = list(sig.parameters)
            module_name = mod_name if mod_name else func.__module__.split('.')[-1]
            func_name = name if name else func.__name__
            if func_name == 'on_start':
                func_name = 'on_startup'
            if func_name == 'on_exit':
                func_name = 'on_close'
            if api or pre_compute is not None or post_compute is not None or memory_cache or file_cache:
                if asyncio.iscoroutinefunction(func):
                    func = a_additional_process(func)
                else:
                    func = additional_process(func)
            if api and str(sig.return_annotation) == 'Result':
                raise ValueError(f"Fuction {module_name}.{func_name} registered as "
                                 f"Api fuction but uses {str(sig.return_annotation)}\n"
                                 f"Please change the sig from ..)-> Result to ..)-> ApiResult")
            data = {
                "type": type_,
                "module_name": module_name,
                "func_name": func_name,
                "level": level,
                "restrict_in_virtual_mode": restrict_in_virtual_mode,
                "func": func,
                "api": api,
                "helper": helper,
                "version": version,
                "initial": initial,
                "exit_f": exit_f,
                "api_methods": api_methods if api_methods is not None else ["AUTO"],
                "__module__": func.__module__,
                "signature": sig,
                "params": params,
                "row": row,
                "state": (
                    False if len(params) == 0 else params[0] in ["self", "state", "app"]
                )
                if state is None
                else state,
                "do_test": test,
                "samples": samples,
                "request_as_kwarg": request_as_kwarg,
                "websocket_context": websocket_context,
            }

            if websocket_handler:
                # Die dekorierte Funktion sollte ein Dict mit den Handlern zurückgeben
                try:
                    handler_config = func(self)  # Rufe die Funktion auf, um die Konfiguration zu erhalten
                    if not isinstance(handler_config, dict):
                        raise TypeError(
                            f"WebSocket handler function '{func.__name__}' must return a dictionary of handlers.")

                    # Handler-Identifikator, z.B. "ChatModule/room_chat"
                    handler_id = f"{module_name}/{websocket_handler}"
                    self.websocket_handlers[handler_id] = {}

                    for event_name, handler_func in handler_config.items():
                        if event_name in ["on_connect", "on_message", "on_disconnect"] and callable(handler_func):
                            if asyncio.iscoroutinefunction(handler_func):
                                handler_func = a_additional_process(handler_func)
                            else:
                                handler_func = additional_process(handler_func)
                            self.websocket_handlers[handler_id][event_name] = handler_func
                        else:
                            self.logger.warning(f"Invalid WebSocket handler event '{event_name}' in '{handler_id}'.")

                    self.logger.info(f"Registered WebSocket handlers for '{handler_id}'.")

                except Exception as e:
                    self.logger.error(f"Failed to register WebSocket handlers for '{func.__name__}': {e}",
                                      exc_info=True)
            else:
                self._register_function(module_name, func_name, data)

            if exit_f:
                if "on_exit" not in self.functions[module_name]:
                    self.functions[module_name]["on_exit"] = []
                self.functions[module_name]["on_exit"].append(func_name)
            if initial:
                if "on_start" not in self.functions[module_name]:
                    self.functions[module_name]["on_start"] = []
                self.functions[module_name]["on_start"].append(func_name)

            return func

        decorator.tb_init = True

        return decorator

    def export(self, *args, **kwargs):
        return self.tb(*args, **kwargs)

    def tb(self, name=None,
           mod_name: str = "",
           helper: str = "",
           version: str | None = None,
           test: bool = True,
           restrict_in_virtual_mode: bool = False,
           api: bool = False,
           initial: bool = False,
           exit_f: bool = False,
           test_only: bool = False,
           memory_cache: bool = False,
           file_cache: bool = False,
           request_as_kwarg: bool = False,
           row: bool = False,
           state: bool | None = None,
           level: int = -1,
           memory_cache_max_size: int = 100,
           memory_cache_ttl: int = 300,
           samples: list or dict or None = None,
           interface: ToolBoxInterfaces or None or str = None,
           pre_compute=None,
           post_compute=None,
           api_methods=None,
           websocket_handler: str | None = None,
           websocket_context: bool=False,
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
        api_methods (list[str], optional): default ["AUTO"] (GET if not params, POST if params) , GET, POST, PUT or DELETE.
        websocket_handler (str, optional): The name of the websocket handler to use.
        websocket_context (bool, optional): Flag to indicate if the function should receive the websocket context.

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
                                      request_as_kwarg=request_as_kwarg,
                                      row=row,
                                      api_methods=api_methods,
                                      memory_cache_max_size=memory_cache_max_size,
                                      memory_cache_ttl=memory_cache_ttl,
                                      websocket_handler=websocket_handler,
                                      websocket_context=websocket_context,
                                      )

    def save_autocompletion_dict(self):
        autocompletion_dict = {}
        for module_name, _module in self.functions.items():
            data = {}
            for function_name, function_data in self.functions[module_name].items():
                if not isinstance(function_data, dict):
                    continue
                data[function_name] = {arg: None for arg in
                                       function_data.get("params", [])}
                if len(data[function_name].keys()) == 0:
                    data[function_name] = None
            autocompletion_dict[module_name] = data if len(data.keys()) > 0 else None
        self.config_fh.add_to_save_file_handler("auto~~~~~~", str(autocompletion_dict))

    def get_autocompletion_dict(self):
        return self.config_fh.get_file_handler("auto~~~~~~")

    def save_registry_as_enums(self, directory: str, filename: str):
        # Ordner erstellen, falls nicht vorhanden
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Dateipfad vorbereiten
        filepath = os.path.join(directory, filename)

        # Enum-Klassen als Strings generieren
        enum_classes = [f'"""Automatic generated by ToolBox v = {self.version}"""'
                        f'\nfrom enum import Enum\nfrom dataclasses import dataclass'
                        f'\n\n\n']
        for module, functions in self.functions.items():
            if module.startswith("APP_INSTANCE"):
                continue
            class_name = module
            enum_members = "\n    ".join(
                [
                    f"{func_name.upper().replace('-', '')}"
                    f" = '{func_name}' "
                    f"# Input: ({fuction_data['params'] if isinstance(fuction_data, dict) else ''}),"
                    f" Output: {fuction_data['signature'].return_annotation if isinstance(fuction_data, dict) else 'None'}"
                    for func_name, fuction_data in functions.items()])
            enum_class = (f'@dataclass\nclass {class_name.upper().replace(".", "_").replace("-", "")}(Enum):'
                          f"\n    NAME = '{class_name}'\n    {enum_members}")
            enum_classes.append(enum_class)

        # Enums in die Datei schreiben
        data = "\n\n\n".join(enum_classes)
        if len(data) < 12:
            raise ValueError(
                "Invalid Enums Loosing content pleas delete it ur self in the (utils/system/all_functions_enums.py) or add mor new stuff :}")
        with open(filepath, 'w') as file:
            file.write(data)

        print(Style.Bold(Style.BLUE(f"Enums gespeichert in {filepath}")))


    # WS logic

    def _set_rust_ws_bridge(self, bridge_object: Any):
        """
        Diese Methode wird von Rust aufgerufen, um die Kommunikationsbrücke zu setzen.
        Sie darf NICHT manuell von Python aus aufgerufen werden.
        """
        self.print(f"Rust WebSocket bridge has been set for instance {self.id}.")
        self._rust_ws_bridge = bridge_object

    async def ws_send(self, conn_id: str, payload: dict):
        """
        Sendet eine Nachricht asynchron an eine einzelne WebSocket-Verbindung.

        Args:
            conn_id: Die eindeutige ID der Zielverbindung.
            payload: Ein Dictionary, das als JSON gesendet wird.
        """
        if self._rust_ws_bridge is None:
            self.logger.error("Cannot send WebSocket message: Rust bridge is not initialized.")
            return

        try:
            # Ruft die asynchrone Rust-Methode auf und wartet auf deren Abschluss
            await self._rust_ws_bridge.send_message(conn_id, json.dumps(payload))
        except Exception as e:
            self.logger.error(f"Failed to send WebSocket message to {conn_id}: {e}", exc_info=True)

    async def ws_broadcast(self, channel_id: str, payload: dict, source_conn_id: str = "python_broadcast"):
        """
        Sendet eine Nachricht asynchron an alle Clients in einem Kanal/Raum.

        Args:
            channel_id: Der Kanal, an den gesendet werden soll.
            payload: Ein Dictionary, das als JSON gesendet wird.
            source_conn_id (optional): Die ID der ursprünglichen Verbindung, um Echos zu vermeiden.
        """
        if self._rust_ws_bridge is None:
            self.logger.error("Cannot broadcast WebSocket message: Rust bridge is not initialized.")
            return

        try:
            # Ruft die asynchrone Rust-Broadcast-Methode auf
            await self._rust_ws_bridge.broadcast_message(channel_id, json.dumps(payload), source_conn_id)
        except Exception as e:
            self.logger.error(f"Failed to broadcast WebSocket message to channel {channel_id}: {e}", exc_info=True)



