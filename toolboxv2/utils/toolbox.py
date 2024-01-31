"""Main module."""
import concurrent.futures
import os
import sys
import time
import types
from enum import Enum
from platform import node, system
from importlib import import_module
from inspect import signature, getouterframes, currentframe
from types import ModuleType
from functools import partial, wraps
import requests
import shelve
from cachetools import TTLCache

from toolboxv2.utils.file_handler import FileHandler
from toolboxv2.utils import Singleton
from toolboxv2.utils.helper_functions import generate_test_cases
from toolboxv2.utils.types import Result, AppArgs, ToolBoxInterfaces, ApiResult
from toolboxv2.utils.tb_logger import setup_logging, get_logger
from toolboxv2.utils.Style import Style
import toolboxv2

import logging
from dotenv import load_dotenv
import dill

load_dotenv()


class FileCache:
    def __init__(self, folder='', filename='cache.db'):
        self.filename = filename
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    def get(self, key):
        with shelve.open(self.filename) as db:
            return db.get(key)

    def set(self, key, value):
        with shelve.open(self.filename, writeback=True) as db:
            db[key] = value


class MemoryCache:
    def __init__(self, maxsize=100, ttl=300):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value


def load_module_dill(filename):
    try:
        with open(filename, 'rb') as file:
            data = dill.load(file)
            return data
    except FileNotFoundError:
        return {}


def save_module_dill(data_to_serialize, filename):
    with open(filename, 'wb') as file:
        dill.dump(data_to_serialize, file)


class App(metaclass=Singleton):

    def __init__(self, prefix: str = "", args=AppArgs().default()):

        t0 = time.perf_counter()
        abspath = os.path.abspath(__file__)
        self.system_flag = system()  # Linux: Linux Mac: Darwin Windows: Windows
        if self.system_flag == "Darwin" or self.system_flag == "Linux":
            dir_name = os.path.dirname(abspath).replace("/utils", "")
        else:
            dir_name = os.path.dirname(abspath).replace("\\utils", "")
        os.chdir(dir_name)

        self.start_dir = dir_name

        lapp = dir_name + '\\.data\\'

        if not prefix:
            if not os.path.exists(f"{lapp}last-app-prefix.txt"):
                os.makedirs(lapp, exist_ok=True)
                open(f"{lapp}last-app-prefix.txt", "a").close()
            with open(f"{lapp}last-app-prefix.txt", "r") as prefix_file:
                cont = prefix_file.read()
                if cont:
                    prefix = cont
        else:
            if not os.path.exists(f"{lapp}last-app-prefix.txt"):
                os.makedirs(lapp, exist_ok=True)
                open(f"{lapp}last-app-prefix.txt", "a").close()
            with open(f"{lapp}last-app-prefix.txt", "w") as prefix_file:
                prefix_file.write(prefix)

        self.prefix = prefix
        self.id = prefix + '-' + node()

        identification = self.id
        if args.mm:
            identification = "MainNode"

        if "test" in prefix:
            self.start_dir += '\\tests'
            os.makedirs(self.start_dir, exist_ok=True)
            self.data_dir = self.start_dir + '\\.data\\' + "test"
            self.config_dir = self.start_dir + '\\.config\\' + "test"
        else:
            self.data_dir = self.start_dir + '\\.data\\' + identification
            self.config_dir = self.start_dir + '\\.config\\' + identification

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)

        print(f"Starting ToolBox as {prefix} from :", Style.Bold(Style.CYAN(f"{os.getcwd()}")))

        logger_info_str, self.logger, self.logging_filename = self.set_logger(args.debug)

        print("Logger " + logger_info_str)
        print("================================")
        self.logger.info("Logger initialized")
        get_logger().info(Style.GREEN("Starting Application instance"))
        if args.init and args.init is not None:
            if self.start_dir not in sys.path:
                sys.path.append(self.start_dir)
            _initialize_toolBox(args.init, args.init_file, self.id)

        self.version = toolboxv2.__version__

        self.keys = {
            "MACRO": "macro~~~~:",
            "MACRO_C": "m_color~~:",
            "HELPER": "helper~~~:",
            "debug": "debug~~~~:",
            "id": "name-spa~:",
            "st-load": "mute~load:",
            "comm-his": "comm-his~:",
            "develop-mode": "dev~mode~:",
            "all_main": "all~main~:",
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
            "all_main": True,
        }
        FileHandler.all_main = args.mm
        self.config_fh = FileHandler(self.id + ".config", keys=self.keys, defaults=defaults)
        self.config_fh.load_file_handler()
        self._debug = args.debug
        self.runnable = {}
        self.dev_modi = self.config_fh.get_file_handler(self.keys["develop-mode"])
        if self.config_fh.get_file_handler("provider::") is None:
            self.config_fh.add_to_save_file_handler("provider::", "https://simplecore.app")
        self.functions = {}

        self.interface_type = ToolBoxInterfaces.native
        self.PREFIX = Style.CYAN(f"~{node()}@>")
        self.MOD_LIST = {}
        self.alive = True
        self.print(
            f"SYSTEM :: {node()}\nID -> {self.id},\nVersion -> {self.version},\n")

        if args.update:
            self.run_any("CloudM", "update_core")

        if args.get_version:
            v = self.version
            if args.mod_version_name != "mainTool":
                v = self.run_any(args.mod_version_name, 'Version')
            self.print(f"Version {args.mod_version_name} : {v}")

        self.logger.info(
            Style.GREEN(
                f"Finish init up in t-{time.perf_counter() - t0}s"
            )
        )

        self.args_sto = args

    def set_logger(self, debug=False):
        if "test" in self.prefix and not debug:
            logger, logging_filename = setup_logging(logging.NOTSET, name="toolbox-test", interminal=True,
                                                     file_level=logging.NOTSET)
            logger_info_str = "in Test Mode"
        elif "live" in self.prefix and not debug:
            logger, logging_filename = setup_logging(logging.DEBUG, name="toolbox-live", interminal=False,
                                                     file_level=logging.WARNING)
            logger_info_str = "in Live Mode"
            # setup_logging(logging.WARNING, name="toolbox-live", is_online=True
            #              , online_level=logging.WARNING).info("Logger initialized")
        elif "debug" in self.prefix:
            self.prefix = self.prefix.replace("-debug", '').replace("debug", '')
            logger, logging_filename = setup_logging(logging.DEBUG, name="toolbox-debug", interminal=True,
                                                     file_level=logging.WARNING)
            logger_info_str = "in debug Mode"
            self.debug = True
        elif debug:
            logger, logging_filename = setup_logging(logging.DEBUG, name=f"toolbox-{self.prefix}-debug",
                                                     interminal=True,
                                                     file_level=logging.DEBUG)
            logger_info_str = "in args debug Mode"
        elif debug:
            logger, logging_filename = setup_logging(logging.DEBUG, name=f"toolbox-{self.prefix}-debug",
                                                     interminal=True,
                                                     file_level=logging.DEBUG)
            logger_info_str = "in args debug Mode"
        else:
            logger, logging_filename = setup_logging(logging.ERROR, name=f"toolbox-{self.prefix}")
            logger_info_str = "in Default"

        return logger_info_str, logger, logging_filename

    @property
    def debug(self):
        return self._debug

    def debug_rains(self, e):
        if self.debug:
            raise e

    def set_runnable(self, r):
        self.runnable = r

    def run_runnable(self, name, **kwargs):
        if name in self.runnable.keys():
            return self.runnable[name](self, self.args_sto, **kwargs)
        self.print("Runnable Not Available")

    @debug.setter
    def debug(self, value):
        if not isinstance(value, bool):
            self.logger.debug(f"Value must be an boolean. is : {value} type of {type(value)}")
            raise ValueError("Value must be an boolean.")

        # self.logger.info(f"Setting debug {value}")
        self._debug = value

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

    def inplace_load_instance(self, mod_name, loc="toolboxv2.mods.", spec='app', save=True):
        if self.dev_modi and loc == "toolboxv2.mods.":
            loc = "toolboxv2.mods_dev."
        if self.mod_online(mod_name):
            self.logger.info(f"Reloading mod from : {loc + mod_name}")
            self.remove_mod(mod_name, spec=spec, delete=False)

        try:
            modular_file_object = import_module(loc + mod_name)
        except ModuleNotFoundError as e:
            self.logger.error(Style.RED(f"module {loc + mod_name} not found is type sensitive {e}"))
            self.print(Style.RED(f"module {loc + mod_name} not found is type sensitive {e}"))
            return None
        try:
            tools_class = getattr(modular_file_object, "Tools")
        except AttributeError:
            tools_class = None

        modular_id = None
        instance = modular_file_object
        app_instance_type = "file/application"

        if tools_class is None:
            modular_id = getattr(modular_file_object, "Name")

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
                raise ImportError(f"Module already known {modular_id}")

            on_start = self.functions[modular_id].get("on_start")

            if on_start is not None:
                i = 1
                for f in on_start:
                    try:
                        f_, e = self.get_function((modular_id, f), state=True, specification=spec)
                        if e == 0:
                            self.logger.info(Style.GREY(f"Running On start {f} {i}/{len(on_start)}"))
                            o = f_()
                            if o is not None:
                                self.print(f"Function On start result: {o}")
                        else:
                            self.logger.warning(f"starting function not found {e}")
                    except Exception as e:
                        self.logger.debug(Style.YELLOW(
                            Style.Bold(f"modular:{modular_id}.{f} on_start error {i}/{len(on_start)} -> {e}")))
                    finally:
                        i += 1

        elif tools_class is not None:
            if modular_id not in self.functions:
                self.functions[modular_id] = {}
            self.functions[modular_id][f"{spec}_instance"] = tools_class
            self.functions[modular_id][f"{spec}_instance_type"] = instance_type

            try:
                for function_name in list(tools_class.tools.keys()):
                    if function_name != "all" and function_name != "name":
                        self.tb(function_name, mod_name=modular_id)(tools_class.tools.get(function_name))
                self.functions[modular_id][f"{spec}_instance_type"] += "/BC"
            except Exception as e:
                self.logger.error(f"Starting Module {modular_id} compatibility failed with : {e}")
                pass
        elif modular_id not in self.functions and tools_class is None:
            self.functions[modular_id] = {}
            self.functions[modular_id][f"{spec}_instance"] = instance
            self.functions[modular_id][f"{spec}_instance_type"] = instance_type

            def is_decorated(func: types.FunctionType):
                return getattr(func, 'tb_init', False)

            def is_tb_function(func: types.FunctionType):
                return isinstance(func.__annotations__.get('return'), Result)

            def is_tbapi_function(func: types.FunctionType):
                return isinstance(func.__annotations__.get('return'), ApiResult)

            for name in dir(instance):
                obj = getattr(instance, name)
                if isinstance(obj, types.FunctionType) and not is_decorated(func=obj):
                    obj.__init__()
                    obj.__init_subclass__()
                    # self.tb(name, mod_name=modular_id)(obj)

            # raise ImportError(f"Modular {modular_id} is not a valid mod 2")
        else:
            raise ImportError(f"Modular {modular_id} is not a valid mod")

        return instance if tools_class is None else tools_class

    def save_initialized_module(self, tools_class, spec):
        tools_class.spec = spec
        live_tools_class = tools_class(app=self)
        return live_tools_class

    def mod_online(self, mod_name):
        return mod_name in self.functions

    def _get_function(self,
                      name: Enum or None,
                      state: bool = True,
                      specification: str = "app",
                      metadata=False, as_str: tuple or None = None, r=0):

        if as_str is None:
            modular_id = str(name.NAME.value)
            function_id = str(name.value)
        else:
            modular_id, function_id = as_str

        self.logger.info(f"getting function : {specification}.{modular_id}.{function_id}")

        if modular_id not in self.functions.keys():
            if r == 0:
                self.save_load(modular_id)
                return self.get_function(name=(modular_id, function_id),
                                         state=state,
                                         specification=specification,
                                         metadata=metadata,
                                         r=1)
            self.logger.warning(f"function modular not found {modular_id} 404")
            return "404", 100

        if function_id not in self.functions[modular_id]:
            self.logger.warning(f"function data not found {modular_id}.{function_id} 404")
            return "404", 200

        function_data = self.functions[modular_id][function_id]

        function = function_data.get("func")
        params = function_data.get("params")

        state_ = function_data.get("state")
        if state_ is not None and state != state_:
            state = state_

        if function is None:
            self.logger.warning(f"No function found")
            return "404", 300

        if params is None:
            self.logger.warning(f"No function (params) found")
            return "404", 301

        if metadata and not state:
            self.logger.info(f"returning metadata stateless")
            return (function_data, function), 0

        if not state:  # mens a stateless function
            self.logger.info(f"returning stateless function")
            return function, 0

        instance = self.functions[modular_id].get(f"{specification}_instance")
        # instance_type = self.functions[modular_id].get(f"{specification}_instance_type", "functions/class")

        if params[0] == 'app':
            instance = self

        if instance is None:
            self.logger.warning(f"No live Instance found")
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

        self.logger.info(f"wrapping in higher_order_function")

        self.logger.info(f"returned fuction {specification}.{modular_id}.{function_id}")
        higher_order_function = partial(function, instance)

        if metadata:
            self.logger.info(f"returning metadata stateful")
            return (function_data, higher_order_function), 0

        self.logger.info(f"returning stateful function")
        return higher_order_function, 0

    def save_exit(self):
        self.logger.info(f"save exiting saving data to {self.config_fh.file_handler_filename} states of {self.debug=}")
        self.config_fh.add_to_save_file_handler(self.keys["debug"], str(self.debug))

    def load_mod(self, mod_name, mlm='I', **kwargs):

        self.logger.info(f"try opening module {mod_name}")
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

    def load_all_mods_in_file(self, working_dir="mods"):
        t0 = time.perf_counter()
        opened = 0
        # Get the list of all modules
        module_list = self.get_all_mods(working_dir)

        open_modules = self.functions.keys()

        for om in open_modules:
            if om in module_list:
                module_list.remove(om)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Load modules in parallel using threads
            futures = {executor.submit(self.load_mod, mod) for mod in module_list}

            for _ in concurrent.futures.as_completed(futures):
                opened += 1
        self.logger.info(f"Opened {opened} modules in {time.perf_counter() - t0:.2f}s")
        return True

    def get_all_mods(self, working_dir="mods", path_to="./runtime"):
        self.logger.info(f"collating all mods in working directory {working_dir}")

        w_dir = self.id.replace(".", "_")

        # if self.mlm == "C":
        #     if os.path.exists(f"{path_to}/{w_dir}/mod_lib"):
        #         working_dir = f"{path_to}/{w_dir}/mod_lib/"
        if working_dir == "mods":
            pr = "_dev" if self.dev_modi else ""
            working_dir = f"./mods{pr}"

        res = os.listdir(working_dir)

        self.logger.info(f"found : {len(res)} files")

        def do_helper(_mod):
            if "mainTool" in _mod:
                return False
            # if not _mod.endswith(".py"):
            #     return False
            if _mod.startswith("__"):
                return False
            if _mod.startswith("test_"):
                return False
            return True

        def r_endings(word: str):
            if word.endswith(".py"):
                return word[:-3]
            return word

        return list(map(r_endings, filter(do_helper, res)))

    def remove_all_modules(self, delete=False):
        for mod in list(self.functions.keys()):
            self.logger.info(f"closing: {mod}")
            self.remove_mod(mod, delete=delete)

    def print_ok(self):
        self.logger.info("OK")

    def remove_mod(self, mod_name, spec='app', delete=True):
        if mod_name not in self.functions:
            self.logger.info(f"mod not active {mod_name}")
            return
        on_exit = self.functions[mod_name].get("on_exit")

        def helper():
            if f"{spec}_instance" in self.functions[mod_name]:
                del self.functions[mod_name][f"{spec}_instance"]
            if f"{spec}_instance_type" in self.functions[mod_name]:
                del self.functions[mod_name][f"{spec}_instance_type"]

        if on_exit is None and self.functions[mod_name].get(f"{spec}_instance_type", "").endswith("/BC"):
            instanc = self.functions[mod_name].get(f"{spec}_instance", None)
            if instanc is not None:
                instanc._on_exit()

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
                f_, e = self.get_function((mod_name, f), state=True, specification=spec)
                if e == 0:
                    self.logger.info(Style.GREY(f"Running On exit {f} {i}/{len(on_exit)}"))
                    o = f_()
                    if o is not None:
                        self.print(f"Function On Exit result: {o}")
                else:
                    self.logger.warning("closing function not found")
            except Exception as e:
                self.logger.debug(
                    Style.YELLOW(Style.Bold(f"modular:{mod_name}.{f} on_exit error {i}/{len(on_exit)} -> {e}")))
            finally:
                i += 1

            helper()

            if delete:
                self.functions[mod_name] = {}
                del self.functions[mod_name]

    def exit(self):
        self.remove_all_modules()
        self.logger.info("Exiting ToolBox")
        self.print(Style.Bold(Style.CYAN("EXIT See U")))
        self.print('\033', end="")
        self.alive = False
        self.save_exit()
        self.config_fh.save_file_handler()

    def save_load(self, modname, spec='app'):
        self.logger.debug(f"Save load module {modname}")
        if not modname:
            self.logger.warning("no filename specified")
            return False
        avalabel_mods = self.get_all_mods()
        i = 0
        fw = modname
        for mod in avalabel_mods:
            if fw == mod:
                modname = avalabel_mods[i]
            i += 1
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
        else:
            modular_name, function_name = mod_function_name.__class__.NAME.value, mod_function_name.value

        function_data, error_code = self.get_function(mod_function_name, state=tb_run_function_with_state,
                                                      metadata=True, specification=tb_run_with_specification)
        self.logger.info(f"Received fuction : {mod_function_name}, with execode: {error_code}")
        if error_code == 1 or error_code == 3 or error_code == 400:
            self.get_mod(modular_name)
            function_data, error_code = self.get_function(mod_function_name, state=tb_run_function_with_state,
                                                          metadata=True, specification=tb_run_with_specification)

        if error_code == 2:
            self.logger.warning(Style.RED(f"Function Not Found"))
            return (Result.default_user_error(interface=self.interface_type,
                                              exec_code=404,
                                              info=f"function not found function is not decorated").
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
                                                 info=f"function not found function").set_origin(mod_function_name)

        self.logger.info(f"Profiling function")
        parameters = function_data.get('params')
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
            self.logger.info(f"Execution done")
            if isinstance(res, Result):
                formatted_result = res
                if formatted_result.origin is None:
                    formatted_result.set_origin(mod_function_name)
            else:
                # Wrap the result in a Result object
                formatted_result = Result.ok(
                    interface=self.interface_type,
                    data_info="Auto generated result",
                    data=res,
                    info="Function executed successfully"
                ).set_origin(mod_function_name)
            self.logger.info(
                f"Function Exec coed: {formatted_result.info.exec_code} Info's: {formatted_result.info.help_text}")
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

        if isinstance(mod_function_name, str) and isinstance(backwords_compability_variabel_string_holder, str):
            mod_function_name = (mod_function_name, backwords_compability_variabel_string_holder)

        res: Result = self.run_function(mod_function_name,
                                        tb_run_function_with_state=tb_run_function_with_state,
                                        tb_run_with_specification=tb_run_with_specification,
                                        args_=args, kwargs_=kwargs)

        if not get_results and isinstance(res, Result):
            return res.get()

        return res

    def get_mod(self, name, spec='app') -> ModuleType or toolboxv2.MainTool:
        if name not in self.functions.keys():
            if self.save_load(name, spec=spec) is False:
                self.logger.warning(f"Could not find {name} in {list(self.functions.keys())}")
                raise ValueError(f"Could not find {name} in {list(self.functions.keys())} pleas install the module")
        # private = self.functions[name].get(f"{spec}_private")
        # if private is not None:
        #     if private and spec != 'app':
        #         raise ValueError("Module is private")
        instance = self.functions[name].get(f"{spec}_instance")
        if instance is None:
            return self.load_mod(name, spec=spec)
        return self.functions[name].get(f"{spec}_instance")

    @staticmethod
    def print(text, *args, **kwargs):
        # self.logger.info(f"Output : {text}")
        print(text, *args, **kwargs)

    # ----------------------------------------------------------------
    # Decorators for the toolbox

    def _register_function(self, module_name, func_name, data):
        if module_name not in self.functions:
            self.functions[module_name] = {}
        if func_name in self.functions[module_name]:
            count = sum(1
                        for existing_key in self.functions[module_name] if
                        existing_key.startswith(module_name))
            new_key = f"{module_name}_{count}"
            self.functions[module_name][new_key] = data
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
                          initial=False,
                          exit_f=False,
                          test=True,
                          samples=None,
                          state=None,
                          pre_compute=None,
                          post_compute=None,
                          memory_cache=False,
                          file_cache=False,
                          memory_cache_max_size=100,
                          memory_cache_ttl=300):

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

        def additional_process(func):

            def executor(*args, **kwargs):

                if pre_compute is not None:
                    args, kwargs = pre_compute(*args, **kwargs)
                result = func(*args, **kwargs)
                if post_compute is not None:
                    result = post_compute(result)
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
            if api or pre_compute is not None or post_compute is not None or memory_cache or file_cache:
                func = additional_process(func)
            if api and 'Result' == str(sig.return_annotation):
                raise ValueError(f"Fuction {module_name}.{func_name} registered as "
                                 f"Api fuction but uses {str(sig.return_annotation)}\n"
                                 f"Please change the sig from ..)-> Result to ..)-> ApiResult")
            data = {
                "type": type_,
                "level": level,
                "restrict_in_virtual_mode": restrict_in_virtual_mode,
                "func": func,
                "api": api,
                "helper": helper,
                "version": version,
                "initial": initial,
                "exit_f": exit_f,
                "__module__": func.__module__,
                "signature": sig,
                "params": params,
                "state": (
                    False if len(params) == 0 else params[0] in ['self', 'state', 'app']) if state is None else state,
                "do_test": test,
                "samples": samples,

            }
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
           state: bool or None = None,
           level: int = -1,
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
        file_cache (bool, optional): Flag to enable file caching for the function.
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
                                      memory_cache_max_size=memory_cache_max_size,
                                      memory_cache_ttl=memory_cache_ttl)

    def print_functions(self):
        if not self.functions:
            print("Nothing to see")
            return

        for module, functions in self.functions.items():
            print(f"\nModule: {module}; Type: {functions.get('app_instance_type', 'Unknown')}")

            for func_name, data in functions.items():
                if not isinstance(data, dict):
                    continue

                func_type = data.get('type', 'Unknown')
                func_level = 'r' if data['level'] == -1 else data['level']
                api_status = 'Api' if data.get('api', False) else 'Non-Api'

                print(f"  Function: {func_name}{data.get('signature', '()')}; "
                      f"Type: {func_type}, Level: {func_level}, {api_status}")

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
                [f"{func_name.upper().replace('-', '')}: str = '{func_name}'" for func_name in functions])
            enum_class = (f'@dataclass\nclass {class_name.upper().replace(".", "_").replace("-", "")}(Enum):'
                          f"\n    NAME = '{class_name}'\n    {enum_members}")
            enum_classes.append(enum_class)

        # Enums in die Datei schreiben
        with open(filepath, 'w') as file:
            file.write("\n\n\n".join(enum_classes))

        print(Style.Bold(Style.BLUE(f"Enums gespeichert in {filepath}")))

    def execute_all_functions(self, m_query='', f_query=''):
        print("Executing all functions")
        all_data = {
            "modular_run": 0,
            "modular_fatal_error": 0,
            "errors": 0,
            "modular_sug": 0,
        }
        items = list(self.functions.items()).copy()
        for module_name, functions in items:
            infos = {
                "functions_run": 0,
                "functions_fatal_error": 0,
                "error": 0,
                "functions_sug": 0,
                'calls': {},
                'callse': {}
            }
            all_data['modular_run'] += 1
            if not module_name.startswith(m_query):
                all_data['modular_sug'] += 1
                continue
            f_items = list(functions.items()).copy()
            for function_name, function_data in f_items:
                if not isinstance(function_data, dict):
                    continue
                if not function_name.startswith(f_query):
                    continue
                test: list = function_data.get('do_test')
                print(test, module_name, function_name, function_data)
                if test is False:
                    continue
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
                print(module_name, function_name, test_kwargs_list)
                for test_kwargs in test_kwargs_list:
                    try:
                        print(f"test Running {state}.{module_name}.{function_name}")
                        result = self.run_function((module_name, function_name),
                                                   tb_run_function_with_state=state,
                                                   **test_kwargs)
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
        print(f"{all_data['modular_run']=}\n{all_data['modular_sug']=}\n{all_data['modular_fatal_error']=}")

        return Result.ok(data=all_data, data_info=analyze_data(all_data))


def analyze_data(data):
    report = []

    for mod_name, mod_info in data.items():
        if mod_name in ['modular_run', 'modular_fatal_error', 'modular_sug']:
            continue  # Überspringen der allgemeinen Statistiken
        if mod_name in ['errors']:
            report.append(f"Total errors: {mod_info}")
            continue
        report.append(f"Modul: {mod_name}")
        report.append(f"  Funktionen ausgeführt: {mod_info.get('functions_run', 0)}")
        report.append(f"  Funktionen mit Fatalen Fehler: {mod_info.get('functions_fatal_error', 0)}")
        report.append(f"  Funktionen mit Fehler: {mod_info.get('error', 0)}")
        report.append(f"  Funktionen erfolgreich: {mod_info.get('functions_sug', 0)}")

        if 'callse' in mod_info and mod_info['callse']:
            report.append("  Fehler:")
            for func_name, errors in mod_info['callse'].items():
                for error in errors:
                    if isinstance(error, str):
                        error = error.replace('\n', ' - ')
                    report.append(f"    - {func_name}, Fehler: {error}")

    return "\n".join(report)


def _initialize_toolBox(init_type, init_from, name):
    logger = get_logger()

    logger.info("Initialing ToolBox: " + init_type)
    if init_type.startswith("http"):
        logger.info("Download from url: " + init_from + "\n->temp_config.config")
        try:
            data = requests.get(init_from).json()["res"]
        except TimeoutError:
            logger.error(Style.RED("Error retrieving config information "))
            exit(1)
        # init_type = "main"
    else:
        data = open(init_from, 'r+').read()

    fh = FileHandler(name + ".config")
    fh.open_s_file_handler()
    fh.file_handler_storage.write(str(data))
    fh.file_handler_storage.close()

    logger.info("Done!")


def get_app(from_=None, name=None, args=AppArgs().default()) -> App:
    logger = get_logger()
    logger.info(Style.GREYBG(f"get app requested from: {from_}"))
    if name:
        app = App(name, args=args)
    else:
        app = App()
    logger.info(Style.Bold(f"App instance, returned ID: {app.id}"))
    return app
