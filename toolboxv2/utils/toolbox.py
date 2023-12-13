"""Main module."""
import concurrent.futures
import os
import sys
import time
from enum import Enum
from platform import node, system
from importlib import import_module
from inspect import signature

import requests

from toolboxv2.utils.file_handler import FileHandler
from toolboxv2.utils import Result, ToolBoxError, ToolBoxResult, ToolBoxInfo, Singleton, AppArgs, ApiOb, \
    ToolBoxInterfaces
from toolboxv2.utils.tb_logger import setup_logging, get_logger
from toolboxv2.utils.Style import Style
import toolboxv2

import logging
from dotenv import load_dotenv

load_dotenv()


class App(metaclass=Singleton):
    def __init__(self, prefix: str = "", args=AppArgs().default()):

        t0 = time.perf_counter()
        abspath = os.path.abspath(__file__)
        self.system_flag = system()  # Linux: Linux Mac: Darwin Windows: Windows
        if self.system_flag == "Darwin" or self.system_flag == "Linux":
            dname = os.path.dirname(abspath).replace("/utils", "")
        else:
            dname = os.path.dirname(abspath).replace("\\utils", "")
        os.chdir(dname)

        self.start_dir = dname

        self.prefix = prefix
        self.id = prefix + '-' + node()

        identification = self.id
        if args.mm:
            identification = "MainNode"

        self.data_dir = dname + '\\.data\\' + identification
        self.config_dir = dname + '\\.config\\' + identification

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)

        if not prefix:
            if not os.path.exists(f"./.data/last-app-prefix"):
                open(f"./.data/last-app-prefix", "a").close()
            with open(f"./.data/last-app-prefix", "r") as prefix_file:
                cont = prefix_file.read()
                if cont:
                    prefix = cont
        else:
            if not os.path.exists(f"./.data/last-app-prefix"):
                open(f"./.data/last-app-prefix", "a").close()
            with open(f"./.data/last-app-prefix", "w") as prefix_file:
                prefix_file.write(prefix)

        print(f"Starting ToolBox as {prefix} from : ", Style.Bold(Style.CYAN(f"{os.getcwd()}")))

        logger_info_str, self.logger, self.logging_filename = self.set_logger(args.debug)

        print("Logger " + logger_info_str)
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
            "module-load-mode": "load~mode:",
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
            "module-load-mode": 'I',
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
        self.mlm = self.config_fh.get_file_handler(self.keys["module-load-mode"])

        self.functions = {}

        self.interface_type = ToolBoxInterfaces.native
        self.PREFIX = Style.CYAN(f"~{node()}@>")
        self.MOD_LIST = {}
        self.alive = True

        self.print(
            f"SYSTEM :: {node()}\nID -> {self.id},\nVersion -> {self.version},\n"
            f"load_mode -> {'coppy' if self.mlm == 'C' else ('Inplace' if self.mlm == 'I' else 'pleas use I or C')}\n")

        if args.update:
            self.run_any("cloudM", "#update-core", [])

        if args.get_version:
            v = self.version
            if args.mod_version_name != "mainTool":
                v = self.run_any(args.mod_version_name, 'Version', [])
            self.print(f"Version {args.mod_version_name} : {v}")

        self.logger.info(
            Style.GREEN(
                f"Finish init up in t-{time.perf_counter() - t0}s"
            )
        )

        self.args_sto = args

    def set_logger(self, debug=False):
        logger_info_str = "is unknown"
        if "test" in self.prefix and not debug:
            logger, logging_filename = setup_logging(logging.NOTSET, name="toolbox-test", interminal=True,
                                                     file_level=logging.NOTSET)
            logger_info_str = "in Test Mode"
        elif "live" in self.prefix and not debug:
            logger, logging_filename = setup_logging(logging.DEBUG, name="toolbox-debug", interminal=True,
                                                     file_level=logging.WARNING)
            logger_info_str = "in Live Mode"
            # setup_logging(logging.WARNING, name="toolbox-live", is_online=True
            #              , online_level=logging.WARNING).info("Logger initialized")
        elif "debug" in self.prefix:
            self.prefix = self.prefix.replace("-debug", '').replace("debug", '')
            logger, logging_filename = setup_logging(logging.DEBUG, name="toolbox-debug", interminal=True,
                                                     file_level=logging.WARNING)
            logger_info_str = "in debug Mode"
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

        logger_info_str, self.logger, self.logging_filename = self.set_logger(value)

        print("Logger " + logger_info_str)
        self.logger.info(f"Setting debug {value}")
        self._debug = value

    def _coppy_mod(self, content, new_mod_dir, mod_name):

        mode = 'xb'
        self.logger.info(f" coppy mod {mod_name} to {new_mod_dir} size : {sys.getsizeof(content) / 8388608:.3f} mb")

        if not os.path.exists(new_mod_dir):
            os.makedirs(new_mod_dir)
            with open(f"{new_mod_dir}/__init__.py", "w") as nmd:
                nmd.write(f"__version__ = '{self.version}'")

        if os.path.exists(f"{new_mod_dir}/{mod_name}.py"):
            mode = False
            with open(f"{new_mod_dir}/{mod_name}.py", 'rb') as d:
                runtime_mod = d.read()  # Testing version but not efficient
            if len(content) != len(runtime_mod):
                mode = 'wb'

        if mode:
            with open(f"{new_mod_dir}/{mod_name}.py", mode) as f:
                f.write(content)

    def _pre_lib_mod(self, mod_name, path_to="./runtime"):
        working_dir = self.id.replace(".", "_")
        lib_mod_dir = f"toolboxv2.runtime.{working_dir}.mod_lib."

        self.logger.info(f"pre_lib_mod {mod_name} from {lib_mod_dir}")

        postfix = "_dev" if self.dev_modi else ""
        mod_file_dir = f"./mods{postfix}/{mod_name}.py"
        new_mod_dir = f"{path_to}/{working_dir}/mod_lib"
        with open(mod_file_dir, "rb") as c:
            content = c.read()
        self._coppy_mod(content, new_mod_dir, mod_name)
        return lib_mod_dir

    def _copy_load(self, mod_name):
        loc = self._pre_lib_mod(mod_name)
        return self.inplace_load(mod_name, loc=loc)

    def inplace_load(self, mod_name, loc="toolboxv2.mods."):
        if self.dev_modi and loc == "toolboxv2.mods.":
            loc = "toolboxv2.mods_dev."
        if self.mod_online(mod_name):
            self.logger.info(f"Reloading mod from : {loc + mod_name}")
            self.remove_mod(mod_name)

        modular_file_object = import_module(loc + mod_name)
        try:
            tools_class = getattr(modular_file_object, "Tools")
        except AttributeError:
            tools_class = None

        modular_id = None
        instance = modular_file_object
        app_instance_type = "file/application"
        try:
            private = getattr(modular_file_object, "private")
        except AttributeError:
            private = False

        if tools_class is None:
            modular_id = getattr(modular_file_object, "Name").lower()

        if tools_class is None and modular_id is None:
            self.logger.warning(f"Unknown instance loaded {mod_name}")
            return modular_file_object

        if tools_class is not None:
            live_tools_class = self.save_initialized_module(tools_class)
            modular_id = live_tools_class.name.lower()
            instance = live_tools_class
            app_instance_type = "functions/class"

        if modular_id in self.functions:
            self.functions[modular_id]["app_instance"] = instance
            self.functions[modular_id]["app_instance_type"] = app_instance_type

            on_start = self.functions[modular_id].get("on_start")

            if on_start is not None:
                i = 1
                for f in on_start:
                    try:
                        f_, e = self._get_function(None, as_str=(modular_id, f), state=True)
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

        else:
            self.functions[modular_id] = {}
            self.functions[modular_id]["app_instance"] = instance
            self.functions[modular_id]["app_instance_type"] = app_instance_type
            self.logger.warning(f"Starting Module {modular_id} without functions")

            ## Back compatibility

            try:
                for function_name in list(instance.tools.keys()):
                    if function_name != "all" and function_name != "name":
                        self.tb(function_name, mod_name=modular_id)(instance.tools.get(function_name))
                self.functions[modular_id]["app_instance_type"] += "/BC"
            except Exception:
                pass

        if private:
            self.functions[modular_id]["private"] = private

        return instance

    def save_initialized_module(self, tools_class):
        live_tools_class = tools_class(app=self)  # save_as_app_instance
        return live_tools_class

    def mod_online(self, mod_name):
        return mod_name.lower() in self.functions

    def _get_function(self,
                      name: Enum or None,
                      state: bool = True,
                      specification: str = "app",
                      metadata=False, as_str: tuple or None = None):

        if as_str is None:
            modular_id = str(name.__class__.__name__).lower()
            function_id = str(name.value)
        else:
            modular_id, function_id = as_str
            modular_id = modular_id.lower()

        self.logger.info(f"getting function : {specification}.{modular_id}.{function_id}")

        if modular_id not in list(map(lambda x: x.lower(), self.functions)):
            self.logger.warning(f"function modular not found {modular_id} 404")
            return "404", 100

        if function_id not in self.functions[modular_id]:
            self.logger.warning(f"function data not found {modular_id}.{function_id} 404")
            return "404", 200

        function_data = self.functions[modular_id][function_id]

        function = function_data.get("func")

        if function is None:
            self.logger.warning(f"No function found")
            return "404", 300

        if metadata and not state:
            self.logger.info(f"returning metadata stateless")
            return (function_data, None), 0

        if not state:  # mens a stateless function
            self.logger.info(f"returning stateless function")
            return function, 0

        instance = self.functions[modular_id].get(f"{specification}_instance")
        instance_type = self.functions[modular_id].get(f"{specification}_instance_type")

        if instance is None:
            self.logger.warning(f"No live Instance found")
            return "404", 400

        if instance_type != "functions/class":  # for backwards compatibility  functions/class/BC old modules
            # returning as stateless
            # return "422", -1
            self.logger.info(
                f"returning stateless function, cant find tools class for state handling found {instance_type}")
            return function, 0

        self.logger.info(f"wrapping in higher_order_function")

        def higher_order_function(*args, **kwargs):
            self.logger.info(f"{specification}.{modular_id}.{function_id} got execute with '{specification}' state ")
            return function(self=instance, *args, **kwargs)

        if metadata:
            self.logger.info(f"returning metadata stateful")
            return (function_data, higher_order_function), 0

        self.logger.info(f"returning stateful function")
        return higher_order_function, 0

    def save_exit(self):
        self.logger.info(f"save exiting saving data to {self.config_fh.file_handler_filename} states of {self.debug=}"
                         f"{self.mlm=}")
        self.config_fh.add_to_save_file_handler(self.keys["debug"], str(self.debug))
        self.config_fh.add_to_save_file_handler(self.keys["module-load-mode"], self.mlm)

    def load_mod(self, mod_name):

        self.logger.info(f"try opening module {mod_name} in mode {self.mlm}")
        if self.debug:
            if self.mlm == "I":
                return self.inplace_load(mod_name)
            elif self.mlm == "C":
                return self._copy_load(mod_name)
        try:
            if self.mlm == "I":
                return self.inplace_load(mod_name)
            elif self.mlm == "C":
                return self._copy_load(mod_name)
            else:
                self.logger.critical(
                    f"config mlm must bee I (inplace load) or C (coppy to runtime load) is {self.mlm=}")
                raise ValueError(f"config mlm must bee I (inplace load) or C (coppy to runtime load) is {self.mlm=}")
        except ImportError as e:
            self.logger.error(Style.YELLOW(f"Error Loading Module '{mod_name}', with error :{e}"))
        except Exception as e:
            self.logger.critical(Style.RED(f"Error Loading Module '{mod_name}', with critical error :{e}"))

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

            for future in concurrent.futures.as_completed(futures):
                opened += 1

        self.save_registry_as_enums("utils", "all_functions_enums.py")
        self.logger.info(f"Opened {opened} modules in {time.perf_counter() - t0:.2f}s")
        return True

    def get_all_mods(self, working_dir="mods", path_to="./runtime"):
        self.logger.info(f"collating all mods in working directory {working_dir}")

        w_dir = self.id.replace(".", "_")

        if self.mlm == "C":
            if os.path.exists(f"{path_to}/{w_dir}/mod_lib"):
                working_dir = f"{path_to}/{w_dir}/mod_lib/"
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

    def remove_all_modules(self):
        for mod in self.functions:
            self.logger.info(f"closing: {mod}")
            self.remove_mod(mod, delete=False)
        del self.functions
        self.functions = {}

    def print_ok(self):
        self.logger.info("OK")

    def remove_mod(self, mod_name, delete=True):
        if mod_name not in self.functions:
            self.logger.info(f"mod not active {mod_name}")
        on_exit = self.functions[mod_name].get("on_exit")
        if on_exit is None and delete:
            self.functions[mod_name] = {}
            del self.functions[mod_name]
            return
        if on_exit is None:
            return
        i = 1
        for f in on_exit:
            try:
                f_, e = self._get_function(None, as_str=(mod_name, f), state=True)
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

    def exit(self):
        self.remove_all_modules()
        self.logger.info("Exiting ToolBox")
        self.print(Style.Bold(Style.CYAN("EXIT See U")))
        self.print('\033', end="")
        self.alive = False
        self.config_fh.save_file_handler()

    def save_load(self, modname):
        self.logger.debug(f"Save load module {modname}")
        if not modname:
            self.logger.warning("no filename specified")
            return False
        avalabel_mods = self.get_all_mods()
        i = 0
        fw = modname.lower()
        for mod in list(map(lambda x: x.lower(), avalabel_mods)):
            if fw == mod:
                modname = avalabel_mods[i]
            i += 1
        if self.debug:
            return self.load_mod(modname)
        try:
            return self.load_mod(modname)
        except ModuleNotFoundError:
            self.logger.error(Style.RED(f"Module {modname} not found"))

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

    def run_function(self, mod_function_name: Enum or tuple, tb_run_function_with_state=True, *args,
                     **kwargs) -> Result:

        if isinstance(mod_function_name, tuple):
            modular_name, function_name = mod_function_name
        else:
            modular_name, function_name = mod_function_name.__class__.__name__.lower(), mod_function_name.value

        function, error_code = self.get_function(mod_function_name, state=tb_run_function_with_state)

        if error_code == 1 or error_code == 3:
            self.get_mod(modular_name)
            function, error_code = self.get_function(mod_function_name, state=tb_run_function_with_state)

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

        if not function:
            self.logger.warning(Style.RED(f"Function {function_name} not found"))
            return Result.default_internal_error(interface=self.interface_type,
                                                 exec_code=404,
                                                 info=f"function not found function").set_origin(mod_function_name)

        self.logger.info(f"Profiling function")
        sig = signature(function)
        self.logger.debug(f"Signature: {sig}")
        parameters = list(sig.parameters)

        # mod_name = self.AC_MOD.name
        # self.print(f"\nStart function {mod_name}:{mod_function_name}\n")
        app_position = None
        for i, param in enumerate(parameters):
            if param == 'app':
                app_position = i

        if app_position is not None:
            args = list(args)
            args.insert(app_position, self)

        if self.debug:
            if len(parameters) == 0:
                res = function()
            elif len(parameters) == 1:
                res = function(*args)
            else:
                res = function(*args, **kwargs)

            if isinstance(res, Result):
                formatted_result = res
                if formatted_result.origin is not None:
                    formatted_result.set_origin(mod_function_name)
            else:
                # Wrap the result in a Result object
                formatted_result = Result.ok(
                    interface=self.interface_type,
                    data_info="Auto generated result",
                    data=res,
                    info="Function executed successfully"
                ).set_origin(mod_function_name)

            self.logger.info(f"Function Exec coed: {formatted_result.info.exec_code} Info's: {formatted_result.info.help_text}")
            return formatted_result
        try:
            if len(parameters) == 0:
                res = function()
            elif len(parameters) == 1:
                res = function(*args)
            else:
                res = function(*args, **kwargs)
            self.logger.info(f"Execution done")
            if isinstance(res, Result):
                formatted_result = res
                if function_name.origine is not None:
                    formatted_result.set_origin(mod_function_name)
            else:
                # Wrap the result in a Result object
                formatted_result = Result.ok(
                    interface=self.interface_type,
                    data_info="Auto generated result",
                    data=res,
                    info="Function executed successfully"
                ).set_origin(mod_function_name)

        except Exception as e:
            self.logger.error(
                Style.YELLOW(Style.Bold(
                    f"! Function ERROR: in {modular_name}.{function_name}")))
            # Wrap the exception in a Result object
            formatted_result = Result.default_internal_error(info=str(e)).set_origin(mod_function_name)
            # res = formatted_result
            self.logger.error(
                f"Function {modular_name}.{function_name}"
                f" executed wit an error {e}")

        else:
            self.print_ok()

            self.logger.info(
                f"Function {modular_name}.{function_name}"
                f" executed successfully")

        return formatted_result

    def run_any(self, mod_function_name: Enum or str, backwords_compability_variabel_string_holder=None,
                get_results=False, tb_run_function_with_state=True,
                *args, **kwargs):

        if isinstance(mod_function_name, str) and isinstance(backwords_compability_variabel_string_holder, str):
            mod_function_name = (mod_function_name, backwords_compability_variabel_string_holder)

        res: Result = self.run_function(mod_function_name,
                                        tb_run_function_with_state=tb_run_function_with_state,
                                        *args, **kwargs)

        if not get_results and isinstance(res, Result):
            return res.get()

        return res

    def get_mod(self, name):
        self.print(f"NAME: {name}")
        if name.lower() not in list(map(lambda x: x.lower(), self.functions.keys())):
            mod = self.save_load(name)
            if mod:
                return mod
            self.logger.warning(f"Could not find {name} in {list(self.functions.keys())}")
            raise ValueError(f"Could not find {name} in {list(self.functions.keys())} pleas install the module")
        private = self.functions[name.lower()].get("private")
        if private is not None:
            if private:
                raise ValueError("Module is not private")
        return self.functions[name.lower()].get("app_instance")

    def print(self, text, *args, **kwargs):
        # self.logger.info(f"Output : {text}")
        print(text, *args, **kwargs)

    # ----------------------------------------------------------------
    # Decorators for the toolbox

    def _register_function(self, module_name, func_name, data):
        if module_name not in self.functions:
            self.functions[module_name] = {}
        if module_name in self.functions and func_name in self.functions[module_name]:
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
                          exit_f=False):

        if isinstance(type_, Enum):
            type_ = type_.value

        version = self.version if version is None else self.version + ':' + version

        def decorator(func):
            sig = signature(func)
            params = list(sig.parameters)
            module_name = mod_name.lower() if mod_name else func.__module__.split('.')[-1].lower()
            func_name = name if name else func.__name__
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

        return decorator

    def tb(self, name=None,
           mod_name: str = "",
           level=-1,
           restrict_in_virtual_mode=False,
           helper="",
           api=False,
           version=None,
           initial=False,
           exit_f=False,
           interface=None):
        if interface is None:
            interface = "tb"
        return self._create_decorator(interface,
                                      name,
                                      mod_name,
                                      level=level,
                                      restrict_in_virtual_mode=restrict_in_virtual_mode,
                                      helper=helper,
                                      api=api,
                                      version=version,
                                      initial=initial,
                                      exit_f=exit_f)

    def print_functions(self):
        for module, functions in self.functions.items():
            print(f"\nModule: {module} type:{functions.get('app_instance_type')}")
            for func_name, data in functions.items():
                if not isinstance(data, dict):
                    continue
                print(
                    f"  Function: {func_name}{data['signature']}; Type: {data['type']}, Level: {'r' if data['level'] == -1 else data['level']}, {'Api' if data['api'] else ''}")
        if self.functions == {}:
            print("Noting to see")

    def save_registry_as_enums(self, directory, filename):
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
            class_name = module.split('.')[-1]  # Verwende den letzten Teil des Modulnamens als Klassenname
            enum_members = "\n\t".join(
                [f"{func_name.upper().replace('-', '')}: str = '{func_name}'" for func_name in functions])
            enum_class = (f'@dataclass\nclass {class_name.upper()}(Enum):'
                          f'\n\t{enum_members}')
            enum_classes.append(enum_class)

        # Enums in die Datei schreiben
        with open(filepath, 'w') as file:
            file.write("\n\n\n".join(enum_classes))

        print(Style.Bold(Style.BLUE(f"Enums gespeichert in {filepath}")))


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

        init_type = "main"
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
