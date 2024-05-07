"""Main module."""
import asyncio
import inspect
import os
import sys
import threading
import time
import types
from asyncio import Task
from enum import Enum
from platform import node, system
from importlib import import_module, reload
from inspect import signature
from types import ModuleType
from functools import partial, wraps

import requests
from yaml import safe_load

from .singelton_class import Singleton

from .system.cache import FileCache, MemoryCache
from .system.tb_logger import get_logger, setup_logging
from .system.types import AppArgs, ToolBoxInterfaces, ApiResult, Result, AppType, MainToolType
from .system.getting_and_closing_app import get_app
from .system.file_handler import FileHandler

from .extras.Style import Style, stram_print, Spinner

import logging
from dotenv import load_dotenv

load_dotenv()


class App(AppType, metaclass=Singleton):

    def __init__(self, prefix: str = "", args=AppArgs().default()):
        super().__init__(prefix, args)
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

        self.globals = {
            "root": {**globals()},
        }
        self.locals = {
            "user": {'app': self, **locals()},
        }

        identification = self.id

        if "test" in prefix:

            if self.system_flag == "Darwin" or self.system_flag == "Linux":
                start_dir = self.start_dir.replace("/toolboxv2", "")
            else:
                start_dir = self.start_dir.replace("\\toolboxv2", "")
            self.data_dir = start_dir + '\\.data\\' + "test"
            self.config_dir = start_dir + '\\.config\\' + "test"
            self.info_dir = start_dir + '\\.info\\' + "test"
        else:
            self.data_dir = self.start_dir + '\\.data\\' + identification
            self.config_dir = self.start_dir + '\\.config\\' + identification
            self.info_dir = self.start_dir + '\\.info\\' + identification

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
        if not os.path.exists(self.info_dir):
            os.makedirs(self.info_dir, exist_ok=True)

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

        with open(os.getenv('CONFIG_FILE', f'{dir_name}/toolbox.yaml'), 'r') as config_file:
            _version = safe_load(config_file)
            __version__ = _version.get('main', {}).get('version', '-.-.-')

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
        self.config_fh = FileHandler(self.id + ".config", keys=self.keys, defaults=defaults)
        self.config_fh.load_file_handler()
        self._debug = args.debug
        self.runnable = {}
        self.dev_modi = self.config_fh.get_file_handler(self.keys["develop-mode"])
        if self.config_fh.get_file_handler("provider::") is None:
            self.config_fh.add_to_save_file_handler("provider::", "http://localhost:" + str(
                self.args_sto.port) if "localhost" == os.environ.get("HOSTNAME",
                                                                     "localhost") else "https://simplecore.app")
        self.functions = {}
        self.modules = {}

        self.interface_type = ToolBoxInterfaces.native
        self.PREFIX = Style.CYAN(f"~{node()}@>")
        self.alive = True
        self.called_exit = False, time.time()

        self.print(f"Infos:\n  {'Name':<8} -> {node()}\n  {'ID':<8} -> {self.id}\n  {'Version':<8} -> {self.version}\n")

        if args.update:
            os.system("git pull")
            # self.save_load("CloudM")
            # self.run_any("CloudM", "update_core")

        if args.get_version:
            v = self.version
            self.print(f"Version main : {v}")

        self.logger.info(
            Style.GREEN(
                f"Finish init up in t-{time.perf_counter() - t0}s"
            )
        )

        self.args_sto = args

    def get_username(self, get_input=False):
        user_name = self.config_fh.get_file_handler("ac_user:::")
        if user_name is None and user_name != "None":
            user_name = input("Input your username\nbe sure to make no typos: ")
            self.config_fh.add_to_save_file_handler("ac_user:::", user_name)
        return user_name

    def reset_username(self):
        self.config_fh.add_to_save_file_handler("ac_user:::", "None")

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

    async def run_runnable(self, name, **kwargs):
        if name in self.runnable.keys():
            if inspect.iscoroutinefunction(self.runnable[name]):
                return await self.runnable[name](get_app(from_="runner"), self.args_sto, **kwargs)
            else:
                return self.runnable[name](get_app(from_="runner"), self.args_sto, **kwargs)
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

    def helper_install_pip_module(self, module_name):
        self.print(f"Installing {module_name} GREEDY")
        os.system(f"pip install {module_name}")

    def python_module_import_classifier(self, mod_name, error_message):

        if error_message.startswith("No module named 'toolboxv2.utils"):
            return Result.default_internal_error(f"404 {error_message.split('utils')[1]} not found")
        if error_message.startswith("No module named 'toolboxv2.mods"):
            # TODO: install from remote optional
            return Result.default_internal_error(f"404 {error_message.split('mods')[1]} not found")
        if error_message.startswith("No module named '"):
            # TODO: install from remote optional
            pip_requ = error_message.split("'")[1].replace("'", "").strip()
            # if 'y' in input(f"\t\t\tAuto install {pip_requ} Y/n").lower:
            return self.helper_install_pip_module(pip_requ)
            # return Result.default_internal_error(f"404 {pip_requ} not found")

    def inplace_load_instance(self, mod_name, loc="toolboxv2.mods.", spec='app', save=True, mfo=None):
        if self.dev_modi and loc == "toolboxv2.mods.":
            loc = "toolboxv2.mods_dev."
        if self.mod_online(mod_name):
            self.logger.info(f"Reloading mod from : {loc + mod_name}")
            self.remove_mod(mod_name, spec=spec, delete=False)

        if (os.path.exists(self.start_dir + '/mods/' + mod_name) or os.path.exists(
            self.start_dir + '/mods/' + mod_name + '.py')) and (
            os.path.isdir(self.start_dir + '/mods/' + mod_name) or os.path.isfile(
            self.start_dir + '/mods/' + mod_name + '.py')):
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
                return None
        else:
            self.print(f"module {loc + mod_name} is not valid")
            return None
        if hasattr(modular_file_object, "Tools"):
            tools_class = getattr(modular_file_object, "Tools")
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
                self.print("ERROR OVERRIDE")
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
                if not hasattr(tools_class, 'tools'):
                    tools_class.tools = {"Version": tools_class.get_version, 'name': tools_class.name}
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

        else:
            raise ImportError(f"Modular {modular_id} is not a valid mod")

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
                      metadata=False, as_str: tuple or None = None, r=0):

        if as_str is None and isinstance(name, Enum):
            modular_id = str(name.NAME.value)
            function_id = str(name.value)
        elif as_str is None and isinstance(name, list):
            modular_id, function_id = name[0], name[1]
        else:
            modular_id, function_id = as_str

        self.logger.info(f"getting function : {specification}.{modular_id}.{function_id}")

        if modular_id not in self.functions.keys():
            if r == 0:
                self.save_load(modular_id, spec=specification)
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
            instance = get_app(from_=f"fuction {specification}.{modular_id}.{function_id}")

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

    def load_mod(self, mod_name: str, mlm='I', **kwargs):

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

    async def load_all_mods_in_file(self, working_dir="mods"):
        t0 = time.perf_counter()
        # Get the list of all modules
        module_list = self.get_all_mods(working_dir)
        open_modules = self.functions.keys()
        start_len = len(open_modules)

        for om in open_modules:
            if om in module_list:
                module_list.remove(om)

        tasks: set[Task] = set()

        if 'isaa' in module_list:
            threading.Thread(target=self.save_load, args=("isaa", 'app'), daemon=True).start()
            module_list.remove('isaa')

        _ = {tasks.add(asyncio.create_task(asyncio.to_thread(self.save_load, mod, 'app'))) for mod in module_list}
        for t in asyncio.as_completed(tasks):
            try:
                result = await t
                if hasattr(result, 'Name'):
                    print('Opened :', result.Name)
                elif hasattr(result, 'name'):
                    print('Opened :', result.name)
                else:
                    print('Opened :', result)
            except Exception as e:
                self.logger.error(Style.RED(f"An Error occurred while opening all modules error: {str(e)}"))
        opened = len(self.functions.keys()) - start_len

        self.logger.info(f"Opened {opened} modules in {time.perf_counter() - t0:.2f}s")
        return f"Opened {opened} modules in {time.perf_counter() - t0:.2f}s"

    def get_all_mods(self, working_dir="mods", path_to="./runtime", use_wd=True):
        self.logger.info(f"collating all mods in working directory {working_dir}")

        pr = "_dev" if self.dev_modi else ""
        if working_dir == "mods" and use_wd:
            working_dir = f"./mods{pr}"
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
            if _mod.startswith("test_"):
                return False
            return True

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

        def helper():
            if f"{spec}_instance" in self.functions[mod_name]:
                del self.functions[mod_name][f"{spec}_instance"]
            if f"{spec}_instance_type" in self.functions[mod_name]:
                del self.functions[mod_name][f"{spec}_instance_type"]

        if on_exit is None and self.functions[mod_name].get(f"{spec}_instance_type", "").endswith("/BC"):
            instance = self.functions[mod_name].get(f"{spec}_instance", None)
            if instance is not None and hasattr(instance, 'on_exit'):
                if inspect.iscoroutinefunction(instance.on_exit):
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
        for f in on_exit:
            try:
                f_, e = self.get_function((mod_name, f), state=True, specification=spec)
                if e == 0:
                    self.logger.info(Style.GREY(f"Running On exit {f} {i}/{len(on_exit)}"))
                    if inspect.iscoroutinefunction(f_):
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

        def helper():
            if f"{spec}_instance" in self.functions[mod_name]:
                del self.functions[mod_name][f"{spec}_instance"]
            if f"{spec}_instance_type" in self.functions[mod_name]:
                del self.functions[mod_name][f"{spec}_instance_type"]

        if on_exit is None and self.functions[mod_name].get(f"{spec}_instance_type", "").endswith("/BC"):
            instance = self.functions[mod_name].get(f"{spec}_instance", None)
            if instance is not None and hasattr(instance, 'on_exit'):
                if inspect.iscoroutinefunction(instance.on_exit):
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
                f_, e = self.get_function((mod_name, f), state=True, specification=spec)
                if e == 0:
                    self.logger.info(Style.GREY(f"Running On exit {f} {i}/{len(on_exit)}"))
                    if inspect.iscoroutinefunction(f_):
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
            finally:
                i += 1

        helper()

        if delete:
            self.functions[mod_name] = {}
            del self.functions[mod_name]

    def exit(self, remove_all=True):
        if self.args_sto.debug:
            self.hide_console()
        self.disconnect()
        if remove_all:
            self.remove_all_modules()
        self.logger.info("Exiting ToolBox interface")
        self.alive = False
        self.called_exit = True, time.time()
        self.save_exit()
        try:
            self.config_fh.save_file_handler()
        except SystemExit:
            print("If u ar testing this is fine else ...")

        import threading

        # for thread in threading.enumerate():
        #     if thread.name == "MainThread":
        #         continue
        #     try:
        #         with Spinner(f"closing Thread {thread.name:^50}|", symbols="s", count_down=True,
        #                      time_in_s=1.251 if not self.debug else 1.1):
        #             thread.join(timeout=1.251 if not self.debug else 1.1)
        #     except TimeoutError as e:
        #         self.logger.error(f"Timeout error on exit {thread.name} {str(e)}")
        #         print(str(e), f"Timeout {thread.name}")

    async def a_exit(self):
        print("Exit a_remove_all_modules")
        await self.a_remove_all_modules()
        results = await asyncio.gather(*[asyncio.create_task(f()) for f in self.exit_tasks if inspect.iscoroutinefunction(f)])
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
        if inspect.iscoroutinefunction(function):
            if function_data.get('row', False):
                return Result.future(asyncio.create_task(self.fuction_runner(function, function_data, args, kwargs)))
            return Result.future(asyncio.create_task(self.fuction_runner(function, function_data, args, kwargs).get()))
        else:
            return self.fuction_runner(function, function_data, args, kwargs)

    def run_a_from_sync(self, function, *args):
        self.loop.call_soon_threadsafe(function, *args)

    def fuction_runner(self, function, function_data: dict, args: list, kwargs: dict):

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
            self.logger.info(f"Execution done")
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
                    f"Function Exec coed: {formatted_result.info.exec_code} Info's: {formatted_result.info.help_text}")
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

    def run_local(self, *args, **kwargs):
        return self.run_any(*args, **kwargs)

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
                                        args_=args, kwargs_=kwargs).as_result()
        if self.debug:
            res.log(show_data=False)
        if not get_results and isinstance(res, Result):
            return res.get()

        return res

    def get_mod(self, name, spec='app') -> ModuleType or MainToolType:
        if name not in self.functions.keys():
            mod = self.save_load(name, spec=spec)
            if mod is False or (isinstance(mod, Result) and mod.is_error()):
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

    def print(self, text, *args, **kwargs):
        # self.logger.info(f"Output : {text}")
        print(Style.CYAN(f"System${self.id}:"), end=" ")
        print(text, *args, **kwargs)

    def sprint(self, text, *args, **kwargs):
        # self.logger.info(f"Output : {text}")
        print(Style.CYAN(f"System${self.id}:"), end=" ")
        if isinstance(text, str) and kwargs == {} and text:
            stram_print(text + ' '.join(args))
            print()
        else:
            print(text, *args, **kwargs)

    # ----------------------------------------------------------------
    # Decorators for the toolbox

    def reload_mod(self, mod_name, spec='app', is_file=True, loc="toolboxv2.mods."):
        if not is_file:
            mods = self.get_all_mods("./mods/" + mod_name)
            for mod in mods:
                try:
                    reload(import_module(loc + mod_name + '.' + mod))
                    self.print(f"Reloaded {mod_name}.{mod}")
                except ImportError:
                    self.print(f"Could not load {mod_name}.{mod}")
        self.inplace_load_instance(mod_name, spec=spec, mfo=reload(self.modules[mod_name]))

    def watch_mod(self, mod_name, spec='app', loc="toolboxv2.mods.", use_thread=True):
        is_file = os.path.isfile(self.start_dir + '/mods/' + mod_name + '.py')
        import watchfiles
        def helper():
            paths = f'mods/{mod_name}' + ('.py' if is_file else '')
            self.print(f'Watching Path: {paths}')
            for changes in watchfiles.watch(paths):
                print(changes)
                self.reload_mod(mod_name, spec, is_file, loc)

        if not use_thread:
            helper()
        else:
            threading.Thread(target=helper, daemon=True).start()

    def _register_function(self, module_name, func_name, data):
        if module_name not in self.functions:
            self.functions[module_name] = {}
        if func_name in self.functions[module_name]:
            self.print(f"Overriding function {func_name} from {module_name}")
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
                          initial=False,
                          exit_f=False,
                          test=True,
                          samples=None,
                          state=None,
                          pre_compute=None,
                          post_compute=None,
                          memory_cache=False,
                          file_cache=False,
                          request_as_kwarg=False,
                          row=False,
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

        def a_additional_process(func):

            async def executor(*args, **kwargs):

                if pre_compute is not None:
                    args, kwargs = await pre_compute(*args, **kwargs)
                if inspect.iscoroutinefunction(func):
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

            def executor(*args, **kwargs):

                if pre_compute is not None:
                    args, kwargs = pre_compute(*args, **kwargs)
                if inspect.iscoroutinefunction(func):
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
                if inspect.iscoroutinefunction(func):
                    func = a_additional_process(func)
                else:
                    func = additional_process(func)
            if api and 'Result' == str(sig.return_annotation):
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
                "__module__": func.__module__,
                "signature": sig,
                "params": params,
                "row": row,
                "state": (
                    False if len(params) == 0 else params[0] in ['self', 'state', 'app']) if state is None else state,
                "do_test": test,
                "samples": samples,
                "request_as_kwarg": request_as_kwarg,

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
           request_as_kwarg: bool = False,
           row: bool = False,
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
                                      request_as_kwarg=request_as_kwarg,
                                      row=row,
                                      memory_cache_max_size=memory_cache_max_size,
                                      memory_cache_ttl=memory_cache_ttl)

    def save_autocompletion_dict(self):
        autocompletion_dict = {}
        for module_name, module in self.functions.items():
            data = {}
            for function_name, function_data in self.functions[module_name].items():
                if not isinstance(function_data, dict):
                    continue
                data[function_name] = {arg: None for arg in
                                       function_data.get("params", [])}  # TODO get default from sig
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
                    f"{func_name.upper().replace('-', '')}:"
                    f" str = '{func_name}'  "
                    f"# Input: ({fuction_data['params'] if isinstance(fuction_data, dict) else ''}),"
                    f" Output: {fuction_data['signature'].return_annotation if isinstance(fuction_data, dict) else 'None'}"
                    for func_name, fuction_data in functions.items()])
            enum_class = (f'@dataclass\nclass {class_name.upper().replace(".", "_").replace("-", "")}(Enum):'
                          f"\n    NAME = '{class_name}'\n    {enum_members}")
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
        # init_type = "main"
    else:
        data = open(init_from, 'r+').read()

    fh = FileHandler(name + ".config")
    fh.open_s_file_handler()
    fh.file_handler_storage.write(str(data))
    fh.file_handler_storage.close()

    logger.info("Done!")
