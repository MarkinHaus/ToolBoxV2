"""Main module."""
import os
import sys
import time
from platform import node, system
from importlib import import_module
from inspect import signature

import requests

from toolboxv2.utils.file_handler import FileHandler
from toolboxv2.utils.tb_logger import setup_logging, get_logger
from toolboxv2.utils.Style import Style
import toolboxv2

import logging
from dotenv import load_dotenv

import json
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter

load_dotenv()


class AppArgs:
    init = None
    init_file = None
    update = False
    update_mod = None,
    delete_ToolBoxV2 = None
    delete_mod = None
    get_version = False
    mod_version_name = 'mainTool'
    name = 'main'
    modi = 'cli'
    port = 68945
    host = '0.0.0.0'
    load_all_mod_in_files = False
    live = False

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


class App(metaclass=Singleton):
    def __init__(self, prefix: str = "", args=AppArgs().default()):

        t0 = time.time()
        abspath = os.path.abspath(__file__)
        self.system_flag = system()  # Linux: Linux Mac: Darwin Windows: Windows
        if self.system_flag == "Darwin" or self.system_flag == "Linux":
            dname = os.path.dirname(abspath).replace("/utils", "")
        else:
            dname = os.path.dirname(abspath).replace("\\utils", "")
        os.chdir(dname)

        self.start_dir = dname

        if not os.path.exists("./.data"):
            os.mkdir("./.data")
        if not os.path.exists("./.config"):
            os.mkdir("./.config")

        if not prefix:
            if not os.path.exists("./.data/last-app-prefix"):
                open("./.data/last-app-prefix", "a").close()
            with open("./.data/last-app-prefix", "r") as prefix_file:
                cont = prefix_file.read()
                if cont:
                    prefix = cont
        else:
            if not os.path.exists("./.data/last-app-prefix"):
                open("./.data/last-app-prefix", "a").close()
            with open("./.data/last-app-prefix", "w") as prefix_file:
                prefix_file.write(prefix)

        print(f"Starting ToolBox as {prefix} from : ", Style.Bold(Style.CYAN(f"{os.getcwd()}")))

        debug = False

        if "test" in prefix:
            self.logger, self.logging_filename = setup_logging(logging.NOTSET, name="toolbox-test", interminal=True,
                          file_level=logging.NOTSET)
        elif "live" in prefix:
            self.logger, self.logging_filename = setup_logging(logging.DEBUG, name="toolbox-debug", interminal=True,
                          file_level=logging.WARNING)
            #setup_logging(logging.WARNING, name="toolbox-live", is_online=True
            #              , online_level=logging.WARNING).info("Logger initialized")
        elif "debug" in prefix:
            prefix = prefix.replace("-debug", '').replace("debug", '')
            self.logger, self.logging_filename = setup_logging(logging.DEBUG, name="toolbox-debug", interminal=True,
                          file_level=logging.WARNING)
            debug = True
        else:
            self.logger, self.logging_filename = setup_logging(logging.ERROR, name=f"toolbox-{prefix}")
        self.logger.info("Logger initialized")
        get_logger().info(Style.GREEN("Starting Tool"))

        if args.init:
            _initialize_toolBox(args.init, args.init_file)

        name = prefix + '-' + node()
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
        }

        defaults = {
            "MACRO": ['Exit'],
            "MACRO_C": {},
            "HELPER": {},
            "debug": not args.live,
            "id": name,
            "st-load": False,
            "module-load-mode": 'I',
            "comm-his": [[]],
            "develop-mode": False,
        }

        self.config_fh = FileHandler(name + ".config", keys=self.keys, defaults=defaults)
        self.config_fh.load_file_handler()

        self._debug = debug

        self.runnable = {}

        self._debug = self.config_fh.get_file_handler(self.keys["debug"])
        self.command_history = self.config_fh.get_file_handler(self.keys["comm-his"])
        self.dev_modi = self.config_fh.get_file_handler(self.keys["develop-mode"])
        self.MACRO = self.config_fh.get_file_handler(self.keys["MACRO"])
        self.MACRO_color = self.config_fh.get_file_handler(self.keys["MACRO_C"])
        self.HELPER = self.config_fh.get_file_handler(self.keys["HELPER"])
        self.id = name  # self.config_fh.get_file_handler(self.keys["id"])
        self.stuf_load = self.config_fh.get_file_handler(self.keys["st-load"])
        self.mlm = self.config_fh.get_file_handler(self.keys["module-load-mode"])

        self.auto_save = True
        self.PREFIX = Style.CYAN(f"~{node()}@>")
        self.MOD_LIST = {}
        self.SUPER_SET = []
        self.AC_MOD = None
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
                f"Finish init up in t-{time.time() - t0}s"
            )
        )

    @property
    def debug(self):
        return self._debug

    def set_runnable(self, r):
        self.runnable = r

    def show_runnable(self):
        self.print(self.pretty_print(list(self.runnable.keys())))
        return self.runnable

    def run_runnable(self, name, args):
        if name in self.runnable.keys():
            self.runnable[args.modi.lower()](self, args)

    @debug.setter
    def debug(self, value):
        if not isinstance(value, bool):
            self.logger.debug(f"Value must be an boolean. is : {value} type of {type(value)}")
            raise ValueError("Value must be an boolean.")

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

    def _pre_lib_mod(self, mod_name):
        working_dir = self.id.replace(".", "_")
        lib_mod_dir = f"toolboxv2.runtime.{working_dir}.mod_lib."

        self.logger.info(f"pre_lib_mod {mod_name} from {lib_mod_dir}")

        postfix = "_dev" if self.dev_modi else ""
        mod_file_dir = f"./mods{postfix}/{mod_name}.py"
        new_mod_dir = f"./runtime/{working_dir}/mod_lib"
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
        if mod_name.lower() in list(self.MOD_LIST.keys()):
            self.logger.info(f"Reloading mod from : {loc + mod_name}")
            self.remove_mod(mod_name)
        mod = import_module(loc + mod_name)
        mod = getattr(mod, "Tools")
        mod = mod(app=self)
        mod_name = mod.name
        self.save_init_mod(mod_name, mod)
        return mod

    def save_init_mod(self, name, mod):
        self.MOD_LIST[name.lower()] = mod
        color = mod.color if mod.color else "WHITE"
        self.MACRO.append(name.lower())
        self.MACRO_color[name.lower()] = color
        self.HELPER[name.lower()] = mod.tools["all"]

    def mod_online(self, mod_name):
        return mod_name.lower() in self.MOD_LIST.keys()

    def _get_function(self, name):

        self.logger.info(f"getting function : {name}")

        if not self.AC_MOD:
            self.logger.debug(Style.RED("No module Active"))
            return None

        for key, func in self.AC_MOD.tools.items():
            if name.lower() == key.lower():
                return func

        if name.lower() not in self.SUPER_SET:
            self.logger.debug(Style.RED(f"KeyError: {name} function not found 404"))
            return None

        return self.AC_MOD.tools[self.AC_MOD.tools["all"][self.SUPER_SET.index(name.lower())][0]]

    def save_exit(self):
        self.logger.info(f"save exiting saving data to {self.config_fh.file_handler_filename} states of {self.debug=}"
                         f" {self.stuf_load=} {self.mlm=}")
        self.config_fh.add_to_save_file_handler(self.keys["debug"], str(self.debug))
        self.config_fh.add_to_save_file_handler(self.keys["st-load"], str(self.stuf_load))
        self.config_fh.add_to_save_file_handler(self.keys["module-load-mode"], self.mlm)
        self.config_fh.add_to_save_file_handler(self.keys["comm-his"], str(self.command_history))

    def load_mod(self, mod_name):

        self.logger.info(f"try opening module {mod_name} in mode {self.mlm}")

        if self.mlm == "I":
            return self.inplace_load(mod_name)
        elif self.mlm == "C":
            return self._copy_load(mod_name)
        else:
            self.logger.critical(f"config mlm must bee I (inplace load) or C (coppy to runtime load) is {self.mlm=}")
            raise ValueError(f"config mlm must bee I (inplace load) or C (coppy to runtime load) is {self.mlm=}")

    def load_all_mods_in_file(self, working_dir="mods"):

        t0 = time.time()

        iter_res = self.get_all_mods(working_dir)

        opened = 0
        for mod in iter_res:
            opened += 1
            self.logger.info(f"Loading module : {mod}")
            if self.debug:
                self.load_mod(mod)
            else:
                try:
                    self.load_mod(mod)
                except Exception as e:
                    self.logger.error(Style.RED("Error") + f" loading module {mod} {e}")

        self.logger.info(f"opened  : {opened} modules in t-{time.time() - t0}s")

        return True

    def get_all_mods(self, working_dir="mods"):
        self.logger.info(f"collating all mods in working directory {working_dir}")

        w_dir = self.id.replace(".", "_")

        if self.mlm == "C":
            if os.path.exists(f"./runtime/{w_dir}/mod_lib"):
                working_dir = f"./runtime/{w_dir}/mod_lib/"
        if working_dir == "mods":
            pr = "_dev" if self.dev_modi else ""
            working_dir = f"./mods{pr}"

        res = os.listdir(working_dir)

        self.logger.info(f"found : {len(res)} files")

        def do_helper(_mod):
            if "mainTool" in _mod:
                return False
            if not _mod.endswith(".py"):
                return False
            if _mod.startswith("__"):
                return False
            if _mod.startswith("test_"):
                return False
            return True

        def r_endings(word: str):
            return word[:-3]

        return list(map(r_endings, filter(do_helper, res)))

    def remove_all_modules(self):
        iter_list = self.MOD_LIST.copy()

        self.exit_all_modules()

        for mod_name in iter_list.keys():
            self.logger.info(f"removing module : {mod_name}")
            try:
                self.remove_mod(mod_name)
            except Exception as e:
                self.logger.error(f"Error removing module {mod_name} {e}")

    def exit_all_modules(self):
        for mod in self.MOD_LIST.items():
            self.logger.info(f"closing: {mod[0]}")
            if mod[1]._on_exit:
                try:
                    mod[1]._on_exit()
                    self.print_ok()
                except Exception as e:
                    self.logger.debug(Style.YELLOW(Style.Bold(f"closing ERROR : {e}")))

    def print_ok(self):
        self.logger.info("OK")

    def remove_mod(self, mod_name):

        self.logger.info(f"Removing mod from sto")
        del self.MOD_LIST[mod_name.lower()]
        del self.MACRO_color[mod_name.lower()]
        del self.HELPER[mod_name.lower()]
        self.MACRO.remove(mod_name.lower())

    def colorize(self, obj):
        for pos, o in enumerate(obj):
            if not isinstance(o, str):
                o = str(o)
            if o.lower() in self.MACRO:
                if o.lower() in self.MACRO_color.keys():
                    obj[pos] = f"{Style.style_dic[self.MACRO_color[o.lower()]]}{o}{Style.style_dic['END']}"
        return obj

    def pretty_print(self, obj: list):
        obj_work = obj.copy()
        obj_work = self.colorize(obj_work)
        s = ""
        for i in obj_work:
            s += str(i) + " "
        return s

    def pretty_print_dict(self, data):
        json_str = json.dumps(data, sort_keys=True, indent=4)
        self.print(highlight(json_str, JsonLexer(), TerminalFormatter()))

    def autocompletion(self, command):
        options = []
        if command == "":
            return options
        for macro in self.MACRO + self.SUPER_SET:
            if macro.startswith(command.lower()):
                options.append(macro)
        self.logger.info(f"Autocompletion in {command} aut : {options}")
        return options

    def exit(self):
        self.exit_all_modules()
        self.logger.info("Exiting ToolBox")
        self.print(Style.Bold(Style.CYAN("OK - EXIT ")))
        self.print('\033[?25h', end="")
        self.alive = False
        self.config_fh.save_file_handler()

    def help(self, command: str):
        if not self.AC_MOD and command == "":
            self.print(f"All commands: {self.pretty_print(self.MACRO)} \nfor mor information type : help [command]")
            return "invalid"
        elif self.AC_MOD:
            self.print(Style.Bold(self.AC_MOD.name))
            self.command_viewer(self.AC_MOD.tools["all"])
            return self.AC_MOD.tools["all"]

        elif command.lower() in self.HELPER.keys():
            helper = self.HELPER[command.lower()]
            self.print(Style.Bold(command.lower()))
            self.command_viewer(helper)
            return helper
        else:
            self.print(Style.RED(f"HELPER {command} is not a valid | valid commands ar"
                            f" {self.pretty_print(list(self.HELPER.keys()))}"))
            return "invalid"

    def save_load(self, modname):
        self.logger.info(f"Save load module {modname}")
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

    def reset(self):
        self.AC_MOD = None
        self.PREFIX = Style.CYAN(f"~{node()}@>")
        self.SUPER_SET = []

    def get_file_handler_name(self):
        if not self.AC_MOD:
            self.logger.debug(Style.RED("No module Active"))
            return None
        try:
            if self.AC_MOD.file_handler_filename:
                return self.AC_MOD.file_handler_filename
        except AttributeError as e:
            self.logger.error(Style.RED(f"AttributeError: {e} has no file handler 404"))
        return None

    def run_function(self, name, *args, **kwargs):
        self.logger.info(f"Start setup for: {name} function mod:{self.AC_MOD.name}")

        function = self._get_function(name)
        if not function:
            self.logger.debug(Style.RED(f"Function {name} not found in {self.AC_MOD.name}"))
            return False

        self.logger.info(f"Profiling function")
        sig = signature(function)
        self.logger.info(f"Signature: {sig}")
        parameters = list(sig.parameters)

        mod_name = self.AC_MOD.name
        self.print(f"\nStart function {mod_name}:{name}\n")
        app_position = None
        for i, param in enumerate(parameters):
            if param == 'app':
                app_position = i
                break
        if app_position is not None:
            args = list(args)
            args.insert(app_position, self)
        try:
            if len(parameters) == 0:
                res = function()
            elif len(parameters) == 1:
                res = function(*args)
            else:
                res = function(*args, **kwargs)
            self.logger.info(f"Execution done")
        except Exception as e:
            self.logger.error(Style.YELLOW(Style.Bold(f"! Function ERROR: in {mod_name}:{name} {e}")))
            res = {'error-in': mod_name, 'error-func': name}
        else:
            self.print_ok()
        return res

    def run_any(self, module_name: str, function_name: str, command=None, **kwargs):

        if command is None:
            command = [ApiOb(), ""]

        do_sto = self.AC_MOD is not None
        ac_sto = ""
        if do_sto:
            ac_sto = self.AC_MOD.name

        if module_name.lower() not in list(self.MOD_LIST.keys()):
            self.logger.warning(f"Module : {module_name}.{function_name} not online")
            self.save_load(module_name)

        if ac_sto != module_name:
            self.new_ac_mod(module_name)
        res = self.run_function(function_name, command, **kwargs)

        if do_sto:
            self.new_ac_mod(ac_sto)

        return res

    def set_spec(self):
        self.SUPER_SET = []
        for spec in self.AC_MOD.tools["all"]:
            self.SUPER_SET.append(spec[0].lower())

    def new_ac_mod(self, name):
        self.logger.info(f"New ac mod : {name}")
        if name.lower() not in self.MOD_LIST.keys():
            self.logger.warning(f"Could not find {name} in {list(self.MOD_LIST.keys())}")
            return f"valid mods ar : {list(self.MOD_LIST.keys())}"
        self.AC_MOD = self.MOD_LIST[name.lower()]
        self.AC_MOD.stuf = self.stuf_load
        self.PREFIX = Style.CYAN(
            f"~{node()}:{Style.Bold(self.pretty_print([name.lower()]).strip())}{Style.CYAN('@>')}")
        self.set_spec()
        return True

    def get_mod(self, name):
        if name.lower() not in self.MOD_LIST.keys():
            mod = self.save_load(name)
            if mod:
                return mod
            self.logger.warning(f"Could not find {name} in {list(self.MOD_LIST.keys())}")
            raise ValueError(f"Could not find {name} in {list(self.MOD_LIST.keys())} pleas install the module")
        return self.MOD_LIST[name.lower()]

    def command_viewer(self, mod_command):
        mod_command_names = []
        mod_command_dis = []
        self.print(f"\n")
        for msg in mod_command:
            if msg[0] not in mod_command_names:
                mod_command_names.append(msg[0])
                mod_command_dis.append([])

            for dis in msg[1:]:
                mod_command_dis[mod_command_names.index(msg[0])].append(dis)

        for tool_address in mod_command_names:
            self.print(Style.GREEN(f"{tool_address}, "))
            for log_info in mod_command_dis[mod_command_names.index(tool_address)]:
                self.print(Style.YELLOW(f"    {log_info}"))
            self.print("\n")

        return mod_command_names

    @debug.setter
    def debug(self, value):
        self._debug = value

    def print(self, text, *args, **kwargs):
        self.logger.info(f"Output : {text}")
        print(text, *args, **kwargs)


def _initialize_toolBox(init_type, init_from):
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

    fh = FileHandler(init_type + '-' + node() + ".config")
    fh.open_s_file_handler()
    fh.file_handler_storage.write(str(data))
    fh.file_handler_storage.close()

    logger.info("Done!")


def get_app(name=None) -> App:
    logger = get_logger()
    logger.info(Style.GREYBG(f"get app requested name: {name}"))
    if name:
        app = App(name)
    else:
        app = App()
    logger.info(Style.Bold(f"App instance, returned ID: {app.id}"))
    return app
