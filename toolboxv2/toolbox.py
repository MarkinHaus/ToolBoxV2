"""Main module."""
import os
from platform import node
from importlib import import_module
from inspect import signature

import requests

import toolboxv2
from toolboxv2.Style import Style


class MainTool:
    def __init__(self, *args, **kwargs):
        self.version = kwargs["v"]
        self.tools = kwargs["tool"]
        self.name = kwargs["name"]
        self.logs = kwargs["logs"]
        self.color = kwargs["color"]
        self.todo = kwargs["load"]
        self._on_exit = kwargs["on_exit"]
        self.stuf = False
        self.load()

    def load(self):
        if self.todo:
            self.todo()
            self.logs.append([self, "load successfully"])
        else:
            self.logs.append([self, "no load require"])
        print(f"TOOL successfully loaded : {self.name}")
        self.logs.append([self, "TOOL successfully loaded"])

    def print(self, message, *args, end="\n"):
        if self.stuf:
            return

        print(Style.style_dic[self.color] + self.name + Style.style_dic["END"] + ":", message, end=end)

    def log(self, message):
        self.logs.append([self, message])


class Code:
    @staticmethod
    def decode_code(data):
        # letters = string.ascii_letters + string.digits + string.punctuation
        # decode_str = ''
        # data_n = data.split('#')
        # data = []
        # for data_z in data_n[:-1]:
        #    data.append(float(data_z))
        # i = 0
        # for data_z in data:
        #    ascii_ = data_z * 2
        #    decode_str += letters[int(ascii_)]
        #    i += 1
        # decode_str = decode_str.replace('-ou-', 'u')
        # decode_str = decode_str.split('@')
        # return decode_str
        return data

    @staticmethod
    def encode_code(data):
        # letters = string.ascii_letters + string.digits + string.punctuation
        # encode_str = ''
        # data = data.replace(' ', '@')
        # leng = data.__len__()
        # for data_st in range(leng):
        #    i = -1
        #    while data[data_st] != letters[i]:
        #        i += 1
        #        if data[data_st] == letters[i]:
        #            encode_str += str(i / 2) + '#'
        #    data_st += 1
        # return encode_str
        return str(data)


class FileHandler(Code):

    def __init__(self, filename, name='mainTool'):
        assert filename.endswith(".config") or filename.endswith(".data"), \
            f"filename must end with .config or .data {filename=}"
        self.file_handler_save = []
        self.file_handler_load = []
        self.file_handler_auto_save = {}
        self.file_handler_filename = filename
        self.file_handler_storage = None
        self.file_handler_index_ = -1
        self.file_handler_file_prefix = f".{filename.split('.')[1]}/{name.replace('.', '-')}/"

    def _open_file_handler(self, mode: str, rdu):
        if self.file_handler_storage:
            self.file_handler_storage.close()
        try:
            self.file_handler_storage = open(self.file_handler_file_prefix + self.file_handler_filename, mode)
            self.file_handler_index_ = -1
        except FileNotFoundError:
            if self.file_handler_index_ >= 1000:
                print(Style.RED(f"pleas create this file to prosed : {self.file_handler_file_prefix}"
                                f"{self.file_handler_filename}"))
                exit(0)
            print(Style.YELLOW(f"Try Creating File: {self.file_handler_file_prefix}{self.file_handler_filename}"),
                  end=" ")
            # if input("Do you want to create this path | file ? (y/n) ") in ['y', 'yes', 'Y']:
            if not os.path.exists(f"{self.file_handler_file_prefix}"):
                os.makedirs(f"{self.file_handler_file_prefix}")
            open(self.file_handler_file_prefix + self.file_handler_filename, 'a').close()
            print(Style.GREEN("File created successfully"))
            self.file_handler_index_ = 1000
            rdu()

    def open_s_file_handler(self):
        self._open_file_handler('w+', self.open_s_file_handler)
        return self

    def open_l_file_handler(self):
        self._open_file_handler('r+', self.open_l_file_handler)
        return self

    def save_file_handler(self):
        for pos, data in enumerate(self.file_handler_auto_save.keys()):
            if self.file_handler_auto_save[data]:
                self.add_to_save_file_handler(data, self.file_handler_load[pos][1])
        if not self.file_handler_storage:
            # self.open_s_file_handler()
            print("WARNING pleas open storage")
        for line in self.file_handler_save:
            self.file_handler_storage.write(line)
            self.file_handler_storage.write('\n')

        return self

    def add_to_save_file_handler(self, key: str, value: str):
        if len(key) != 10:
            print('WARNING: key length is not 10 characters')
            return
        try:
            self._set_auto_save_file_handler(key)
            self.file_handler_save.append(key + self.encode_code(value))
        except ValueError:
            print(Style.RED(f"{value=}\n\n{key=} was not saved.\n{type(value)=}!=str"))

    def load_file_handler(self):
        if not self.file_handler_storage:
            # self.open_l_file_handler()
            print("WARNING pleas open storage")
        for line in self.file_handler_storage:
            line = line[:-1]
            heda = line[:10]
            enc = self.decode_code(line[10:])
            append = [heda, enc]
            self.file_handler_auto_save[heda] = True
            self.file_handler_load.append(append)
        return self

    def _set_auto_save_file_handler(self, key: str):
        self.file_handler_auto_save[key] = False

    def get_file_handler(self, obj: str) -> str or None:
        self.file_handler_index_ = -1
        for objects in self.file_handler_load:
            self.file_handler_index_ += 1
            # print(objects)
            if obj == objects[0]:
                return objects[1]
        return None


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

    def default(self):
        return self


class App:
    def __init__(self, prefix: str = "", args=AppArgs().default()):

        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)
        print(Style.Bold(Style.CYAN(f"using local mods = {os.getcwd()}")))

        if args.init:
            if not args.init_file:
                args.init_file = "init.config"
            self._initialize_toolBox(args.init, args.init_file)

        name = prefix + '-' + node()
        self.version = toolboxv2.__version__
        self.config_fh = FileHandler(name + ".config")
        self.config_fh.open_l_file_handler()
        self.config_fh.load_file_handler()

        self.keys = {
            "MACRO": "macro~~~~:",
            "MACRO_C": "m_color~~:",
            "HELPER": "helper~~~:",
            "debug": "debug~~~~:",
            "id": "name-spa~:",
            "st-load": "mute~load:",
            "module-load-mode": "load~mode:",
        }
        self.MACRO = self._get_config_data("MACRO", [])
        self.MACRO_color = self._get_config_data("MACRO_C", {})
        self.HELPER = self._get_config_data("HELPER", {})
        self.id = self._get_config_data("id", [name])[0]
        self.stuf_load = self._get_config_data("st-load", False)
        self.mlm = self._get_config_data("module-load-mode", ["I"])[0]
        self.auto_save = True
        self.PREFIX = Style.CYAN(f"~{node()}@>")
        self.MOD_LIST = {}
        self.logs_ = []
        self.SUPER_SET = []
        self.AC_MOD = None
        self.alive = True
        self.debug = self._get_config_data("debug", args.live)

        print("SYSTEM :: " + node())

        if args.update:
            self.run_any("cloudM", "#update-core", [])

        if args.get_version:
            v = self.version
            if args.mod_version_name != "mainTool":
                v = self.run_any(args.mod_version_name, 'Version', [])
            print(f"Version {args.mod_version_name} : {v}")
            self.save_exit()
            self.exit()

    def _initialize_toolBox(self, init_type, init_from):
        data = ""
        print("Initialing ToolBox: " + init_type)
        if init_type.startswith("http"):
            print("Download from url: " + init_from + "\n->temp_config.config")
            try:
                data = requests.get(init_from).json()["res"]
            except Exception:
                print(Style.RED("Error retrieving config information "))
                exit(1)

            init_type = "init_config"
        else:
            data = open(init_from, 'r+').read()

        fh = FileHandler(init_type + '-' + node() + ".config")
        fh.open_s_file_handler()
        fh.file_handler_storage.write(str(data))
        fh.file_handler_storage.close()

    def _test_repeat(self):
        if self.config_fh.file_handler_index_ == -1:
            self.debug_print("Config - Installation Don")
        if self.config_fh.file_handler_index_ == 0:
            self.debug_print("Darten Wurden Wärend Runtim Entfernt")
        return self.config_fh.file_handler_index_ <= 0

    def _get_config_data(self, key, t):
        data = self.config_fh.get_file_handler(self.keys[key])
        if data is not None:
            try:
                return eval(data)
            except ValueError:
                self.debug_print(f"Error Loading {key}")
        return t

    def _coppy_mod(self, content, new_mod_dir, mod_name):
        if not os.path.exists(new_mod_dir):
            os.makedirs(new_mod_dir)
            open(f"{new_mod_dir}/__init__.py", "w").write(
                f"__version__ = '{self.version}'")
        if os.path.exists(f"{new_mod_dir}/{mod_name}.py"):
            with open(f"{new_mod_dir}/{mod_name}.py", 'rb') as d:
                runtime_mod = d.read()  # Testing version but not efficient
                if len(content) != len(runtime_mod):
                    with open(f"{new_mod_dir}/{mod_name}.py", 'wb') as f:
                        f.write(content)
        else:
            with open(f"{new_mod_dir}/{mod_name}.py", 'xb') as f:
                f.write(content)

    def _pre_lib_mod(self, mod_name):
        working_dir = self.id.replace(".", "_")
        lib_mod_dir = f"toolboxv2.runtime.{working_dir}.mod_lib."
        postfix = "_dev" if self.debug else ""
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
        if self.debug and loc == "toolboxv2.mods.":
            loc = "toolboxv2.mods_dev."
        if mod_name.upper() in list(self.MOD_LIST.keys()):
            print("Reloading mod from = ", loc)
            self.remove_mod(mod_name)
        mod = import_module(loc + mod_name)
        mod = getattr(mod, "Tools")
        #
        mod = mod(app=self)
        mod_name = mod.name
        self.MOD_LIST[mod_name.upper()] = mod
        color = mod.color if mod.color else "WHITE"
        self.MACRO.append(mod_name.upper())
        self.MACRO_color[mod_name.upper()] = color
        self.HELPER[mod_name.upper()] = mod.tools["all"]
        # # for spec, _ in mod.tools["all"]:
        # #     self.spec.append(mod_name.upper() + "-" + spec.upper())

        return mod

    def _get_function(self, name):
        if not self.AC_MOD:
            self.debug_print(Style.RED("No module Active"))
            return None
        if self.debug:
            return self.AC_MOD.tools[self.AC_MOD.tools["all"][self.SUPER_SET.index(name.upper())][0]]
        try:
            return self.AC_MOD.tools[self.AC_MOD.tools["all"][self.SUPER_SET.index(name.upper())][0]]
        except KeyError as e:
            print(Style.RED(f"KeyError: {e} function not found 404"))
            return None

    def save_exit(self):

        if self._test_repeat():
            self.config_fh.add_to_save_file_handler(self.keys["MACRO"], str(self.MACRO))

        if self._test_repeat():
            self.config_fh.add_to_save_file_handler(self.keys["MACRO_C"], str(self.MACRO_color))

        if self._test_repeat():
            self.config_fh.add_to_save_file_handler(self.keys["debug"], str(self.debug))

        if self._test_repeat():
            self.config_fh.add_to_save_file_handler(self.keys["st-load"], str(self.stuf_load))

        if self._test_repeat():
            self.config_fh.add_to_save_file_handler(self.keys["id"], str([self.id]))

        if self._test_repeat():
            self.config_fh.add_to_save_file_handler(self.keys["module-load-mode"], str([self.mlm]))

    def test_repeat_ac_mod(self):
        if self.AC_MOD.file_handler_index_ == -1:
            self.debug_print("Config - Installation Don")
        if self.AC_MOD.file_handler_index_ == 0:
            self.debug_print("Darten Wurden Wärend Runtim Entfernt")
        return self.AC_MOD.file_handler_index_ <= 0

    def load_mod(self, mod_name):

        if self.mlm == "I":
            return self.inplace_load(mod_name)
        if self.mlm == "C":
            return self._copy_load(mod_name)
        else:
            raise ValueError(f"config mlm must bee I or C is {self.mlm}")

    def load_all_mods_in_file(self, working_dir="mods"):#
        w_dir = self.id.replace(".", "_")
        if self.mlm == "C":
            if os.path.exists(f"./runtime/{w_dir}/mod_lib"):
                working_dir = f"./runtime/{w_dir}/mod_lib/"
        if working_dir == "mods":
            pr = "_dev" if self.debug else ""
            working_dir = f"./mods{pr}"

        res = os.listdir(working_dir)

        def do_helper(_mod):
            if "mainTool" in mod:
                return False
            if not mod.endswith(".py"):
                return False
            if mod.startswith("__"):
                return False
            if mod.startswith("test_"):
                return False
            return True

        for mod in res:
            if do_helper(mod):
                print(f"Loading module : {mod[:-3]}", end=' ')
                if self.debug:
                    self.load_mod(mod[:-3])
                try:
                    self.load_mod(mod[:-3])
                except Exception as e:
                    print(Style.RED("Error") + f" loading modules : {e}")

        return True

    def remove_all_modules(self):
        iter_list = self.MOD_LIST.copy()

        self.exit_all_modules()

        for mod_name in iter_list.keys():
            try:
                self.remove_mod(mod_name)
            except Exception as e:
                self.debug_print(Style.RED("ERROR: %s %e" % mod_name % e))

    def exit_all_modules(self):
        for mod in self.MOD_LIST.items():
            print("closing:", mod[0], ": ", end="")
            if mod[1]._on_exit:
                try:
                    mod[1]._on_exit()
                    self.print_ok()
                except Exception as e:
                    self.debug_print(Style.YELLOW(Style.Bold(f"closing ERROR : {e}")))

    def print_ok(self):
        if self.id.startswith("test"):
            print("OK")
            return
        print(Style.GREEN(Style.Bold(f"🆗")))

    def remove_mod(self, mod_name):

        del self.MOD_LIST[mod_name.upper()]
        del self.MACRO_color[mod_name.upper()]
        del self.HELPER[mod_name.upper()]
        self.MACRO.remove(mod_name.upper())

    def colorize(self, obj):
        for pos, o in enumerate(obj):
            if o.upper() in self.MACRO:
                if o.upper() in self.MACRO_color.keys():
                    obj[pos] = f"{Style.style_dic[self.MACRO_color[o.upper()]]}{o}{Style.style_dic['END']}"
        return obj

    def pretty_print(self, obj: list):
        obj_work = obj.copy()
        obj_work = self.colorize(obj_work)
        s = ""
        for i in obj_work:
            s += str(i) + " "
        return s

    def autocompletion(self, command):
        options = []
        if command == "":
            return options
        for macro in self.MACRO + self.SUPER_SET:
            if macro.startswith(command.upper()):
                options.append(macro)

        return options

    def logs(self):
        print(f"PREFIX={self.PREFIX}"
              f"\nMACRO={self.pretty_print(self.MACRO[:7])}"
              f"\nMODS={self.pretty_print(self.MACRO[7:])}"
              f"\nSUPER_SET={self.pretty_print(self.SUPER_SET)}")
        self.command_viewer(self.logs_)

    def exit(self):
        self.exit_all_modules()
        print(Style.Bold(Style.CYAN("OK - EXIT ")))
        print('\033[?25h', end="")
        self.alive = False
        self.config_fh.open_s_file_handler()
        self.config_fh.save_file_handler()
        self.config_fh.file_handler_storage.close()

    def help(self, command: str):
        if not self.AC_MOD and command == "":
            print(f"All commands: {self.pretty_print(self.MACRO)} \nfor mor information type : help [command]")
            return "intern-error"
        elif self.AC_MOD:
            print(Style.Bold(self.AC_MOD.name))
            self.command_viewer(self.AC_MOD.tools["all"])
            return self.AC_MOD.tools["all"]

        elif command.upper() in self.HELPER.keys():
            helper = self.HELPER[command.upper()]
            print(Style.Bold(command.upper()))
            self.command_viewer(helper)
            return helper
        else:
            print(Style.RED(f"HELPER {command} is not a valid | valid commands ar"
                            f" {self.pretty_print(list(self.HELPER.keys()))}"))
            return "invalid commands"

    def save_load(self, filename):
        if self.debug:
            return self.load_mod(filename)
        try:
            return self.load_mod(filename)
        except ModuleNotFoundError:
            print(Style.RED(f"Module {filename} not found"))

    def reset(self):
        self.AC_MOD = None
        self.PREFIX = Style.CYAN(f"~{node()}@>")
        self.SUPER_SET = []

    def get_file_handler_name(self):
        if not self.AC_MOD:
            self.debug_print(Style.RED("No module Active"))
            return None
        try:
            if self.AC_MOD.file_handler_filename:
                return self.AC_MOD.file_handler_filename
        except AttributeError as e:
            print(Style.RED(f"AttributeError: {e} has no file handler 404"))
            return None

    def run_function(self, name, command):
        # get function
        function = self._get_function(name)
        res = {}
        if not function:
            self.debug_print(Style.RED(f"Function {name} not found"))
            return False
        # signature function
        sig = signature(function)
        args = len(sig.parameters)

        if args == 0:
            if self.debug:
                res = function()
            else:
                try:
                    self.print_ok()
                    print("\nStart function\n")
                    res = function()
                    self.debug_print(Style.GREEN(f"\n-"))
                except Exception as e:
                    self.debug_print(Style.YELLOW(Style.Bold(f"! function ERROR : {e}")))

        elif args == 1:
            if self.debug:
                res = function(command)
            else:
                try:
                    self.print_ok()
                    print("\nStart function\n")
                    res = function(command)
                    self.debug_print(Style.GREEN(f"\n-"))
                except Exception as e:
                    self.debug_print(Style.YELLOW(Style.Bold(f"! function ERROR : {e}")))

        elif args == 2:
            if self.debug:
                res = function(command, self)
            else:
                try:
                    self.print_ok()
                    print("\nStart function\n")
                    res = function(command, self)
                    self.debug_print(Style.GREEN(f"\n-"))
                except Exception as e:
                    self.debug_print(Style.YELLOW(Style.Bold(f"! function ERROR : {e}")))
        else:
            self.debug_print(Style.YELLOW(f"! to many args {args} def ...(u): | -> {str(sig)}"))

        self.debug_print(res)

        self.debug_print(f"Name: {self.id} : {__name__}")

        return res

    def run_any(self, module_name: str, function_name: str, command: list):

        if module_name.upper() not in list(self.MOD_LIST.keys()):
            print(f"Module : {module_name}.{function_name} not online")
            self.save_load(module_name)

        do_sto = self.AC_MOD is not None
        ac_sto = ""
        if do_sto:
            ac_sto = self.AC_MOD.name

        self.new_ac_mod(module_name)
        res = self.run_function(function_name, command)

        if do_sto:
            self.new_ac_mod(ac_sto)
        if self.debug:
            print(res)

        return res

    def set_spec(self):
        self.SUPER_SET = []
        for spec in self.AC_MOD.tools["all"]:
            self.SUPER_SET.append(spec[0].upper())

    def new_ac_mod(self, name):
        if name.upper() not in self.MOD_LIST.keys():
            return f"valid mods ar : {self.MOD_LIST.keys()}"
        self.AC_MOD = self.MOD_LIST[name.upper()]
        self.AC_MOD.stuf = self.stuf_load
        self.PREFIX = Style.CYAN(
            f"~{node()}:{Style.Bold(self.pretty_print([name.upper()]).strip())}{Style.CYAN('@>')}")
        self.set_spec()

    def debug_print(self, message, *args, end="\n"):
        if self.debug:
            print(message, *args, end=end)

    @staticmethod
    def command_viewer(mod_command):
        mod_command_names = []
        mod_command_dis = []
        print(f"\n")
        for msg in mod_command:
            if msg[0] not in mod_command_names:
                mod_command_names.append(msg[0])
                mod_command_dis.append([])

            for dis in msg[1:]:
                mod_command_dis[mod_command_names.index(msg[0])].append(dis)

        for tool_address in mod_command_names:
            print(Style.GREEN(f"{tool_address}, "))
            for log_info in mod_command_dis[mod_command_names.index(tool_address)]:
                print(Style.YELLOW(f"    {log_info}"))
            print("\n")

        return mod_command_names
