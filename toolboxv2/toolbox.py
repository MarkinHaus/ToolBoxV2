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

    def __init__(self, filename, name='mainTool', keys=None, defaults=None):
        if defaults is None:
            defaults = {}
        if keys is None:
            keys = {}
        assert filename.endswith(".config") or filename.endswith(".data"), \
            f"filename must end with .config or .data {filename=}"
        self.file_handler_save = {}
        self.file_handler_load = []
        self.file_handler_filename = filename
        self.file_handler_storage = None
        self.file_handler_index_ = 0
        self.file_handler_file_prefix = f".{filename.split('.')[1]}/{name.replace('.', '-')}/"
        self.load_file_handler()
        self.set_defaults_keys_file_handler(keys, defaults)

    def _open_file_handler(self, mode: str, rdu):
        if self.file_handler_storage:
            self.file_handler_storage.close()
            self.file_handler_storage = None
        try:
            self.file_handler_storage = open(self.file_handler_file_prefix + self.file_handler_filename, mode)
            self.file_handler_index_ += 1
        except FileNotFoundError:
            if self.file_handler_index_ >= 5:
                print(Style.RED(f"pleas create this file to prosed : {self.file_handler_file_prefix}"
                                f"{self.file_handler_filename}"))
                exit(0)
            self.file_handler_index_ += 1
            print(Style.YELLOW(f"Try Creating File: {self.file_handler_file_prefix}{self.file_handler_filename}"),
                  end=" ")

            if not os.path.exists(f"{self.file_handler_file_prefix}"):
                os.makedirs(f"{self.file_handler_file_prefix}")

            with open(self.file_handler_file_prefix + self.file_handler_filename, 'a'):
                print(Style.GREEN("File created successfully"))
                self.file_handler_index_ = -1
            rdu()

    def open_s_file_handler(self):
        self._open_file_handler('w+', self.open_s_file_handler)
        return self

    def open_l_file_handler(self):
        self._open_file_handler('r+', self.open_l_file_handler)
        return self

    def save_file_handler(self):
        if self.file_handler_storage:
            print(f"WARNING file is already open (S): {self.file_handler_filename} {self.file_handler_storage}")

        self.open_s_file_handler()

        for key in self.file_handler_save.keys():
            data = self.file_handler_save[key]
            self.file_handler_storage.write(key+str(data))
            self.file_handler_storage.write('\n')

        self.file_handler_storage.close()
        self.file_handler_storage = None

        return self

    def add_to_save_file_handler(self, key: str, value: str):
        if len(key) != 10:
            print('WARNING: key length is not 10 characters')
            return
        if key not in self.file_handler_save.keys():
            print(Style.YELLOW(f"{key} wos not found in file set new"))
            w = 'None'
        else:
            w = self.file_handler_save[key]

        self.file_handler_save[key] = self.encode_code(value)
        return w, self.decode_code(w)

    def load_file_handler(self):

        if self.file_handler_storage:
            print(f"WARNING file is already open (L) {self.file_handler_filename} {self.file_handler_storage}")

        self.open_l_file_handler()

        for line in self.file_handler_storage:
            line = line[:-1]
            heda = line[:10]
            self.file_handler_save[heda] = line[10:]
            enc = self.decode_code(line[10:])
            append = [heda, enc]
            self.file_handler_load.append(append)

        self.file_handler_storage.close()
        self.file_handler_storage = None

        return self

    def get_file_handler(self, obj: str) -> str or None:
        self.file_handler_index_ = -1
        f = []
        for objects in self.file_handler_load:
            self.file_handler_index_ += 1
            f.append(objects[0])
            if obj == objects[0]:

                try:
                    return eval(objects[1])
                except ValueError:
                    print(Style.RED(f"Error Loading {obj} use default if provided"))
                except SyntaxError:
                    pass  # print(Style.YELLOW(f"Data frc : {obj} ; {objects[1]}"))
                except NameError:
                    return str(objects[1])

        if obj in list(self.file_handler_save.keys()):
            return self.decode_code(self.file_handler_save[obj])
        if obj == '-all-f-data':
            return f

        return None

    def set_defaults_keys_file_handler(self, keys: dict, defaults: dict):
        list_keys = iter(list(keys.keys()))
        df_keys = defaults.keys()
        file_keys = self.get_file_handler("-all-f-data")
        for key in list_keys:

            if key in file_keys:
                continue

            if key in df_keys:
                self.file_handler_load.append([keys[key], str(defaults[key])])
                self.file_handler_save[keys[key]] = self.encode_code(defaults[key])
            else:
                self.file_handler_load.append([keys[key], "None"])


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
        print("Starting Tool - Box from : ", Style.Bold(Style.CYAN(f"{os.getcwd()}")))

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

        self.debug = False

        self.debug = self.config_fh.get_file_handler(self.keys["debug"])
        self.command_history = self.config_fh.get_file_handler(self.keys["comm-his"])
        self.dev_modi = self.config_fh.get_file_handler(self.keys["develop-mode"])
        self.MACRO = self.config_fh.get_file_handler(self.keys["MACRO"])
        self.MACRO_color = self.config_fh.get_file_handler(self.keys["MACRO_C"])
        self.HELPER = self.config_fh.get_file_handler(self.keys["HELPER"])
        self.id = self.config_fh.get_file_handler(self.keys["id"])
        self.stuf_load = self.config_fh.get_file_handler(self.keys["st-load"])
        self.mlm = self.config_fh.get_file_handler(self.keys["module-load-mode"])

        self.auto_save = True
        self.PREFIX = Style.CYAN(f"~{node()}@>")
        self.MOD_LIST = {}
        self.logs_ = []
        self.SUPER_SET = []
        self.AC_MOD = None
        self.alive = True

        print(f"SYSTEM :: {node()}\nVersion -> {self.version},\nID -> {self.id},\nload_mode -> {'coppy' if self.mlm == 'C' else ('Inplace' if self.mlm == 'I' else 'pleas use I or C')}\n")

        if args.update:
            self.run_any("cloudM", "#update-core", [])

        if args.get_version:
            v = self.version
            if args.mod_version_name != "mainTool":
                v = self.run_any(args.mod_version_name, 'Version', [])
            print(f"Version {args.mod_version_name} : {v}")
            self.save_exit()
            self.exit()

    def _save_data(self, key, data):
        if self.config_fh.file_handler_index_ == -1:
            self.debug_print(f"Config - Installation Don for {key}")
        if self.config_fh.file_handler_index_ == 6:
            self.debug_print(f"data missing : {key=}")

        self.config_fh.add_to_save_file_handler(key, data)

    def _coppy_mod(self, content, new_mod_dir, mod_name):
        mode = 'xb'

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
            print("Reloading mod from =", loc+mod_name)
            self.remove_mod(mod_name)
        mod = import_module(loc + mod_name)
        mod = getattr(mod, "Tools")
        mod = mod(app=self)
        mod_name = mod.name
        self.MOD_LIST[mod_name.lower()] = mod
        color = mod.color if mod.color else "WHITE"
        self.MACRO.append(mod_name.lower())
        self.MACRO_color[mod_name.lower()] = color
        self.HELPER[mod_name.lower()] = mod.tools["all"]

        return mod

    def _get_function(self, name):
        if not self.AC_MOD:
            self.debug_print(Style.RED("No module Active"))
            return None

        if self.debug:
            return self.AC_MOD.tools[self.AC_MOD.tools["all"][self.SUPER_SET.index(name.lower())][0]]

        if name.lower() not in self.SUPER_SET:
            self.debug_print(Style.RED(f"KeyError: {name} function not found 404"))
            return None

        return self.AC_MOD.tools[self.AC_MOD.tools["all"][self.SUPER_SET.index(name.lower())][0]]

    def save_exit(self):
        self._save_data(self.keys["debug"], str(self.debug))
        self._save_data(self.keys["st-load"], str(self.stuf_load))
        self._save_data(self.keys["module-load-mode"], self.mlm)

    def load_mod(self, mod_name):

        if self.mlm == "I":
            return self.inplace_load(mod_name)
        if self.mlm == "C":
            return self._copy_load(mod_name)
        else:
            raise ValueError(f"config mlm must bee I (inplace load) or C (coppy to runtime load) is {self.mlm=}")

    def load_all_mods_in_file(self, working_dir="mods"):
        w_dir = self.id.replace(".", "_")
        if self.mlm == "C":
            if os.path.exists(f"./runtime/{w_dir}/mod_lib"):
                working_dir = f"./runtime/{w_dir}/mod_lib/"
        if working_dir == "mods":
            pr = "_dev" if self.dev_modi else ""
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
        iter_res = iter(res)
        for mod in iter_res:
            if do_helper(mod):
                print(f"Loading module : {mod[:-3]}", end=' ')
                if self.debug:
                    self.load_mod(mod[:-3])
                    continue
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

        del self.MOD_LIST[mod_name.lower()]
        del self.MACRO_color[mod_name.lower()]
        del self.HELPER[mod_name.lower()]
        self.MACRO.remove(mod_name.lower())

    def colorize(self, obj):
        for pos, o in enumerate(obj):
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

    def autocompletion(self, command):
        options = []
        if command == "":
            return options
        for macro in self.MACRO + self.SUPER_SET:
            if macro.startswith(command.lower()):
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
        self.config_fh.save_file_handler()

    def help(self, command: str):
        if not self.AC_MOD and command == "":
            print(f"All commands: {self.pretty_print(self.MACRO)} \nfor mor information type : help [command]")
            return "intern-error"
        elif self.AC_MOD:
            print(Style.Bold(self.AC_MOD.name))
            self.command_viewer(self.AC_MOD.tools["all"])
            return self.AC_MOD.tools["all"]

        elif command.lower() in self.HELPER.keys():
            helper = self.HELPER[command.lower()]
            print(Style.Bold(command.lower()))
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

        if module_name.lower() not in list(self.MOD_LIST.keys()):
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
            self.SUPER_SET.append(spec[0].lower())

    def new_ac_mod(self, name):
        if name.lower() not in self.MOD_LIST.keys():
            return f"valid mods ar : {self.MOD_LIST.keys()}"
        self.AC_MOD = self.MOD_LIST[name.lower()]
        self.AC_MOD.stuf = self.stuf_load
        self.PREFIX = Style.CYAN(
            f"~{node()}:{Style.Bold(self.pretty_print([name.lower()]).strip())}{Style.CYAN('@>')}")
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


def _initialize_toolBox(init_type, init_from):
    data = ""
    print("Initialing ToolBox: " + init_type)
    if init_type.startswith("http"):
        print("Download from url: " + init_from + "\n->temp_config.config")
        try:
            data = requests.get(init_from).json()["res"]
        except TimeoutError:
            print(Style.RED("Error retrieving config information "))
            exit(1)

        init_type = "main"
    else:
        data = open(init_from, 'r+').read()

    fh = FileHandler(init_type + '-' + node() + ".config")
    fh.open_s_file_handler()
    fh.file_handler_storage.write(str(data))
    fh.file_handler_storage.close()

    print("Done!")
