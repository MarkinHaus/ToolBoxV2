"""Console script for toolboxv2. min dep readchar Style"""
import json
import os
# Import default Pages
import sys
from platform import system, node

# Import public Pages
import readchar
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import JsonLexer

from toolboxv2 import App, Style
from toolboxv2.utils import Singleton

NAME = "cli-leg"

prefix = Style.CYAN(f"~{node()}@>")


def user_input(app: App):
    get_input = True
    command = ""
    print_command = []
    helper = ""
    helper_index = 0
    options = []
    sh_index = 0
    session_history = [[]]
    # session_history += [c for c in app.command_history]

    while get_input:

        key = readchar.readkey()

        if key == b'\x05' or key == '\x05':
            print('\033', end="")
            get_input = False
            command = "EXIT"

        elif key == readchar.key.LEFT:
            if helper_index > 0:
                helper_index -= 1

        elif key == readchar.key.RIGHT:
            if helper_index < len(options) - 1:
                helper_index += 1

        elif key == readchar.key.UP:
            sh_index -= 1
            if sh_index <= 0:
                sh_index = len(session_history) - 1
            command = ""
            print_command = session_history[sh_index]

        elif key == readchar.key.DOWN:
            sh_index += 1
            if sh_index >= len(session_history):
                sh_index = 0
            command = ""
            print_command = session_history[sh_index]

        elif key == b'\x08' or key == b'\x7f' or key == '\x08' or key == '\x7f':
            if len(command) == 0 and len(print_command) != 0:
                command = print_command[-1]
                command = command[:-1]
                print_command = print_command[:-1]
            else:
                command = command[:-1]
        elif key == b' ' or key == ' ':
            print_command.append(command)
            command = ""
        elif key == readchar.key.ENTER:
            get_input = False
            print_command.append(command)
        elif key == b'\t' or key == '\t':
            command += helper
        else:
            if type(key) == str:
                command += key
            else:
                command += str(key, "ISO-8859-1")

        options = list(set(A().autocompletion(command)))

        if helper_index > len(options) - 1:
            helper_index = 0

        helper = ""
        do = len(options) > 0
        if do:
            helper = options[helper_index][len(command):].lower()

        to_print = prefix + A().pretty_print(print_command + [command + Style.Underline(Style.Bold(helper))])
        if do:
            to_print += " | " + Style.Bold(options[helper_index]) + " " + str(options)
        sys.stdout.write("\033[K")
        print(to_print, end="\r")

    sys.stdout.write("\033[K")
    print(prefix + A().pretty_print(print_command) + "\n")

    # app.command_history.append(print_command)

    return print_command


def command_runner(app, command, args):
    if command[0] == '':  # log(helper)
        print("Pleas enter a command or help for mor information")

    elif command[0].lower() == '_hr':
        if len(command) == 2:
            if input(f"Do you to hot-reloade {'alle mods' if len(command) <= 1 else command[1]}? (y/n): ") in \
                ["y", "yes", "Y"]:

                if command[1] in app.MOD_LIST.keys():
                    app.reset()
                    try:
                        app.remove_mod(command[1])
                    except Exception as e:
                        print(Style.RED(f"Error removing module {command[1]}\nERROR:\n{e}"))

                    try:
                        app.save_load(command[1])
                    except Exception as e:
                        print(Style.RED(f"Error adding module {command[1]}\nERROR:\n{e}"))
                elif command[1] == "-x":
                    app.reset()
                    app.remove_all_modules()
                    while 1:
                        try:
                            com = " ".join(sys.orig_argv)
                        except AttributeError:
                            com = "python3 "
                            com += " ".join(sys.argv)
                        os.system(com)
                        print("Restarting..")
                        exit(0)
                else:
                    print(f"Module not found {command[1]} |  is case sensitive")
        else:
            app.reset()
            app.remove_all_modules()
            app.load_all_mods_in_file()

    elif command[0].lower() == 'app-info':
        app.print_functions()
        print(f"{app.id = }\n\n"
              f"\n{app.debug = }")
        print(f"PREFIX={app.PREFIX}"
              f"\nMACRO={A().pretty_print(app.MACRO[:7])}"
              f"\nMODS={A().pretty_print(app.MACRO[7:])}"
              f"\nSUPER_SET={A().pretty_print(app.SUPER_SET)}")

    elif command[0].lower() == "exit" or command[0].lower() == 'e':  # builtin events(exit)
        app.exit()

    elif command[0].lower() == "help":  # logs(event(helper))
        n = command[1] if len(command) > 2 else ''
        # app.help(n)

    elif command[0].lower() == 'load-mod':  # builtin events(event(cloudM(_)->event(Build)))
        if len(command) == 2:
            print(app.save_load(command[1]))
        else:
            p = "_dev" if app.dev_modi else ""

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

            res = list(filter(do_helper, os.listdir(f"./mods{p}/")))
            for mod_name in res:
                mod_name_refracted = mod_name.replace(".py", '')
                print(f"Mod name : {mod_name_refracted}")
                A().SUPER_SET += [mod_name_refracted]
            print()

    elif command[0] == '..':
        app.reset()

    elif command[0] == 'cls':
        if system() == "Windows":
            os.system("cls")
        else:
            os.system("clear")

    elif command[0] == 'mode':
        help_ = ['mode:live', 'mode:debug', 'mode', 'mode:stuf', 'app-info']
        A().SUPER_SET += help_
        A().MACRO += help_
        print(f"{'debug' if app.debug else 'live'} \n{app.debug=}\n{app.id=}\n")

    elif command[0] == 'mode:live':
        app.debug = False
        app.dev_modi = False

    elif command[0] == 'mode:debug':
        app.debug = True
        app.dev_modi = True

    elif command[0] == 'run-i':

        if len(command) > 1:
            app.run_runnable(command[1])
        else:
            print(app.runnable.keys())

    elif command[0].lower() in app.functions:
        if len(command) > 1:
            if command[1].lower() in A().SUPER_SET:
                app.run_any(command[0], command[1], False, True, command[1:])

    else:  # error(->)
        print(Style.YELLOW("[-] Unknown command:") + A().pretty_print(command))


def run(app: App, *args):
    while app.alive:
        print("", end="" + "->>\r")
        command = user_input(app)
        commands = []
        for com in command:
            commands.append(com.strip())
        command_runner(app, command, args)

        #
        # self.print(f"Function Exec coed: {formatted_result.info.exec_code}"
        #            f"\nInfo's: {formatted_result.info.help_text}"
        #            f"\nData: {formatted_result.result.data}")


DEFAULT_HELPER = {"help": [["Information", "version : 0.1.2", "color : GREEN", "syntax : help (in scope)",
                            "help is available in all subsets"]], "load-mod": [
    ["Information", "version : 0.1.0", "color : BLUE", "syntax : load-mod[filename]", "file must be in mods folder "]],
                  "exit": [["Information", "version : 0.1.0", "color : RED", "syntax : exit",
                            "The only way to exit in TOOL BOX"]],
                  "..": [["Information", "version : 0.1.0", "color : MAGENTA", "syntax : ..", "Brings u Back to Main"]],
                  "logs": [["Information", "version : 0.1.0", "color : MAGENTA", "syntax : LOGS", "show logs"]],
                  "_hr": [["Information", "version : ----", "Hotreload all mods"]],
                  "cls": [["Information", "version : ----", "Clear Screen"]],
                  "mode": [["Information", "version : ----", "go in monit mode"]],
                  "app-info": [["Information", "version : ----", "app - status - info"]],
                  "mode:live": [["Test Function", "version : ----", "\x1b[31mCode can no loger crash\x1b[0m"]],
                  "mode:debug": [["Test Function", "version : ----", "\x1b[31mCode can crash\x1b[0m"]],
                  "mode:stuf": [["Test Function", "version : ----", "mmute mods on loding and prossesig\x1b[0m"]]}
DEFAULT_MACRO = ["help", "load-mod", "exit", "_hr", "..", "cls", "mode"]
DEFAULT_MACRO_color = {"help": "GREEN", "load-mod": "BLUE", "exit": "RED", "monit": "YELLOW", "..": "MAGENTA",
                       "logs": "MAGENTA", "cls": "WHITE"}


class A(metaclass=Singleton):

    def __init__(self):

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

        self.HELPER = DEFAULT_HELPER
        self.MACRO = DEFAULT_MACRO
        self.MACRO_color = DEFAULT_MACRO_color

        self.SUPER_SET = []
        self.command_history = []
        # self.config_fh.add_to_save_file_handler(self.keys["comm-his"], str(self.command_history))

    def save_init_mod(self, name, mod):  #
        color = mod.color if mod.color else "WHITE"
        self.MACRO.append(name.lower())
        self.MACRO_color[name.lower()] = color
        self.HELPER[name.lower()] = mod.tools["all"]

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
        print(highlight(json_str, JsonLexer(), TerminalFormatter()))

    def autocompletion(self, command):
        options = []
        if command == "":
            return options
        for macro in self.MACRO + self.SUPER_SET:
            if macro.startswith(command.lower()):
                options.append(macro)
        return options

    def command_viewer(self, mod_command):
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
