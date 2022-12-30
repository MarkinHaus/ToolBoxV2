"""Console script for toolboxv2. min dep readchar Style"""

# Import default Pages
import sys
import os
from platform import system
from .Style import Style

# Import public Pages
import readchar
from .toolbox import App

session_history = [[]]


def user_input(app: App):
    get_input = True
    command = ""
    print_command = []
    helper = ""
    helper_index = 0
    options = []
    sh_index = 0

    while get_input:

        key = readchar.readkey()

        if key == b'\x05' or key == '\x05':
            print('\033[?25h', end="")
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

        elif key == b'\x08' or key == '\x7f':
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

        options = list(set(app.autocompletion(command)))

        if helper_index > len(options) - 1:
            helper_index = 0

        helper = ""
        do = len(options) > 0
        if do:
            helper = options[helper_index][len(command):].lower()

        to_print = app.PREFIX + app.pretty_print(print_command + [command + Style.Underline(Style.Bold(helper))])
        if do:
            to_print += " | " + Style.Bold(options[helper_index]) + " " + str(options)
        sys.stdout.write("\033[K")
        print(to_print, end="\r")

    sys.stdout.write("\033[K")
    print(app.PREFIX + app.pretty_print(print_command) + "\n")
    session_history.append(print_command)
    return print_command


def run_cli(app: App):
    mode = "live"
    while app.alive:
        print("", end="" + "->>\r")
        command = user_input(app)

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
                img = app.MOD_LIST["WELCOME"].tools["printT"]
                img()

        elif command[0].lower() == 'logs':
            app.logs()

        elif command[0].lower() == 'app-info':
            print(f"{app.id = }\n{app.stuf_load = }\n{app.mlm = }\n{app.auto_save = }"
                  f"\n{app.AC_MOD = }\n{app.debug = }")

        elif command[0].upper() == "EXIT":  # builtin events(exit)
            if input("Do you want to exit? (y/n): ") in ["y", "yes", "Y"]:
                app.save_exit()
                app.exit()

        elif command[0].lower() == "help":  # logs(event(helper))
            n = command[1] if len(command) > 2 else ''
            app.help(n)

        elif command[0].upper() == 'LOAD-MOD':  # builtin events(event(cloudM(_)->event(Build)))
            if len(command) == 2:
                app.save_load(command[1])
                app.new_ac_mod(command[1])
            else:
                p = "_dev" if app.debug else ""
                res = os.listdir(f"./mods{p}/")
                app.SUPER_SET += res
                app.MACRO += res
                if system() == "Windows":
                    os.system(f"dir .\mods{p}")
                else:
                    os.system(f"ls ./mods{p}")

        elif command[0] == '..':
            app.reset()

        elif command[0] == 'cls':
            if system() == "Windows":
                os.system("cls")
            else:
                os.system("clear")

        elif command[0] == 'mode':
            help_ = ['mode:live', 'mode:debug', 'mode', 'mode:stuf', 'app-info']
            app.SUPER_SET += help_
            app.MACRO += help_
            print(f"{mode=} \n{app.debug=}\n{app.id=}\n{app.stuf_load=}")

        elif command[0] == 'mode:live':
            mode = 'live'
            app.debug = False

        elif command[0] == 'mode:debug':
            mode = 'debug'
            app.debug = True

        elif command[0] == 'mode:stuf':
            app.stuf_load = not app.stuf_load

        elif command[0].upper() in app.MOD_LIST.keys():
            app.new_ac_mod(command[0])

            if len(command) > 1:
                if command[1].upper() in app.SUPER_SET:
                    app.run_function(command[1], command[1:])

        elif app.AC_MOD:  # builtin events(AC_MOD(MOD))
            if command[0].upper() in app.SUPER_SET:
                app.run_function(command[0], command)
            else:
                print(Style.RED("function could not be found"))

        else:  # error(->)
            print(Style.YELLOW("[-] Unknown command:") + app.pretty_print(command))
