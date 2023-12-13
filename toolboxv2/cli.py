"""Console script for toolboxv2."""
# Import default Pages
import sys
import argparse
from platform import system

# Import public Pages
from toolboxv2 import App, MainTool, runnable_dict as runnable_dict_func
from toolboxv2.utils.toolbox import get_app
import ctypes


def show_console(show=True):
    """Brings up the Console Window."""
    if show:
        # Show console
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 4)
    else:
        # Hide console
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)


try:
    from toolboxv2.utils.tb_logger import edit_log_files, loggerNameOfToolboxv2, unstyle_log_files
except ModuleNotFoundError:
    from .utils.tb_logger import edit_log_files, loggerNameOfToolboxv2, unstyle_log_files

import os
import subprocess


def create_service_file(user, group, working_dir):
    service_content = f"""[Unit]
Description=My Python App
After=network.target

[Service]
User={user}
Group={group}
WorkingDirectory={working_dir}
ExecStart=toolboxv2 -m app -n app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
    with open("myapp.service", "w") as f:
        f.write(service_content)


def init_service():
    user = input("Enter the user name: ")
    group = input("Enter the group name: ")
    working_dir = input("Enter the working directory path (/path/to/your/app): ")

    create_service_file(user, group, working_dir)

    subprocess.run(["sudo", "mv", "myapp.service", "/etc/systemd/system/"])
    subprocess.run(["sudo", "systemctl", "daemon-reload"])


def manage_service(action):
    subprocess.run(["sudo", "systemctl", action, "myapp.service"])


def show_service_status():
    subprocess.run(["sudo", "systemctl", "status", "myapp.service"])


def uninstall_service():
    subprocess.run(["sudo", "systemctl", "disable", "myapp.service"])
    subprocess.run(["sudo", "systemctl", "stop", "myapp.service"])
    subprocess.run(["sudo", "rm", "/etc/systemd/system/myapp.service"])
    subprocess.run(["sudo", "systemctl", "daemon-reload"])


def setup_service_windows():
    path = "C:/ProgramData/Microsoft/Windows/Start Menu/Programs/Startup"
    print("Select mode:")
    print("1. Init (first-time setup)")
    print("2. Uninstall")
    mode = int(input("Enter the mode number: "))

    if not os.path.exists(path):
        print("pleas press win + r and enter")
        print("1. for system -> shell:common startup")
        print("2. for user -> shell:startup")
        path = input("Enter the path: ")

    if mode == 1:
        with open(path + '/tb_start.bat', "a") as f:
            f.write(
                """toolboxV2 -m demon -bg -n AutoStart"""
            )
    else:
        os.remove(path + '/tb_start.pyw')


def setup_service_linux():
    print("Select mode:")
    print("1. Init (first-time setup)")
    print("2. Start / Stop / Restart")
    print("3. Status")
    print("4. Uninstall")

    mode = int(input("Enter the mode number: "))

    if mode == 1:
        init_service()
    elif mode == 2:
        action = input("Enter 'start', 'stop', or 'restart': ")
        manage_service(action)
    elif mode == 3:
        show_service_status()
    elif mode == 4:
        uninstall_service()
    else:
        print("Invalid mode")


def parse_args():
    parser = argparse.ArgumentParser(description="Welcome to the ToolBox cli")

    parser.add_argument("-init",
                        help="ToolBoxV2 init (name) -> default : -n name = main", type=str or None, default=None)

    parser.add_argument('-f', '--init-file',
                        type=str,
                        default="init.config",
                        help="optional init flag init from config file or url")

    parser.add_argument("-v", "--get-version",
                        help="get version of ToolBox and all mods with -l",
                        action="store_true")

    parser.add_argument("--mm",
                        help="all Main all files ar saved in Main Node",
                        action="store_true")

    parser.add_argument("--sm", help=f"Service Manager for {system()} manage auto start and auto restart",
                        default=False,
                        action="store_true")
    parser.add_argument("--lm", help=f"Log Manager remove and edit log files", default=False,
                        action="store_true")

    parser.add_argument("-m", "--modi",
                        type=str,
                        help="Start a ToolBox interface default build in cli",
                        default="cli")

    parser.add_argument("--kill", help="Kill current local tb instance", default=False,
                        action="store_true")

    parser.add_argument("--remote", help=f"Open a remote toolbox session", default=False,
                        action="store_true")
    parser.add_argument("--remote-direct-key", type=str or None, help=f"Open a remote toolbox session", default=None)

    parser.add_argument("-bg", "--background-application", help="Start an interface as background application",
                        default=False,
                        action="store_true")
    parser.add_argument("--docker", help="start the toolbox in docker (in remote mode is no local docker engin "
                                         "required)", default=False,
                        action="store_true")

    parser.add_argument("-i", "--install", help="Install a mod or interface via name", type=str or None, default=None)
    parser.add_argument("-r", "--remove", help="Uninstall a mod or interface via name", type=str or None, default=None)
    parser.add_argument("-u", "--update", help="Update a mod or interface via name", type=str or None, default=None)

    parser.add_argument('-mvn', '--mod-version-name',
                        metavar="name",
                        type=str,
                        help="Name of mod",
                        default="mainTool")

    parser.add_argument('-n', '--name',
                        metavar="name",
                        type=str,
                        help="Specify an id for the ToolBox instance",
                        default="main")

    parser.add_argument("-p", "--port",
                        metavar="port",
                        type=int,
                        help="Specify a port for interface",
                        default=8000)  # 1268945

    parser.add_argument("-w", "--host",
                        metavar="host",
                        type=str,
                        help="Specify a host for interface",
                        default="0.0.0.0")

    parser.add_argument("-l", "--load-all-mod-in-files",
                        help="load all modules in mod file",
                        action="store_true")

    # parser.add_argument("--mods-folder",
    #                     help="specify loading package folder",
    #                     type=str,
    #                     default="toolboxv2.mods.")

    parser.add_argument("--debug",
                        help="start in debug mode",
                        action="store_true")

    parser.add_argument("--test",
                        help="run all tests",
                        action="store_true")

    parser.add_argument("--profiler",
                        help="run all measurements",
                        action="store_true")

    return parser.parse_args()


def edit_logs():
    name = input(f"Name of logger \ndefault {loggerNameOfToolboxv2}\n:")
    name = name if name else loggerNameOfToolboxv2

    def date_in_format(_date):
        ymd = _date.split('-')
        if len(ymd) != 3:
            print("Not enough segments")
            return False
        if len(ymd[1]) != 2:
            print("incorrect format")
            return False
        if len(ymd[2]) != 2:
            print("incorrect format")
            return False

        return True

    def level_in_format(_level):

        if _level in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']:
            _level = [50, 40, 30, 20, 10, 0][['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'].index(_level)]
            return True, _level
        try:
            _level = int(_level)
        except ValueError:
            print("incorrect format pleas enter integer 50, 40, 30, 20, 10, 0")
            return False, -1
        return _level in [50, 40, 30, 20, 10, 0], _level

    date = input(f"Date of log format : YYYY-MM-DD replace M||D with xx for multiple editing\n:")

    while not date_in_format(date):
        date = input("Date of log format : YYYY-MM-DD :")

    level = input(
        f"Level : {list(zip(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'], [50, 40, 30, 20, 10, 0]))}"
        f" : enter number\n:")

    while not level_in_format(level)[0]:
        level = input("Level : ")

    level = level_in_format(level)[1]

    do = input("Do function : default remove (r) or uncoler (uc)")
    if do == 'uc':
        edit_log_files(name=name, date=date, level=level, n=0, do=unstyle_log_files)
    else:
        edit_log_files(name=name, date=date, level=level, n=0)


def main():
    """Console script for toolboxv2."""
    args = parse_args()

    def dev_helper():
        dev_args = f"streamlit run streamlit_web_dev_tools.py ata:{args.name}.config:{args.name}" \
                   f" {(f'--server.port={args.port} --server.address={args.host}' if args.host != '0.0.0.0' else '')}"
        os.system(dev_args)

    # (
    # init=None,
    # init_file='init.config',
    # get_version=False,
    # mm=False,
    # sm=False,
    # lm=False,
    # modi='cli',
    # kill=False,
    # remote=False,
    # remote_direct_key=None,
    # background_application=False,
    # docker=False,
    # install=None,
    # remove=None,
    # update=None,
    # mod_version_name='mainTool',
    # name='main',
    # port=8000,
    # host='0.0.0.0',
    # load_all_mod_in_files=False,
    # mods_folder='toolboxv2.mods.',
    # debug=None,
    # test=None,
    # profiler=None
    # )

    # print(args)

    app_pid = str(os.getpid())

    tb_app = get_app(from_="InitialStartUp", name=args.name, args=args)

    # tb_app.load_all_mods_in_file()
    # tb_app.save_registry_as_enums("utils", "all_functions_enums.py")

    pid_file = f"{tb_app.start_dir}/{tb_app.config_fh.file_handler_file_prefix}{args.modi}-{args.name}.pid"

    if args.background_application and system() == "Windows":
        show_console(False)

    if args.lm:
        edit_logs()
        tb_app.save_exit()
        tb_app.exit()
        exit(0)

    if args.sm:
        print("1. App")
        setup = input("Set up for :")
        if tb_app.system_flag == "Linux":
            setup_service_linux()
        if tb_app.system_flag == "Windows":
            setup_service_windows()

    if args.load_all_mod_in_files:
        tb_app.load_all_mods_in_file()
        if args.debug:
            tb_app.print_functions()
        if args.get_version:

            for mod_name in tb_app.functions:
                if isinstance(tb_app.functions[mod_name]["app_instance"], MainTool):
                    print(f"{mod_name} : {tb_app.functions[mod_name]['app_instance'].version}")

    if not args.kill and not args.docker:

        with open(f"{tb_app.start_dir}/{tb_app.config_fh.file_handler_file_prefix}/{args.modi}-{args.name}.pid",
                  "w") as f:
            f.write(app_pid)

        runnable_dict = runnable_dict_func()

        if args.modi.lower() in runnable_dict.keys():
            tb_app.set_runnable(runnable_dict)
            # open(f"./config/{args.modi}.pid", "w").write(app_pid)
            runnable_dict[args.modi.lower()](tb_app, args)
        else:
            print(f"Modi : [{args.modi}] not found on device installed modi : {runnable_dict.keys()}")

    elif args.docker:

        runnable_dict = runnable_dict_func('docker')

        if 'docker' not in runnable_dict.keys():
            print("No docker")
            return

        runnable_dict['docker'](tb_app, args)

    elif args.kill:
        if not os.path.exists(pid_file):
            print("You must first run the mode")
        else:
            with open(pid_file,
                      "r") as f:
                app_pid = f.read()
            print(f"Exit app {app_pid}")
            if system() == "Windows":
                os.system(f"taskkill /pid {app_pid} /F")
            else:
                os.system(f"kill -9 {app_pid}")

            # if args.docker:
            #     tb_app.run_any()

    if tb_app.alive:
        tb_app.save_exit()
        tb_app.exit()

    if os.path.exists(pid_file):
        os.remove(pid_file)

    # print(
    #    f"\n\nPython-loc: {init_args[0]}\nCli-loc: {init_args[1]}\nargs: {tb_app.pretty_print(init_args[2:])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    # init main : ToolBoxV2 -init main -f init.config
    # Exit
    # y
    # ToolBoxV2 -l || ToolBoxV2 -n main -l
