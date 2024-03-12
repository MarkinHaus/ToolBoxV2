"""Console script for toolboxv2."""
# Import default Pages
import sys
import argparse
import time
from functools import wraps
from platform import system, node
# Import public Pages
from toolboxv2 import App, MainTool, runnable_dict as runnable_dict_func, Style, __version__, Spinner
from toolboxv2.utils import show_console
from toolboxv2.utils import get_app
from toolboxv2.utils.daemon import DaemonApp
from toolboxv2.utils.proxy import ProxyApp
from toolboxv2.utils.system import override_main_app


DEFAULT_MODI = "minicli"

try:
    import hmr

    HOT_RELOADER = True
except ImportError:
    HOT_RELOADER = False

try:
    import cProfile
    import pstats
    import io


    def profile_execute_all_functions(app=None, m_query='', f_query=''):
        # Erstellen Sie eine Instanz Ihrer Klasse
        instance = app if app is not None else get_app(from_="Profiler")

        # Erstellen eines Profilers
        profiler = cProfile.Profile()

        def timeit(func_):
            @wraps(func_)
            def timeit_wrapper(*args, **kwargs):
                profiler.enable()
                start_time = time.perf_counter()
                result = func_(*args, **kwargs)
                end_time = time.perf_counter()
                profiler.disable()
                total_time_ = end_time - start_time
                print(f'Function {func_.__name__}{args} {kwargs} Took {total_time_:.4f} seconds')
                return result

            return timeit_wrapper

        items = list(instance.functions.items()).copy()
        for module_name, functions in items:
            if not module_name.startswith(m_query):
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
                instance.functions[module_name][function_name]['func'] = timeit(function_data.get('func'))

                # Starten des Profilers und Ausführen der Funktion
        instance.execute_all_functions(m_query=m_query, f_query=f_query)

        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        print("\n================================" * 12)
        s = io.StringIO()
        sortby = 'time'  # Sortierung nach der Gesamtzeit, die in jeder Funktion verbracht wird

        # Erstellen eines pstats-Objekts und Ausgabe der Top-Funktionen
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()

        # Ausgabe der Ergebnisse
        print(s.getvalue())

        # Erstellen eines Streams für die Profilergebnisse

except ImportError as e:
    profile_execute_all_functions = lambda *args: print(args);
    raise ValueError(f"Failed to import function for profiling")

try:
    from toolboxv2.utils.system.tb_logger import edit_log_files, loggerNameOfToolboxv2, unstyle_log_files
except ModuleNotFoundError:
    from toolboxv2.utils.system.tb_logger import edit_log_files, loggerNameOfToolboxv2, unstyle_log_files

import os
import subprocess


def start(pidname, args):
    caller = args[0]
    args = args[1:]
    args = ["-bgr" if arg == "-bg" else arg for arg in args]

    if '-m' not in args or args[args.index('-m') + 1] == "toolboxv2":
        args += ["-m", "bg"]
    if caller.endswith('toolboxv2'):
        args = ["toolboxv2"] + args
    else:
        args = [sys.executable, "-m", "toolboxv2"] + args
    if system() == "Windows":
        DETACHED_PROCESS = 0x00000008
        p = subprocess.Popen(args, creationflags=DETACHED_PROCESS)
    else:  # sys.executable, "-m",
        p = subprocess.Popen(args, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    pdi = p.pid
    get_app().sprint(f"Service {pidname} started")


def stop(pidfile, pidname):
    try:
        with open(pidfile, "r") as f:
            procID = f.readline().strip()
    except IOError:
        print("Process file does not exist")
        return

    if procID:
        if system() == "Windows":
            subprocess.Popen(['taskkill', '/PID', procID, '/F'])
        else:
            subprocess.Popen(['kill', '-SIGTERM', procID])

        print(f"Service {pidname} {procID} stopped")
        os.remove(pidfile)


def create_service_file(user, group, working_dir):
    service_content = f"""[Unit]
Description=ToolBoxService
After=network.target

[Service]
User={user}
Group={group}
WorkingDirectory={working_dir}
ExecStart=toolboxv2 -bgr -m bg
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
    with open("tb.service", "w") as f:
        f.write(service_content)


def init_service():
    user = input("Enter the user name: ")
    group = input("Enter the group name: ")
    working_dir = get_app().start_dir

    create_service_file(user, group, working_dir)

    subprocess.run(["sudo", "mv", "tb.service", "/etc/systemd/system/"])
    subprocess.run(["sudo", "systemctl", "daemon-reload"])


def manage_service(action):
    subprocess.run(["sudo", "systemctl", action, "tb.service"])


def show_service_status():
    subprocess.run(["sudo", "systemctl", "status", "tb.service"])


def uninstall_service():
    subprocess.run(["sudo", "systemctl", "disable", "tb.service"])
    subprocess.run(["sudo", "systemctl", "stop", "tb.service"])
    subprocess.run(["sudo", "rm", "/etc/systemd/system/tb.service"])
    subprocess.run(["sudo", "systemctl", "daemon-reload"])


def setup_service_windows():
    path = "C:/ProgramData/Microsoft/Windows/Start Menu/Programs/Startup"
    print("Select mode:")
    print("1. Init (first-time setup)")
    print("2. Uninstall")
    print("3. Show window")
    print("4. hide window")
    print("0. Exit")

    mode = input("Enter the mode number: ").strip()

    if not os.path.exists(path):
        print("pleas press win + r and enter")
        print("1. for system -> shell:common startup")
        print("2. for user -> shell:startup")
        path = input("Enter the path that opened: ")

    if mode == "1":
        if os.path.exists(path + '/tb_start.bat'):
            os.remove(path + '/tb_start.bat')
        with open(path + '/tb_start.bat', "a") as f:
            f.write(
                f"""{sys.executable} -m toolboxv2 -bg"""
            )
        print(f"Init Service in {path}")
        print(f"run toolboxv2 -bg to start the service")
    elif mode == "3":
        get_app().show_console()
    elif mode == "4":
        get_app().hide_console()
    elif mode == "0":
        pass
    elif mode == "2":
        os.remove(path + '/tb_start.bat')
        print(f"Removed Service from {path}")
    else:
        setup_service_windows()


def setup_service_linux():
    print("Select mode:")
    print("1. Init (first-time setup)")
    print("2. Start / Stop / Restart")
    print("3. Status")
    print("4. Uninstall")

    print("5. Show window")
    print("6. hide window")

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
    elif mode == 5:
        get_app().show_console()
    elif mode == 6:
        get_app().hide_console()
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
                        default=DEFAULT_MODI)

    parser.add_argument("--kill", help="Kill current local tb instance", default=False,
                        action="store_true")

    parser.add_argument("--remote", help=f"Open a remote toolbox session", default=False,
                        action="store_true")
    parser.add_argument("--remote-direct-key", type=str or None, help=f"Open a remote toolbox session", default=None)

    parser.add_argument("-bg", "--background-application", help="Start an interface as background application",
                        default=False,
                        action="store_true")
    parser.add_argument("-bgr", "--background-application-runner", help="Start an interface as background application",
                        default=False,
                        action="store_true")
    parser.add_argument("-fg", "--proxy-application", help="Start an interface as proxy application",
                        default=True,
                        action="store_false")

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

    parser.add_argument("-sfe", "--save-function-enums-in-file",
                        help="run with -l to gather to generate all_function_enums.py files",
                        action="store_true")

    parser.add_argument("-hr", "--hot-reload",
                        help="run -hr automatically reload the toolboxv2 module on changes good for development and"
                             " for long running instances in save environments",
                        action="store_true")

    # parser.add_argument("--mods-folder",
    #                     help="specify loading package folder",
    #                     type=str,
    #                     default="toolboxv2.mods.")

    parser.add_argument("--debug",
                        help="start in debug mode",
                        action="store_true")

    parser.add_argument("--delete-config-all",
                        help="start in debug mode",
                        action="store_true")

    parser.add_argument("--delete-data-all",
                        help="start in debug mode",
                        action="store_true")

    parser.add_argument("--delete-config",
                        help="start in debug mode",
                        action="store_true")

    parser.add_argument("--delete-data",
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
    # abspath = os.path.dirname(os.path.abspath(__file__))
    abspath = os.path.dirname(os.path.abspath(__file__))

    identification = args.name + '-' + node() + '\\'
    if args.mm:
        identification = "MainNode\\"

    data_folder = abspath + '\\.data\\'
    config_folder = abspath + '\\.config\\'
    info_folder = abspath + '\\.info\\'

    os.makedirs(info_folder, exist_ok=True)

    app_config_file = config_folder + identification
    app_data_folder = data_folder + identification

    if args.delete_config_all:
        os.remove(config_folder)
    if args.delete_data_all:
        os.remove(data_folder)
    if args.delete_config:
        os.remove(app_config_file)
    if args.delete_data:
        os.remove(app_data_folder)

    if args.test:
        test_path = os.path.dirname(os.path.abspath(__file__))
        if system() == "Windows":
            test_path = test_path.replace("toolboxv2\\toolboxv2", "toolboxv2") + "\\tests\\test_mods"
        else:
            test_path = test_path.replace("ToolBoxV2/toolboxv2", "ToolBoxV2") + "/tests/test_mods"
        print(f"Testing in {test_path}")

        if os.system(f"python -m unittest discover -s {test_path}") != 0:
            os.system(f"python3 -m unittest discover -s {test_path}")
        return 1

    app_pid = str(os.getpid())

    pid_file = f"{info_folder}{args.modi}-{args.name}.pid"

    tb_app = get_app(from_="InitialStartUp", name=args.name, args=args, app_con=App)
    daemon_app = None
    if args.background_application_runner:
        daemon_app = DaemonApp(tb_app, args.host, args.port if args.port != 8000 else 6587, t=args.modi != 'bg')
        if not args.debug:
            show_console(False)
        tb_app.daemon_app = daemon_app
        with open(pid_file + '-app.pid', 'w') as f:
            f.write(app_pid)
        args.proxy_application = False
    elif args.background_application:
        if not args.kill and not args.hot_reload:
            start(args.name, sys.argv)
        elif args.hot_reload:
            try:
                _ = ProxyApp(tb_app, args.host if args.host != "0.0.0.0" else "localhost",
                             args.port if args.port != 8000 else 6587, timeout=6)
                if _.exit_main() != "No data look later":
                    stop(pid_file + '-app.pid', args.name)
            except Exception:
                stop(pid_file + '-app.pid', args.name)
            time.sleep(2)
            start(args.name, sys.argv)
        else:
            if '-m ' not in sys.argv:
                pid_file = f"{info_folder}bg-{args.name}.pid"
            try:
                _ = ProxyApp(tb_app, args.host if args.host != "0.0.0.0" else "localhost",
                             args.port if args.port != 8000 else 6587, timeout=6)
                if _.exit_main() != "No data look later":
                    stop(pid_file + '-app.pid', args.name)
            except Exception:
                stop(pid_file + '-app.pid', args.name)
    elif args.proxy_application:
        tb_app = override_main_app(ProxyApp(tb_app, args.host if args.host != "0.0.0.0" else "localhost",
                                            args.port if args.port != 8000 else 6587))

        tb_app.verify()
        if args.debug:
            tb_app.show_console()

    # tb_app.load_all_mods_in_file()
    # tb_app.save_registry_as_enums("utils", "all_functions_enums.py")

    if args.lm:
        edit_logs()
        tb_app.exit()
        exit(0)

    if args.sm:
        if tb_app.system_flag == "Linux":
            setup_service_linux()
        if tb_app.system_flag == "Windows":
            setup_service_windows()
        tb_app.exit()
        exit(0)

    if args.load_all_mod_in_files or args.save_function_enums_in_file or args.get_version or args.profiler or args.background_application_runner:
        if not args.proxy_application:
            tb_app.load_all_mods_in_file()
        if args.save_function_enums_in_file:
            tb_app.save_registry_as_enums("utils", "all_functions_enums.py")
            tb_app.alive = False
            tb_app.exit()
            return 0
        if args.debug:
            tb_app.print_functions()
        if args.get_version:
            print(f"\n{' Version ':-^45}\n\n{Style.Bold(Style.CYAN(Style.ITALIC('RE')))+Style.ITALIC('Simple')+'ToolBox':<35}:{__version__:^10}\n")
            for mod_name in tb_app.functions:
                if isinstance(tb_app.functions[mod_name].get("app_instance"), MainTool):
                    print(f"{mod_name:^35}:{tb_app.functions[mod_name]['app_instance'].version:^10}")
                else:
                    v = tb_app.functions[mod_name].get(list(tb_app.functions[mod_name].keys())[0]).get("version",
                                                                                                       "unknown (functions only)").replace(f"{__version__}:", '')
                    print(f"{mod_name:^35}:{v:^10}")
            print("\n")
            tb_app.alive = False
            tb_app.exit()
            return 0

    if args.profiler:
        profile_execute_all_functions(tb_app)
        tb_app.alive = False
        tb_app.exit()
        return 0

    if not args.kill and not args.docker and tb_app.alive and not args.background_application:

        tb_app.save_autocompletion_dict()
        with open(pid_file, "w") as f:
            f.write(app_pid)

        if not args.proxy_application:
            runnable_dict = runnable_dict_func()
            tb_app.set_runnable(runnable_dict)
            if args.modi in runnable_dict.keys():
                pass
            else:
                raise ValueError(
                    f"Modi : [{args.modi}] not found on device installed modi : {list(runnable_dict.keys())}")
            # open(f"./config/{args.modi}.pid", "w").write(app_pid)
            tb_app.run_runnable(args.modi)
        elif 'cli' in args.modi:
            runnable_dict = runnable_dict_func('cli')
            tb_app.set_runnable(runnable_dict)
            tb_app.run_runnable(args.modi)
        else:
            tb_app.rrun_runnable(args.modi)

    elif args.docker:

        runnable_dict = runnable_dict_func('docker')

        if 'docker' not in runnable_dict.keys():
            print("No docker")
            return

        runnable_dict['docker'](tb_app, args)

    elif args.kill and not args.background_application:
        if not os.path.exists(pid_file):
            print("You must first run the mode")
        else:
            with open(pid_file, "r") as f:
                app_pid = f.read()
            print(f"Exit app {app_pid}")
            if system() == "Windows":
                os.system(f"taskkill /pid {app_pid} /F")
            else:
                os.system(f"kill -9 {app_pid}")

    if args.proxy_application and args.debug:
        tb_app.hide_console()

    if tb_app.alive:
        tb_app.exit()
        return 0

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
