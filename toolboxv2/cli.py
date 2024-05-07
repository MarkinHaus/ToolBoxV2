"""Console script for toolboxv2."""
# Import default Pages
import sys
import argparse
import time
import asyncio
from functools import wraps
from platform import system, node

from yaml import safe_load

from toolboxv2.runabel import runnable_dict as runnable_dict_func
from toolboxv2.utils.system.main_tool import MainTool
from toolboxv2.utils.extras.Style import Style, Spinner
from toolboxv2.utils.system.session import Session
from toolboxv2.utils.system import all_functions_enums as tbef
# Import public Pages
from toolboxv2.utils.toolbox import App

from toolboxv2.utils import show_console
from toolboxv2.utils import get_app
from toolboxv2.utils.daemon import DaemonApp
from toolboxv2.utils.proxy import ProxyApp
from toolboxv2.utils.system import override_main_app, get_state_from_app, CallingObject

DEFAULT_MODI = "cli"

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


def create_service_file(user, group, working_dir, runner):
    service_content = f"""[Unit]
Description=ToolBoxService
After=network.target

[Service]
User={user}
Group={group}
WorkingDirectory={working_dir}
ExecStart=toolboxv2 -bgr -m {runner}
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
    runner = "bg"
    if runner_ := input("enter a runner default bg: ").strip():
        runner = runner_
    working_dir = get_app().start_dir

    create_service_file(user, group, working_dir, runner)

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
        runner = "bg"
        if runner_ := input("enter a runner default bg: ").strip():
            runner = runner_
        if os.path.exists(path + '/tb_start.bat'):
            os.remove(path + '/tb_start.bat')
        with open(path + '/tb_start.bat', "a") as f:
            f.write(
                f"""{sys.executable} -m toolboxv2 -bg -m {runner}"""
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

    parser.add_argument("-v", "--get-version",
                        help="get version of ToolBox and all mods with -l",
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

    parser.add_argument("-bg", "--background-application", help="Start an interface in the background",
                        default=False,
                        action="store_true")

    parser.add_argument("-bgr", "--background-application-runner",
                        help="The Flag to run the background runner in the current terminal/process",
                        default=False,
                        action="store_true")

    parser.add_argument("-fg", "--live-application",
                        help="Start an interface as live application not using a background runner",
                        default=True,
                        action="store_false")

    parser.add_argument("--docker", help="start the toolbox in docker (in remote mode is no local docker engin "
                                         "required)", default=False,
                        action="store_true")

    parser.add_argument("-i", "--install", help="Install a mod or interface via name", type=str or None, default=None)
    parser.add_argument("-r", "--remove", help="Uninstall a mod or interface via name", type=str or None, default=None)
    parser.add_argument("-u", "--update", help="Update a mod or interface via name", type=str or None, default=None)

    parser.add_argument('-n', '--name',
                        metavar="name",
                        type=str,
                        help="Specify an id for the ToolBox instance",
                        default="main")

    parser.add_argument("-p", "--port",
                        metavar="port",
                        type=int,
                        help="Specify a port for interface",
                        default=5000)  # 1268945

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

    # parser.add_argument("--mods-folder",
    #                     help="specify loading package folder",
    #                     type=str,
    #                     default="toolboxv2.mods.")

    parser.add_argument("--debug",
                        help="start in debug mode",
                        action="store_true")

    parser.add_argument("--remote",
                        help="start in remote mode",
                        action="store_true")

    parser.add_argument("--delete-config-all",
                        help="!!! DANGER !!! deletes all config files. incoming data loss",
                        action="store_true")

    parser.add_argument("--delete-data-all",
                        help="!!! DANGER !!! deletes all data folders. incoming data loss",
                        action="store_true")

    parser.add_argument("--delete-config",
                        help="!! Warning !! deletes named data folders."
                             " incoming data loss. useful if an tb instance is not working properly",
                        action="store_true")

    parser.add_argument("--delete-data",
                        help="!! Warning !! deletes named data folders."
                             " incoming data loss. useful if an tb instance is not working properly",
                        action="store_true")

    parser.add_argument("--test",
                        help="run all tests",
                        action="store_true")

    parser.add_argument("--profiler",
                        help="run all registered functions and make measurements",
                        action="store_true")

    parser.add_argument("-c", "--command", nargs='*', action='append',
                        help="run all registered functions and make measurements")

    parser.add_argument("--sysPrint", action="store_true", default=False,
                        help="activate system prints / verbose output")

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


def run_tests(test_path):
    # Konstruiere den Befehl für den Unittest-Testaufruf
    command = [sys.executable, "-m", "unittest", "discover", "-s", test_path]

    # Führe den Befehl mit subprocess aus
    try:
        result = subprocess.run(command, check=True)
        # Überprüfe den Rückgabewert des Prozesses und gib entsprechend True oder False zurück
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Ausführen der Unittests: {e}")
        return False
    except FileNotFoundError:
        print("Fehler: Python-Interpreter nicht gefunden. Stellen Sie sicher, dass Python installiert ist.")
        return False


async def main(loop=None):
    """Console script for toolboxv2."""
    args = parse_args()

    with open(os.getenv('CONFIG_FILE', f'{os.path.abspath(__file__).replace("cli.py", "")}toolbox.yaml'),
              'r') as config_file:
        _version = safe_load(config_file)
        __version__ = _version.get('main', {}).get('version', '-.-.-')
    # print(args)
    # abspath = os.path.dirname(os.path.abspath(__file__))
    abspath = os.path.dirname(os.path.abspath(__file__))

    identification = args.name + '-' + node() + '\\'

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
            test_path = test_path + "\\tests\\test_mods"
        else:
            test_path = test_path + "/tests/test_mods"
        print(f"Testing in {test_path}")
        if not run_tests(test_path):
            print("Error in tests")
            return 1
        return 0

    app_pid = str(os.getpid())

    pid_file = f"{info_folder}{args.modi}-{args.name}.pid"

    tb_app = get_app(from_="InitialStartUp", name=args.name, args=args, app_con=App)
    if not args.sysPrint and not (args.debug or args.background_application_runner or args.install or args.kill):
        tb_app.print = lambda text, *args, **kwargs: None

    tb_app.loop = loop
    with Spinner("Crating State"):
        await asyncio.to_thread(get_state_from_app, tb_app,
                                os.environ.get("TOOLBOXV2_REMOTE_BASE", "https://simplecore.app"),
                                "https://github.com/MarkinHaus/ToolBoxV2/tree/master/toolboxv2/")
    daemon_app = None
    tb_app.print("OK")
    if args.background_application_runner:
        daemon_app = await DaemonApp(tb_app, args.host, args.port if args.port != 5000 else 6587, t=False)
        if not args.debug:
            show_console(False)
        tb_app.daemon_app = daemon_app
        with open(pid_file + '-app.pid', 'w') as f:
            f.write(app_pid)
        args.live_application = False
    elif args.background_application:
        if not args.kill:
            start(args.name, sys.argv)
        else:
            if '-m ' not in sys.argv:
                pid_file = f"{info_folder}bg-{args.name}.pid"
            try:
                _ = await ProxyApp(tb_app, args.host if args.host != "0.0.0.0" else "localhost",
                                   args.port if args.port != 5000 else 6587, timeout=6)
                await _.verify()
                if _.exit_main() != "No data look later":
                    stop(pid_file + '-app.pid', args.name)
            except Exception:
                stop(pid_file + '-app.pid', args.name)
    elif args.live_application:
        try:
            _ = await ProxyApp(tb_app, args.host if args.host != "0.0.0.0" else "localhost",
                               args.port if args.port != 5000 else 6587)
            time.sleep(1)
            await _.verify()
            time.sleep(1)
            tb_app = override_main_app(_)
            if args.debug:
                await tb_app.show_console()
        except:
            print("Auto starting Starting Local if u know ther is no bg instance use -fg to run in the frond ground")

    # tb_app.load_all_mods_in_file()
    # tb_app.save_registry_as_enums("utils", "all_functions_enums.py")

    if args.install:
        report = False  # tb_app.run_any("CloudM", "install", module_name=args.install)
        if not report:
            if 'n' not in input("Mod not found in local mods_sto install from remote ? (yes,no)"):
                session = Session(tb_app.get_username(), base=os.getenv("MOD_PROVIDER"))
                if not await session.login():
                    mk = input(f"bitte geben sie ihren magik link ein {session.base}/")
                    if '/web/' not in mk:
                        print("Link is not in Valid format")
                        return
                    if not await session.init_log_in_mk_link(mk):
                        print("Link is not in Valid")
                        return
                response = await session.fetch("/api/CloudM/get_latest?module_name=" + args.install, method="GET")
                json_response = await response.json()
                print(json_response)
                print(json_response['data'])

                r2 = await session.fetch(json_response['data'], method="GET")
                json_response = await response.json()
                print(json_response)
                print(json_response['data'])

    if args.lm:
        edit_logs()
        await tb_app.a_exit()
        exit(0)

    if args.sm:
        if tb_app.system_flag == "Linux":
            setup_service_linux()
        if tb_app.system_flag == "Windows":
            setup_service_windows()
        await tb_app.a_exit()
        exit(0)

    if args.load_all_mod_in_files or args.save_function_enums_in_file or args.get_version or args.profiler or args.background_application_runner or args.test:
        _min_info = ""
        if not args.live_application:
            _min_info = await tb_app.load_all_mods_in_file()
        if args.save_function_enums_in_file:
            tb_app.save_registry_as_enums("utils\\system", "all_functions_enums.py")
            tb_app.alive = False
            await tb_app.a_exit()
            return 0
        if args.debug:
            tb_app.print_functions()
        if _min_info:
            print(_min_info)
        if args.get_version:
            print(
                f"\n{' Version ':-^45}\n\n{Style.Bold(Style.CYAN(Style.ITALIC('RE'))) + Style.ITALIC('Simple') + 'ToolBox':<35}:{__version__:^10}\n")
            for mod_name in tb_app.functions:
                if isinstance(tb_app.functions[mod_name].get("app_instance"), MainTool):
                    print(f"{mod_name:^35}:{tb_app.functions[mod_name]['app_instance'].version:^10}")
                else:
                    try:
                        v = tb_app.functions[mod_name].get(list(tb_app.functions[mod_name].keys())[0]).get("version",
                                                                                                           "unknown (functions only)").replace(
                            f"{__version__}:", '')
                    except AttributeError:
                        v = 'unknown'
                    print(f"{mod_name:^35}:{v:^10}")
            print("\n")
            tb_app.alive = False
            await tb_app.a_exit()
            return 0

    if args.profiler:
        profile_execute_all_functions(tb_app)
        tb_app.alive = False
        await tb_app.a_exit()
        return 0

    if args.command and not args.background_application:
        for command in args.command:
            if len(command) < 1:
                tb_app.print_functions()
                tb_app.print(
                    "minimum command length is 2 {module_name} {function_name} optional args...")
                continue
            if len(command) == 1:
                if command[0].lower() == 'true':
                    while tb_app.alive:
                        await asyncio.sleep(1)
                tb_app.get_mod(command[0])
                tb_app.print_functions(command[0])
                tb_app.print(
                    "minimum command length is 2 {module_name} {function_name} optional args...")
                continue

            tb_app.print(f"Running command: {' '.join(command)}")
            call = CallingObject().empty()
            tb_app.get_mod(command[0], spec='app')
            call.module_name = command[0]
            call.function_name = command[1]
            call.args = command[1:]
            spec = 'app' if not args.live_application else tb_app.id
            r = tb_app.run_any((call.module_name, call.function_name), tb_run_with_specification=spec, args_=call.args,
                               get_results=True)
            if asyncio.iscoroutine(r):
                r = await r
            if isinstance(r, asyncio.Task):
                r = await r

            print("Running", spec, r)

    elif not args.kill and not args.docker and tb_app.alive and not args.background_application:

        tb_app.save_autocompletion_dict()
        with open(pid_file, "w") as f:
            f.write(app_pid)
        if args.background_application_runner and args.modi == 'bg':
            await daemon_app.online
        if not args.live_application:
            runnable_dict = runnable_dict_func()
            tb_app.set_runnable(runnable_dict)
            if args.modi in runnable_dict.keys():
                pass
            else:
                raise ValueError(
                    f"Modi : [{args.modi}] not found on device installed modi : {list(runnable_dict.keys())}")
            # open(f"./config/{args.modi}.pid", "w").write(app_pid)
            await tb_app.run_runnable(args.modi)
        elif 'cli' in args.modi:
            runnable_dict = runnable_dict_func('cli')
            tb_app.set_runnable(runnable_dict)
            await tb_app.run_runnable(args.modi)
        elif args.remote:
            await tb_app.rrun_runnable(args.modi)
        else:
            runnable_dict = runnable_dict_func(args.modi[:2])
            tb_app.set_runnable(runnable_dict)
            await tb_app.run_runnable(args.modi)

    elif args.docker:

        runnable_dict = runnable_dict_func('docker')

        if 'docker' not in runnable_dict.keys():
            print("No docker")
            return 1

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

    if args.live_application and args.debug:
        hide = tb_app.hide_console()
        if hide is not None:
            await hide
    if tb_app.alive:
        await tb_app.a_exit()
        return 0

    if os.path.exists(pid_file):
        os.remove(pid_file)

    # print(
    #    f"\n\nPython-loc: {init_args[0]}\nCli-loc: {init_args[1]}\nargs: {tb_app.pretty_print(init_args[2:])}")
    return 0


def main_runner():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main(loop))


if __name__ == "__main__":
    print("STARTED START FROM CLI")
    sys.exit(main_runner())
    # init main : ToolBoxV2 -init main -f init.config
    # Exit
    # y
    # ToolBoxV2 -l || ToolBoxV2 -n main -l
