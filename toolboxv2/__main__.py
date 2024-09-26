"""Console script for toolboxv2."""
import json
# Import default Pages
import sys
import argparse
import threading
import time
import asyncio
from functools import wraps
from platform import system, node

from yaml import safe_load

from toolboxv2.runabel import runnable_dict as runnable_dict_func
from toolboxv2.tests.a_util import async_test
from toolboxv2.utils.system.conda_runner import conda_runner_main
from toolboxv2.utils.system.getting_and_closing_app import a_get_proxy_app
from toolboxv2.utils.system.main_tool import MainTool
from toolboxv2.utils.extras.Style import Style, Spinner
from toolboxv2.utils.system.session import Session

from toolboxv2.utils.toolbox import App

from toolboxv2.utils import show_console
from toolboxv2.utils import get_app
from toolboxv2.utils.daemon import DaemonApp
from toolboxv2.utils.proxy import ProxyApp
from toolboxv2.utils.system import get_state_from_app, CallingObject

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
                        help="Start an Proxy interface optional using -p -w",
                        action="store_true",  # Ändere zu store_true
                        default=False)

    parser.add_argument("--docker", help="start the toolbox in docker Enables 4 modi [test,live,live0,dev]\n\trun as "
                                         "$ tb --docker -m [modi] optional -p -w\n\tvalid with -fg", default=False,
                        action="store_true")
    parser.add_argument("--build", help="build docker image from local source", default=False,
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

    parser.add_argument("--ipy", action="store_true", default=False,
                        help="activate toolbox in IPython")

    args = parser.parse_args()
    # args.live_application = not args.live_application
    return args


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


async def setup_app():
    args = parse_args()

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
            test_path = test_path + "\\tests"
        else:
            test_path = test_path + "/tests"
        print(f"Testing in {test_path}")
        if not run_tests(test_path):
            print("Error in tests")
            exit(1)
        exit(0)

    abspath = os.path.dirname(os.path.abspath(__file__))
    info_folder = abspath + '\\.info\\'
    pid_file = f"{info_folder}{args.modi}-{args.name}.pid"
    app_pid = str(os.getpid())

    tb_app = get_app(from_="InitialStartUp", name=args.name, args=args, app_con=App)

    if not args.sysPrint and not (args.debug or args.background_application_runner or args.install or args.kill):
        tb_app.sprint = lambda text, *_args, **kwargs: [time.sleep(0.1), False][-1]

    tb_app.loop = asyncio.get_running_loop()

    if args.load_all_mod_in_files:
        _min_info = await tb_app.load_all_mods_in_file()
        with Spinner("Crating State"):
            st = threading.Thread(target=get_state_from_app, args=(tb_app,
                                                                   os.environ.get("TOOLBOXV2_REMOTE_BASE",
                                                                                  "https://simplecore.app"),
                                                                   "https://github.com/MarkinHaus/ToolBoxV2/tree/master/toolboxv2/"),
                                  daemon=True)
        st.start()
        tb_app.print_functions()
        if _min_info:
            print(_min_info)

    if args.update:
        if args.update == "main":
            await tb_app.save_load("CloudM")
            tb_app.run_any("CloudM", "update_core")
        else:
            tb_app.run_any("CloudM", "install", module_name=args.update, update=True, get_results=True).print()
        # os.system("git pull")
        # self.save_load("CloudM")
        # self.run_any("CloudM", "update_core")

    tb_app.print("OK")
    if args.background_application_runner:
        daemon_app = await DaemonApp(tb_app, args.host, args.port if args.port != 5000 else 6587, t=False)
        if not args.debug:
            show_console(False)
        tb_app.daemon_app = daemon_app
        args.live_application = False
    elif args.background_application:
        if not args.kill:
            start(args.name, sys.argv)
        else:
            if '-m ' not in sys.argv:
                pid_file = f"{info_folder}bg-{args.name}.pid"
            try:
                _ = await ProxyApp(tb_app, args.host if args.host != "0.0.0.0" else "localhost",
                                   args.port if args.port != 5000 else 6587, timeout=4)
                await _.verify()
                if await _.exit_main() != "No data look later":
                    stop(pid_file, args.name)
            except Exception:
                stop(pid_file, args.name)
    elif args.live_application:
        try:
            tb_app = await a_get_proxy_app(tb_app, host=args.host if args.host != "0.0.0.0" else "localhost",
                                           port=args.port if args.port != 5000 else 6587,
                                           key=os.getenv("TB_R_KEY", "user@phfrase"))
            if args.debug:
                await tb_app.show_console()
        except:
            print("Auto starting Starting Local if u know ther is no bg instance use -fg to run in the frond ground")

    with open(pid_file, "w") as f:
        f.write(app_pid)

    return tb_app, args


async def command_runner(tb_app, command):
    if len(command) < 1:
        tb_app.print_functions()
        tb_app.print(
            "minimum command length is 2 {module_name} {function_name} optional args...")
        return

    tb_app.print(f"Running command: {' '.join(command)}")
    call = CallingObject().empty()
    mod = tb_app.get_mod(command[0], spec='app')
    if hasattr(mod, "async_initialized") and not mod.async_initialized:
        await mod
    call.module_name = command[0]

    if len(command) < 2:
        tb_app.print_functions(command[0])
        tb_app.print(
            "minimum command length is 2 {module_name} {function_name} optional args...")
        return

    call.function_name = command[1]
    call.args = command[2:]
    spec = 'app'  #  if not args.live_application else tb_app.id
    r = await tb_app.a_run_any((call.module_name, call.function_name), tb_run_with_specification=spec,
                               args_=call.args,
                               get_results=True)
    if asyncio.iscoroutine(r):
        r = await r
    if isinstance(r, asyncio.Task):
        r = await r

    print("Running", spec, r)


async def mod_installer(tb_app, module_name):
    report = tb_app.run_any("CloudM", "install", module_name=module_name)
    if not report:
        await asyncio.sleep(0.1)
        if 'n' not in input(f"Mod '{module_name}' not found in local mods_sto install from remote ? (yes,no)"):
            session = Session(tb_app.get_username(), os.getenv("TOOLBOXV2_REMOTE_BASE"))
            if not await session.login():
                mk = input(f"bitte geben sie ihren magik link ein {session.base}/")
                if 'web/' not in mk:
                    print("Link is not in Valid format")
                    return
                if not await session.init_log_in_mk_link(mk):
                    print("Link is not in Valid")
                    return
            response = await session.fetch("/api/CloudM/get_latest?module_name=" + module_name, method="GET")
            json_response = await response.json()
            response = json.loads(json_response)

            do_install = True
            if response['error'] != "none":
                print("Error while fetching mod data")
                do_install = False

            if response['result']['data'] == 'mod not found':
                print("404 mod not found")
                do_install = False
            if not do_install:
                return
            print(f"mod url is : {session.base + response['result']['data']}")
            if not await session.download_file(response['result']['data'], tb_app.start_dir + '/mods_sto'):
                print("failed to download mod")
                print("optional download it ur self and put the zip in the mods_sto folder")
                if 'y' not in input("Done ? will start set up from the mods_sto folder").lower():
                    return
            os.rename(
                tb_app.start_dir + '/mods_sto/' + response['result']['data'].split('/')[-1].replace("$",
                                                                                                    '').replace(
                    "&", '').replace("§", ''),
                tb_app.start_dir + '/mods_sto/' + response['result']['data'].split('/')[-1])
            report = tb_app.run_any("CloudM", "install", module_name=module_name)
            if not report:
                print("Set up error")
            return


async def main():
    """Console script for toolboxv2."""
    tb_app, args = await setup_app()
    with open(os.getenv('CONFIG_FILE', f'{os.path.abspath(__file__).replace("__main__.py", "")}toolbox.yaml'),
              'r') as config_file:
        _version = safe_load(config_file)
        __version__ = _version.get('main', {}).get('version', '-.-.-')

    abspath = os.path.dirname(os.path.abspath(__file__))
    info_folder = abspath + '\\.info\\'
    pid_file = f"{info_folder}{args.modi}-{args.name}.pid"

    if args.install:
        await mod_installer(tb_app, args.install)

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
        if args.save_function_enums_in_file:
            tb_app.save_registry_as_enums("utils\\system", "all_functions_enums.py")
            tb_app.alive = False
            await tb_app.a_exit()
            return 0
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
            await command_runner(tb_app, command)

    if not args.kill and not args.docker and tb_app.alive and not args.background_application and (
        not args.command or '-m' in sys.argv):

        tb_app.save_autocompletion_dict()
        if args.background_application_runner and args.modi == 'bg' and hasattr(tb_app, 'daemon_app'):
            await tb_app.daemon_app.online
        if not args.live_application:
            runnable_dict = runnable_dict_func(remote=False)
            if args.modi not in runnable_dict.keys():
                runnable_dict = {**runnable_dict, **runnable_dict_func(s=args.modi, remote=True)}
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

    if os.path.exists(pid_file):
        os.remove(pid_file)

    if not tb_app.called_exit[0]:
        await tb_app.a_exit()
        return 0
    # print(
    #    f"\n\nPython-loc: {init_args[0]}\nCli-loc: {init_args[1]}\nargs: {tb_app.pretty_print(init_args[2:])}")
    return 0


def install_ipython():
    os.system('pip install ipython prompt_toolkit')


def tb_pre_ipy(app, eo):
    # print(f"In Data:  \n\t{eo.raw_cell}\n\t{eo.store_history}\n\t{eo.silent}\n\t{eo.shell_futures}\n\t{eo.cell_id}")
    # app.print(f"{eo.raw_cell=}{eo.raw_cell.split(' ')=}")
    if eo.raw_cell != 'exit':
        eo.raw_cell = ''
    # start information getering


def tb_post_ipy(app, rest):
    # print(f"Out Data:  \n\t{rest.execution_count}\n\t{rest.error_before_exec}\n\t{rest.error_in_exec}
    # \n\t{rest.info.raw_cell}\n\t{rest.info.store_history}\n\t{rest.info.silent}\n\t{rest.info.shell_futures}
    # \n\t{rest.info.cell_id}\n\t{rest.result} ")
    # return information
    return ""


def line_magic_ipy(app, ipython, line):
    app.mod_online(line.split(' ')[0].strip(), True)
    if line.split(' ')[0].strip() in app.functions:
        async_test(command_runner)(app, line.split(' '))
    else:
        app.print_functions()

def configure_ipython(argv):
    from traitlets.config import Config

    c = Config()

    # Autocompletion with prompt_toolkit
    c.InteractiveShellCompleter.use_jedi = True
    c.InteractiveShell.automagic = True
    # Enable contextual help
    c.InteractiveShellApp.exec_lines = []

    c.TerminalInteractiveShell.editor = 'nano'

    c.PrefilterManager.multi_line_specials = True

    c.InteractiveShell.colors = 'LightBG'
    c.InteractiveShell.confirm_exit = True
    c.TerminalIPythonApp.display_banner = False
    c.AliasManager.user_aliases = [
        ("TB", "tb"),
        ("@", "!tb -c "),
    ]
    c.InteractiveShellApp.exec_lines.append("""import os
import sys
import toolboxv2 as tb
from toolboxv2.tests.a_util import async_test
from threading import Thread
# from toolboxv2.utils.system.ipy_completer import get_completer

from IPython.core.magic import register_line_magic, register_cell_magic
sys.argv = """+str(argv)+"""
app, args = await tb.__main__.setup_app()
if hasattr(app, "daemon_app"):
    Thread(target=async_test(app.daemon_app.connect), args=(app,), daemon=True).start()


def pre_run_code_hook(eo):
    tb.__main__.tb_pre_ipy(app, eo)


def post_run_code_hook(result):
    tb.__main__.tb_post_ipy(app, result)


def load_ipython_extension(ipython):
    @register_line_magic
    def my_line_magic(line):
        parts = line.split(' ')
        f_name = "ipy_sessions/"+("tb_session" if len(parts) <= 1 else parts[-1])

        os.makedirs(f'{app.data_dir}/ipy_sessions/',exist_ok=True)
        if "save" in parts[0]:
            do_inj = not os.path.exists(f'{app.data_dir}/{f_name}.ipy')
            if do_inj:
                ipython.run_line_magic('save', f'{app.data_dir}/{f_name}.ipy')
            else:
                ipython.run_line_magic('save', f'-r {app.data_dir}/{f_name}.ipy')
        if "inject" in parts[0]:
                file_path = f'{app.data_dir}/{f_name}.ipy'
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                # Insert lines after the first line
                lines[1:1] = [line + '\\n' for line in
                              ["import toolboxv2 as tb", "app, args = await tb.__main__.setup_app()"]]
                with open(file_path, 'w') as file:
                    file.writelines(lines)
        elif "loadX" in parts[0]:
            # ipython.run_line_magic('store', '-r')
            ipython.run_line_magic('run', f'{app.data_dir}/{f_name}.ipy')
        elif "load" in parts[0]:
            # ipython.run_line_magic('store', '-r')
            ipython.run_line_magic('load', f'{app.data_dir}/{f_name}.ipy')
        elif "open" in parts[0]:
            file_path = f'{app.data_dir}/{f_name}.ipy'
            if os.path.exists(f'{app.data_dir}/{f_name}.ipy'):
                l = "notebook" if not 'lab' in parts else 'labs'
                os.system(f"jupyter {l} {file_path}")
            else:
                print("Pleas save first")
        else:
            tb.__main__.line_magic_ipy(app, ipython, line)

    @register_cell_magic
    def my_cell_magic(line, cell):
        print(f"Custom cell magic {line} |CELL| {cell}")
        line = line + '\\n' + cell
        tb.__main__.line_magic_ipy(app, ipython, line)

    def apt_completers(self, event):
        return ['save', 'loadX', 'load', 'open', 'inject']

    ipython.set_hook('complete_command', apt_completers, re_key = '%tb')

    ipython.register_magic_function(my_line_magic, 'line', 'tb')
    ipython.register_magic_function(my_cell_magic, 'cell', 'tb')


load_ipython_extension(get_ipython())

# get_ipython().set_custom_completer(get_completer())
get_ipython().events.register("pre_run_cell", pre_run_code_hook)
get_ipython().events.register("post_run_cell", post_run_code_hook)

""")
    ()
    return c


def start_ipython_session(argv):
    from IPython import start_ipython, get_ipython
    config = configure_ipython(argv)

    shell = start_ipython(argv=None, config=config)


def main_runner():
    sys.excepthook = sys.__excepthook__
    if '--ipy' in sys.argv:
        argv = sys.argv[1:]
        sys.argv = sys.argv[:1]
        start_ipython_session(argv)
    else:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(main())


if __name__ == "__main__":
    print("STARTED START FROM CLI")
    if sys.argv[1] == "conda":
        sys.argv[1:] = sys.argv[2:]
        sys.exit(conda_runner_main())
    sys.exit(main_runner())
