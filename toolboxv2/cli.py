"""Console script for toolboxv2."""
import logging
# Import default Pages
import os
import sys
import argparse
import socketserver
from platform import system

# Import public Pages
from toolboxv2 import App, run_cli, AppServerHandler, MainTool
from toolboxv2.util import edit_log_files, loggerNameOfToolboxv2, unstyle_log_files


def parse_args():
    parser = argparse.ArgumentParser(description="Welcome to the ToolBox cli")

    parser.add_argument("-init",
                        help="ToolBoxV2 init (name) -> default : -n name = main")

    parser.add_argument('-f', '--init-file',
                        type=str,
                        default="init.config",
                        help="optional init flag init from config file or url")

    parser.add_argument("-update",
                        help="update ToolBox",
                        action="store_true")

    parser.add_argument("--update-mod",
                        help="update ToolBox mod", )

    parser.add_argument("--delete-ToolBoxV2",
                        help="delete ToolBox or mod | ToolBoxV2 --delete-ToolBoxV2 ",
                        nargs=2,
                        choices=["all" "cli", "dev", "api", 'config', 'data', 'src', 'all'])

    parser.add_argument("--delete-mod",
                        help="delete ToolBoxV2 mod | ToolBox --delete-mod (mod-name)")

    parser.add_argument("-v", "--get-version",
                        help="get version of ToolBox | ToolBoxV2 -v -n (mod-name)",
                        action="store_true")

    parser.add_argument('-mvn', '--mod-version-name',
                        metavar="name",
                        type=str,
                        help="Name of mod",
                        default="mainTool")

    parser.add_argument('-n', '--name',
                        metavar="name",
                        type=str,
                        help="Name of ToolBox",
                        default="main")

    parser.add_argument("-m", "--modi",
                        type=str,
                        help="Start ToolBox in different modes",
                        choices=["cli", "dev", "api", "app"],
                        default="cli")

    parser.add_argument("-p", "--port",
                        metavar="port",
                        type=int,
                        help="Specify a port for dev | api",
                        default=12689)  # 1268945

    parser.add_argument("-w", "--host",
                        metavar="host",
                        type=str,
                        help="Specify a host for dev | api",
                        default="0.0.0.0")

    parser.add_argument("-l", "--load-all-mod-in-files",
                        help="yeah",
                        action="store_true")

    parser.add_argument("--log-editor",
                        help="yeah",
                        action="store_true")

    parser.add_argument("--live",
                        help="yeah",
                        action="store_true")

    return parser.parse_args()


def edit_logs():

    name = input(f"Name of logger \ndefault {loggerNameOfToolboxv2} \n:")
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

    level = input(f"Level : {list(zip(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'], [50, 40, 30, 20, 10, 0]))}"
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

    # (init=None,
    # init_file=None,
    # update=False,
    # delete_ToolBoxV2=None,
    # delete_mod=None,
    # get_version=False,
    # name='main',
    # modi='cli',
    # port=12689,
    # host='0.0.0.0',
    # load_all_mod_in_files=False,
    # live=False
    # )

    try:
        init_args = " ".join(sys.orig_argv)
    except AttributeError:
        init_args = "python3 "
        init_args += " ".join(sys.argv)
    init_args = init_args.split(" ")

    tb_app = App(args.name, args=args)

    if args.log_editor:
        edit_logs()
        tb_app.save_exit()
        tb_app.exit()
        exit(0)

    if args.load_all_mod_in_files:
        tb_app.load_all_mods_in_file()

        if args.get_version:

            for mod_name in tb_app.MACRO:
                if mod_name in tb_app.MOD_LIST.keys():
                    if isinstance(tb_app.MOD_LIST[mod_name], MainTool):
                        print(f"{mod_name} : {tb_app.MOD_LIST[mod_name].version}")

    if args.modi == 'api':
        tb_app.run_any('api_manager', 'start-api', ['start-api', args.name])
    if args.modi == 'dev':
        dev_helper()
    if args.modi == 'app':
        print(args.host, args.port)
        httpd = socketserver.TCPServer((args.host, args.port), AppServerHandler)
        httpd.serve_forever()

    if args.modi == 'cli':
        run_cli(tb_app)

    if args.modi == "app" and args.name == "kill":

        app_pid = str(os.getpid())
        print(f"Exit app {app_pid}")
        if system() == "Windows":
            os.system(f"taskkill /pid {app_pid}")
        else:
            os.system(f"kill -9 {app_pid}")

    if tb_app.alive:
        tb_app.save_exit()
        tb_app.exit()

    print("\n\tSee u")
    print(
        f"\n\nPython-loc: {init_args[0]}\nCli-loc: {init_args[1]}\nargs: {tb_app.pretty_print(init_args[2:])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    # init main : ToolBoxV2 -init main -f init.config
    # Exit
    # y
    # ToolBoxV2 -l || ToolBoxV2 -n main -l
