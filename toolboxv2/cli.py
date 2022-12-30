"""Console script for toolboxv2."""
# Import default Pages
import os
import sys
import argparse
import socketserver
from platform import system
from threading import Thread

# Import public Pages
from toolboxv2 import App, run_cli, AppServerHandler


def parse_args():
    parser = argparse.ArgumentParser(description="Welcome to the ToolBox cli")

    parser.add_argument("-init",
                        help="ToolBoxV2 init (name) -> default : -n name = main")

    parser.add_argument('-f', '--init-file',
                        type=str,
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

    parser.add_argument("--live",
                        help="yeah",
                        action="store_true")

    return parser.parse_args()


def main():

    """Console script for toolboxv2."""
    args = parse_args()

    def dev_helper():
        dev_args = f"streamlit run streamlit_web_dev_tools.py ata:{args.name}.config:{args.name}" \
                   f" {(f'--server.port={args.port} --server.address={args.host}' if args.host != '0.0.0.0' else '')}"
        os.system(dev_args)

    print("Arguments: ", args)

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

    if args.load_all_mod_in_files:
        tb_app.load_all_mods_in_file()

    if args.modi == 'api':
        tb_app.run_any('api_manager', 'start-api', ['start-api', args.name])
    if args.modi == 'dev':
        dev_helper()
    if args.modi == 'app':
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

    print("\n\tSee u")
    print(
        f"commands: \nPython-loc: {init_args[0]}\nCli-loc: {init_args[1]}\nargs: {tb_app.pretty_print(init_args[2:])}\n{' '.join(init_args)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

    # init main : ToolBoxV2 -init main -f init.config
    # Exit
    # y
    # ToolBoxV2 -l || ToolBoxV2 -n main -l
