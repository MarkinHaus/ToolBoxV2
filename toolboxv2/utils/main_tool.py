import uuid

from toolboxv2.utils.toolbox import App
from toolboxv2.utils.Style import Style
from toolboxv2.utils.tb_logger import get_logger


class MainTool:

    toolID = ""

    def __init__(self, *args, **kwargs):
        self.version = kwargs["v"]
        self.tools = kwargs["tool"]
        self.name = kwargs["name"]
        self.logger = kwargs["logs"]
        if kwargs["logs"] is None:
            self.logger = get_logger()
        self.color = kwargs["color"]
        self.todo = kwargs["load"]
        self._on_exit = kwargs["on_exit"]
        self.stuf = False
        if not hasattr(self, 'config'):
            self.config = {}
        self.load()
        self.ac_user_data_sto = {}

    def load(self):
        if self.todo:
            #try:
                self.todo()
            #except Exception as e:
            #    get_logger().error(f" Error loading mod {self.name} {e}")
        else:
            get_logger().info(f"{self.name} no load require")

        print(f"TOOL : {self.name} online")

    def print(self, message, end="\n", **kwargs):
        if self.stuf:
            return

        print(Style.style_dic[self.color] + self.name + Style.style_dic["END"] + ":", message, end=end, **kwargs)

    def add_str_to_config(self, command):
        if len(command) != 2:
            self.logger.error('Invalid command must be key value')
            return False
        self.config[command[0]] = command[1]

    def webInstall(self, user_instance, construct_render) -> str:
        """"Returns a web installer for the given user instance and construct render template"""

    def get_uid(self, command, app: App):

        # if "cloudm" not in list(app.MOD_LIST.keys()):
        #     return f"Server has no cloudM module", True
        #
        # if "db" not in list(app.MOD_LIST.keys()):
        #     return "Server has no database module", True

        res = app.run_any('cloudM', "validate_jwt", command)

        if type(res) is str:
            return res, True

        self.ac_user_data_sto = res
        return res["uid"], False

    def get_user_instance(self, uid, app: App):
        return app.run_any('cloudM', "get_user_instance", [uid])

