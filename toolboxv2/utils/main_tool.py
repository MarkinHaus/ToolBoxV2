from toolboxv2.utils import Result, ToolBoxInterfaces, ToolBoxError, ToolBoxInfo, ToolBoxResult
from toolboxv2.utils.toolbox import App, get_app
from toolboxv2.utils.Style import Style
from toolboxv2.utils.tb_logger import get_logger


class MainTool:
    toolID = ""
    app = None

    def __init__(self, *args, **kwargs):
        self.version = kwargs["v"]
        self.tools = kwargs.get("tool", {})
        self.name = kwargs["name"]
        self.logger = kwargs.get("logs")
        if self.logger is None:
            self.logger = get_logger()
        self.color = kwargs.get("color", "WHITE")
        self.todo = kwargs.get("load", lambda: None)
        self._on_exit = kwargs.get("on_exit", lambda: None)
        self.stuf = False
        if not hasattr(self, 'config'):
            self.config = {}
        if self.app is None:
            self.app = get_app()
        self.ac_user_data_sto = {}
        self.description = "A toolbox mod"
        # Result.default(self.app.interface)
        self.load()

    @staticmethod
    def return_result(error: ToolBoxError = ToolBoxError.none,
                      exec_code: int = 0,
                      help_text: str = "",
                      data_info=None,
                      data=None,
                      data_to=ToolBoxInterfaces.cli):

        if data is None:
            data = {}

        if data_info is None:
            data_info = {}

        return Result(
            error,
            ToolBoxResult(data_info=data_info, data=data, data_to=data_to),
            ToolBoxInfo(exec_code=exec_code, help_text=help_text)
        )

    def load(self):
        if self.todo:
            # try:
            self.todo()
        # except Exception as e:
        #    get_logger().error(f" Error loading mod {self.name} {e}")
        else:
            get_logger().info(f"{self.name} no load require")

        self.app.print(f"TOOL : {self.name} online")

    def print(self, message, end="\n", **kwargs):
        if self.stuf:
            return

        self.app.print(Style.style_dic[self.color] + self.name + Style.style_dic["END"] + ":", message, end=end,
                       **kwargs)

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
