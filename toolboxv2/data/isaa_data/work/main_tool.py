from toolboxv2.utils.toolbox import App
from toolboxv2.utils.Style import Style
from toolboxv2.utils.tb_logger import get_logger


class MainTool:
    def __init__(self, *args, **kwargs):
        self.version = kwargs["v"]
        self.tools = kwargs["tool"]
        self.name = kwargs["name"]
        l = get_logger()
        if kwargs["logs"] != l:
            l.warning("Mor then one login object")
        self.color = kwargs["color"]
        self.todo = kwargs["load"]
        self._on_exit = kwargs["on_exit"]
        self.stuf = False
        self.load()

    def load(self):
        if self.todo:
            try:
                self.todo()
            except Exception as e:
                get_logger().error(f" Error loading mod {self.name} {e}")
        else:
            get_logger().info(f"{self.name} no load require")

        print(f"TOOL : {self.name} online")

    def print(self, message, end="\n", **kwargs):
        if self.stuf:
            return

        print(Style.style_dic[self.color] + self.name + Style.style_dic["END"] + ":", message, end=end, **kwargs)

    def get_uid(self, command, app: App):

        # if "cloudm" not in list(app.MOD_LIST.keys()):
        #     return f"Server has no cloudM module", True
#
        # if "db" not in list(app.MOD_LIST.keys()):
        #     return "Server has no database module", True

        res = app.run_any('cloudM', "validate_jwt", command)

        if type(res) is str:
            return res, True

        return res["uid"], False
