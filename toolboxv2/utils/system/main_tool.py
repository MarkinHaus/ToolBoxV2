import asyncio
import inspect

from toolboxv2.utils.extras import Style

from .types import Result, ToolBoxInterfaces, ToolBoxError, ToolBoxInfo, ToolBoxResult
from .getting_and_closing_app import get_app
from .tb_logger import get_logger
from .all_functions_enums import CLOUDM_AUTHMANAGER


class MainTool:
    toolID: str = ""
    # app = None
    interface = None
    spec = "app"
    name = ""
    color = "Bold"
    stuf = False

    def __init__(self, *args, **kwargs):
        """
        Standard constructor used for arguments pass
        Do not override. Use __ainit__ instead
        """
        self.__storedargs = args, kwargs
        self.async_initialized = False

    async def __ainit__(self, *args, **kwargs):
        self.version = kwargs["v"]
        self.tools = kwargs.get("tool", {})
        self.name = kwargs["name"]
        self.logger = kwargs.get("logs", get_logger())
        self.color = kwargs.get("color", "WHITE")
        self.todo = kwargs.get("load", lambda: None)
        if not hasattr(self, 'config'):
            self.config = {}
        self.user = None
        self.description = "A toolbox mod" if kwargs.get("description") is None else kwargs.get("description")
        if MainTool.interface is None:
            MainTool.interface = self.app.interface_type
        # Result.default(self.app.interface)
        if self.todo:
            try:
                if inspect.iscoroutinefunction(self.todo):
                    self.app.loop.call_soon_threadsafe(self.todo)
                else:
                    self.todo()
                await asyncio.sleep(0)
            except Exception as e:
                get_logger().error(f" Error loading mod {self.name} {e}")
        else:
            get_logger().info(f"{self.name} no load require")

        self.app.print(f"TOOL : {self.spec}.{self.name} online")

    @property
    def app(self):
        return get_app(
            from_=f"{self.spec}.{self.name}|{self.toolID if self.toolID else '*' + MainTool.toolID} {self.interface if self.interface else MainTool.interface}")

    @app.setter
    def app(self, v):
        raise PermissionError(f"You cannot set the App Instance! {v=}")

    @staticmethod
    def return_result(error: ToolBoxError = ToolBoxError.none,
                      exec_code: int = 0,
                      help_text: str = "",
                      data_info=None,
                      data=None,
                      data_to=None):

        if data_to is None:
            data_to = MainTool.interface if MainTool.interface is not None else ToolBoxInterfaces.cli

        if data is None:
            data = {}

        if data_info is None:
            data_info = {}

        return Result(
            error,
            ToolBoxResult(data_info=data_info, data=data, data_to=data_to),
            ToolBoxInfo(exec_code=exec_code, help_text=help_text)
        )

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

    def get_version(self) -> str:
        """"Returns the version"""
        return self.version

    def get_user(self, username: str) -> Result:
        return self.app.run_any(CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=username, get_results=True)

    async def __initobj(self):
        """Crutch used for __await__ after spawning"""
        assert not self.async_initialized
        self.async_initialized = True
        # pass the parameters to __ainit__ that passed to __init__
        await self.__ainit__(*self.__storedargs[0], **self.__storedargs[1])
        return self

    def __await__(self):
        return self.__initobj().__await__()
