# toolboxv2/mods/videoFlow/tools.py

from toolboxv2 import MainTool, App
from toolboxv2.mods.videoFlow.api.auth import register_api_endpoints as register_auth_api
from toolboxv2.mods.videoFlow.api.projects import register_api_endpoints as register_projects_api
from toolboxv2.mods.videoFlow.api.generation import register_api_endpoints as register_generation_api
from toolboxv2.mods.videoFlow.api.credits import register_api_endpoints as register_credits_api
from toolboxv2.mods.videoFlow.api.update_story import register_api_endpoints as register_update_story_api

class VideoFlowTool(MainTool):
    name = "videoFlow"

    def __init__(self, app: App):
        super().__init__(app)
        self.app = app
        self._register_all_api_endpoints()

    def _register_all_api_endpoints(self):
        register_auth_api(self.app)
        register_projects_api(self.app)
        register_generation_api(self.app)
        register_credits_api(self.app)
        register_update_story_api(self.app)

    async def start(self, *args, **kwargs):
        # This method will be called when the tool starts
        # You can add any startup logic here
        print("VideoFlowTool started!")

    async def stop(self, *args, **kwargs):
        # This method will be called when the tool stops
        # You can add any cleanup logic here
        print("VideoFlowTool stopped!")

    async def run(self, *args, **kwargs):
        # This method will be called when the tool is executed directly
        print("VideoFlowTool running!")
