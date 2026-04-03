# toolboxv2/mods/videoFlow/api/update_story.py

from toolboxv2 import App, RequestData
from toolboxv2.mods.videoFlow.engine.project_manager import ProjectManager
from toolboxv2.mods.videoFlow.engine.config import CostTracker
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData

def register_api_endpoints(app: App):
    @app.export(api=True, mod_name="videoFlow", route="/update_story/{project_id}", method="PUT")
    async def update_story(request_data: RequestData, project_id: str) -> dict:
        user_id = request_data.get("user_id")
        story_data_dict = request_data.get("storyData")

        if not user_id:
            return {"status": "error", "message": "Unauthorized: User ID not found in session.", "status_code": 401}
        if not story_data_dict:
            return {"status": "error", "message": "Story data is required.", "status_code": 400}

        pm = ProjectManager(CostTracker())
        project_path = pm.get_project_path(project_id)

        if not project_path or not project_path.exists():
            return {"status": "error", "message": "Project not found.", "status_code": 404}

        try:
            story_data = StoryData(**story_data_dict)
            pm.save_story_yaml(story_data, project_path)
            return {"status": "success", "message": "Story updated successfully.", "status_code": 200}
        except Exception as e:
            return {"status": "error", "message": f"Invalid story data: {e}", "status_code": 400}
