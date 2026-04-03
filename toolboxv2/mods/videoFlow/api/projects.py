# toolboxv2/mods/videoFlow/api/projects.py

from toolboxv2 import App, RequestData
from toolboxv2.mods.videoFlow.engine.project_manager import ProjectManager
from toolboxv2.mods.videoFlow.engine.config import CostTracker
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData

def register_api_endpoints(app: App):
    @app.export(api=True, mod_name="videoFlow", route="/create_project", method="POST")
    async def create_project(request_data: RequestData) -> dict:
        user_id = request_data.get("user_id") # Assuming user_id is available from session/JWT
        project_name = request_data.get("projectName")

        if not user_id:
            return {"status": "error", "message": "Unauthorized: User ID not found in session.", "status_code": 401}
        if not project_name:
            return {"status": "error", "message": "Project name is required.", "status_code": 400}

        pm = ProjectManager(CostTracker()) # CostTracker is a placeholder, should be managed globally
        project_path = pm.create_project(project_name) # Using project_name as prompt for simplicity

        # In a real scenario, you'd associate the project with the user_id in ProjectManager
        # For now, we'll just return the project_id based on the path
        project_id = project_path.name # Using folder name as project_id

        return {"status": "success", "projectId": project_id, "status_code": 201}

    @app.export(api=True, mod_name="videoFlow", route="/project_status/{project_id}", method="GET")
    async def get_project_status(request_data: RequestData, project_id: str) -> dict:
        user_id = request_data.get("user_id")

        if not user_id:
            return {"status": "error", "message": "Unauthorized: User ID not found in session.", "status_code": 401}

        pm = ProjectManager(CostTracker())
        project_path = pm.get_project_path(project_id)

        if not project_path or not project_path.exists():
            return {"status": "error", "message": "Project not found.", "status_code": 404}

        # In a real app, verify user_id owns this project_id
        # For now, we'll assume ownership if project_path is found

        status = pm.check_project_status(project_path)
        
        # Load story data if available
        story_data = pm.load_story_yaml(project_path)
        story_dict = story_data.model_dump() if story_data else None

        return {
            "projectId": project_id,
            "status": status,
            "storyData": story_dict,
            "status_code": 200
            # Add other relevant project data here
        }
