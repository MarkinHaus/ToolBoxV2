# toolboxv2/mods/videoFlow/api/generation.py

from toolboxv2 import App, RequestData
from toolboxv2.mods.videoFlow.engine.project_manager import ProjectManager
from toolboxv2.mods.videoFlow.engine.config import CostTracker
from toolboxv2.mods.videoFlow.engine.pipeline.steps import (
    run_story_generation_step,
    run_image_generation_step,
    run_audio_generation_step,
    run_video_generation_step,
    run_pdf_generation_step,
    run_clip_generation_step,
    run_html_generation_step,
)

def register_api_endpoints(app: App):
    @app.export(api=True, mod_name="videoFlow", route="/run_step/{project_id}/{step_name}", method="POST")
    async def run_step(request_data: RequestData, project_id: str, step_name: str) -> dict:
        user_id = request_data.get("user_id")

        if not user_id:
            return {"status": "error", "message": "Unauthorized: User ID not found in session.", "status_code": 401}

        pm = ProjectManager(CostTracker()) # CostTracker is a placeholder, should be managed globally
        project_path = pm.get_project_path(project_id)

        if not project_path or not project_path.exists():
            return {"status": "error", "message": "Project not found.", "status_code": 404}

        # In a real app, verify user_id owns this project_id

        # Placeholder for credit system check
        # if not check_credits(user_id, step_name):
        #    return {"status": "error", "message": "Not enough credits.", "status_code": 402}

        task_mapping = {
            "story": run_story_generation_step,
            "images": run_image_generation_step,
            "audio": run_audio_generation_step,
            "video": run_video_generation_step,
            "pdf": run_pdf_generation_step,
            "clips": run_clip_generation_step,
            "html": run_html_generation_step,
        }

        step_func = task_mapping.get(step_name)
        if not step_func:
            return {"status": "error", "message": f"Invalid step name: {step_name}", "status_code": 400}

        # Get prompt for story generation if it's the story step
        prompt = request_data.get("prompt") if step_name == "story" else None

        # Run the step in a background task
        # app.run_bg_task(step_func, project_id, prompt, pm.cost_tracker) # Assuming cost_tracker is passed
        # For now, run directly for testing purposes, will switch to run_bg_task later
        if step_name == "story":
            await step_func(project_id, prompt, pm.cost_tracker)
        elif step_name == "audio":
            use_elevenlabs = request_data.get("use_elevenlabs", False)
            await step_func(project_id, pm.cost_tracker, use_elevenlabs=use_elevenlabs)
        else:
            await step_func(project_id, pm.cost_tracker)

        return {"status": "success", "message": f"The '{step_name}' generation step has started.", "status_code": 202}
