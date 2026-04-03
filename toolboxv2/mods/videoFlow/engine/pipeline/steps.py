# toolboxv2/mods/videoFlow/engine/pipeline/steps.py

import logging
from pathlib import Path

from toolboxv2 import App
from toolboxv2.mods.videoFlow.engine.generators.story_generator import StoryGenerator
from toolboxv2.mods.videoFlow.engine.generators.image_generator import ImageGenerator
from toolboxv2.mods.videoFlow.engine.generators.audio_generator import AudioGenerator
from toolboxv2.mods.videoFlow.engine.generators.video_generator import VideoGenerator
from toolboxv2.mods.videoFlow.engine.generators.pdf_generator import PDFGenerator
from toolboxv2.mods.videoFlow.engine.generators.clip_generator import ClipGenerator
from toolboxv2.mods.videoFlow.engine.generators.html_generator import MultiMediaStoryHTMLGenerator
from toolboxv2.mods.videoFlow.engine.project_manager import ProjectManager
from toolboxv2.mods.videoFlow.engine.config import CostTracker

# This would be initialized properly in the main application
logger = logging.getLogger(__name__)

# A mock app for now
class MockApp:
    def get_mod(self, mod_name):
        return None
app = MockApp()


async def run_story_generation_step(project_id: str, prompt: str, cost_tracker: CostTracker):
    project_manager = ProjectManager(cost_tracker)
    project_path = project_manager.get_project_path(project_id)
    if not project_path:
        logger.error(f"Project with id {project_id} not found.")
        return None
        
    story_generator = StoryGenerator(app.get_mod("isaa"), logger)
    story = await story_generator.generate_story(prompt)
    if story:
        project_manager.save_story_yaml(story, project_path)
        project_manager.update_project_status(project_id, "story_complete")
    return story

async def run_image_generation_step(project_id: str, cost_tracker: CostTracker):
    project_manager = ProjectManager(cost_tracker)
    project_path = project_manager.get_project_path(project_id)
    story = project_manager.load_story_yaml(project_path)
    if story:
        image_generator = ImageGenerator(logger, cost_tracker, app.get_mod("isaa"))
        images = await image_generator.generate_all_images(story, project_path)
        project_manager.update_project_status(project_id, "images_complete")
        return images
    return None

async def run_audio_generation_step(project_id: str, cost_tracker: CostTracker, use_elevenlabs: bool = False):
    project_manager = ProjectManager(cost_tracker)
    project_path = project_manager.get_project_path(project_id)
    story = project_manager.load_story_yaml(project_path)
    if story:
        audio_generator = AudioGenerator(logger, cost_tracker, project_path, use_elevenlabs=use_elevenlabs)
        audio = await audio_generator.generate_audio(story, project_path)
        project_manager.update_project_status(project_id, "audio_complete")
        return audio
    return None

async def run_video_generation_step(project_id: str, cost_tracker: CostTracker):
    project_manager = ProjectManager(cost_tracker)
    project_path = project_manager.get_project_path(project_id)
    story = project_manager.load_story_yaml(project_path)
    images = list((project_path / "images").glob("*.png"))
    audio = list((project_path / "audio").glob("*.wav"))
    if story and images and audio:
        video_generator = VideoGenerator(logger, project_path)
        video = await video_generator.create_video(story, images, audio[0], project_path)
        project_manager.update_project_status(project_id, "video_complete")
        return video
    return None

async def run_pdf_generation_step(project_id: str, cost_tracker: CostTracker):
    project_manager = ProjectManager(cost_tracker)
    project_path = project_manager.get_project_path(project_id)
    story = project_manager.load_story_yaml(project_path)
    images = {'all_images_complete': list((project_path / "images").glob("*.png"))}
    if story:
        pdf_generator = PDFGenerator(logger)
        pdf = pdf_generator.create_complete_pdf(story, images, project_path, cost_tracker.get_summary())
        project_manager.update_project_status(project_id, "pdf_complete")
        return pdf
    return None

async def run_clip_generation_step(project_id: str, cost_tracker: CostTracker):
    project_manager = ProjectManager(cost_tracker)
    project_path = project_manager.get_project_path(project_id)
    story = project_manager.load_story_yaml(project_path)
    images = [str(p) for p in (project_path / "images").glob("*.png")]
    if story and images:
        image_generator = ImageGenerator(logger, cost_tracker, app.get_mod("isaa"))
        clip_generator = ClipGenerator(logger, cost_tracker, app.get_mod("isaa"), image_generator)
        clips = await clip_generator.generate_all_clips(story, images, project_path, image_generator)
        project_manager.update_project_status(project_id, "clips_complete")
        return clips
    return None

async def run_html_generation_step(project_id: str, cost_tracker: CostTracker):
    project_manager = ProjectManager(cost_tracker)
    project_path = project_manager.get_project_path(project_id)
    story = project_manager.load_story_yaml(project_path)
    if story:
        html_generator = MultiMediaStoryHTMLGenerator(logger)
        html = html_generator.create_complete_html_experience(story, project_path)
        project_manager.update_project_status(project_id, "html_complete")
        return html
    return None
