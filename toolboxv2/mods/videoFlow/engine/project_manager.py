# toolboxv2/mods/videoFlow/engine/project_manager.py

import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import yaml
from types import MappingProxyType

from toolboxv2.mods.videoFlow.engine.config import Config, CostTracker
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData, VoiceType, CharacterRole, ImageStyle, VideoStyle


def path_to_str(obj):
    """Convert Path objects to strings recursively"""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: path_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [path_to_str(item) for item in obj]
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, '__dict__'):  # Handle custom objects
        return path_to_str(obj.__dict__)
    return obj


def str_to_path(obj, path_keys=None):
    """Convert specific string keys back to Path objects"""
    if path_keys is None:
        path_keys = ['path', 'project_path', 'image_path', 'audio_path', 'video_path', 'pdf_path']

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in path_keys and isinstance(v, str):
                result[k] = Path(v)
            elif k == 'path_obj' and isinstance(v, str):
                result[k] = Path(v)
            else:
                result[k] = str_to_path(v, path_keys)
        return result
    elif isinstance(obj, list):
        return [str_to_path(item, path_keys) for item in obj]
    return obj


def make_json_serializable(obj):
    """Convert any object to JSON serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (dict, MappingProxyType)):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, 'model_dump') and callable(getattr(obj, 'model_dump')):  # Pydantic models
        return make_json_serializable(obj.model_dump())
    elif hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):  # Pydantic models
        return make_json_serializable(obj.dict())
    elif hasattr(obj, '__dict__'):  # Custom objects
        return make_json_serializable(vars(obj))
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return str(obj)


class ProjectManager:
    """Enhanced project management with autoresume functionality"""

    def __init__(self, cost_tracker: CostTracker):
        self.base_dir = Config.BASE_OUTPUT_DIR
        self.base_dir.mkdir(exist_ok=True)
        self.projects_index_file = self.base_dir / "projects_index.json"
        self.load_projects_index()
        self.cost_tracker = cost_tracker

    def load_projects_index(self):
        """Load projects index from file with proper deserialization"""
        if self.projects_index_file.exists():
            with open(self.projects_index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.projects_index = str_to_path(data)
                self.cost_tracker = CostTracker.from_summary(self.projects_index.get('cost_summary', {}))
        else:
            self.projects_index = {}

    def save_projects_index(self):
        """Save projects index to file with robust serialization"""
        try:
            serializable_index = make_json_serializable(self.projects_index)
            with open(self.projects_index_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_index, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"⚠️ Failed to save projects index: {e}")

    def get_prompt_hash(self, prompt: str) -> str:
        """Generate consistent hash for prompt"""
        return hashlib.md5(prompt.strip().lower().encode()).hexdigest()

    def get_project_path(self, project_id: str) -> Optional[Path]:
        """Get the path for a project from the index."""
        project_info = self.projects_index.get(project_id)
        if project_info:
            return Path(project_info['path'])
        return None

    def update_project_status(self, project_id: str, status: str):
        """Update the status of a project."""
        if project_id in self.projects_index:
            self.projects_index[project_id]['status'] = status
            self.save_projects_index()

    def find_existing_projects(self, prompt: str) -> List[Dict]:
        """Find existing projects for given prompt"""
        prompt_hash = self.get_prompt_hash(prompt)
        existing_projects = []

        for project_id, project_info in self.projects_index.items():
            if project_info.get('prompt_hash') == prompt_hash:
                # Convert path string to Path object
                project_path = Path(str(project_info['path']))  # Ensure it's a string first
                if project_path.exists():
                    # Get project status
                    status = self.check_project_status(project_path)
                    # Create a copy to avoid modifying original
                    project_copy = path_to_str(project_info.copy())
                    project_copy['status'] = status
                    project_copy['path_obj'] = project_path
                    existing_projects.append(project_copy)

        return existing_projects


    def check_project_status(self, project_dir: Path) -> Dict:
        """Check what assets exist in project - FIXED PDF detection"""
        status = {
            'story_yaml': (project_dir / "story.yaml").exists(),
            'metadata': (project_dir / "project_metadata.json").exists(),
            'images': len(list((project_dir / "images").glob("*.png"))) if (project_dir / "images").exists() else 0,
            'audio': len(list((project_dir / "audio").glob("*.wav"))) if (project_dir / "audio").exists() else 0,
            'video': len(list((project_dir / "video").glob("*.mp4"))) if (project_dir / "video").exists() else 0,
            'pdf': 0,  # Initialize
            'clips': len(list((project_dir / "video").glob("*.mp4"))) if (project_dir / "video").exists() else 0
        }

        # FIXED: Check for PDFs in multiple locations
        pdf_locations = [
            project_dir,  # Root directory
            project_dir / "pdf",  # PDF subdirectory
        ]

        for location in pdf_locations:
            if location.exists():
                pdf_files = list(location.glob("*.pdf"))
                status['pdf'] += len(pdf_files)

        # Calculate completion percentage
        total_steps = 6  # story, images, audio, video, pdf, clips
        completed_steps = sum([
            status['story_yaml'],
            status['images'] > 0,
            status['audio'] > 0,
            status['video'] > 0,
            status['pdf'] > 0,
            status['clips'] > 0
        ])
        status['completion_percentage'] = (completed_steps / total_steps) * 100

        return status

    def create_project(self, prompt: str, resume_project: Optional[Path] = None) -> Path:
        """Create new project or return existing for resume"""
        if resume_project:
            return resume_project

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_hash = self.get_prompt_hash(prompt)
        clean_prompt = re.sub(r'[^\\w\\s-]', '', prompt)[:20].replace(' ', '_')

        project_name = f"{timestamp}_{clean_prompt}_{prompt_hash[:8]}"
        project_dir = self.base_dir / project_name
        project_dir.mkdir(exist_ok=True)

        # Create subdirs
        for subdir in ["images", "audio", "video", "pdf", "transitions"]:
            (project_dir / subdir).mkdir(exist_ok=True)

        # Register project in index
        project_id = f"{timestamp}_{prompt_hash[:8]}"
        self.projects_index[project_id] = {
            'prompt': prompt,
            'prompt_hash': prompt_hash,
            'path': str(project_dir),
            'created': timestamp,
            'last_accessed': timestamp
        }
        self.save_projects_index()

        return project_dir

    def update_project_access(self, project_dir: Path):
        """Update last accessed time for project"""
        project_name = project_dir.name
        for project_id, project_info in self.projects_index.items():
            if Path(project_info['path']).name == project_name:
                project_info['last_accessed'] = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.save_projects_index()
                break

    def validate_story_structure(self, story: StoryData) -> Tuple[bool, List[str]]:
        """Validate story structure"""
        errors = []

        if not story.title or len(story.title.strip()) < 3:
            errors.append("Story title too short or missing")

        if not story.scenes or len(story.scenes) < 1:
            errors.append("Story needs at least one scene")

        if not story.world_desc or len(story.world_desc.strip()) < 10:
            errors.append("World description too short")

        for i, char in enumerate(story.characters):
            if not char.visual_desc or len(char.visual_desc.strip()) < 5:
                errors.append(f"Character {i + 1} visual_desc too short")

        return len(errors) == 0, errors

    def validate_generated_assets(self, project_dir: Path, expected_counts: Dict) -> Tuple[bool, Dict]:
        """Validate generated assets meet expectations - FIXED PDF validation"""
        status = self.check_project_status(project_dir)
        validation_results = {}

        # Check images
        expected_images = expected_counts.get('images', 0)
        validation_results['images'] = {
            'expected': expected_images,
            'actual': status['images'],
            'valid': status['images'] >= max(1, expected_images * 0.8)  # Allow 20% tolerance, minimum 1
        }

        # Check audio
        validation_results['audio'] = {
            'expected': expected_counts.get('audio', 1),
            'actual': status['audio'],
            'valid': status['audio'] >= 1
        }

        # Check video
        validation_results['video'] = {
            'expected': expected_counts.get('video', 1),
            'actual': status['video'],
            'valid': status['video'] >= 1
        }

        # FIXED: Check PDF properly
        validation_results['pdf'] = {
            'expected': expected_counts.get('pdf', 1),
            'actual': status['pdf'],
            'valid': status['pdf'] >= 1
        }

        all_valid = all(result['valid'] for result in validation_results.values())

        return all_valid, validation_results

    def get_resume_choice(self, existing_projects: List[Dict]) -> Tuple[str, Optional[Path]]:
        """Interactive choice for resume or new project"""
        print(f"\n🔍 Found {len(existing_projects)} existing project(s) for this prompt:")
        print("=" * 60)

        for i, project in enumerate(existing_projects, 1):
            status = project['status']
            print(f"{i}. Project: {Path(project['path']).name}")
            print(f"   📅 Created: {project['created']}")
            print(f"   📊 Completion: {status['completion_percentage']:.1f}%")
            print(f"   📁 Images: {status['images']}, 🎵 Audio: {status['audio']}, 🎬 Video: {status['video']}")
            print(f"   📄 PDF: {status['pdf']}, 🎞️ Clips: {status['clips']}")
            print()

        print("Options:")
        print("0. Create NEW project")
        for i in range(len(existing_projects)):
            print(f"{i + 1}. Resume project {i + 1}")

        while True:
            try:
                choice = input(f"\nEnter choice (0-{len(existing_projects)}): ").strip()
                choice_num = int(choice)

                if choice_num == 0:
                    return "new", None
                elif 1 <= choice_num <= len(existing_projects):
                    selected_project = existing_projects[choice_num - 1]
                    return "resume", selected_project['path_obj']
                else:
                    print(f"❌ Invalid choice. Enter 0-{len(existing_projects)}")
            except ValueError:
                print("❌ Invalid input. Enter a number.")

    def save_story_yaml(self, story: StoryData, project_dir: Path):
        """Save story as YAML with proper enum serialization"""
        is_valid, errors = self.validate_story_structure(story)
        if not is_valid:
            print("⚠️ Story validation warnings:")
            for error in errors:
                print(f"   - {error}")

        yaml_path = project_dir / "story.yaml"

        # Convert story to dict and serialize enums properly
        story_dict = self._serialize_enums_for_yaml(story.dict())

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(story_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"✅ Story YAML saved: {is_valid and 'Valid' or 'With warnings'}")

    def load_story_yaml(self, project_dir: Path) -> Optional[StoryData]:
        """Load story from YAML with proper enum deserialization"""
        story_yaml_path = project_dir / "story.yaml"
        if not story_yaml_path.exists():
            return None

        try:
            with open(story_yaml_path, 'r', encoding='utf-8') as f:
                story_data = yaml.safe_load(f)

            # Deserialize enums from string values
            story_data = self._deserialize_enums_from_yaml(story_data)

            existing_story = StoryData(**story_data)
            return existing_story

        except Exception as e:
            print(f"⚠️ Could not load existing story: {e}")
            return None

    def _serialize_enums_for_yaml(self, data):
        """Convert enum objects to their string values for YAML serialization"""
        if isinstance(data, dict):
            return {key: self._serialize_enums_for_yaml(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._serialize_enums_for_yaml(item) for item in data]
        elif isinstance(data, (VoiceType, CharacterRole, ImageStyle, VideoStyle)):
            return data.value  # Convert enum to its string value
        else:
            return data

    def _deserialize_enums_from_yaml(self, data):
        """Convert string values back to enum objects after YAML loading"""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key == 'voice' and isinstance(value, str):
                    try:
                        result[key] = VoiceType(value)
                    except ValueError:
                        result[key] = value  # Keep original if conversion fails
                elif key == 'role' and isinstance(value, str):
                    try:
                        result[key] = CharacterRole(value)
                    except ValueError:
                        result[key] = value
                elif key == 'image_style' and isinstance(value, str):
                    try:
                        result[key] = ImageStyle(value)
                    except ValueError:
                        result[key] = value
                elif key == 'camera_style' and isinstance(value, str):
                    try:
                        result[key] = VideoStyle(value)
                    except ValueError:
                        result[key] = value
                else:
                    result[key] = self._deserialize_enums_from_yaml(value)
            return result
        elif isinstance(data, list):
            return [self._deserialize_enums_from_yaml(item) for item in data]
        else:
            return data
    def save_metadata(self, story: StoryData, cost_summary: Dict, generated_files: Dict, project_dir: Path):
        """Save complete project metadata with robust serialization"""
        # Convert generated_files paths to strings
        serializable_files = make_json_serializable(generated_files)

        metadata = {
            'project_info': {
                'title': story.title,
                'generated': datetime.now().isoformat(),
                'version': '5.0 Enhanced with AutoResume'
            },
            'cost': make_json_serializable(self.cost_tracker.get_summary()),
            'story_structure': make_json_serializable(story),
            'cost_summary': make_json_serializable(cost_summary),
            'generated_files': serializable_files,
            'validation': {
                'story_valid': self.validate_story_structure(story)[0],
                'last_validated': datetime.now().isoformat()
            }
        }

        metadata_path = project_dir / "project_metadata.json"
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            # Fallback with even more aggressive serialization
            print(f"⚠️ Standard serialization failed, using fallback: {e}")
            metadata_safe = make_json_serializable(metadata)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_safe, f, indent=2, ensure_ascii=False, default=str)

    def create_summary(self, story: StoryData, cost_summary: Dict, generated_files: Dict, project_dir: Path):
        """Create project summary with validation info"""
        status = self.check_project_status(project_dir)

        summary = f"""# {story.title} - Enhanced Production Summary with AutoResume

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Story Details
- **Genre:** {story.genre}
- **Style:** {story.style_preset.image_style.value.title()}
- **Characters:** {len(story.characters)}
- **Scenes:** {len(story.scenes)}
- **World:** {story.world_desc}

### Project Status & Validation
- **Completion:** {status['completion_percentage']:.1f}%
- **Story Structure:** {'✅ Valid' if self.validate_story_structure(story)[0] else '⚠️ Has warnings'}

### Generated Assets (Enhanced)
- **Images:** {status['images']} {'✅' if status['images'] > 0 else '❌'}
- **Audio:** {status['audio']} files {'✅' if status['audio'] > 0 else '❌'}
- **Video:** {status['video']} files {'✅' if status['video'] > 0 else '❌'}
- **PDF:** {status['pdf']} files {'✅' if status['pdf'] > 0 else '❌'}

### Enhanced Features
- ✅ AutoResume functionality - continue interrupted projects
- ✅ Validation at each step
- ✅ Multiple world establishment images (2)
- ✅ Multiple scene perspectives (2-4 per scene)
- ✅ Chronological video sequence (one scene per story beat)
- ✅ Complete PDF with all generated images
- ✅ Different camera angles and viewpoints
- ✅ Character interaction variations

### Cost Summary
- **Total Cost:** ${cost_summary.get('total_cost_usd', 0):.3f}
- **Kokoro TTS:** {cost_summary['breakdown']['kokoro']['calls']} calls (${cost_summary['breakdown']['kokoro']['cost']:.3f})
- **Flux Schnell:** {cost_summary['breakdown']['flux_schnell']['calls']} calls (${cost_summary['breakdown']['flux_schnell']['cost']:.3f})
- **Flux KREA:** {cost_summary['breakdown']['flux_krea']['calls']} calls (${cost_summary['breakdown']['flux_krea']['cost']:.3f})
- **Flux kontext:** {cost_summary['breakdown']['flux_kontext']['calls']} calls (${cost_summary['breakdown']['flux_kontext']['cost']:.3f})
- **BANAN:** {cost_summary['breakdown']['banana']['calls']} calls (${cost_summary['breakdown']['banana']['cost']:.3f})
- **MINIMAX (clips):** {cost_summary['breakdown']['minimax']['calls']} calls (${cost_summary['breakdown']['minimax']['cost']:.3f})
- **ElevenLabs:** {cost_summary['breakdown']['elevenlabs']['calls']} tokens (${cost_summary['breakdown']['elevenlabs']['cost']:.3f})

### Project Location
`{project_dir}`

### Resume Information
- **Prompt Hash:** {self.get_prompt_hash(story.world_desc)}
- **Can Resume:** Yes - use same prompt to continue this project

---
*Generated by Enhanced Multimedia Story Generator v5.0 with AutoResume*
*Features: AutoResume | Validation | Multiple World Images | Scene Perspectives | Chronological Video | Complete PDF*
"""

        summary_path = project_dir / "PROJECT_SUMMARY.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)

    def move_final_media(self):
        """Move the final 2 videos (images and clip based), audio book, and PDF to a final media folder"""

        # Get the project directory from the project manager
        project_dir = self.base_dir

        # Create final media folder in the project directory
        final_media_dir = project_dir / "final_media"
        final_media_dir.mkdir(exist_ok=True)

        # Find and move the final videos
        video_dir = project_dir / "video"
        if video_dir.exists():
            video_files = list(video_dir.glob("*_final.mp4"))
            clip_video_files = list(video_dir.glob("*_clips_final.mp4"))

            # Move up to 2 final videos
            final_videos = video_files + clip_video_files
            for video_file in final_videos[:2]:  # Limit to 2 videos
                if video_file.exists():
                    destination = final_media_dir / video_file.name
                    shutil.move(str(video_file), str(destination))
                    print(f"Moved video: {video_file.name}")

        # Find and move the audio book
        audio_dir = project_dir / "audio"
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*_complete.wav"))
            for audio_file in audio_files:
                if audio_file.exists():
                    destination = final_media_dir / audio_file.name
                    shutil.move(str(audio_file), str(destination))
                    print(f"Moved audio: {audio_file.name}")
                    break  # Only move first audio file

        # Find and move the PDF
        pdf_locations = [project_dir / "pdf", project_dir]
        for location in pdf_locations:
            if location.exists():
                pdf_files = list(location.glob("*.pdf"))
                for pdf_file in pdf_files:
                    if pdf_file.exists():
                        destination = final_media_dir / pdf_file.name
                        shutil.move(str(pdf_file), str(destination))
                        print(f"Moved PDF: {pdf_file.name}")
                        break  # Only move first PDF file

        print(f"Final media moved to: {final_media_dir}")
