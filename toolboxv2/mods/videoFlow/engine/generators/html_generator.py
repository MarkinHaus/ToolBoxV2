# toolboxv2/mods/videoFlow/engine/generators/html_generator.py

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from toolboxv2.mods.videoFlow.engine.generators.html_assets import get_dark_mode_css, get_fixed_javascript
from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData


class MultiMediaStoryHTMLGenerator:
    """
    Enhanced HTML generator v2.0 with DARK MODE and FIXED media integration
    Creates complete single-page multimedia experience with all media properly loaded
    """

    def __init__(self, logger=None):
        self.logger = logger or self._create_default_logger()

    def _create_default_logger(self):
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def create_complete_html_experience(self, story_data: StoryData, project_dir: Path,
                                        output_filename: str = None) -> Optional[Path]:
        """
        Create complete single-page HTML multimedia experience with FIXED media integration
        """
        try:
            # Create HTML directory
            html_dir = project_dir / "html"
            html_dir.mkdir(exist_ok=True)

            # Setup HTML file path
            if not output_filename:
                safe_title = re.sub(r'[<>:"/\\|?* ]', '_', story_data.title)[:30]
                output_filename = f"{safe_title}_complete_experience.html"

            html_path = html_dir / output_filename

            # FIXED: Copy all media files to html directory for proper access
            self._copy_media_files(project_dir, html_dir)

            # Organize all media with CORRECTED paths
            organized_media = self._organize_media_with_correct_paths(project_dir, html_dir, story_data)

            # Generate complete HTML with DARK MODE
            html_content = self._generate_complete_dark_html(story_data, organized_media, project_dir)

            # Write HTML file
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info(f"Enhanced HTML experience created: {html_path.name}")
            self.logger.info(f"Media files organized: {organized_media.get('all_media_count', 0)} total files")

            return html_path

        except Exception as e:
            self.logger.error(f"Failed to create HTML experience: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _copy_media_files(self, project_dir: Path, html_dir: Path):
        """Copy all media files to HTML directory for proper access"""
        media_dirs = ['images', 'audio', 'video', 'clips', 'transitions']

        for media_type in media_dirs:
            source_dir = project_dir / media_type
            if source_dir.exists():
                dest_dir = html_dir / media_type
                dest_dir.mkdir(exist_ok=True)

                # Copy all files
                for file_path in source_dir.iterdir():
                    if file_path.is_file():
                        dest_path = dest_dir / file_path.name
                        try:
                            shutil.copy2(file_path, dest_path)
                            self.logger.info(f"Copied: {file_path.name}")
                        except Exception as e:
                            self.logger.warning(f"Failed to copy {file_path.name}: {e}")

    def _classify_image_by_name(self, filename: str) -> str:
        """Classify image by filename using regex patterns"""
        name = filename.lower()

        # Character files (highest priority)
        if re.search(r'char|character', name):
            return 'character'

        # Scene files
        if re.search(r'scene_\d+', name):
            return 'scene'

        # Cover files
        if re.search(r'cover|^00_(?!char)', name):
            return 'cover'

        # World files
        if re.search(r'world|^01_(?!char)', name):
            return 'world'

        # End files
        if re.search(r'end|^99_', name):
            return 'end'

        return 'unknown'
    def _organize_media_with_correct_paths(self, project_dir: Path, html_dir: Path, story_data) -> Dict:
        """Organize media with CORRECTED paths that actually exist"""
        organized = {
            'cover_image': None,
            'final_videos': [],
            'world_images': [],
            'character_images': [],
            'scenes': [],
            'audio_complete': None,
            'audio_segments': [],
            'video_clips': [],
            'end_image': None,
            'all_media_count': 0,
            'pdf_files': [],
            'metadata': {
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'style': story_data.style_preset.image_style.value,
                'camera': story_data.style_preset.camera_style.value
            }
        }

        # Initialize scenes
        for i in range(len(story_data.scenes)):
            organized['scenes'].append({
                'scene_idx': i,
                'images': [],
                'clips': [],
                'audio': [],
                'story_data': story_data.scenes[i]
            })

        # FIXED: Find images with correct paths
        images_dir = html_dir / "images"
        if images_dir.exists():
            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    relative_path = f"images/{img_path.name}"
                    name = img_path.name.lower()
                    organized['all_media_count'] += 1
                    img_type = self._classify_image_by_name(img_path.name)

                    if img_type == 'character':
                        organized['character_images'].append(relative_path)
                        self.logger.info(f"Found character: {img_path.name}")
                    elif img_type == 'cover':
                        organized['cover_image'] = relative_path
                        self.logger.info(f"Found cover: {img_path.name}")
                    elif img_type == 'world':
                        organized['world_images'].append(relative_path)
                        self.logger.info(f"Found world: {img_path.name}")
                    elif 'scene' in name:
                        scene_match = re.search(r'scene_(\d+)', name)
                        if scene_match:
                            scene_idx = int(scene_match.group(1))
                            if scene_idx < len(organized['scenes']):
                                organized['scenes'][scene_idx]['images'].append(relative_path)
                                self.logger.info(f"Found scene image: {img_path.name} -> Scene {scene_idx}")
                    elif name.startswith('99_') or 'end' in name:
                        organized['end_image'] = relative_path
                        self.logger.info(f"Found end: {img_path.name}")
                    else:
                        self.logger.warning(f"Unclassified image: {img_path.name}")

        transitions_dir = html_dir / "transitions"
        if transitions_dir.exists():
            for img_path in sorted(transitions_dir.iterdir()):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    relative_path = f"transitions/{img_path.name}"
                    name = img_path.name.lower()
                    organized['all_media_count'] += 1

                    # Match transition to scene: transition_scene_XX_to_perspective_YY.png
                    transition_match = re.search(r'transition_scene_(\d+)_to_perspective_(\d+)', name)
                    if transition_match:
                        scene_idx = int(transition_match.group(1))
                        perspective_idx = int(transition_match.group(2))

                        if scene_idx < len(organized['scenes']):
                            if 'transitions' not in organized['scenes'][scene_idx]:
                                organized['scenes'][scene_idx]['transitions'] = []
                            organized['scenes'][scene_idx]['transitions'].append({
                                'path': relative_path,
                                'perspective_idx': perspective_idx
                            })
                            self.logger.info(
                                f"Found transition: {img_path.name} -> Scene {scene_idx}, Perspective {perspective_idx}")

        pdf_locations = [html_dir.parent, html_dir.parent / "pdf"]
        for pdf_location in pdf_locations:
            if pdf_location.exists():
                for pdf_path in pdf_location.glob("*.pdf"):
                    relative_path = f"../{pdf_path.relative_to(html_dir.parent)}"
                    organized['pdf_files'].append({
                        'path': relative_path,
                        'name': pdf_path.name,
                        'size': pdf_path.stat().st_size
                    })
                    organized['all_media_count'] += 1
                    self.logger.info(f"Found PDF: {pdf_path.name}")

        # FIXED: Find audio files
        audio_dir = html_dir / "audio"
        if audio_dir.exists():
            for audio_path in sorted(audio_dir.iterdir()):
                if audio_path.suffix.lower() in ['.wav', '.mp3']:
                    relative_path = f"audio/{audio_path.name}"
                    name = audio_path.name.lower()
                    organized['all_media_count'] += 1

                    if 'complete' in name:
                        organized['audio_complete'] = relative_path
                        self.logger.info(f"Found complete audio: {audio_path.name}")
                    else:
                        organized['audio_segments'].append(relative_path)

                        # Match to scene
                        scene_match = re.search(r'scene_(\d+)', name)
                        if scene_match:
                            scene_idx = int(scene_match.group(1))
                            if scene_idx < len(organized['scenes']):
                                organized['scenes'][scene_idx]['audio'].append(relative_path)

        # FIXED: Find video files
        video_dir = html_dir / "video"
        if video_dir.exists():
            for video_path in sorted(video_dir.iterdir()):
                if video_path.suffix.lower() in ['.mp4', '.webm']:
                    relative_path = f"video/{video_path.name}"
                    organized['all_media_count'] += 1

                    if 'final' in video_path.name.lower():
                        organized['final_videos'].append(relative_path)
                        self.logger.info(f"Found final video: {video_path.name}")

        # FIXED: Find clip files
        clips_dir = html_dir / "clips"
        if clips_dir.exists():
            for clip_path in sorted(clips_dir.iterdir()):
                if clip_path.suffix.lower() in ['.mp4', '.webm']:
                    relative_path = f"clips/{clip_path.name}"
                    name = clip_path.name.lower()
                    organized['all_media_count'] += 1
                    organized['video_clips'].append(relative_path)

                    # Match to scene
                    scene_match = re.search(r'scene_(\d+)', name)
                    if scene_match:
                        scene_idx = int(scene_match.group(1))
                        if scene_idx < len(organized['scenes']):
                            organized['scenes'][scene_idx]['clips'].append(relative_path)
                            self.logger.info(f"Found scene clip: {clip_path.name} -> Scene {scene_idx}")

        total_transitions = sum(len(scene.get('transitions', [])) for scene in organized['scenes'])
        if total_transitions > 0:
            self.logger.info(f"Found {total_transitions} transition images across {len(organized['scenes'])} scenes")
        else:
            self.logger.info("No transition images found")

        self.logger.info(f"Media organization complete: {organized['all_media_count']} files found")
        return organized

    def _generate_complete_dark_html(self, story_data, organized_media: Dict, project_dir: Path) -> str:
        """Generate complete HTML with DARK MODE and FIXED media display"""

        # Generate dark color scheme
        colors = self._generate_dark_colors(story_data.style_preset)

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{story_data.title} - Complete Multimedia Experience</title>

    <style>
        {get_dark_mode_css(story_data, colors)}
    </style>
</head>
<body>
    <!-- FIXED Audio Book Player Popup -->
    {self._generate_fixed_audio_player(organized_media)}

    <!-- Main Content -->
    <div class="main-container">

        <!-- Header: Title Image + Final Video -->
        {self._generate_header_with_media(story_data, organized_media)}

        <!-- Production Details -->
        {self._generate_production_info(story_data, organized_media)}

        <!-- World & Setting with ALL world images -->
        {self._generate_world_section(story_data, organized_media)}

        <!-- Characters with character images -->
        {self._generate_characters_gallery(story_data, organized_media)}

        <!-- COMPLETE Story with ALL scene media -->
        {self._generate_complete_story_experience(story_data, organized_media)}


        <!-- PDF Documents Section -->
        {self._generate_pdf_section(organized_media)}

        <!-- Final ending with end image -->
        {self._generate_finale_section(organized_media)}

        <!-- End Card -->
        {self._generate_end_card(story_data, organized_media)}

    </div>

    <!-- FIXED JavaScript -->
    <script>
        {get_fixed_javascript()}
    </script>

</body>
</html>"""

        return html_template

    def _generate_pdf_section(self, organized_media: Dict) -> str:
        """Generate PDF display and download section"""

        pdf_files = organized_media.get('pdf_files', [])

        if not pdf_files:
            return ''

        content = f"""
        <section class="content-section">
            <h2 class="section-title fade-in-up">📄 Complete Story Documents</h2>

            <div class="pdf-showcase">
        """

        for pdf_info in pdf_files:
            pdf_path = pdf_info['path']
            pdf_name = pdf_info['name']
            pdf_size = pdf_info['size'] / (1024 * 1024)  # Convert to MB

            content += f"""
            <div class="pdf-container fade-in-up">
                <div class="pdf-header">
                    <h3 class="pdf-title">📖 {pdf_name}</h3>
                    <div class="pdf-meta">
                        <span>Size: {pdf_size:.1f} MB</span>
                        <button class="download-btn" onclick="downloadPDF('{pdf_path}', '{pdf_name}')">
                            💾 Download PDF
                        </button>
                    </div>
                </div>

                <div class="pdf-viewer-container">
                    <iframe
                        src="{pdf_path}#toolbar=1&navpanes=1&scrollbar=1"
                        class="pdf-viewer"
                        title="PDF Viewer - {pdf_name}"
                        loading="lazy">
                        <p>Your browser doesn't support PDF viewing.
                        <a href="{pdf_path}" download="{pdf_name}">Download the PDF instead</a></p>
                    </iframe>

                    <div class="pdf-controls">
                        <button onclick="openPDFFullscreen('{pdf_path}')" class="pdf-btn">🔍 View Fullscreen</button>
                        <button onclick="printPDF('{pdf_path}')" class="pdf-btn">🖨️ Print</button>
                        <a href="{pdf_path}" download="{pdf_name}" class="pdf-btn">📥 Download</a>
                    </div>
                </div>
            </div>
            """

        content += """
            </div>
        </section>
        """

        return content

    def _generate_dark_colors(self, style_preset) -> Dict[str, str]:
        """Generate DARK MODE color schemes for ALL updated styles"""

        dark_style_colors = {
            # Original styles (updated)
            "realistic": {"bg": "#0a0a0a", "surface": "#1a1a1a", "primary": "#ffffff", "secondary": "#b0b0b0",
                          "accent": "#4a9eff"},
            "cartoon": {"bg": "#1a1a2e", "surface": "#16213e", "primary": "#ffffff", "secondary": "#a0a0a0",
                        "accent": "#00d4ff"},
            "anime": {"bg": "#0f0f23", "surface": "#1a1a2e", "primary": "#ffffff", "secondary": "#c9c9c9",
                      "accent": "#ff6b9d"},
            "watercolor": {"bg": "#1e1e2e", "surface": "#2d3748", "primary": "#f7fafc", "secondary": "#cbd5e0",
                           "accent": "#9f7aea"},
            "oil_painting": {"bg": "#1a1a1a", "surface": "#2d2d2d", "primary": "#f5f5f5", "secondary": "#d4d4d4",
                             "accent": "#f6ad55"},
            "digital_art": {"bg": "#000000", "surface": "#111111", "primary": "#00ff88", "secondary": "#888888",
                            "accent": "#ff0066"},
            "pencil_sketch": {"bg": "#1a1a1a", "surface": "#2c2c2c", "primary": "#f0f0f0", "secondary": "#b0b0b0",
                              "accent": "#718096"},
            "cyberpunk": {"bg": "#000000", "surface": "#0a0a0a", "primary": "#00ffff", "secondary": "#ff00ff",
                          "accent": "#00ff00"},
            "fantasy": {"bg": "#0d1117", "surface": "#161b22", "primary": "#f0e6ff", "secondary": "#c9d1d9",
                        "accent": "#da70d6"},
            "noir": {"bg": "#000000", "surface": "#1a1a1a", "primary": "#ffffff", "secondary": "#808080",
                     "accent": "#c0c0c0"},

            # NEW STYLES
            "minimalist": {"bg": "#0f0f0f", "surface": "#1f1f1f", "primary": "#ffffff", "secondary": "#cccccc",
                           "accent": "#666666"},
            "abstract": {"bg": "#1a1a2e", "surface": "#16213e", "primary": "#ffffff", "secondary": "#a8a8a8",
                         "accent": "#ff4757"},
            "retro": {"bg": "#2c1810", "surface": "#3d2817", "primary": "#ffeaa7", "secondary": "#fdcb6e",
                      "accent": "#e17055"},
            "steampunk": {"bg": "#1a1611", "surface": "#2d241b", "primary": "#d4af37", "secondary": "#cd853f",
                          "accent": "#b8860b"},
            "comic_style": {"bg": "#1a1a2e", "surface": "#16213e", "primary": "#ffffff", "secondary": "#feca57",
                            "accent": "#ff6348"},
        }

        style = style_preset.image_style.value
        return dark_style_colors.get(style, dark_style_colors["realistic"])

    def _generate_camera_specific_css(self, camera_style: str) -> str:
        """Generate camera-specific CSS effects"""

        camera_effects = {
            # Black & White styles
            "black_white_classic": "filter: grayscale(100%) contrast(1.2);",
            "film_noir": "filter: grayscale(100%) contrast(1.5) brightness(0.8);",
            "high_contrast_bw": "filter: grayscale(100%) contrast(2.0);",
            "vintage_bw": "filter: grayscale(100%) sepia(0.3) contrast(1.1);",

            # Bright/Colorful effects
            "neon_cyberpunk": "filter: saturate(1.8) contrast(1.3) hue-rotate(10deg);",
            "vaporwave": "filter: saturate(1.5) hue-rotate(270deg) contrast(1.2);",
            "psychedelic": "filter: saturate(2.0) hue-rotate(180deg) contrast(1.4);",
            "rainbow_bright": "filter: saturate(1.8) brightness(1.1) contrast(1.2);",
            "candy_colors": "filter: saturate(1.6) brightness(1.05) contrast(1.1);",
            "miami_vice": "filter: saturate(1.4) hue-rotate(315deg) contrast(1.2);",

            # Special effects
            "glitch_art": "filter: saturate(1.5) contrast(1.3); animation: glitch 2s infinite;",
            "holographic": "filter: saturate(1.4) brightness(1.1); animation: hologram 3s infinite;",
            "thermal_camera": "filter: hue-rotate(200deg) saturate(1.8) contrast(1.3);",
            "infrared": "filter: hue-rotate(270deg) saturate(1.2) contrast(1.4);",

            # Vintage effects
            "polaroid_vintage": "filter: sepia(0.4) saturate(0.8) contrast(0.9);",
            "film_35mm": "filter: sepia(0.2) saturate(1.1) contrast(1.05);",

            # Art styles
            "impressionist": "filter: blur(0.5px) saturate(1.2) contrast(0.9);",
            "expressionist": "filter: saturate(1.6) contrast(1.4) brightness(0.95);",
        }

        return camera_effects.get(camera_style.lower().replace(" ", "_").replace("&", ""), "")

    def _generate_fixed_audio_player(self, organized_media: Dict) -> str:
        """Generate FIXED audio player with proper toggle functionality"""

        audio_complete = organized_media.get('audio_complete', '')

        if not audio_complete:
            return '<!-- No audio file found -->'

        return f"""
        <div class="audio-player-floating" id="audioPlayerPopup">
            <div class="audio-header">
                <button class="close-audio-btn" id="closeAudioBtn" onclick="toggleAudioPlayer()" title="Toggle Audio Player">×</button>
                <strong>📚 Complete Audiobook</strong>
            </div>

            <audio id="mainAudioPlayer" style="width: 100%; margin: 10px 0;" preload="metadata">
                <source src="{audio_complete}" type="audio/wav">
                <source src="{audio_complete}" type="audio/mp3">
                Your browser does not support audio playback.
            </audio>

            <div class="audio-controls">
                <button class="audio-btn" onclick="playPauseAudio()" id="playPauseBtn">▶️ Play</button>
                <button class="audio-btn" onclick="skipBackward()">⏪ -10s</button>
                <button class="audio-btn" onclick="skipForward()">⏩ +10s</button>
            </div>

            <div class="audio-progress" id="audioProgressDisplay">
                Ready to play audiobook
            </div>
        </div>
        """

    def _generate_header_with_media(self, story_data, organized_media: Dict) -> str:
        """Generate header section with CLICKABLE media files"""

        cover_image = organized_media.get('cover_image', '')
        final_videos = organized_media.get('final_videos', [])
        main_video = final_videos[0] if final_videos else ''

        return f"""
        <section class="hero-section">
            {f'<img src="{cover_image}" alt="{story_data.title} Cover" class="hero-title-image clickable-image fade-in-up">' if cover_image else '<div class="loading-shimmer" style="width: 600px; height: 400px; border-radius: 20px; margin-bottom: 2rem;"></div>'}

            <h1 class="hero-title fade-in-up">{story_data.title}</h1>
            <p class="hero-subtitle fade-in-up">A {story_data.genre} Multimedia Experience</p>
            <p class="hero-subtitle fade-in-up">
                {story_data.style_preset.image_style.value.replace('_', ' ').title()}
            </p>

            {f'<video class="hero-video fade-in-up" controls preload="metadata"><source src="{main_video}" type="video/mp4">Your browser does not support video.</video>' if main_video else '<div class="loading-shimmer" style="width: 100%; max-width: 900px; height: 300px; border-radius: 15px;"></div>'}
        </section>
        """

    def _get_camera_description(self, camera_style: str) -> str:
        """Get description for camera style"""
        descriptions = {
            "Black & White Classic": "Timeless monochrome photography",
            "Film Noir": "High contrast dramatic lighting",
            "Neon Cyberpunk": "Futuristic neon-enhanced visuals",
            "Studio Ghibli": "Hand-drawn animation aesthetic",
            "Glitch Art": "Digital distortion effects",
            "Polaroid Vintage": "Instant camera retro feel",
            # Add more as needed
        }
        return descriptions.get(camera_style, "Custom visual treatment")

    def _generate_production_info(self, story_data, organized_media: Dict) -> str:
        """Generate production information section"""

        metadata = organized_media.get('metadata', {})
        media_count = organized_media.get('all_media_count', 0)

        return f"""
        <section class="content-section">
            <h2 class="section-title fade-in-up">Production Details</h2>

            <div class="info-grid">
                <div class="info-card fade-in-up">
                    <h3>🎬 Visual Direction</h3>
                    <p><strong>Style:</strong> {metadata.get('style', 'Unknown').replace('_', ' ').title()}</p>
                    <p><strong>Camera:</strong> {story_data.style_preset.camera_style.value}</p>
                    <p><strong>Effect:</strong> {self._get_camera_description(story_data.style_preset.camera_style.value)}</p>
                </div>

                <div class="info-card fade-in-up">
                    <h3>📊 Story Statistics</h3>
                    <p><strong>Characters:</strong> {len(story_data.characters)} main characters</p>
                    <p><strong>Scenes:</strong> {len(story_data.scenes)} narrative scenes</p>
                    <p><strong>Genre:</strong> {story_data.genre}</p>
                </div>

                <div class="info-card fade-in-up">
                    <h3>🎭 Character Cast</h3>
                    {self._generate_character_list_for_info(story_data)}
                </div>

                <div class="info-card fade-in-up">
                    <h3>📁 Media Assets</h3>
                    <p><strong>Total Files:</strong> {media_count} media files</p>
                    <p><strong>Transitions:</strong> {sum(len(scene.get('transitions', [])) for scene in organized_media.get('scenes', []))} transition effects</p>
                    <p><strong>Generated:</strong> {metadata.get('generation_time', 'Unknown')}</p>
                    <p><strong>Format:</strong> Interactive HTML Experience</p>
                </div>
            </div>
        </section>
        """

    def _generate_character_list_for_info(self, story_data) -> str:
        """Generate compact character list"""
        char_list = ""
        for char in story_data.characters[:4]:  # Max 4 characters for compact display
            char_list += f"<p><strong>{char.name}:</strong> {char.role.value.title()}</p>"
        if len(story_data.characters) > 4:
            char_list += f"<p><em>+{len(story_data.characters) - 4} more characters</em></p>"
        return char_list

    def _generate_world_section(self, story_data, organized_media: Dict) -> str:
        """Generate world section with CLICKABLE world images"""

        world_images = organized_media.get('world_images', [])

        content = f"""
        <section class="content-section">
            <h2 class="section-title fade-in-up">World & Setting</h2>

            <div class="narrator-content fade-in-up">
                {story_data.world_desc}
            </div>
        """

        if world_images:
            content += '<div class="media-showcase">'
            for i, world_img in enumerate(world_images):
                content += f"""
                <div class="media-card fade-in-up">
                    <img src="{world_img}" alt="World View {i + 1}" class="clickable-image" loading="lazy" onerror="this.style.display='none'">
                    <div class="media-info">
                        <div class="media-title">World Environment {i + 1}</div>
                        <div class="media-description">Environmental establishment shot showing the atmospheric setting of our story world.</div>
                    </div>
                </div>
                """
            content += '</div>'
        else:
            content += '<div class="media-info fade-in-up"><p>No world images found in media directory.</p></div>'

        content += '</section>'
        return content

    def _generate_characters_gallery(self, story_data, organized_media: Dict) -> str:
        """Generate characters section with CLICKABLE character images"""

        character_images = organized_media.get('character_images', [])

        content = f"""
        <section class="content-section">
            <h2 class="section-title fade-in-up">Character Gallery</h2>

            <div class="media-showcase">
        """

        for i, character in enumerate(story_data.characters):
            char_img = character_images[i] if i < len(character_images) else ''

            content += f"""
            <div class="media-card fade-in-up">
                {f'<img src="{char_img}" alt="{character.name}" class="clickable-image" loading="lazy" onerror="this.style.display=\'none\'">' if char_img else '<div style="height: 280px; background: var(--glass-bg); display: flex; align-items: center; justify-content: center; color: var(--secondary-color);">Character Image Not Found</div>'}
                <div class="media-info">
                    <div class="media-title">{character.name}</div>
                    <div class="media-description">
                        <strong>Role:</strong> {character.role.value.title()}<br>
                        <strong>Voice:</strong> {character.voice.value.replace('_', ' ').title()}<br>
                        {character.visual_desc}
                    </div>
                </div>
            </div>
            """

        content += """
            </div>
        </section>
        """

        return content

    def _generate_complete_story_experience(self, story_data, organized_media: Dict) -> str:
        """Generate complete story with CLICKABLE scene media"""

        scenes = organized_media.get('scenes', [])

        content = f"""
        <section class="content-section">
            <h2 class="section-title fade-in-up">Complete Story Experience</h2>
        """

        for i, scene_data in enumerate(scenes):
            if i >= len(story_data.scenes):
                continue

            story_scene = story_data.scenes[i]
            scene_images = scene_data.get('images', [])
            scene_transitions = scene_data.get('transitions', [])
            scene_clips = scene_data.get('clips', [])
            content += f"""
            <div class="story-scene fade-in-up" >
                <h3 class="scene-title">Scene {i + 1}: {story_scene.title}</h3>

                <div class="narrator-content">
                    <strong>Setting:</strong> {story_scene.setting}<br><br>
                    {story_scene.narrator}
                </div>
            """

            # Add ALL scene images with clickable functionality
            if scene_images:
                content += '<div class="media-showcase" style="margin: 2rem 0;">'
                for img_idx, scene_img in enumerate(scene_images):
                    perspective_match = re.search(r'perspective_(\d+)', scene_img)
                    perspective_info = f"Perspective {int(perspective_match.group(1)) + 1}" if perspective_match else f"View {img_idx + 1}"

                    content += f"""
                    <div class="media-card">
                        <img src="{scene_img}" alt="Scene {i + 1} {perspective_info}" class="clickable-image" loading="lazy" onerror="this.style.display='none'">
                        <div class="media-info">
                            <div class="media-title">{perspective_info}</div>
                            <div class="media-description">{story_scene.setting}</div>
                        </div>
                    </div>
                    """
                content += '</div>'

            # Add ALL scene clips (UNCUT!)
            if scene_clips:
                content += '<div class="media-showcase" style="margin: 2rem 0;">'
                for clip_idx, scene_clip in enumerate(scene_clips):
                    content += f"""
                    <div class="media-card{'-gold' if story_scene.duration >= 11 else ''}">
                        <video controls preload="metadata" style="width: 100%;">
                            <source src="{scene_clip}" type="video/mp4">
                            Your browser does not support video.
                        </video>
                        <div class="media-info">
                            <div class="media-title">Scene {i + 1} Video Clip {clip_idx + 1}</div>
                            <div class="media-description">AI-generated scene animation - uncut full clip</div>
                        </div>
                    </div>
                    """
                content += '</div>'

            # Add clickable transitions
            if scene_transitions:
                content += '<div class="transitions-section" style="margin: 1.5rem 0;">'
                content += '<h4 style="color: var(--accent-color); font-size: 1.1rem; margin-bottom: 1rem; text-align: center;">🎬 Scene Transitions</h4>'
                content += '<div class="media-showcase" style="grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));">'

                for trans_idx, transition in enumerate(sorted(scene_transitions, key=lambda x: x['perspective_idx'])):
                    content += f"""
                    <div class="media-card transition-card">
                        <img src="{transition['path']}" alt="Scene {i + 1} Transition {transition['perspective_idx'] + 1}" class="clickable-image" loading="lazy" onerror="this.style.display='none'">
                        <div class="media-info">
                            <div class="media-title">Transition {transition['perspective_idx'] + 1}</div>
                            <div class="media-description">Scene transition effect</div>
                        </div>
                    </div>
                    """
                content += '</div></div>'

            # Add dialogue
            if story_scene.dialogue:
                for dialogue in story_scene.dialogue:
                    content += f"""
                    <div class="dialogue-container">
                        <div class="character-speaker">{dialogue.character}:</div>
                        <div class="dialogue-speech">"{dialogue.text}"</div>
                    </div>
                    """

            content += '</div>'

        content += '</section>'
        return content

    def _generate_finale_section(self, organized_media: Dict) -> str:
        """Generate finale section with CLICKABLE end image"""

        end_image = organized_media.get('end_image', '')

        if not end_image:
            return ''

        return f"""
        <section class="content-section">
            <h2 class="section-title fade-in-up">The End</h2>

            <div class="media-showcase" style="justify-content: center;">
                <div class="media-card fade-in-up" style="max-width: 600px;">
                    <img src="{end_image}" alt="Story Conclusion" class="clickable-image" loading="lazy" onerror="this.style.display='none'">
                    <div class="media-info">
                        <div class="media-title">Story Conclusion</div>
                        <div class="media-description">The final moment of our multimedia journey</div>
                    </div>
                </div>
            </div>
        </section>
        """

    def _generate_end_card(self, story_data, organized_media: Dict) -> str:
        """Generate final end card"""

        return f"""
        <section class="finale-section">
            <h1 class="finale-title">THE END</h1>
            <p style="font-size: 1.5rem; margin-bottom: 2rem; color: var(--secondary-color);">Thank you for experiencing</p>
            <p style="font-size: 2.5rem; margin-bottom: 2rem; font-weight: bold; color: var(--accent-color);">{story_data.title}</p>
            <p style="font-size: 1.3rem; color: var(--secondary-color); opacity: 0.8;">A complete multimedia story experience</p>
            <p style="font-size: 1rem; margin-top: 3rem; opacity: 0.6; color: var(--secondary-color);">
                Style: {story_data.style_preset.image_style.value.title()} •
                Generated with AI
            </p>
        </section>
        """
