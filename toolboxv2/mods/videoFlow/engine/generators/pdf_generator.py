# toolboxv2/mods/videoFlow/engine/generators/pdf_generator.py

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak

from toolboxv2.mods.videoFlow.engine.models.base_models import StoryData


class PDFGenerator:
    """Production-ready PDF generator with complete image integration and generation data"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.styles = self._create_styles()

    def _create_styles(self):
        """Create professional PDF styles optimized for complete story presentation"""
        base_styles = getSampleStyleSheet()

        return {
            # Title page styles
            'title_main': ParagraphStyle(
                'TitleMain',
                parent=base_styles['Title'],
                fontSize=32,
                alignment=TA_CENTER,
                spaceAfter=20,
                textColor=HexColor('#1a1a1a'),
                fontName='Helvetica-Bold'
            ),
            'title_subtitle': ParagraphStyle(
                'TitleSubtitle',
                parent=base_styles['Normal'],
                fontSize=16,
                alignment=TA_CENTER,
                spaceAfter=15,
                textColor=HexColor('#4a4a4a'),
                fontName='Helvetica'
            ),

            # Story page styles - narrator at top
            'narrator_text': ParagraphStyle(
                'NarratorText',
                parent=base_styles['Normal'],
                fontSize=12,
                alignment=TA_JUSTIFY,
                spaceAfter=10,
                spaceBefore=10,
                leftIndent=20,
                rightIndent=20,
                textColor=HexColor('#2c3e50'),
                fontName='Helvetica',
                leading=16
            ),

            # Story page styles - dialogue at bottom
            'dialogue_text': ParagraphStyle(
                'DialogueText',
                parent=base_styles['Normal'],
                fontSize=11,
                alignment=TA_LEFT,
                spaceAfter=6,
                spaceBefore=3,
                leftIndent=30,
                textColor=HexColor('#34495e'),
                fontName='Helvetica-Oblique',
                leading=14
            ),
            'character_name': ParagraphStyle(
                'CharacterName',
                parent=base_styles['Normal'],
                fontSize=11,
                alignment=TA_LEFT,
                spaceAfter=3,
                leftIndent=30,
                textColor=HexColor('#e74c3c'),
                fontName='Helvetica-Bold'
            ),

            # Section headers
            'section_header': ParagraphStyle(
                'SectionHeader',
                parent=base_styles['Heading1'],
                fontSize=18,
                alignment=TA_CENTER,
                spaceBefore=20,
                spaceAfter=15,
                textColor=HexColor('#2c3e50'),
                fontName='Helvetica-Bold'
            ),

            # Scene headers
            'scene_header': ParagraphStyle(
                'SceneHeader',
                parent=base_styles['Heading2'],
                fontSize=14,
                alignment=TA_CENTER,
                spaceBefore=15,
                spaceAfter=10,
                textColor=HexColor('#34495e'),
                fontName='Helvetica-Bold'
            ),

            # Image captions
            'image_caption': ParagraphStyle(
                'ImageCaption',
                parent=base_styles['Normal'],
                fontSize=9,
                alignment=TA_CENTER,
                spaceAfter=8,
                textColor=HexColor('#7f8c8d'),
                fontName='Helvetica-Oblique'
            ),

            # Character info
            'character_info': ParagraphStyle(
                'CharacterInfo',
                parent=base_styles['Normal'],
                fontSize=10,
                alignment=TA_LEFT,
                spaceAfter=8,
                textColor=HexColor('#2c3e50'),
                fontName='Helvetica'
            ),
            # Metadata info
            'metadata_info': ParagraphStyle(
                'MetadataInfo',
                parent=base_styles['Normal'],
                fontSize=9,
                alignment=TA_LEFT,
                spaceAfter=5,
                textColor=HexColor('#666666'),
                fontName='Helvetica'
            )
        }

    def create_complete_pdf(self, story: StoryData, images: Dict[str, List[Path]], project_dir: Path,
                            cost_summary: Dict = None) -> Optional[Path]:
        """Create complete PDF with all images, full story integration, and generation data"""
        pdf_dir = project_dir / "pdf"
        pdf_dir.mkdir(exist_ok=True)

        safe_title = re.sub(r'[<>:"/\\|?* ]', '_', story.title)[:30]
        pdf_path = pdf_dir / f"{safe_title}_complete_full.pdf"

        try:
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=letter,
                rightMargin=50,
                leftMargin=50,
                topMargin=50,
                bottomMargin=50
            )

            flowables = []

            # Find all images properly
            all_images_organized = self._organize_all_images(images, project_dir)
            self.logger.info(
                f"Found images: cover={bool(all_images_organized['cover'])}, end={bool(all_images_organized['end'])}, scenes={len(all_images_organized['scenes'])}")

            # 1. TITLE PAGE with 00_cover.png
            flowables.extend(self._create_title_page_with_cover(story, all_images_organized['cover'], cost_summary))
            flowables.append(PageBreak())

            # 2. COMPLETE STORY PAGES (each scene with ALL available images)
            flowables.extend(self._create_complete_story_pages(story, all_images_organized))
            flowables.append(PageBreak())

            # 3. END PAGE with 99_end.png
            flowables.extend(self._create_end_page_with_image(all_images_organized['end']))
            flowables.append(PageBreak())

            # 4. ADDITIONAL GENERATION DATA AND IMAGES
            flowables.extend(self._create_generation_data_section(story, all_images_organized, cost_summary))

            # Build PDF
            doc.build(flowables)

            self.logger.info(f"Complete PDF with all images created: {pdf_path.name}")
            return pdf_path

        except Exception as e:
            self.logger.error(f"PDF creation failed: {e}")
            return None

    def _organize_all_images(self, images: Dict[str, List[Path]], project_dir: Path) -> Dict:
        """Organize all images from various sources and find cover/end images"""
        organized = {
            'cover': None,
            'end': None,
            'world': [],
            'characters': [],
            'scenes': {},  # {scene_idx: [images]}
            'all_scene_images': []
        }

        # Search for images in project directory
        images_dir = project_dir / "images"
        if images_dir.exists():
            for img_file in images_dir.glob("*.png"):
                name = img_file.name.lower()

                # Find cover image (00_cover.png)
                if name.startswith('00_') or 'cover' in name:
                    organized['cover'] = img_file
                    self.logger.info(f"Found cover image: {img_file.name}")

                # Find end image (99_end.png)
                elif name.startswith('99_') or 'end' in name:
                    organized['end'] = img_file
                    self.logger.info(f"Found end image: {img_file.name}")

                # Find world images
                elif name.startswith('01_') or 'world' in name:
                    organized['world'].append(img_file)

                # Find character images
                elif 'char' in name or name.startswith('02_'):
                    organized['characters'].append(img_file)

                # Find scene images
                elif 'scene' in name:
                    scene_match = re.search(r'scene_(\d+)', name)
                    if scene_match:
                        scene_idx = int(scene_match.group(1))
                        if scene_idx not in organized['scenes']:
                            organized['scenes'][scene_idx] = []
                        organized['scenes'][scene_idx].append(img_file)
                        organized['all_scene_images'].append(img_file)

        # Also check from images dict parameter
        for key, img_list in images.items():
            if key == 'cover' and img_list and not organized['cover']:
                organized['cover'] = img_list[0] if img_list[0].exists() else None
            elif key == 'end' and img_list and not organized['end']:
                organized['end'] = img_list[0] if img_list[0].exists() else None
            elif key in ['world', 'world_images']:
                organized['world'].extend([img for img in img_list if img.exists()])
            elif key in ['character_refs', 'characters']:
                organized['characters'].extend([img for img in img_list if img.exists()])
            elif 'scene' in key:
                organized['all_scene_images'].extend([img for img in img_list if img.exists()])

        # Sort scene images
        for scene_idx in organized['scenes']:
            organized['scenes'][scene_idx].sort(key=lambda x: x.name)

        organized['world'].sort(key=lambda x: x.name)
        organized['characters'].sort(key=lambda x: x.name)

        return organized

    def _create_title_page_with_cover(self, story: StoryData, cover_image: Optional[Path],
                                      cost_summary: Dict = None) -> List:
        """Create title page with 00_cover.png and generation metadata"""
        elements = []

        # Cover image 00_cover.png
        if cover_image and cover_image.exists():
            try:
                cover_img = Image(str(cover_image), width=6 * inch, height=4.5 * inch)
                cover_img.hAlign = 'CENTER'
                elements.append(cover_img)
                elements.append(Spacer(1, 0.3 * inch))
                self.logger.info(f"Added cover image to PDF: {cover_image.name}")
            except Exception as e:
                self.logger.warning(f"Could not add cover image: {e}")
                elements.append(Spacer(1, 3 * inch))
        else:
            self.logger.warning("No cover image found (00_cover.png)")
            elements.append(Spacer(1, 3 * inch))

        # Title and basic info
        elements.append(Paragraph(story.title, self.styles['title_main']))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(f"A {story.genre} Story", self.styles['title_subtitle']))
        elements.append(
            Paragraph(f"Visual Style: {story.style_preset.image_style.value.title()}", self.styles['title_subtitle']))
        elements.append(
            Paragraph(f"Camera Style: {story.style_preset.camera_style.value.title()}", self.styles['title_subtitle']))

        # Generation metadata
        elements.append(Spacer(1, 0.4 * inch))
        elements.append(
            Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", self.styles['metadata_info']))
        elements.append(Paragraph(f"Characters: {len(story.characters)} | Scenes: {len(story.scenes)}",
                                  self.styles['metadata_info']))

        if cost_summary:
            total_cost = cost_summary.get('total_cost_usd', 0)
            elements.append(Paragraph(f"Generation Cost: ${total_cost:.3f} USD", self.styles['metadata_info']))

            # Cost breakdown
            breakdown = cost_summary.get('breakdown', {})
            cost_details = []
            for service, details in breakdown.items():
                if details.get('calls', 0) > 0:
                    cost_details.append(f"{service}: {details['calls']} calls (${details['cost']:.3f})")

            if cost_details:
                elements.append(Paragraph("Cost Breakdown: " + " | ".join(cost_details), self.styles['metadata_info']))

        # World description
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(story.world_desc, self.styles['narrator_text']))

        return elements

    def _create_complete_story_pages(self, story: StoryData, all_images_organized: Dict) -> List:
        """Create complete story pages with ALL scene images integrated"""
        elements = []

        elements.append(Paragraph("Complete Visual Story", self.styles['section_header']))
        elements.append(Spacer(1, 0.3 * inch))

        for scene_idx, scene in enumerate(story.scenes):
            # Scene header
            elements.append(Paragraph(f"Scene {scene_idx + 1}: {scene.title}", self.styles['scene_header']))
            elements.append(Spacer(1, 0.2 * inch))

            # Narrator text at TOP
            if scene.narrator:
                elements.append(Paragraph(scene.narrator, self.styles['narrator_text']))
                elements.append(Spacer(1, 0.2 * inch))

            # ALL SCENE IMAGES for this scene
            scene_images = all_images_organized['scenes'].get(scene_idx, [])
            if scene_images:
                elements.append(
                    Paragraph(f"Visual Perspectives ({len(scene_images)} images)", self.styles['character_info']))
                elements.append(Spacer(1, 0.1 * inch))

                for img_idx, scene_img in enumerate(scene_images):
                    if scene_img.exists():
                        try:
                            # Adjust image size based on number of images
                            if len(scene_images) <= 2:
                                img_width, img_height = 5.5 * inch, 4 * inch
                            else:
                                img_width, img_height = 4.5 * inch, 3.2 * inch

                            img = Image(str(scene_img), width=img_width, height=img_height)
                            img.hAlign = 'CENTER'
                            elements.append(img)

                            # Image caption with perspective info
                            perspective_match = re.search(r'scene_(\d+)', scene_img.name)
                            perspective_info = f"Perspective {int(perspective_match.group(1)) + 1}" if perspective_match else f"View {img_idx + 1}"
                            elements.append(
                                Paragraph(f"{perspective_info}: {scene.setting}", self.styles['image_caption']))
                            elements.append(Spacer(1, 0.15 * inch))

                        except Exception as e:
                            self.logger.warning(f"Could not add scene image {scene_img.name}: {e}")
            else:
                # Show world image as fallback
                if all_images_organized['world']:
                    try:
                        world_img = all_images_organized['world'][scene_idx % len(all_images_organized['world'])]
                        img = Image(str(world_img), width=5 * inch, height=3.8 * inch)
                        img.hAlign = 'CENTER'
                        elements.append(img)
                        elements.append(Paragraph(f"World Setting: {scene.setting}", self.styles['image_caption']))
                        elements.append(Spacer(1, 0.2 * inch))
                    except Exception as e:
                        self.logger.warning(f"Could not add world fallback image: {e}")
                        elements.append(Spacer(1, 2 * inch))

            # Show relevant characters in this scene
            scene_characters = list(set([d.character for d in scene.dialogue if d.character != "Narrator"]))
            if scene_characters and all_images_organized['characters']:
                elements.append(Paragraph("Characters in this scene:", self.styles['character_info']))
                elements.append(Spacer(1, 0.1 * inch))

                for char_name in scene_characters[:2]:  # Max 2 characters per scene page
                    # Find matching character
                    for story_char_idx, story_char in enumerate(story.characters):
                        if story_char.name == char_name and story_char_idx < len(all_images_organized['characters']):
                            char_img = all_images_organized['characters'][story_char_idx]
                            if char_img.exists():
                                try:
                                    char_image = Image(str(char_img), width=2 * inch, height=2 * inch)
                                    char_image.hAlign = 'CENTER'
                                    elements.append(char_image)
                                    elements.append(Paragraph(f"{story_char.name}: {story_char.visual_desc[:50]}...",
                                                              self.styles['image_caption']))
                                    elements.append(Spacer(1, 0.1 * inch))
                                    break
                                except Exception as e:
                                    self.logger.warning(f"Could not add character image for {char_name}: {e}")

            # Dialogue at BOTTOM
            if scene.dialogue:
                elements.append(Spacer(1, 0.2 * inch))
                elements.append(Paragraph("Dialogue:", self.styles['character_info']))
                for dialogue in scene.dialogue:
                    elements.append(Paragraph(f"{dialogue.character}:", self.styles['character_name']))
                    elements.append(Paragraph(dialogue.text, self.styles['dialogue_text']))

            # Page break between scenes
            if scene_idx < len(story.scenes) - 1:
                elements.append(PageBreak())

        return elements

    def _create_end_page_with_image(self, end_image: Optional[Path]) -> List:
        """Create end page with 99_end.png"""
        elements = []

        elements.append(Spacer(1, 1 * inch))
        elements.append(Paragraph("The End", self.styles['title_main']))
        elements.append(Spacer(1, 0.5 * inch))

        # End image 99_end.png
        if end_image and end_image.exists():
            try:
                end_img = Image(str(end_image), width=5.5 * inch, height=4 * inch)
                end_img.hAlign = 'CENTER'
                elements.append(end_img)
                elements.append(Paragraph("Story Conclusion", self.styles['image_caption']))
                self.logger.info(f"Added end image to PDF: {end_image.name}")
            except Exception as e:
                self.logger.warning(f"Could not add end image: {e}")
        else:
            self.logger.warning("No end image found (99_end.png)")

        elements.append(Spacer(1, 0.5 * inch))
        elements.append(
            Paragraph("Thank you for experiencing this complete visual story!", self.styles['narrator_text']))

        return elements

    def _create_generation_data_section(self, story: StoryData, all_images_organized: Dict,
                                        cost_summary: Dict = None) -> List:
        """Create section with complete generation data and remaining images"""
        elements = []

        elements.append(Paragraph("Generation Data & Complete Image Gallery", self.styles['section_header']))
        elements.append(Spacer(1, 0.3 * inch))

        # Generation statistics
        elements.append(Paragraph("Generation Statistics", self.styles['scene_header']))
        elements.append(Paragraph(f"Story Title: {story.title}", self.styles['metadata_info']))
        elements.append(Paragraph(f"Genre: {story.genre}", self.styles['metadata_info']))
        elements.append(
            Paragraph(f"Visual Style: {story.style_preset.image_style.value.title()}", self.styles['metadata_info']))
        elements.append(
            Paragraph(f"Camera Style: {story.style_preset.camera_style.value.title()}", self.styles['metadata_info']))
        elements.append(Paragraph(f"Total Characters: {len(story.characters)}", self.styles['metadata_info']))
        elements.append(Paragraph(f"Total Scenes: {len(story.scenes)}", self.styles['metadata_info']))

        # Count all images
        total_images = 0
        if all_images_organized['cover']: total_images += 1
        if all_images_organized['end']: total_images += 1
        total_images += len(all_images_organized['world'])
        total_images += len(all_images_organized['characters'])
        total_images += len(all_images_organized['all_scene_images'])

        elements.append(Paragraph(f"Total Generated Images: {total_images}", self.styles['metadata_info']))

        if cost_summary:
            elements.append(Paragraph(f"Total Generation Cost: ${cost_summary.get('total_cost_usd', 0):.3f}",
                                      self.styles['metadata_info']))

        elements.append(Spacer(1, 0.4 * inch))

        # Complete character gallery
        if all_images_organized['characters']:
            elements.append(Paragraph("Complete Character Gallery", self.styles['scene_header']))
            for i, character in enumerate(story.characters):
                if i < len(all_images_organized['characters']):
                    char_img = all_images_organized['characters'][i]
                    if char_img.exists():
                        try:
                            elements.append(Paragraph(character.name, self.styles['character_info']))
                            img = Image(str(char_img), width=3 * inch, height=3 * inch)
                            img.hAlign = 'CENTER'
                            elements.append(img)
                            elements.append(
                                Paragraph(f"Role: {character.role.value.title()}", self.styles['metadata_info']))
                            elements.append(
                                Paragraph(f"Description: {character.visual_desc}", self.styles['metadata_info']))
                            elements.append(Paragraph(f"Voice: {character.voice.value.replace('_', ' ').title()}",
                                                      self.styles['metadata_info']))
                            elements.append(Spacer(1, 0.3 * inch))
                        except Exception as e:
                            self.logger.warning(f"Could not add character {character.name}: {e}")

            elements.append(PageBreak())

        # World environment gallery
        if all_images_organized['world']:
            elements.append(Paragraph("Complete World Gallery", self.styles['scene_header']))
            elements.append(Paragraph(f"World Description: {story.world_desc}", self.styles['character_info']))
            elements.append(Spacer(1, 0.2 * inch))

            for i, world_img in enumerate(all_images_organized['world']):
                if world_img.exists():
                    try:
                        img = Image(str(world_img), width=5 * inch, height=3.8 * inch)
                        img.hAlign = 'CENTER'
                        elements.append(img)
                        elements.append(Paragraph(f"World Environment View {i + 1}", self.styles['image_caption']))
                        elements.append(Spacer(1, 0.3 * inch))
                    except Exception as e:
                        self.logger.warning(f"Could not add world image {i}: {e}")

        # Footer with complete generation info
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(
            f"Complete multimedia story generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f" using Enhanced Story Generator v5.0 with {story.style_preset.image_style.value.title()} visual style "
            f"and {story.style_preset.camera_style.value.title()} camera work.",
            self.styles['metadata_info']
        ))

        return elements
