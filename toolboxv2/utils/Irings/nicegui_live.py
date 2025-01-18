import random

from nicegui import ui
from typing import Dict, Set, Callable, Any, Optional
from dataclasses import dataclass
import colorsys
import math
import networkx as nx
from datetime import datetime
import asyncio
import queue
from functools import partial


@dataclass
class ProcessResult:
    query: str
    result: str
    timestamp: datetime
    markdown_result: str


class NetworkVisualizer:
    def __init__(self, network_manager):
        self.network = network_manager
        self.update_queue = queue.Queue()
        self.running = True
        self.active_rings = set()
        self.callback: Optional[Callable[[ProcessResult], None]] = None

        # UI State
        self.is_processing = False
        self.selected_ring = None

        # Calculate centrality
        self.centrality_scores = self._calculate_centrality()

        # Create UI
        self.create_ui()

    def _calculate_centrality(self) -> Dict[str, float]:
        """Calculate centrality scores for rings"""
        G = nx.Graph()

        # Add nodes and edges
        for ring_id in self.network.rings:
            G.add_node(ring_id)

        for ring_id, connections in self.network.connections.items():
            for target in connections:
                G.add_edge(ring_id, target)

        # Calculate eigenvector centrality
        try:
            return nx.eigenvector_centrality(G)
        except:
            # Fallback to degree centrality if eigenvector fails
            return nx.degree_centrality(G)

    async def create_ui(self):
        # Add Tailwind CSS classes
        ui.add_head_html("""
            <style>
                .message-container { max-height: 40vh; overflow-y: auto; }
                .user-message { background: #e3f2fd; border-radius: 10px; margin: 5px; padding: 10px; }
                .system-message { background: #f5f5f5; border-radius: 10px; margin: 5px; padding: 10px; }
                .inspector-panel { height: 300px; overflow-y: auto; }
            </style>
        """)

        # Main container with flex layout
        with ui.element('div').classes('flex flex-col h-screen p-4 bg-gray-100'):
            # Top section: Messages and Input
            with ui.element('div').classes('flex-none h-1/3'):
                with ui.element('div').classes('message-container mb-4'):
                    self.messages_container = ui.element('div').classes('space-y-2')

                with ui.element('div').classes('flex items-center space-x-2'):
                    self.input_area = ui.textarea().props('rows=3').classes('flex-grow')
                    with ui.element('div').classes('flex flex-col space-y-2'):
                        self.loading_indicator = ui.spinner('dots').classes('hidden')
                        ui.button('Process', on_click=self._handle_input).classes('bg-blue-500 text-white')

            # Bottom section: Network and Inspector
            with ui.element('div').classes('flex flex-grow gap-4'):
                # Network visualization
                with ui.element('div').classes('flex-1'):
                    self._create_network_visualization()

            # Ring Inspector
            with ui.card().classes('flex-1 inspector-panel'):
                self.inspector_title = ui.label('Ring Inspector').classes('text-xl font-bold mb-4')

                with ui.row().classes('w-full'):
                    self.stats_container = ui.element('div').classes('space-y-2')

                ui.label('Concepts').classes('text-lg font-bold mt-4')
                self.concepts_container = ui.element('div').classes('space-y-2')

                ui.label('Concept Details').classes('text-lg font-bold mt-4')
                self.concept_details = ui.markdown('')

        # Initial update
        await self._update()

    def _calculate_ring_position(self, ring_id: str) -> tuple:
        """Calculate 3D position based on centrality"""
        centrality = self.centrality_scores.get(ring_id, 0)
        connections = len(self.network.connections.get(ring_id, []))

        # Spiral placement
        idx = list(self.network.rings.keys()).index(ring_id)
        angle = idx * math.pi * 0.618033988749895  # Golden ratio for even distribution

        # Height based on centrality
        height = centrality * 10

        # Radius based on connections
        radius = 5 + (connections * 0.5)

        return (
            radius * math.cos(angle),
            radius * math.sin(angle),
            height
        )


    def _init_network_objects(self):
        """Initialize network objects with 3D positioning"""
        # Calculate positions
        self.ring_positions = {
            ring_id: self._calculate_ring_position(ring_id)
            for ring_id in self.network.rings.keys()
        }

        # Create rings with event handling
        self.ring_objects = {}
        for ring_id in self.ring_positions:
            self.ring_objects[ring_id] = self._create_ring_object(ring_id)


    def _create_ring_object(self, ring_id: str):
        """Create a single interactive ring object"""
        pos = self.ring_positions[ring_id]

        # Create ring geometry
        ring = self.scene.sphere(
            radius=0.4,
        )
        ring.move(*pos)

        # Color based on centrality
        color = self._generate_color(self.centrality_scores.get(ring_id, 0))
        ring.material(color)

        # Add click handler
        ring.draggable(True)

        return ring

    def _handle_ring_click(self, ring_id: str):
        """Handle ring selection"""
        self.selected_ring = ring_id
        self._update_inspector()

    def _update_inspector(self):
        """Update ring inspector panel"""
        if not self.selected_ring or self.selected_ring not in self.network.rings:
            return

        ring = self.network.rings[self.selected_ring]

        # Update title
        self.inspector_title.text = f"Ring Inspector: {self.selected_ring}"

        # Update statistics
        with self.stats_container:
            ui.clear()
            ui.markdown(f"""
                **Statistics:**
                - Total Concepts: {len(ring.concept_graph.concepts)}
                - Connected Rings: {len(ring.adapter.connected_rings)}
                - Centrality Score: {self.centrality_scores.get(self.selected_ring, 0):.3f}
            """)

        # Update concepts list
        with self.concepts_container:
            ui.clear()
            for concept_id, concept in ring.concept_graph.concepts.items():
                with ui.card().classes('p-2 cursor-pointer hover:bg-gray-100'):
                    ui.label(f"{concept.name[:30]}... (Stage: {concept.stage})")
                    ui.button('Details', on_click=partial(self._show_concept_details, concept))

    def _show_concept_details(self, concept):
        """Show detailed information for a concept"""
        details = f"""
        ### Concept Details
        - **ID:** {concept.id}
        - **Name:** {concept.name}
        - **Created:** {concept.created_at}
        - **Stage:** {concept.stage}
        - **TTL:** {concept.ttl}
        - **Similar Concepts:** {len(concept.similar_concepts)}
        - **Contradictions:** {len(concept.contradictions)}

        **Metadata:**
        ```json
        {concept.metadata}
        ```
        """
        self.concept_details.content = details

    def _generate_color(self, centrality: float) -> str:
        """Generate color based on centrality score"""
        # Use centrality for both saturation and value
        hue = 0.6  # Blue
        sat = 0.5 + (centrality * 0.5)  # Higher centrality = more saturated
        val = 0.7 + (centrality * 0.3)  # Higher centrality = brighter

        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        return f'#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}'

    async def _update(self):
        """Update visualization state"""
        if not self.running:
            return

        try:
            while True:
                ring_id = self.update_queue.get_nowait()
                self.active_rings.add(ring_id)
                if ring_id in self.ring_objects:
                    self.ring_objects[ring_id].material('#00ff00')
                self.update_queue.task_done()
        except queue.Empty:
            pass

        if self.running:
            await asyncio.sleep(0.1)
            asyncio.create_task(self._update())

    def _create_metrics_section(self):
        with ui.card().classes('w-full mb-4'):
            ui.label('Network Metrics').classes('text-xl font-bold mb-2')
            self.network_stats = ui.label()

            ui.separator()

            ui.label('Connectivity').classes('text-lg font-bold mt-2')
            self.connectivity_stats = ui.label()

            ui.separator()

            ui.label('Complexity').classes('text-lg font-bold mt-2')
            self.complexity_stats = ui.label()

    def _create_input_section(self):
        with ui.card().classes('w-full mb-4'):
            ui.label('Process Text').classes('text-lg font-bold mb-2')
            self.input_area = ui.textarea().classes('w-full')
            ui.button('Process', on_click=self._process_input).classes('mt-2')

    def _create_activity_log(self):
        with ui.card().classes('w-full'):
            ui.label('Recent Activity').classes('text-lg font-bold mb-2')
            self.log_area = ui.textarea().classes('w-full').props('readonly')

    def _create_network_visualization(self):
        with ui.scene().classes('w-full h-[600px]') as self.scene:
            # Initialisiere 3D-Szene
            self.scene.background_color = '#1a1a1a'  # Dunklerer Hintergrund
            self.scene.grid = (20, 20)  # Feineres Gitter
            # Initialisiere Ringe als Kugeln

            self.ring_objects = {}
            # Initialize network objects
            self._init_network_objects()
            self._create_rings()

        for ring_id in self.ring_objects:
            ui.button(ring_id, on_click=partial(self._handle_ring_click, ring_id))

    def _calculate_ring_positions(self):
        """Berechne Positionen für Ringe in kreisförmiger Anordnung"""
        self.ring_positions = {}
        num_rings = len(self.network.rings)
        if num_rings == 0:
            return

        radius = 5  # Größerer Radius für bessere Sichtbarkeit
        for i, ring_id in enumerate(self.network.rings.keys()):
            angle = (2 * math.pi * i) / num_rings
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            self.ring_positions[ring_id] = (x, y, 0)

    def _create_rings(self):
        """Erstelle 3D-Ringe mit Verbindungen"""
        # Erstelle Ringe
        for ring_id, pos in self.ring_positions.items():
            # Erstelle Kugel für Ring
            sphere = self.scene.sphere(radius=0.5)
            sphere.move(*pos)

            # Färbe basierend auf Ring-ID
            hue = hash(ring_id) % 360 / 360.0
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            color = f'#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}'
            sphere.material(color)

            # Speichere Referenz
            self.ring_objects[ring_id] = sphere

        # Erstelle Verbindungen
        for ring_id, connections in self.network.connections.items():
            if ring_id not in self.ring_positions:
                continue

            start_pos = self.ring_positions[ring_id]
            for target_id in connections:
                if target_id not in self.ring_positions:
                    continue

                end_pos = self.ring_positions[target_id]
                self.scene.line(start_pos, end_pos).material('#404040')

    def _update_metrics(self):
        """Update Metrik-Anzeigen"""
        # Netzwerk-Statistiken
        network_text = f"Total Rings: {len(self.network.rings)}\n"
        network_text += f"Active Rings: {len(self.active_rings)}\n"
        network_text += f"Total Concepts: {sum(len(ring.concept_graph.concepts) for ring in self.network.rings.values())}"
        self.network_stats.text = network_text

        # Konnektivitäts-Statistiken
        avg_connections = sum(len(connections) for connections in self.network.connections.values()) / len(
            self.network.rings) if self.network.rings else 0
        connectivity_text = f"Average Connections: {avg_connections:.1f}\n"
        connectivity_text += f"Max Connections: {self.network.max_connections}"
        self.connectivity_stats.text = connectivity_text

        # Komplexitäts-Metriken
        total_concepts = sum(len(ring.concept_graph.concepts) for ring in self.network.rings.values())
        avg_concepts = total_concepts / len(self.network.rings) if self.network.rings else 0
        density = len(self.network.connections) / (len(self.network.rings) * (len(self.network.rings) - 1) / 2) if len(
            self.network.rings) > 1 else 0
        complexity_text = f"Average Concepts/Ring: {avg_concepts:.1f}\n"
        complexity_text += f"Network Density: {density:.2f}"
        self.complexity_stats.text = complexity_text

    def _process_input(self):
        """Verarbeite Eingabetext"""
        text = self.input_area.value
        if text:
            # Verarbeite Text durch Netzwerk-Manager
            activated_rings = self.network.process_input(text)

            # Lösche Eingabe
            self.input_area.value = ''

            # Logge Aktivität
            self._log_activity(f"Processed: {text[:30]}...")
            for ring_id in activated_rings:
                self._log_activity(f"Activated: {ring_id}")
                self.mark_active(ring_id)

    def _log_activity(self, message: str):
        """Füge Nachricht zum Aktivitätslog hinzu"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        current_log = self.log_area.value
        new_log = f"{timestamp} - {message}\n{current_log}"
        # Behalte nur die letzten 8 Zeilen
        log_lines = new_log.splitlines()[:8]
        self.log_area.value = '\n'.join(log_lines)

    def mark_active(self, ring_id: str):
        """Markiere einen Ring als aktiv"""
        self.update_queue.put(ring_id)

    def stop(self):
        """Stoppe die Visualisierung"""
        self.running = False

    async def _handle_input(self):
        """Handle input processing and visualization updates"""
        if self.is_processing:
            return

        text = self.input_area.value
        if not text:
            return

        try:
            self.is_processing = True
            self.loading_indicator.classes(remove='hidden')

            # Add user message
            await self._add_message(text, is_user=True)

            # Clear input
            self.input_area.value = ''

            # Process input and get results
            result = await self._process_input_async(text)

            # Update visualization for activated rings
            activated_rings = self.network.process_input(text)
            for ring_id in activated_rings:
                if ring_id in self.ring_objects:
                    # Highlight the ring
                    self.ring_objects[ring_id].material('#00ff00')

                    # Scale up the ring temporarily
                    original_scale = self.ring_objects[ring_id].scale
                    self.ring_objects[ring_id].scale = (1.5, 1.5, 1.5)

                    # Create particle effect for activation
                    self._create_activation_effect(ring_id)

                    # Schedule return to normal scale
                    asyncio.create_task(self._reset_ring_scale(ring_id, original_scale))

            # Update statistics and inspector if a ring is selected
            if self.selected_ring:
                self._update_inspector()

            # Generate and display network analysis
            analysis = self._generate_network_analysis(activated_rings)
            await self._add_message(analysis, is_user=False)

            # Call callback if registered
            if self.callback:
                self.callback(result)

        except Exception as e:
            error_msg = f"Error processing input: {str(e)}"
            await self._add_message(error_msg, is_user=False)

        finally:
            self.is_processing = False
            self.loading_indicator.classes('hidden')

    async def _reset_ring_scale(self, ring_id: str, original_scale: tuple, delay: float = 1.0):
        """Reset ring scale after delay"""
        await asyncio.sleep(delay)
        if ring_id in self.ring_objects:
            self.ring_objects[ring_id].scale = original_scale

    def _create_activation_effect(self, ring_id: str):
        """Create particle effect for ring activation"""
        if ring_id not in self.ring_positions:
            return

        pos = self.ring_positions[ring_id]

        # Create multiple particles
        for _ in range(5):
            particle = self.scene.sphere(radius=0.1)
            particle.material('#00ff00')
            particle.move(*pos)

            # Random direction for particle
            dx = (random.random() - 0.5) * 2
            dy = (random.random() - 0.5) * 2
            dz = (random.random() - 0.5) * 2

            # Animate particle
            asyncio.create_task(self._animate_particle(particle, dx, dy, dz))

    async def _animate_particle(self, particle, dx, dy, dz, duration: float = 1.0):
        """Animate a particle effect"""
        start_pos = particle.position
        steps = 20

        for i in range(steps):
            t = i / steps
            # Ease out cubic
            scale = 1 - math.pow(1 - t, 3)

            # Update position
            new_pos = (
                start_pos[0] + dx * scale,
                start_pos[1] + dy * scale,
                start_pos[2] + dz * scale
            )
            particle.move(*new_pos)

            # Fade out
            opacity = 1 - scale
            particle.opacity = opacity

            await asyncio.sleep(duration / steps)

        # Remove particle
        particle.delete()

    def _generate_network_analysis(self, activated_rings: Set[str]) -> str:
        """Generate detailed network analysis"""
        if not activated_rings:
            return "No rings were activated by this input."

        # Get statistics for activated rings
        activated_stats = []
        for ring_id in activated_rings:
            if ring_id in self.network.rings:
                ring = self.network.rings[ring_id]
                stats = {
                    'id': ring_id,
                    'concepts': len(ring.concept_graph.concepts),
                    'centrality': self.centrality_scores.get(ring_id, 0),
                    'connections': len(self.network.connections.get(ring_id, [])),
                }
                activated_stats.append(stats)

        # Sort by centrality
        activated_stats.sort(key=lambda x: x['centrality'], reverse=True)

        # Generate markdown report
        analysis = f"""### Network Activity Analysis
    - **Activated Rings:** {len(activated_rings)}
    - **Most Central Ring:** {activated_stats[0]['id']} (centrality: {activated_stats[0]['centrality']:.3f})

    #### Ring Details:
    """

        for stats in activated_stats:
            analysis += f"""
    - **{stats['id']}**
      - Centrality: {stats['centrality']:.3f}
      - Concepts: {stats['concepts']}
      - Connections: {stats['connections']}
    """

        return analysis

    async def _add_message(self, text: str, is_user: bool):
        """Add message to UI with appropriate styling"""
        classes = 'user-message ml-auto' if is_user else 'system-message'
        with self.messages_container:
            if isinstance(text, str) and text.startswith('###'):
                # Handle markdown content
                ui.markdown(text).classes(classes)
            else:
                # Handle plain text
                ui.label(text).classes(classes)


# Hauptanwendung starten
def run_visualizer(network_manager):
    visualizer = NetworkVisualizer(network_manager)
    return visualizer.create_ui


def process_with_visualization(network, visualizer, text):
    results = network.process_input(text)
    for ring_id in results:
        visualizer.mark_active(ring_id)
    return results
