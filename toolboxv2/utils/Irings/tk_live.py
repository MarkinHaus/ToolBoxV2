import tkinter as tk
from typing import Dict, Set
import math
import colorsys
import threading
import queue
from datetime import datetime
import tkinter as tk
from typing import Dict, Set
import math
import colorsys
import threading
import queue
from datetime import datetime


class NetworkVisualizer:

    def __init__(self, network_manager):
        self.network = network_manager
        self.update_queue = queue.Queue()
        self.running = True
        self._ui_initialized = threading.Event()

        # Start UI thread
        self.ui_thread = threading.Thread(target=self._run_ui)
        self.ui_thread.daemon = True
        self.ui_thread.start()

        # Wait for UI to initialize
        self._ui_initialized.wait()

    def _run_ui(self):
        """Run UI in separate thread"""
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Network Overview")

        # Configure main window
        self.WIDTH = 800
        self.HEIGHT = 600

        # Create main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Create left panel for metrics and input
        self.left_panel = self._create_left_panel(main_container)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Create canvas
        self.canvas = tk.Canvas(
            main_container,
            width=self.WIDTH - 200,
            height=self.HEIGHT,
            bg='black'
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Visual elements tracking
        self.ring_positions: Dict[str, tuple] = {}
        self.concept_counts: Dict[str, int] = {}
        self.active_rings: Set[str] = set()
        self.animation_count = 0
        self.hover_text = None

        # Bind mouse events
        self.canvas.bind('<Motion>', self._handle_hover)
        self.canvas.bind('<Leave>', self._clear_hover)

        # Signal that UI is initialized
        self._ui_initialized.set()

        # Configure auto-update
        self._schedule_update()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        try:
            # Start mainloop
            self.root.mainloop()
        except:
            self.stop()

    def _create_left_panel(self, parent):
        """Create the left panel with metrics and input area"""
        frame = tk.Frame(parent, width=200)

        # Network stats
        stats_frame = tk.LabelFrame(frame, text="Network Metrics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.network_stats = tk.Label(stats_frame, text="", justify=tk.LEFT, anchor="w")
        self.network_stats.pack(fill=tk.X, padx=5, pady=5)

        # Connectivity stats
        tk.Label(stats_frame, text="Connectivity", font=('Arial', 10, 'bold')).pack(pady=(10, 0))
        self.connectivity_stats = tk.Label(stats_frame, text="", justify=tk.LEFT, anchor="w")
        self.connectivity_stats.pack(fill=tk.X, padx=5, pady=5)

        # Complexity metrics
        tk.Label(stats_frame, text="Complexity", font=('Arial', 10, 'bold')).pack(pady=(10, 0))
        self.complexity_stats = tk.Label(stats_frame, text="", justify=tk.LEFT, anchor="w")
        self.complexity_stats.pack(fill=tk.X, padx=5, pady=5)

        # Input area
        input_frame = tk.LabelFrame(frame, text="Process Text")
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.input_text = tk.Text(input_frame, height=4, width=25)
        self.input_text.pack(fill=tk.X, padx=5, pady=5)

        process_button = tk.Button(input_frame, text="Process", command=self._process_input)
        process_button.pack(pady=(0, 5))

        # Activity log
        log_frame = tk.LabelFrame(frame, text="Recent Activity")
        log_frame.pack(fill=tk.X, padx=5, pady=5)

        self.activity_log = tk.Text(log_frame, height=8, width=25, state='disabled')
        self.activity_log.pack(fill=tk.X, padx=5, pady=5)

        return frame

    def _process_input(self):
        """Process the input text through the network"""
        text = self.input_text.get("1.0", tk.END).strip()
        if text:
            # Process the text using the network manager
            activated_rings = self.network.process_input(text)

            # Clear input
            self.input_text.delete("1.0", tk.END)

            # Log activity
            self._log_activity(f"Processed: {text[:30]}...")
            for ring_id in activated_rings:
                self._log_activity(f"Activated: {ring_id}")
                self.mark_active(ring_id)

    def _log_activity(self, message: str):
        """Add a message to the activity log"""
        self.activity_log.config(state='normal')
        self.activity_log.insert("1.0", f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        # Keep only last 8 lines
        content = self.activity_log.get("1.0", tk.END).splitlines()
        if len(content) > 8:
            self.activity_log.delete("1.0", tk.END)
            self.activity_log.insert("1.0", "\n".join(content[:8]) + "\n")
        self.activity_log.config(state='disabled')

    def _update_metrics_panel(self):
        """Update the metrics panel with current network stats"""
        # Update network stats
        network_text = f"Total Rings: {len(self.network.rings)}\n"
        network_text += f"Active Rings: {len(self.active_rings)}\n"
        network_text += f"Total Concepts: {sum(len(ring.concept_graph.concepts) for ring in self.network.rings.values())}"
        self.network_stats.config(text=network_text)

        # Update connectivity stats
        avg_connections = sum(len(connections) for connections in self.network.connections.values()) / len(
            self.network.rings)
        connectivity_text = f"Avg Connections: {avg_connections:.1f}\n"
        connectivity_text += f"Max Connections: {self.network.max_connections}"
        self.connectivity_stats.config(text=connectivity_text)

        # Update complexity metrics
        total_concepts = sum(len(ring.concept_graph.concepts) for ring in self.network.rings.values())
        avg_concepts = total_concepts / len(self.network.rings) if self.network.rings else 0
        complexity_text = f"Avg Concepts/Ring: {avg_concepts:.1f}\n"
        complexity_text += f"Network Density: {len(self.network.connections) / (len(self.network.rings) * (len(self.network.rings) - 1) / 2):.2f}"
        self.complexity_stats.config(text=complexity_text)

    def _create_metrics_panel(self, parent):
        """Create the metrics panel"""
        frame = tk.LabelFrame(parent, text="Network Metrics", width=200)

        # Network stats
        self.network_stats = tk.Label(frame, text="", justify=tk.LEFT, anchor="w")
        self.network_stats.pack(fill=tk.X, padx=5, pady=5)

        # Connectivity stats
        tk.Label(frame, text="Connectivity", font=('Arial', 10, 'bold')).pack(pady=(10, 0))
        self.connectivity_stats = tk.Label(frame, text="", justify=tk.LEFT, anchor="w")
        self.connectivity_stats.pack(fill=tk.X, padx=5, pady=5)

        # Complexity metrics
        tk.Label(frame, text="Complexity", font=('Arial', 10, 'bold')).pack(pady=(10, 0))
        self.complexity_stats = tk.Label(frame, text="", justify=tk.LEFT, anchor="w")
        self.complexity_stats.pack(fill=tk.X, padx=5, pady=5)

        # Activity log
        tk.Label(frame, text="Recent Activity", font=('Arial', 10, 'bold')).pack(pady=(10, 0))
        self.activity_log = tk.Text(frame, height=8, width=25)
        self.activity_log.pack(fill=tk.X, padx=5, pady=5)

        return frame


    def _handle_hover(self, event):
        """Handle mouse hover events"""
        x, y = event.x, event.y

        # Clear previous hover text
        if self.hover_text:
            self.canvas.delete(self.hover_text)
            self.hover_text = None

        # Check if mouse is over any ring
        for ring_id, (ring_x, ring_y) in self.ring_positions.items():
            size = max(10, min(15, 10 + len(self.network.rings[ring_id].concept_graph.concepts) // 100))
            if abs(x - ring_x) <= size and abs(y - ring_y) <= size:
                # Create hover text with ring info
                ring = self.network.rings[ring_id]
                hover_text = f"{ring_id}\nConcepts: {len(ring.concept_graph.concepts)}"
                self.hover_text = self.canvas.create_text(
                    x + 10, y - 10,
                    text=hover_text,
                    fill='white',
                    anchor='w',
                    font=('Arial', 8)
                )
                break

    def _clear_hover(self, event):
        """Clear hover text when mouse leaves canvas"""
        if self.hover_text:
            self.canvas.delete(self.hover_text)
            self.hover_text = None

    def _draw_rings(self):
        """Draw rings with size based on concept count"""
        for ring_id, pos in self.ring_positions.items():
            x, y = pos

            # Calculate ring size based on number of concepts
            num_concepts = len(self.network.rings[ring_id].concept_graph.concepts)
            size = max(10, min(15, 10 + num_concepts // 100))

            # Generate color based on ring activity
            hue = hash(ring_id) % 360 / 360.0
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            color = f'#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}'

            # Draw ring
            ring_area = self.canvas.create_oval(
                x - size / 2, y - size / 2,
                x + size / 2, y + size / 2,
                fill=color if ring_id in self.active_rings else 'gray',
                outline='white',
                tags=(f"ring_{ring_id}",)
            )

            # Bind click event
            self.canvas.tag_bind(f"ring_{ring_id}", '<Button-1>',
                                 lambda e, rid=ring_id: self._open_inspector(rid))

    def _update(self):
        """Update the visualization"""
        if not self.running:
            return

        # Process any pending updates
        try:
            while True:
                ring_id = self.update_queue.get_nowait()
                self.active_rings.add(ring_id)
                self.update_queue.task_done()
        except queue.Empty:
            pass

        self.canvas.delete("all")  # Clear canvas

        # Calculate ring positions if needed
        if not self.ring_positions:
            self._calculate_ring_positions()

        # Draw network elements
        self._draw_connections()
        self._draw_rings()
        self._draw_activity()

        # Update metrics panel
        self._update_metrics_panel()

        # Schedule next update
        self._schedule_update()

    def stop(self):
        """Stop the visualizer"""
        self.running = False
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

    def _schedule_update(self):
        """Schedule the next update"""
        if self.running:
            self.root.after(100, self._update)

    def _calculate_ring_positions(self):
        """Calculate positions for rings in a circle layout"""
        num_rings = len(self.network.rings)
        if num_rings == 0:
            return

        center_x = self.WIDTH / 2
        center_y = self.HEIGHT / 2
        radius = min(self.WIDTH, self.HEIGHT) * 0.35

        for i, ring_id in enumerate(self.network.rings.keys()):
            angle = (2 * math.pi * i) / num_rings
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.ring_positions[ring_id] = (x, y)

    def _draw_connections(self):
        """Draw connections between rings"""
        for ring_id, connections in self.network.connections.items():
            if ring_id not in self.ring_positions:
                continue

            start_x, start_y = self.ring_positions[ring_id]

            for target_id, weight in connections.items():
                if target_id not in self.ring_positions:
                    continue

                end_x, end_y = self.ring_positions[target_id]

                # Calculate color based on connection weight
                intensity = min(255, int(weight * 255))
                color = f'#{intensity:02x}{intensity:02x}ff'

                # Draw connection line
                self.canvas.create_line(
                    start_x, start_y, end_x, end_y,
                    fill=color, width=1, dash=(2, 2)
                )

    def _draw_activity(self):
        """Draw recent activity indicators"""
        last_active = self.network.metrics.last_activated
        if last_active and last_active in self.ring_positions:
            x, y = self.ring_positions[last_active]
            size = 40 + (self.animation_count % 20)

            # Draw ripple effect
            self.canvas.create_oval(
                x - size / 2, y - size / 2,
                x + size / 2, y + size / 2,
                outline='white',
                width=1
            )

        self.animation_count += 1

    def _open_inspector(self, ring_id: str):
        """Open ring inspector window"""
        RingInspector(self.network.rings[ring_id])

    def mark_active(self, ring_id: str):
        """Mark a ring as active"""
        self.update_queue.put(ring_id)


class RingInspector:
    def __init__(self, ring: 'IntelligenceRing'):
        self.ring = ring
        self.window = tk.Toplevel()
        self.window.title(f"Ring Inspector: {ring.ring_id}")

        # Configure window
        self.window.geometry("400x500")

        # Create UI elements
        self._create_ui()

        # Start updates
        self._schedule_update()

    def _create_ui(self):
        """Create inspector UI elements"""
        # Stats frame
        stats_frame = tk.LabelFrame(self.window, text="Statistics", padx=5, pady=5)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.stats_label = tk.Label(stats_frame, text="", justify=tk.LEFT)
        self.stats_label.pack(anchor=tk.W)

        # Concepts list
        concepts_frame = tk.LabelFrame(self.window, text="Concepts", padx=5, pady=5)
        concepts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.concepts_list = tk.Listbox(concepts_frame)
        self.concepts_list.pack(fill=tk.BOTH, expand=True)

        # Bind selection event
        self.concepts_list.bind('<<ListboxSelect>>', self._show_concept_details)

        # Details frame
        self.details_frame = tk.LabelFrame(self.window, text="Concept Details", padx=5, pady=5)
        self.details_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.details_text = tk.Text(self.details_frame, height=8)
        self.details_text.pack(fill=tk.BOTH, expand=True)

    def _schedule_update(self):
        """Schedule the next update"""
        self.window.after(500, self._update)

    def _update(self):
        """Update the inspector view"""
        if not self.window.winfo_exists():
            return

        # Update statistics
        stats_text = f"Total Concepts: {len(self.ring.concept_graph.concepts)}\n"
        stats_text += f"Connected Rings: {len(self.ring.adapter.connected_rings)}\n"
        self.stats_label.config(text=stats_text)

        # Update concepts list
        current_selection = self.concepts_list.curselection()
        self.concepts_list.delete(0, tk.END)

        for concept_id, concept in self.ring.concept_graph.concepts.items():
            list_text = f"{concept.name[:30]}... (Stage: {concept.stage})"
            self.concepts_list.insert(tk.END, list_text)

        # Restore selection
        if current_selection:
            self.concepts_list.selection_set(current_selection)

        self._schedule_update()

    def _show_concept_details(self, event):
        """Show details for selected concept"""
        selection = self.concepts_list.curselection()
        if not selection:
            return

        index = selection[0]
        concept = list(self.ring.concept_graph.concepts.values())[index]

        details = f"ID: {concept.id}\n"
        details += f"Name: {concept.name}\n"
        details += f"Created: {concept.created_at}\n"
        details += f"Stage: {concept.stage}\n"
        details += f"TTL: {concept.ttl}\n"
        details += f"Similar Concepts: {len(concept.similar_concepts)}\n"
        details += f"Contradictions: {len(concept.contradictions)}\n"
        details += f"Metadata: {concept.metadata}\n"

        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, details)


def process_with_visualization(network, visualizer, text):
    """Process input text and update visualization"""
    results = network.process_input(text)
    for ring_id in results:
        visualizer.mark_active(ring_id)
    return results
