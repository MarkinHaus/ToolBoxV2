'''import colorsys
import json
import time
from datetime import datetime, timedelta
from queue import Queue
from typing import Dict, Union, List, Any

import os
import random
from threading import Thread, Event

import networkx as nx
from dataclasses import asdict

from toolboxv2 import get_app
from toolboxv2.mods.FastApi.fast_nice import register_nicegui

import asyncio

from nicegui import ui

from pathlib import Path
import stripe

from toolboxv2.mods.TruthSeeker.arXivCrawler import Paper
from toolboxv2.mods.isaa.base.AgentUtils import anything_from_str_to_dict

# Set your secret key (use environment variables in production!)
stripe.api_key = os.getenv('STRIPE_SECRET_KEY', 'sk_test_YourSecretKey')

def create_landing_page():
    # Set up dynamic background
    ui.query("body").style("background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)")

    # Main container with enhanced responsive design
    with ui.column().classes(
    "w-full max-w-md p-8 rounded-3xl shadow-2xl "
    "items-center self-center mx-auto my-8"
    ):
        # Advanced styling for glass-morphism effect
        ui.query(".nicegui-column").style("""
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease-in-out;
        """)

        # Animated logo/brand icon
        with ui.element("div").classes("animate-fadeIn"):
            ui.icon("science").classes(
            "text-7xl mb-6 text-primary "
            "transform hover:scale-110 transition-transform"
            )

        # Enhanced typography for title
        ui.label("TruthSeeker").classes(
        "text-5xl font-black text-center "
        "text-primary mb-2 animate-slideDown"
        )

        # Stylized subtitle with brand message
        ui.label("Precision. Discovery. Insights.").classes(
        "text-xl font-medium text-center "
        "mb-10 animate-fadeIn"
        )

        # Button container for consistent spacing
        ui.button(
        "Start Research",
        on_click=lambda: ui.navigate.to("/open-Seeker.seek")
        ).classes(
        "w-full px-6 py-4 text-lg font-bold "
        "bg-primary hover:bg-primary-dark "
        "transform hover:-translate-y-0.5 "
        "transition-all duration-300 ease-in-out "
        "rounded-xl shadow-lg animate-slideUp"
        )

        # Navigation links container
        with ui.element("div").classes("mt-8 space-y-3 text-center"):
            ui.link(
            "Demo video",
            ).classes(
            "block text-lg text-gray-200 hover:text-primary "
            "transition-colors duration-300 animate-fadeIn"
            ).on("click", lambda: ui.navigate.to("/open-Seeker.demo"))

            ui.link(
            "About Us",
            ).classes(
            "block text-lg text-gray-400 hover:text-primary "
            "transition-colors duration-300 animate-fadeIn"
            ).on("click", lambda: ui.navigate.to("/open-Seeker.about"))

def create_video_demo():
    with ui.card().classes('w-full max-w-3xl mx-auto').style(
        'background: var(--background-color); color: var(--text-color)'):
        # Video container with responsive aspect ratio
        with ui.element('div').classes('relative w-full aspect-video'):
            video = ui.video('../api/TruthSeeker/video').classes('w-full h-full object-cover')

            # Custom controls overlay
            with ui.element('div').classes('absolute bottom-0 left-0 right-0 bg-black/50 p-2'):
                with ui.row().classes('items-center gap-2'):
                    #play_btn = ui.button(icon='play_arrow', on_click=lambda: video.props('playing=true'))
                    #pause_btn = ui.button(icon='pause', on_click=lambda: video.props('playing=false'))
                    ui.slider(min=0, max=100, value=0).classes('w-full').bind_value(video, 'time')
                    #mute_btn = ui.button(icon='volume_up', on_click=lambda: video.props('muted=!muted'))
                    #fullscreen_btn = ui.button(icon='fullscreen', on_click=lambda: video.props('fullscreen=true'))


        # Video description
        ui.markdown('Walkthrough of TruthSeeker features and capabilities.')
        # Back to Home Button
        ui.button('Back to Home', on_click=lambda: ui.navigate.to('/open-Seeker')).classes(
            'mt-6 w-full bg-primary text-white hover:opacity-90'
        )

    return video

def create_about_page():
    """Create a comprehensive About page for TruthSeeker"""
    with ui.column().classes('w-full max-w-4xl mx-auto p-6'):
        # Page Header
        ui.label('About TruthSeeker').classes('text-4xl font-bold text-primary mb-6')

        # Mission Statement
        with ui.card().classes('w-full mb-6').style(
            'background: var(--background-color); color: var(--text-color); padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'
        ):
            ui.label('Our Mission').classes('text-2xl font-semibold text-primary mb-4')
            ui.markdown("""
                TruthSeeker aims to democratize access to scientific knowledge,
                transforming complex academic research into comprehensible insights.
                We bridge the gap between raw data and meaningful understanding.
            """).classes('text-lg').style('color: var(--text-color);')

        # Core Technologies
        with ui.card().classes('w-full mb-6').style(
            'background: var(--background-color); color: var(--text-color); padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'
        ):
            ui.label('Core Technologies').classes('text-2xl font-semibold text-primary mb-4')
            with ui.row().classes('gap-4 w-full'):
                with ui.column().classes('flex-1 text-center'):
                    ui.icon('search').classes('text-4xl text-primary mb-2')
                    ui.label('Advanced Query Processing').classes('font-bold')
                    ui.markdown('Intelligent algorithms that extract nuanced research insights.').style(
                        'color: var(--text-color);')
                with ui.column().classes('flex-1 text-center'):
                    ui.icon('analytics').classes('text-4xl text-primary mb-2')
                    ui.label('Semantic Analysis').classes('font-bold')
                    ui.markdown('Deep learning models for comprehensive research verification.').style(
                        'color: var(--text-color);')
                with ui.column().classes('flex-1 text-center'):
                    ui.icon('verified').classes('text-4xl text-primary mb-2')
                    ui.label('Research Validation').classes('font-bold')
                    ui.markdown('Multi-layered verification of academic sources.').style('color: var(--text-color);')
        # Research Process
        with ui.card().classes('w-full').style('background: var(--background-color);color: var(--text-color);'):
            ui.label('Research Discovery Process').classes('text-2xl font-semibold text-primary mb-4')
            with ui.card().classes('q-pa-md q-mx-auto').style(
                'max-width: 800px; background: var(--background-color); border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'
            ) as card:
                ui.markdown("# Research Workflow").style(
                    "color: var(--primary-color); text-align: center; margin-bottom: 20px;")
                ui.markdown(
                    """
                    Welcome to TruthSeeker’s interactive research assistant. Follow the steps below to transform your initial inquiry into a refined, actionable insight.
                    """
                ).style("color: var(--text-color); text-align: center; margin-bottom: 30px;")

                # The stepper component
                with ui.stepper().style('background: var(--background-color); color: var(--text-color);') as stepper:
                    # Step 1: Query Initialization
                    with ui.step('Query Initialization'):
                        ui.markdown("### Step 1: Query Initialization").style("color: var(--primary-color);")
                        ui.markdown(
                            """
                            Begin by entering your research question or selecting from popular academic domains.
                            This sets the direction for our semantic analysis engine.
                            """
                        ).style("color: var(--text-color); margin-bottom: 20px;")
                        with ui.stepper_navigation():
                            ui.button('Next', on_click=stepper.next).props('rounded color=primary')

                    # Step 2: Semantic Search
                    with ui.step('Semantic Search'):
                        ui.markdown("### Step 2: Semantic Search").style("color: var(--primary-color);")
                        ui.markdown(
                            """
                            Our advanced algorithms now process your input to generate context-rich queries.
                            This stage refines the search context by understanding the deeper intent behind your question.
                            """
                        ).style("color: var(--text-color); margin-bottom: 20px;")
                        with ui.stepper_navigation():
                            ui.button('Back', on_click=stepper.previous).props('flat')
                            ui.button('Next', on_click=stepper.next).props('rounded color=primary')

                    # Step 3: Document Analysis
                    with ui.step('Document Analysis'):
                        ui.markdown("### Step 3: Document Analysis").style("color: var(--primary-color);")
                        ui.markdown(
                            """
                            The system then dives into a detailed analysis of academic papers, parsing content to extract key insights and connections.
                            This ensures that even subtle but crucial information is captured.
                            """
                        ).style("color: var(--text-color); margin-bottom: 20px;")
                        with ui.stepper_navigation():
                            ui.button('Back', on_click=stepper.previous).props('flat')
                            ui.button('Next', on_click=stepper.next).props('rounded color=primary')

                    # Step 4: Insight Generation
                    with ui.step('Insight Generation'):
                        ui.markdown("### Step 4: Insight Generation").style("color: var(--primary-color);")
                        ui.markdown(
                            """
                            Finally, we synthesize the analyzed data into clear, actionable research summaries.
                            These insights empower you with concise guidance to drive further inquiry or practical application.
                            """
                        ).style("color: var(--text-color); margin-bottom: 20px;")
                        with ui.stepper_navigation():
                            ui.button('Back', on_click=stepper.previous).props('flat')

        # Back to Home Button
        ui.button('Back to Home', on_click=lambda: ui.navigate.to('/open-Seeker')).classes(
            'mt-6 w-full bg-primary text-white hover:opacity-90'
        )
# Dummy-Implementierung für get_tools()
def get_tools():
    """
    Hier solltest du dein richtiges Werkzeug-Objekt zurückliefern.
    In diesem Beispiel gehen wir davon aus, dass du über eine Funktion wie get_app verfügst.
    """
    return get_app("ArXivPDFProcessor", name=None).get_mod("isaa")

def create_graph_tab(processor_instance: Dict, graph_ui: ui.element, main_ui: ui.element):
    """Create and update the graph visualization"""

    # Get HTML graph from processor
    _html_content = processor_instance["instance"].tools.get_memory(processor_instance["instance"].mem_name)
    html_content = "" if isinstance(_html_content, list) else _html_content.vis(get_output_html=True)

    # Ensure static directory exists
    static_dir = Path('dist/static')
    static_dir.mkdir(exist_ok=True)

    # Save HTML to static file
    graph_file = static_dir / f'graph{processor_instance["instance"].mem_name}.html'
    # Save HTML to static file with added fullscreen functionality

    # Add fullscreen JavaScript
    graph_file.write_text(html_content, encoding='utf-8')

    with main_ui:
        # Clear existing content except fullscreen button
        graph_ui.clear()

        with graph_ui:
            ui.html(f"""

                <iframe
                     src="/static/graph{processor_instance["instance"].mem_name}.html"
                    style="width: 100%; height: 800px; border: none; background: #1a1a1a;"
                    >
                </iframe>
            """).classes('w-full h-full')


is_init = [False]
# --- Database Setup ---
def get_db():
    db = get_app().get_mod("DB")
    if not is_init[0]:
        is_init[0] = True
        db.edit_cli("LD")
        db.initialize_database()
    return db

import pickle
# --- Session State Management ---
def get_user_state(session_id: str, is_new=False) -> dict:
    db = get_db()
    state_ = {
        'balance': .5,
        'last_reset': datetime.utcnow().isoformat(),
        'research_history': [],
        'payment_id': '',
    }
    if session_id is None:
        state_['balance'] *= -1
        if is_new:
            return state_, True
        return state_
    state = db.get(f"TruthSeeker::session:{session_id}")
    if state.get() is None:
        state = state_
        if is_new:
            return state_, True
    else:
        try:
            state = pickle.loads(state.get())
        except Exception as e:
            print(e)
            state = {
        'balance': 0.04,
        'last_reset': datetime.utcnow().isoformat(),
        'research_history': ["Sorry we had an error recreating your state"],
        'payment_id': '',
            }
            if is_new:
                return state, True
    if is_new:
        return state, False
    return state


def save_user_state(session_id: str, state: dict):
    db = get_db()
    print("Saving state")
    db.set(f"TruthSeeker::session:{session_id}", pickle.dumps(state)).print()

def delete_user_state(session_id: str):
    db = get_db()
    print("Saving state")
    db.delete(f"TruthSeeker::session:{session_id}").print()

def reset_daily_balance(state: dict, valid=False) -> dict:
    now = datetime.utcnow()
    last_reset = datetime.fromisoformat(state.get('last_reset', now.isoformat()))
    if now - last_reset > timedelta(hours=24):
        state['balance'] = max(state.get('balance', 1.6 if valid else 0.5), 1.6 if valid else 0.5)
        state['last_reset'] = now.isoformat()
    return state


class MemoryResultsDisplay:
    def __init__(self, results: List[Dict[str, Any]], main_ui: ui.element):
        self.results = results
        self.main_ui = main_ui
        self.setup_ui()

    def setup_ui(self):
        """Set up the main UI for displaying memory results"""
        with self.main_ui:
            self.main_ui.clear()
            with ui.column().classes('w-full'):
                for mem_result in self.results:
                    self.create_memory_card(mem_result)

    def create_memory_card(self, mem_result: Dict[str, Any]):
        """Create a card for each memory result"""
        result = mem_result.get("result", {})
        with self.main_ui:
            if isinstance(result, dict):
                self.display_dict_result(result)
            elif hasattr(result, 'overview'):  # Assuming RetrievalResult type
                self.display_retrieval_result(result)
            else:
                ui.label("Unsupported result type").classes('--text-color:error')

    def display_dict_result(self, result: Dict[str, Any]):
        """Display dictionary-based result with collapsible sections"""
        # Summary Section
        summary = result.get("summary", {})
        if isinstance(summary, str):
            try:
                summary = json.loads(summary[:-1])
            except json.JSONDecodeError:
                summary = {"error": "Could not parse summary"}

        # Raw Results Section
        raw_results = result.get("raw_results", {})
        if isinstance(raw_results, str):
            try:
                raw_results = json.loads(raw_results[:-1])
            except json.JSONDecodeError:
                raw_results = {"error": "Could not parse raw results"}

        # Metadata Section
        metadata = result.get("metadata", {})
        with self.main_ui:
            # Collapsible Sections
            with ui.column().classes('w-full space-y-2').style("max-width: 100%;"):
                # Summary Section
                with ui.expansion('Summary', icon='description').classes('w-full') as se:
                    self.display_nested_data(summary, main_ui=se)

                # Raw Results Section
                with ui.expansion('Raw Results', icon='work').classes('w-full') as re:
                    self.display_nested_data(raw_results, main_ui=re)

                # Metadata Section
                if metadata:
                    with ui.expansion('Metadata', icon='info').classes('w-full'):
                        ui.markdown(f"```json\n{json.dumps(metadata, indent=2)}\n```").style("max-width: 100%;")

    def display_retrieval_result(self, result):
        """Display retrieval result with detailed sections"""
        with self.main_ui:
            with ui.column().classes('w-full space-y-4').style("max-width: 100%;"):
                # Overview Section
                with ui.expansion('Overview', icon='visibility').classes('w-full') as ov:
                    for overview_item in result.overview:
                        if isinstance(overview_item, str):
                            overview_item = json.loads(overview_item)
                        self.display_nested_data(overview_item, main_ui=ov)

                # Details Section
                with ui.expansion('Details', icon='article').classes('w-full'):
                    for chunk in result.details:
                        with ui.card().classes('w-full p-3 mb-2').style("background: var(--background-color)"):
                            ui.label(chunk.text).classes('font-medium mb-2 --text-color:secondary')

                            with ui.row().classes('w-full justify-between').style("background: var(--background-color)"):
                                ui.label(f"Embedding Shape: {chunk.embedding.shape}").classes('text-sm')
                                ui.label(f"Content Hash: {chunk.content_hash}").classes('text-sm')

                            if chunk.cluster_id is not None:
                                ui.label(f"Cluster ID: {chunk.cluster_id}").classes('text-sm')

                # Cross References Section
                with ui.expansion('Cross References', icon='link').classes('w-full'):
                    for topic, chunks in result.cross_references.items():
                        with ui.card().classes('w-full p-3 mb-2').style("background: var(--background-color)"):
                            ui.label(topic).classes('font-semibold mb-2 --text-color:secondary')
                            for chunk in chunks:
                                ui.label(chunk.text).classes('text-sm mb-1')

    def display_nested_data(self, data: Union[Dict, List], indent: int = 0, main_ui=None):
        """Recursively display nested dictionary or list data"""
        with (self.main_ui if main_ui is None else main_ui):
            if isinstance(data, dict):
                with ui.column().classes(f'ml-{indent * 2}').style("max-width: 100%;"):
                    for key, value in data.items():
                        with ui.row().classes('items-center'):
                            ui.label(f"{key}:").classes('font-bold mr-2 --text-color:primary')
                            if isinstance(value, list):
                                if key == "main_chunks":
                                    continue
                                self.display_nested_data(value, indent + 1, main_ui=main_ui)
                            if isinstance(value, dict):
                                ui.markdown(f"```json\n{json.dumps(value, indent=2)}\n```").classes("break-words w-full").style("max-width: 100%;")
                            else:
                                ui.label(str(value)).classes('--text-color:secondary')
            elif isinstance(data, list):
                with ui.column().classes(f'ml-{indent * 2}').style("max-width: 100%;"):
                    for item in data:
                        if isinstance(item, str):
                            item = json.loads(item)
                        if isinstance(item, list):
                            self.display_nested_data(item, indent + 1, main_ui=main_ui)
                        if isinstance(item, dict):
                            ui.markdown(f"```json\n{json.dumps(item, indent=2)}\n```").classes("break-words w-full").style("max-width: 100%;")
                        else:
                            ui.label(str(item)).classes('--text-color:secondary')

def create_followup_section(processor_instance: Dict, main_ui: ui.element, session_id, balance):
    main_ui.clear()
    with main_ui:
        ui.label("Query Interface  (1ct)").classes("text-xl font-semibold mb-4")

        # Container for query inputs
        query_container = ui.column().classes("w-full gap-4")
        query = ""  # Store references to query inputs
        # Query parameters section
        with ui.expansion("Query Parameters", icon="settings").classes("w-full") as query_e:
            with ui.grid(columns=2).classes("w-full gap-4"):
                k_input = ui.number("Results Count (k)", value=2, min=1, max=20)
                min_sim = ui.number("Min Similarity", value=.3, min=0, max=1, step=0.1)
                cross_depth = ui.number("Cross Reference Depth", value=2, min=1, max=5)
                max_cross = ui.number("Max Cross References", value=10, min=1, max=20)
                max_sent = ui.number("Max Sentences", value=10, min=1, max=50)
                unified = ui.switch("Unified Retrieve (+3ct)", value=True)

        # Results display
        with ui.element("div").classes("w-full mt-4") as results_display:
            pass
        results_display = results_display
        with query_container:
            query_input = ui.input("Query", placeholder="Enter your query...") \
                .classes("w-full")
        # Control buttons
        with ui.row().classes("w-full gap-4 mt-4"):
            ui.button("Execute Query", on_click=lambda: asyncio.create_task(execute_query())) \
                .classes("bg-green-600 hover:bg-green-700")
            ui.button("Clear Results", on_click=lambda: results_display.clear()) \
                .classes("bg-red-600 hover:bg-red-700")
    query_input = query_input

    async def execute_query():
        """Execute a single query with parameters"""
        nonlocal query_input, results_display, main_ui
        try:
            query_text = query_input.value
            if not query_text.strip():
                with main_ui:
                    ui.notify("No Input", type="warning")
                return ""

            if not processor_instance.get("instance"):
                with main_ui:
                    ui.notify("No active processor instance", type="warning")
                return
            # Collect parameters
            params = {
                "k": int(k_input.value),
                "min_similarity": min_sim.value,
                "cross_ref_depth": int(cross_depth.value),
                "max_cross_refs": int(max_cross.value),
                "max_sentences": int(max_sent.value),
                "unified": unified.value
            }
            # Construct query parameters
            query_params = {
                "k": params["k"],
                "min_similarity": params["min_similarity"],
                "cross_ref_depth": params["cross_ref_depth"],
                "max_cross_refs": params["max_cross_refs"],
                "max_sentences": params["max_sentences"]
            }

            # Execute query
            results = await processor_instance["instance"].extra_query(
                query=query_text,
                query_params=query_params,
                unified_retrieve=params["unified"]
            )
            print("results",results)
            s = get_user_state(session_id)
            s['balance'] -= .04 if unified.value else .01
            save_user_state(session_id, s)
            with main_ui:
                balance.set_text(f"Balance: {s['balance']:.2f}€")
            # Format results
            with main_ui:
                with results_display:
                    MemoryResultsDisplay(results, results_display)

        except Exception as e:
            return f"Error executing query: {str(e)}\n\n"


    # Add initial query input

online_states = [0]
def create_research_interface(Processor):

    def helpr(request, session: dict):

        state = {'balance':0, 'research_history': []}
        main_ui = None
        with ui.column().classes("w-full max-w-6xl mx-auto p-6 space-y-6") as loading:
            ui.spinner(size='lg')
            ui.label('Initializing...').classes('ml-2')

        # Container for main content (initially hidden)
        content = ui.column().classes('hidden')

        # Extract session data before spawning thread
        session_id = session.get('ID')
        session_id_h = session.get('IDh')
        session_rid = request.row.query_params.get('session_id') if hasattr(request, 'row') else request.query_params.get('session_id')
        session_valid = session.get('valid')

        # Thread communication
        result_queue = Queue()
        ready_event = Event()

        def init_background():
            nonlocal session_id, session_id_h, session_rid, session_valid
            try:
                # Original initialization logic
                _state, is_new = get_user_state(session_id, is_new=True)

                if is_new and session_id_h != "#0":
                    _state = get_user_state(session_id_h)
                    save_user_state(session_id, _state)
                    delete_user_state(session_id_h)
                if session_rid:
                    state_: dict
                    state_, is_new_ = get_user_state(session_rid, is_new=True)
                    if not is_new_:
                        _state = state_.copy()
                        state_['payment_id'] = ''
                        state_['last_reset'] = datetime.utcnow().isoformat()
                        state_['research_history'] = state_['research_history'][:3]
                        state_['balance'] = 0
                        save_user_state(session_id, _state)
                _state = reset_daily_balance(_state, session_valid)
                save_user_state(session_id, _state)

                # Send result back to main thread
                result_queue.put(_state)
                ready_event.set()
            except Exception as e:
                result_queue.put(e)
                ready_event.set()

            # Start background initialization

        Thread(target=init_background).start()

        def check_ready():
            nonlocal state
            if ready_event.is_set():
                result = result_queue.get()

                # Check if initialization failed
                if isinstance(result, Exception):
                    loading.clear()
                    with loading:
                        ui.label(f"Error during initialization: {str(result)}").classes('text-red-500')
                    return

                # Get state and build main UI
                state = result
                loading.classes('hidden')
                content.classes(remove='hidden')
                main_ui.visible = True
                with main_ui:
                    balance.set_text(f"Balance: {state['balance']:.2f}€")
                    show_history()
                return  # Stop the timer

            # Check again in 100ms
            ui.timer(0.1, check_ready, once=True)

        # Start checking for completion
        check_ready()

        # Wir speichern die aktive Instanz, damit Follow-Up Fragen gestellt werden können
        processor_instance = {"instance": None}

        # UI-Elemente als Platzhalter; wir definieren sie später in der UI und machen sie so
        # in den Callback-Funktionen über "nonlocal" verfügbar.
        overall_progress = None
        status_label = None
        results_card = None
        summary_content = None
        analysis_content = None
        references_content = None
        followup_card = None
        research_card = None
        config_cart = None
        progress_card = None
        balance = None
        graph_ui = None

        sr_button = None
        r_button = None
        r_text = None


        # Global config storage with default values
        config = {
            'chunk_size': 21000,
            'overlap': 600,
            'num_search_result_per_query': 3,
            'max_search': 3,
            'num_workers': None
        }

        def update_estimates():
            """
            Dummy estimation based on query length and configuration.
            (Replace with your own non-linear formula if needed.)
            """
            query_text = query.value or ""
            query_length = len(query_text)
            # For example: estimated time scales with chunk size and query length.
            estimated_time ,estimated_price = Processor.estimate_processing_metrics(query_length, **config)
            estimated_time *= max(1, online_states[0] * 6)
            if processor_instance["instance"] is not None:
                estimated_price += .25
            if estimated_time < 60:
                time_str = f"~{int(estimated_time)}s"
            elif estimated_time < 3600:
                minutes = estimated_time // 60
                seconds = estimated_time % 60
                time_str = f"~{int(minutes)}m {int(seconds)}s"
            else:
                hours = estimated_time // 3600
                minutes = (estimated_time % 3600) // 60
                time_str = f"~{int(hours)}h {int(minutes)}m"
            with main_ui:
                query_length_label.set_text(f"Total Papers: {config['max_search']*config['num_search_result_per_query']}")
                time_label.set_text(f"Processing Time: {time_str}")
                price_label.set_text(f"Price: {estimated_price:.2f}€")

            return estimated_price

        def on_config_change(event):
            """
            Update the global config based on input changes and recalc estimates.
            """
            try:
                config['chunk_size'] = int(chunk_size_input.value)
            except ValueError:
                pass
            try:
                config['overlap'] = int(overlap_input.value)
                if config['overlap'] > config['chunk_size'] / 4:
                    config['overlap'] = int(config['chunk_size'] / 4)
                    with main_ui:
                        overlap_input.value = config['overlap']
            except ValueError:
                pass
            try:
                config['num_search_result_per_query'] = int(num_search_result_input.value)
            except ValueError:
                pass
            try:
                config['max_search'] = int(max_search_input.value)
            except ValueError:
                pass
            try:
                config['num_workers'] = int(num_workers_input.value) if num_workers_input.value != 0 else None
            except ValueError:
                config['num_workers'] = None

            update_estimates()

        def on_query_change():
            update_estimates()

        # Callback, der vom Processor (über processor_instance.callback) aufgerufen wird.
        def update_status(data: dict):
            nonlocal overall_progress, status_label
            if not data:
                return
            # Aktualisiere den Fortschrittsbalken und den aktuellen Schritt (wenn vorhanden)
            with main_ui:
                if isinstance(data, dict):
                    progress = data.get("progress", 0)
                    step = data.get("step", "Processing...")
                    overall_progress.value =round( progress ,2) # nicegui.linear_progress erwartet einen Wert zwischen 0 und 1
                    status_label.set_text(f"{step} {data.get('info','')}")
                else:
                    status_label.set_text(f"{data}")

        def start_search():
            nonlocal balance

            async def helper():
                nonlocal processor_instance, overall_progress, status_label, results_card, \
                    summary_content, analysis_content,config, references_content, followup_card,sr_button,r_button,r_text

                try:
                    if not validate_inputs():
                        with main_ui:
                            state['balance'] += est_price
                            save_user_state(session_id, state)
                            balance.set_text(f"Balance: {state['balance']:.2f}€")
                        return
                    reset_interface()
                    show_progress_indicators()

                    query_text = query.value.strip()
                    # Erzeuge das "tools"-Objekt (abhängig von deiner konkreten Implementation)
                    tools = get_tools()
                    with main_ui:
                        research_card.visible = False
                        config_cart.visible = False
                        config_section.visible = False
                        query.set_value("")
                    # Direkt instanziieren: Eine neue ArXivPDFProcessor-Instanz
                    if processor_instance["instance"] is not None:
                        processor = processor_instance["instance"]
                        processor.chunk_size = config['chunk_size']
                        processor.overlap = config['overlap']
                        processor.num_search_result_per_query = config['num_search_result_per_query']
                        processor.max_search = config['max_search']
                        processor.num_workers = config['num_workers']
                        papers, insights = await processor.process(query_text)
                    else:
                        processor = Processor(query_text, tools=tools, **config)
                    # Setze den Callback so, dass Updates in der GUI angezeigt werden
                        processor.callback = update_status
                        processor_instance["instance"] = processor
                        papers, insights = await processor.process()

                    update_results({
                        "papers": papers,
                        "insights": insights
                    })
                    with main_ui:
                        research_card.visible = True
                        config_cart.visible = True
                        show_history()

                except Exception as e:
                    import traceback

                    with main_ui:
                        update_status({"progress": 0, "step": "Error", "info": str(e)})
                        state['balance'] += est_price
                        save_user_state(session_id, state)
                        balance.set_text(f"Balance: {state['balance']:.2f}€")
                        ui.notify(f"Error {str(e)})", type="negative")
                        research_card.visible = True
                        config_cart.visible = True
                        config_section.visible = True
                    print(traceback.format_exc())

            def target():
                get_app().run_a_from_sync(helper, )

            est_price = update_estimates()
            if est_price > state['balance']:
                with main_ui:
                    ui.notify(f"Insufficient balance. Need €{est_price:.2f}", type='negative')
            else:
                state['balance'] -= est_price
                save_user_state(session_id, state)
                with main_ui:
                    online_states[0] += 1
                    balance.set_text(f"Balance: {state['balance']:.2f}€ Running Queries: {online_states[0]}")

                Thread(target=target, daemon=True).start()
                with main_ui:
                    online_states[0] -= 1
                    balance.set_text(f"Balance: {get_user_state(session_id)['balance']:.2f}€")


        def show_history():
            with config_cart:
                for idx, entry in enumerate(state['research_history']):
                    with ui.card().classes("w-full backdrop-blur-lg bg-white/10 p-4"):
                        ui.label(entry['query']).classes('text-sm')
                        ui.button("Open").on_click(lambda _, i=idx: load_history(i))

        def reset():
            nonlocal processor_instance, results_card, followup_card, sr_button, r_button, r_text
            processor_instance["instance"] = None
            show_progress_indicators()
            with main_ui:
                config_cart.visible = False
                config_section.visible = False
                followup_card.visible = False
                results_card.visible = False
                r_button.visible = False
                r_text.set_text("Research Interface")
                sr_button.set_text("Start Research")
            start_search()
        # UI-Aufbau

        with ui.column().classes("w-full max-w-6xl mx-auto p-6 space-y-6") as main_ui:
            balance = ui.label(f"Balance: {state['balance']:.2f}€").classes("text-s font-semibold")

            config_cart = config_cart

            # --- Research Input UI Card ---
            with ui.card().classes("w-full backdrop-blur-lg bg-white/10 p-4") as research_card:
                r_text = ui.label("Research Interface").classes("text-3xl font-bold mb-4")

                # Query input section with auto-updating estimates
                query = ui.input("Research Query",
                                    placeholder="Gib hier deine Forschungsfrage ein...",
                                    value="") \
                    .classes("w-full min-h-[100px]") \
                    .on('change', lambda e: on_query_change()).style("color: var(--text-color)")

                # --- Action Buttons ---
                with ui.row().classes("mt-4"):
                    sr_button =ui.button("Start Research", on_click=start_search) \
                        .classes("bg-blue-600 hover:bg-blue-700 py-3 rounded-lg")
                    ui.button("toggle config",
                              on_click=lambda: setattr(config_section, 'visible', not config_section.visible) or show_progress_indicators()).style(
                        "color: var(--text-color)")
                    r_button = ui.button("Start new Research",
                              on_click=reset).style(
                        "color: var(--text-color)")
            sr_button = sr_button
            r_button = r_button
            r_button.visible = False
            research_card = research_card

            # --- Options Cart / Configurations ---
            with ui.card_section().classes("w-full backdrop-blur-lg bg-white/10 hidden") as config_section:
                ui.separator()
                ui.label("Configuration Options").classes("text-xl font-semibold mt-4 mb-2")
                with ui.row():
                    chunk_size_input = ui.number(label="Chunk Size",
                                                 value=config['chunk_size'], format='%.0f', max=64_000, min=1000,
                                                 step=100) \
                        .on('change', on_config_change).style("color: var(--text-color)")
                    overlap_input = ui.number(label="Overlap",
                                              value=config['overlap'], format='%.0f', max=6400, min=100, step=50) \
                        .on('change', on_config_change).style("color: var(--text-color)")

                with ui.row():
                    num_search_result_input = ui.number(label="Results per Query",
                                                        value=config['num_search_result_per_query'], format='%.0f',
                                                        min=1, max=100, step=1) \
                        .on('change', on_config_change).style("color: var(--text-color)")
                    max_search_input = ui.number(label="Max Search Queries",
                                                 value=config['max_search'], format='%.0f', min=1, max=100, step=1) \
                        .on('change', on_config_change).style("color: var(--text-color)")
                    num_workers_input = ui.number(label="Number of Workers (leave empty for default)",
                                                  value=0, format='%.0f', min=0, max=32, step=1) \
                        .on('change', on_config_change).style("color: var(--text-color)")
            config_section = config_section
            config_section.visible = False
            # --- Ergebnisse anzeigen ---
            with ui.card().classes("w-full backdrop-blur-lg p-4 bg-white/10") as results_card:
                ui.label("Research Results").classes("text-xl font-semibold mb-4")
                with ui.tabs() as tabs:
                    ui.tab("Summary")
                    ui.tab("References")
                    ui.tab("SystemStates")
                with ui.tab_panels(tabs, value="Summary").classes("w-full").style("background-color: var(--background-color)"):
                    with ui.tab_panel("Summary"):
                        summary_content = ui.markdown("").style("color : var(--text-color)")
                    with ui.tab_panel("References"):
                        references_content = ui.markdown("").style("color : var(--text-color)")
                    with ui.tab_panel("SystemStates"):
                        analysis_content = ui.markdown("").style("color : var(--text-color)")


            # Ergebnisse sichtbar machen, sobald sie vorliegen.
            results_card = results_card
            results_card.visible = False

            # --- Follow-Up Bereich mit mehrfachen Folgefragen und Suchparametern ---
            with ui.card().classes("w-full backdrop-blur-lg bg-white/10 p-4 hidden") as followup_card:
                pass

            # Zugriff auf followup_card (falls später benötigt)
            followup_card = followup_card
            followup_card.visible = False

            # --- Fortschrittsanzeige ---
            with ui.card().classes("w-full backdrop-blur-lg bg-white/10 p-4") as progress_card:
                with ui.row():
                    ui.label("Research Progress").classes("text-xl font-semibold mb-4")
                    query_length_label = ui.label("").classes("mt-6 hover:text-primary transition-colors duration-300")
                    time_label = ui.label("Time: ...").classes("mt-6 hover:text-primary transition-colors duration-300")
                    price_label = ui.label("Price: ...").classes(
                        "mt-6 hover:text-primary transition-colors duration-300")

                overall_progress = ui.linear_progress(0).classes("w-full mb-4")
                status_label = ui.label("Warte auf Start...").classes("text-base")
            # Wir merken uns progress_card, falls wir ihn zurücksetzen wollen.
            progress_card = progress_card

            query_length_label = query_length_label
            time_label = time_label
            price_label = price_label

            with ui.card().classes("w-full backdrop-blur-lg bg-white/10 p-4") as config_cart:
                # --- Process Code Section ---
                # --- Estimated Time and Price ---
                # ui.label("History").classes("text-xl font-semibold mt-4 mb-2")
                ui.label('Research History').classes('text-xl p-4')
                show_history()

            ui.button('Add Credits', on_click=lambda: balance_overlay(session_id)).props('icon=paid')
            ui.label('About TruthSeeker').classes(
                'mt-6 text-gray-500 hover:text-primary '
                'transition-colors duration-300'
            ).on('click', lambda: ui.navigate.to('/open-Seeker.about', new_tab=True))

            with ui.element('div').classes("w-full").style("white:100%; height:100%") as graph_ui:
                pass

            with ui.card().classes("w-full p-4").style("background-color: var(--background-color)"):
                ui.label("Private Session link (restore the session on a different device)")
                base_url = f'https://{os.getenv("HOSTNAME")}/gui/open-Seeker.seek' if not 'localhost' in os.getenv("HOSTNAME") else 'http://localhost:5000/gui/open-Seeker.seek'
                ui.label(f"{base_url}?session_id={session_id}").style("white:100%")
                ui.label("Changes each time!")

            graph_ui = graph_ui
            graph_ui.visible = False
        main_ui = main_ui
        main_ui.visible = False

        # --- Hilfsfunktionen ---
        def validate_inputs() -> bool:
            if not query.value.strip():
                with main_ui:
                    ui.notify("Bitte gib eine Forschungsfrage ein.", type="warning")
                return False
            return True

        def reset_interface():
            nonlocal overall_progress, status_label, results_card, followup_card
            overall_progress.value = 0
            with main_ui:
                status_label.set_text("Research startet...")
            # Ergebnisse und Follow-Up Bereich verstecken
            results_card.visible = False
            followup_card.visible = False
            graph_ui.visible = False

        def show_progress_indicators():
            nonlocal progress_card
            progress_card.visible = True

        def update_results(data: dict, save=True):
            nonlocal summary_content, analysis_content, references_content, results_card,\
                followup_card,graph_ui, r_button, r_text, sr_button
            with main_ui:
                r_button.visible = True
                r_text.set_text("Add to current Results or press 'Start new Research'")
                sr_button.set_text("Add to current Results")
            # Handle papers (1-to-1 case)
            papers = data.get("papers", [])
            if not isinstance(papers, list):
                papers = [papers]

            # Get insights
            insights = data.get("insights", [])

            if save:
                history_entry = data.copy()
                history_entry['papers'] = [paper.model_dump_json() for paper in papers]
                if processor_instance is not None and processor_instance['instance'] is not None:
                    history_entry["mam_name"] = processor_instance['instance'].mem_name
                    history_entry["query"] = processor_instance['instance'].query

                    history_entry["processor_memory"] = processor_instance['instance'].tools.get_memory(

                    ).save_memory(history_entry["mam_name"], None)
                state['research_history'].append(history_entry)
                save_user_state(session_id, state)
            else:
                papers = [Paper(**json.loads(paper)) for paper in papers]
            create_followup_section(processor_instance, followup_card, session_id, balance)
            with main_ui:
                progress_card.visible = False
                # Build summary from insights
                summaries = []
                for insight in insights:
                    if 'result' in insight and 'summary' in insight['result']:
                        if isinstance(insight['result']['summary'], str):
                            # print(insight['result']['summary'], "NEXT", json.loads(insight['result']['summary'][:-1]),"NEXT22",  type(json.loads(insight['result']['summary'][:-1])))
                            insight['result']['summary'] = json.loads(insight['result']['summary'][:-1])
                        main_summary = insight['result']['summary'].get('main_summary', '')
                        if main_summary:
                            summaries.append(main_summary)
                summary_text = "\n\n".join(summaries) if summaries else "No summary available."
                summary_content.set_content(f"# Research Summary\n\n{summary_text}")

                # Analysis section (unchanged if processor details haven't changed)
                if processor_instance["instance"] is not None:
                    inst = processor_instance["instance"]
                    analysis_md = (
                        f"# Analysis\n"
                        f"- **query:** {inst.query}\n"
                        f"- **chunk_size:** {inst.chunk_size}\n"
                        f"- **overlap:** {inst.overlap}\n"
                        f"- **max_workers:** {inst.max_workers}\n"
                        f"- **num_search_result_per_query:** {inst.nsrpq}\n"
                        f"- **max_search:** {inst.max_search}\n"
                        f"- **download_dir:** {inst.download_dir}\n"
                        f"- **mem_name:** {inst.mem_name}\n"
                        f"- **current_session:** {inst.current_session}\n"
                        f"- **all_ref_papers:** {inst.all_ref_papers}\n"
                        f"- **all_texts_len:** {inst.all_texts_len}\n"
                        f"- **final_texts_len:** {inst.f_texts_len}\n"
                        f"- **num_workers:** {inst.num_workers}"
                    )
                    analysis_content.set_content(analysis_md)

                # References and Insights section
                references_md = "# References\n"
                # Add papers
                references_md += "\n".join(
                    f"- ({i}) [{getattr(paper, 'title', 'Unknown Title')}]({getattr(paper, 'pdf_url', 'Unknown URL')})"
                    for i, paper in enumerate(papers)
                )

                # Add detailed insights
                references_md += "\n\n# Insights\n"
                for i, insight in enumerate(insights):
                    print(insight)
                    result = insight.get('result', {})
                    summary = result.get('summary', {})

                    if isinstance(summary, str):
                        summary = json.loads(summary)

                    # Main summary
                    references_md += f"\n## Insight {i + 1}\n"
                    references_md += f"### Main Summary\n{summary.get('main_summary', 'No summary available.')}\n"

                    # Concept Analysis
                    concept_analysis = summary.get('concept_analysis', {})
                    if concept_analysis:
                        references_md += "\n### Concept Analysis\n"
                        references_md += "#### Key Concepts\n- " + "\n- ".join(
                            concept_analysis.get('key_concepts', [])) + "\n"
                        references_md += "\n#### Relationships\n- " + "\n- ".join(
                            concept_analysis.get('relationships', [])) + "\n"
                        references_md += "\n#### Importance Hierarchy\n- " + "\n- ".join(
                            concept_analysis.get('importance_hierarchy', [])) + "\n"

                    # Topic Insights
                    topic_insights = summary.get('topic_insights', {})
                    if topic_insights:
                        references_md += "\n### Topic Insights\n"
                        references_md += "#### Primary Topics\n- " + "\n- ".join(
                            topic_insights.get('primary_topics', [])) + "\n"
                        references_md += "\n#### Cross References\n- " + "\n- ".join(
                            topic_insights.get('cross_references', [])) + "\n"
                        references_md += "\n#### Knowledge Gaps\n- " + "\n- ".join(
                            topic_insights.get('knowledge_gaps', [])) + "\n"

                    # Relevance Assessment
                    relevance = summary.get('relevance_assessment', {})
                    if relevance:
                        references_md += "\n### Relevance Assessment\n"
                        references_md += f"- Query Alignment: {relevance.get('query_alignment', 'N/A')}\n"
                        references_md += f"- Confidence Score: {relevance.get('confidence_score', 'N/A')}\n"
                        references_md += f"- Coverage Analysis: {relevance.get('coverage_analysis', 'N/A')}\n"

                references_content.set_content(references_md)

                # nx concpts graph
                if processor_instance["instance"] is not None:
                    create_graph_tab(
                        processor_instance,
                        graph_ui,main_ui
                    )

                # Show results and followup cards
                results_card.visible = True
                followup_card.visible = True
                graph_ui.visible = True

        def load_history(index: int):
            entry = state['research_history'][index]
            if processor_instance is not None and processor_instance['instance'] is not None:

                processor_instance["instance"].mem_name = entry["mam_name"]
                processor_instance['instance'].query = entry["query"]

                pass
            else:
                processor = Processor(entry["query"], tools=get_tools(), **config)
                # Setze den Callback so, dass Updates in der GUI angezeigt werden
                processor.callback = update_status
                processor.mem_name = entry["mam_name"]
                processor_instance["instance"] = processor

            processor_instance["instance"].tools.get_memory().load_memory(entry["mam_name"], entry["processor_memory"])
            processor_instance["instance"].mem_name = entry["mam_name"]
            update_results(entry, save=False)

    return helpr

# --- Stripe Integration ---
def regiser_stripe_integration(is_scc=True):
    def stripe_callback(request: Request):

        sid = request.row.query_params.get('session_id') if hasattr(request, 'row') else request.query_params.get('session_id')
        state = get_user_state(sid)

        if state['payment_id'] == '':
            with ui.card().classes("w-full items-center").style("background-color: var(--background-color)"):
                ui.label(f"No payment id!").classes("text-lg font-bold")
                ui.button(
                    "Start Research",
                    on_click=lambda: ui.navigate.to("/open-Seeker.seek?session_id="+sid)
                ).classes(
                    "w-full px-6 py-4 text-lg font-bold "
                    "bg-primary hover:bg-primary-dark "
                    "transform hover:-translate-y-0.5 "
                    "transition-all duration-300 ease-in-out "
                    "rounded-xl shadow-lg animate-slideUp"
                )
            return

        try:
            session_data = stripe.checkout.Session.retrieve(state['payment_id'])
        except Exception as e:
            with ui.card().classes("w-full items-center").style("background-color: var(--background-color)"):
                ui.label(f"No Transactions Details !{e}").classes("text-lg font-bold")
                ui.button(
                    "Start Research",
                    on_click=lambda: ui.navigate.to("/open-Seeker.seek")
                ).classes(
                    "w-full px-6 py-4 text-lg font-bold "
                    "bg-primary hover:bg-primary-dark "
                    "transform hover:-translate-y-0.5 "
                    "transition-all duration-300 ease-in-out "
                    "rounded-xl shadow-lg animate-slideUp"
                )
                return
        with ui.card().classes("w-full items-center").style("background-color: var(--background-color)"):
            if is_scc and state['payment_id'] != '' and session_data.payment_status == 'paid':
                state = get_user_state(sid)
                amount = session_data.amount_total / 100  # Convert cents to euros
                state['balance'] += amount
                state['payment_id'] = ''
                save_user_state(sid, state)

            # ui.navigate.to(f'/session?session={session}')
                ui.label(f"Transaction Complete - New balance :{state['balance']}").classes("text-lg font-bold")
                with ui.card().classes("w-full p-4").style("background-color: var(--background-color)"):
                    ui.label("Private Session link (restore the session on a different device)")
                    base_url = f'https://{os.getenv("HOSTNAME")}/gui/open-Seeker.seek' if not 'localhost' in os.getenv("HOSTNAME")else 'http://localhost:5000/gui/open-Seeker.seek'
                    ui.label(f"{base_url}?session_id={sid}").style("white:100%")
                    ui.label("Changes each time!")
            else:
                ui.label(f"Transaction Error! {session_data}, {dir(session_data)}").classes("text-lg font-bold")
            ui.button(
                "Start Research",
                on_click=lambda: ui.navigate.to("/open-Seeker.seek")
            ).classes(
                "w-full px-6 py-4 text-lg font-bold "
                "bg-primary hover:bg-primary-dark "
                "transform hover:-translate-y-0.5 "
                "transition-all duration-300 ease-in-out "
                "rounded-xl shadow-lg animate-slideUp"
            )


    return stripe_callback


def handle_stripe_payment(amount: float, session_id):
    base_url = f'https://{os.getenv("HOSTNAME")}/gui/open-Seeker.stripe' if not 'localhost' in os.getenv("HOSTNAME") else 'http://localhost:5000/gui/open-Seeker.stripe'
    session = stripe.checkout.Session.create(
        payment_method_types=['card',
                              "link",
                              ],
        line_items=[{
            'price_data': {
                'currency': 'eur',
                'product_data': {'name': 'Research Credits'},
                'unit_amount': int(amount * 100),
            },
            'quantity': 1,
        }],
        automatic_tax={"enabled": True},
        mode='payment',
        success_url=f'{base_url}?session_id={session_id}',
        cancel_url=f'{base_url}.error'
    )
    state = get_user_state(session_id)
    state['payment_id'] = session.id
    save_user_state(session_id, state)
    ui.navigate.to(session.url, new_tab=True)

# --- UI Components ---
def balance_overlay(session_id):
    with ui.dialog().classes('w-full max-w-md bg-white/20 backdrop-blur-lg rounded-xl') as dialog:
        with ui.card().classes('w-full p-6 space-y-4').style("background-color: var(--background-color)"):
            ui.label('Add Research Credits').classes('text-2xl font-bold')
            amount = ui.number('Amount (€) min 2', value=5, format='%.2f', min=2, max=9999, step=1).classes('w-full')
            with ui.row().classes('w-full justify-between'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                ui.button('Purchase', on_click=lambda: handle_stripe_payment(amount.value, session_id))
    return dialog


def create_ui(processor):
    # ui_instance =
    register_nicegui("open-Seeker", create_landing_page
                     , additional="""<style>.nicegui-content {padding:0 !important} .ellipsis { color: var(--text-color) !important} #span {color: var(--text-color) !important} textarea:focus, input:focus {color:  var(--text-color) !important;}

            body {
        background: var(--background-color);
        color: var(--text-color);
        min-height: 100vh;
        font-family: "Inter", sans-serif;
        transition: background-color 0.3s, color 0.3s;
            }
            </style>""")
    register_nicegui("open-Seeker.stripe", regiser_stripe_integration(True)
                     , additional="""<style>.nicegui-content {padding:0 !important} .ellipsis { color: var(--text-color) !important} #span {color: var(--text-color) !important} textarea:focus, input:focus {color:  var(--text-color) !important;}

            body {
        background: var(--background-color);
        color: var(--text-color);
        min-height: 100vh;
        font-family: "Inter", sans-serif;
        transition: background-color 0.3s, color 0.3s;
            }
            </style>""", show=False)
    register_nicegui("open-Seeker.error", regiser_stripe_integration(False)
                     , additional="""<style>.nicegui-content {padding:0 !important} .ellipsis { color: var(--text-color) !important} #span {color: var(--text-color) !important} textarea:focus, input:focus {color:  var(--text-color) !important;}

            body {
        background: var(--background-color);
        color: var(--text-color);
        min-height: 100vh;
        font-family: "Inter", sans-serif;
        transition: background-color 0.3s, color 0.3s;
            }
            </style>""", show=False)
    register_nicegui("open-Seeker.about", create_about_page
                     , additional="""<style>.nicegui-content {padding:0 !important} .ellipsis { color: var(--text-color) !important} #span {color: var(--text-color) !important} textarea:focus, input:focus {color:  var(--text-color) !important;}

            body {
        background: var(--background-color);
        color: var(--text-color);
        min-height: 100vh;
        font-family: "Inter", sans-serif;
        transition: background-color 0.3s, color 0.3s;
            }
            </style>""", show=False)

    register_nicegui("open-Seeker.seek", create_research_interface(processor), additional="""
    <style>
    body {
        background: var(--background-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
        text-alignment: center
    }
#div {color:  var(--text-color) !important;}
#input {color:  var(--text-color) !important;}
.q-field__label {color:  var(--text-color) !important;}
.q-field__native {color:  var(--text-color) !important;}
    textarea:focus, input:focus, textarea {color:  var(--text-color) !important;}
    </style>
    """, show=False)
    register_nicegui("open-Seeker.demo", create_video_demo, additional="""
    <style>
    body {
        background: var(--background-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, show=False)
'''
