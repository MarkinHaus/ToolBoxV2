import json
from datetime import datetime, timedelta
from fastapi import Request
import os
import random
from threading import Thread

import networkx as nx
from lightrag import QueryParam
from dataclasses import asdict

from toolboxv2 import get_app
from toolboxv2.mods.FastApi.fast_nice import register_nicegui

import asyncio

from nicegui import ui

from pathlib import Path
import stripe

from toolboxv2.mods.TruthSeeker.arXivCrawler import Paper
# Set your secret key (use environment variables in production!)
stripe.api_key = os.getenv('STRIPE_SECRET_KEY_', 'sk_test_YourSecretKey')

def create_landing_page():
    # Set up dynamic background
    ui.query("body").style("background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)")

    # Main container with enhanced responsive design
    with ui.column().classes(
    "w-full max-w-md p-8 rounded-3xl shadow-2xl "
    "items-center self-center mx-auto my-8"
    ):
        # Advanced styling for glass-morphism effect
        ui.query(".nicegui-column").style(''''
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease-in-out;
        ''')

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
            ui.markdown('''
                TruthSeeker aims to democratize access to scientific knowledge,
                transforming complex academic research into comprehensible insights.
                We bridge the gap between raw data and meaningful understanding.
            ''').classes('text-lg').style('color: var(--text-color);')

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


def create_graph_tab(memory, processor_instance, summary_content, analysis_content, ui_main):
    # Load GraphML file
    if processor_instance is None or processor_instance[
        "instance"] is None:
        return
    graph_path = Path(memory.get_memory_base(processor_instance[
        "instance"].mem_name)) / "graph_chunk_entity_relation.graphml"

    if not graph_path.exists():
        ui.label("Error finding graph file")
        return
    G = nx.read_graphml(graph_path)

    # Precompute layouts
    pos_2d = nx.spring_layout(G, dim=2, seed=42)
    pos_3d = nx.spring_layout(G, dim=3, seed=42)

    # Prepare data for ECharts
    nodes = list(G.nodes())
    edges = list(G.edges())

    echart = None
    is_3d = None

    # 2D data: note that each node has a name and a value (the coordinates)
    nodes_2d = [{'name': node, 'value': list(pos_2d[node])} for node in nodes]
    # For links, we use the index positions of the node in the nodes list.
    edges_2d = [{'source': nodes.index(u), 'target': nodes.index(v)} for u, v in edges]

    # 3D data: add a random z value so that nodes have 3 coordinates
    nodes_3d = [{'name': node, 'value': list(pos_3d[node]) + [random.uniform(0, 1)]} for node in nodes]
    edges_3d = []
    for u, v in edges:
        # Each edge is given two endpoints; each endpoint gets its own random z value
        edges_3d.append([
            list(pos_3d[u]) + [random.uniform(0, 1)],
            list(pos_3d[v]) + [random.uniform(0, 1)]
        ])

    # Create UI elements in the Analysis tab
    with ui_main:
        # 3D toggle switch
        is_3d = ui.switch('3D Mode').classes('mb-4')
        # The chart container: note the height and width classes
        echart

    def update_graph():
        nonlocal echart, is_3d
        if is_3d.value:
            # 3D configuration (ECharts will automatically load 3D libraries if keys contain "3D")
            options = {
                'xAxis3D': {'type': 'value'},
                'yAxis3D': {'type': 'value'},
                'zAxis3D': {'type': 'value'},
                'grid3D': {'show': True},
                'series': [
                    {
                        'type': 'scatter3D',
                        'data': [n['value'] for n in nodes_3d],
                        'symbolSize': 12,
                        'itemStyle': {'color': '#4CAF50'}
                    },
                    {
                        'type': 'lines3D',
                        'data': edges_3d,
                        'lineStyle': {'color': '#607D8B', 'width': 1}
                    }
                ]
            }
        else:
            # 2D configuration
            options = {
                'xAxis': {'show': False},
                'yAxis': {'show': False},
                'series': [
                    {
                        'type': 'graph',
                        'layout': 'none',
                        'data': nodes_2d,
                        'links': edges_2d,
                        'roam': True,
                        'label': {'show': True, 'position': 'right'},
                        'edgeSymbol': ['circle', 'arrow'],
                        'edgeSymbolSize': [4, 10],
                        'itemStyle': {'color': '#4CAF50'},
                        'lineStyle': {'color': '#607D8B', 'width': 1}
                    }
                ]
            }
        echart = ui.echart(options).classes('w-full h-96')
        echart.update()

    # Initial render
    update_graph()

    # Update graph when switching modes
    is_3d.on('update:model-value', update_graph)

    # Define the click handler for points in the chart
    def handle_click(e):
        # e.series_type can be 'scatter3D' or 'graph'
        if e.series_type in ('scatter3D', 'graph'):
            # e.data_index holds the index of the clicked node in our nodes list.
            node_name = nodes[e.data_index]
            summary_content.set_content(f"**Selected Node:** {node_name}")
            num_neighbors = len(list(G.neighbors(node_name)))
            analysis_content.set_content(f"## Analysis for {node_name}\nNode connections: {num_neighbors}")

    # Register the click event for the chart
    echart.on_point_click(handle_click)

is_init = [False]
# --- Database Setup ---
def get_db():
    db = get_app().get_mod("DB")
    if not is_init[0]:
        is_init[0] = True
        db.edit_cli("LD")
        db.initialize_database()
    return db

# --- Session State Management ---
def get_user_state(session_id: str) -> dict:
    db = get_db()
    state_ = {
        'balance': 1.0,
        'last_reset': datetime.utcnow().isoformat(),
        'research_history': [],
        'payment_id': '',
    }
    if session_id is None:
        state_['balance'] *= -1
        return state_
    state = db.get(f"TruthSeeker::session:{session_id}")
    print("STAR>E:::", state, state.get(), "###")
    if state.get() is None:
        state = state_
    else:
        try:
            state = json.loads(state.get().decode('utf-8').replace("'", '"'))
        except Exception as e:
            print(e)
            state = state_
    return state


def save_user_state(session_id: str, state: dict):
    db = get_db()
    print("Saving state")
    db.set(f"TruthSeeker::session:{session_id}", json.dumps(state).encode('utf-8')).print()

def reset_daily_balance(state: dict, valid=False) -> dict:
    now = datetime.utcnow()
    last_reset = datetime.fromisoformat(state.get('last_reset', now.isoformat()))
    if now - last_reset > timedelta(hours=24):
        state['balance'] = max(state.get('balance', 1.2 if valid else 0.5), 1.2 if valid else 0.5)
        state['last_reset'] = now.isoformat()
    return state


def create_research_interface(Processor):

    def helpr(request, session: dict):
        session_id = session.get('ID')
        state = get_user_state(session_id)
        if session_rid := request.row.query_params.get('session_id'): # MACh schluer gege trikser uws
            if state.get('balance') < get_user_state(session_rid).get('balance') and get_user_state(session_rid).get('balance') != 1.0:
                pass
            else:
                state = get_user_state(session_rid)
        state = reset_daily_balance(state, session.get('valid'))
        save_user_state(session_id, state)
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
        followup_results_content = None
        progress_card = None
        balance = None

        # Global config storage with default values
        config = {
            'chunk_size': 1000,
            'overlap': 100,
            'num_search_result_per_query': 2,
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

            if estimated_time < 60:
                time_str = f"~{estimated_time:.2f}s"
            elif estimated_time < 3600:
                minutes = estimated_time // 60
                seconds = estimated_time % 60
                time_str = f"~{int(minutes)}m {int(seconds)}s"
            else:
                hours = estimated_time // 3600
                minutes = (estimated_time % 3600) // 60
                time_str = f"~{int(hours)}h {int(minutes)}m"
            with main_ui:
                query_length_label.set_text(f"Query len: {query_length} Total Papers: {config['max_search']*config['num_search_result_per_query']}")
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
                    summary_content, analysis_content,config, references_content, followup_card

                try:
                    if not validate_inputs():
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
                    # Direkt instanziieren: Eine neue ArXivPDFProcessor-Instanz
                    processor = Processor(query_text, tools=tools, **config)
                    # Setze den Callback so, dass Updates in der GUI angezeigt werden
                    processor.callback = update_status
                    processor_instance["instance"] = processor

                    # p_button[0].disabled = True
                    # Starte den Prozess (dieser gibt am Ende die verarbeiteten Papers und Insights zurück)
                    papers, insights = await processor.process()
                    # p_button[0].disabled = False
                    # Update Ergebnisse in der GUI
                    update_results({
                        "papers": papers,
                        "summary": insights.get("answer", "Keine Zusammenfassung erhalten."),
                        "insights": insights.get("sources", [])
                    })
                    with main_ui:
                        research_card.visible = True
                        config_cart.visible = True
                        show_history()

                except Exception as e:
                    # import traceback

                    with main_ui:
                        update_status({"progress": 0, "step": "Error", "info": str(e)})
                        ui.notify(f"Error {str(e)})", type="negative")

                    # print(traceback.format_exc())

            def target():
                get_app().run_a_from_sync(helper, )

            est_price = update_estimates()
            if est_price > state['balance']:
                ui.notify(f"Insufficient balance. Need €{est_price:.2f}", type='negative')
            else:
                state['balance'] -= est_price
                save_user_state(session_id, state)
                balance.set_text(f"Balance: {state['balance']:.2f}€")
                Thread(target=target, daemon=True).start()

        async def start_followup():
            nonlocal processor_instance, progress_card, response_type_input

            research_card.visible = False
            config_cart.visible = False
            config_section.visible = False

            # Sammle die Suchparameter aus den UI-Elementen
            try:
                qp = QueryParam(
                    mode=mode_select.value,
                    only_need_context=False,
                    only_need_prompt=False,
                    response_type=response_type_input.value,
                    stream=False,
                    top_k=60,
                    max_token_for_text_unit=4000
                )
            except Exception as e:
                ui.notify(f"Fehler bei den Suchparametern: {e}", type="warning")
                return
            with main_ui:
                combined_result = ""
                # Verarbeite jede eingegebene Folgefrage
                for input_comp in followup_inputs:
                    question = (input_comp.value or "").strip()
                    if not question:
                        continue  # Leere Eingaben überspringen
                    # Prüfe, ob eine aktive Research-Session vorhanden ist
                    if not processor_instance["instance"]:
                        ui.notify("Keine aktive Research-Session vorhanden.", type="warning")
                        return

                    progress_card.visible = True
                    # Rufe extra_query mit den Suchparametern auf
                    results = await processor_instance["instance"].extra_query(question, query_params=qp)
                    progress_card.visible = False
                    answer = results.get("answer", "Keine Antwort erhalten.")
                    combined_result += f"### Frage: {question}\n{answer}\n\n"

                if not combined_result:
                    ui.notify("Bitte mindestens eine gültige Folgefrage eingeben.", type="warning")
                    return

                research_card.visible = True
                config_cart.visible = True

                followup_results_content.set_content(combined_result)

        def show_history():
            with config_cart:
                for idx, entry in enumerate(state['research_history']):
                    with ui.card().classes("w-full backdrop-blur-lg bg-white/10 p-4").on('click',
                                                                                         lambda _, i=idx: load_history(
                                                                                             i)):
                        ui.label(entry['query']).classes('text-sm')
        # UI-Aufbau
        with ui.column().classes("w-full max-w-6xl mx-auto p-6 space-y-6") as main_ui:
            balance = ui.label(f"Balance: {state['balance']:.2f}€").classes("text-s font-semibold")

            config_cart = config_cart

            # --- Research Input UI Card ---
            with ui.card().classes("w-full backdrop-blur-lg bg-white/10 p-4") as research_card:
                ui.label("Research Interface").classes("text-3xl font-bold mb-4")

                # Query input section with auto-updating estimates
                query = ui.input("Research Query",
                                    placeholder="Gib hier deine Forschungsfrage ein...",
                                    value="") \
                    .classes("w-full min-h-[100px]") \
                    .on('change', lambda e: on_query_change()).style("color: var(--text-color)")

                # --- Action Buttons ---
                with ui.row().classes("mt-4"):
                    ui.button("Start Research", on_click=start_search) \
                        .classes("bg-blue-600 hover:bg-blue-700 py-3 rounded-lg")
                    ui.button("toggle config",
                              on_click=lambda: setattr(config_section, 'visible', not config_section.visible)).style(
                        "color: var(--text-color)")

            research_card = research_card

            # --- Options Cart / Configurations ---
            with ui.card_section().classes("w-full backdrop-blur-lg bg-white/10 hidden") as config_section:
                ui.separator()
                ui.label("Configuration Options").classes("text-xl font-semibold mt-4 mb-2")
                with ui.row():
                    chunk_size_input = ui.number(label="Chunk Size",
                                                 value=config['chunk_size'], format='%.0f', max=32_000, min=1000,
                                                 step=100) \
                        .on('change', on_config_change).style("color: var(--text-color)")
                    overlap_input = ui.number(label="Overlap",
                                              value=config['overlap'], format='%.0f', max=3200, min=100, step=50) \
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
            with ui.card().classes("w-full backdrop-blur-lg p-4 bg-white/10 hidden") as results_card:
                ui.label("Research Results").classes("text-xl font-semibold mb-4")
                with ui.tabs() as tabs:
                    ui.tab("Summary")
                    ui.tab("References")
                    ui.tab("Graph")
                    ui.tab("SystemStates")
                with ui.tab_panels(tabs, value="Summary").classes("w-full").style("background-color: var(--background-color)"):
                    with ui.tab_panel("Summary"):
                        summary_content = ui.markdown("").style("color : var(--text-color)")
                    with ui.tab_panel("References"):
                        references_content = ui.markdown("").style("color : var(--text-color)")
                    with ui.tab_panel("Graph") as graph_ui:
                        # ...and then add the graph (chart)
                        md_node = ui.markdown("").style("color : var(--text-color)")
                        analysis_node = ui.markdown("").style("color : var(--text-color)")
                        ui.button("Show Graph", on_click= lambda :create_graph_tab(get_tools().get_memory(), processor_instance, md_node, analysis_node, graph_ui))

                    with ui.tab_panel("SystemStates"):
                        analysis_content = ui.markdown("").style("color : var(--text-color)")

            # Ergebnisse sichtbar machen, sobald sie vorliegen.
            results_card = results_card

            # --- Follow-Up Bereich mit mehrfachen Folgefragen und Suchparametern ---
            with ui.card().classes("w-full backdrop-blur-lg bg-white/10 p-4 hidden") as followup_card:
                ui.label("Follow-Up Fragen & Suchparameter").classes("text-xl font-semibold mb-4")

                followup_inputs = []  # Liste zur Speicherung der Referenzen auf die Eingabefelder

                def add_followup_input():
                    # Erstelle ein neues Textarea für eine Folgefrage und füge es der Liste hinzu
                    input_comp = ui.input("Follow-Up Frage", placeholder="Gib deine Folgefrage ein...") \
                        .classes("w-full min-h-[60px] mb-2")
                    followup_inputs.append(input_comp)

                # Erstes Eingabefeld sofort anzeigen
                add_followup_input()

                # --- Suchparameter (QueryParam) ---
                ui.label("Suchparameter").classes("text-lg font-semibold mb-2")
                with ui.row():
                    mode_select = ui.select(label="Modus", value="global",
                                            options=["local", "global", "hybrid", "naive", "mix"]).style("color: var(--text-color)")
                    response_type_input = ui.input(label="Antwort-Typ", value="Multiple Paragraphs").style("color: var(--text-color)")

                # Ausgabe der Ergebnisse
                followup_results_content = ui.markdown("")

                # Absende-Button für Follow-Up Anfragen
                ui.button("Follow-Up absenden", on_click=lambda: asyncio.create_task(start_followup())) \
                    .classes("bg-green-600 hover:bg-green-700 py-3 rounded-lg")

            # Zugriff auf followup_card (falls später benötigt)
            followup_card = followup_card

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

        main_ui = main_ui
        # --- Hilfsfunktionen ---
        def validate_inputs() -> bool:
            if not query.value.strip():
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

        def show_progress_indicators():
            nonlocal progress_card
            progress_card.visible = True

        def update_results(data: dict, save=True):
            nonlocal summary_content, analysis_content, references_content, results_card, followup_card
            papers = data.get("papers", [])
            summary = data.get("summary", "")
            insights = data.get("insights", [])
            if save:
                history_entry = data.copy()
                history_entry['papers'] = [paper.model_dump_json() for paper in papers]
                if processor_instance is not None and processor_instance['instance'] is not None:
                    history_entry["mam_name"] = processor_instance['instance'].mem_name
                    history_entry["query"] = processor_instance['instance'].query
                state['research_history'].append(history_entry)
                save_user_state(session_id, state)
            else:
                papers = [Paper(**paper) for paper in papers]
            with main_ui:
                progress_card.visible = False
                # Zusammenfassung
                summary_content.set_content(f"# Research Summary\n\n{summary}")

                # Analyse: Hier werden die Inhalte der Quellen als Liste dargestellt.
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

                    # Set the markdown content
                    analysis_content.set_content(analysis_md)

                # Referenzen: Falls Paper-Objekte Attribute wie title oder relevance besitzen
                references_md = "# Referenzen\n" + "\n".join(
                    f"- ({i}) {getattr(paper, 'title', 'Unknown Title')} *{getattr(paper, 'pdf_url', 'Unknown URL')}*"
                    for i, paper in enumerate(papers)
                )+ "# Summarys\n"+"\n".join(
                    f"- ({i}) {getattr(paper, 'summary', 'Unknown summary')}"
                    for i, paper in enumerate(papers)
                )+"\n".join(
                    f"- ({i}) insight: {insight}"
                    for i, insight in enumerate(insights)
                )
                references_content.set_content(references_md)

                # Ergebnisse und Follow-Up Bereich einblenden
                results_card.visible = True
                followup_card.visible = True

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
            update_results(entry, save=False)

    return helpr

# --- Stripe Integration ---
def regiser_stripe_integration(is_scc=True):
    def stripe_callback(request: Request, session: str):

        state = get_user_state(request.row.query_params.get('session_id'))

        if state['payment_id'] is '':
            with ui.card().classes("w-full items-center").style("background-color: var(--background-color)"):
                ui.label(f"No payment id!").classes("text-lg font-bold")
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
                state = get_user_state(session)
                amount = session_data.amount_total / 100  # Convert cents to euros
                state['balance'] += amount
                state['payment_id'] = ''
                save_user_state(request.row.query_params.get('session_id'), state)

            # ui.navigate.to(f'/session?session={session}')
                ui.label(f"Transaction Complete - New Balace :{state['balance']}").classes("text-lg font-bold")
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
    base_url = f'{os.environ["HOSTNAME"]}/gui/open-Seeker.stripe' if not 'localhost' in os.environ["HOSTNAME"] else 'http://localhost:5000/gui/open-Seeker.stripe'
    session = stripe.checkout.Session.create(
        payment_method_types=['card', 'paypal',
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


