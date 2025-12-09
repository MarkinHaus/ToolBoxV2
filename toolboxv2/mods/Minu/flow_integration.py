"""
Minu Flow Integration V3
=========================
Automatische Generierung von UIs für Toolbox Flows mit stabilem Callback-System.

WICHTIGE ÄNDERUNGEN:
- Callbacks werden pro View-Instanz gespeichert, nicht global
- Callback-IDs sind stabil (basierend auf Funktionsname, nicht Counter)
- State-Updates werden korrekt propagiert
- Nur Flows mit Custom UI werden im Dashboard angezeigt
"""

import asyncio
import inspect
import hashlib
from typing import Any, Dict, Callable, Optional, get_type_hints, List

from toolboxv2 import get_app

from .core import (
    MinuView, State, Component, Card, Text, Button, Form,
    Spinner, Alert, Row, Column, Divider, Spacer, Heading,
    Grid, Badge, Custom, Checkbox, Input, Select, Textarea
)


# ============================================================================
# STABLE CALLBACK SYSTEM - Pro View, nicht global
# ============================================================================

class ViewCallbackRegistry:
    """
    Callback-Registry die an eine View-Instanz gebunden ist.
    Verwendet stabile IDs basierend auf Funktionsnamen.
    """

    def __init__(self, view_id: str):
        self.view_id = view_id
        self._callbacks: Dict[str, Callable] = {}
        self._name_to_id: Dict[str, str] = {}

    def register(self, callback: Callable, hint: str = "") -> str:
        """
        Registriert Callback mit stabiler ID.

        Args:
            callback: Die Callback-Funktion
            hint: Optionaler Hint für bessere ID-Generierung

        Returns:
            Stabile Handler-ID
        """
        # Generiere stabile ID basierend auf:
        # - View ID
        # - Funktionsname oder Hint
        # - Code-Location (für Lambdas)

        func_name = getattr(callback, '__name__', '')
        if func_name == '<lambda>' or not func_name:
            # Für Lambdas: verwende Hint oder Code-Hash
            if hint:
                key = f"{self.view_id}_{hint}"
            else:
                # Hash des Bytecodes für Stabilität
                code = getattr(callback, '__code__', None)
                if code:
                    code_id = f"{code.co_filename}:{code.co_firstlineno}"
                else:
                    code_id = str(id(callback))
                key = f"{self.view_id}_{hashlib.md5(code_id.encode()).hexdigest()[:8]}"
        else:
            key = f"{self.view_id}_{func_name}"

        # Wenn bereits registriert, wiederverwende ID
        if key in self._name_to_id:
            handler_id = self._name_to_id[key]
        else:
            handler_id = f"_cb_{hashlib.md5(key.encode()).hexdigest()[:12]}"
            self._name_to_id[key] = handler_id

        self._callbacks[handler_id] = callback
        return handler_id

    def get(self, handler_id: str) -> Optional[Callable]:
        return self._callbacks.get(handler_id)

    def get_all(self) -> Dict[str, Callable]:
        return self._callbacks.copy()

    def clear(self):
        self._callbacks.clear()
        self._name_to_id.clear()


# ============================================================================
# ENHANCED FLOW WRAPPER VIEW
# ============================================================================

class FlowWrapperView(MinuView):
    """
    Generischer View-Wrapper für Toolbox-Flows.

    Features:
    - Stabile Callback-IDs
    - Korrekte State-Propagation
    - Custom UI Support mit View-Referenz
    """

    # Reactive State
    inputs = State({})
    result = State(None)
    status = State("idle")  # idle, running, success, error
    error_msg = State("")

    def __init__(self, flow_name: str, run_func: Callable, custom_ui_func: Optional[Callable] = None):
        super().__init__(view_id=f"flow-{flow_name}")
        self.flow_name = flow_name
        self.run_func = run_func
        self.custom_ui_func = custom_ui_func

        # Pro-View Callback Registry
        self._callback_registry = ViewCallbackRegistry(self._view_id)

        # Schema für Auto-UI
        self.schema = self._generate_schema()

    def register_callback(self, callback: Callable, hint: str = "") -> str:
        """Registriert einen Callback und gibt die Handler-ID zurück."""
        handler_id = self._callback_registry.register(callback, hint)

        # Binde den Callback als Methode an diese View
        async def bound_handler(event, cb=callback):
            try:
                result = cb(event)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except Exception as e:
                self.error_msg.value = f"Error: {str(e)}"
                self.status.value = "error"
                raise

        setattr(self, handler_id, bound_handler)
        return handler_id

    def _generate_schema(self) -> Dict[str, Any]:
        """Analysiert Run-Funktion und erstellt Formular-Schema."""
        schema = {}
        try:
            sig = inspect.signature(self.run_func)
            type_hints = get_type_hints(self.run_func) if hasattr(self.run_func, '__annotations__') else {}

            for name, param in sig.parameters.items():
                if name in ('app', 'args_sto', 'kwargs', 'self'):
                    continue

                param_type = type_hints.get(name, str)
                default = param.default if param.default != inspect.Parameter.empty else ""

                field_config = {
                    "label": name.replace("_", " ").title(),
                    "default": default
                }

                if param_type == bool:
                    field_config["type"] = "checkbox"
                elif param_type == int:
                    field_config["type"] = "number"
                elif param_type == dict or param_type == list:
                    field_config["type"] = "textarea"
                    field_config["rows"] = 4
                else:
                    field_config["type"] = "text"
                    if any(kw in name.lower() for kw in ["prompt", "content", "text", "description", "body"]):
                        field_config["type"] = "textarea"
                        field_config["rows"] = 3

                schema[name] = field_config

        except Exception as e:
            print(f"[Minu] Error generating schema for {self.flow_name}: {e}")

        return schema

    async def run_flow(self, event: Dict[str, Any]):
        """Handler für Flow-Ausführung."""
        # Event kann formData enthalten oder direkt die Daten
        form_data = event.get("formData", event) if isinstance(event, dict) else {}

        self.status.value = "running"
        self.inputs.value = form_data
        self.result.value = None
        self.error_msg.value = ""

        app = get_app(from_="minu_flow_wrapper")

        try:
            res = await app.run_flows(self.flow_name, **form_data)

            if hasattr(res, 'is_error') and res.is_error():
                self.error_msg.value = res.info.info or "Unknown error"
                self.status.value = "error"
            else:
                # Daten extrahieren
                if hasattr(res, 'data'):
                    self.result.value = res.data
                elif hasattr(res, 'result') and hasattr(res.result, 'data'):
                    self.result.value = res.result.data
                else:
                    self.result.value = res

                self.status.value = "success"

        except Exception as e:
            self.error_msg.value = str(e)
            self.status.value = "error"

    async def reset(self, event):
        """Zurück zum Idle-State."""
        self.status.value = "idle"
        self.result.value = None
        self.error_msg.value = ""
        self.inputs.value = {}

    def __getattr__(self, name: str):
        """
        Fallback für dynamische Callback-Handler.
        Sucht in der lokalen Registry.
        """
        if name.startswith('_cb_'):
            callback = self._callback_registry.get(name)
            if callback:
                async def async_wrapper(event, cb=callback):
                    result = cb(event)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return result
                return async_wrapper

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def render(self) -> Component:
        """Rendert die UI."""
        # Header mit Reset-Button
        header = Row(
            Heading(self.flow_name.replace("_", " ").title(), level=2),
            Button("Reset", on_click="reset", variant="ghost")
                if self.status.value != "idle" else None,
            justify="between",
            className="mb-4"
        )

        # Custom UI verwenden wenn vorhanden
        if self.custom_ui_func:
            try:
                if isinstance(self.custom_ui_func, Callable):
                    # Custom UI bekommt self als Parameter für State-Zugriff
                    return self.custom_ui_func(self)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Minu] Error rendering custom UI for {self.flow_name}: {self.custom_ui_func}")
                return Column(
                    header,
                    Alert(f"Custom UI Error: {e}", variant="error"),
                    gap="4"
                )

        # Auto-generierte UI
        return self._render_auto_ui(header)

    def _render_auto_ui(self, header: Component) -> Component:
        """Rendert die automatisch generierte UI."""
        content = []

        if self.status.value == "running":
            content.append(Card(
                Column(
                    Spinner(size="lg"),
                    Text("Processing...", className="text-secondary"),
                    gap="4",
                    className="items-center py-8"
                )
            ))

        elif self.status.value == "error":
            content.append(Alert(self.error_msg.value, variant="error", title="Error"))
            content.append(Button("Try Again", on_click="reset", variant="secondary"))

        elif self.status.value == "success":
            content.append(Alert("Flow completed successfully!", variant="success"))
            content.append(self._render_result())
            content.append(Spacer())
            content.append(Button("Run Again", on_click="reset", variant="primary"))

        else:
            content.append(self._render_form())

        return Column(header, *content, gap="4")

    def _render_form(self) -> Component:
        """Rendert das Auto-Formular."""
        fields = []

        for name, config in self.schema.items():
            field_type = config.get("type", "text")
            label = config.get("label", name)
            default = config.get("default", "")
            value = self.inputs.value.get(name, default)

            if field_type == "checkbox":
                fields.append(Checkbox(
                    label=label,
                    checked=bool(value),
                    bind=name
                ))
            elif field_type == "textarea":
                fields.append(Column(
                    Text(label, className="text-sm font-medium"),
                    Textarea(
                        value=str(value) if value else "",
                        placeholder=f"Enter {label.lower()}...",
                        bind=name,
                        rows=config.get("rows", 3)
                    ),
                    gap="1"
                ))
            elif field_type == "number":
                fields.append(Input(
                    label=label,
                    value=str(value) if value else "",
                    input_type="number",
                    bind=name
                ))
            else:
                fields.append(Input(
                    label=label,
                    value=str(value) if value else "",
                    placeholder=f"Enter {label.lower()}...",
                    bind=name
                ))

        fields.append(Spacer())
        fields.append(Button(
            f"Run {self.flow_name.replace('_', ' ').title()}",
            on_click="run_flow",
            variant="primary",
            className="w-full"
        ))

        return Card(*fields, gap="3")

    def _render_result(self) -> Component:
        """Rendert das Ergebnis."""
        result = self.result.value

        if result is None:
            return Text("No result", className="text-secondary")

        if isinstance(result, dict):
            rows = []
            for key, value in result.items():
                rows.append(Row(
                    Text(key.replace("_", " ").title() + ":", className="font-medium"),
                    Text(str(value)[:200] + ("..." if len(str(value)) > 200 else "")),
                    justify="between",
                    className="py-2 border-b border-neutral-100"
                ))
            return Card(*rows, title="Result")

        if isinstance(result, (list, tuple)):
            items = [Text(f"• {item}") for item in result[:20]]
            if len(result) > 20:
                items.append(Text(f"... and {len(result) - 20} more", className="text-secondary"))
            return Card(*items, title=f"Result ({len(result)} items)")

        return Card(
            Text(str(result), className="whitespace-pre-wrap"),
            title="Result"
        )


# ============================================================================
# DASHBOARD & REGISTRATION - Nur Custom UI Flows
# ============================================================================

def scan_and_register_flows(app, only_custom_ui: bool = True) -> str:
    """
    Scannt Flows und registriert Views.

    Args:
        app: Toolbox App-Instanz
        only_custom_ui: Wenn True, nur Flows mit Custom UI anzeigen

    Returns:
        HTML-String des Dashboards
    """
    # Flows laden
    if not hasattr(app, "flows") or not app.flows:
        try:
            from toolboxv2.flows import flows_dict
            app.flows = flows_dict()
        except Exception as e:
            return f'<div class="alert alert-error">Could not load flows: {e}</div>'

    # Custom UIs laden
    try:
        from toolboxv2.flows import flows_dict
        custom_uis = flows_dict(ui=True)
    except:
        custom_uis = {}

    # Flows filtern und registrieren
    flow_cards = []

    flow_data = [

    ]

    for flow_name, run_func in app.flows.items():
        custom_ui = custom_uis.get(flow_name)

        # Wenn only_custom_ui, überspringe Flows ohne Custom UI
        if only_custom_ui and not custom_ui:
            continue

        try:
            # Dynamische View-Klasse erstellen
            def make_init(fn, rf, cu):
                def __init__(self):
                    FlowWrapperView.__init__(self, fn, rf, cu)
                return __init__

            DynamicView = type(
                f"FlowView_{flow_name}",
                (FlowWrapperView,),
                {
                    "__init__": make_init(flow_name, run_func, custom_ui.get("ui") if custom_ui else None),
                    "__doc__": run_func.__doc__ or f"Flow: {flow_name}",
                }
            )

            # Registrieren
            from toolboxv2.mods.Minu import register_view
            register_view(flow_name, DynamicView)

            # Card für Dashboard
            doc = (run_func.__doc__ or "No description").strip().split('\n')[0][:80]
            badge_variant = "success" if custom_ui else "secondary"
            badge_text = "Custom UI" if custom_ui else "Auto UI"

            card_html = f'''
            <div class="flow-card" onclick="window.location.href='/api/Minu/render?view={flow_name}&ssr=True'">
                <div class="flow-card-header">
                    <h4>{flow_name.replace("_", " ").title()}</h4>
                    <span class="badge badge-{badge_variant}">{badge_text}</span>
                </div>
                <p class="flow-card-desc">{doc}</p>
            </div>
            '''
            flow_cards.append(card_html)

        except Exception as e:
            print(f"[Minu] Error registering {flow_name}: {e}")

    # Dashboard HTML
    return f'''
    <div class="flow-dashboard">
        <div class="flow-header">
            <h1>Flow Apps</h1>
            <span class="badge badge-info">{len(flow_cards)} Available</span>
        </div>
        <div class="flow-grid">
            {"".join(flow_cards) if flow_cards else '<p class="text-secondary">No flows with Custom UI found.</p>'}
        </div>
    </div>
    <style>
        .flow-dashboard {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
        .flow-header {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-default); }}
        .flow-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1.5rem; }}
        .flow-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .flow-card:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-md); }}
        .flow-card-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem; }}
        .flow-card-header h4 {{ margin: 0; font-size: 1.1rem; }}
        .flow-card-desc {{ color: var(--text-secondary); font-size: 0.875rem; margin: 0; }}
        .badge {{ padding: 0.25rem 0.5rem; border-radius: var(--radius-sm); font-size: 0.75rem; font-weight: 500; }}
        .badge-success {{ background: var(--color-success); color: white; }}
        .badge-secondary {{ background: var(--bg-sunken); color: var(--text-secondary); }}
        .badge-info {{ background: var(--color-info); color: white; }}
    </style>
    '''


# ============================================================================
# UNIFIED DASHBOARD - Apps + Flows
# ============================================================================

def render_unified_dashboard(app, user_authenticated: bool = False) -> str:
    """
    Rendert ein einheitliches Dashboard mit Apps und Flows.

    Args:
        app: Toolbox App-Instanz
        user_authenticated: Ob der User eingeloggt ist

    Returns:
        Vollständiges HTML für das Dashboard
    """
    import json

    # 1. CloudM UIs laden
    try:
        ui_config = app.config_fh.get_file_handler("CloudM::UI", "{}")
        all_uis = json.loads(ui_config)
    except:
        all_uis = {}

    # 2. Flows mit Custom UI laden
    try:
        from toolboxv2.flows import flows_dict
        all_flows = flows_dict()
        custom_uis = flows_dict(ui=True)
    except:
        all_flows = {}
        custom_uis = {}

    # 3. Apps filtern basierend auf Auth
    app_cards = []
    for name, ui_info in all_uis.items():
        requires_auth = ui_info.get("auth", False)

        # Wenn Auth erforderlich aber User nicht eingeloggt, überspringen
        if requires_auth and not user_authenticated:
            continue

        title = ui_info.get("title", name)
        description = ui_info.get("description", "")[:100]
        path = ui_info.get("path", f"/app/{name}")
        icon = ui_info.get("icon", "apps")

        app_cards.append({
            "type": "app",
            "name": name,
            "title": title,
            "description": description,
            "path": path,
            "icon": icon,
            "auth": requires_auth
        })

    # 4. Flows mit Custom UI hinzufügen
    for flow_name, run_func in all_flows.items():
        if flow_name not in custom_uis:
            continue

        # Flow View registrieren
        custom_ui = custom_uis.get(flow_name)

        def make_init(fn, rf, cu):
            def __init__(self):
                FlowWrapperView.__init__(self, fn, rf, cu)
            return __init__

        DynamicView = type(
            f"FlowView_{flow_name}",
            (FlowWrapperView,),
            {"__init__": make_init(flow_name, run_func, custom_ui)}
        )

        from toolboxv2.mods.Minu import register_view
        register_view(flow_name, DynamicView)

        doc = (run_func.__doc__ or "").strip().split('\n')[0][:100]

        app_cards.append({
            "type": "flow",
            "name": flow_name,
            "title": flow_name.replace("_", " ").title(),
            "description": doc or "Interactive Flow Application",
            "path": f"/api/Minu/render?view={flow_name}&ssr=True",
            "icon": "account_tree",
            "auth": False
        })

    # 5. Nach Titel sortieren
    app_cards.sort(key=lambda x: x["title"].lower())

    # 6. HTML generieren
    cards_html = []
    for card in app_cards:
        badge = "Flow" if card["type"] == "flow" else ("🔒" if card["auth"] else "")
        cards_html.append(f'''
        <div class="app-card" data-search="{card['title'].lower()} {card['description'].lower()}" onclick="window.location.href='{card['path']}'">
            <div class="app-card-icon">
                <span class="material-symbols-outlined">{card['icon']}</span>
            </div>
            <div class="app-card-content">
                <div class="app-card-header">
                    <h3>{card['title']}</h3>
                    {f'<span class="app-badge">{badge}</span>' if badge else ''}
                </div>
                <p>{card['description']}</p>
            </div>
        </div>
        ''')

    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>App Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
        <style>
            :root {{
                --bg-base: #f8fafc;
                --bg-surface: #ffffff;
                --bg-sunken: #f1f5f9;
                --text-primary: #1e293b;
                --text-secondary: #64748b;
                --border-subtle: #e2e8f0;
                --interactive: #3b82f6;
                --radius-lg: 12px;
                --radius-md: 8px;
                --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
                --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1);
            }}

            * {{ box-sizing: border-box; margin: 0; padding: 0; }}

            body {{
                font-family: system-ui, -apple-system, sans-serif;
                background: var(--bg-base);
                color: var(--text-primary);
                min-height: 100vh;
            }}

            .dashboard {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 2rem;
            }}

            .dashboard-header {{
                text-align: center;
                margin-bottom: 2rem;
            }}

            .dashboard-header h1 {{
                font-size: 2rem;
                margin-bottom: 0.5rem;
            }}

            .dashboard-header p {{
                color: var(--text-secondary);
            }}

            .search-container {{
                max-width: 500px;
                margin: 0 auto 2rem;
            }}

            .search-input {{
                width: 100%;
                padding: 0.875rem 1rem 0.875rem 3rem;
                border: 1px solid var(--border-subtle);
                border-radius: var(--radius-lg);
                font-size: 1rem;
                background: var(--bg-surface);
                transition: all 0.2s;
            }}

            .search-input:focus {{
                outline: none;
                border-color: var(--interactive);
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
            }}

            .search-container {{ position: relative; }}
            .search-container .material-symbols-outlined {{
                position: absolute;
                left: 1rem;
                top: 50%;
                transform: translateY(-50%);
                color: var(--text-secondary);
            }}

            .app-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 1rem;
            }}

            .app-card {{
                display: flex;
                gap: 1rem;
                padding: 1.25rem;
                background: var(--bg-surface);
                border: 1px solid var(--border-subtle);
                border-radius: var(--radius-lg);
                cursor: pointer;
                transition: all 0.2s;
            }}

            .app-card:hover {{
                transform: translateY(-2px);
                box-shadow: var(--shadow-md);
                border-color: var(--interactive);
            }}

            .app-card.hidden {{
                display: none;
            }}

            .app-card-icon {{
                width: 48px;
                height: 48px;
                background: var(--bg-sunken);
                border-radius: var(--radius-md);
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            }}

            .app-card-icon .material-symbols-outlined {{
                font-size: 24px;
                color: var(--interactive);
            }}

            .app-card-content {{
                flex: 1;
                min-width: 0;
            }}

            .app-card-header {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 0.25rem;
            }}

            .app-card-header h3 {{
                font-size: 1rem;
                font-weight: 600;
            }}

            .app-badge {{
                font-size: 0.7rem;
                padding: 0.125rem 0.375rem;
                background: var(--bg-sunken);
                border-radius: 4px;
                color: var(--text-secondary);
            }}

            .app-card-content p {{
                font-size: 0.875rem;
                color: var(--text-secondary);
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}

            .no-results {{
                text-align: center;
                padding: 3rem;
                color: var(--text-secondary);
            }}

            @media (max-width: 640px) {{
                .dashboard {{ padding: 1rem; }}
                .app-grid {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="dashboard-header">
                <h1>Applications</h1>
                <p>{len(app_cards)} apps available</p>
            </div>

            <div class="search-container">
                <span class="material-symbols-outlined">search</span>
                <input type="text" class="search-input" id="search" placeholder="Search apps..." autocomplete="off">
            </div>

            <div class="app-grid" id="app-grid">
                {"".join(cards_html)}
            </div>

            <div class="no-results" id="no-results" style="display: none;">
                No apps match your search.
            </div>
        </div>

        <script>
            const searchInput = document.getElementById('search');
            const appGrid = document.getElementById('app-grid');
            const noResults = document.getElementById('no-results');
            const cards = document.querySelectorAll('.app-card');

            searchInput.addEventListener('input', (e) => {{
                const term = e.target.value.toLowerCase().trim();
                let hasVisible = false;

                cards.forEach(card => {{
                    const searchText = card.dataset.search || '';
                    const matches = !term || searchText.includes(term);
                    card.classList.toggle('hidden', !matches);
                    if (matches) hasVisible = true;
                }});

                noResults.style.display = hasVisible ? 'none' : 'block';
            }});
        </script>
    </body>
    </html>
    '''


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "FlowWrapperView",
    "ViewCallbackRegistry",
    "scan_and_register_flows",
    "render_unified_dashboard",
]
