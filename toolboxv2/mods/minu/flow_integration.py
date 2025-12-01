"""
Minu Flow Integration V2
=========================
Automatische Generierung von UIs für Toolbox Flows mit Python-Callback-Support.
"""

import asyncio
import inspect
from typing import Any, Dict, Callable, Optional, get_type_hints

from toolboxv2 import get_app

# Importiere benötigte Komponenten aus core
from .core import (
    MinuView, State, Component, Card, Text, Button, Form,
    Spinner, Alert, Row, Column, Divider, Spacer, Heading,
    Grid, Badge, Custom
)
from .flows import (
    form_for, ui_for_data, get_all_callbacks
)


class FlowWrapperView(MinuView):
    """
    Generischer View-Wrapper für Toolbox-Flows mit Callback-Injection.
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

        # Schema für Formular-Generierung
        self.schema = self._generate_schema()

        # WICHTIG: Callbacks aus flows.py injizieren
        self._inject_flow_callbacks()

    def _inject_flow_callbacks(self):
        """
        Injiziert alle registrierten Callbacks aus flows.py als Handler-Methoden.
        """
        callbacks = get_all_callbacks()

        for handler_name, callback_func in callbacks.items():
            async def handler_wrapper(event, func=callback_func):
                try:
                    result = func(event)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return result
                except Exception as e:
                    self.error_msg.value = f"Callback error: {str(e)}"
                    self.status.value = "error"
                    raise

            setattr(self, handler_name, handler_wrapper)

    def _generate_schema(self) -> Dict[str, Any]:
        """Analysiert die Run-Funktion und erstellt ein Formular-Schema"""
        schema = {}
        try:
            sig = inspect.signature(self.run_func)
            type_hints = get_type_hints(self.run_func)

            for name, param in sig.parameters.items():
                if name in ('app', 'args_sto', 'kwargs'):
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
                elif param_type == dict:
                    field_config["type"] = "textarea"
                else:
                    field_config["type"] = "text"
                    if "prompt" in name or "content" in name or "text" in name:
                        field_config["type"] = "textarea"
                        field_config["rows"] = 3

                schema[name] = field_config
        except Exception as e:
            print(f"Error generating schema for {self.flow_name}: {e}")

        return schema

    async def run_flow(self, form_data: Dict[str, Any]):
        """Handler für Formular-Submit"""
        self.status.value = "running"
        self.inputs.value = form_data
        self.result.value = None
        self.error_msg.value = ""

        app = get_app(from_="minu_flow_wrapper")

        try:
            kwargs = {**form_data}
            res = await app.run_flows(self.flow_name, **kwargs)

            if hasattr(res, 'is_error') and res.is_error():
                self.error_msg.value = res.info.info or "Unknown error"
                self.status.value = "error"
            else:
                if hasattr(res, 'data'):
                    self.result.value = res.data
                elif hasattr(res, 'result'):
                    self.result.value = res.result.data
                else:
                    self.result.value = res

                self.status.value = "success"

        except Exception as e:
            self.error_msg.value = str(e)
            self.status.value = "error"

    def reset(self, _):
        """Zurück zum Start"""
        self.status.value = "idle"
        self.result.value = None
        self.error_msg.value = ""

    def render(self) -> Component:
        """Rendert die UI."""
        header = Row(
            Heading(self.flow_name.replace("_", " ").title(), level=2),
            Button("Reset", on_click="reset", variant="ghost", size="sm")
                if self.status.value != "idle" else None,
            justify="between",
            className="mb-4"
        )

        # 1. Custom UI Logic
        if self.custom_ui_func:
            try:
                return self.custom_ui_func(self)
            except Exception as e:
                return Column(
                    header,
                    Alert(f"Error in custom UI: {e}", variant="error")
                )

        # 2. Auto UI Logic
        content = []

        if self.status.value == "running":
            content.append(
                Card(
                    Column(
                        Spinner(size="lg"),
                        Text("Processing Flow...", className="text-secondary"),
                        align="center",
                        gap="4"
                    ),
                    className="py-12"
                )
            )
        elif self.status.value == "error":
            content.append(Alert(self.error_msg.value, variant="error", title="Flow Failed"))
            content.append(Button("Try Again", on_click="reset", variant="secondary"))
        elif self.status.value == "success":
            result_comp = ui_for_data(self.result.value, title="Result")
            content.append(result_comp)
            content.append(Spacer())
            content.append(Button("New Run", on_click="reset", variant="primary"))
        else:
            form = form_for(
                self.schema,
                values=self.inputs.value,
                on_submit="run_flow",
                submit_label=f"Run {self.flow_name}",
            )
            content.append(form)

        return Column(header, *content)



def render_to_html(component: Component) -> str:
    """Convert a Minu component to static HTML string."""
    def _render(comp: Component) -> str:
        if not comp:
            return ""

        tag_map = {
            "card": "div",
            "row": "div",
            "column": "div",
            "grid": "div",
            "text": "span",
            "heading": f"h{comp.props.get('level', 1)}",
            "button": "button",
            "input": "input",
            "divider": "hr",
            "spacer": "div",
            "form": "form",
        }

        tag = tag_map.get(comp.type.value, "div")
        attrs = []
        if comp.className:
            attrs.append(f'class="{comp.className}"')
        if comp.id:
            attrs.append(f'id="{comp.id}"')

        if comp.style:
            style_str = "; ".join(f"{k}: {v}" for k, v in comp.style.to_dict().items())
            attrs.append(f'style="{style_str}"')

        attr_str = " " + " ".join(attrs) if attrs else ""

        if tag in ["input", "hr"]:
            return f"<{tag}{attr_str} />"

        content = ""
        if comp.props.get("text"):
            content = comp.props["text"]
        elif comp.props.get("label"):
            content = comp.props["label"]

        if comp.children:
            content = "".join(_render(child) for child in comp.children)

        return f"<{tag}{attr_str}>{content}</{tag}>"

    return _render(component)


def scan_and_register_flows(app, specific_flow: str = None) -> str:
    """
    Scannt geladene Flows, registriert MinuViews und gibt ein schönes Dashboard zurück.
    """
    # 1. Flows laden
    if not hasattr(app, "flows") or not app.flows:
        try:
            from toolboxv2.flows import flows_dict
            app.flows = flows_dict()
        except Exception as e:
            return render_to_html(Alert(f"Could not load flows: {e}", variant="error"))

    try:
        from toolboxv2.flows import flows_dict
        uis = flows_dict(ui=True)
    except:
        uis = {}

    flows_to_process = app.flows.items()
    if specific_flow:
        flows_to_process = [(k, v) for k, v in app.flows.items() if k == specific_flow]

    # 2. Karten für jeden Flow erstellen
    flow_cards = []

    for flow_name, run_func in flows_to_process:
        try:
            # Custom UI prüfen
            custom_ui = uis.get(flow_name)
            is_custom = bool(custom_ui)

            # View registrieren
            DynamicFlowView = type(
                f"FlowView_{flow_name}",
                (FlowWrapperView,),
                {
                    "__init__": lambda self, fn=flow_name, rf=run_func, cu=custom_ui:
                        FlowWrapperView.__init__(self, fn, rf, cu),
                    "__doc__": f"Auto-generated view for flow {flow_name}",
                },
            )
            from toolboxv2.mods.minu import register_view
            register_view(flow_name, DynamicFlowView)

            # Karte bauen
            badge = Badge("Custom UI", variant="success") if is_custom else Badge("Auto UI", variant="secondary")

            # Docstring bereinigen
            doc = (run_func.__doc__ or "No description available.").strip().split('\n')[0]
            if len(doc) > 60:
                doc = doc[:57] + "..."

            # WICHTIG: Echter HTML Link für Navigation statt Button-Event
            action_btn_html = (
                f'<a href="/api/Minu/render?view={flow_name}&ssr=True" '
                f'class="btn btn-primary text-center" '
                f'style="text-decoration: none; height: 1em;">Open Flow</a>'
            )

            card = Card(
                Row(
                    Heading(flow_name.replace("_", " ").title(), level=4),
                    badge,
                    justify="between",
                    align="start",
                    className="mb-2"
                ),
                Text(doc, className="text-secondary text-sm mb-4 h-12 overflow-hidden"),
                Custom(html=action_btn_html),
                className="hover:shadow-md transition-shadow duration-200 h-full flex flex-col"
            )

            flow_cards.append(card)

        except Exception as e:
            print(f"[Minu] Error registering flow UI for {flow_name}: {e}")
            flow_cards.append(Alert(f"Error loading {flow_name}: {e}", variant="error"))

    # 3. Dashboard zusammenbauen
    dashboard = Column(
        Row(
            Heading("Flow Registry", level=1),
            Badge(f"{len(flow_cards)} Flows", variant="info"),
            gap="4",
            align="center",
            className="mb-8 pb-4 border-b border-neutral-200"
        ),
        Grid(*flow_cards, cols=3, gap="6") if flow_cards else Alert("No flows found.", variant="warning"),
        className="container mx-auto p-8 max-w-7xl"
    )

    # 4. Render HTML
    return render_to_html(dashboard)


def inject_callbacks_into_view(view: MinuView, callbacks: Dict[str, Callable]):
    """Manuell Callbacks in eine View injizieren."""
    for handler_name, callback_func in callbacks.items():
        async def handler_wrapper(event, func=callback_func):
            result = func(event)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        setattr(view, handler_name, handler_wrapper)
