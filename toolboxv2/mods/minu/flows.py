"""
Minu UI Framework - Flow Helpers (Enhanced mit Python Callbacks)
=================================================================
Utility functions für generating UIs in Toolbox Flows mit direkter
Python-Callback-Unterstützung.

NEU: Python-Funktionen können als on_click/on_change Handler übergeben werden.
Diese werden automatisch in der View registriert.

Usage:
    def my_handler(event):
        print("Button clicked!", event)

    return Button("Click me", on_click=my_handler)
"""

from typing import Any, Dict, List, Union, Optional, Callable
import json
import asyncio
import inspect

from .core import (
    Component,
    Card,
    Text,
    Heading,
    Row,
    Column,
    Grid,
    Table,
    Button,
    Input,
    Select,
    Form,
    Badge,
    Icon,
    Divider,
    Spacer,
    Alert,
    Progress,
    List as MinuList,
    ListItem,
)


import html
import json
from typing import Any, Dict, Optional
from .core import Component, ComponentType

# ============================================================================
# CALLBACK REGISTRY - Verwaltung von Python-Callbacks
# ============================================================================

class CallbackRegistry:
    """
    Verwaltet Python-Callbacks für Flow-generierte Components.
    Ermöglicht direkte Funktionsübergabe statt nur String-Handler-Namen.
    """

    def __init__(self):
        self._callbacks: Dict[str, Callable] = {}
        self._counter = 0

    def register(self, callback: Callable) -> str:
        """
        Registriert einen Callback und gibt Handler-Namen zurück.

        Args:
            callback: Python-Funktion (sync oder async)

        Returns:
            Handler-Name als String
        """
        if not callable(callback):
            raise ValueError("callback must be callable")

        # Generiere eindeutigen Handler-Namen
        handler_name = f"_flow_callback_{self._counter}"
        self._counter += 1

        self._callbacks[handler_name] = callback
        return handler_name

    def get(self, handler_name: str) -> Optional[Callable]:
        """Holt einen registrierten Callback"""
        return self._callbacks.get(handler_name)

    def get_all(self) -> Dict[str, Callable]:
        """Gibt alle registrierten Callbacks zurück"""
        return self._callbacks.copy()

    def clear(self):
        """Löscht alle Callbacks"""
        self._callbacks.clear()
        self._counter = 0


# Globale Registry pro Thread/Context
_callback_registry = CallbackRegistry()


def register_callback(callback: Callable) -> str:
    """Convenience-Funktion zum Registrieren von Callbacks"""
    return _callback_registry.register(callback)


def get_callback(handler_name: str) -> Optional[Callable]:
    """Convenience-Funktion zum Abrufen von Callbacks"""
    return _callback_registry.get(handler_name)


def get_all_callbacks() -> Dict[str, Callable]:
    """Gibt alle registrierten Callbacks zurück"""
    return _callback_registry.get_all()


def clear_callbacks():
    """Löscht alle registrierten Callbacks"""
    _callback_registry.clear()


# ============================================================================
# ENHANCED COMPONENT FACTORIES MIT CALLBACK-UNTERSTÜTZUNG
# ============================================================================

def _normalize_handler(handler: Union[str, Callable, None]) -> Optional[str]:
    """
    Konvertiert Handler (String oder Callable) zu Handler-Name.

    Args:
        handler: String-Handler-Name oder Python-Funktion

    Returns:
        Handler-Name als String oder None
    """
    if handler is None:
        return None

    if isinstance(handler, str):
        return handler

    if callable(handler):
        return register_callback(handler)

    raise ValueError(f"Invalid handler type: {type(handler)}")


# ============================================================================
# ENHANCED BUTTON MIT CALLBACK
# ============================================================================

def CallbackButton(
    label: str,
    on_click: Union[str, Callable, None] = None,
    variant: str = "primary",
    disabled: bool = False,
    icon: str | None = None,
    className: str | None = None,
    **props,
) -> Component:
    """
    Button mit Python-Callback-Unterstützung.

    Args:
        label: Button-Text
        on_click: String-Handler ODER Python-Funktion
        variant: Button-Stil
        disabled: Deaktiviert?
        icon: Optional Icon
        className: CSS-Klassen
        **props: Weitere Props

    Example:
        def handle_save(event):
            print("Saving...", event)

        CallbackButton("Save", on_click=handle_save)
    """
    handler_name = _normalize_handler(on_click)
    return Button(
        label,
        on_click=handler_name,
        variant=variant,
        disabled=disabled,
        icon=icon,
        className=className,
        **props
    )


def CallbackInput(
    placeholder: str = "",
    value: str = "",
    input_type: str = "text",
    bind: str | None = None,
    on_change: Union[str, Callable, None] = None,
    on_submit: Union[str, Callable, None] = None,
    label: str | None = None,
    className: str | None = None,
    **props,
) -> Component:
    """Input mit Callback-Unterstützung"""
    change_handler = _normalize_handler(on_change)
    submit_handler = _normalize_handler(on_submit)

    return Input(
        placeholder=placeholder,
        value=value,
        input_type=input_type,
        bind=bind,
        on_change=change_handler,
        on_submit=submit_handler,
        label=label,
        className=className,
        **props
    )


# ============================================================================
# AUTO UI GENERATION (ENHANCED)
# ============================================================================

def ui_for_data(
    data: Any,
    title: Optional[str] = None,
    editable: bool = False,
    on_save: Union[str, Callable, None] = None,
) -> Component:
    """
    Auto-generate UI mit Callback-Support.

    Args:
        data: Python-Daten
        title: Optional Titel
        editable: Editierbar?
        on_save: String-Handler ODER Python-Funktion
    """
    save_handler = _normalize_handler(on_save)

    if data is None:
        return Alert("No data", variant="info")

    if isinstance(data, dict):
        return _dict_to_ui(data, title, editable, save_handler)

    elif isinstance(data, (list, tuple)):
        if data and all(isinstance(item, dict) for item in data):
            return _list_of_dicts_to_table(data, title)
        else:
            return _list_to_ui(data, title)

    elif isinstance(data, bool):
        return _value_badge(data, "success" if data else "error")

    elif isinstance(data, (int, float)):
        return _value_display(data, title)

    elif isinstance(data, str):
        if len(data) > 100:
            return Card(Text(data), title=title or "Text")
        return _value_display(data, title)

    else:
        return _value_display(str(data), title)


def _dict_to_ui(
    data: Dict[str, Any],
    title: Optional[str] = None,
    editable: bool = False,
    on_save: Optional[str] = None,
) -> Component:
    """Convert dict mit Callback-Support"""
    rows = []

    for key, value in data.items():
        label = key.replace("_", " ").title()

        if editable:
            rows.append(_generate_input_for_value(key, label, value))
        else:
            if isinstance(value, dict):
                rows.append(
                    Column(
                        Heading(label, level=4),
                        _dict_to_ui(value),
                        className="ml-4"
                    )
                )
            elif isinstance(value, (list, tuple)):
                if value and all(isinstance(item, dict) for item in value):
                    rows.append(
                        Column(
                            Heading(label, level=4),
                            _list_of_dicts_to_table(value),
                            className="ml-4",
                        )
                    )
                else:
                    rows.append(_key_value_row(label, ", ".join(str(v) for v in value)))
            elif isinstance(value, bool):
                rows.append(
                    Row(
                        Text(f"{label}:", className="font-medium w-32"),
                        Badge(
                            "Yes" if value else "No",
                            variant="success" if value else "error",
                        ),
                        gap="4",
                    )
                )
            else:
                rows.append(
                    _key_value_row(label, str(value) if value is not None else "—")
                )

    content = [*rows]

    if editable and on_save:
        content.append(Divider())
        content.append(
            Row(
                Button("Save", on_click=on_save, variant="primary"),
                Button("Cancel", on_click="cancel_edit", variant="secondary"),
                justify="end",
            )
        )

    if editable:
        return Form(*content, on_submit=on_save)

    return Card(*content, title=title)


def _key_value_row(key: str, value: str) -> Component:
    """Key-Value Row"""
    return Row(
        Text(f"{key}:", className="font-medium text-secondary w-32"),
        Text(value, className="flex-1"),
        gap="4",
        className="py-1",
    )


def _list_of_dicts_to_table(data: List[Dict], title: Optional[str] = None) -> Component:
    """List to Table"""
    if not data:
        return Alert("No data", variant="info")

    columns = [
        {"key": key, "label": key.replace("_", " ").title()} for key in data[0].keys()
    ]

    table = Table(columns=columns, data=data)

    if title:
        return Card(table, title=title)
    return table


def _list_to_ui(data: List, title: Optional[str] = None) -> Component:
    """List to UI"""
    items = [
        ListItem(ui_for_data(item) if isinstance(item, (dict, list)) else Text(str(item)))
        for item in data
    ]

    list_comp = MinuList(*items)

    if title:
        return Card(list_comp, title=title)
    return list_comp


def _value_display(value: Any, title: Optional[str] = None) -> Component:
    """Single Value Display"""
    if title:
        return Row(
            Text(f"{title}:", className="font-medium text-secondary"),
            Text(str(value)),
            gap="2",
        )
    return Text(str(value))


def _value_badge(value: Any, variant: str = "default") -> Component:
    """Value as Badge"""
    return Badge(str(value), variant=variant)


def _generate_input_for_value(key: str, label: str, value: Any) -> Component:
    """Generate Input for Value"""
    if isinstance(value, bool):
        from .core import Switch
        return Switch(label=label, checked=value, bind=key)

    elif isinstance(value, (int, float)):
        return Input(label=label, value=str(value), input_type="number", bind=key)

    elif isinstance(value, str):
        if len(value) > 100:
            from .core import Textarea
            return Column(
                Text(label, className="font-medium"),
                Textarea(value=value, bind=key, rows=4),
            )
        if "@" in value:
            return Input(label=label, value=value, input_type="email", bind=key)
        return Input(label=label, value=value, bind=key)

    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
        return Input(
            label=label,
            value=", ".join(value),
            placeholder="Comma-separated values",
            bind=key,
        )

    else:
        from .core import Textarea
        return Column(
            Text(label, className="font-medium"),
            Textarea(
                value=json.dumps(value, indent=2) if value else "", bind=key, rows=4
            ),
        )


# ============================================================================
# ENHANCED CONVENIENCE FUNCTIONS
# ============================================================================

def data_card(
    data: Dict[str, Any],
    title: Optional[str] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
) -> Component:
    """
    Data Card mit Callback-Actions.

    Args:
        data: Daten-Dict
        title: Titel
        actions: Liste von Action-Dicts mit optionalen Callbacks

    Example:
        def edit_handler(e):
            print("Edit clicked")

        data_card(
            {"name": "John"},
            actions=[
                {"label": "Edit", "handler": edit_handler},
                {"label": "Delete", "handler": "delete_user"}
            ]
        )
    """
    rows = [
        _key_value_row(key.replace("_", " ").title(), str(value))
        for key, value in data.items()
    ]

    if actions:
        rows.append(Divider())

        action_buttons = []
        for action in actions:
            handler = action.get("handler")
            normalized_handler = _normalize_handler(handler)

            action_buttons.append(
                Button(
                    action["label"],
                    on_click=normalized_handler,
                    variant=action.get("variant", "secondary"),
                )
            )

        rows.append(Row(*action_buttons, justify="end", gap="2"))

    return Card(*rows, title=title)


def data_table(
    data: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    on_row_click: Union[str, Callable, None] = None,
    searchable: bool = False,
) -> Component:
    """
    Data Table mit Callback-Support.

    Args:
        data: Liste von Dicts
        columns: Optional Spalten
        title: Titel
        on_row_click: String-Handler ODER Python-Funktion
        searchable: Suchfeld?
    """
    if not data:
        return Alert("No data available", variant="info")

    if columns:
        col_defs = [{"key": c, "label": c.replace("_", " ").title()} for c in columns]
    else:
        col_defs = [
            {"key": k, "label": k.replace("_", " ").title()} for k in data[0].keys()
        ]

    row_handler = _normalize_handler(on_row_click)

    elements = []

    if searchable:
        elements.append(Input(placeholder="Search...", bind="table_search"))
        elements.append(Spacer())

    elements.append(Table(columns=col_defs, data=data, on_row_click=row_handler))

    if title:
        return Card(*elements, title=title)
    return Column(*elements)


def form_for(
    schema: Dict[str, Dict[str, Any]],
    values: Optional[Dict[str, Any]] = None,
    on_submit: Union[str, Callable, None] = "submit_form",
    title: Optional[str] = None,
    submit_label: str = "Submit",
) -> Component:
    """
    Form mit Callback-Support.

    Args:
        schema: Feld-Schema
        values: Initial Values
        on_submit: String-Handler ODER Python-Funktion
        title: Titel
        submit_label: Submit-Button-Text

    Example:
        def handle_submit(event):
            print("Form submitted:", event)

        form_for(
            {"name": {"type": "text", "label": "Name"}},
            on_submit=handle_submit
        )
    """
    values = values or {}
    fields = []

    submit_handler = _normalize_handler(on_submit)

    for name, config in schema.items():
        field_type = config.get("type", "text")
        label = config.get("label", name.replace("_", " ").title())
        placeholder = config.get("placeholder", "")
        value = values.get(name, config.get("default", ""))

        if field_type == "select":
            fields.append(
                Select(
                    options=config.get("options", []),
                    value=str(value),
                    label=label,
                    bind=name,
                )
            )

        elif field_type == "checkbox":
            from .core import Checkbox
            fields.append(Checkbox(label=label, checked=bool(value), bind=name))

        elif field_type == "textarea":
            from .core import Textarea
            fields.append(
                Column(
                    Text(label, className="font-medium"),
                    Textarea(
                        value=str(value),
                        placeholder=placeholder,
                        bind=name,
                        rows=config.get("rows", 4),
                    ),
                )
            )

        else:
            fields.append(
                Input(
                    value=str(value) if value else "",
                    placeholder=placeholder,
                    input_type=field_type,
                    label=label,
                    bind=name,
                )
            )

    fields.append(Spacer())
    fields.append(Row(Button(submit_label, variant="primary"), justify="end"))

    form = Form(*fields, on_submit=submit_handler)

    if title:
        return Card(form, title=title)
    return form


def stats_grid(stats: List[Dict[str, Any]], cols: int = 4) -> Component:
    """Stats Grid"""
    cards = []

    for stat in stats:
        elements = []

        if stat.get("icon"):
            elements.append(Icon(stat["icon"], size="32"))

        elements.append(Heading(str(stat.get("value", 0)), level=2))
        elements.append(Text(stat.get("label", ""), className="text-secondary"))

        if stat.get("change"):
            change = stat["change"]
            is_positive = change.startswith("+") or (
                isinstance(change, (int, float)) and change > 0
            )
            elements.append(
                Badge(str(change), variant="success" if is_positive else "error")
            )

        cards.append(Card(*elements, className="card text-center"))

    return Grid(*cards, cols=cols)


def action_bar(
    actions: List[Dict[str, Any]],
    title: Optional[str] = None
) -> Component:
    """
    Action Bar mit Callback-Support.

    Args:
        actions: Liste von Actions mit Callbacks
        title: Optional Titel

    Example:
        def new_item(e):
            print("New item")

        action_bar(
            [
                {"label": "New", "handler": new_item, "icon": "add"},
                {"label": "Export", "handler": "export_data"}
            ]
        )
    """
    left = []
    if title:
        left.append(Heading(title, level=3))

    buttons = []
    for action in actions:
        handler = _normalize_handler(action.get("handler"))

        btn = Button(
            action.get("label", ""),
            on_click=handler,
            variant=action.get("variant", "secondary"),
            icon=action.get("icon"),
        )
        buttons.append(btn)

    return Row(
        Row(*left) if left else Spacer(),
        Row(*buttons, gap="2"),
        justify="between",
        className="mb-4",
    )


# ============================================================================
# RESULT WRAPPERS
# ============================================================================

def ui_result(component: Component, title: Optional[str] = None) -> dict:
    """Wrap Component für Flow-Return"""
    result = {"minu": True, "component": component.to_dict()}
    if title:
        result["title"] = title
    return result


# Export all
__all__ = [
    # Callback System
    "CallbackRegistry",
    "register_callback",
    "get_callback",
    "get_all_callbacks",
    "clear_callbacks",
    # Enhanced Components
    "CallbackButton",
    "CallbackInput",
    # UI Generators
    "ui_for_data",
    "data_card",
    "data_table",
    "form_for",
    "stats_grid",
    "action_bar",
    # Utils
    "ui_result",
]
