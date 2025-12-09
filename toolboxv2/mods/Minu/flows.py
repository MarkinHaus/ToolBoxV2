"""
Minu UI Framework - Flow Helpers V3
====================================
Utility functions für UIs mit stabilem Callback-System.

WICHTIG: Callbacks werden jetzt an die View gebunden, nicht global gespeichert.
Die View muss `register_callback` Methode haben.
"""

from typing import Any, Dict, List, Union, Optional, Callable
import json
import html as html_module

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
    Checkbox,
    Textarea,
    List as MinuList,
    ListItem,
)


# ============================================================================
# CONTEXT-AWARE CALLBACK REGISTRATION
# ============================================================================

# Thread-local storage für den aktuellen View-Context
import threading
_view_context = threading.local()


def set_current_view(view):
    """Setzt den aktuellen View-Context für Callback-Registrierung."""
    _view_context.current_view = view


def get_current_view():
    """Holt den aktuellen View-Context."""
    return getattr(_view_context, 'current_view', None)


def clear_current_view():
    """Löscht den View-Context."""
    _view_context.current_view = None


def _normalize_handler(handler: Union[str, Callable, None], hint: str = "") -> Optional[str]:
    """
    Konvertiert Handler zu Handler-Name.

    Wenn es eine Callable ist, wird sie beim aktuellen View registriert.

    Args:
        handler: String-Handler-Name oder Python-Funktion
        hint: Optionaler Hint für stabile ID-Generierung

    Returns:
        Handler-Name als String
    """
    if handler is None:
        return None

    if isinstance(handler, str):
        return handler

    if callable(handler):
        view = get_current_view()
        if view and hasattr(view, 'register_callback'):
            return view.register_callback(handler, hint)
        else:
            # Fallback: Funktionsname verwenden
            return getattr(handler, '__name__', f'callback_{id(handler)}')

    raise ValueError(f"Invalid handler type: {type(handler)}")


# ============================================================================
# ENHANCED COMPONENTS MIT CALLBACK-UNTERSTÜTZUNG
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
        variant: Button-Stil (primary, secondary, ghost)
        disabled: Deaktiviert?
        icon: Optional Icon-Name
        className: CSS-Klassen

    Example:
        def handle_click(event):
            print("Clicked!", event)

        CallbackButton("Click Me", on_click=handle_click)
    """
    handler_name = _normalize_handler(on_click, hint=f"btn_{label[:20]}")
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
    """Input mit Callback-Unterstützung."""
    change_handler = _normalize_handler(on_change, hint=f"input_change_{bind or 'anon'}")
    submit_handler = _normalize_handler(on_submit, hint=f"input_submit_{bind or 'anon'}")

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


def CallbackCheckbox(
    label: str = "",
    checked: bool = False,
    bind: str | None = None,
    on_change: Union[str, Callable, None] = None,
    className: str | None = None,
    **props,
) -> Component:
    """Checkbox mit Callback-Unterstützung."""
    change_handler = _normalize_handler(on_change, hint=f"checkbox_{bind or 'anon'}")

    return Checkbox(
        label=label,
        checked=checked,
        bind=bind,
        on_change=change_handler,
        className=className,
        **props
    )


def CallbackSelect(
    options: List[Dict[str, str]],
    value: str = "",
    bind: str | None = None,
    on_change: Union[str, Callable, None] = None,
    label: str | None = None,
    placeholder: str = "Select...",
    className: str | None = None,
    **props,
) -> Component:
    """Select mit Callback-Unterstützung."""
    change_handler = _normalize_handler(on_change, hint=f"select_{bind or 'anon'}")

    return Select(
        options=options,
        value=value,
        bind=bind,
        on_change=change_handler,
        label=label,
        placeholder=placeholder,
        className=className,
        **props
    )


# ============================================================================
# AUTO UI GENERATION
# ============================================================================

def ui_for_data(
    data: Any,
    title: Optional[str] = None,
    editable: bool = False,
    on_save: Union[str, Callable, None] = None,
) -> Component:
    """
    Generiert automatisch eine UI für beliebige Daten.

    Args:
        data: Python-Daten (dict, list, primitiv)
        title: Optional Titel
        editable: Editierbar?
        on_save: Save-Handler
    """
    save_handler = _normalize_handler(on_save, hint="save_data")

    if data is None:
        return Alert("No data", variant="info")

    if isinstance(data, dict):
        return _dict_to_ui(data, title, editable, save_handler)

    if isinstance(data, (list, tuple)):
        if data and all(isinstance(item, dict) for item in data):
            return _list_of_dicts_to_table(data, title)
        return _list_to_ui(data, title)

    if isinstance(data, bool):
        return Badge("Yes" if data else "No", variant="success" if data else "error")

    if isinstance(data, (int, float)):
        return _value_display(data, title)

    if isinstance(data, str):
        if len(data) > 200:
            return Card(Text(data, className="whitespace-pre-wrap"), title=title or "Text")
        return _value_display(data, title)

    return _value_display(str(data), title)


def _dict_to_ui(
    data: Dict[str, Any],
    title: Optional[str] = None,
    editable: bool = False,
    on_save: Optional[str] = None,
) -> Component:
    """Dict zu UI."""
    rows = []

    for key, value in data.items():
        label = key.replace("_", " ").title()

        if isinstance(value, dict):
            rows.append(Column(
                Heading(label, level=4),
                _dict_to_ui(value),
                className="ml-4 mt-2"
            ))
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(item, dict) for item in value):
                rows.append(Column(
                    Heading(label, level=4),
                    _list_of_dicts_to_table(value),
                    className="mt-2"
                ))
            else:
                rows.append(_key_value_row(label, ", ".join(str(v) for v in value[:5])))
        else:
            rows.append(_key_value_row(label, str(value)[:100]))

    if editable and on_save:
        rows.append(Divider())
        rows.append(Row(
            Button("Save", on_click=on_save, variant="primary"),
            justify="end"
        ))

    return Card(*rows, title=title) if title else Column(*rows, gap="2")


def _list_to_ui(data: List[Any], title: Optional[str] = None) -> Component:
    """List zu UI."""
    items = [Text(f"• {str(item)[:100]}") for item in data[:20]]
    if len(data) > 20:
        items.append(Text(f"... and {len(data) - 20} more items", className="text-secondary"))

    return Card(*items, title=title or f"List ({len(data)} items)")


def _list_of_dicts_to_table(data: List[Dict], title: Optional[str] = None) -> Component:
    """List of Dicts zu Table."""
    if not data:
        return Alert("No data", variant="info")

    columns = [{"key": k, "label": k.replace("_", " ").title()} for k in data[0].keys()]

    table = Table(columns=columns, data=data[:50])

    if title:
        return Card(table, title=title)
    return table


def _key_value_row(key: str, value: str) -> Component:
    """Key-Value Zeile."""
    return Row(
        Text(f"{key}:", className="font-medium text-secondary"),
        Text(value),
        justify="between",
        className="py-1"
    )


def _value_display(value: Any, label: Optional[str] = None) -> Component:
    """Einfache Wert-Anzeige."""
    if label:
        return Row(
            Text(f"{label}:", className="font-medium"),
            Text(str(value)),
            gap="2"
        )
    return Text(str(value))


# ============================================================================
# FORM GENERATION
# ============================================================================

def form_for(
    schema: Dict[str, Dict[str, Any]],
    values: Optional[Dict[str, Any]] = None,
    on_submit: Union[str, Callable, None] = "submit_form",
    title: Optional[str] = None,
    submit_label: str = "Submit",
) -> Component:
    """
    Generiert ein Formular aus einem Schema.

    Args:
        schema: Feld-Schema {name: {type, label, default, options, ...}}
        values: Initiale Werte
        on_submit: Submit-Handler
        title: Formular-Titel
        submit_label: Text für Submit-Button

    Example:
        schema = {
            "name": {"type": "text", "label": "Name"},
            "email": {"type": "email", "label": "Email"},
            "role": {"type": "select", "options": [{"value": "user", "label": "User"}]}
        }
        form_for(schema, on_submit=my_handler)
    """
    values = values or {}
    fields = []
    submit_handler = _normalize_handler(on_submit, hint="form_submit")

    for name, config in schema.items():
        field_type = config.get("type", "text")
        label = config.get("label", name.replace("_", " ").title())
        placeholder = config.get("placeholder", "")
        default = config.get("default", "")
        value = values.get(name, default)

        if field_type == "select":
            fields.append(Select(
                options=config.get("options", []),
                value=str(value) if value else "",
                label=label,
                bind=name,
                placeholder=placeholder or "Select..."
            ))

        elif field_type == "checkbox":
            fields.append(Checkbox(
                label=label,
                checked=bool(value),
                bind=name
            ))

        elif field_type == "textarea":
            fields.append(Column(
                Text(label, className="text-sm font-medium mb-1"),
                Textarea(
                    value=str(value) if value else "",
                    placeholder=placeholder,
                    bind=name,
                    rows=config.get("rows", 4)
                ),
                gap="1"
            ))

        else:
            fields.append(Input(
                value=str(value) if value else "",
                placeholder=placeholder,
                input_type=field_type,
                label=label,
                bind=name
            ))

    fields.append(Spacer())
    fields.append(Button(submit_label, on_click=submit_handler, variant="primary", className="w-full"))

    form_content = Column(*fields, gap="3")

    if title:
        return Card(form_content, title=title)
    return form_content


# ============================================================================
# CONVENIENCE COMPONENTS
# ============================================================================

def data_card(
    data: Dict[str, Any],
    title: Optional[str] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
) -> Component:
    """
    Data Card mit Actions.

    Args:
        data: Daten-Dict
        title: Titel
        actions: Liste von {label, handler, variant, icon}
    """
    rows = [
        _key_value_row(key.replace("_", " ").title(), str(value)[:100])
        for key, value in data.items()
    ]

    if actions:
        rows.append(Divider())
        buttons = []
        for action in actions:
            handler = _normalize_handler(
                action.get("handler"),
                hint=f"action_{action.get('label', 'btn')}"
            )
            buttons.append(Button(
                action.get("label", "Action"),
                on_click=handler,
                variant=action.get("variant", "secondary"),
                icon=action.get("icon")
            ))
        rows.append(Row(*buttons, justify="end", gap="2"))

    return Card(*rows, title=title)


def data_table(
    data: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    on_row_click: Union[str, Callable, None] = None,
) -> Component:
    """Data Table mit optionalem Row-Click-Handler."""
    if not data:
        return Alert("No data available", variant="info")

    if columns:
        col_defs = [{"key": c, "label": c.replace("_", " ").title()} for c in columns]
    else:
        col_defs = [{"key": k, "label": k.replace("_", " ").title()} for k in data[0].keys()]

    row_handler = _normalize_handler(on_row_click, hint="table_row_click")

    table = Table(columns=col_defs, data=data, on_row_click=row_handler)

    if title:
        return Card(table, title=title)
    return table


def stats_grid(stats: List[Dict[str, Any]], cols: int = 4) -> Component:
    """Stats Grid für KPIs."""
    cards = []

    for stat in stats:
        elements = []

        if stat.get("icon"):
            elements.append(Icon(stat["icon"], size="32"))

        elements.append(Heading(str(stat.get("value", 0)), level=2))
        elements.append(Text(stat.get("label", ""), className="text-secondary"))

        if stat.get("change"):
            change = stat["change"]
            is_positive = str(change).startswith("+") or (isinstance(change, (int, float)) and change > 0)
            elements.append(Badge(str(change), variant="success" if is_positive else "error"))

        cards.append(Card(*elements, className="text-center"))

    return Grid(*cards, cols=cols)


def action_bar(
    actions: List[Dict[str, Any]],
    title: Optional[str] = None
) -> Component:
    """Action Bar mit Buttons."""
    left = []
    if title:
        left.append(Heading(title, level=3))

    buttons = []
    for i, action in enumerate(actions):
        handler = _normalize_handler(
            action.get("handler"),
            hint=f"actionbar_{i}"
        )
        buttons.append(Button(
            action.get("label", ""),
            on_click=handler,
            variant=action.get("variant", "secondary"),
            icon=action.get("icon")
        ))

    return Row(
        Row(*left) if left else Spacer(),
        Row(*buttons, gap="2"),
        justify="between",
        className="mb-4"
    )


# ============================================================================
# RESULT WRAPPER
# ============================================================================

def ui_result(component: Component, title: Optional[str] = None) -> dict:
    """Wrap Component für Flow-Return."""
    result = {"minu": True, "component": component.to_dict()}
    if title:
        result["title"] = title
    return result


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Context
    "set_current_view",
    "get_current_view",
    "clear_current_view",
    # Callback Components
    "CallbackButton",
    "CallbackInput",
    "CallbackCheckbox",
    "CallbackSelect",
    # UI Generators
    "ui_for_data",
    "form_for",
    "data_card",
    "data_table",
    "stats_grid",
    "action_bar",
    # Utils
    "ui_result",
]
