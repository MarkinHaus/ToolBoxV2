"""
Minu UI Framework for Toolbox V2
================================
A lightweight, reactive UI framework that generates JSON-based UI definitions
and sends them via WebSocket for real-time rendering in TBJS.

Design Principles:
1. Simple Python API - UI als Python-Objekte
2. Reactive State - Automatische Updates bei Änderungen
3. Minimal Payloads - Nur Diffs werden gesendet
4. Native Toolbox - Volle Integration mit Result, Export, etc.
"""

from __future__ import annotations

import asyncio
import json
import uuid
import weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from toolboxv2 import RequestData, get_app
from toolboxv2.mods.Minu.shared import SharedSection
from toolboxv2.mods.Minu.user import AuthenticatedUserWrapper, AnonymousUser

# Type definitions
T = TypeVar("T")
EventHandler = Callable[..., Any]
Children = Union["Component", List["Component"], str, None]

class MinuJSONEncoder(json.JSONEncoder):
    """
    Automatische Umwandlung von ReactiveState in den eigentlichen Wert.
    Verhindert Fehler, wenn man aus Versehen 'self.state' statt 'self.state.value' übergibt.
    """
    def default(self, obj):
        # Wenn es ein ReactiveState ist, nimm den Wert
        if isinstance(obj, ReactiveState):
            return obj.value
        # Wenn das Objekt eine to_dict Methode hat (z.B. Component), nutze diese
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        # Fallback auf Standard-Verhalten (z.B. für datetime)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj) # Letzter Ausweg: String-Repräsentation
# ============================================================================
# REACTIVE STATE SYSTEM
# ============================================================================


class StateChange:
    """Represents a single state change for diffing"""

    __slots__ = ("path", "old_value", "new_value", "timestamp")

    def __init__(self, path: str, old_value: Any, new_value: Any):
        self.path = path
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = (
            asyncio.get_event_loop().time()
            if asyncio.get_event_loop().is_running()
            else 0
        )


class ReactiveState(Generic[T]):
    """
    A reactive state container that tracks changes and notifies observers.

    Usage:
        name = ReactiveState("initial")
        name.value = "changed"  # Triggers observers
    """

    _observers: weakref.WeakSet
    _value: T
    _path: str
    _str_hash: str

    def __init__(self, initial: T, path: str = ""):
        self._value = initial
        self._path = path
        self._observers = weakref.WeakSet()
        self._str_hash = f"ReactiveState({self._value!r})"

    def update_hash(self):
        self._str_hash = f"ReactiveState({self._value!r})"

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, new_value: T):

        if self._value != new_value or self._str_hash != f"ReactiveState({new_value!r})":
            old = self._value
            self._value = new_value
            change = StateChange(self._path, old, new_value)
            self._notify(change)
            self.update_hash()
        else:
            print("Same value, no change", new_value == self._value)

    def _notify(self, change: StateChange):
        for observer in self._observers:
            if hasattr(observer, "_on_state_change"):
                observer._on_state_change(change)

    def bind(self, observer: MinuView):
        """Bind this state to a view for automatic updates"""
        self._observers.add(observer)

    def __repr__(self):
        return f"ReactiveState({self._value!r})"


def State(initial: T, path: str = "") -> ReactiveState[T]:
    """Factory function for creating reactive state"""
    return ReactiveState(initial, path)


# ============================================================================
# COMPONENT SYSTEM
# ============================================================================


class ComponentType(str, Enum):
    """All supported component types"""

    # Layout
    CARD = "card"
    ROW = "row"
    COLUMN = "column"
    GRID = "grid"
    SPACER = "spacer"
    DIVIDER = "divider"

    # Content
    TEXT = "text"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    ICON = "icon"
    IMAGE = "image"
    BADGE = "badge"

    # Input
    BUTTON = "button"
    INPUT = "input"
    TEXTAREA = "textarea"
    SELECT = "select"
    CHECKBOX = "checkbox"
    SWITCH = "switch"
    SLIDER = "slider"

    # Feedback
    ALERT = "alert"
    TOAST = "toast"
    PROGRESS = "progress"
    SPINNER = "spinner"

    # Navigation
    LINK = "link"
    TABS = "tabs"
    TAB = "tab"
    NAV = "nav"

    # Data
    TABLE = "table"
    LIST = "list"
    LISTITEM = "listitem"

    # Special
    MODAL = "modal"
    WIDGET = "widget"
    FORM = "form"
    CUSTOM = "custom"

    DYNAMIC = "dynamic"


@dataclass
class ComponentStyle:
    """CSS-like styling for components"""

    margin: str | None = None
    padding: str | None = None
    width: str | None = None
    height: str | None = None
    color: str | None = None
    background: str | None = None
    border: str | None = None
    borderRadius: str | None = None
    fontSize: str | None = None
    fontWeight: str | None = None
    display: str | None = None
    flexDirection: str | None = None
    alignItems: str | None = None
    justifyContent: str | None = None
    gap: str | None = None

    def to_dict(self) -> Dict[str, str]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_str(cls, css_string: str) -> ComponentStyle:
        """
        Parse CSS string into ComponentStyle.

        Examples:
            "margin: 10px; padding: 5px; background: red;"
            "width: 100%; height: auto; display: flex; gap: 1rem;"
        """
        if not css_string or not css_string.strip():
            return cls()

        # CSS property name -> dataclass field name mapping
        css_to_field = {
            "margin": "margin",
            "padding": "padding",
            "width": "width",
            "height": "height",
            "color": "color",
            "background": "background",
            "background-color": "background",
            "border": "border",
            "border-radius": "borderRadius",
            "font-size": "fontSize",
            "font-weight": "fontWeight",
            "display": "display",
            "flex-direction": "flexDirection",
            "align-items": "alignItems",
            "justify-content": "justifyContent",
            "gap": "gap",
        }

        parsed = {}

        # Split by semicolon and process each declaration
        declarations = css_string.split(";")

        for decl in declarations:
            decl = decl.strip()
            if not decl or ":" not in decl:
                continue

            # Split property: value
            parts = decl.split(":", 1)
            if len(parts) != 2:
                continue

            prop = parts[0].strip().lower()
            value = parts[1].strip()

            # Map CSS property to field name
            field_name = css_to_field.get(prop)
            if field_name:
                parsed[field_name] = value

        return cls(**parsed)


@dataclass(eq=False)
class Component:
    """
    Base component class representing a UI element.

    All components serialize to JSON for transport to the frontend.
    """

    type: ComponentType
    id: str = field(default_factory=lambda: f"minu-{uuid.uuid4().hex[:8]}")
    children: List[Component] = field(default_factory=list)
    props: Dict[str, Any] = field(default_factory=dict)
    style: ComponentStyle | None = None
    className: str | None = None
    events: Dict[str, str] = field(default_factory=dict)  # event -> handler_name
    bindings: Dict[str, str] = field(default_factory=dict)  # prop -> state_path

    def __post_init__(self):
        # Normalize children
        if isinstance(self.children, str):
            self.children = [Text(self.children)]
        elif isinstance(self.children, Component):
            self.children = [self.children]
        elif self.children is None:
            self.children = []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to JSON-compatible dict"""
        result = {
            "type": self.type.value,
            "id": self.id,
            "props": self.props,
        }

        if self.children:
            result["children"] = [
                c.to_dict() if isinstance(c, Component) else c for c in self.children
            ]

        if self.style:
            if isinstance(self.style, str):
                self.style = ComponentStyle.from_str(self.style)
            result["style"] = self.style.to_dict()

        if self.className:
            result["className"] = self.className

        if self.events:
            result["events"] = self.events

        if self.bindings:
            result["bindings"] = self.bindings

        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ============================================================================
# COMPONENT FACTORY FUNCTIONS
# ============================================================================


class Dynamic(Component):
    """
    A container that re-renders its content on the server when bound state changes.
    Allows for true branching logic (if/else) in the UI.
    """

    def __init__(
        self,
        render_fn: Callable[[], Component | List[Component]],
        bind: List[ReactiveState] | ReactiveState,
        className: str = None,
    ):
        super().__init__(type=ComponentType.DYNAMIC, className=className)
        self.render_fn = render_fn
        # Normalize bind to list
        self.bound_states = [bind] if isinstance(bind, ReactiveState) else (bind or [])

        # Initial render
        self._update_content()

    def _update_content(self):
        """Executes the render function and updates children"""
        content = self.render_fn()
        if isinstance(content, list):
            self.children = content
        elif isinstance(content, Component):
            self.children = [content]
        else:
            self.children = [] if content is None else [Text(str(content))]

    def to_dict(self) -> Dict[str, Any]:
        # Dynamic components render as a simple generic container (like a div/Column)
        # but with a stable ID so we can target it for replacements.
        d = super().to_dict()
        d["type"] = "column"  # Render as a column container on client
        return d

def Card(
    *children: Children,
    title: str | None = None,
    subtitle: str | None = None,
    className: str = "card",
    style: ComponentStyle | None = None,
    **props,
) -> Component:
    """
    A card container with optional header.

    Usage:
        Card(
            Text("Content"),
            title="My Card",
            className="card animate-fade-in"
        )
    """
    child_list = []

    if title or subtitle:
        header_children = []
        if title:
            header_children.append(
                Component(
                    type=ComponentType.HEADING,
                    props={"level": 3, "text": title},
                    className="card-title",
                )
            )
        if subtitle:
            header_children.append(
                Component(
                    type=ComponentType.TEXT,
                    props={"text": subtitle},
                    className="text-secondary text-sm",
                )
            )
        child_list.append(
            Component(
                type=ComponentType.ROW, children=header_children, className="card-header"
            )
        )

    for child in children:
        if isinstance(child, (list, tuple)):
            child_list.extend(child)
        elif child is not None:
            child_list.append(child if isinstance(child, Component) else Text(str(child)))

    return Component(
        type=ComponentType.CARD,
        children=child_list,
        className=className,
        style=style,
        props=props,
    )


def Text(
    content: str,
    variant: str = "body",  # body, caption, overline
    className: str | None = None,
    bind: str | None = None,
    **props,
) -> Component:
    """Simple text component"""
    class_name = className or f"text-{variant}"
    bindings = {"text": bind} if bind else {}

    return Component(
        type=ComponentType.TEXT,
        props={"text": content, **props},
        className=class_name,
        bindings=bindings,
    )


def Heading(
    text: str, level: int = 1, className: str | None = None, **props
) -> Component:
    """Heading component (h1-h6)"""
    return Component(
        type=ComponentType.HEADING,
        props={"text": text, "level": level, **props},
        className=className
        or f"text-{['4xl', '3xl', '2xl', 'xl', 'lg', 'base'][level - 1]}",
    )


def Button(
    label: str,
    on_click: str | None = None,
    variant: str = "primary",  # primary, secondary, ghost
    disabled: bool = False,
    icon: str | None = None,
    className: str | None = None,
    **props,
) -> Component:
    """
    Interactive button component.

    Usage:
        Button("Save", on_click="handle_save", variant="primary")
    """
    events = {"click": on_click} if on_click else {}
    class_name = className or f"btn btn-{variant}"

    children = []
    if icon:
        children.append(Icon(icon))
    children.append(Text(label))

    return Component(
        type=ComponentType.BUTTON,
        children=children if len(children) > 1 else [],
        props={"disabled": disabled, **props},
        className=class_name,
        events=events,
    )


def Input(
    placeholder: str = "",
    value: str = "",
    input_type: str = "text",
    bind: str | None = None,
    on_change: str | None = None,
    on_submit: str | None = None,
    label: str | None = None,
    className: str | None = None,
    **props,
) -> Component:
    """
    Text input component with optional label and bindings.

    Usage:
        Input(
            placeholder="Enter name",
            bind="user.name",
            on_change="validate_name"
        )
    """
    events = {}
    if on_change:
        events["change"] = on_change
    if on_submit:
        events["submit"] = on_submit

    bindings = {"value": bind} if bind else {}

    input_comp = Component(
        type=ComponentType.INPUT,
        props={
            "placeholder": placeholder,
            "value": value,
            "inputType": input_type,
            **props,
        },
        className=className,
        events=events,
        bindings=bindings,
    )

    if label:
        return Column(
            Text(label, className="text-sm font-medium mb-1"),
            input_comp,
            className="form-field",
        )

    return input_comp


def Textarea(
    placeholder: str = "",
    value: str = "",
    bind: str | None = None,
    on_change: str | None = None,
    on_submit: str | None = None,
    label: str | None = None,
    rows: int | None = None,
    className: str | None = None,
    **props,
) -> Component:
    """
    Multiline textarea component with optional label, bindings and events.

    Usage:
        Textarea(
            placeholder="Enter description",
            bind="user.bio",
            rows=4,
            on_change="handle_bio_change"
        )
    """
    events = {}
    if on_change:
        events["change"] = on_change
    if on_submit:
        events["submit"] = on_submit

    bindings = {"value": bind} if bind else {}

    textarea_props = {
        "placeholder": placeholder,
        "value": value,
        "inputType": "textarea",  # falls dein Renderer das unterscheidet
        **props,
    }

    if rows:
        textarea_props["rows"] = rows

    textarea_comp = Component(
        type=ComponentType.TEXTAREA if hasattr(ComponentType, "TEXTAREA") else ComponentType.INPUT,
        props=textarea_props,
        className=className,
        events=events,
        bindings=bindings,
    )

    if label:
        return Column(
            Text(label, className="text-sm font-medium mb-1"),
            textarea_comp,
            className="form-field",
        )

    return textarea_comp


def Select(
    options: List[Dict[str, str]],
    value: str = "",
    placeholder: str = "Select...",
    bind: str | None = None,
    on_change: str | None = None,
    label: str | None = None,
    **props,
) -> Component:
    """
    Dropdown select component.

    Usage:
        Select(
            options=[
                {"value": "opt1", "label": "Option 1"},
                {"value": "opt2", "label": "Option 2"}
            ],
            bind="selected_option"
        )
    """
    events = {"change": on_change} if on_change else {}
    bindings = {"value": bind} if bind else {}

    select_comp = Component(
        type=ComponentType.SELECT,
        props={"options": options, "value": value, "placeholder": placeholder, **props},
        events=events,
        bindings=bindings,
    )

    if label:
        return Column(
            Text(label, className="text-sm font-medium mb-1"),
            select_comp,
            className="form-field",
        )

    return select_comp


def Checkbox(
    label: str,
    checked: bool = False,
    bind: str | None = None,
    on_change: str | None = None,
    **props,
) -> Component:
    """Checkbox input with label"""
    events = {"change": on_change} if on_change else {}
    bindings = {"checked": bind} if bind else {}

    return Component(
        type=ComponentType.CHECKBOX,
        props={"label": label, "checked": checked, **props},
        events=events,
        bindings=bindings,
    )


def Switch(
    label: str = "",
    checked: bool = False,
    bind: str | None = None,
    on_change: str | None = None,
    **props,
) -> Component:
    """Toggle switch component"""
    events = {"change": on_change} if on_change else {}
    bindings = {"checked": bind} if bind else {}

    return Component(
        type=ComponentType.SWITCH,
        props={"label": label, "checked": checked, **props},
        events=events,
        bindings=bindings,
    )


# Layout Components


def Row(
    *children: Children,
    gap: str = "4",
    align: str = "center",
    justify: str = "start",
    wrap: bool = False,
    className: str | None = None,
    **props,
) -> Component:
    """Horizontal flex container"""
    class_parts = ["flex", f"gap-{gap}", f"items-{align}", f"justify-{justify}"]
    if wrap:
        class_parts.append("flex-wrap")

    return Component(
        type=ComponentType.ROW,
        children=list(children),
        className=className or " ".join(class_parts),
        props=props,
    )


def Column(
    *children: Children,
    gap: str = "4",
    align: str = "stretch",
    className: str | None = None,
    **props,
) -> Component:
    """Vertical flex container"""
    return Component(
        type=ComponentType.COLUMN,
        children=list(children),
        className=className or f"flex flex-col gap-{gap} items-{align}",
        props=props,
    )


def Grid(
    *children: Children,
    cols: int = 2,
    gap: str = "4",
    className: str | None = None,
    **props,
) -> Component:
    """CSS Grid container"""
    return Component(
        type=ComponentType.GRID,
        children=list(children),
        className=className or f"grid grid-cols-{cols} gap-{gap}",
        props=props,
    )


def Spacer(size: str = "4", **props) -> Component:
    """Empty space component"""
    return Component(type=ComponentType.SPACER, className=f"h-{size}", props=props)


def Divider(className: str | None = None, **props) -> Component:
    """Horizontal divider line"""
    return Component(
        type=ComponentType.DIVIDER,
        className=className or "border-t border-neutral-200 my-4",
        props=props,
    )


# Feedback Components


def Alert(
    message: str,
    variant: str = "info",  # info, success, warning, error
    title: str | None = None,
    dismissible: bool = False,
    on_dismiss: str | None = None,
    **props,
) -> Component:
    """Alert/notification component"""
    events = {"dismiss": on_dismiss} if on_dismiss else {}

    return Component(
        type=ComponentType.ALERT,
        props={
            "message": message,
            "variant": variant,
            "title": title,
            "dismissible": dismissible,
            **props,
        },
        className=f"alert alert-{variant}",
        events=events,
    )


def Progress(
    value: int = 0,
    max_value: int = 100,
    label: str | None = None,
    bind: str | None = None,
    **props,
) -> Component:
    """Progress bar component"""
    bindings = {"value": bind} if bind else {}

    return Component(
        type=ComponentType.PROGRESS,
        props={"value": value, "max": max_value, "label": label, **props},
        bindings=bindings,
    )


def Spinner(size: str = "md", className: str | None = None) -> Component:
    """Loading spinner"""
    return Component(
        type=ComponentType.SPINNER,
        props={"size": size},
        className=className or "animate-spin",
    )


# Data Display


def Table(
    columns: List[Dict[str, str]],
    data: List[Dict[str, Any]],
    bind_data: str | None = None,
    on_row_click: str | None = None,
    **props,
) -> Component:
    """
    Data table component.

    Usage:
        Table(
            columns=[
                {"key": "name", "label": "Name"},
                {"key": "email", "label": "Email"}
            ],
            data=[
                {"name": "John", "email": "john@example.com"}
            ],
            bind_data="users"
        )
    """
    events = {"rowClick": on_row_click} if on_row_click else {}
    bindings = {"data": bind_data} if bind_data else {}

    return Component(
        type=ComponentType.TABLE,
        props={"columns": columns, "data": data, **props},
        events=events,
        bindings=bindings,
    )


def List(
    *items: Children, ordered: bool = False, className: str | None = None, **props
) -> Component:
    """List component"""
    return Component(
        type=ComponentType.LIST,
        children=list(items),
        props={"ordered": ordered, **props},
        className=className,
    )


def ListItem(
    *children: Children,
    on_click: str | None = None,
    className: str | None = None,
    **props,
) -> Component:
    """List item component"""
    events = {"click": on_click} if on_click else {}

    return Component(
        type=ComponentType.LISTITEM,
        children=list(children),
        className=className,
        events=events,
        props=props,
    )


# Special Components


def Icon(name: str, size: str = "24", className: str | None = None) -> Component:
    """Material icon component"""
    return Component(
        type=ComponentType.ICON,
        props={"name": name, "size": size},
        className=className or "material-symbols-outlined",
    )


def Image(
    src: str,
    alt: str = "",
    width: str | None = None,
    height: str | None = None,
    className: str | None = None,
    **props,
) -> Component:
    """Image component"""
    return Component(
        type=ComponentType.IMAGE,
        props={"src": src, "alt": alt, "width": width, "height": height, **props},
        className=className,
    )


def Badge(
    text: str,
    variant: str = "default",  # default, primary, success, warning, error
    className: str | None = None,
) -> Component:
    """Small badge/tag component"""
    return Component(
        type=ComponentType.BADGE,
        props={"text": text, "variant": variant},
        className=className or f"badge badge-{variant}",
    )


def Modal(
    *children: Children,
    title: str | None = None,
    open: bool = False,
    bind_open: str | None = None,
    on_close: str | None = None,
    **props,
) -> Component:
    """Modal dialog component"""
    events = {"close": on_close} if on_close else {}
    bindings = {"open": bind_open} if bind_open else {}

    return Component(
        type=ComponentType.MODAL,
        children=list(children),
        props={"title": title, "open": open, **props},
        events=events,
        bindings=bindings,
    )


def Widget(
    *children: Children,
    title: str = "",
    collapsible: bool = False,
    className: str | None = None,
    **props,
) -> Component:
    """Floating widget container (uses .widget CSS class)"""
    return Component(
        type=ComponentType.WIDGET,
        children=list(children),
        props={"title": title, "collapsible": collapsible, **props},
        className=className or "widget",
    )


def Form(
    *children: Children,
    on_submit: str | None = None,
    className: str | None = None,
    **props,
) -> Component:
    """Form container with submit handling"""
    events = {"submit": on_submit} if on_submit else {}

    return Component(
        type=ComponentType.FORM,
        children=list(children),
        className=className,
        events=events,
        props=props,
    )


def Tabs(
    tabs: List[Dict[str, Any]],
    active: int = 0,
    bind_active: str | None = None,
    on_change: str | None = None,
    **props,
) -> Component:
    """
    Tab navigation component.

    Usage:
        Tabs(
            tabs=[
                {"label": "Tab 1", "content": Card(Text("Content 1"))},
                {"label": "Tab 2", "content": Card(Text("Content 2"))}
            ],
            bind_active="active_tab"
        )
    """
    events = {"change": on_change} if on_change else {}
    bindings = {"active": bind_active} if bind_active else {}

    # Serialize tab content
    serialized_tabs = []
    for tab in tabs:
        serialized_tab = {"label": tab.get("label", "")}
        if "content" in tab:
            content = tab["content"]
            serialized_tab["content"] = (
                content.to_dict() if isinstance(content, Component) else content
            )
        serialized_tabs.append(serialized_tab)

    return Component(
        type=ComponentType.TABS,
        props={"tabs": serialized_tabs, "active": active, **props},
        events=events,
        bindings=bindings,
    )


def Custom(html: str = "", component_name: str | None = None, **props) -> Component:
    """
    Custom HTML or registered component.

    Usage:
        Custom(html="<div class='custom'>Custom HTML</div>")
        Custom(component_name="MyCustomComponent", data={"key": "value"})
    """
    return Component(
        type=ComponentType.CUSTOM,
        props={"html": html, "componentName": component_name, **props},
    )


# ============================================================================
# VIEW SYSTEM
# ============================================================================


class MinuView:
    """
    Base class for Minu UI views with integrated User and Shared support.

    Features:
    - Reactive state management
    - User property (authenticated or anonymous)
    - Shared sections for multi-user collaboration

    Usage:
        class MyDashboard(MinuView):
            title = State("Dashboard")

            def render(self):
                # User ist automatisch verfügbar
                if self.user.is_authenticated:
                    greeting = f"Willkommen, {self.user.name}!"
                else:
                    greeting = "Willkommen, Gast!"

                return Column(
                    Heading(self.title.value),
                    Text(greeting),
                    Button("Click me", on_click="handle_click")
                )

            async def handle_click(self, event):
                # User-Daten speichern
                if self.user.is_authenticated:
                    await self.user.set_mod_data('MyMod', {'clicked': True})
                else:
                    self.user.set_mod_data('MyMod', {'clicked': True})

    Multi-User Example:
        class GameLobby(MinuView):
            async def on_mount(self):
                # Shared Section erstellen oder beitreten
                self.game = await self.create_shared(
                    name="game_123",
                    initial_data={'players': [], 'state': 'waiting'}
                )

                # Auf Änderungen reagieren
                self.game.on_change('state', self.on_game_state_change)

            async def on_join(self, event):
                await self.game.append('players', {
                    'id': self.user.uid,
                    'name': self.user.name,
                    'score': 0
                })
    """

    _view_id: str
    _session: MinuSession | None
    _pending_changes: List[StateChange]
    _state_attrs: Dict[str, ReactiveState]
    _dynamic_components: set

    # User Integration
    _user_cache: AuthenticatedUserWrapper | AnonymousUser | None = None
    _app: Any | None = None
    request_data: RequestData | None = None

    # Shared Integration
    _shared_sections: Dict[str, SharedSection] = None

    def __init__(self, view_id: str | None = None):
        self._view_id = view_id or f"view-{uuid.uuid4().hex[:8]}"
        self._session = None
        self._pending_changes = []
        self._state_attrs = {}
        self._dynamic_components = set()
        self._user_cache = None
        self._shared_sections = {}

        # State-Attribute initialisieren
        for attr_name in dir(self.__class__):
            if not attr_name.startswith("_"):
                attr = getattr(self.__class__, attr_name)
                if isinstance(attr, ReactiveState):
                    state_copy = State(attr.value, f"{self._view_id}.{attr_name}")
                    state_copy.bind(self)
                    self._state_attrs[attr_name] = state_copy
                    setattr(self, attr_name, state_copy)

    # =================== User Property ===================

    @property
    def user(self) -> AuthenticatedUserWrapper | AnonymousUser:
        """
        Aktueller User (angemeldet oder anonym).

        Für angemeldete Nutzer:
            - user.name, user.uid, user.email, etc.
            - user.get_mod_client('ModName') für ModDataClient
            - await user.get_mod_data('ModName')
            - await user.set_mod_data('ModName', {...})

        Für anonyme Nutzer:
            - user.name == "anonymous"
            - user.level == -1
            - user.uid == "anon_<session_id>"
            - user.get_mod_data('ModName') (synchron, Session-basiert)
            - user.set_mod_data('ModName', {...}) (synchron, Session-basiert)
        """
        if self._user_cache is not None:
            return self._user_cache

        # Import hier um Circular Imports zu vermeiden
        from .user import AnonymousUser, MinuUser

        # Sync fallback wenn async nicht möglich
        if self.request_data:
            self._user_cache = MinuUser.from_request_sync(
                self._app, self.request_data
            )
            return self._user_cache

        # Default: Anonymous ohne Session
        return AnonymousUser(session_id=f"no-session-{uuid.uuid4().hex[:8]}")

    async def ensure_user(self) -> AuthenticatedUserWrapper | AnonymousUser:
        """
        Async User-Laden. Sollte zu Beginn eines Event-Handlers aufgerufen werden.

        Usage:
            async def on_submit(self, event):
                user = await self.ensure_user()
                if user.is_authenticated:
                    await user.set_mod_data('MyMod', {'score': 100})
        """
        from .user import AnonymousUser, MinuUser

        if self._user_cache is not None and self._user_cache.is_authenticated:
            return self._user_cache

        if self.request_data and self._app:
            self._user_cache = await MinuUser.from_request(
                self._app, self.request_data
            )
            # Cache im Request für spätere Zugriffe
            if self.request_data:
                self.request_data._cached_minu_user = self._user_cache

        return self._user_cache or AnonymousUser()

    def set_app(self, app):
        """App-Referenz setzen (wird von Session-Handler aufgerufen)"""
        self._app = app

    # =================== Shared Section Methods ===================

    @property
    def shared_manager(self) -> 'SharedManager':
        """SharedManager Instanz"""
        from .shared import SharedManager

        return SharedManager.get_(self._app)

    async def create_shared(
        self, name: str, initial_data: Dict[str, Any] = None, **kwargs
    ) -> SharedSection:
        """
        Neue Shared Section erstellen.

        Args:
            name: Name der Section
            initial_data: Initiale Daten
            **kwargs: Weitere Optionen (max_participants, allow_anonymous, etc.)

        Returns:
            SharedSection Instanz
        """
        from .shared import SharedManager

        section = await self.shared_manager.create(
            self.request_data, name, initial_data, **kwargs
        )

        self._shared_sections[section.id] = section
        return section

    async def join_shared(self, section_id: str) -> SharedSection | None:
        """
        Shared Section beitreten.

        Args:
            section_id: ID der Section

        Returns:
            SharedSection oder None wenn nicht erlaubt
        """
        section = await self.shared_manager.join(
            section_id, self.request_data, self._session
        )

        if section:
            self._shared_sections[section.id] = section

        return section

    async def leave_shared(self, section_id: str) -> bool:
        """Shared Section verlassen"""
        result = await self.shared_manager.leave(section_id, self.request_data)

        if result and section_id in self._shared_sections:
            del self._shared_sections[section_id]

        return result

    def get_shared(self, section_id: str) -> SharedSection | None:
        """Lokale Shared Section abrufen"""
        return self._shared_sections.get(section_id)

    def render(self) -> Component:
        raise NotImplementedError("Subclass must implement render()")

    def _on_state_change(self, change: StateChange):
        """Called when any bound state changes"""
        self._pending_changes.append(change)

        # Debug logging

        if self._session:
            # Check for structural updates needed
            for dyn in self._dynamic_components:
                # Check if the changed state is in the dyn component's bindings
                # Match by full path OR by state name only
                # change.path could be "view-xxx.input_text" or just "input_text"
                # s._path is always "view-xxx.state_name"
                is_bound = False
                bound_paths = [s._path for s in dyn.bound_states]

                for s in dyn.bound_states:
                    # Extract just the state name from both paths
                    state_name = s._path.split('.')[-1]
                    change_name = change.path.split('.')[-1]

                    if s._path == change.path or state_name == change_name:
                        is_bound = True
                        break

                if is_bound:
                    dyn._update_content()
                    # Schedule a structural replacement
                    self._session._mark_structure_dirty(dyn)

            self._session._mark_dirty(self)

    def register_dynamic(self, dyn: Dynamic):
        """Helper to register dynamic components during render"""
        self._dynamic_components.add(dyn)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize view to dict, setting context for callback registration."""
        # Setze den aktuellen View-Context für Callback-Registrierung
        try:
            from .flows import clear_current_view, set_current_view
            set_current_view(self)
        except ImportError:
            pass

        try:
            rendered = self.render()
            return {
                "viewId": self._view_id,
                "component": rendered.to_dict(),
                "state": {name: state.value for name, state in self._state_attrs.items()},
                "handlers": self._get_handlers(),
            }
        finally:
            # Context aufräumen
            try:
                from .flows import clear_current_view
                clear_current_view()
            except ImportError:
                pass

    def _get_handlers(self) -> List[str]:
        handlers = []
        for name in dir(self):
            if not name.startswith("_") and name not in ("render", "to_dict"):
                attr = getattr(self, name)
                if callable(attr) and not isinstance(attr, ReactiveState):
                    handlers.append(name)
        return handlers

    def get_patches(self) -> List[Dict[str, Any]]:
        patches = []
        for change in self._pending_changes:
            patches.append({
                "type": "state_update",
                "viewId": self._view_id,
                "path": change.path,
                "value": change.new_value,
            })
        self._pending_changes.clear()
        return patches

    def __getattr__(self, name: str):
        """
        Fallback für dynamisch registrierte Callback-Handler.
        Sucht in der lokalen _callback_registry wenn vorhanden.
        """
        # Verhindere Rekursion bei internen Attributen
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        # Prüfe ob wir eine callback_registry haben
        if '_callback_registry' in self.__dict__:
            registry = self.__dict__['_callback_registry']
            if hasattr(registry, 'get'):
                callback = registry.get(name)
                if callback:
                    import asyncio
                    async def async_wrapper(event, cb=callback):
                        result = cb(event)
                        if asyncio.iscoroutine(result):
                            result = await result
                        return result
                    return async_wrapper

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


# ============================================================================
# SESSION & TRANSPORT
# ============================================================================


class MinuSession:
    _views: Dict[str, MinuView]
    _pending_updates: set[MinuView]  # Changed to Set for unique tracking
    _send_callback: Callable[[str], Any] | None
    _pending_replacements: set[Component]

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        self._views = {}
        self._pending_updates = set()
        self._pending_replacements = set()
        self._send_callback = None

    def _mark_structure_dirty(self, component: Component):
        """Mark a component for full structural replacement"""
        self._pending_replacements.add(component)

    def set_send_callback(self, callback: Callable[[str], Any]):
        self._send_callback = callback

    def register_view(self, view: MinuView, app=None) -> str:
        view._session = self
        app = app or get_app(f"minu.register_view.{view._view_id}")
        view.set_app(app)
        self._views[view._view_id] = view
        return view._view_id

    def unregister_view(self, view_id: str):
        if view_id in self._views:
            self._views[view_id]._session = None
            del self._views[view_id]

    def get_view(self, view_id: str) -> MinuView | None:
        return self._views.get(view_id)

    def _mark_dirty(self, view: MinuView):
        """Mark a view as needing updates (Synchronous)"""
        self._pending_updates.add(view)

    async def force_flush(self):
        """
        Immediately send all pending updates.
        Must be awaited at the end of every event handler.
        """
        all_patches = []

        # 1. Handle Structural Replacements - convert to component_update patches
        if self._pending_replacements:
            replacements = list(self._pending_replacements)
            self._pending_replacements.clear()

            for comp in replacements:
                # Find the viewId that owns this component
                owner_view_id = None
                for view_id, view in self._views.items():
                    if comp in view._dynamic_components:
                        owner_view_id = view_id
                        break

                # Add as component_update patch instead of separate message
                all_patches.append({
                    "type": "component_update",
                    "viewId": owner_view_id,
                    "componentId": comp.id,
                    "component": comp.to_dict(),
                })

        # 2. Collect state patches from dirty views
        if self._pending_updates:
            dirty_views = list(self._pending_updates)
            self._pending_updates.clear()

            for view in dirty_views:
                patches = view.get_patches()
                all_patches.extend(patches)

        # 3. Send all patches in one message
        if all_patches and self._send_callback:
            message = {
                "type": "patches",
                "sessionId": self.session_id,
                "patches": all_patches,
            }
            await self._send(json.dumps(message, cls=MinuJSONEncoder))

    async def _send(self, message: str):
        if self._send_callback:
            result = self._send_callback(message)
            if asyncio.iscoroutine(result):
                await result

    async def send_full_render(self, view: MinuView):
        message = {"type": "render", "sessionId": self.session_id, "view": view.to_dict()}
        await self._send(json.dumps(message, cls=MinuJSONEncoder))


    async def handle_event(self, event_data: Dict[str, Any], request = None, app = None):
        """Handle an event from the client with improved callback lookup."""
        view_id = event_data.get("viewId")
        handler_name = event_data.get("handler")
        payload = event_data.get("payload", {})

        view = self._views.get(view_id)
        if not view:
            return {"error": f"View {view_id} not found"}
        if request:
            view.request_data = request
        if app:
            view.set_app(app)
        handler = getattr(view, handler_name, None)

        # 2. Wenn nicht gefunden, prüfe _callback_registry der View
        if handler is None and hasattr(view, '_callback_registry'):
            callback = view._callback_registry.get(handler_name)
            if callback:
                async def handler(event, cb=callback):
                    result = cb(event)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return result

        # 3. Prüfe ob es ein dynamischer Handler ist (via __getattr__)
        if handler is None:
            try:
                handler = getattr(view, handler_name)
            except AttributeError:
                pass

        if not handler or not callable(handler):
            return {"error": f"Handler '{handler_name}' not found on view '{view_id}'"}

        if hasattr(view, 'request_data'):
            view.request_data = request

        try:
            result = handler(payload)
            if asyncio.iscoroutine(result):
                result = await result

            # Wichtig: Updates flushen
            await self.force_flush()

            return {"success": True, "result": result}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


def minu_handler(view_class: type):
    """
    Decorator to create a Minu UI endpoint from a View class.

    Usage:
        @minu_handler
        class MyDashboard(MinuView):
            ...

        # This creates:
        # - WebSocket handler for live updates
        # - API endpoint for initial render
    """

    def create_handler(app, request):
        session = MinuSession()
        view = view_class()
        session.register_view(view)
        return view, session

    return create_handler
# Convenience re-exports
__all__ = [
    # State
    "State",
    "ReactiveState",
    "StateChange",
    # Components
    "Component",
    "ComponentType",
    "ComponentStyle",
    "Card",
    "Text",
    "Heading",
    "Button",
    "Input",
    "Select",
    "Checkbox",
    "Switch",
    "Row",
    "Column",
    "Grid",
    "Spacer",
    "Divider",
    "Alert",
    "Progress",
    "Spinner",
    "Table",
    "List",
    "ListItem",
    "Icon",
    "Image",
    "Badge",
    "Modal",
    "Widget",
    "Form",
    "Tabs",
    "Custom",
    # View System
    "MinuView",
    "MinuSession",
    # Integration
    "minu_handler",
]


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
# Beispiel 1: Einfache View mit User-Zugriff

class UserDashboard(MinuView):
    greeting = State("")

    def render(self):
        return Column(
            Heading("Dashboard"),
            Text(self.greeting.value or f"Hallo, {self.user.name}!"),

            # Zeige verschiedene Inhalte basierend auf Auth-Status
            *self._render_content()
        )

    def _render_content(self):
        if self.user.is_authenticated:
            return [
                Text(f"Level: {self.user.level}"),
                Text(f"Email: {self.user.email}"),
                Button("Abmelden", on_click="logout")
            ]
        else:
            return [
                Text("Du bist nicht angemeldet."),
                Button("Anmelden", on_click="login")
            ]

    async def on_mount(self):
        # User async laden für vollständige Daten
        user = await self.ensure_user()

        # Mod-Daten laden
        if user.is_authenticated:
            data = await user.get_mod_data('Dashboard')
            if data.get('last_visit'):
                self.greeting.value = f"Willkommen zurück, {user.name}!"


# Beispiel 2: Multi-User Chat

class ChatRoom(MinuView):
    messages = State([])
    input_text = State("")

    async def on_mount(self):
        # Shared Section für den Chat-Room
        self.chat = await self.join_shared('chat_room_general')

        if self.chat:
            # Existierende Nachrichten laden
            self.messages.value = self.chat.get('messages', [])

            # Auf neue Nachrichten reagieren
            self.chat.on_change('messages', self._on_new_message)

    def _on_new_message(self, change):
        # Update UI wenn neue Nachrichten ankommen
        if change.operation == 'append':
            current = self.messages.value.copy()
            current.append(change.value)
            self.messages.value = current

    def render(self):
        return Column(
            Heading("Chat Room"),

            # Message List
            List(*[
                ListItem(
                    Text(f"{msg['author']}: {msg['text']}")
                ) for msg in self.messages.value
            ]),

            # Input
            Row(
                Input(
                    placeholder="Nachricht...",
                    bind_value="input_text"
                ),
                Button("Senden", on_click="send_message")
            )
        )

    async def send_message(self, event):
        text = self.input_text.value.strip()
        if not text:
            return

        # Nachricht an alle Teilnehmer senden
        await self.chat.append('messages', {
            'author': self.user.name,
            'author_id': self.user.uid,
            'text': text,
            'timestamp': time.time()
        }, author_id=self.user.uid)

        self.input_text.value = ""


# Beispiel 3: Multiplayer Game

class GameLobby(MinuView):
    players = State([])
    game_state = State("waiting")  # waiting, playing, finished

    async def on_mount(self):
        # Game Session erstellen oder beitreten
        game_id = self.props.get('game_id', 'default_game')

        self.game = await self.join_shared(f'game_{game_id}')

        if not self.game:
            # Neues Spiel erstellen
            self.game = await self.create_shared(
                name=f'game_{game_id}',
                initial_data={
                    'players': [],
                    'state': 'waiting',
                    'scores': {}
                },
                max_participants=4,
                allow_anonymous=True
            )

        # State synchronisieren
        self.players.value = self.game.get('players', [])
        self.game_state.value = self.game.get('state', 'waiting')

        # Auf Änderungen reagieren
        self.game.on_change('players', self._on_players_change)
        self.game.on_change('state', self._on_state_change)

        # Selbst als Spieler hinzufügen
        await self._join_game()

    async def _join_game(self):
        players = self.game.get('players', [])

        # Prüfen ob bereits im Spiel
        if any(p['id'] == self.user.uid for p in players):
            return

        await self.game.append('players', {
            'id': self.user.uid,
            'name': self.user.name,
            'ready': False
        }, author_id=self.user.uid)

    def _on_players_change(self, change):
        self.players.value = self.game.get('players', [])

    def _on_state_change(self, change):
        self.game_state.value = change.value

    def render(self):
        return Column(
            Heading(f"Game Lobby ({self.game_state.value})"),

            # Spielerliste
            Card(
                Heading("Spieler", level=3),
                List(*[
                    ListItem(
                        Row(
                            Text(p['name']),
                            Badge("Bereit" if p.get('ready') else "Wartet",
                                  variant="success" if p.get('ready') else "default")
                        )
                    ) for p in self.players.value
                ])
            ),

            # Aktionen
            Row(
                Button("Bereit", on_click="toggle_ready",
                       variant="primary" if not self._am_ready() else "default"),
                Button("Spiel starten", on_click="start_game",
                       disabled=not self._can_start())
            ) if self.game_state.value == "waiting" else None
        )

    def _am_ready(self) -> bool:
        for p in self.players.value:
            if p['id'] == self.user.uid:
                return p.get('ready', False)
        return False

    def _can_start(self) -> bool:
        if len(self.players.value) < 2:
            return False
        return all(p.get('ready') for p in self.players.value)

    async def toggle_ready(self, event):
        players = self.game.get('players', [])
        for i, p in enumerate(players):
            if p['id'] == self.user.uid:
                players[i]['ready'] = not p.get('ready', False)
                await self.game.set('players', players, author_id=self.user.uid)
                break

    async def start_game(self, event):
        if self._can_start():
            await self.game.set('state', 'playing', author_id=self.user.uid)
"""
