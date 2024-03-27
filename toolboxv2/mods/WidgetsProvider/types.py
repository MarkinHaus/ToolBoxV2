from typing import Dict, List

# Datenmodell für Widgets
class Widget:
    def __init__(self, name: str, widget_id: str, widget_type: str, data: dict):
        self.name = name
        self.id = widget_id
        self.type = widget_type
        self.data = data

# Datenmodell für Stos
class Sto:
    def __init__(self, name: str, widgets: Dict[str, Widget]):
        self.name = name
        self.widgets = widgets
