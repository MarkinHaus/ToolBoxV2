"""icli 2.0 - UI und Ausfuehrung strikt getrennt.

Schichten (Importe nur von oben nach unten):

    app.py       Textual-UI. Kennt nur protocol + Engine-Fassade.
    engine.py    Eigener Thread + eigener asyncio-Loop. Kennt kein Textual.
    backend.py   Adapter auf den isaa-Mod (einzige Stelle mit isaa-Importen).
    protocol.py  Reine Daten: Command (UI -> Engine), Event (Engine -> UI).

Die UI wird nie von der Engine aufgerufen: sie holt Events pro Frame aus
einer Queue (Pull statt Push) und kann deshalb nicht blockiert oder
ueberflutet werden.
"""

VERSION = "2.0.0a1"
