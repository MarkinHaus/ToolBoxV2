"""icli 2.0 - Starter. Die Logik liegt in toolboxv2/mods/isaa/extras/icli2/.

Start:  tb -m icli2      (alte icli bleibt unveraendert unter 'icli')
"""
NAME = "icli2"


def build(app):
    """Engine (mit IsaaBackend) und UI so zusammensetzen wie im Betrieb; die Engine laeuft danach.
    Eigene Funktion, damit der Betriebstest denselben Aufbau benutzt (tests/test_isaa/icli2_widget_prozess.py)."""
    from toolboxv2.mods.isaa.extras.icli2.app import ICli2
    from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend
    from toolboxv2.mods.isaa.extras.icli2.engine import Engine

    engine = Engine(IsaaBackend(app))
    engine.start()                     # Agent laedt im Engine-Thread, UI ist sofort da
    return engine, ICli2(engine)


async def run(app=None, *args):
    try:
        import textual  # noqa: F401
    except ImportError:
        print("icli2 benoetigt Textual:  pip install 'textual>=8'   (Teil des Extras 'isaa')")
        return

    from toolboxv2 import get_app
    from toolboxv2.mods.isaa.extras.icli2 import protocol as P
    from toolboxv2.mods.isaa.extras.icli2.termguard import quiet_terminal

    app = app or get_app("isaa-host")
    # Log-Handler aufs Terminal schweigen waehrend der TUI; alles landet im ToolBox-Log (get_logger)
    with quiet_terminal():
        engine, ui = build(app)
        try:
            await ui.run_async()
        finally:
            engine.send(P.SHUTDOWN)
            engine.join(15)
    await app.a_exit()
