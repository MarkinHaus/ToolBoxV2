"""icli 2.0 - Starter. Die Logik liegt in toolboxv2/mods/isaa/extras/icli2/.

Start:  tb -m icli2      (alte icli bleibt unveraendert unter 'icli')
"""
NAME = "icli2"


async def run(app=None, *args):
    try:
        import textual  # noqa: F401
    except ImportError:
        print("icli2 benoetigt Textual:  pip install 'textual>=8'   (Teil des Extras 'isaa')")
        return

    from toolboxv2 import get_app
    from toolboxv2.mods.isaa.extras.icli2 import protocol as P
    from toolboxv2.mods.isaa.extras.icli2.app import ICli2
    from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend
    from toolboxv2.mods.isaa.extras.icli2.engine import Engine

    from pathlib import Path
    from toolboxv2.mods.isaa.extras.icli2.termguard import log_to_file

    app = app or get_app("isaa-host")
    # Logging, das direkt ins Terminal schreiben wuerde, landet waehrend der TUI in einer Datei
    with log_to_file(Path(app.appdata) / "icli2" / "icli2.log"):
        engine = Engine(IsaaBackend(app))
        engine.start()                 # Agent laedt im Engine-Thread, UI ist sofort da
        try:
            await ICli2(engine).run_async()
        finally:
            engine.send(P.SHUTDOWN)
            engine.join(15)
    await app.a_exit()
