NAME = 'bg'


async def run(app, api=True):
    app.print("Running...")
    from toolboxv2.utils.clis.api import  manage_server, handle_debug
    if api:
        if app.debug:
            handle_debug()
        else:
            manage_server("start")
    await app.daemon_app.connect(app)
