NAME = 'bg'


async def run(_, __):
    _.print("Running...")
    await _.daemon_app.connect(_)
