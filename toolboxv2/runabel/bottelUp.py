NAME = "BOTTUP"


def run(app, app_args, mod="DoNext"):
    from toolboxv2.utils.extras.bottleup import bottle_up

    app.get_mod("DB").edit_cli("RR")
    app.get_mod(mod)
    bottle_up(app, main_route=mod, host=app_args.host, port=app_args.port)
