NAME = "BOTTUP"


def run(app, app_args, mod="DoNext", db_mode="RR"):
    from toolboxv2.utils.extras.bottleup import bottle_up

    app.get_mod("DB").edit_cli(db_mode)
    app.get_mod(mod)
    bottle_up(app, main_route=mod, host=app_args.host, port=app_args.port)
