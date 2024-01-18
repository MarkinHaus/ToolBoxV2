from toolboxv2 import tbef

NAME = "api"


def run(_, _0):
    if _0.host or _0.port:
        _.run_any(tbef.API_MANAGER.EDITAPI, api_name=_0.name, host=_0.host if _0.host else "localhost",
                  port=_0.port if _0.port else 5000)
    return _.run_any(tbef.API_MANAGER.STARTAPI, api_name=_0.name, live=_0.remote, reload=_.debug)
