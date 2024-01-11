from toolboxv2 import tbef
NAME = "api"


def run(_, _0):
    return _.run_any(tbef.API_MANAGER.STARTAPI, api_name=_0.name)
