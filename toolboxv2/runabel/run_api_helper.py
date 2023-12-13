NAME = "api"
from toolboxv2 import tbef

def run(_, _1):
    _.run_any(tbef.API_MANAGER.STARTAPI, command=['start-api', _1.name])
    return 0
