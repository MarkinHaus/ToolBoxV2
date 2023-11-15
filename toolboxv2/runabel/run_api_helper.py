NAME = "api"


def run(_, _1):
    _.run_any('api_manager', 'start-api', command=['start-api', _1.name])
    return 0
