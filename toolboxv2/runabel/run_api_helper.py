NAME = "api"


def run(_, _1):
    _.run_any("api_manager", "start-api", "main")
    return 0
