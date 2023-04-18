from toolboxv2.util.TBConfig import Config


def next_mode(ai_output: str, current_mode: str, config=Config()):
    if config.get().mode0token in ai_output:
        return 'mode0token'
    if config.get().mode1token in ai_output:
        return 'mode0token'
    if config.get().mode2token in ai_output:
        return 'mode0token'
    return current_mode

