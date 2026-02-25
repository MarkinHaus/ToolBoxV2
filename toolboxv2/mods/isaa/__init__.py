try:

    from .module import Tools, version

    tools = Tools
    Name = 'isaa'
    version = version

except ImportError:
    import traceback
    print(traceback.format_exc())
    pass

