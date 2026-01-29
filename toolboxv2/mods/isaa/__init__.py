#from .kernel import Kernel, __all__ as kernel_all

try:

    from .module import Tools, version

    tools = Tools
    Name = 'isaa'
    version = version

except ImportError:
    import traceback
    print(traceback.format_exc())
    pass

from .hud import Name as isaa_hud_name
