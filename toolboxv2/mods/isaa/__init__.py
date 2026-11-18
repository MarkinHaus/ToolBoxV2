"""isaa-Mod. Der Mod selbst (module.py) wird erst geladen, wenn Tools/tools/version gebraucht werden.

Grund: module.py ruft auf Modulebene get_app("isaa.Export") auf (Export-Muster der Mods). Ein
direkter Import hier legte bei JEDEM Import eines Unterpakets (z. B. extras/icli2, base/...) eine
ToolBox-App an - mit dem Profil, das gerade in der Umgebung steht, und schwer (alle isaa-Abhaengigkeiten).
Der Mod-Lader (utils/toolbox.py: hasattr(mod, "Tools")) und `from toolboxv2.mods.isaa import Tools`
gehen ueber __getattr__ (PEP 562) und laden module.py dann wie bisher.
"""
Name = 'isaa'

_LAZY = ("Tools", "tools", "version")


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from .module import Tools, version
    except ImportError as e:                    # wie bisher: Mod ohne Tools statt Absturz - aber laut im Log
        from toolboxv2 import get_logger
        get_logger().error(f"isaa: module.py nicht ladbar: {e}", exc_info=True)
        raise AttributeError(f"isaa nicht ladbar ({name}): {e}") from e
    globals().update(Tools=Tools, tools=Tools, version=version)
    return globals()[name]
