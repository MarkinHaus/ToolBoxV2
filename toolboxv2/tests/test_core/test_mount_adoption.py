"""R9 B4b: mount_app adoptiert konkurrierende Routen derselben Modul-App.

Kaltpfad-Race: _auto_mount_features mounted isaa beim Owner-Init, bevor
restore/start_app source-app:isaa registrieren konnte -> zweiter mount_app
desselben Moduls sieht eigene Routen als Konflikte und lieferte
merged:false + conflicts, statt {"already": true}.

Fix: sind ALLE konfliktierenden Routen identisch (gleiches Handler-Objekt
oder gleiches Handler-Modul) mit den Routen der zu mountenden App,
uebernimmt mount_app sie in die eigene Source-Registry (Adoption)
und meldet Erfolg.

Run: python -m unittest toolboxv2.tests.test_core.test_mount_adoption -v
"""
import unittest
from toolboxv2.utils.workers.fast_tb import FastTB, Route


def _handler():
    def h(request):
        return {}
    return h


def _add(app, path, module="toolboxv2.mods.isaa.ui.app"):
    h = _handler()
    h.__module__ = module
    r = Route(path=path, method="GET", handler=h, pattern=None,
              param_names=[], name=path.strip("/") or "root", auth=False)
    app._routes.append(r)
    return r


class TestMountAdoption(unittest.TestCase):
    def test_same_module_conflicts_are_adopted(self):
        owner = FastTB()
        # Auto-Mount hat isaa bereits eingeh?ngt (ohne saubere Source-Registry
        # im Kaltpfad simuliert: Routen da, source "app:isaa" fehlt)
        _add(owner, "/isaa")
        _add(owner, "/api/chats")
        _add(owner, "/api/vfs/list")
        other = FastTB()
        _add(other, "/isaa")
        _add(other, "/api/chats")
        _add(other, "/api/vfs/list")
        conflicts = owner.mount_app(other, "isaa", "app:isaa")
        self.assertEqual(conflicts, [], str(conflicts))
        self.assertIn("app:isaa", owner._mounted_sources)

    def test_foreign_conflicts_still_reported(self):
        owner = FastTB()
        _add(owner, "/api/chats", module="toolboxv2.mods.other.app")
        other = FastTB()
        _add(other, "/isaa")
        _add(other, "/api/chats")
        conflicts = owner.mount_app(other, "isaa", "app:isaa")
        self.assertNotEqual(conflicts, [])
        self.assertNotIn("app:isaa", owner._mounted_sources)

    def test_partial_overlap_adopts_only_own(self):
        # eigene Routen identisch, Fremdroute /own bleibt unberuehrt ->
        # /api/chats gehoert anderem Modul -> echter Konflikt
        owner = FastTB()
        _add(owner, "/api/chats", module="toolboxv2.mods.other.app")
        _add(owner, "/isaa", module="toolboxv2.mods.isaa.ui.app")
        other = FastTB()
        _add(other, "/isaa")
        _add(other, "/api/chats")
        conflicts = owner.mount_app(other, "isaa", "app:isaa")
        self.assertNotEqual(conflicts, [])


if __name__ == "__main__":
    unittest.main()
