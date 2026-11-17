"""Pruefung von models.py mit Fake-Routern (ohne installiertes toolboxv2/litellm).
Aufruf: cd tests && python check_models.py"""
import os, sys, types
import stub

_before = set(sys.modules)
from toolboxv2.mods.isaa.extras.icli2 import models
_new = set(sys.modules) - _before

# 6. Import darf weder litellm noch fremde toolboxv2-Module laden
assert not any(m == "litellm" or m.startswith("litellm.") for m in _new), sorted(_new)
bad = [m for m in _new if m.split(".")[0] == "toolboxv2" and not m.startswith("toolboxv2.mods.isaa.extras.icli2")]
assert not bad, bad


def mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m


class FakeGW:
    """Wie GatewayProvider: get_available_models() mit Cache-Attribut _available_models."""
    def __init__(s, names): s.models = list(names); s._available_models = None
    def get_available_models(s):
        if s._available_models is None:
            s._available_models = list(s.models)
        return s._available_models


class FakeHandler:
    def __init__(s, names): s.models = list(names)
    def get_available_models(s): return list(s.models)


class Boom:
    def get_available_models(s): raise RuntimeError("kaputt")


# Fakes unter den echten Modulnamen, die models.py lazy importiert
mod("toolboxv2.flows"); mod("toolboxv2.flows.isaa", __file__=stub.R + "/flows/isaa/__init__.py")
gw = FakeGW(["m1"])
mod("toolboxv2.mods.isaa.base.IntelligentRateLimiter", gateway=gw)
litellm = mod("litellm", custom_provider_map=[])
os.environ["TB_LLM_GATEWAY_URL"] = "http://fake-gateway/v1"

# 1. statische Modelle aus der echten icli.py
sm = models.static_models()
assert len(sm) > 20, len(sm)
assert all(isinstance(k, str) and isinstance(v, str) for k, v in sm.items())
assert sm["gpt-5.2"] == "openrouter/openai/gpt-5.2", sm.get("gpt-5.2")

# 2. Router-Modelle mit vollem Namen und Routername
r = models.router_models()
assert ("gateway/m1", "m1", "gateway") in r, r

# 3. dynamisch: neue Modelle und neuer Router ohne Neuladen sichtbar
gw.models.append("m2")
assert ("gateway/m2", "m2", "gateway") in models.router_models()
handler = FakeHandler(["x1"])
litellm.custom_provider_map.append({"provider": "fresh", "custom_handler": handler})
assert ("fresh/x1", "x1", "fresh") in models.router_models()
handler.models.append("x2")
assert ("fresh/x2", "x2", "fresh") in models.router_models()

# 4. ein fehlerhafter Router bricht weder router_models noch all_models
litellm.custom_provider_map.append({"provider": "boom", "custom_handler": Boom()})
r = models.router_models()
assert ("gateway/m1", "m1", "gateway") in r and ("fresh/x1", "x1", "fresh") in r, r
rows = models.all_models()
values = {x["value"] for x in rows}
assert "gateway/m1" in values and "fresh/x2" in values, values
assert any(x["label"] == "gpt-5.2" for x in rows)
orig = gw.get_available_models
def _raise(): raise RuntimeError("gw kaputt")
gw.get_available_models = _raise
r = models.router_models()
assert ("fresh/x1", "x1", "fresh") in r and not any(f.startswith("gateway/") for f, _, _ in r), r
gw.get_available_models = orig
litellm.custom_provider_map.pop()  # boom entfernen

# 5. all_models: Reihenfolge, Dopplungen, Zeilenform
rows = models.all_models([("openai/x", "aktuell schnell"), ("openai/x", "aktuell komplex")])
assert rows[0] == {"value": "openai/x", "label": "openai/x", "hint": "aktuell schnell"}, rows[0]
assert rows[1] == {"value": "openai/x", "label": "openai/x", "hint": "aktuell komplex"}, rows[1]
for row in rows:
    assert set(row) == {"value", "label", "hint"}, row
    assert all(isinstance(row[k], str) for k in row), row
values = [x["value"] for x in rows[2:]]
assert len(values) == len(set(values)), "doppelte value nach den ersten zwei Zeilen"
assert "openai/x" not in values
ridx = [i for i, x in enumerate(rows) if x["hint"].startswith("Router: ")]
sidx = [i for i, x in enumerate(rows) if i >= 2 and not x["hint"].startswith("Router: ")]
assert ridx and sidx and max(ridx) < min(sidx), (ridx[:3], sidx[:3])
assert {rows[i]["hint"] for i in ridx} == {"Router: gateway", "Router: fresh"}, {rows[i]["hint"] for i in ridx}

print("models ok")
