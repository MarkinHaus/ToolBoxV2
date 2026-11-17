"""Features und Skills der icli 2.0: SimpleFeatureManager samt den sieben kleinen Ladern.

Wird in IsaaBackend eingemischt. Alle toolboxv2-Importe passieren erst in der
jeweiligen Methode bzw. im Lader (das Paket ist beim Start der UI nicht zwingend geladen).

Vorlage (1:1 uebernommen, nur ohne print_status-Ausgabe, die das TUI stoeren wuerde):
  flows/isaa/icli.py  SimpleFeatureManager, load_desktop_auto_feature, load_web_auto_feature,
                      load_isis_feature, load_full_web_auto_feature, load_coder_toolkit,
                      load_chain_toolkit, load_execute
Skills: SkillsManager (base/Agent/skills.py) ueber agent._get_execution_engine().skills_manager.
Die grossen Lader (docs, autodoc, autotest, autofix) sind bewusst nicht enthalten.
"""
from __future__ import annotations

import asyncio
import sys


class SimpleFeatureManager:
    def __init__(self):
        self.agent_ref = None
        self.features = {}

    def list_features(self):
        return list(self.features.keys())

    async def enable(self, feature):
        if feature in self.features:
            self.features[feature]["is_enabled"] = True
            if self.features[feature]["activation_f"]:
                res = self.features[feature]["activation_f"](self.agent_ref)
                if asyncio.iscoroutine(res):
                    await res

    def is_enabled(self, feature):
        return self.features.get(feature, {"is_enabled": False})["is_enabled"]

    def disable(self, feature):
        if feature in self.features:
            self.features[feature]["is_enabled"] = False
            if self.features[feature]["deactivation_f"]:
                self.features[feature]["deactivation_f"](self.agent_ref)

    def add_feature(self, feature, activation_f=None, deactivation_f=None):
        self.features[feature] = {"is_enabled": False, "activation_f": activation_f,
                                  "deactivation_f": deactivation_f}

    def set_agent(self, agent):
        self.agent_ref = agent


# ============================ Lader (Importe lazy) ============================
def load_desktop_auto_feature(fm):
    def enable(agent):
        from toolboxv2.mods.isaa.extras.destop_auto import register_enhanced_tools
        kit, tools = register_enhanced_tools()
        agent.add_tools(tools)

    def disable(agent):
        from toolboxv2.mods.isaa.extras.destop_auto import register_enhanced_tools
        kit, tools = register_enhanced_tools()
        agent.remove_tools(tools)
    fm.add_feature("desktop_auto", activation_f=enable, deactivation_f=disable)


def load_web_auto_feature(fm):
    from toolboxv2.mods.isaa.extras.web_helper.tooklit import get_minimal_tools
    tools_set = [None]
    toolkit, tools_set[0] = get_minimal_tools(out_of_process=True, headless=True)

    async def enable(agent):
        await toolkit.start_browser()
        agent.add_tools(tools_set[0])

    async def disable(agent):
        await toolkit.stop_browser()
        agent.remove_tools(tools_set[0])
    fm.add_feature("mini_web_auto", activation_f=enable, deactivation_f=disable)


def load_isis_feature(fm):
    from toolboxv2.mods.isaa.extras.toolkit.isis_toolkit import IsisToolkit
    toolkit = IsisToolkit(out_of_process=True, headless=True)
    tools_set = [None]

    async def enable(agent):
        await toolkit.start()
        tools_set[0] = toolkit.get_tools()
        agent.add_tools(tools_set[0])

    async def disable(agent):
        await toolkit.stop()
        agent.remove_tools(tools_set[0])
    fm.add_feature("isis_auto", activation_f=enable, deactivation_f=disable)


def load_full_web_auto_feature(fm):
    from toolboxv2.mods.isaa.extras.web_helper.tooklit import WebAgentToolkit
    toolkit = WebAgentToolkit(out_of_process=True, headless=True)
    tools_set = [None]

    async def enable(agent):
        from toolboxv2 import tb_root_dir
        if sys.platform == "win32":
            asyncio.set_event_loop(asyncio.ProactorEventLoop())
        await toolkit.start_browser()
        tools_set[0] = toolkit.get_tools()
        agent.add_tools(tools_set[0])
        session = await agent.session_manager.get_or_create(agent.active_session)
        session.vfs.add_system_file(str(tb_root_dir.parent / "docs" / "mods" / "isaa" / "web_toolkit.md"),
                                    "web_toolkit.md", open_state=True)

    async def disable(agent):
        await toolkit.stop_browser()
        agent.remove_tools(tools_set[0])
        session = await agent.session_manager.get_or_create(agent.active_session)
        session.vfs.remove_system_file("web_toolkit.md")
    fm.add_feature("full_web_auto", activation_f=enable, deactivation_f=disable)


def load_coder_toolkit(fm):
    from toolboxv2.mods.isaa.CodingAgent.coder_toolset import coder_register_flow_tools
    pool = [None]

    def enable(agent):
        from toolboxv2 import init_cwd
        _pool, tools = coder_register_flow_tools(agent, str(init_cwd))
        pool[0] = _pool
        agent.add_tools(tools)

    def disable(agent):
        from toolboxv2 import init_cwd
        _pool, tools = coder_register_flow_tools(agent, init_cwd)
        agent.remove_tools(tools)
    fm.add_feature("coder", activation_f=enable, deactivation_f=disable)


def load_chain_toolkit(fm):
    from toolboxv2.mods.isaa.base.chain.chain_tools import create_chain_tools
    tools_set = [None]
    agent_registry, coder_registry, format_registry = {}, {}, {}

    def enable(agent):
        tools_set[0] = create_chain_tools(agent, agent_registry=agent_registry,
                                          coder_registry=coder_registry, format_registry=format_registry)
        agent.add_tools(tools_set[0])

    def disable(agent):
        agent.remove_tools(tools_set[0])
    fm.add_feature("chain", activation_f=enable, deactivation_f=disable)


def load_execute(fm):
    from toolboxv2.mods.isaa.base.Agent.executors import register_code_exec_tools
    tools_set = [None]

    def enable(agent):
        tools_set[0] = register_code_exec_tools(agent)

    def disable(agent):
        for t in (tools_set[0] or []):
            agent.remove_tool(t["name"] if isinstance(t, dict) else t)
        tools_set[0] = None
    fm.add_feature("execute", activation_f=enable, deactivation_f=disable)


LOADERS = (load_desktop_auto_feature, load_web_auto_feature, load_full_web_auto_feature,
           load_coder_toolkit, load_chain_toolkit, load_execute, load_isis_feature)


class FeatureMixin:
    def _features_init(self) -> None:
        """Idempotent: Manager anlegen, Lader registrieren (jeder einzeln abgesichert)."""
        if getattr(self, "_fm", None) is not None:
            return
        self._fm = SimpleFeatureManager()
        self._skill_context: dict[str, str] = {}
        for loader in LOADERS:
            try:
                loader(self._fm)
            except Exception as e:  # fehlender Import darf die anderen Lader nicht stoppen
                try:
                    self.app.logger.warning(f"Feature-Lader {loader.__name__} uebersprungen: {e}")
                except Exception:
                    pass

    def _features_active(self) -> list[str]:
        """Namen der aktiven Features (Kopfzeile) - wirft nie."""
        try:
            self._features_init()
            return [f for f in self._fm.list_features() if self._fm.is_enabled(f)]
        except Exception:
            return []

    async def _rows_features(self) -> tuple[str, list[dict]]:
        self._features_init()
        rows = []
        for name in self._fm.list_features():
            on = self._fm.is_enabled(name)
            rows.append({"value": name, "label": ("✓ " if on else "· ") + name, "hint": "an" if on else "aus"})
        return "Features", rows

    async def _do_feature(self, name: str) -> dict:
        self._features_init()
        if name not in self._fm.features:
            raise ValueError(f"Unbekanntes Feature: {name}")
        self._fm.set_agent(await self._agent())
        if self._fm.is_enabled(name):
            # fm.disable() wuerde async-Deaktivierer nicht abwarten -> hier selbst behandeln
            feat = self._fm.features[name]
            feat["is_enabled"] = False
            if feat["deactivation_f"]:
                res = feat["deactivation_f"](self._fm.agent_ref)
                if asyncio.iscoroutine(res):
                    await res
            state = "aus"
        else:
            try:
                await self._fm.enable(name)
            except Exception:
                self._fm.features[name]["is_enabled"] = False
                raise
            state = "an"
        return {"ready": await self._ready(), "text": f"Feature {name} {state}"}

    # ---------------------------------------------------------------- Skills
    async def _skills_manager(self):
        return (await self._agent())._get_execution_engine().skills_manager

    async def _rows_skills(self) -> tuple[str, list[dict]]:
        self._features_init()
        sm = await self._skills_manager()
        rows = []
        for sid, sk in sm.skills.items():
            on = sid in self._skill_context
            trig = ", ".join(getattr(sk, "triggers", None) or [])
            desc = (getattr(sk, "description", "") or "").strip().splitlines()
            hint = (trig or (desc[0] if desc else ""))[:60]
            rows.append({"value": sid, "label": ("✓ " if on else "· ") + (getattr(sk, "name", None) or sid),
                         "hint": hint})
        return f"Skills: {self.agent}", rows

    async def _do_skill(self, name: str) -> dict:
        self._features_init()
        sm = await self._skills_manager()
        sk = sm.skills.get(name)
        if sk is None:
            raise ValueError(f"Skill nicht gefunden: {name}")
        label = getattr(sk, "name", None) or name
        if name in self._skill_context:
            self._skill_context.pop(name, None)
            try:
                sk.confidence = 0.0
            except Exception:
                pass
            return {"text": f"Skill {label} aus"}
        self._skill_context[name] = getattr(sk, "instruction", "") or ""
        try:
            sk.confidence = 0.95
        except Exception:
            pass
        return {"text": f"Skill {label} an - Agent hat den vollen Kontext"}

    def _skill_wrap(self, query: str) -> str:
        """Aktive Skills der Anfrage voranstellen (Format wie icli.py ~14505) - wirft nie."""
        try:
            ctx = getattr(self, "_skill_context", None)
            if not ctx:
                return query
            blocks = "\n\n".join(f"[ACTIVE SKILL: {sid}]\n{instr}" for sid, instr in ctx.items())
            return f"{blocks}\n\n---\nUser request:\n{query}"
        except Exception:
            return query
