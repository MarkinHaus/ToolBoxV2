"""Discord- und Telegram-Anbindung der icli 2.0 ueber die unveraenderten Erweiterungen der alten icli.

Die Erweiterungen erwarten einen ISAA_Host. HostShim bietet genau das, was sie davon benutzen:
  isaa_tools, active_agent_name, _handle_command, _build_completer, run_agent_monitored.
Alle toolboxv2-Importe passieren erst in der jeweiligen Methode (das Paket ist beim Start nicht
zwingend geladen). Die Erweiterungen drucken per print(); die Ausgabe wird per redirect_stdout
abgefangen und als Notice zurueckgegeben.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import re

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_END = object()                                        # Ende-Marke fuer den Spiegel-Iterator
TG_COMMANDS = {                                        # wie das Completer-Dict der alten icli
    "setup": "Anleitung: Bot anlegen, Token und eigene ID", "connect": "Bot starten [token]",
    "disconnect": "Bot stoppen", "status": "Verbindungsstatus", "chats": "Bekannte Chats",
    "send": "Nachricht senden <chat_id> <text>", "search": "Kontakte suchen <begriff>",
    "active": "Aktive Unterhaltungen", "admin": "Admins verwalten [add|remove] <id>",
}
DC_HINTS = {
    "connect": "Bot starten [token]", "disconnect": "Bot stoppen", "status": "Verbindungsstatus",
    "channels": "Aktive Kanaele", "active": "Aktive Unterhaltungen", "send": "Nachricht senden <adresse> <text>",
    "search": "Adressbuch durchsuchen <begriff>",
}


def _clean(text: str) -> str:
    return ANSI.sub("", text or "").strip()


class HostShim:
    """Minimaler Ersatz fuer ISAA_Host - nur was Discord/Telegram wirklich benutzen."""

    def __init__(self, backend):
        self.backend = backend
        self.discord_ext = None                        # setzt patch_cli_for_discord
        self.telegram_ext = None
        self._mirrors: set[asyncio.Task] = set()

    @property
    def isaa_tools(self):
        return self.backend.isaa

    @property
    def active_agent_name(self) -> str:
        return self.backend.agent

    @active_agent_name.setter
    def active_agent_name(self, name: str) -> None:
        self.backend.agent = name

    async def _handle_command(self, user_input: str) -> None:
        """Basis-Handler: tut nichts (patch_cli_for_discord umhuellt ihn)."""

    def _build_completer(self):
        return {}, None

    async def run_agent_monitored(
        self, agent_name: str, query: str, kind: str = "native", session_id: str | None = None,
        take_focus: bool = False, should_speak: bool = False,
        system_prompt_override: str | None = None, user_lightning_model: bool = False,
    ):
        """Wie das Original: Chunks an den Aufrufer, zusaetzlich gespiegelt in die UI.
        Beachtet: agent_name, query, kind, session_id, system_prompt_override, user_lightning_model.
        Ignoriert (nur altes Terminal): take_focus, should_speak."""
        b = self.backend
        session = session_id or b.session
        try:
            agent = await b._icli2_agent(agent_name)          # mit icli2-Tools (Widgets)
        except Exception as e:
            yield {"type": "error", "message": f"agent '{agent_name}': {e}"}
            return
        text = system_prompt_override.replace("{query}", query) if system_prompt_override else query
        queue: asyncio.Queue = asyncio.Queue()
        state = {"alive": True}

        async def mirror():                            # Iterator fuer _spawn: liest bis zur Ende-Marke
            while True:
                item = await queue.get()
                if item is _END:
                    return
                yield item

        async def feed_ui():
            try:
                await b._spawn(agent_name, session, query, kind=kind, stream=mirror())
            except Exception as e:                     # Spiegel darf den echten Stream nie stoeren
                state["alive"] = False
                b.app.logger.warning(f"icli2 bridge mirror: {e}")

        task = asyncio.ensure_future(feed_ui())
        self._mirrors.add(task)
        task.add_done_callback(self._mirrors.discard)
        try:
            async for chunk in agent.a_stream(query=text, session_id=session,
                                              max_iterations=b.max_iterations,
                                              user_lightning_model=user_lightning_model):
                if state["alive"]:
                    queue.put_nowait(chunk)
                yield chunk
        except Exception as e:
            if state["alive"]:
                queue.put_nowait({"type": "error", "message": str(e)})
            raise
        finally:
            queue.put_nowait(_END)                     # auch bei Fehler / fruehem Abbruch des Aufrufers


class BridgeMixin:
    async def _bridges_boot(self) -> None:
        """Shim anlegen, Discord- und Telegram-Erweiterung anhaengen. Wirft nie, idempotent."""
        try:
            if getattr(self, "_shim", None) is None:
                self._shim = HostShim(self)
                self._bridge_err = {}
            shim, err = self._shim, self._bridge_err
            if shim.discord_ext is None:
                try:
                    from toolboxv2.mods.isaa.extras.discord_interface.integration_example import patch_cli_for_discord
                    with contextlib.redirect_stdout(io.StringIO()):   # Erfolgsmeldung stoert das TUI
                        patch_cli_for_discord(shim)
                    err.pop("discord", None)
                except Exception as e:
                    err["discord"] = f"{type(e).__name__}: {e}"
                    self.app.logger.warning(f"icli2 discord: {err['discord']}")
            if shim.telegram_ext is None:
                try:
                    from toolboxv2.mods.isaa.extras.telegram_interface.telegram_cli_extension import TelegramCliExtension
                    shim.telegram_ext = TelegramCliExtension(shim)
                    err.pop("telegram", None)
                except Exception as e:
                    err["telegram"] = f"{type(e).__name__}: {e}"
                    self.app.logger.warning(f"icli2 telegram: {err['telegram']}")
        except Exception as e:
            self.app.logger.warning(f"icli2 bridges: {e}")

    async def _bridge_ext(self, which: str):
        await self._bridges_boot()
        shim = self._shim
        ext = shim.discord_ext if which == "discord" else shim.telegram_ext
        if ext is None:
            raise RuntimeError(f"{which.capitalize()}-Integration nicht verfügbar: "
                               f"{getattr(self, '_bridge_err', {}).get(which, 'unbekannt')}")
        return shim, ext

    async def _rows_discord(self) -> tuple[str, list[dict]]:
        await self._bridge_ext("discord")
        from toolboxv2.mods.isaa.extras.discord_interface.discord_cli_extension import get_completer_dict
        rows = []
        for name, sub in get_completer_dict().items():
            hint = DC_HINTS.get(name) or ("Unterbefehle: " + ", ".join(sub) if sub else "")
            rows.append({"value": name, "label": name, "hint": hint})
        return "Discord", rows

    async def _do_discord(self, args: str = "") -> dict:
        shim, _ = await self._bridge_ext("discord")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            await shim._handle_command(f"/discord {args}".strip())
        return {"text": _clean(buf.getvalue()) or "ok"}

    async def _rows_telegram(self) -> tuple[str, list[dict]]:
        await self._bridge_ext("telegram")
        return "Telegram", [{"value": k, "label": k, "hint": v} for k, v in TG_COMMANDS.items()]

    async def _do_telegram(self, args: str = "") -> dict:
        _, ext = await self._bridge_ext("telegram")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            result = await ext.handle_command(args)
        text = _clean("\n".join(x for x in (buf.getvalue(), str(result or "")) if x.strip()))
        return {"text": text or "ok"}

    async def _bridges_close(self) -> None:
        """Verbindungen beenden, soweit vorhanden. Wirft nie."""
        shim = getattr(self, "_shim", None)
        if shim is None:
            return
        try:
            dc = shim.discord_ext
            if dc is not None and getattr(dc, "_connected", False):
                with contextlib.redirect_stdout(io.StringIO()):
                    await dc._disconnect()
        except Exception as e:
            self.app.logger.warning(f"icli2 discord close: {e}")
        try:
            tg = shim.telegram_ext
            if tg is not None and tg.is_connected:
                await tg.cmd_disconnect()
        except Exception as e:
            self.app.logger.warning(f"icli2 telegram close: {e}")
        for task in list(shim._mirrors):
            task.cancel()
