"""CloudM-Anbindung der icli 2.0: Login, ToolBox-Funktionen, LiveSync-Shares.

Wird in IsaaBackend eingemischt. Alle toolboxv2/CloudM-Importe passieren erst in
der jeweiligen Methode (das Paket ist beim Start der UI nicht zwingend geladen).

Genutzt werden ausschliesslich vorhandene Funktionen:
  app.session.valid / .username            (utils/system/session.py)
  CloudM.LogInSystem._find_cli_session     (gespeicherte CLI-Sitzung)
  CloudM.LogInSystem.cli_logout            (ohne Eingabe, nur Ausgabe)
  app.functions / app.a_run_any            (utils/toolbox.py)
  CloudM.LiveSync.vfs_adapter.VFSSyncManager.connect/disconnect/status
Gespeicherte Shares: <appdata>/icli/vfs_shares.json (gleiche Datei wie die alte icli).
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import shlex
from pathlib import Path

from . import fmt

LOGIN_HELP = ("Anmeldung braucht Tastatureingaben (Menue/E-Mail/Code) und geht nur im "
              "normalen Terminal: dort  tb login  ausfuehren, danach hier erneut /login.")
SHARE_MARK = {"live": "verbunden", "connecting": "verbindet", "auth_failed": "abgelehnt"}


class CloudMMixin:
    # ---------------------------------------------------------------- Login
    async def _cloud_user(self) -> str:
        """Name des angemeldeten Benutzers oder "" - wirft nie."""
        try:
            sess = getattr(self.app, "session", None)
            if sess is not None and getattr(sess, "valid", False) and getattr(sess, "username", None):
                return str(sess.username)
        except Exception:
            pass
        try:
            from toolboxv2.mods.CloudM.LogInSystem import _find_cli_session
            found = await asyncio.to_thread(_find_cli_session, self.app)
            return str((found or {}).get("username") or "")
        except Exception:
            return ""

    async def _do_login(self) -> dict:
        name = await self._cloud_user()
        if name:
            return {"text": f"Angemeldet als {name}"}
        # cli_login/login_screen fragen per input()/Menue ab -> im TUI nicht aufrufbar
        return {"text": LOGIN_HELP, "level": "warn"}

    async def _do_logout(self) -> dict:
        name = await self._cloud_user()
        if not name:
            return {"text": "Nicht angemeldet", "level": "warn"}
        from toolboxv2.mods.CloudM.LogInSystem import cli_logout
        with contextlib.redirect_stdout(io.StringIO()):  # Box-Ausgabe soll das TUI nicht stoeren
            await cli_logout(self.app)
        return {"text": f"Abgemeldet: {name}"}

    # ------------------------------------------------------------- ToolBox
    async def _rows_tb(self) -> tuple[str, list[dict]]:
        rows = []
        for mod, funcs in sorted((getattr(self.app, "functions", None) or {}).items()):
            if not isinstance(funcs, dict):
                continue
            for fn, meta in sorted(funcs.items()):
                if not isinstance(meta, dict) or not callable(meta.get("func")):
                    continue
                params = [p for p in (meta.get("params") or []) if p not in ("self", "app", "state")]
                hint = (meta.get("helper") or "").strip().split("\n")[0]
                if params:
                    hint = (hint + "  " if hint else "") + "(" + ", ".join(params) + ")"
                rows.append({"value": f"{mod}.{fn}", "label": f"{mod}.{fn}", "hint": hint})
        return "ToolBox-Funktionen", rows

    @staticmethod
    def _parse_tb_args(args: str) -> dict:
        out = {}
        for tok in shlex.split(args or ""):
            key, sep, val = tok.partition("=")
            if not sep or not key:
                raise ValueError(f"Argument '{tok}' hat nicht die Form key=wert")
            try:
                out[key] = json.loads(val)
            except ValueError:
                out[key] = val
        return out

    async def _do_tb(self, target: str, args: str = "") -> dict:
        mod, dot, fn = (target or "").strip().rpartition(".")
        if not dot or not mod or not fn:
            raise ValueError("Ziel muss 'modul.funktion' sein")
        kwargs = self._parse_tb_args(args)
        res = await self.app.a_run_any((mod, fn), get_results=True, **kwargs)
        if hasattr(res, "is_error") and res.is_error():
            info = getattr(res, "info", None)
            raise RuntimeError(f"{target} fehlgeschlagen: {getattr(info, 'help_text', None) or info or 'Fehler'}")
        data = res.get() if hasattr(res, "get") else res
        return {"text": fmt.detail(data)}

    # -------------------------------------------------------------- Shares
    def _shares_file(self) -> Path:
        return Path(self.app.appdata) / "icli" / "vfs_shares.json"

    def _saved_shares(self) -> list:
        try:
            data = json.loads(self._shares_file().read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _share_mgr(self, session):
        mgr = getattr(session, "_vfs_sync_mgr", None)
        if mgr is None:
            from toolboxv2.mods.CloudM.LiveSync.vfs_adapter import VFSSyncManager
            mgr = VFSSyncManager(session.vfs)
            session._vfs_sync_mgr = mgr
        return mgr

    async def _rows_shares(self) -> tuple[str, list[dict]]:
        session = await self._session()
        mgr = getattr(session, "_vfs_sync_mgr", None)
        states = mgr.status() if mgr is not None else {}
        rows = []
        for sh in self._saved_shares():
            vp = sh.get("vfs_path", "")
            st = states.get(vp, {}).get("status", "stopped")
            hint = SHARE_MARK.get(st, "getrennt") + (" · nur lesen" if sh.get("readonly") else "")
            rows.append({"value": vp, "label": f"{vp} ← {sh.get('local_dir', '')}", "hint": hint})
        return "LiveSync-Shares", rows

    async def _do_share(self, share: str) -> dict:
        sh = next((x for x in self._saved_shares() if x.get("vfs_path") == share), None)
        if sh is None:
            raise RuntimeError(f"Share '{share}' ist nicht gespeichert (anlegen mit /vfs share connect in der alten CLI)")
        session = await self._session()
        mgr = self._share_mgr(session)
        state = mgr.status().get(share, {}).get("status")
        if state in ("live", "connecting"):
            await mgr.disconnect(share)
            return {"ready": await self._ready(), "text": f"Share getrennt: {share}"}
        if state is not None:                            # stopped / auth_failed: alten Adapter abraeumen
            await mgr.disconnect(share)
        await mgr.connect(share, sh["local_dir"], token=sh["token"], readonly=sh.get("readonly", False))
        return {"ready": await self._ready(), "text": f"Share verbindet: {share} ← {sh['local_dir']}"}
