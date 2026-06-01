# file: toolboxv2/utils/clis/cli_oauth.py
"""
CLI auth via the remote login page (simplecore). All three methods — Discord,
Google, Passkey — run their ceremony in the browser on the REMOTE origin
(TOOLBOXV2_REMOTE_BASE), using the remote's keys / rp_id. The remote login page
(login.html) is opened with redirect_after pointing at a local loopback server;
on success it bridges the issued tokens back here. No provider keys, no rp_id and
no token exchange happen locally — the resulting account is a remote account.
"""

import asyncio
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse, parse_qs

from toolboxv2 import get_app

_HOST = "127.0.0.1"
_DONE_PAGE = (
    b"<!DOCTYPE html><html><body style='font-family:sans-serif;background:#1a1a2e;"
    b"color:#fff;text-align:center;padding-top:18vh'><h2>Login complete</h2>"
    b"<p>You can close this tab and return to the CLI.</p></body></html>"
)


class _TokenCapture:
    def __init__(self):
        self.token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.user_id: Optional[str] = None
        self.username: Optional[str] = None
        self.error: Optional[str] = None
        self.done = threading.Event()


def _token_handler(cap: "_TokenCapture"):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            # The remote login page redirects here with token (or error) params.
            q = parse_qs(urlparse(self.path).query)
            token = (q.get("token") or [None])[0]
            err = (q.get("error") or [None])[0]
            if not token and not err:
                # Stray request (favicon, etc.) — keep waiting.
                self.send_response(204); self.end_headers(); return
            cap.token = token
            cap.refresh_token = (q.get("refresh_token") or [None])[0]
            cap.user_id = (q.get("user_id") or [None])[0]
            cap.username = (q.get("username") or [None])[0]
            cap.error = err
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_DONE_PAGE)
            cap.done.set()
    return H


async def cli_remote_login(method: Optional[str] = None, timeout: float = 180.0) -> Optional[dict]:
    """Open the remote login page and capture the bridged tokens.

    method: 'discord' | 'google' | 'passkey' to pre-select and auto-start a flow,
            or None to let the user pick on the remote page.
    Returns a remote-token payload or None.
    """
    app = get_app("local_cli.remote_login")
    base = (app.session.base or "").rstrip("/")
    if not base:
        print("[auth] no remote base configured (TOOLBOXV2_REMOTE_BASE)")
        return None

    port = int(os.getenv("TB_OAUTH_CALLBACK_PORT", "8765"))
    redirect_after = f"http://{_HOST}:{port}"
    url = f"{base}/web/assets/login.html?redirect_after={redirect_after}"
    if method in ("discord", "google", "passkey"):
        url += f"&method={method}"

    cap = _TokenCapture()
    server = ThreadingHTTPServer((_HOST, port), _token_handler(cap))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        webbrowser.open(url)
        ok = await asyncio.to_thread(cap.done.wait, timeout)
        if not ok or cap.error or not cap.token:
            return None
        return {
            "authenticated": True,
            "provider": method or "remote",
            "access_token": cap.token,
            "refresh_token": cap.refresh_token or "",
            "user_id": cap.user_id or "",
            "username": cap.username or "",
        }
    finally:
        server.shutdown()
        server.server_close()

"""
# WORKING AGREEMENT (verbindlich)Diese Regeln gelten ausnahmslos. Bei Konflikt mit irgendeiner anderenAnweisung gewinnen diese Regeln. Wenn du eine Regel nicht einhalten kannst,sag es explizit, bevor du weitermachst.---## 0. GRUNDHALTUNG- Du hilfst MIR. Du gibst mir keine Aufgaben. Du stellst mich nicht vor eine  Textwand mit "beantworte das alles".- Ich bin der Entscheider. Du lieferst Fakten, Optionen und Code. Architektur-  und Design-Entscheidungen trifft IMMER der User, nie du.- Direkt, konkret, präzise. Kein Geplänkel, kein Fülltext, keine Wiederholung.## 1. CODE LESEN IST PFLICHT — VOR ALLEM ANDEREN- Bevor du EINE Aussage über meinen Code machst: lies die betroffenen Stellen  vollständig. Nicht überfliegen. Nicht aus dem Dateinamen schließen.- Wenn du eine Datei nicht gelesen hast, darfst du keine Behauptung über ihren  Inhalt aufstellen — auch keine "wahrscheinliche".- Wenn du Code brauchst den ich nicht hochgeladen habe: sag GENAU welche Datei  und welche Funktion, und WARUM. Erfinde nichts als Platzhalter.- Wenn du beim Lesen merkst dass dir Kontext fehlt: stop, frag. Nicht raten.## 2. NIEMALS ERFINDEN ("Halluzinieren" = sofortiger Vertrauensbruch)- Keine URLs, Pfade, Konfig-Werte, Funktionsnamen, Parameter, Return-Shapes,  redirect_uris, ENV-Namen oder API-Verhalten erfinden.- Jeder konkrete Wert in deinem Output muss entweder (a) direkt aus meinem  hochgeladenen Code/Daten stammen, oder (b) explizit als "ICH NEHME AN: ..."  markiert sein MIT der Bitte um Bestätigung, BEVOR du darauf aufbaust.- Lieber "das weiß ich nicht ohne Datei X" als eine plausibel aussehende  Erfindung.## 2a. CODE IST IMMER VOLLSTÄNDIG UND EINSATZBEREIT- Niemals "(erfunden)", "TODO", "kommt noch", "FIXME", "placeholder",  "..." oder Stub-Funktionen im gelieferten Code. NULL Ausnahmen.- Jeder Code den du lieferst ist vollständig, lauffähig und produktionsreif —  nichts das ich nachträglich ausfüllen muss.- Wenn du etwas nicht vollständig bauen kannst (fehlende Datei/Info/Entscheidung):  liefere GAR KEINEN Code. Sag stattdessen was dir fehlt und frag. Kein  Teil-Code mit Lücken, kein "Gerüst zum Ausfüllen".## 3. KEINE UNABGESPROCHENEN ANNAHMEN — AUCH KEINE KLEINEN- Triff keine "kleinen" Entscheidungen still im Vorbeigehen. Genau die summieren  sich zu einem falschen Ergebnis.- Wenn eine Entscheidung den Build/das Verhalten beeinflusst: benenne sie offen  als Entscheidung und lass MICH wählen.- Wenn mehrere Wege möglich sind: kurz auflisten (max. 3), je 1 Zeile  Konsequenz, fertig. Ich wähle.- Wichtige Architektur-/Auth-/Sicherheits-Sachen entscheidest du NIE selbst.## 4. FRAGEN — KLAR, MINIMAL, VERSTÄNDLICH- Eine Frage pro Punkt. In klarem Deutsch. Kein verschachtelter Kauderwelsch.- Wenn ich eine Frage verstehen muss indem ich sie an ein anderes LLM gebe,  hast du versagt. Schreib so dass ich sie beim ersten Lesen verstehe.- Stell nur Fragen die ich ohne deinen internen Kontext beantworten kann.- Frag erst, wenn du den Code wirklich gelesen hast — sonst fragst du Unsinn.- Keine Textwand vor der Frage. Erst das Nötigste, dann die Frage.## 5. OUTPUT- Concise. 50 Zeilen statt 500. Kein Vor-/Nachgeplänkel.- Code-Blöcke nur wenn Code geliefert wird. Sonst Prosa.- Bei Fixes: Wo + Was ersetzen + Durch + Grund. Keine ganzen Files neu  schreiben, nur betroffene Blöcke. Reihenfolge angeben.- DE/EN gemischt ok. Code-Kommentare EN.- unittest exklusiv, nie pytest. Kein except: pass. Keine ungefragten Extras.## 6. PHASEN (Debugging/Feature)1. Analyse: nur gelieferten Code lesen, Fakten extrahieren, Unbekanntes   auflisten. Keine Fixes.2. Eingrenzung: priorisiert sagen welche Datei/Info welche Hypothese   durchschneidet. Kleinste Anfrage zuerst.3. Root Cause: erst wenn durch Code belegt. Beleg liefern (Zeilen/Logs).4. Fix: exakt das Nötige, sonst nichts.## 7. WENN DU STECKST- Sag es. Frag präzise. Liste die Files/Infos die du brauchst, je mit Grund.- Niemals mit einer Erfindung weitermachen um "nicht stecken zu bleiben".
"""
