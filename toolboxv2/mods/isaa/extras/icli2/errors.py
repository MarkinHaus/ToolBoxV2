"""Fehler eines Laufs verstaendlich machen. Reine Funktionen, nur Standardbibliothek.

explain(roh, ...) liefert:
  kind      Art (rate_limit, auth, credits, model, context, connection, overloaded, timeout,
            policy, bad_request, empty, crash, limit, stream_closed, unknown)
  title     kurze Ueberschrift
  hint      was der Nutzer jetzt tun kann (konkret, mit Befehlen der icli2)
  provider  Anbieter, wenn erkennbar (z. B. "Openrouter")
  message   die eigentliche Meldung des Anbieters (aus eingebettetem JSON) bzw. erste Zeile
  retry     empfohlene Wartezeit, wenn der Anbieter eine nennt (z. B. "8 s")
  detail    vollstaendiger Rohtext, JSON lesbar eingerueckt, ggf. mit Traceback
"""
from __future__ import annotations

import json
import re

DETAIL_LIMIT = 8000

# (Art, Muster) - Reihenfolge zaehlt: spezifisch vor allgemein
_RULES: list[tuple[str, str]] = [
    ("limit", r"max_iterations|iterationslimit|max(imum)? iterations"),
    ("context", r"ContextWindowExceeded|context (length|window)|maximum context|too many tokens|"
                r"prompt is too long|reduce the length|token limit"),
    ("credits", r"\b402\b|insufficient[_ ](credits|quota|funds|balance)|requires more credits|"
                r"payment required|billing|out of credits|credit balance"),
    ("rate_limit", r"RateLimit|\b429\b|rate[_ -]?limit|too many requests|RESOURCE_EXHAUSTED|quota exceeded"),
    ("auth", r"AuthenticationError|PermissionDenied|\b401\b|\b403\b|invalid[_ ]api[_ ]key|"
             r"incorrect api key|api key not valid|unauthori[sz]ed|no api key|missing api key"),
    ("model", r"NotFoundError|model[_ ]not[_ ]found|does not exist|no endpoints found|"
              r"unknown model|invalid model|\b404\b"),
    ("policy", r"ContentPolicy|content[_ ]policy|safety|blocked by|flagged"),
    ("timeout", r"\bTimeout\b|APITimeoutError|timed out|ReadTimeout|deadline exceeded"),
    ("overloaded", r"ServiceUnavailable|\b503\b|\b502\b|\b529\b|overloaded|InternalServerError|\b500\b"),
    ("stream_closed", r"Event loop is closed|Stream aborted"),
    ("connection", r"APIConnectionError|Connection (error|refused|reset)|ConnectError|"
                   r"Name or service not known|getaddrinfo|network is unreachable|SSL"),
    ("empty", r"leere Response|empty (llm )?response"),
    ("crash", r"Agent crashed|INTERNAL ERROR|Traceback \(most recent call last\)"),
    ("bad_request", r"BadRequestError|\b400\b|invalid_request|validation failed"),
]

_TITLES = {
    "rate_limit": "Anbieter-Limit erreicht",
    "auth": "Zugang abgelehnt",
    "credits": "Guthaben oder Budget aufgebraucht",
    "model": "Modell nicht verfügbar",
    "context": "Kontext zu groß für das Modell",
    "connection": "Anbieter nicht erreichbar",
    "overloaded": "Anbieter überlastet oder gestört",
    "timeout": "Zeitüberschreitung beim Anbieter",
    "policy": "Vom Anbieter blockiert",
    "bad_request": "Anfrage vom Anbieter abgelehnt",
    "empty": "Modell antwortete leer",
    "crash": "Interner Fehler im Agenten",
    "limit": "Iterationslimit erreicht",
    "stream_closed": "Verbindung zum Modell abgebrochen",
    "unknown": "Lauf mit Fehler beendet",
}

_GENERIC = {"API", "Litellm", "Http", "Base", "Value", "Runtime", "Type", "Key", "Attribute", "Assertion",
            "Content", "Context", "Index", "OS", "IO", "Timeout", "Connection"}


def _provider(text: str) -> str:
    for name in re.findall(r"\b([A-Z][A-Za-z]+)Exception\b", text):
        if name not in _GENERIC:
            return name
    m = re.search(r"\b(openrouter|groq|gemini|anthropic|openai|ollama|mistral|deepseek|gateway)\b", text, re.I)
    return m.group(1).capitalize() if m else ""


def _json_part(text: str):
    """Erstes gueltiges JSON-Objekt im Text (Anbieter betten ihre Antwort oft so ein)."""
    for start in [m.start() for m in re.finditer(r"\{", text)][:8]:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc, in_str = (not esc and ch == "\\"), (in_str if esc or ch != '"' else False)
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return start, i + 1, json.loads(text[start:i + 1])
                    except ValueError:
                        break
    return None


def _message(text: str, data) -> str:
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            for key in ("message", "detail", "error"):
                if isinstance(err.get(key), str) and err[key].strip():
                    return err[key].strip()
        elif isinstance(err, str) and err.strip():
            return err.strip()
    m = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        return m.group(1).encode().decode("unicode_escape", errors="ignore")
    # "Agent crashed: ProviderError: litellm.XError: XError: AnbieterException - eigentliche Meldung"
    line = text.strip().splitlines()[0] if text.strip() else ""
    return _strip_wrappers(line)


_OUTER = [                                               # immer nur Huelle
    r"^(Agent crashed|Es ist ein Fehler aufgetreten|Stream aborted|LLM call failed)\s*:\s*",
    r"^\W*INTERNAL ERROR\s*:\s*",
    r"^[A-Za-z]+Exception\s*-\s*",                    # "OpenAIException - " (litellm)
]
_CLASS = r"^(litellm\.)?[A-Za-z]*(Error|Exception)\s*:\s*"  # Huelle nur, wenn danach noch eine kommt


def _is_wrapped(line: str) -> bool:
    return any(re.match(rx, line) for rx in _OUTER) or bool(re.match(_CLASS, line))


def _strip_wrappers(line: str) -> str:
    """Huellen vor der eigentlichen Meldung entfernen. Eine Fehlerklasse ("KeyError: ...") bleibt
    stehen, wenn danach nur noch die Meldung kommt - bei Python-Fehlern ist sie die Information."""
    line = line.strip()
    while line:
        for rx in _OUTER:
            if re.match(rx, line):
                line = re.sub(rx, "", line).strip()
                break
        else:
            rest = re.sub(_CLASS, "", line).strip()
            if rest != line and _is_wrapped(rest):
                line = rest
                continue
            return line
    return line


def _retry(text: str) -> str:
    m = (re.search(r"try again in\s*([\d.]+)\s*(ms|s|sec|seconds|m|min)?", text, re.I)
         or re.search(r"retry[_ -]?(?:after|delay)\W{0,4}([\d.]+)\s*(ms|s|sec|seconds|m|min)?", text, re.I))
    if not m:
        return ""
    value, unit = float(m.group(1)), (m.group(2) or "s").lower()
    seconds = value / 1000 if unit == "ms" else value * 60 if unit.startswith("m") and unit != "ms" else value
    return f"{max(1, round(seconds))} s" if seconds < 120 else f"{round(seconds / 60)} min"


def _hint(kind: str, provider: str, model: str, retry: str, iters: tuple[int, int] | None) -> str:
    who = provider or "dem Anbieter"
    mdl = f"„{model}“" if model else "dieses Modell"
    wait = f" (empfohlen: {retry} warten)" if retry else ""
    return {
        "rate_limit": f"Der Anbieter nimmt gerade keine weiteren Anfragen an{wait}. Kurz warten und eine "
                      f"neue Nachricht schicken – der Lauf setzt fort. Dauerhaft: anderes Modell über /model.",
        "auth": f"Der API-Schlüssel für {who} fehlt oder ist ungültig. Schlüssel in der .env prüfen "
                f"oder über /model ein Modell eines anderen Anbieters wählen.",
        "credits": f"Bei {who} ist das Guthaben aufgebraucht. Aufladen oder über /model ein anderes "
                   f"(z. B. kostenloses) Modell wählen.",
        "model": f"{mdl} ist bei {who} nicht (mehr) verfügbar. Über /model ein anderes Modell wählen.",
        "context": "Der Verlauf passt nicht mehr ins Modell. /context zeigt, was Platz belegt; /reset leert "
                   "den Verlauf, /new startet eine frische Session – oder über /model ein Modell mit "
                   "größerem Kontext wählen.",
        "connection": "Das Modell war nicht erreichbar. Netzwerk oder Proxy prüfen; eine neue Nachricht "
                      "setzt den Lauf fort.",
        "overloaded": f"{who.capitalize() if provider else 'Der Anbieter'} meldet eine Störung oder "
                      f"Überlastung{wait}. Später erneut versuchen oder über /model ausweichen.",
        "timeout": "Das Modell hat nicht rechtzeitig geantwortet. Erneut senden – bei Wiederholung über "
                   "/model ein schnelleres Modell wählen.",
        "policy": "Der Anbieter hat die Anfrage wegen seiner Inhaltsrichtlinien blockiert. Anfrage "
                  "umformulieren oder ein anderes Modell wählen.",
        "bad_request": f"{mdl} hat die Anfrage abgelehnt. Häufige Ursache: das Modell kann keine Tools "
                       f"oder Bilder – über /model ein anderes wählen. Genauer Grund in den Details.",
        "empty": f"{mdl} hat mehrfach nichts zurückgegeben. Über /model ein anderes Modell versuchen.",
        "crash": "Ein Fehler im Agenten selbst, nicht beim Anbieter. Der Traceback steht in den Details; "
                 "das Log liegt unter <appdata>/icli2/icli2.log.",
        "limit": ("Der Agent hat sein Limit" + (f" von {iters[1]} Schritten" if iters and iters[1] else "")
                  + " erreicht, ohne fertig zu werden. Eine neue Nachricht setzt fort; das Limit änderst "
                    "du mit /set max_iterations 60."),
        "stream_closed": "Die Verbindung wurde beim Schließen getrennt (bekannt unter Windows). Nachricht "
                         "erneut senden.",
        "unknown": "Genauer Grund in den Details.",
    }[kind]


def explain(raw: str, *, kind: str = "", model: str = "", traceback: str = "",
            iters: tuple[int, int] | None = None) -> dict:
    """Rohen Fehlertext in eine verstaendliche, nuetzliche Darstellung uebersetzen."""
    text = str(raw or "").strip()
    probe = f"{text}\n{traceback}"
    if not kind:
        kind = next((k for k, rx in _RULES if re.search(rx, probe, re.I)), "unknown")
    found = _json_part(text)
    provider = _provider(probe)
    retry = _retry(probe)
    message = _message(text, found[2] if found else None)
    if kind == "limit":
        message = message if message and "max_iterations" not in message else ""
    detail = text
    if found:
        start, end, data = found
        detail = f"{text[:start].rstrip()}\n{json.dumps(data, indent=2, ensure_ascii=False)}\n{text[end:].lstrip()}".strip()
    if traceback:
        detail = f"{detail}\n\n{traceback.strip()}"
    if len(detail) > DETAIL_LIMIT:
        detail = detail[:DETAIL_LIMIT] + f"\n… (+{len(detail) - DETAIL_LIMIT} Zeichen)"
    title = _TITLES[kind]
    if kind == "limit" and iters and iters[1]:
        title += f" ({iters[0]}/{iters[1]})"
    code = re.search(r"\b(4\d\d|5\d\d)\b", text)
    if code and kind in ("rate_limit", "auth", "credits", "model", "overloaded", "bad_request"):
        title += f" ({code.group(1)})"
    return {"kind": kind, "title": title, "hint": _hint(kind, provider, model, retry, iters),
            "provider": provider, "message": message[:600], "retry": retry, "detail": detail}


def render(info: dict) -> str:
    """Mehrzeilige Ansicht fuer die aufgeklappte Fehlerzeile."""
    lines = [info["hint"], ""]
    if info.get("message"):
        who = f" von {info['provider']}" if info.get("provider") else ""
        lines += [f"Meldung{who}:", f"  {info['message']}", ""]
    if info.get("detail") and info["detail"] != info.get("message"):
        lines += ["Details:", *[f"  {line}" for line in info["detail"].splitlines()]]
    return "\n".join(lines).rstrip()
