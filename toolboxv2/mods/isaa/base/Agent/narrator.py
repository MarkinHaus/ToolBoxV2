"""
AgentLiveNarrator
=================
Generates short live-status text for AgentLiveState.thought via a fast
"Blitz" LLM (e.g. groq/llama-3.1-8b-instant).

Activation:
    BLITZ_MODEL=groq/llama-3.1-8b-instant   # required – if empty, narrator is off
    AGENT_NARRATOR_ENABLED=true              # default true
    NARRATOR_LANG=auto                       # auto | de | en

Design rules
------------
* Fire-and-forget via asyncio.create_task – never blocks the engine.
* Cancel-pending: a new task cancels the old one unless the old is a
  think-triggered task (highest priority).
* Rate-limit: min 0.75 s between Blitz calls; soft token budget per run.
* Mock strings are set *synchronously before* any async call so the UI
  always has something to show immediately.
* Think-tool path: Blitz gets the full thinking chunk → builds / updates
  NarratorMiniState (plan, drift, repeat) → used as context in all future
  calls of this run.
"""
import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
import re

from toolboxv2 import get_logger, get_app, Spinner

logger = get_logger()

# ---------------------------------------------------------------------------
# Config / Constants
# ---------------------------------------------------------------------------

BLITZ_MODEL: str = os.getenv("BLITZMODEL", "")
NARRATOR_ENABLED: bool = os.getenv("AGENT_NARRATOR_ENABLED", "true").lower() == "true"
NARRATOR_LANG: str = os.getenv("NARRATOR_LANG", "auto")  # auto | de | en

# Token budget per minute (sliding window, matches Groq free-tier limits)
NARRATOR_MAX_INPUT_TOKENS_PM: int = int(os.getenv("NARRATOR_MAX_INPUT_TPM", "10000"))
NARRATOR_MAX_OUTPUT_TOKENS_PM: int = int(os.getenv("NARRATOR_MAX_OUTPUT_TPM", "1000"))
_BUDGET_WINDOW: float = 60.0  # seconds

# Timing
NARRATOR_MIN_INTERVAL: float = 0.75   # seconds between blitz calls
NARRATOR_STALE_THRESHOLD: float = 0.3  # if live.thought is fresher than this, discard blitz result

# Diff compression: keep only first N chars of each message content
DIFF_CONTENT_MAXCHARS: int = 140
THINK_CONTENT_MAXCHARS: int = 3600

# Blitz prompt tokens (rough estimate per call for budget tracking)
BLITZ_APPROX_SYSTEM_TOKENS: int = 80
BLITZ_APPROX_PER_MSG: int = 25

# ---------------------------------------------------------------------------
# Mock string tables  (DE + EN, with {tool} placeholder where needed)
# ---------------------------------------------------------------------------

_MOCKS: dict[str, dict[str, list[str]]] = {
    "init": {
        "de": [
            "analysiere anfrage …", "starte verarbeitung …", "bereite mich vor …", "initialisiere lauf …",
            "prüfe systemstatus …", "lade module …", "stelle verbindung her …", "konfiguriere parameter …",
            "sichte eingabedaten …", "bereite arbeitsbereich vor …", "schalte system aktiv …", "initialisiere kernel …",
            "werte kontext aus …", "bereite datenstrom vor …", "setze variablen …", "starte ablaufroutine …",
            "checke systemintegrität …", "initialisiere speicher …", "aktiviere schnittstellen …", "bereite anfrage auf …",
            "lade notwendige ressourcen …", "synchronisiere daten …", "führe selbsttest aus …", "initialisiere logik …",
            "prüfe nutzerrechte …", "bereite antwortpuffer vor …", "verifiziere eingabe …", "lade konfigurationsdateien …",
            "baue umgebung auf …", "initialisiere instanz …", "starte prozesskette …", "bereite engine vor …",
            "prüfe netzwerkstatus …", "initialisiere cache …", "schalte logik frei …", "starte zugriff …",
            "analysiere auftrag …", "bereite ausführung vor …", "initialisiere runtime …", "checke anfrageformat …",
            "lade abhängigkeiten …", "starte initialisierung …", "bereite speicherplatz vor …", "prüfe systemlast …",
            "aktiviere rechenmodule …", "initialisiere basis …", "fahre system hoch …", "bereite task vor …",
            "initialisiere workflow …", "starte vorgang …"
        ],
        "en": [
            "analysing request …", "starting up …", "getting ready …", "initialising run …",
            "checking system status …", "loading modules …", "establishing connection …", "configuring parameters …",
            "reviewing input data …", "preparing workspace …", "activating system …", "initialising kernel …",
            "evaluating context …", "preparing data stream …", "setting variables …", "starting routine …",
            "checking system integrity …", "initialising memory …", "activating interfaces …", "preparing request …",
            "loading resources …", "synchronising data …", "running self-test …", "initialising logic …",
            "checking user permissions …", "preparing response buffer …", "verifying input …", "loading config files …",
            "setting up environment …", "initialising instance …", "starting process chain …", "preparing engine …",
            "checking network status …", "initialising cache …", "unlocking logic …", "starting access …",
            "analysing task …", "preparing execution …", "initialising runtime …", "checking request format …",
            "loading dependencies …", "starting initialisation …", "preparing storage …", "checking system load …",
            "activating compute modules …", "initialising base …", "booting system …", "preparing task …",
            "initialising workflow …", "starting process …"
        ],
    },
    "llm_pre": {
        "de": [
            "denke nach …", "formuliere nächsten schritt …", "überlege weiter …", "plane vorgehen …",
            "analysiere situation …", "erwäge optionen …", "sortiere gedanken …", "beurteile lage …",
            "wäge fakten ab …", "erschließe zusammenhänge …", "entwirf lösungsansatz …", "prüfe logik …",
            "reflektiere inhalt …", "berechne wahrscheinlichkeit …", "erarbeite konzept …", "kombiniere informationen …",
            "stelle hypothese auf …", "überdenke strategie …", "analysiere daten …", "erwäge alternativen …",
            "schärfe den blick …", "sortiere prioritäten …", "tue mich schlau …", "entfalte gedankengang …",
            "prüfe plausibilität …", "verknüpfe wissen …", "erkenne muster …", "plane die taktik …",
            "überlege möglichkeiten …", "sichte fakten …", "fokussiere problem …", "erarbeite struktur …",
            "schätze aufwand ab …", "reflektiere vorgang …", "ordne informationen …", "analysiere kontext …",
            "erwäge vorteile …", "entwirf schritte …", "überprüfe plausibilität …", "konkretisiere plan …",
            "denke konzeptionell …", "abstrahiere problem …", "gewichte argumente …", "durchlaufe möglichkeiten …",
            "identifiziere kern …", "formuliere ansatz …", "berate mich intern …", "erwäge sichtweisen …",
            "überlege notwendigkeit …", "plane das weitere …"
        ],
        "en": [
            "thinking …", "planning next step …", "reasoning …", "considering options …",
            "analysing situation …", "evaluating choices …", "sorting thoughts …", "assessing state …",
            "weighing facts …", "identifying connections …", "drafting approach …", "checking logic …",
            "reflecting on content …", "calculating probability …", "developing concept …", "combining data …",
            "forming hypothesis …", "rethinking strategy …", "analysing data …", "considering alternatives …",
            "sharpening focus …", "sorting priorities …", "gathering insights …", "unfolding thought process …",
            "checking plausibility …", "connecting knowledge …", "recognising patterns …", "planning tactics …",
            "considering possibilities …", "reviewing facts …", "focusing on problem …", "developing structure …",
            "estimating effort …", "reflecting on process …", "organising information …", "analysing context …",
            "considering advantages …", "drafting steps …", "verifying plausibility …", "concretising plan …",
            "thinking conceptually …", "abstracting problem …", "weighing arguments …", "exploring options …",
            "identifying core …", "formulating approach …", "consulting internally …", "considering perspectives …",
            "thinking about necessity …", "planning next moves …"
        ],
    },
    "tool_start": {
        "de": [
            "starte {tool} …", "rufe {tool} auf …", "führe {tool} aus …", "wende {tool} an …",
            "aktiviere {tool} …", "initialisiere {tool} …", "starte ausführung von {tool} …", "nutze {tool} …",
            "lade {tool} …", "starte prozess: {tool} …", "führe {tool} durch …", "rufe funktion {tool} ab …",
            "setze {tool} ein …", "starte anwendung von {tool} …", "initiiere {tool} …", "starte tool-sequenz: {tool} …",
            "wähle {tool} …", "gebe {tool} frei …", "arbeite mit {tool} …", "schalte {tool} scharf …",
            "starte instanz {tool} …", "führe {tool} für dich aus …", "rufe auf {tool} …", "verwende nun {tool} …",
            "starte den einsatz von {tool} …", "aktiviere {tool} schnittstelle …", "führe {tool} befehl aus …",
            "starte das tool {tool} …", "initiiere nutzung von {tool} …", "starte {tool} arbeitsablauf …",
            "rufe externes {tool} …", "wende {tool} methode an …", "starte {tool} operation …", "setze {tool} in gang …",
            "führe {tool} aufgabe aus …", "starte {tool} abfrage …", "nutze {tool} modul …", "aktiviere {tool} funktion …",
            "rufe {tool} prozess …", "starte {tool} analyse …", "führe {tool} berechnung …", "initiiere {tool} aufruf …",
            "setze {tool} aktiv …", "starte {tool} task …", "rufe {tool} dienst …", "wende {tool} tool an …",
            "starte {tool} einheit …", "führe {tool} logik aus …", "aktiviere {tool} modul …", "starte {tool} jetzt …"
        ],
        "en": [
            "running {tool} …", "calling {tool} …", "executing {tool} …", "invoking {tool} …",
            "activating {tool} …", "initialising {tool} …", "starting execution of {tool} …", "using {tool} …",
            "loading {tool} …", "starting process: {tool} …", "carrying out {tool} …", "calling function {tool} …",
            "deploying {tool} …", "starting application of {tool} …", "initiating {tool} …", "starting sequence: {tool} …",
            "selecting {tool} …", "releasing {tool} …", "working with {tool} …", "activating {tool} …",
            "starting instance {tool} …", "running {tool} for you …", "calling on {tool} …", "using {tool} now …",
            "starting deployment of {tool} …", "activating {tool} interface …", "executing {tool} command …",
            "starting tool {tool} …", "initiating use of {tool} …", "starting {tool} workflow …",
            "calling external {tool} …", "applying {tool} method …", "starting {tool} operation …", "setting {tool} in motion …",
            "executing {tool} task …", "starting {tool} query …", "using {tool} module …", "activating {tool} function …",
            "calling {tool} process …", "starting {tool} analysis …", "executing {tool} calculation …", "initiating {tool} call …",
            "setting {tool} active …", "starting {tool} task …", "calling {tool} service …", "applying {tool} tool …",
            "starting {tool} unit …", "executing {tool} logic …", "activating {tool} module …", "starting {tool} now …"
        ],
    },
    "tool_end": {
        "de": [
            "werte ergebnis aus …", "verarbeite antwort …", "lass mich das zusammenfassen …", "prüfe resultat …",
            "analysiere ergebnis …", "fasse die daten zusammen …", "interpretiere ausgabe …", "sichte den output …",
            "gehe das resultat durch …", "verifiziere die antwort …", "bereite ergebnis auf …", "werte die rückgabe aus …",
            "betrachte die ergebnisse …", "prüfe den erfolg …", "fasse zusammen …", "analysiere den output …",
            "verarbeite das erhaltene …", "checke die daten …", "sichte das ergebnis …", "werte rückmeldung aus …",
            "fasse punkt für punkt zusammen …", "betrachte die antwort …", "prüfe die datenqualität …", "interpretiere resultat …",
            "schließe die analyse ab …", "verarbeite output …", "fasse die antwort zusammen …", "werte dies aus …",
            "prüfe die ausgabe …", "analysiere rückgabe …", "fasse die erkenntnisse zusammen …", "bereite zusammenfassung vor …",
            "verifiziere ergebnis …", "sichte die antwort …", "werte daten aus …", "betrachte resultat …",
            "verarbeite rückmeldung …", "fasse ergebnis zusammen …", "prüfe verarbeitung …", "analysiere antwort …",
            "werte die ergebnisse aus …", "fasse das wichtige zusammen …", "interpretiere daten …", "checke ergebnis …",
            "verarbeite die antwort …", "fasse punktuell zusammen …", "werte aus …", "prüfe die antwort …",
            "analysiere resultat …", "schließe auswertung ab …"
        ],
        "en": [
            "processing result …", "evaluating output …", "let me summarise …", "checking result …",
            "analysing result …", "summarising data …", "interpreting output …", "reviewing the output …",
            "going through the result …", "verifying the answer …", "preparing result …", "evaluating feedback …",
            "observing results …", "checking for success …", "summarising …", "analysing output …",
            "processing what was received …", "checking the data …", "reviewing result …", "evaluating feedback …",
            "summarising point by point …", "observing the answer …", "checking data quality …", "interpreting result …",
            "concluding analysis …", "processing output …", "summarising the answer …", "evaluating this …",
            "checking the output …", "analysing feedback …", "summarising insights …", "preparing summary …",
            "verifying result …", "reviewing the answer …", "evaluating data …", "observing result …",
            "processing feedback …", "summarising result …", "checking processing …", "analysing answer …",
            "evaluating the results …", "summarising key points …", "interpreting data …", "checking result …",
            "processing the answer …", "summarising punctually …", "evaluating …", "checking the answer …",
            "analysing result …", "concluding evaluation …"
        ],
    },
    "think": {
        "de": [
            "denke gründlich nach …", "analysiere problem …", "durchdenke lösung …", "erarbeite strategie …",
            "vertiefe gedankengang …", "betrachte alle facetten …", "wäge argumente ab …", "analysiere komplexität …",
            "erwäge den besten weg …", "strukturiere den gedanken …", "reflektiere intensiv …", "überdenke lösungswege …",
            "konzentriere mich …", "untersuche den sachverhalt …", "plane den schritt …", "kombiniere ansätze …",
            "erkenne logische pfade …", "beurteile die situation …", "erarbeite eine lösung …", "denke in die tiefe …",
            "analysiere die details …", "durchleuchte den kontext …", "wäge konsequenzen ab …", "plane die vorgehensweise …",
            "vertiefe das verständnis …", "betrachte die logik …", "entwirf schritt für schritt …", "erkenne das muster …",
            "überlege sorgfältig …", "analysiere den kern …", "durchdenke die optionen …", "erarbeite den plan …",
            "fokussiere das ziel …", "betrachte alternativen …", "wäge die möglichkeiten ab …", "denke strukturiert …",
            "analysiere das grundproblem …", "durchleuchte den bereich …", "erarbeite neue wege …", "reflektiere die logik …",
            "vertiefe die planung …", "betrachte die auswirkungen …", "erkenne logische schlüsse …", "überlege den ansatz …",
            "analysiere die herausforderung …", "durchdenke das vorgehen …", "erarbeite taktik …", "fokussiere das wesentliche …",
            "betrachte die strategie …", "denke gründlich durch …"
        ],
        "en": [
            "thinking deeply …", "analysing problem …", "working through solution …", "elaborating strategy …",
            "deepening thoughts …", "considering all facets …", "weighing arguments …", "analysing complexity …",
            "considering best path …", "structuring thought …", "reflecting intensely …", "rethinking solutions …",
            "focusing …", "investigating facts …", "planning the step …", "combining approaches …",
            "identifying logic paths …", "assessing the situation …", "developing a solution …", "thinking in depth …",
            "analysing details …", "examining the context …", "weighing consequences …", "planning the procedure …",
            "deepening understanding …", "considering the logic …", "drafting step by step …", "recognising patterns …",
            "thinking carefully …", "analysing the core …", "thinking through options …", "developing the plan …",
            "focusing on the goal …", "considering alternatives …", "weighing possibilities …", "thinking structured …",
            "analysing the root problem …", "examining the area …", "developing new paths …", "reflecting logic …",
            "deepening the plan …", "considering impacts …", "identifying logical conclusions …", "considering the approach …",
            "analysing the challenge …", "thinking through the process …", "developing tactics …", "focusing on the essentials …",
            "considering strategy …", "thinking thoroughly …"
        ],
    },
    "summarise": {
        "de": [
            "lass mich das zusammenfassen …", "bereite abschlussbericht vor …", "fasse ergebnisse zusammen …",
            "ziehe ein fazit …", "bringe es auf den punkt …", "erstelle die zusammenfassung …", "fasse den inhalt zusammen …",
            "schließe die antwort ab …", "gebe kurzes resümee …", "fasse die punkte zusammen …", "bereite die finale antwort vor …",
            "fasse alles kurz zusammen …", "erarbeite das fazit …", "bringe die essenz hervor …", "fasse die arbeit zusammen …",
            "schließe den prozess ab …", "formuliere das ergebnis …", "fasse dies übersichtlich zusammen …", "gebe zusammenfassenden bericht …",
            "fasse das wesen zusammen …", "bereite abschluss vor …", "bringe ergebnisse kompakt zusammen …", "fasse alles in worte …",
            "schließe die zusammenfassung ab …", "fasse den kern zusammen …", "erstelle resümee …", "fasse die antwort zusammen …",
            "bereite finale schritte vor …", "bringe es auf die essenz …", "fasse übersichtlich …", "schließe ab …",
            "fasse die fakten zusammen …", "gebe überblick …", "fasse die essenz zusammen …", "bereite resultat vor …",
            "bringe es kurz zusammen …", "fasse die logik zusammen …", "schließe mit zusammenfassung …", "formuliere abschluss …",
            "fasse das ergebnis präzise zusammen …", "gebe kurzes fazit …", "fasse alles strukturiert zusammen …",
            "bereite den abschluss vor …", "bringe die fakten auf den punkt …", "fasse die antwort kompakt zusammen …",
            "schließe die zusammenfassung …", "formuliere das resümee …", "fasse dies prägnant zusammen …", "bereite ende vor …",
            "fasse schlussfolgerung zusammen …"
        ],
        "en": [
            "let me summarise …", "preparing final answer …", "wrapping up results …",
            "drawing a conclusion …", "getting to the point …", "creating summary …", "summarising content …",
            "completing the answer …", "giving a brief resume …", "summarising points …", "preparing final response …",
            "summarising everything …", "developing conclusion …", "highlighting essence …", "summarising work …",
            "concluding process …", "formulating result …", "summarising clearly …", "providing summary report …",
            "summarising the core …", "preparing conclusion …", "compacting results …", "summarising in words …",
            "finishing summary …", "summarising the core …", "creating resume …", "summarising answer …",
            "preparing final steps …", "distilling the essence …", "summarising overview …", "concluding …",
            "summarising facts …", "giving overview …", "summarising essence …", "preparing result …",
            "summarising briefly …", "summarising logic …", "concluding with summary …", "formulating conclusion …",
            "summarising result precisely …", "giving brief conclusion …", "summarising structured …",
            "preparing conclusion …", "getting facts to the point …", "summarising answer compactly …",
            "closing summary …", "formulating resume …", "summarising concisely …", "preparing end …",
            "summarising conclusion …"
        ],
    },
    "drift": {
        "de": " ⚠ plan-abweichung",
        "en": " ⚠ drift detected",
    },
    "repeat": {
        "de": " ↩ wiederholung",
        "en": " ↩ repetition",
    },
}

# ---------------------------------------------------------------------------
# NarratorMiniState – built from think-tool results
# ---------------------------------------------------------------------------

@dataclass
class NarratorMiniState:
    """
    Compact state the narrator maintains across iterations of one run.
    Updated exclusively from think-tool Blitz calls.
    """
    plan_summary: str = ""   # ≤ 1 sentence: what the agent intends to do
    drift: bool = False      # Blitz detected deviation from plan
    repeat: bool = False     # Blitz detected repetition
    last_think_hash: str = ""  # dedup: skip if identical thinking chunk


# ---------------------------------------------------------------------------
# Helper: language detection
# ---------------------------------------------------------------------------

def _lang(query: str = "") -> str:
    """Return 'de' or 'en'."""
    if NARRATOR_LANG in ("de", "en"):
        return NARRATOR_LANG
    # simple heuristic: count German stop-words
    de_words = {"ich", "du", "ist", "das", "die", "der", "und", "was", "wie",
                "nicht", "mit", "auf", "für", "von"}
    tokens = set(query.lower().split())
    return "de" if len(tokens & de_words) >= 2 else "en"


def _get_best_match(query: str, pool: list[str]) -> str:
    """
    Super schnelle Suche nach dem Satz mit den meisten gemeinsamen Wörtern.
    """
    if not query:
        return random.choice(pool)

    query_words = set(query.lower().split())
    best_sentence = pool[0]
    max_matches = -1

    for sentence in pool:
        # Zähle wie viele Wörter der query im Satz vorkommen
        sentence_words = set(sentence.lower().split())
        matches = len(query_words.intersection(sentence_words))

        # Wenn wir einen exzellenten Match finden, sofort zurückgeben
        if matches > max_matches:
            max_matches = matches
            best_sentence = sentence

        # Wenn wir schon viele Matches haben (z.B. > 3), reicht das aus
        if max_matches >= 3:
            break

    return best_sentence


import re
import random
import time


def _extract_technical_entities(text: str) -> list[str]:
    """
    Sucht im echten Text nach aussagekräftigen Dingen:
    Dateinamen (main.py), CamelCase/snake_case Variablen, Pfade oder in Quotes gesetzte Wörter.
    """
    if not text:
        return []

    # Matches: "word", 'word', datei.ext, Pfade/zu/datei, snake_case, CamelCase
    pattern = r'(?:"([^"]+)"|\'([^\']+)\'|\b(\w+\.\w{2,5})\b|\b(?:[a-zA-Z0-9]+/[a-zA-Z0-9/]+)\b|\b([a-z]+_[a-z_]+)\b|\b([A-Z][a-z]+[A-Z]\w+)\b)'

    matches = re.finditer(pattern, text[-2000:])  # Nur die letzten 2000 Zeichen (temporaler Fokus)
    entities = []
    for match in matches:
        # Finde die erste Gruppe im Regex, die nicht None ist
        found = next((m for m in match.groups() if m), match.group(0))
        # Bereinigen und filtern (mindestens 3 Zeichen)
        clean = found.strip("'\".,;:()[]{}\\")
        if len(clean) > 2 and clean.lower() not in {"the", "and", "der", "die", "das"}:
            entities.append(clean)

    # Rückgabe der letzten 5 einzigartigen Entitäten (die aktuellsten)
    return list(dict.fromkeys(entities))[-5:]


def _mock(key: str, lang: str, mini_state=None, context_str: str = "", **fmt) -> str:
    # 1. Flags (Prio 1)
    if mini_state:
        if getattr(mini_state, "drift", False) and "drift" in _MOCKS:
            return _MOCKS["drift"].get(lang, " ⚠ drift detected")
        if getattr(mini_state, "repeat", False) and "repeat" in _MOCKS:
            return _MOCKS["repeat"].get(lang, " ↩ repetition")

    # 2. Reale Daten extrahieren (Der Kern für die 99% Informativität)
    real_entities = _extract_technical_entities(context_str)
    plan_text = (getattr(mini_state, "plan_summary", "") or "").lower()

    # 3. Pool holen und mischen
    category_data = _MOCKS.get(key, {})
    pool = category_data.get(lang, category_data.get("de", ["…"]))[:]
    random.shuffle(pool)

    # Filter: Nur Wörter >= 4 Zeichen
    def get_clean_words(text):
        return {w for w in re.findall(r'\w{4,}', text)}

    plan_words = get_clean_words(plan_text)
    context_words = get_clean_words(context_str[-1000:].lower() if context_str else "")

    # 4. Scoring
    scored_candidates = []
    for text_template in pool:
        mock_words = get_clean_words(text_template.lower())
        score = (len(plan_words.intersection(mock_words)) * 3) + \
                (len(context_words.intersection(mock_words)) * 1)
        scored_candidates.append((score, text_template))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # 5. Remix & Entity Injection
    top_texts = [txt for _, txt in scored_candidates[:3]]

    res = top_texts[0]  # Fallback, falls zusammenbauen fehlschlägt

    if len(top_texts) >= 2:
        p1 = top_texts[0].replace("…", "").strip()
        p2 = top_texts[1].replace("…", "").strip()

        p1_word_set = {w.lower() for w in re.findall(r'\w{2,}', p1)}
        p2_filtered_words = [w for w in p2.split() if w.lower().strip(",.;") not in p1_word_set]

        connector = " und " if lang == "de" else " and "
        if p2_filtered_words:
            res = f"{p1}{connector}{' '.join(p2_filtered_words)}"
        else:
            res = p1

    # === NEU: REALE DATEN IN DEN MOCK EINBAUEN ===
    if real_entities:
        # Wir hängen die aktuellste Entität an, falls sinnvoll, z.B. (-> main.py)
        # Oder wir ersetzen generische Wörter wie "problem", "daten", "logik" durch die echte Entität
        latest_entity = real_entities[-1]

        # Ersetze generische Wörter (intelligent!)
        generic_words = ["problem", "daten", "logik", "situation", "fakten", "data", "informationen", "details"]
        replaced = False
        for gw in generic_words:
            if gw in res.lower():
                # RegEx für Case-Insensitive Replace
                res = re.sub(rf'\b{gw}\b', f"'{latest_entity}'", res, flags=re.IGNORECASE)
                replaced = True
                break

        # Wenn kein generisches Wort da war, hängen wir den Kontext elegant an
        if not replaced:
            target_connector = " -> " if " " in res else " "
            res = f"{res}{target_connector}{latest_entity}"

    # Abschließende Punkte anfügen
    if not res.endswith("…"):
        res = f"{res.strip()} …"

    return res.format(**fmt) if fmt else res
# ---------------------------------------------------------------------------
# History compression for diff
# ---------------------------------------------------------------------------

def _compress_diff(history: list[dict], cursor: int) -> list[dict]:
    """
    Return only new messages since cursor with content truncated.
    Skips system messages (loop-warning injections etc.).
    """
    new_msgs = history[cursor:]
    compressed = []
    for m in new_msgs:
        role = m.get("role", "")
        if role == "system":
            continue
        content = m.get("content") or ""
        if isinstance(content, list):
            # multipart – flatten to text
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        compressed.append({
            "role": role,
            "content": content[:DIFF_CONTENT_MAXCHARS],
        })
    return compressed


def _estimate_tokens(msgs: list[dict]) -> int:
    total_chars = sum(len(m.get("content", "")) for m in msgs)
    return BLITZ_APPROX_SYSTEM_TOKENS + len(msgs) * BLITZ_APPROX_PER_MSG + total_chars // 4


# ---------------------------------------------------------------------------
# Blitz prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_NORMAL_DE = (
    "Du bist ein Agent-Monitor. Antworte NUR mit kompaktem JSON, kein Markdown:\n"
    '{"t":"<max 8 Wörter was agent jetzt tut>","d":0,"r":0}\n'
    "d=1 bei plan-abweichung, r=1 bei wiederholung. Kein anderes Feld."
)
_SYSTEM_NORMAL_EN = (
    "You are an agent monitor. Reply ONLY with compact JSON, no markdown:\n"
    '{"t":"<max 8 words what agent is doing now>","d":0,"r":0}\n'
    "d=1 if drifting from plan, r=1 if repeating. No other fields."
)

_SYSTEM_THINK_DE = (
    "Du bist ein Monitor. Lese die folgenden Gedanken des Agenten und fasse den Inhalt zusammen.\n"
    "Antworte AUSSCHLIESSLICH mit JSON:\n"
    '{"t": "<max 8 Worte Zusammenfassung WAS er gerade denkt>", "plan": "<1 kurzer Satz WAS er als nächstes tun will>", "d": 0, "r": 0}\n'
    "d=1 bei Plan-Abweichung, r=1 bei Wiederholung. Kein Markdown."
)

_SYSTEM_THINK_EN = (
    "You are a monitor. Read the following thoughts of the agent and summarize the content.\n"
    "Reply ONLY with JSON:\n"
    '{"t": "<max 8 words summary of WHAT the agent is thinking>", "plan": "<1 short sentence on WHAT he will do next>", "d": 0, "r": 0}\n'
    "d=1 if plan drift, r=1 if repeating. No markdown."
)


def _build_normal_prompt(
    lang: str,
    diff_msgs: list[dict],
    mini: NarratorMiniState,
    tool_hint: str = "",
) -> tuple[str, list[dict]]:
    system = _SYSTEM_NORMAL_DE if lang == "de" else _SYSTEM_NORMAL_EN
    parts = []
    if mini.plan_summary:
        label = "Plan" if lang == "en" else "Plan"
        parts.append(f"{label}: {mini.plan_summary}")
    if tool_hint:
        label = "Tool" if lang == "en" else "Tool"
        parts.append(f"{label}: {tool_hint}")
    if diff_msgs:
        label = "Recent" if lang == "en" else "Neu"
        parts.append(f"{label}:\n" + "\n".join(
            f"[{m['role']}] {m['content']}" for m in diff_msgs[-4:]
        ))
    user_content = "\n".join(parts) or ("." if lang == "en" else ".")
    return system, [{"role": "user", "content": user_content}]


def _build_think_prompt(
    lang: str,
    thinking_content: str,
    mini: NarratorMiniState,
) -> tuple[str, list[dict]]:
    system = _SYSTEM_THINK_DE if lang == "de" else _SYSTEM_THINK_EN
    label_prev = "Vorheriger Plan" if lang == "de" else "Previous plan"
    label_think = "Denken" if lang == "de" else "Thinking"
    parts = []
    if mini.plan_summary:
        parts.append(f"{label_prev}: {mini.plan_summary}")
    parts.append(f"{label_think}:\n{thinking_content[:int(THINK_CONTENT_MAXCHARS/2)] + '...' + thinking_content[-int(THINK_CONTENT_MAXCHARS/2):]}")
    return system, [{"role": "user", "content": "\n".join(parts)}]


# ---------------------------------------------------------------------------
# Core: blitz API call  (raw litellm, NOT through agent machinery)
# ---------------------------------------------------------------------------
async def _call_blitz(system: str, messages: list[dict], max_tokens: int = 60) -> dict | None:
    """
    Call BLITZ_MODEL via LiteLLMRateLimitHandler (with fallback + key rotation).
    Falls back to raw litellm.acompletion if no handler is initialised.
    Returns parsed JSON dict or None on any error.
    """
    try:
        all_messages = [{"role": "system", "content": system}] + messages

        if _blitz_handler is not None:
            import litellm as _litellm_mod

            response = await asyncio.wait_for(
                _blitz_handler.completion_with_rate_limiting(
                    _litellm_mod,
                    model = BLITZ_MODEL,
                    messages = all_messages,
                    max_tokens = max_tokens,
                    temperature = 0.3,
                    stream = False,
                    drop_params = True,
                ),
                timeout = 10.0  # ← HARD TIMEOUT
            )
        else:
            # Fallback: direkt litellm ohne handler
            import litellm  # type: ignore

            response = await asyncio.wait_for(
                litellm.acompletion(
                    model = BLITZ_MODEL,
                    messages = all_messages,
                    max_tokens = max_tokens,
                    temperature = 0.3,
                    stream = False,
                    drop_params = True,
                ),
                timeout = 10.0  # ← HARD TIMEOUT
            )

        raw = response.choices[0].message.content or ""
        raw = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        usage = getattr(response, "usage", None)
        return {
            "data": data,
            "raw": raw,
            "in": getattr(usage, "prompt_tokens", 0),
            "out": getattr(usage, "completion_tokens", 0),
        }
    except asyncio.TimeoutError:
        logger.debug("Blitz call timed out (10s)")
        return None
    except asyncio.CancelledError:
        raise
    except json.JSONDecodeError as exc:
        logger.debug("Blitz JSON decode failed (truncated?): %s", exc)
        return None
    except Exception as exc:
        logger.debug("Blitz call failed: %s", exc)
        return None
# ---------------------------------------------------------------------------
# AgentLiveNarrator
# ---------------------------------------------------------------------------

_GROQ_DEFAULT_FALLBACKS: dict[str, list[str]] = {
    # Primary model → ordered fallback list
    #   Strategy: same-class first, then smaller/cheaper
    "groq/llama-3.1-8b-instant": [
        "groq/llama-3.3-70b-versatile",
        "groq/openai/gpt-oss-20b",
    ],
    "groq/llama-3.3-70b-versatile": [
        "groq/llama-3.1-8b-instant",
        "groq/openai/gpt-oss-20b",
    ],
    "groq/openai/gpt-oss-120b": [
        "groq/openai/gpt-oss-20b",
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
    ],
    "groq/openai/gpt-oss-20b": [
        "groq/llama-3.1-8b-instant",
        "groq/llama-3.3-70b-versatile",
    ],
}

# Generic fallback when BLITZ_MODEL is a groq model not in the table above
_GROQ_GENERIC_FALLBACKS: list[str] = [
    "groq/llama-3.1-8b-instant",
    "groq/llama-3.3-70b-versatile",
    "groq/openai/gpt-oss-20b",
]


def _resolve_groq_fallbacks(primary: str) -> list[str]:
    """
    Return a fallback chain for a groq primary model.
    Uses the explicit table if available, otherwise the generic list
    (filtering out the primary itself).
    """
    if primary in _GROQ_DEFAULT_FALLBACKS:
        return _GROQ_DEFAULT_FALLBACKS[primary]
    # unknown groq model → generic chain minus itself
    return [m for m in _GROQ_GENERIC_FALLBACKS if m != primary]

_blitz_handler: "LiteLLMRateLimitHandler | None" = None


def init_narrator_handler(
    handler: "LiteLLMRateLimitHandler | None" = None,
    fallback_models: list[str] | None = None,
) -> "LiteLLMRateLimitHandler":
    """
    Initialise (or replace) the module-level rate-limit handler used by
    all Narrator Blitz calls.

    Args:
        handler:          Pass an existing handler to share with the rest of
                          the engine.  If None a fresh one is created.
        fallback_models:  Optional fallback chain for BLITZ_MODEL.
                          If None AND BLITZ_MODEL is a groq model,
                          a sensible default chain is auto-selected.

    Returns the active handler so the caller can further configure it.
    """
    global _blitz_handler

    if handler is not None:
        _blitz_handler = handler
        return _blitz_handler

    # Auto-detect groq fallbacks when none provided
    if fallback_models is None and "groq" in BLITZ_MODEL.lower():
        fallback_models = _resolve_groq_fallbacks(BLITZ_MODEL)
        logger.info(
            "Narrator: auto-selected groq fallbacks for %s → %s",
            BLITZ_MODEL, fallback_models,
        )

    # lazy import – keep module importable without the handler installed
    from toolboxv2.mods.isaa.base.IntelligentRateLimiter import LiteLLMRateLimitHandler

    _blitz_handler = LiteLLMRateLimitHandler(
        enable_model_fallback=bool(fallback_models),
        enable_key_rotation=True,
        key_rotation_mode="balance",
        max_retries=2,
    )

    if fallback_models:
        _blitz_handler.add_fallback_chain(
            primary_model=BLITZ_MODEL,
            fallback_models=fallback_models,
            fallback_duration=90.0,
        )

    return _blitz_handler


class AgentLiveNarrator:
    """
    Attach to ExecutionEngine as ``self._narrator``.

    Usage pattern in engine::

        # synchronous set (immediate UI update)
        self._narrator.mock("tool_start", tool="calc")

        # async schedule (fire-and-forget, non-blocking)
        asyncio.create_task(self._narrator.on_tool_end(ctx))
    """
    def __init__(self, live: "AgentLiveState", agent: Any, do_narator=True,
             handler: "LiteLLMRateLimitHandler | None" = None,
             fallback_models: list[str] | None = None):
        self._inspier = None
        self.live = live
        self.agent = agent  # kept for future: model preference routing
        self._enabled: bool = bool(BLITZ_MODEL) and NARRATOR_ENABLED and do_narator

        if self._enabled:
            init_narrator_handler(handler=handler, fallback_models=fallback_models)

        # Per-run state (reset on each execute call)
        self._mini: NarratorMiniState = NarratorMiniState()
        self._lang: str = NARRATOR_LANG if NARRATOR_LANG != "auto" else "de"

        # Async task management
        self._pending_tasks: list[asyncio.Task] = []
        self._pending_is_think: bool = False  # think tasks are NOT cancelled by normal tasks

        # Rate-limiting
        self._last_blitz_time: float = 0.0
        self._thought_set_time: float = 0.0   # when we last wrote live.thought

        # Per-minute sliding window budget  [(timestamp, in_tokens, out_tokens), ...]
        # NOT reset per-run – persists across runs (shared across the engine lifetime)
        self._token_log: list[tuple[float, int, int]] = []

        # History cursor (diff tracking)
        self._history_cursor: int = 0

        # Print Toggle (Default True, CLI kann es auf False setzen)
        self.enable_console_print = os.getenv("NARRATOR_CONSOLE_PRINT", "true").lower() == "true"

        #  Callback für Live-Streaming
        self.on_live_update_callback = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def reset(self, query: str = ""):
        """Call at the start of each execute() run."""
        self._mini = NarratorMiniState()
        self._lang = _lang(query)
        self._pending_task = None
        self._pending_is_think = False
        self._last_blitz_time = 0.0
        self._thought_set_time = 0.0
        # NOTE: _token_log is NOT reset here – it's a per-minute sliding window
        # that lives across runs so the PM limit is respected correctly.
        self._history_cursor = 0

    # ------------------------------------------------------------------
    # Synchronous mock setters (always instant, no async needed)
    # ------------------------------------------------------------------

    def mock(self, key: str, **fmt) -> None:
        """Set live.thought immediately with a predefined mock string."""
        text = _mock(key, self._lang, mini_state=self._mini, context_str=self._inspier, **fmt)
        self._set_thought(text, moc=True)

    def _set_thought(self, text: str, moc=False) -> None:
        from toolboxv2 import get_app

        now = time.monotonic()

        # Debouncing: Ignoriere Mocks, wenn vor kurzem ein ECHTER Text gesetzt wurde
        if moc and hasattr(self, "_thought_set_time") and (now - self._thought_set_time) < 1.5:
            return

        if getattr(self, "_pending_is_think", False) or (moc and (now - getattr(self, "_thought_set_time", 0)) > 600):
            self._inspier = ""

        if not moc:
            # Puffer pflegen (Maximalgröße beschränken, z.B. 2000 Zeichen, spart RAM/CPU)
            if self._inspier is None:
                self._inspier = ""
            self._inspier = (self._inspier + " " + text)[-2000:]
            self._thought_set_time = now
        elif moc:
            self._thought_set_time = now

        # UI Update
        # Das Löschen mit " "*25 verhindert Artefakte bei kürzeren Strings im Terminal
        formatted_text = text.strip()
        if self.enable_console_print:
            from toolboxv2 import get_app
            get_app().print(formatted_text + " " * 25, end="\r")
        self.live.narrator_msg = formatted_text
        if self.on_live_update_callback:
            self.on_live_update_callback(text)

    def _append_state_flags(self, text: str) -> str:
        if self._mini.drift:
            text += _MOCKS["drift"][self._lang]
        if self._mini.repeat:
            text += _MOCKS["repeat"][self._lang]
        return text

    # ------------------------------------------------------------------
    # Event handlers  (called from engine – all are async but lightweight)
    # ------------------------------------------------------------------

    def on_init(self, query: str) -> None:
        """Call right after live.enter(INIT, ...). Sync – no blitz yet."""
        self._lang = _lang(query)
        self.mock("init")

    def on_llm_pre_call(self, history: list[dict]) -> None:
        """
        Called just before each LLM call.
        Always sets mock immediately.
        Does NOT schedule blitz (no new tool results yet → last info is better).
        """
        self.mock("llm_pre")
        # advance cursor so next diff only contains what happened AFTER this call
        self._history_cursor = len(history)

    def on_tool_start(self, tool_name: str) -> None:
        """Sync: set mock for tool start immediately."""
        if tool_name == "think":
            self.mock("think")
        else:
            self.mock("tool_start", tool=tool_name)

    def schedule_tool_end(
        self,
        tool_name: str,
        result_snippet: str,
        history: list[dict],
    ) -> None:
        """
        After tool finished. Sets mock immediately, then schedules
        blitz in background *if* rate-limit and budget allow.
        """
        self.mock("tool_end")

        if not self._enabled:
            return
        if not self._budget_ok(history):
            return
        if time.monotonic() - self._last_blitz_time < NARRATOR_MIN_INTERVAL:
            # still inside cool-down – mock string is fine
            return

        diff = _compress_diff(history, self._history_cursor)
        self._history_cursor = len(history)

        self._schedule(
            self._blitz_normal(diff, tool_hint=f"{tool_name}: {result_snippet[:60]}"),
            is_think=False,
        )

    def schedule_think_result(
        self,
        thinking_content: str,
        history: list[dict],
    ) -> None:
        """
        Special path: agent finished a think-tool call.
        Always tries to call Blitz (cancels any non-think pending task).
        """
        try:
            # Fallback falls thinking_content None oder kein String ist
            if not thinking_content:
                thinking_content = ""
            thinking_content = str(thinking_content)

            # dedup: same thinking chunk → skip
            chunk_hash = hashlib.md5(thinking_content[:200].encode('utf-8', errors='ignore')).hexdigest()[:8]
            if chunk_hash == self._mini.last_think_hash:
                return
            self._mini.last_think_hash = chunk_hash

            self._history_cursor = len(history)

            if not self._enabled:
                self.mock("think")
                return

            if not self._budget_ok(history):
                return

            # cancel old tasks
            self._cancel_pending(force=True)

            self._schedule(
                self._blitz_think(thinking_content),
                is_think=True,
            )
        except Exception as exc:
            # WICHTIG: Fehler abfangen, damit der Main-Agent NICHT crasht!
            logger.error(f"CRITICAL ERROR in Narrator schedule_think_result: {exc}", exc_info=True)
            self.mock("think") # Fallback UI Update

    async def blitz(
        self,
        system: str,
        messages: list[dict],
        schema: dict | None = None,
        respect_ratelimit: bool = False,
        history: list[dict] | None = None,
    ) -> dict | str | None:
        """
        Public raw Blitz-Model call – now routed through the rate-limit handler.

        Args:
            system:            System prompt.
            messages:          User/assistant messages.
            schema:            Optional expected JSON schema as dict.
            respect_ratelimit: If True, checks internal narrator PM budget
                               before calling (handler has its own limits too).
            history:           Working history – for internal budget estimation.

        Returns:
            Parsed dict if schema given and valid,
            raw string if no schema,
            None on error / budget exceeded.
        """
        if not self._enabled:
            return None

        if respect_ratelimit:
            if not self._budget_ok(history or []):
                return None
        with Spinner(message="calling blitz", symbols="q"):
            result = await _call_blitz(system, messages)
        if result is None:
            return None

        if respect_ratelimit:
            self._record_tokens(result["in"], result["out"])

        data = result["data"]

        if schema is None:
            return data if isinstance(data, dict) else result.get("raw", "")

        for key, expected_type in schema.items():
            if key not in data:
                logger.debug("Blitz schema mismatch: missing key %r", key)
                return None
            if not isinstance(data[key], expected_type):
                logger.debug(
                    "Blitz schema mismatch: key %r expected %s got %s",
                    key, expected_type.__name__, type(data[key]).__name__,
                )
                return None

        return data


    def finalize(self) -> None:
        """Cancel ALL pending narrator tasks. Call before agent shutdown."""

        self._cancel_pending(force=True)
        # Zusätzliche Sicherheit: Alle noch nicht erfassten Tasks killen
        for task in self._pending_tasks[:]:
            if not task.done():
                task.cancel()
            self._pending_tasks.clear()
            self._pending_is_think = False

    def on_summarise(self) -> None:
        """Called when agent is about to give final_answer."""
        self.mock("summarise")
        self.finalize()
    # ------------------------------------------------------------------

    def _schedule(self, coro, *, is_think: bool) -> None:
        # Wenn es ein Think-Task ist, löschen wir ALLES alte (force=True).
        # Wenn es ein Normal-Task ist, löschen wir nur alte Normal-Tasks (force=False).
        self._cancel_pending(force=is_think)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            coro.close()
            return

        task = loop.create_task(coro)

        # Markiere den Task eindeutig!
        task.set_name("think" if is_think else "normal")

        task.add_done_callback(self._on_task_done)
        self._pending_tasks.append(task)
        if is_think:
            self._pending_is_think = True

    def _cancel_pending(self, *, force: bool) -> None:
        surviving = []
        for task in self._pending_tasks:
            if task.done():
                continue

            # force=True -> Alles wird gecancelt.
            # force=False -> NUR Tasks canceln, die NICHT "think" heißen.
            if force or task.get_name() != "think":
                task.cancel()
            else:
                surviving.append(task)

        self._pending_tasks = surviving
        if force:
            self._pending_is_think = False

    def _on_task_done(self, task: asyncio.Task) -> None:
        if task in self._pending_tasks:
            self._pending_tasks.remove(task)

        # Überprüfen, ob noch ein Think-Task existiert
        if not any(t.get_name() == "think" and not t.done() for t in self._pending_tasks):
            self._pending_is_think = False

        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.debug("Narrator task raised: %s", exc)

    # ------------------------------------------------------------------
    # Internal: budget guard
    # ------------------------------------------------------------------

    def _budget_ok(self, history: list[dict]) -> bool:
        """
        Return True if per-minute token budget allows another Blitz call.
        Uses a sliding 60-second window – entries older than 60 s are pruned
        before every check so the limit is rolling, not fixed-interval.
        """
        now = time.monotonic()
        cutoff = now - _BUDGET_WINDOW
        # prune stale entries
        self._token_log = [(t, i, o) for t, i, o in self._token_log if t > cutoff]

        used_in  = sum(i for _, i, _ in self._token_log)
        used_out = sum(o for _, _, o in self._token_log)

        # estimate cost of the upcoming call
        diff = _compress_diff(history, self._history_cursor)
        est_in = _estimate_tokens(diff)

        if used_in + est_in > NARRATOR_MAX_INPUT_TOKENS_PM:
            logger.debug("Narrator: input TPM budget full (%d/%d) – skipping blitz",
                         used_in, NARRATOR_MAX_INPUT_TOKENS_PM)
            return False
        if used_out >= NARRATOR_MAX_OUTPUT_TOKENS_PM:
            logger.debug("Narrator: output TPM budget full (%d/%d) – skipping blitz",
                         used_out, NARRATOR_MAX_OUTPUT_TOKENS_PM)
            return False
        return True

    def _record_tokens(self, tokens_in: int, tokens_out: int) -> None:
        """Log actual token usage from a completed Blitz call."""
        self._token_log.append((time.monotonic(), tokens_in, tokens_out))

    # ------------------------------------------------------------------
    # Internal: blitz call coroutines
    # ------------------------------------------------------------------

    async def _blitz_normal(
        self,
        diff_msgs: list[dict],
        tool_hint: str = "",
    ) -> None:
        self._last_blitz_time = time.monotonic()
        system, messages = _build_normal_prompt(
            self._lang, diff_msgs, self._mini, tool_hint
        )
        with Spinner(message="calling blitz internal", symbols="p"):
            result = await _call_blitz(system, messages)
        if result is None:
            return

        # Stale guard: if thought was refreshed <0.3 s ago by a more recent mock,
        # discard blitz result to avoid overwriting fresher info.
        if time.monotonic() - self._thought_set_time < NARRATOR_STALE_THRESHOLD:
            logger.debug("Narrator: blitz result discarded (stale)")
            return

        self._record_tokens(result["in"], result["out"])

        data = result["data"]
        thought = data.get("t", "")
        if not thought:
            return

        if data.get("d"):
            self._mini.drift = True
        if data.get("r"):
            self._mini.repeat = True

        self._set_thought(self._append_state_flags(thought), moc=False)

    async def _blitz_think(self, thinking_content: str) -> None:
        self._last_blitz_time = time.monotonic()
        system, messages = _build_think_prompt(
            self._lang, thinking_content, self._mini
        )
        with Spinner(message="meta thinking", symbols="i"):
            result = await _call_blitz(system, messages)
        if result is None:
            self.mock("think")
            return

        self._record_tokens(result["in"], result["out"])

        data = result["data"]
        thought = data.get("t", "")
        plan = data.get("plan", "")

        # Update mini-state with new plan if provided
        if plan:
            self._mini.plan_summary = plan
        if data.get("d"):
            self._mini.drift = True
        if data.get("r"):
            self._mini.repeat = True

        if thought:
            self._set_thought(self._append_state_flags(thought), moc=False)
        else:
            self.mock("think")


    # ------------------------------------------------------------------
    # 1. Skills – select relevant skills for current context
    # ------------------------------------------------------------------

    def schedule_skills_update(
        self,
        query: str,
        history: list[dict],
        skills_manager: Any,
        ctx:None=None,
    ) -> None:
        """
        Fire-and-forget: ask Blitz to pick the most relevant skills for the
        current context, then update live.skills.

        Call after on_init() and after schedule_think_result().
        """
        if not self._enabled:
            return
        if not self._budget_ok(history):
            return

        skill_index = _compress_skills_to_index(skills_manager)
        if skill_index == "(no skills)":
            return

        diff = _compress_diff(history, self._history_cursor)
        self._schedule(
            self._blitz_skills(query, diff, skill_index, skills_manager, ctx),
            is_think=False,
        )

    async def _blitz_skills(
        self,
        query: str,
        diff_msgs: list[dict],
        skill_index: str,
        skills_manager: Any,
        ctx: None = None
    ) -> None:
        system, messages = _build_skills_prompt(
            self._lang, query, diff_msgs, skill_index, self._mini
        )
        with Spinner(message="skill select", symbols="a"):
            result = await _call_blitz(system, messages)
        if result is None:
            return

        self._record_tokens(result["in"], result["out"])

        data = result["data"]
        selected_ids: list[str] = data.get("ids", [])
        if not isinstance(selected_ids, list):
            return

        # Validate ids exist in skills_manager
        skills = getattr(skills_manager, "skills", {})
        valid_ids = [sid for sid in selected_ids if sid in skills]

        # Update live state
        valid_names = [skills[sid].name for sid in valid_ids]
        self.live.skills = valid_names
        if ctx:
            ctx.matched_skills = valid_ids

        logger.debug(
            "Narrator: skills updated → %s (reason: %s)",
            valid_ids, data.get("reason", ""),
        )

    # ------------------------------------------------------------------
    # 2. RuleSet – extract situation/intent and activate matching rules
    # ------------------------------------------------------------------

    def schedule_ruleset_update(
        self,
        history: list[dict],
        session: Any,
        ctx: Any,
    ) -> None:
        """
        Fire-and-forget: ask Blitz to extract situation + intent from history,
        update session.rule_set, and write result to VFS.

        Call after on_init() and after substantial tool results.
        """
        if not self._enabled:
            return
        rule_set = getattr(session, "rule_set", None)
        if rule_set is None:
            return
        if not self._budget_ok(history):
            return

        diff = _compress_diff(history, self._history_cursor)
        self._schedule(
            self._blitz_ruleset(diff, session, rule_set, ctx),
            is_think=False,
        )

    async def _blitz_ruleset(
        self,
        diff_msgs: list[dict],
        session: Any,
        rule_set: Any,
        ctx: Any,
    ) -> None:
        # Reiche rule_set an den Builder weiter, damit der LLM die IDs kennt
        system, messages = _build_ruleset_prompt(self._lang, diff_msgs, self._mini, rule_set)

        with Spinner(message="Extracting rules", symbols="+"):
            # MAX_TOKENS ERHÖHT, damit das JSON nicht abgeschnitten wird!
            result = await _call_blitz(system, messages, max_tokens=300)

        if result is None:
            return

        self._record_tokens(result["in"], result["out"])
        data = result["data"]

        # Neue Extrakte
        action = data.get("action", "set_context").strip().lower()
        situation = data.get("situation", "").strip()
        intent = data.get("intent", "").strip()
        confidence = float(data.get("confidence", 0.0))
        rule_id = data.get("rule_id", "").strip()

        if confidence < 0.3:
            logger.debug("Narrator: ruleset action skipped (low confidence %.2f)", confidence)
            return

        # 1. Aktion auf dem rule_set sicher ausführen (mit hasattr Checks)
        try:
            if action == "add_rule" and data.get("instructions"):
                if hasattr(rule_set, "add_rule"):
                    new_rule = rule_set.add_rule(
                        situation=situation,
                        intent=intent,
                        instructions=data.get("instructions", []),
                        required_tool_groups=data.get("tools", []),
                        learned=True,
                        confidence=confidence
                    )
                    logger.debug("Narrator: added new rule %r", getattr(new_rule, "id", "unknown"))
                else:
                    logger.debug("Narrator: add_rule skipped (method not found on RuleSet)")

            elif action == "update_rule" and rule_id:
                if hasattr(rule_set, "update_rule"):
                    updates = {}
                    if situation: updates["situation"] = situation
                    if intent: updates["intent"] = intent
                    if data.get("instructions"): updates["instructions"] = data.get("instructions")
                    if data.get("tools"): updates["required_tool_groups"] = data.get("tools")

                    success = rule_set.update_rule(rule_id, **updates)
                    logger.debug("Narrator: updated rule %r -> %s", rule_id, success)
                else:
                    logger.debug("Narrator: update_rule skipped (method not found on RuleSet)")

            elif action == "remove_rule" and rule_id:
                if hasattr(rule_set, "remove_rule"):
                    success = rule_set.remove_rule(rule_id)
                    logger.debug("Narrator: removed rule %r -> %s", rule_id, success)
                else:
                    logger.debug("Narrator: remove_rule skipped (method not found on RuleSet)")

            # WICHTIG: Den aktiven Zustand (Situation) für den aktuellen Lauf immer setzen
            if situation and intent and hasattr(rule_set, "set_situation"):
                rule_set.set_situation(situation, intent)

        except Exception as exc:
            logger.error("Narrator: rule_set action %r failed: %s", action, exc)
            return

        # 2. VFS SOFORT persistieren
        try:
            if hasattr(rule_set, "build_vfs_content") and hasattr(rule_set, "get_vfs_filename"):
                vfs_content: str = rule_set.build_vfs_content()
                vfs_filename: str = rule_set.get_vfs_filename()
                vfs = getattr(session, "vfs", None)

                if vfs is not None and hasattr(vfs, "set_rules_file"):
                    vfs.set_rules_file(vfs_content)

                    # Optional mark_clean() aufrufen, wenn es existiert
                    if hasattr(rule_set, "mark_clean"):
                        rule_set.mark_clean()

                    logger.debug("Narrator: VFS %r updated (action: %s)", vfs_filename, action)
        except Exception as exc:
            logger.error("Narrator: VFS write failed: %s", exc)

    # ------------------------------------------------------------------
    # 3. Live Memory Extraction
    # ------------------------------------------------------------------

    def schedule_memory_extraction(
        self,
        query: str,
        history: list[dict],
        ctx: Any,
        session: Any,
    ) -> None:
        """
        Fire-and-forget: ask Blitz to extract useful facts from recent history,
        inject working-memory facts into ctx.working_history and persist
        user-level facts via session memory.

        Call after schedule_tool_end() for substantive tool results.
        """
        if not self._enabled:
            return
        if not self._budget_ok(history):
            return
        if time.monotonic() - self._last_blitz_time < NARRATOR_MIN_INTERVAL:
            return

        diff = _compress_diff(history, self._history_cursor)
        if not diff:
            return

        self._schedule(
            self._blitz_memory(query, diff, ctx, session),
            is_think=False,
        )

    async def _blitz_memory(
        self,
        query: str,
        diff_msgs: list[dict],
        ctx: Any,
        session: Any,
    ) -> None:
        self._last_blitz_time = time.monotonic()
        system, messages = _build_memory_prompt(self._lang, query, diff_msgs)
        with Spinner(message="Extracting memorys", symbols="w"):
            result = await _call_blitz(system, messages)
        if result is None:
            return

        self._record_tokens(result["in"], result["out"])

        data = result["data"]
        if not data.get("found", False):
            return

        working_facts: list[str] = data.get("working", [])
        user_facts: list[str] = data.get("user", [])

        # --- Inject working facts into ctx.working_history ---
        if working_facts and hasattr(ctx, "working_history"):
            label = "Extrahierte Kontext-Fakten" if self._lang == "de" else "Extracted Context Facts"
            facts_str = "\n".join(f"- {f}" for f in working_facts+user_facts)
            inject_msg = {
                "role": "system",
                "content": f"[{label}]\n{facts_str}",
            }
            # Insert after the last system message to stay near context start
            insert_at = 0
            for i, m in enumerate(ctx.working_history):
                if m.get("role") == "system":
                    insert_at = i + 1
            ctx.working_history.insert(insert_at, inject_msg)
            logger.debug("Narrator: injected %d working facts", len(working_facts))


# =============================================================================
# NARRATOR EXTENSION: Skills · RuleSet · Live Memory
# =============================================================================
# Each of the three subsystems is self-contained:
#   - schedule_*   → public fire-and-forget entry points (called from Engine)
#   - _blitz_*     → async coroutines that do the actual Blitz call + side-effect
#
# All three respect the shared PM token budget and the cancel/priority rules
# already established for the core narrator tasks.
# =============================================================================

# ---------------------------------------------------------------------------
# System prompts  (kept outside the class so they are importable for tests)
# ---------------------------------------------------------------------------

_SKILLS_SYSTEM_DE = """\
Du bist ein Skill-Selektor für einen KI-Agenten. Deine einzige Aufgabe:
Wähle aus der gegebenen Skill-Liste die 1-3 am besten passenden Skills für den aktuellen Kontext aus.

Regeln:
- Antworte AUSSCHLIESSLICH mit kompaktem JSON, kein Markdown:
  {"ids": ["id1", "id2"], "reason": "<max 8 deutsche Wörter warum>"}
- Wähle NUR Skills die direkt zur aktuellen Aufgabe passen
- Lieber weniger als zu viele
- "ids" darf leer sein wenn kein Skill passt"""

_SKILLS_SYSTEM_EN = """\
You are a skill selector for an AI agent. Your sole task:
Select 1-3 skills from the provided list that best fit the current context.

Rules:
- Reply ONLY with compact JSON, no markdown:
  {"ids": ["id1", "id2"], "reason": "<max 8 English words why>"}
- Select ONLY skills directly relevant to the current task
- Fewer is better than too many
- "ids" may be empty if no skill fits"""

_RULESET_SYSTEM_DE = """\
Du bist ein Regel- & Kontext-Analyzer für einen KI-Agenten. Deine Aufgabe:
1. Erfasse die aktuelle 'situation' und den 'intent' des Agenten.
2. Wenn der User oder Agent eine klare, WIEDERVERWENDBARE Vorgehensweise (Prozedur) formuliert, speichere sie als Regel.

Antworte AUSSCHLIESSLICH mit kompaktem JSON:
{
  "action": "set_context" | "add_rule" | "update_rule" | "remove_rule",
  "situation": "<WO arbeitet der Agent?>",
  "intent": "<WAS will er erreichen?>",
  "confidence": 0.9,
  "rule_id": "<nur bei update/remove aus bestehenden Regeln>",
  "instructions": ["schritt 1", "schritt 2"],
  "tools": ["tool_group1"]
}
Regeln:
- Standard-Aktion ist "set_context".
- "add_rule" NUR nutzen, wenn eine echte Schritt-für-Schritt Prozedur erkennbar ist.
- Keine Spekulationen."""

_RULESET_SYSTEM_EN = """\
You are a Rule & Context Analyzer for an AI agent. Your task:
1. Track the agent's current 'situation' and 'intent'.
2. If the user or agent formulates a clear, REUSABLE procedure, save it as a rule.

Reply ONLY with compact JSON:
{
  "action": "set_context" | "add_rule" | "update_rule" | "remove_rule",
  "situation": "<WHERE is the agent working?>",
  "intent": "<WHAT is the goal?>",
  "confidence": 0.9,
  "rule_id": "<only for update/remove from existing rules>",
  "instructions": ["step 1", "step 2"],
  "tools": ["tool_group1"]
}
Rules:
- Default action is "set_context".
- Use "add_rule" ONLY if a true step-by-step procedure is identifiable.
- No speculation."""

_MEMORY_SYSTEM_DE = """\
Du bist ein Memory-Extraktor für einen KI-Agenten. Deine Aufgabe:
Extrahiere aus dem Agent-Verlauf NUR dauerhaft nützliche Fakten.

Regeln:
- Antworte AUSSCHLIESSLICH mit kompaktem JSON, kein Markdown:
  {"working": ["fakt1", "fakt2"], "user": ["user_fakt1"], "found": true}
- "working": Kurzlebige Fakten relevant für den aktuellen Lauf (z.B. "Datei X liegt in /pfad/Y", "Funktion Z gibt None zurück")
  * Max 3 Einträge, je max 15 Wörter
  * NUR wenn sie in den nächsten Schritten gebraucht werden
- "user": Stabile Fakten ÜBER DEN USER (z.B. "User bevorzugt deutsche Kommentare", "User arbeitet mit Python 3.12")
  * Max 2 Einträge, je max 12 Wörter
  * NUR echte neue Infos, keine Annahmen
- "found": false wenn keine relevanten Fakten vorhanden
- NIEMALS: temporäre Zustände, Zwischen-Ergebnisse, Meinungen"""

_MEMORY_SYSTEM_EN = """\
You are a memory extractor for an AI agent. Your task:
Extract ONLY permanently useful facts from the agent's history.

Rules:
- Reply ONLY with compact JSON, no markdown:
  {"working": ["fact1", "fact2"], "user": ["user_fact1"], "found": true}
- "working": Short-lived facts relevant for the current run (e.g. "File X is at /path/Y", "Function Z returns None")
  * Max 3 entries, max 15 words each
  * ONLY if needed in upcoming steps
- "user": Stable facts ABOUT THE USER (e.g. "User prefers English comments", "User works with Python 3.12")
  * Max 2 entries, max 12 words each
  * ONLY genuine new info, no assumptions
- "found": false if no relevant facts present
- NEVER: temporary states, intermediate results, opinions"""


def _build_skills_prompt(
    lang: str,
    query: str,
    diff_msgs: list[dict],
    skill_index: str,
    mini: "NarratorMiniState",
) -> tuple[str, list[dict]]:
    system = _SKILLS_SYSTEM_DE if lang == "de" else _SKILLS_SYSTEM_EN
    label_q = "Anfrage" if lang == "de" else "Query"
    label_plan = "Plan" if lang == "de" else "Plan"
    label_skills = "Verfügbare Skills" if lang == "de" else "Available Skills"
    label_ctx = "Letzter Kontext" if lang == "de" else "Recent Context"

    parts = [f"{label_q}: {query[:120]}"]
    if mini.plan_summary:
        parts.append(f"{label_plan}: {mini.plan_summary}")
    parts.append(f"{label_skills}:\n{skill_index}")
    if diff_msgs:
        recent = "\n".join(f"[{m['role']}] {m['content'][:80]}" for m in diff_msgs[-3:])
        parts.append(f"{label_ctx}:\n{recent}")

    return system, [{"role": "user", "content": "\n\n".join(parts)}]


def _compress_existing_rules(rule_set: Any) -> str:
    """Erstellt einen ultrakompakten Index existierender Regeln für den Prompt."""
    rules = getattr(rule_set, "situation_rules", {})
    if not rules:
        return "(Keine existierenden Regeln)"

    lines = []
    for rid, r in list(rules.items())[:10]:  # Maximal 10 für Budget-Schutz
        lines.append(f"ID: {rid} | Sit: {r.situation} | Int: {r.intent}")
    return "\n".join(lines)


def _build_ruleset_prompt(
    lang: str,
    diff_msgs: list[dict],
    mini: "NarratorMiniState",
    rule_set: Any,
) -> tuple[str, list[dict]]:
    system = _RULESET_SYSTEM_DE if lang == "de" else _RULESET_SYSTEM_EN

    label_plan = "Bekannter Plan" if lang == "de" else "Known Plan"
    label_rules = "Existierende Regeln" if lang == "de" else "Existing Rules"
    label_ctx = "Agent-Verlauf" if lang == "de" else "Agent History"

    parts = []
    if mini.plan_summary:
        parts.append(f"{label_plan}: {mini.plan_summary}")

    rules_index = _compress_existing_rules(rule_set)
    parts.append(f"{label_rules}:\n{rules_index}")

    if diff_msgs:
        recent = "\n".join(f"[{m['role']}] {m['content'][:150]}" for m in diff_msgs[-5:])
        parts.append(f"{label_ctx}:\n{recent}")

    if not parts:
        parts.append(".")

    return system, [{"role": "user", "content": "\n\n".join(parts)}]


def _build_memory_prompt(
    lang: str,
    query: str,
    diff_msgs: list[dict],
) -> tuple[str, list[dict]]:
    system = _MEMORY_SYSTEM_DE if lang == "de" else _MEMORY_SYSTEM_EN
    label_q = "Ursprüngliche Anfrage" if lang == "de" else "Original Query"
    label_ctx = "Neuer Verlauf" if lang == "de" else "New History"

    parts = [f"{label_q}: {query[:100]}"]
    if diff_msgs:
        recent = "\n".join(f"[{m['role']}] {m['content'][:120]}" for m in diff_msgs[-6:])
        parts.append(f"{label_ctx}:\n{recent}")

    return system, [{"role": "user", "content": "\n\n".join(parts)}]


def _compress_skills_to_index(skills_manager: Any) -> str:
    """
    Convert SkillsManager skills to a compact index string for Blitz prompt.
    Format: "id|name|trigger1,trigger2,trigger3"  (one per line, max 20 skills)
    """
    lines = []
    skills = getattr(skills_manager, "skills", {})
    for skill in list(skills.values()):
        if not getattr(skill, "is_active", lambda: True)():
            continue
        triggers = ",".join(getattr(skill, "triggers", [])[:3])
        lines.append(f"{skill.id}|{skill.name}|{triggers}")
    return "\n".join(lines) if lines else "(no skills)"
