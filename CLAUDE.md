# Arbeitsregeln für ToolBoxV2

Diese Regeln hat Markin für jede Arbeit an diesem Projekt festgelegt.

1. **Vorhandenen Code finden, verstehen und verwenden.** Was gebraucht wird, gibt es in 9 von 10 Fällen schon (z. B. Wiederholen/Backoff/Fallback im `IntelligentRateLimiter`, Logging über `get_logger()`). Vor jeder neuen Funktion im Code suchen; Vorhandenes erweitern statt etwas Paralleles bauen.
2. **Was vorher schon kaputt war, im selben Zug mit beheben.** Rote Tests oder Fehler, auf die man bei der Arbeit stößt, nicht als „war schon so“ stehen lassen. Nur fragen, wenn der Eingriff zu groß ist, Informationen fehlen, die nicht im Code stehen, oder der Code sich widerspricht.
3. **Tests prüfen den echten Code.** Kein Nachbau der zu testenden Klasse (z. B. ein `SimpleNamespace` als `self`): der driftet vom Code weg. Ersetzt werden nur äußere Grenzen (Netz, LLM, Dateisystem außerhalb des Tests).
4. **Ein Logging-System:** `from toolboxv2 import get_logger`. Keine eigenen Log-Dateien oder Logger-Systeme in Modulen.
