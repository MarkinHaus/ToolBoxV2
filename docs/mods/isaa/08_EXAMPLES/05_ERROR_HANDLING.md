# 05_ERROR_HANDLING - Robust coden

## Problem
Wie baue ich Agenten, die mit Fehlern umgehen koennen?

## Loesung
1. try/except - Fehler abfangen
2. Fallbacks - Alternative Wege
3. Retries - Automatische Wiederholung

## Fehlerkategorien

| Kategorie | Reaktion |
|-----------|----------|
| TOOL_NOT_FOUND | Tool laden, Retry |
| CONNECTION_TIMEOUT | Retry mit Backoff |
| INVALID_INPUT | Validierung |

## Code

    def sicherer_agent(task: str) -> dict:
        try:
            if not task:
                return {"success": False, "error": "Task fehlt", "error_code": "INVALID_INPUT"}
            result = vfs_shell(f"echo '{task}'")
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e), "error_code": "UNKNOWN"}

## Retry mit Backoff

    def retry_with_backoff(func, max_retries=3):
        for i in range(max_retries):
            try:
                return func()
            except:
                if i == max_retries - 1:
                    return {"success": False, "error": "Max retries"}
