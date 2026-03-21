# 01_BASIC_AGENT - Einfachster Agent

## Problem
Wie erstelle ich einen minimalen, funktionsfaehigen ISAA-Agenten?

## Loesung
Ein ISAA-Agent benoetigt:
1. Identity - Klare Rolle und Faehigkeiten
2. Tool-Loading - Mindestens eine Action-Faehigkeit
3. Final-Answer - Abschluss-Kontrakt

## Code

    IDENTITY = "Ich bin ein Agent"
    TOOLS = ["vfs", "filesystem"]
    
    def execute_task(task: str) -> dict:
        plan = think(f"Wie loese ich: {task}")
        if not hat_tools():
            load_tools(TOOLS)
        result = vfs_shell(f"echo '{task}'")
        return final_answer(answer=result, success=True)

## Output

    {"stdout": "...", "success": true}

## Schnell-Referenz

| Komponente | Pflicht |
|------------|---------|
| IDENTITY | X |
| Tools | X |
| think() | - |
| final_answer() | X |
