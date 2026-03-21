# 04_MEMORY_USAGE - Memory nutzen

## Problem
Wie speichere und rufe ich Wissen zwischen Sessions ab?

## Loesung
1. memory_save - Fakten permanent speichern
2. memory_recall - Gespeichertes Wissen abrufen
3. memory_analyse - Zusammenhaenge erkennen

## Speichern

    memory_save(
        key="arch_db_choice",
        value={"entscheidung": "PostgreSQL statt MongoDB"},
        tags=["architektur", "database"]
    )

## Abfragen

    result = memory_recall(query="Database-Entscheidung")

## Output

    {"success": true, "results": [{"key": "arch_db_choice", "score": 0.95}]}
