"""
Dreamer V3 — Tool Definitions

All dream_* tool schemas for the DreamerAgent.
Grouped by phase for clarity.

Author: FlowAgent V3
"""

from typing import List


# ═══════════════════════════════════════════════════════════════════
# BUDGET CALCULATION
# ═══════════════════════════════════════════════════════════════════

_SUB_AGENT_BASE_BUDGET = 800
_SUB_AGENT_PER_RECORD = 800
_SUB_AGENT_MAX_BUDGET = 8000


def calculate_sub_agent_budget(cluster_size: int) -> int:
    """
    Auto-budget for cluster analysis sub-agents.

    Formula: base + cluster_size * per_record, capped at max.
    """
    budget = _SUB_AGENT_BASE_BUDGET + cluster_size * _SUB_AGENT_PER_RECORD
    return min(budget, _SUB_AGENT_MAX_BUDGET)


# ═══════════════════════════════════════════════════════════════════
# 3.1 DATA ACCESS TOOLS
# ═══════════════════════════════════════════════════════════════════

DREAM_DATA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dream_get_taskmap",
            "description": (
                "Lade Multi-Run-Intel aus der Task Map (/global/.memory/taskmap), "
                "die das Background-Learning nach JEDEM Run schreibt. "
                "Ohne Argumente: Überblick aller task_types/subtypes mit Indexes "
                "(performance, avg_trace_length, improvement_trend, entry_count, is_new). "
                "Mit task_type+subtype: die letzten formatted rows (Tool-Sequenzen, "
                "Fehler-Tools, Resume-Fakten inkl. user_content-Korrekturen, drift/effort) "
                "+ happypath + guid-Status. Klassen mit is_new=true und Runs mit "
                "resume.type=user_content haben PRIORITÄT. Nutze dies als primäre "
                "Datenbasis für die Auswertung über mehrere Runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "description": "z.B. coding (leer = Überblick)"},
                    "subtype": {"type": "string", "description": "z.B. toolbox (default: general)"},
                    "limit": {"type": "integer", "description": "max rows (default 20)", "default": 20}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_get_records",
            "description": (
                "Lade geparste RunRecords aus dem Harvest. "
                "Optional filterbar nach Query-Keywords oder Erfolg/Misserfolg. "
                "Records enthalten: run_id, query, tools_used, success, error_traces, summary. "
                "Nutze ZUERST ohne Filter um einen Überblick zu bekommen, "
                "dann mit Filtern um spezifische Patterns zu untersuchen."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_filter": {
                        "type": "string",
                        "description": "Keyword-Filter für Queries (optional)"
                    },
                    "success_only": {
                        "type": "boolean",
                        "description": "Nur erfolgreiche Runs",
                        "default": False
                    },
                    "failure_only": {
                        "type": "boolean",
                        "description": "Nur fehlgeschlagene Runs",
                        "default": False
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max Anzahl Records (default: 50)",
                        "default": 50
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_get_skills",
            "description": (
                "Lade alle aktuellen Skills mit Stats. "
                "Gibt zurück: id, name, source, confidence, effectiveness, total_uses, "
                "avg_iterations, trigger_count, tool_count, instruction_length, bloat_score, active."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_get_rules",
            "description": (
                "Lade alle aktuellen RuleSet-Regeln und LearnedPatterns. "
                "Gibt SituationRules (id, situation, intent, instructions, confidence, "
                "success/failure counts) und LearnedPatterns zurück."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_get_personas",
            "description": (
                "Lade alle Personas mit Stats. "
                "Gibt zurück: name, source, success_ratio, avg_iterations, "
                "budget_efficiency, total_uses, trigger_keywords/skills."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
]

# ═══════════════════════════════════════════════════════════════════
# 3.2 CLUSTERING TOOLS
# ═══════════════════════════════════════════════════════════════════

DREAM_CLUSTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dream_cluster_records",
            "description": (
                "Clustere RunRecords nach Ähnlichkeit. "
                "Nutzt Embedding-Clustering (wenn verfügbar) oder Keyword-Overlap. "
                "Gibt Cluster-Map zurück: cluster_id → {intent, records, success_count}. "
                "Einzelne Records ohne Cluster-Match landen in 'unclustered_*' Gruppen."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "record_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs der zu clusternden Records (leer = alle)"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Similarity-Threshold (default: 0.65)",
                        "default": 0.65
                    }
                }
            }
        }
    },
]

# ═══════════════════════════════════════════════════════════════════
# 3.3 SKILL EVOLUTION TOOLS
# ═══════════════════════════════════════════════════════════════════

DREAM_SKILL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dream_evolve_skill",
            "description": (
                "Verfeinere einen existierenden Skill basierend auf Cluster-Analyse. "
                "Regeln: Mature Skills (conf≥0.7, predefined) mit <3 Records → nur Metadata. "
                "Mature + ≥3 Records → Merge. Learned/low-conf → Replace erlaubt. "
                "Failure-Patterns werden als BEKANNTE FALLSTRICKE angehängt. "
                "Rollback wird automatisch gespeichert."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "instruction_update": {
                        "type": "string",
                        "description": "Neue/verbesserte Instruction"
                    },
                    "failure_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Erkannte Anti-Patterns"
                    },
                    "new_triggers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Zusätzliche Trigger-Keywords"
                    },
                    "success_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tools aus erfolgreichen Runs"
                    },
                    "cluster_size": {
                        "type": "integer",
                        "description": "Anzahl Records im Cluster (für Evidence-Gate)"
                    },
                    "success_ratio": {
                        "type": "number",
                        "description": "Erfolgsrate im Cluster (0.0-1.0)"
                    }
                },
                "required": ["skill_id", "cluster_size", "success_ratio"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_create_skill",
            "description": (
                "Erstelle einen neuen Skill aus einem Cluster-Pattern. "
                "Nur aufrufen wenn kein existierender Skill zum Cluster passt. "
                "Neuer Skill startet mit confidence=0.3, source='learned'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill-Name (max 50 chars)"},
                    "triggers": {
                        "type": "array", "items": {"type": "string"},
                        "description": "3-5 Trigger-Keywords"
                    },
                    "instruction": {
                        "type": "string",
                        "description": "Nummerierte Anleitung (4-6 Schritte)"
                    },
                    "tools_used": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Relevante Tools"
                    },
                    "failure_patterns": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Bekannte Fallstricke (optional)"
                    }
                },
                "required": ["name", "triggers", "instruction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_merge_skills",
            "description": (
                "Merge zwei duplizierte Skills. "
                "Der Skill mit höherer Confidence bleibt als Primary. "
                "Triggers, Tools und Instructions werden zusammengeführt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "primary_skill_id": {"type": "string"},
                    "secondary_skill_id": {"type": "string"},
                    "merged_instruction": {
                        "type": "string",
                        "description": "Zusammengeführte Instruction (optional, sonst auto)"
                    }
                },
                "required": ["primary_skill_id", "secondary_skill_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_split_skill",
            "description": (
                "Splitte einen bloated Skill in fokussierte Sub-Skills. "
                "Parent wird deaktiviert (nicht gelöscht, für Rollback). "
                "Sub-Skills erben relevante Tools und Confidence*0.8."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "sub_intents": {
                        "type": "array", "items": {"type": "string"},
                        "description": "2-3 fokussierte Sub-Intents"
                    }
                },
                "required": ["skill_id", "sub_intents"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_compress_skill",
            "description": (
                "Komprimiere einen bloated Skill. "
                "Entfernt alte EVOLVED UPDATE Sections, prunt Triggers, cappt Tools. "
                "Nutze bei Bloat >60%."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"}
                },
                "required": ["skill_id"]
            }
        }
    },
]

# ═══════════════════════════════════════════════════════════════════
# 3.4 RULE EXTRACTION TOOLS
# ═══════════════════════════════════════════════════════════════════

DREAM_RULE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dream_extract_rules",
            "description": (
                "Extrahiere SituationRules aus Cluster-Analysen. "
                "Analysiert wiederkehrende Patterns und erstellt Regeln. "
                "Nur erstellen wenn Pattern in ≥2 Clustern auftaucht."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rules": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "situation": {"type": "string"},
                                "intent": {"type": "string"},
                                "instructions": {
                                    "type": "array", "items": {"type": "string"}
                                },
                                "required_tool_groups": {
                                    "type": "array", "items": {"type": "string"}
                                },
                                "confidence": {"type": "number"}
                            },
                            "required": ["situation", "intent", "instructions"]
                        }
                    }
                },
                "required": ["rules"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_learn_pattern",
            "description": (
                "Speichere ein gelerntes Pattern im RuleSet. "
                "Patterns sind kurze Fakten die in ähnlichen Situationen helfen. "
                "Z.B. 'Discord Embeds brauchen: title, description, color (hex)'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "source_situation": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["api", "formatting", "workflow", "error_handling", "general"]
                    },
                    "tags": {
                        "type": "array", "items": {"type": "string"}
                    }
                },
                "required": ["pattern", "source_situation"]
            }
        }
    },
]

# ═══════════════════════════════════════════════════════════════════
# 3.5 PERSONA EVOLUTION TOOLS
# ═══════════════════════════════════════════════════════════════════

DREAM_PERSONA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dream_evolve_persona",
            "description": (
                "Erstelle oder aktualisiere eine Persona basierend auf Analyse. "
                "Nur erstellen wenn: Klarer Intent-Pattern, success_ratio > 0.3, ≥2 Records."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "snake_case, max 30 chars"},
                    "prompt_modifier": {"type": "string"},
                    "model_preference": {
                        "type": "string", "enum": ["fast", "complex"]
                    },
                    "temperature": {"type": "number"},
                    "max_iterations_factor": {"type": "number"},
                    "verification_level": {
                        "type": "string", "enum": ["none", "basic", "strict"]
                    },
                    "dominant_intent": {"type": "string"},
                    "evidence_count": {"type": "integer"}
                },
                "required": ["name", "prompt_modifier", "model_preference"]
            }
        }
    },
]

# ═══════════════════════════════════════════════════════════════════
# 3.6 CLEANUP & PRUNING TOOLS
# ═══════════════════════════════════════════════════════════════════

DREAM_CLEANUP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dream_cleanup_skills",
            "description": (
                "Aktives Aufräumen des Skill-Systems. Führt automatisch durch: "
                "1. LÖSCHEN: Skills mit conf<0.15 UND total_uses≥5. "
                "2. DEAKTIVIEREN: Skills die 3+ Cycles nie gematcht haben. "
                "3. MERGEN: Skills mit identischem Namen oder >80% Trigger-Overlap. "
                "4. COMPRIMIEREN: Skills mit bloat>0.7. "
                "Predefined Skills werden NIE gelöscht."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_cleanup_rules",
            "description": (
                "Aktives Aufräumen des RuleSet-Systems. Führt automatisch durch: "
                "1. LÖSCHEN: Regeln mit conf<0.2. "
                "2. ZUSAMMENFASSEN: Duplizierte Patterns (>80% Wort-Overlap). "
                "3. LÖSCHEN: Regeln die 5+ Cycles nie getroffen haben. "
                "4. LÖSCHEN: LearnedPatterns mit usage_count=0 nach 3+ Cycles. "
                "5. PRUNEN: patterns auf max 50 cappen."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_prune_personas",
            "description": (
                "Entferne schlechte oder ungenutzte Personas. "
                "Kriterien: conf<0.25 nach ≥5 Evidence ODER 0 Usage nach ≥3 Cycles. "
                "Builtins werden NIE gepruned."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_delete_skill",
            "description": (
                "Lösche einen einzelnen Skill explizit. "
                "Nutze wenn Analyse zeigt: kontraproduktiv, veraltet, oder unerkanntes Duplikat. "
                "Predefined Skills können nur deaktiviert werden."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "reason": {
                        "type": "string",
                        "description": "Begründung (wird im Report angezeigt)"
                    }
                },
                "required": ["skill_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_delete_rule",
            "description": (
                "Lösche eine einzelne Regel explizit. "
                "Nutze wenn eine Regel aktiv schadet oder widersprüchlich ist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["rule_id", "reason"]
            }
        }
    },
]

# ═══════════════════════════════════════════════════════════════════
# 3.7 MEMORY & PERSIST TOOLS
# ═══════════════════════════════════════════════════════════════════

DREAM_PERSIST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dream_extract_memories",
            "description": (
                "Extrahiere dauerhafte Fakten aus Cluster-Analysen. "
                "Speichert: 'Task pattern: X → Success approach: Y', 'Known pitfall: Z'. "
                "Nur FAKTEN, keine Vermutungen."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "concepts": {
                                    "type": "array", "items": {"type": "string"}
                                }
                            },
                            "required": ["text", "concepts"]
                        }
                    }
                },
                "required": ["memories"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_write_taskmap_guide",
            "description": (
                "Schreibe/ersetze guid.md für eine Task-Map-Klasse — dein per-Task-Guide "
                "aus der Multi-Run-Analyse. Wird beim nächsten Run als Pre-Context "
                "injiziert (max ~400 Token — halte ihn KOMPAKT: optimale Tool-Route, "
                "bekannte Fallen, framework-spezifischer Weg). Niemals für task_type=new."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string"},
                    "subtype": {"type": "string", "description": "default: general"},
                    "content": {"type": "string", "description": "Markdown-Guide, kompakt"}
                },
                "required": ["task_type", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dream_persist_checkpoint",
            "description": (
                "Persistiere alle Änderungen in VFS. "
                "Speichert: Skills Checkpoint, RuleSet Checkpoint, Persona Store, DreamReport. "
                "MUSS vor final_answer() aufgerufen werden!"
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
]


# ═══════════════════════════════════════════════════════════════════
# REGISTRY HELPERS
# ═══════════════════════════════════════════════════════════════════

def get_all_dream_tool_definitions() -> List[dict]:
    """Get all dream tool definitions as flat list."""
    return (
        DREAM_DATA_TOOLS
        + DREAM_CLUSTER_TOOLS
        + DREAM_SKILL_TOOLS
        + DREAM_RULE_TOOLS
        + DREAM_PERSONA_TOOLS
        + DREAM_CLEANUP_TOOLS
        + DREAM_PERSIST_TOOLS
    )


def get_dream_tool_names() -> List[str]:
    """Get all dream tool names."""
    return [t["function"]["name"] for t in get_all_dream_tool_definitions()]
