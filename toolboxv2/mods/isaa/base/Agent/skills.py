"""
Skills System - Learned behavioral patterns for FlowAgent

Replaces the overengineered RuleSet with a simpler, more effective system.

Features:
- Skill learning from successful runs (Auto + Confidence Threshold)
- Hybrid matching (Keyword first, Embedding fallback via mem.get_embeddings)
- Confidence-based activation
- Tool relevance scoring (Keyword Overlap)
- Skill sharing between agents (Explicit)
- ToolGroups migration from rule_set.py

Author: FlowAgent V3
"""

import json
import uuid
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Any, Dict, Tuple, Callable


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Skill:
    """
    Learned or predefined behavioral pattern.

    Core concept: Verbally tell the model HOW to do things,
    learned from successful runs or predefined for common patterns.
    """
    id: str
    name: str
    triggers: List[str]  # Keywords für Matching
    instruction: str  # Nummerierte Anleitung (das Herzstück)

    # Tool Context
    tools_used: List[str] = field(default_factory=list)
    tool_groups: List[str] = field(default_factory=list)

    # Learning Metadata
    source: str = "predefined"  # "predefined" | "learned" | "imported"
    confidence: float = 1.0  # 0.0-1.0
    activation_threshold: float = 0.6  # Wird erst aktiv wenn confidence >= threshold
    success_count: int = 0
    failure_count: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None

    def is_active(self) -> bool:
        """Skill ist aktiv wenn confidence >= threshold"""
        return self.confidence >= self.activation_threshold

    def matches_keywords(self, query: str) -> bool:
        """Keyword-basiertes Matching"""
        query_lower = query.lower()
        return any(trigger.lower() in query_lower for trigger in self.triggers)

    def record_usage(self, success: bool):
        """Lernen aus Verwendung"""
        self.last_used = datetime.now()
        if success:
            self.success_count += 1
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.failure_count += 1
            self.confidence = max(0.1, self.confidence - 0.15)

    def to_dict(self) -> dict:
        """Serialize for checkpoint"""
        return {
            'id': self.id,
            'name': self.name,
            'triggers': self.triggers,
            'instruction': self.instruction,
            'tools_used': self.tools_used,
            'tool_groups': self.tool_groups,
            'source': self.source,
            'confidence': self.confidence,
            'activation_threshold': self.activation_threshold,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Skill':
        """Deserialize from checkpoint"""
        skill = cls(
            id=data['id'],
            name=data['name'],
            triggers=data['triggers'],
            instruction=data['instruction'],
            tools_used=data.get('tools_used', []),
            tool_groups=data.get('tool_groups', []),
            source=data.get('source', 'predefined'),
            confidence=data.get('confidence', 1.0),
            activation_threshold=data.get('activation_threshold', 0.6),
            success_count=data.get('success_count', 0),
            failure_count=data.get('failure_count', 0)
        )
        if data.get('created_at'):
            skill.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('last_used'):
            skill.last_used = datetime.fromisoformat(data['last_used'])
        return skill


@dataclass
class ToolGroup:
    """
    Groups multiple tools under a single display name.
    Migrated from rule_set.py - kept because it's useful.

    Instead of showing 50 Discord tools, show "discord_tools: Discord Server APIs"
    """
    name: str  # "discord_tools"
    display_name: str  # "Discord Server APIs"
    description: str  # Short description
    tool_names: List[str]  # Actual tool names in registry
    trigger_keywords: List[str]  # ["discord", "server", "bot"]
    priority: int = 5  # Sorting priority (1=highest)
    icon: str = "🔧"
    auto_generated: bool = False

    def matches_query(self, query: str) -> bool:
        """Check if query matches this group's keywords"""
        query_lower = query.lower()
        return any(kw.lower() in query_lower for kw in self.trigger_keywords)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'tool_names': self.tool_names,
            'trigger_keywords': self.trigger_keywords,
            'priority': self.priority,
            'icon': self.icon,
            'auto_generated': self.auto_generated
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ToolGroup':
        return cls(**data)


# =============================================================================
# SKILLS MANAGER
# =============================================================================

class SkillsManager:
    """
    Manages skills for a FlowAgent instance.

    Features:
    - Hybrid matching (Keyword first, Embedding fallback)
    - Confidence-based activation
    - Tool relevance scoring (Keyword Overlap)
    - Skill sharing (Explicit)
    - ToolGroups management
    """

    def __init__(self, agent_name: str, memory_instance: Any = None):
        """
        Initialize SkillsManager.

        Args:
            agent_name: Name of parent agent
            memory_instance: Memory instance with get_embeddings() method
        """
        self.agent_name = agent_name
        self._memory = memory_instance

        # Skills Storage
        self.skills: Dict[str, Skill] = {}

        # Tool Groups (migrated from rule_set.py)
        self.tool_groups: Dict[str, ToolGroup] = {}

        # Embedding Cache for Skills (lazy loaded)
        self._skill_embeddings_dirty = True
        self._skill_embeddings_cache: Dict[str, np.ndarray] = {}

        # Initialize predefined skills
        self._init_predefined_skills()

    def _init_predefined_skills(self):
        """Initialize predefined skills focused on user management and interaction"""
        predefined = [
            # === USER PREFERENCE SKILLS ===
            Skill(
                id="user_preference_save",
                name="User Preference Save",
                triggers=[
                    "merke", "speicher", "remember", "präferenz", "preference",
                    "mag ich", "bevorzuge", "ich will dass du", "vergiss nicht"
                ],
                instruction="""Für das Speichern von User Präferenzen:
1. Identifiziere die konkrete Präferenz aus der Anfrage
2. Formuliere sie als klares Key-Value Paar
3. Nutze think() um die Präferenz zu strukturieren
4. Speichere mit memory_inject oder geeignetem Tool
5. Bestätige dem User WAS GENAU gespeichert wurde
6. Frage nach falls die Präferenz unklar ist - rate NICHT""",
                tools_used=["think", "memory_inject", "final_answer"],
                tool_groups=["memory"],
                source="predefined"
            ),
            Skill(
                id="user_preference_recall",
                name="User Preference Recall",
                triggers=[
                    "was mag ich", "meine präferenzen", "erinnerst du",
                    "weißt du noch", "meine vorlieben", "was weißt du über mich",
                    "kennst du mich"
                ],
                instruction="""Für das Abrufen von User Präferenzen:
1. Durchsuche den Kontext nach relevanten gespeicherten Präferenzen
2. Falls nicht im Kontext: Nutze memory_query Tool
3. Präsentiere gefundene Präferenzen klar strukturiert
4. Sei EHRLICH: Falls keine gefunden, sage das klar
5. Biete an, Präferenzen zu speichern wenn gewünscht""",
                tools_used=["memory_query", "final_answer"],
                tool_groups=["memory"],
                source="predefined"
            ),
            Skill(
                id="user_context_update",
                name="User Context Update",
                triggers=[
                    "ich bin jetzt", "ich habe gewechselt", "neu bei mir",
                    "update", "aktualisiere", "ändere mein"
                ],
                instruction="""Für das Aktualisieren von User-Kontext:
1. Identifiziere was sich geändert hat (alt → neu)
2. Bestätige das Verständnis mit dem User
3. Aktualisiere die relevanten Einträge
4. Zeige was aktualisiert wurde
5. Frage ob weitere Updates nötig sind""",
                tools_used=["think", "memory_query", "memory_inject", "final_answer"],
                tool_groups=["memory"],
                source="predefined"
            ),

            # === HABITS SKILLS ===
            Skill(
                id="habits_tracking",
                name="Habits Tracking",
                triggers=[
                    "gewohnheit", "habit", "täglich", "routine", "tracking",
                    "streak", "ich habe heute", "erledigt", "done"
                ],
                instruction="""Für Gewohnheits-Tracking:
1. Identifiziere welche Gewohnheit getrackt werden soll
2. Prüfe ob bereits Daten existieren (memory_query)
3. Für neuen Eintrag: Speichere mit Datum und Status
4. Für Abfrage: Zeige Historie und Streak an
5. Motiviere bei Erfolgen, ermutige bei Lücken
6. Schlage Verbesserungen vor wenn Pattern erkennbar""",
                tools_used=["think", "memory_query", "memory_inject", "final_answer"],
                tool_groups=["memory"],
                source="predefined"
            ),
            Skill(
                id="habits_analysis",
                name="Habits Analysis",
                triggers=[
                    "analyse gewohnheit", "habit statistik", "wie läuft",
                    "fortschritt", "mein streak", "gewohnheit übersicht"
                ],
                instruction="""Für Gewohnheits-Analyse:
1. Lade alle Daten zur angefragten Gewohnheit
2. Berechne: Erfolgsrate, längster Streak, aktueller Streak
3. Nutze think() für Muster-Analyse (Wochentage, Zeiten)
4. Gib konstruktives, ehrliches Feedback
5. Schlage KONKRETE Verbesserungen vor
6. Sei motivierend aber realistisch""",
                tools_used=["think", "memory_query", "final_answer"],
                tool_groups=["memory"],
                source="predefined"
            ),
            Skill(
                id="habits_setup",
                name="Habits Setup",
                triggers=[
                    "neue gewohnheit", "habit erstellen", "will anfangen",
                    "möchte tracken", "routine einrichten"
                ],
                instruction="""Für das Einrichten neuer Gewohnheiten:
1. Frage nach: Was genau? Wie oft? Wann?
2. Definiere klare, messbare Kriterien
3. Erstelle initiale Tracking-Struktur
4. Erkläre wie der User Fortschritte melden kann
5. Setze realistische Erwartungen
6. Biete Reminder-Optionen an wenn verfügbar""",
                tools_used=["think", "memory_inject", "final_answer"],
                tool_groups=["memory"],
                source="predefined"
            ),

            # === MULTI-STEP TASK ===
            Skill(
                id="multi_step_task",
                name="Multi-Step Task Planning",
                triggers=[
                    "mehrere schritte", "komplex", "projekt", "plan",
                    "umfangreich", "aufgabe mit", "großes vorhaben"
                ],
                instruction="""Für komplexe Multi-Step Aufgaben:
1. ZUERST: Nutze think() um einen Plan zu erstellen
2. Liste alle benötigten Schritte auf (max 5-7)
3. Identifiziere welche Tools du brauchen wirst
4. Lade benötigte Tools VOR dem Start
5. Führe Schritte SEQUENTIELL aus
6. Nach jedem Schritt: Kurz verifizieren ob erfolgreich
7. Bei Fehler: STOPPEN und User informieren
8. Am Ende: Zusammenfassung was getan wurde""",
                tools_used=["think", "list_tools", "load_tools", "final_answer"],
                tool_groups=[],
                source="predefined"
            ),
            Skill(
                id="autonomous_execution",
                name="Autonomous Task Execution",
                triggers=["analyze", "create", "check", "files", "project", "fix", "solve"],
                instruction="""PROTOCOL FOR COMPLEX TASKS:
        1. ANALYSIS: Use `think()` to map out dependencies. What files/info do I need?
        2. RECON: Use `vfs_list` or `list_tools` to understand the environment.
        3. EXECUTION:
           - If tools are missing: `load_tools(["category"])`
           - If files are missing: Create them or ask for content.
        4. PERSISTENCE: Save intermediate results to VFS (`vfs_write`) if the task is long.
        5. COMPLETION: Present the result definitively.""",
                tools_used=["think", "vfs_list", "load_tools", "final_answer"],
                tool_groups=["vfs", "system"],
                source="predefined"
            ),
            # === INTERACTION SKILLS ===
            Skill(
                id="clarification_needed",
                name="Clarification Request",
                triggers=[
                    "unklar", "was meinst du", "verstehe nicht",
                    "kannst du erklären", "mehr details"
                ],
                instruction="""Wenn Klarstellung nötig ist:
1. Identifiziere WAS genau unklar ist
2. Formuliere SPEZIFISCHE Fragen (nicht "was meinst du?")
3. Biete Optionen an wenn möglich ("Meinst du A oder B?")
4. Erkläre warum die Info wichtig ist
5. Warte auf Antwort bevor du fortfährst
6. NIEMALS raten bei wichtigen Details""",
                tools_used=["think", "final_answer"],
                tool_groups=[],
                source="predefined"
            ),
            Skill(
                id="error_recovery",
                name="Error Recovery",
                triggers=[
                    "fehler", "funktioniert nicht", "problem", "geht nicht",
                    "error", "kaputt", "falsch"
                ],
                instruction="""Für Fehlerbehandlung:
1. Analysiere den Fehler genau (was ging schief?)
2. Nutze think() um Ursachen zu identifizieren
3. Prüfe ob es ein Tool-Fehler oder Logik-Fehler ist
4. Bei Tool-Fehler: Versuche alternativen Ansatz
5. Bei unklarer Ursache: Frage User nach mehr Kontext
6. Sei EHRLICH wenn du den Fehler nicht beheben kannst
7. Schlage nächste Schritte vor""",
                tools_used=["think", "final_answer"],
                tool_groups=[],
                source="predefined"
            ),
            # === VFS PERSISTENCE SKILLS ===

            Skill(
                id="vfs_info_persist",
                name="Information Persistence",
                triggers=[
                    "merke dir", "speicher das", "wichtig", "behalten",
                    "notiz", "note", "remember", "für später",
                    "zusammenfassung", "ergebnis", "resultat"
                ],
                instruction="""Wenn Informationen persistent gespeichert werden sollen:
        1. Identifiziere WAS gespeichert werden soll (Fakten, Ergebnisse, Notizen)
        2. Wähle passenden Speicherort:
           - /info/[thema].md für thematische Sammlungen
           - /[aufgabe]_result.md für Aufgaben-Ergebnisse
           - /notes.md für schnelle Notizen
        3. Strukturiere den Inhalt klar (Überschriften, Listen)
        4. Nutze vfs_write() um zu speichern
        5. Bestätige dem User WO und WAS gespeichert wurde
        6. Bei Updates: Erst vfs_read(), dann ergänzen, dann vfs_write()""",
                tools_used=["think", "vfs_read", "vfs_write", "vfs_list", "final_answer"],
                tool_groups=["vfs"],
                source="predefined"
            ),

            Skill(
                id="vfs_task_planning",
                name="Task Planning with Persistence",
                triggers=[
                    "plane", "projekt", "aufgabe", "task", "todo",
                    "schritte", "workflow", "prozess", "ablauf",
                    "großes vorhaben", "mehrteilig"
                ],
                instruction="""Für komplexe Aufgaben mit Planung:
        1. ZUERST: Nutze think() um die Aufgabe zu analysieren
        2. Erstelle Plan in /plan.md oder /[aufgabe]_plan.md:
           - Ziel der Aufgabe
           - Geschätzte Schritte (nummeriert)
           - Benötigte Tools/Ressourcen
           - Erfolgskriterien
        3. Arbeite Schritte ab und UPDATE den Plan:
           - [x] Erledigt
           - [ ] Offen
           - [!] Problem
        4. Zwischenergebnisse in /[aufgabe]/[schritt].md speichern
        5. Finale Zusammenfassung in /[aufgabe]_result.md
        6. Bei Unterbrechung: Plan zeigt wo weitermachen""",
                tools_used=["think", "vfs_write", "vfs_read", "vfs_mkdir", "vfs_list", "final_answer"],
                tool_groups=["vfs"],
                source="predefined"
            ),

            Skill(
                id="vfs_knowledge_base",
                name="Knowledge Base Management",
                triggers=[
                    "wissen", "knowledge", "dokumentation", "docs",
                    "sammlung", "archiv", "referenz", "nachschlagen",
                    "was weiß ich über", "zeig mir alles zu"
                ],
                instruction="""Für Wissensmanagement im VFS:
        1. Struktur der Knowledge Base:
           /info/
           ├── [thema1].md
           ├── [thema2].md
           └── index.md (Übersicht aller Themen)
        2. Bei neuen Informationen:
           - Prüfe ob /info/[thema].md existiert (vfs_list)
           - Existiert: Lesen, ergänzen, schreiben
           - Neu: Erstellen mit klarer Struktur
           - index.md updaten
        3. Bei Abfragen:
           - Erst /info/ durchsuchen
           - Relevante Dateien lesen
           - Konsolidierte Antwort geben
        4. Format in .md Dateien:
           - Klare Überschriften
           - Datum der letzten Änderung
           - Quellen wenn vorhanden
        5. Regelmäßig aufräumen: Veraltetes markieren oder löschen""",
                tools_used=["think", "vfs_read", "vfs_write", "vfs_list", "vfs_mkdir", "final_answer"],
                tool_groups=["vfs"],
                source="predefined"
            ),
            Skill(
                id="parallel_subtasks",
                name="Parallel Sub-Task Execution",
                triggers=[
                    "parallel", "gleichzeitig", "recherchiere mehrere",
                    "vergleiche", "sammle von verschiedenen", "und dann zusammen"
                ],
                instruction="""Für parallelisierbare Aufgaben:
        1. Identifiziere UNABHÄNGIGE Teilaufgaben (die nicht aufeinander warten)
        2. Für jede Teilaufgabe: spawn_sub_agent(task=..., output_dir=/sub/[name], wait=False)
        3. Wenn alle gestartet: wait_for([alle_ids])
        4. Lese Ergebnisse aus /sub/[name]/result.md
        5. Kombiniere/Vergleiche die Ergebnisse
        6. WICHTIG: Sub-Agents können NUR in ihren output_dir schreiben
        7. WICHTIG: Gib Sub-Agents klare, spezifische Tasks - sie können NICHT zurückfragen""",
                tools_used=["think", "spawn_sub_agent", "wait_for", "vfs_read", "final_answer"],
                tool_groups=["vfs"],
                source="predefined"
            )
        ]

        for skill in predefined:
            self.skills[skill.id] = skill

    # =========================================================================
    # SKILL MATCHING (Hybrid: Keyword → Embedding)
    # =========================================================================

    def match_skills(self, query: str, max_results: int = 3) -> List[Skill]:
        """
        Hybrid Matching: Keyword first, Embedding fallback.
        Returns matched skills sorted by relevance.
        """
        matched: List[Tuple[Skill, float]] = []

        # Phase 1: Keyword Matching (fast, predictable)
        for skill in self.skills.values():
            if not skill.is_active():
                continue
            if skill.matches_keywords(query):
                matched.append((skill, 1.0))  # Score 1.0 für keyword match

        # Phase 2: Embedding Fallback (wenn weniger als max_results gefunden)
        if len(matched) < max_results and self._memory:
            try:
                import asyncio
                # Check if we're in an async context
                try:
                    loop = asyncio.get_running_loop()
                    # We're in async context, need to handle differently
                    # For now, skip embedding matching in sync context
                except RuntimeError:
                    # No running loop, can't do async embedding match
                    pass
            except Exception:
                pass

        # Sort by score, then by confidence
        matched.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)

        return [skill for skill, _ in matched[:max_results]]

    async def match_skills_async(self, query: str, max_results: int = 3) -> List[Skill]:
        """
        Async version of match_skills with embedding fallback.
        """
        matched: List[Tuple[Skill, float]] = []

        # Phase 1: Keyword Matching
        for skill in self.skills.values():
            if not skill.is_active():
                continue
            if skill.matches_keywords(query):
                matched.append((skill, 1.0))

        # Phase 2: Embedding Fallback
        if len(matched) < max_results and self._memory:
            embedding_matches = await self._match_by_embedding(
                query,
                max_results - len(matched)
            )
            for skill, score in embedding_matches:
                if skill.id not in [s.id for s, _ in matched]:
                    matched.append((skill, score))

        matched.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)
        return [skill for skill, _ in matched[:max_results]]

    async def _match_by_embedding(
        self,
        query: str,
        max_results: int
    ) -> List[Tuple[Skill, float]]:
        """Embedding-basiertes Matching mit mem.get_embeddings()"""
        if not self._memory:
            return []

        try:
            # Query Embedding
            query_embedding = await self._memory.get_embeddings([query])

            # Ensure skill embeddings are cached
            await self._ensure_skill_embeddings()

            # Search in skill embeddings
            results = []
            for skill_id, embedding in self._skill_embeddings_cache.items():
                if skill_id not in self.skills:
                    continue
                skill = self.skills[skill_id]
                if not skill.is_active():
                    continue

                # Cosine similarity
                similarity = self._cosine_similarity(query_embedding[0], embedding)
                if similarity >= 0.5:  # Threshold
                    results.append((skill, float(similarity)))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_results]

        except Exception as e:
            print(f"[SkillsManager] Embedding match failed: {e}")
            return []

    async def _ensure_skill_embeddings(self):
        """Lazy load/update skill embeddings"""
        if not self._skill_embeddings_dirty:
            return

        if not self._memory:
            return

        try:
            texts = []
            skill_ids = []

            for skill_id, skill in self.skills.items():
                if skill.is_active():
                    # Combine triggers and instruction for embedding
                    text = f"{skill.name} {' '.join(skill.triggers)} {skill.instruction[:200]}"
                    texts.append(text)
                    skill_ids.append(skill_id)

            if texts:
                embeddings = await self._memory.get_embeddings(texts)
                for i, skill_id in enumerate(skill_ids):
                    self._skill_embeddings_cache[skill_id] = embeddings[i]

            self._skill_embeddings_dirty = False

        except Exception as e:
            print(f"[SkillsManager] Failed to create skill embeddings: {e}")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # =========================================================================
    # TOOL RELEVANCE SCORING (Keyword Overlap)
    # =========================================================================

    def score_tool_relevance(
        self,
        query: str,
        tool_name: str,
        tool_description: str
    ) -> float:
        """
        Calculate keyword-based relevance score for a tool.
        Called once at query start, results cached in ExecutionEngine.
        """
        query_words = set(query.lower().split())
        desc_words = set(tool_description.lower().split())
        name_words = set(tool_name.lower().replace('_', ' ').split())

        # Remove common stop words
        stop_words = {'der', 'die', 'das', 'und', 'oder', 'ein', 'eine', 'the', 'a', 'an', 'is', 'are', 'to', 'for'}
        query_words -= stop_words
        desc_words -= stop_words

        # Overlap berechnen
        query_desc_overlap = len(query_words & desc_words)
        query_name_overlap = len(query_words & name_words)

        # Score: Name match ist wichtiger als Description match
        score = 0.0
        if query_name_overlap > 0:
            score += 0.5 * min(1.0, query_name_overlap / max(1, len(name_words)))
        if query_desc_overlap > 0:
            score += 0.3 * min(1.0, query_desc_overlap / max(1, len(desc_words) / 3))

        # Bonus wenn Tool in matched Skills vorkommt
        matched_skills = self.match_skills(query, max_results=2)
        for skill in matched_skills:
            if tool_name in skill.tools_used:
                score += 0.3
                break

        return min(1.0, score)

    def get_relevant_tools_from_groups(
        self,
        query: str,
        tool_groups: List[str],
        tool_manager: Any,
        max_tools: int
    ) -> List[Tuple[str, float]]:
        """
        Get most relevant tools from given groups, sorted by relevance score.

        Returns: List of (tool_name, relevance_score) tuples
        """
        scored_tools = []

        for group_name in tool_groups:
            group = self.tool_groups.get(group_name)
            if not group:
                continue

            for tool_name in group.tool_names:
                tool_entry = tool_manager.get(tool_name)
                if tool_entry:
                    score = self.score_tool_relevance(
                        query,
                        tool_name,
                        tool_entry.description or ""
                    )
                    scored_tools.append((tool_name, score))

        # Sort by score descending
        scored_tools.sort(key=lambda x: x[1], reverse=True)

        return scored_tools[:max_tools]

    # =========================================================================
    # SKILL LEARNING
    # =========================================================================

    async def learn_from_run(
        self,
        query: str,
        tools_used: List[str],
        final_answer: str,
        success: bool,
        llm_completion_func: Callable
    ) -> Optional[Skill]:
        """
        Learn a new skill from a successful run.
        Uses LLM (fast model) to generate instruction.

        Returns: New skill if created, None otherwise
        """
        if not success:
            return None

        # Mindestens 2 Tools verwendet?
        if len(tools_used) < 2:
            return None

        # Filter out meta-tools
        meaningful_tools = [t for t in tools_used if t not in ['think', 'final_answer', 'list_tools', 'load_tools']]
        if len(meaningful_tools) < 1:
            return None

        # Prüfen ob ähnlicher Skill existiert (keyword match)
        existing = self.match_skills(query, max_results=1)
        if existing and existing[0].matches_keywords(query):
            # Existing skill - nur confidence erhöhen
            existing[0].record_usage(True)
            return None

        # LLM generiert Instruction
        try:
            extraction_prompt = f"""Analysiere diesen erfolgreichen Ablauf und erstelle eine wiederverwendbare Anleitung.

USER ANFRAGE: {query}

TOOLS VERWENDET (in Reihenfolge): {', '.join(tools_used)}

ERGEBNIS: {final_answer[:500]}

Erstelle:
1. Einen kurzen Namen für diesen Skill (max 5 Wörter, deutsch oder englisch)
2. 3-5 Trigger-Keywords die ähnliche Anfragen erkennen würden
3. Eine nummerierte Anleitung (4-6 Schritte) die beschreibt wie man ähnliche Aufgaben löst

Format (EXAKT so):
NAME: [skill name]
TRIGGERS: [keyword1, keyword2, keyword3]
ANLEITUNG:
1. [Schritt 1]
2. [Schritt 2]
3. [Schritt 3]
..."""

            response = await llm_completion_func(
                messages=[{"role": "user", "content": extraction_prompt}],
                model_preference="fast",
                max_tokens=300,
                temperature=0.3,
                with_context=False
            )

            # Parse response
            skill = self._parse_skill_from_llm(response, tools_used, query)
            if skill:
                skill.confidence = 0.3  # Start niedrig (threshold ist 0.6)
                self.skills[skill.id] = skill
                self._skill_embeddings_dirty = True
                print(f"[SkillsManager] Learned new skill: {skill.name} (confidence: {skill.confidence})")
                return skill

        except Exception as e:
            print(f"[SkillsManager] Failed to learn skill: {e}")

        return None

    def _parse_skill_from_llm(
        self,
        response: str,
        tools_used: List[str],
        original_query: str
    ) -> Optional[Skill]:
        """Parse LLM response into Skill object"""
        try:
            lines = response.strip().split('\n')

            name = ""
            triggers = []
            instruction_lines = []
            in_instruction = False

            for line in lines:
                line = line.strip()
                if line.upper().startswith("NAME:"):
                    name = line[5:].strip()
                elif line.upper().startswith("TRIGGERS:"):
                    triggers_str = line[9:].strip().strip("[]")
                    triggers = [t.strip().strip('"\'') for t in triggers_str.split(',')]
                elif line.upper().startswith("ANLEITUNG:"):
                    in_instruction = True
                elif in_instruction and line:
                    instruction_lines.append(line)

            if not name or not triggers or not instruction_lines:
                return None

            # Generiere ID
            skill_id = f"learned_{name.lower().replace(' ', '_')[:20]}_{uuid.uuid4().hex[:6]}"

            return Skill(
                id=skill_id,
                name=name,
                triggers=triggers,
                instruction='\n'.join(instruction_lines),
                tools_used=list(set(tools_used)),
                tool_groups=[],
                source="learned",
                confidence=0.3
            )

        except Exception as e:
            print(f"[SkillsManager] Failed to parse skill: {e}")
            return None

    def record_skill_usage(self, skill_id: str, success: bool):
        """Record skill usage for learning"""
        if skill_id in self.skills:
            self.skills[skill_id].record_usage(success)
            self._skill_embeddings_dirty = True  # Confidence changed

    # =========================================================================
    # TOOL GROUPS (migrated from rule_set.py)
    # =========================================================================

    def register_tool_group(
        self,
        name: str,
        display_name: str,
        tool_names: List[str],
        trigger_keywords: List[str],
        description: str = "",
        priority: int = 5,
        icon: str = "🔧",
        auto_generated: bool = False
    ):
        """Register a tool group"""
        self.tool_groups[name] = ToolGroup(
            name=name,
            display_name=display_name,
            description=description,
            tool_names=tool_names,
            trigger_keywords=trigger_keywords,
            priority=priority,
            icon=icon,
            auto_generated=auto_generated
        )

    def get_matching_tool_groups(self, query: str) -> List[ToolGroup]:
        """Get tool groups that match the query"""
        matched = []
        for group in self.tool_groups.values():
            if group.matches_query(query):
                matched.append(group)
        matched.sort(key=lambda g: g.priority)
        return matched

    def get_tool_group(self, name: str) -> Optional[ToolGroup]:
        """Get tool group by name"""
        return self.tool_groups.get(name)

    def list_tool_groups(self) -> List[str]:
        """List all tool group names"""
        return list(self.tool_groups.keys())

    def get_categories_from_groups(self) -> List[str]:
        """Get unique categories from all tool groups"""
        categories = set()
        for group in self.tool_groups.values():
            # Extract category from group name (e.g., "discord_tools" -> "discord")
            category = group.name.replace("_tools", "")
            categories.add(category)
        return sorted(list(categories))

    # =========================================================================
    # SKILL SHARING (Explicit)
    # =========================================================================

    def export_skill(self, skill_id: str) -> Optional[dict]:
        """Export a skill for sharing"""
        if skill_id in self.skills:
            return self.skills[skill_id].to_dict()
        return None

    def import_skill(self, skill_data: dict, overwrite: bool = False) -> bool:
        """Import a skill from another agent"""
        try:
            skill = Skill.from_dict(skill_data)

            if skill.id in self.skills and not overwrite:
                return False

            # Mark as imported
            skill.source = "imported"
            self.skills[skill.id] = skill
            self._skill_embeddings_dirty = True
            return True

        except Exception as e:
            print(f"[SkillsManager] Failed to import skill: {e}")
            return False

    def share_skill_with(
        self,
        skill_id: str,
        target_skills_manager: 'SkillsManager'
    ) -> bool:
        """Share a skill with another SkillsManager (via BindManager)"""
        skill_data = self.export_skill(skill_id)
        if skill_data:
            return target_skills_manager.import_skill(skill_data)
        return False

    def list_shareable_skills(self) -> List[dict]:
        """List skills that can be shared (active and high confidence)"""
        shareable = []
        for skill in self.skills.values():
            if skill.is_active() and skill.confidence >= 0.7:
                shareable.append({
                    'id': skill.id,
                    'name': skill.name,
                    'confidence': skill.confidence,
                    'success_count': skill.success_count,
                    'source': skill.source
                })
        return shareable

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_checkpoint(self) -> dict:
        """Serialize for checkpoint"""
        return {
            'agent_name': self.agent_name,
            'skills': {sid: s.to_dict() for sid, s in self.skills.items()},
            'tool_groups': {gid: g.to_dict() for gid, g in self.tool_groups.items()}
        }

    def from_checkpoint(self, data: dict):
        """Restore from checkpoint"""
        # Restore skills (but keep predefined)
        predefined_ids = {s.id for s in self.skills.values() if s.source == "predefined"}

        for skill_id, skill_data in data.get('skills', {}).items():
            if skill_id in predefined_ids:
                # Update predefined skill stats only
                if skill_id in self.skills:
                    self.skills[skill_id].success_count = skill_data.get('success_count', 0)
                    self.skills[skill_id].failure_count = skill_data.get('failure_count', 0)
                    self.skills[skill_id].confidence = skill_data.get('confidence', 1.0)
            else:
                # Restore learned/imported skills
                self.skills[skill_id] = Skill.from_dict(skill_data)

        # Restore tool groups
        self.tool_groups.clear()
        for group_id, group_data in data.get('tool_groups', {}).items():
            self.tool_groups[group_id] = ToolGroup.from_dict(group_data)

        self._skill_embeddings_dirty = True

    # =========================================================================
    # PROMPT BUILDING
    # =========================================================================

    def build_skill_prompt_section(self, matched_skills: List[Skill]) -> str:
        """Build prompt section for matched skills"""
        if not matched_skills:
            return ""

        lines = ["", "RELEVANTE SKILLS FÜR DIESE AUFGABE:"]
        for skill in matched_skills:
            lines.append(f"\n📚 [{skill.name}]")
            lines.append(skill.instruction)

        return '\n'.join(lines)

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> dict:
        """Get statistics"""
        active_skills = sum(1 for s in self.skills.values() if s.is_active())
        learned_skills = sum(1 for s in self.skills.values() if s.source == "learned")
        imported_skills = sum(1 for s in self.skills.values() if s.source == "imported")

        return {
            'total_skills': len(self.skills),
            'active_skills': active_skills,
            'learned_skills': learned_skills,
            'imported_skills': imported_skills,
            'predefined_skills': len(self.skills) - learned_skills - imported_skills,
            'tool_groups': len(self.tool_groups),
            'agent_name': self.agent_name
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"<SkillsManager {self.agent_name} [{stats['active_skills']} active skills, {stats['tool_groups']} groups]>"


# =============================================================================
# AUTO GROUPING (migrated from rule_set.py)
# =============================================================================

def auto_group_tools_by_name_pattern(
    tool_manager: Any,
    skills_manager: SkillsManager,
    min_group_size: int = 2,
    separator: str = "_"
) -> Dict[str, List[str]]:
    """
    Automatically create tool groups based on repeating patterns in tool names.
    Migrated from rule_set.py

    Example:
        Tools: discord_send, discord_edit, discord_delete, github_clone, github_push
        Result: {
            "discord_tools": ["discord_send", "discord_edit", "discord_delete"],
            "github_tools": ["github_clone", "github_push"]
        }
    """
    ignore_prefixes = ["mcp", "a2a", "local"]
    ignore_suffixes = ["tool", "helper", "util", "utils"]

    all_tools = tool_manager.list_names()
    if not all_tools:
        return {}

    # Extract prefixes
    prefix_tools: Dict[str, List[str]] = defaultdict(list)

    for tool_name in all_tools:
        parts = tool_name.lower().split(separator)
        if len(parts) < 2:
            continue

        for prefix_len in range(1, min(3, len(parts))):
            prefix_parts = parts[:prefix_len]

            if prefix_parts[0] in ignore_prefixes:
                if len(prefix_parts) > 1:
                    prefix_parts = prefix_parts[1:]
                else:
                    continue

            if prefix_parts[-1] in ignore_suffixes:
                continue

            prefix = separator.join(prefix_parts)
            if len(prefix) >= 2:
                prefix_tools[prefix].append(tool_name)

    # Filter and resolve overlaps
    valid_groups: Dict[str, List[str]] = {}
    assigned_tools: set = set()

    sorted_prefixes = sorted(
        prefix_tools.items(),
        key=lambda x: (-len(x[0].split(separator)), -len(x[1]))
    )

    for prefix, tools in sorted_prefixes:
        available_tools = [t for t in tools if t not in assigned_tools]
        unique_tools = list(dict.fromkeys(available_tools))

        if len(unique_tools) >= min_group_size:
            group_name = f"{prefix}_tools"
            valid_groups[group_name] = unique_tools
            assigned_tools.update(unique_tools)

    # Register in SkillsManager
    for group_name, tool_names in valid_groups.items():
        display_parts = group_name.replace("_tools", "").split(separator)
        display_name = " ".join(word.capitalize() for word in display_parts) + " Tools"

        trigger_keywords = list(set(
            part for part in group_name.replace("_tools", "").split(separator)
            if part and len(part) > 1
        ))

        skills_manager.register_tool_group(
            name=group_name,
            display_name=display_name,
            tool_names=tool_names,
            trigger_keywords=trigger_keywords,
            description=f"Auto-grouped {len(tool_names)} tools",
            auto_generated=True
        )

    return valid_groups
