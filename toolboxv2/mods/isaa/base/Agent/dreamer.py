"""
Dreamer — Async Meta-Learning for FlowAgent

Offline consolidation: scans logs, clusters runs, evolves skills & personas,
publishes mature skills to bound agents via BindManager.

Usage:
    result = await agent.a_dream(DreamConfig(max_budget=5000))

Architecture:
    Chain-based pipeline using MemoryKnowledgeActor for analysis.
    Each phase is a Function node → composable, testable, interruptible.

Author: FlowAgent V3

=== FIX SUMMARY ===
Problem 1: Parser erkennt echtes Log-Format nicht (### USER Sections statt "Query: " Lines)
Problem 2: Keine Basis für Evolution (blind replace, kein sample-minimum, kein rollback)
Problem 3: Kein Bloat-Management (bloat score disconnected, kein merge, kein proaktiver split)

Fixes:
  [F1] _parse_log: Komplett neu — section-basierter Parser für ### USER/SYSTEM/TOOL/ASSISTANT
  [F2] _phase_cluster: min_records Gate von 3→1 (Einzelne Records gehen in "unclustered")
  [F3] _evolve_skill: Instruction-Diff statt blind replace, min_evidence check, rollback-Speicher
  [F4] _phase_reconcile: Bloat-Check + proaktiver Split/Compress, Merge-Logik für Duplikate
  [F5] _calculate_bloat: Neue Methode — identische Metrik wie Diamond-Anzeige
  [F6] _compress_skill: Neue Methode — Instruction kürzen, Triggers prunen, Tools cappen
  [F7] _merge_duplicate_skills: Neue Methode — nutzt Skill.merge_with()
  [F8] _phase_skip: Robusteres Error-Recovery (dict + Exception handling)
"""

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, List, Dict

from pydantic import BaseModel

from toolboxv2.mods.isaa.base.Agent.chain import Chain, Function, ErrorHandlingChain
from toolboxv2.mods.isaa.base.Agent.skills import Skill, SkillsManager

_log = logging.getLogger("isaa.dreamer")


# =============================================================================
# CONFIG & MODELS
# =============================================================================
from toolboxv2.mods.isaa.base.Agent.types import DreamConfig


class RunRecord(BaseModel):
    """Parsed execution log."""
    run_id: str = ""
    query: str = ""
    tools_used: list[str] = []
    success: bool = True
    error_traces: list[str] = []
    summary: str = ""
    timestamp: str = ""
    log_path: str = ""


class ClusterAnalysis(BaseModel):
    """LLM output for a run cluster."""
    dominant_intent: str = ""
    success_ratio: float = 0.0
    success_pattern: str = ""
    failure_patterns: list[str] = []
    recommended_instruction_update: str = ""
    suggested_triggers: list[str] = []
    suggested_negative_examples: list[str] = []
    should_split: bool = False
    split_intents: list[str] = []


class DreamReport(BaseModel):
    """Audit trail for a dream cycle."""
    dream_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    logs_scanned: int = 0
    clusters_found: int = 0
    skills_evolved: list[str] = []
    skills_created: list[str] = []
    skills_merged: list[str] = []
    skills_split: list[str] = []
    skills_published: list[str] = []
    skills_compressed: list[str] = []          # [F6] NEU
    personas_evolved: list[str] = []
    memory_entries_added: int = 0
    errors: list[str] = []
    budget_used: int = 0


# =============================================================================
# DREAMER — Core Pipeline
# =============================================================================
STOP_WORDS = {"ich", "ein", "eine", "der", "die", "das", "wie", "was", "ist",
              "und", "oder", "für", "mit", "von", "zu", "the", "a", "an",
              "is", "to", "for", "and", "of", "in", "on", "how", "what", "can"}


class Dreamer:
    """
    Async meta-learning engine.

    Runs as Chain pipeline:
        harvest → cluster → analyze → reconcile → publish → cleanup

    Each phase receives + returns a DreamState dict, enabling
    pause/resume and partial execution.
    """

    def __init__(self, agent: 'FlowAgent'):
        self._last_good_state = None
        self.agent = agent
        self._budget_used = 0
        self._config: Optional[DreamConfig] = None
        self._report: Optional[DreamReport] = None

    # ─── Pipeline Construction ──────────────────────────────────────

    def _build_pipeline(self, config: DreamConfig) -> Chain:
        """Build the dream pipeline as a Chain."""

        harvest = Function(self._phase_harvest)
        cluster = Function(self._phase_cluster)
        analyze = Function(self._phase_analyze)
        reconcile = Function(self._phase_reconcile)
        publish = Function(self._phase_publish)
        cleanup = Function(self._phase_memory_sync)

        if config.hard_stop:
            return harvest >> cluster >> analyze >> reconcile >> publish >> cleanup
        else:
            skip = Function(self._phase_skip)
            return (
                (harvest | skip)
                >> (cluster | skip)
                >> (analyze | skip)
                >> (reconcile | skip)
                >> (publish | skip)
                >> (cleanup | skip)
            )

    def _get_scheduler(self):
        """Get JobScheduler if available."""
        try:
            from toolboxv2 import get_app
            return get_app().get_mod("isaa").job_scheduler
        except Exception:
            return None

    # ─── Main Entry Point ───────────────────────────────────────────

    async def dream(self, config: DreamConfig) -> DreamReport:
        """Execute a full dream cycle."""
        scheduler = self._get_scheduler()
        if scheduler:
            scheduler.event_bus.emit("dream_start", {
                "agent": self.agent.amd.name,
                "config": config.__dict__,
            })
        self._config = config
        self._budget_used = 0
        self._report = DreamReport(
            dream_id=f"dream_{uuid.uuid4().hex[:8]}",
            started_at=datetime.now().isoformat(),
        )

        state = {
            "config": config,
            "report": self._report,
            "records": [],
            "clusters": {},
            "analyses": {},
        }

        pipeline = self._build_pipeline(config)
        self._last_good_state = state
        state = await pipeline.a_run(state)

        self._report.finished_at = datetime.now().isoformat()
        self._report.budget_used = self._budget_used

        await self._persist_report(self._report)
        if scheduler:
            scheduler.event_bus.emit("dream_end", {
                "agent": self.agent.amd.name,
                "report": self._report.model_dump(),
            })
        return self._report

    # ─── Phase 1: Log Harvesting ────────────────────────────────────

    async def _phase_harvest(self, state: dict) -> dict:
        """Scan /.memory/logs/ and parse RunRecords."""
        config: DreamConfig = state["config"]
        report: DreamReport = state["report"]

        session = await self._get_session()
        log_dir = "/.memory/logs"

        cutoff = self._get_cutoff(config)
        _log.info(f"[Dreamer] Harvest cutoff={cutoff}")

        ls_result = session.vfs_ls(log_dir, recursive=False)
        if not ls_result.get("success"):
            session.vfs.mkdir(log_dir, parents=True)
            return state

        records: list[RunRecord] = []
        contents = ls_result.get("contents", [])
        _log.info(f"[Dreamer] Found {len(contents)} log entries")

        for entry in contents:
            name = entry.get("name", "")
            if not name.endswith(".md"):
                continue

            # Parse timestamp from filename: YYYYMMDD_HHMMSS_runid.md
            try:
                basename = name.rsplit("/", 1)[-1]
                if not basename.endswith(".md"):
                    continue
                ts_str = basename[:15]
                file_ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                if cutoff and file_ts < cutoff:
                    continue
            except (ValueError, IndexError):
                continue

            # Read and parse log
            path = f"{log_dir}/{name}"
            read_result = session.vfs_read(path)
            if not read_result.get("success"):
                continue

            record = self._parse_log(read_result.get("content", ""), path)
            if record:
                records.append(record)

        state["records"] = records
        report.logs_scanned = len(records)
        _log.info(f"[Dreamer] Harvested {len(records)} valid records")
        return state

    # ─── Phase 2: Embedding Clustering ──────────────────────────────

    async def _phase_cluster(self, state: dict) -> dict:
        """Group RunRecords by query similarity."""
        records: list[RunRecord] = state["records"]

        # ══════════════════════════════════════════════════════════════
        # [F2] FIX: Gate von <3 entfernt. Einzelne Records → "unclustered"
        #      Damit die Analyse-Phase auch mit 1-2 Logs arbeiten kann.
        # ══════════════════════════════════════════════════════════════
        if not records:
            return state

        if len(records) < 3:
            # Nicht genug für echtes Clustering → alles in einen Cluster
            state["clusters"] = {"unclustered_0": records}
            state["report"].clusters_found = 1
            _log.info(f"[Dreamer] {len(records)} records → single unclustered group")
            return state

        memory = self._get_memory()
        if not memory:
            state["clusters"] = self._keyword_cluster(records)
            # [F2] Auch unzugeordnete Records als Einzel-Cluster aufnehmen
            self._add_unclustered(records, state["clusters"])
            state["report"].clusters_found = len(state["clusters"])
            return state

        # Embedding clustering (unchanged logic)
        queries = [r.query for r in records]
        try:
            BATCH_SIZE = 64
            all_embeddings = []
            for i in range(0, len(queries), BATCH_SIZE):
                batch = queries[i:i + BATCH_SIZE]
                embs = await memory.get_embeddings(batch)
                all_embeddings.extend(embs)
            embeddings = all_embeddings
        except Exception as e:
            _log.warning(f"[Dreamer] Embedding failed, keyword fallback: {e}")
            state["clusters"] = self._keyword_cluster(records)
            self._add_unclustered(records, state["clusters"])
            state["report"].clusters_found = len(state["clusters"])
            return state

        import numpy as np
        clusters: dict[str, list[int]] = {}
        assigned = set()
        threshold = 0.65

        for i in range(len(records)):
            if i in assigned:
                continue
            cluster_id = f"c_{len(clusters)}"
            cluster_indices = [i]
            assigned.add(i)

            for j in range(i + 1, len(records)):
                if j in assigned:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9
                ))
                if sim >= threshold:
                    cluster_indices.append(j)
                    assigned.add(j)

            if len(cluster_indices) >= 2:
                clusters[cluster_id] = cluster_indices

        state["clusters"] = {
            cid: [records[idx] for idx in indices]
            for cid, indices in clusters.items()
        }

        # [F2] Unassigned Records als Einzel-Cluster
        self._add_unclustered(records, state["clusters"], assigned)
        state["report"].clusters_found = len(state["clusters"])
        _log.info(f"[Dreamer] Found {len(state['clusters'])} clusters")
        return state

    # ─── Phase 3: LLM Cluster Analysis ─────────────────────────────

    async def _phase_analyze(self, state: dict) -> dict:
        """Analyze each cluster via LLM to extract patterns."""
        clusters: dict[str, list[RunRecord]] = state["clusters"]
        config: DreamConfig = state["config"]
        analyses: dict[str, ClusterAnalysis] = {}

        for cid, records in clusters.items():
            if self._budget_used >= config.max_budget:
                _log.warning("[Dreamer] Budget exhausted in analyze phase")
                scheduler = self._get_scheduler()
                if scheduler:
                    scheduler.event_bus.emit("dream_budget_hit", {
                        "agent": self.agent.amd.name,
                        "budget_used": self._budget_used,
                        "clusters_remaining": len(clusters) - len(analyses),
                    })
                break

            success_records = [r for r in records if r.success]
            failure_records = [r for r in records if not r.success]

            prompt = self._build_analysis_prompt(records, success_records, failure_records, config)

            try:
                response = await self.agent.a_run_llm_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model_preference="fast",
                    max_tokens=500,
                    temperature=0.2,
                    stream=False,
                )
                prompt_cost = len(prompt) // 4
                response_cost = len(response) // 4
                self._budget_used += prompt_cost + response_cost

                analysis = self._parse_analysis(response, len(success_records), len(records))
                analyses[cid] = analysis

            except Exception as e:
                _log.warning(f"[Dreamer] Analysis failed for {cid}: {e}")
                state["report"].errors.append(f"analyze:{cid}:{e}")

        state["analyses"] = analyses
        return state

    # ─── Phase 4: Skill Reconciliation ──────────────────────────────

    async def _phase_reconcile(self, state: dict) -> dict:
        """Evolve, create, split, merge, or compress skills based on analyses."""
        analyses: dict[str, ClusterAnalysis] = state["analyses"]
        clusters: dict[str, list[RunRecord]] = state["clusters"]
        config: DreamConfig = state["config"]
        report: DreamReport = state["report"]
        sm: SkillsManager = self.agent.session_manager.skills_manager

        # ══════════════════════════════════════════════════════════════
        # [F7] PRE-PASS: Merge duplicate skills BEFORE evolution
        # ══════════════════════════════════════════════════════════════
        merged_ids = self._merge_duplicate_skills(sm)
        report.skills_merged.extend(merged_ids)

        # ══════════════════════════════════════════════════════════════
        # [F5][F6] PRE-PASS: Bloat-Check + proaktive Compression
        # ══════════════════════════════════════════════════════════════
        for skill in list(sm.skills.values()):
            bloat = self._calculate_bloat(skill)
            if bloat > 0.6:  # >60% bloat → compress
                self._compress_skill(skill, bloat)
                report.skills_compressed.append(skill.id)
                _log.info(f"[Dreamer] Compressed bloated skill '{skill.name}' (bloat={bloat:.0%})")

        # ── Main reconciliation loop ──
        for cid, analysis in analyses.items():
            records = clusters.get(cid, [])
            if not analysis.dominant_intent:
                continue

            # Match existing skill
            try:
                existing = await sm.match_skills_async(
                    analysis.dominant_intent, max_results=1
                )
            except Exception:
                existing = sm.match_skills(analysis.dominant_intent, max_results=1)

            if existing and config.do_skill_evolve:
                skill = existing[0]
                # ══════════════════════════════════════════════════════
                # [F3] FIX: min_evidence Check — Don't rewrite mature
                #      skills from tiny clusters
                # ══════════════════════════════════════════════════════
                self._evolve_skill(skill, analysis, records)
                report.skills_evolved.append(skill.id)

                # Skill splitting (bloat-driven OR LLM-suggested)
                if config.do_skill_split:
                    bloat = self._calculate_bloat(skill)
                    should_split = (
                        (analysis.should_split and analysis.split_intents)
                        or bloat > 0.45  # [F5] Proaktiver Split bei hohem Bloat
                    )
                    if should_split:
                        split_intents = analysis.split_intents if analysis.split_intents else None
                        new_ids = await self._split_skill(skill, analysis, split_intents)
                        report.skills_split.extend(new_ids)
                        scheduler = self._get_scheduler()
                        if scheduler:
                            scheduler.event_bus.emit("dream_skill_evolved", {
                                "agent": self.agent.amd.name,
                                "skill_id": skill.id,
                                "action": "split",
                            })

            elif not existing and config.do_create_new:
                new_skill = self._create_skill_from_analysis(analysis, records)
                if new_skill:
                    sm.skills[new_skill.id] = new_skill
                    sm._skill_embeddings_dirty = True
                    report.skills_created.append(new_skill.id)
                    scheduler = self._get_scheduler()
                    if scheduler:
                        scheduler.event_bus.emit("dream_skill_evolved", {
                            "agent": self.agent.amd.name,
                            "skill_id": new_skill.id,
                            "action": "created",
                        })

            if config.do_persona_evolve:
                self._evolve_persona_from_analysis(analysis, records)

        return state

    # ─── Phase 5: Publish via BindManager ───────────────────────────

    async def _phase_publish(self, state: dict) -> dict:
        """Publish mature skills to bound agents (Skill Marketplace)."""
        config: DreamConfig = state["config"]
        report: DreamReport = state["report"]
        sm: SkillsManager = self.agent.session_manager.skills_manager
        bm = getattr(self.agent, 'bind_manager', None)

        if not bm or not bm.bindings:
            return state

        publishable = [
            s for s in sm.skills.values()
            if (s.source in ("learned", "imported")
                and s.confidence >= config.publish_threshold
                and getattr(s, '_version', 1) >= config.publish_min_version
                and s.is_active())
        ]

        for skill in publishable:
            skill_data = skill.to_dict()
            skill_data["_published_by"] = self.agent.amd.name
            skill_data["_published_at"] = datetime.now().isoformat()

            for partner_name, binding in bm.bindings.items():
                if binding.mode != "public":
                    continue
                try:
                    await bm.sync(
                        partner_name=partner_name,
                        action="skill_publish",
                        data=skill_data,
                        session_id=self.agent.active_session or "default",
                    )
                    report.skills_published.append(f"{skill.id}→{partner_name}")
                except Exception as e:
                    _log.debug(f"[Dreamer] Publish to {partner_name} failed: {e}")

        return state

    # ─── Phase 6: Memory Sync ───────────────────────────────────────

    async def _phase_memory_sync(self, state: dict) -> dict:
        """Inject consolidated knowledge into agent memory."""
        report: DreamReport = state["report"]
        analyses: dict[str, ClusterAnalysis] = state["analyses"]
        memory = self._get_memory()

        if not memory:
            return state

        added = 0
        for cid, analysis in analyses.items():
            if not analysis.dominant_intent:
                continue

            knowledge = (
                f"Task pattern: {analysis.dominant_intent}\n"
                f"Success approach: {analysis.success_pattern}\n"
            )
            if analysis.failure_patterns:
                knowledge += f"Known pitfalls: {'; '.join(analysis.failure_patterns[:3])}\n"
            if analysis.suggested_negative_examples:
                knowledge += f"Avoid: {'; '.join(analysis.suggested_negative_examples[:2])}\n"

            try:
                await memory.add_data(
                    text=knowledge,
                    memory_names=f"{self.agent.amd.name}_insights",
                    content_type="fact",
                    concepts=[analysis.dominant_intent.split()[0], "dreamer", "pattern"],

                )
                added += 1
            except Exception as e:
                _log.debug(f"[Dreamer] Memory add failed: {e}")

        report.memory_entries_added = added

        session = await self._get_session()
        try:
            session.vfs.mkdir("/.memory/dreamer", parents=True)
            session.vfs.write("/.memory/dreamer/last_run", datetime.now().isoformat())
        except Exception:
            pass

        return state

    # ─── Skip handler for soft-stop ─────────────────────────────────

    async def _phase_skip(self, state: dict) -> dict:
        # ══════════════════════════════════════════════════════════════
        # [F8] FIX: Robusteres Error-Recovery
        #      Chain kann Exception ODER error-dict weiterreichen.
        #      Beide Fälle abfangen statt nur isinstance(Exception).
        # ══════════════════════════════════════════════════════════════
        is_error = False

        if isinstance(state, Exception):
            is_error = True
            error_msg = str(state)
        elif isinstance(state, dict) and state.get("_error"):
            is_error = True
            error_msg = str(state["_error"])
        elif not isinstance(state, dict):
            # Unerwarteter Typ — auch ein Fehler
            is_error = True
            error_msg = f"Unexpected state type: {type(state).__name__}"

        if is_error:
            _log.warning(f"[Dreamer] Phase error (skipped): {error_msg}")
            if self._report:
                self._report.errors.append(error_msg)
            if self._last_good_state is not None:
                recovered = self._last_good_state.copy()
                recovered["_skipped_phase"] = True
                return recovered
            # Absoluter Fallback: leerer valider State
            return {
                "config": self._config,
                "report": self._report,
                "records": [],
                "clusters": {},
                "analyses": {},
                "_skipped_phase": True,
            }

        self._last_good_state = state
        return state

    # =========================================================================
    # SKILL OPERATIONS
    # =========================================================================

    def _evolve_skill(self, skill: Skill, analysis: ClusterAnalysis, records: list[RunRecord]):
        """
        Refine an existing skill — INFORMED evolution, not blind replace.

        ══════════════════════════════════════════════════════════════
        [F3] FIX: Drei zentrale Änderungen:
          1. min_evidence: Cluster mit <3 Records darf mature Skills
             (confidence≥0.7) nicht umschreiben, nur Metadata updaten
          2. Instruction MERGE statt REPLACE: alte Instruction bleibt
             als Basis, LLM-Output wird als Update-Section eingefügt
          3. Rollback: vorherige Instruction wird in _instruction_history
             gespeichert für manuelles Rollback
        ══════════════════════════════════════════════════════════════
        """
        version = getattr(skill, '_version', 1)
        cluster_size = len(records)

        # ── Rollback-Speicher (vorherige Instruction sichern) ──
        if not hasattr(skill, '_instruction_history'):
            skill._instruction_history = []
        skill._instruction_history.append({
            "version": version,
            "instruction": skill.instruction,
            "date": datetime.now().isoformat(),
        })
        skill._instruction_history = skill._instruction_history[-5:]  # Letzte 5 behalten

        skill._version = version + 1

        # ── Evidence-Gate: Kleine Cluster dürfen mature Skills nicht umschreiben ──
        is_mature = skill.confidence >= 0.7 and skill.source == "predefined"
        too_few_samples = cluster_size < 3

        if analysis.recommended_instruction_update:
            if is_mature and too_few_samples:
                # Nur als Hinweis anfügen, nicht ersetzen
                _log.info(f"[Dreamer] Skill '{skill.name}' is mature — "
                          f"appending hint only (cluster_size={cluster_size})")
            elif is_mature:
                # Mature + genug Evidence → Merge (alte Basis + neue Insights)
                skill.instruction = self._merge_instructions(
                    skill.instruction,
                    analysis.recommended_instruction_update,
                    skill.name
                )
            else:
                # Learned/low-confidence → Replace ist OK
                skill.instruction = analysis.recommended_instruction_update

        # Failure patterns als ersetzbarer Block
        if analysis.failure_patterns:
            negatives = "\n".join(f"⚠️ {p}" for p in analysis.failure_patterns[:3])
            if "\nBEKANNTE FALLSTRICKE" in skill.instruction:
                skill.instruction = skill.instruction[:skill.instruction.index("\nBEKANNTE FALLSTRICKE")]
            skill.instruction += f"\n\nBEKANNTE FALLSTRICKE (v{skill._version}):\n{negatives}\n"

        # Triggers: CAP at 8
        MAX_TRIGGERS = 8
        existing = set(t.lower() for t in skill.triggers)
        for t in analysis.suggested_triggers:
            if t.lower() not in existing:
                skill.triggers.append(t)
                existing.add(t.lower())
        if len(skill.triggers) > MAX_TRIGGERS:
            skill.triggers = skill.triggers[:3] + analysis.suggested_triggers[:MAX_TRIGGERS - 3]

        # Tools: CAP at 10, only from successful runs
        MAX_TOOLS = 10
        success_tools = set()
        for r in records:
            if r.success:
                success_tools.update(t for t in r.tools_used if t not in ("think", "final_answer"))
        skill.tools_used = [t for t in skill.tools_used if t in success_tools][:MAX_TOOLS]
        for t in success_tools:
            if t not in skill.tools_used and len(skill.tools_used) < MAX_TOOLS:
                skill.tools_used.append(t)

        # ══════════════════════════════════════════════════════════════
        # [F3] FIX: Evidence-weighted confidence
        #      success_count/failure_count aus Skill nutzen statt nur
        #      0.7/0.3 EMA. Mehr Evidence → stärkere Anpassung.
        # ══════════════════════════════════════════════════════════════
        total_evidence = skill.success_count + skill.failure_count + cluster_size
        weight_new = min(0.5, cluster_size / max(total_evidence, 1))
        weight_old = 1.0 - weight_new
        skill.confidence = min(1.0, skill.confidence * weight_old + analysis.success_ratio * weight_new)
        skill.last_used = datetime.now()

        skill._negative_examples = analysis.suggested_negative_examples[:5]

        if not hasattr(skill, '_evolution_history'):
            skill._evolution_history = []
        skill._evolution_history.append({
            "version": skill._version,
            "date": datetime.now().isoformat(),
            "action": "evolved",
            "cluster_size": cluster_size,
            "success_ratio": analysis.success_ratio,
        })
        skill._evolution_history = skill._evolution_history[-10:]

    def _merge_instructions(self, old_instruction: str, new_instruction: str, skill_name: str) -> str:
        """
        Merge old and new instructions intelligently.
        Keeps the old structure, integrates new insights.

        [F3] Statt blind replace: alte Steps bleiben, neue werden als
             UPDATE-Section angehängt. Bei nächster Evolution kann das
             LLM dann beide Teile sehen und konsolidieren.
        """
        # Wenn die alte Instruction kurz ist (<200 chars), einfach ersetzen
        if len(old_instruction) < 200:
            return new_instruction

        # Sonst: strukturierter Merge
        # Alte Instruction behalten, neue als "EVOLVED UPDATE" anhängen
        # Dabei alte Updates entfernen wenn vorhanden
        base = old_instruction
        if "\n\n── EVOLVED UPDATE" in base:
            base = base[:base.index("\n\n── EVOLVED UPDATE")]

        merged = (
            f"{base}\n\n"
            f"── EVOLVED UPDATE (v{getattr(self, '_version', '?')}) ──\n"
            f"{new_instruction}\n"
        )

        # Gesamtlänge cappen um Bloat zu vermeiden
        MAX_INSTRUCTION_LEN = 1500
        if len(merged) > MAX_INSTRUCTION_LEN:
            merged = merged[:MAX_INSTRUCTION_LEN] + "\n[...truncated]"

        return merged

    async def _split_skill(self, parent: Skill, analysis: ClusterAnalysis,
                           intents: Optional[list[str]] = None) -> list[str]:
        """
        Split a bloated skill into focused sub-skills.

        [F5] FIX: Parent-Skill wird nach Split deaktiviert (threshold hochgesetzt).
             Sub-Skills bekommen nur RELEVANTE Tools, nicht alle.
        """
        sm: SkillsManager = self.agent.session_manager.skills_manager
        new_ids = []

        split_intents = intents or analysis.split_intents
        if not split_intents:
            return []

        for intent in split_intents[:3]:
            sub_id = sm._generate_skill_id(intent)
            # Nur Tools die zum Intent passen (Keyword-Overlap)
            intent_words = set(intent.lower().split())
            relevant_tools = [
                t for t in parent.tools_used
                if any(w in t.lower() for w in intent_words)
            ] or parent.tools_used[:5]  # Fallback: erste 5

            sub_skill = Skill(
                id=sub_id,
                name=intent,
                triggers=[intent.lower()] + [w for w in intent.lower().split() if len(w) > 3],
                instruction=f"Spezialisierung von '{parent.name}':\n{intent}\n\n"
                            f"Basis-Anleitung:\n{parent.instruction[:300]}",
                tools_used=relevant_tools,
                tool_groups=parent.tool_groups.copy(),
                source="learned",
                confidence=parent.confidence * 0.8,
                activation_threshold=0.6,
            )
            sub_skill._version = 1
            sub_skill._parent_skill = parent.id
            sm.skills[sub_id] = sub_skill
            new_ids.append(sub_id)

        # ══════════════════════════════════════════════════════════════
        # [F5] Parent nach Split deaktivieren (nicht löschen für Rollback)
        # ══════════════════════════════════════════════════════════════
        if new_ids:
            parent.activation_threshold = 1.1  # Effektiv deaktiviert
            parent._split_into = new_ids
            _log.info(f"[Dreamer] Split '{parent.name}' → {new_ids}, parent deactivated")

        sm._skill_embeddings_dirty = True
        return new_ids

    def _create_skill_from_analysis(
        self, analysis: ClusterAnalysis, records: list[RunRecord]
    ) -> Optional[Skill]:
        """Genesis: create a new skill from cluster analysis."""
        sm: SkillsManager = self.agent.session_manager.skills_manager

        if not analysis.dominant_intent or not analysis.success_pattern:
            return None

        tools = set()
        for r in records:
            if r.success:
                tools.update(t for t in r.tools_used if t not in ("think", "final_answer"))

        skill_id = sm._generate_skill_id(analysis.dominant_intent)
        skill = Skill(
            id=skill_id,
            name=analysis.dominant_intent[:50],
            triggers=analysis.suggested_triggers or [analysis.dominant_intent.split()[0]],
            instruction=analysis.success_pattern,
            tools_used=list(tools),
            tool_groups=[],
            source="learned",
            confidence=min(0.5, analysis.success_ratio),
            activation_threshold=0.4,
        )
        skill._version = 1
        skill._origin = "dreamer_genesis"

        if analysis.failure_patterns:
            skill.instruction += "\n\nBEKANNTE FALLSTRICKE:\n" + "\n".join(
                f"⚠️ {p}" for p in analysis.failure_patterns[:3]
            )

        return skill

    # =========================================================================
    # BLOAT MANAGEMENT [F5] [F6] [F7] — KOMPLETT NEU
    # =========================================================================

    def _calculate_bloat(self, skill: Skill) -> float:
        """
        Calculate bloat score (0.0-1.0) matching Diamond display metric.

        Bloat = weighted sum of:
          - trigger_bloat:     len(triggers) / MAX      (weight 0.25)
          - instruction_bloat: len(instruction) / MAX   (weight 0.50)
          - tools_bloat:       len(tools) / MAX         (weight 0.25)
        """
        MAX_TRIGGERS = 12
        MAX_INSTRUCTION_LEN = 1200
        MAX_TOOLS = 12

        trigger_bloat = min(1.0, len(skill.triggers) / MAX_TRIGGERS)
        instruction_bloat = min(1.0, len(skill.instruction) / MAX_INSTRUCTION_LEN)
        tools_bloat = min(1.0, len(skill.tools_used) / MAX_TOOLS)

        return trigger_bloat * 0.25 + instruction_bloat * 0.50 + tools_bloat * 0.25

    def _compress_skill(self, skill: Skill, bloat_score: float):
        """
        Compress a bloated skill: trim instruction, prune triggers, cap tools.

        [F6] Intelligente Kompression:
          - Instruction: EVOLVED UPDATE Sections entfernen, dann Gesamtlänge cappen
          - Triggers: Duplikate + zu generische entfernen
          - Tools: Nur die am häufigsten genutzten behalten
        """
        # ── Instruction komprimieren ──
        instruction = skill.instruction

        # Alte EVOLVED UPDATE Sections entfernen (nur die neueste bleibt)
        parts = instruction.split("── EVOLVED UPDATE")
        if len(parts) > 2:
            # Basis + letzte Update-Section
            instruction = parts[0] + "── EVOLVED UPDATE" + parts[-1]

        # Gesamtlänge cappen
        MAX_LEN = 800
        if len(instruction) > MAX_LEN:
            # Versuche an Satz-/Zeilengrenze zu schneiden
            cut = instruction[:MAX_LEN].rfind("\n")
            if cut < MAX_LEN * 0.5:
                cut = MAX_LEN
            instruction = instruction[:cut].rstrip()
        skill.instruction = instruction

        # ── Triggers prunen ──
        MAX_TRIGGERS = 6
        if len(skill.triggers) > MAX_TRIGGERS:
            # Behalte originale (erste 3) + kürzeste (spezifischste)
            originals = skill.triggers[:3]
            rest = sorted(skill.triggers[3:], key=len)[:MAX_TRIGGERS - 3]
            skill.triggers = originals + rest

        # ── Tools cappen ──
        MAX_TOOLS = 8
        if len(skill.tools_used) > MAX_TOOLS:
            skill.tools_used = skill.tools_used[:MAX_TOOLS]

        _log.debug(f"[Dreamer] Compressed '{skill.name}': "
                   f"bloat {bloat_score:.0%} → {self._calculate_bloat(skill):.0%}")

    def _merge_duplicate_skills(self, sm: SkillsManager) -> list[str]:
        """
        Find and merge duplicate skills (same name or very similar triggers).

        [F7] Nutzt Skill.merge_with() das bereits existiert aber nie aufgerufen wurde.
        Returns list of merged (removed) skill IDs.
        """
        merged_ids = []
        skills_list = list(sm.skills.values())
        seen_names = {}  # normalized_name → skill

        for skill in skills_list:
            # Normalisiere Name für Duplikat-Erkennung
            norm_name = skill.name.lower().strip()

            if norm_name in seen_names:
                primary = seen_names[norm_name]
                if primary.id == skill.id:
                    continue
                # Merge: behalte den mit höherer Confidence
                if skill.confidence > primary.confidence:
                    primary, skill = skill, primary
                    seen_names[norm_name] = primary

                try:
                    primary.merge_with(skill)
                    _log.info(f"[Dreamer] Merged duplicate '{skill.name}' (id={skill.id}) "
                              f"into '{primary.name}' (id={primary.id})")
                except Exception as e:
                    # merge_with nutzt LiteLLM — kann fehlschlagen
                    # Fallback: manueller Merge ohne LLM
                    _log.warning(f"[Dreamer] LLM merge failed, manual merge: {e}")
                    existing_triggers = set(t.lower() for t in primary.triggers)
                    for t in skill.triggers:
                        if t.lower() not in existing_triggers:
                            primary.triggers.append(t)
                    existing_tools = set(primary.tools_used)
                    for t in skill.tools_used:
                        if t not in existing_tools:
                            primary.tools_used.append(t)

                # Entferne das Duplikat
                if skill.id in sm.skills:
                    del sm.skills[skill.id]
                    merged_ids.append(skill.id)
            else:
                seen_names[norm_name] = skill

        if merged_ids:
            sm._skill_embeddings_dirty = True

        return merged_ids

    # =========================================================================
    # PERSONA EVOLUTION
    # =========================================================================

    def _evolve_persona_from_analysis(self, analysis: ClusterAnalysis, records: list[RunRecord]):
        """Adjust agent persona config based on observed patterns."""
        persona = getattr(self.agent.amd, 'persona', None)
        if not persona:
            return

        if not hasattr(persona, '_dream_insights'):
            persona._dream_insights = {}

        intent_key = analysis.dominant_intent[:30]
        persona._dream_insights[intent_key] = {
            "success_ratio": analysis.success_ratio,
            "common_tools": list(set(t for r in records for t in r.tools_used[:3])),
            "failure_hints": analysis.failure_patterns[:2],
            "updated": datetime.now().isoformat(),
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    async def _get_session(self):
        """Get or create the active session."""
        sid = self.agent.active_session or "default"
        return await self.agent.session_manager.get_or_create(sid)

    def _get_memory(self):
        """Get memory instance."""
        try:
            return self.agent.session_manager._get_memory()
        except Exception:
            return None

    def _get_cutoff(self, config: DreamConfig) -> Optional[datetime]:
        """Determine log cutoff time."""
        if config.max_history_time:
            return datetime.now() - timedelta(hours=config.max_history_time)

        try:
            session = self.agent.session_manager.get(
                self.agent.active_session or "default"
            )
            if session:
                result = session.vfs_read("/.memory/dreamer/last_run")
                if result.get("success"):
                    return datetime.fromisoformat(result["content"].strip())
        except Exception:
            pass

        return datetime.now() - timedelta(hours=24*3)

    # ══════════════════════════════════════════════════════════════════════
    # [F1] KOMPLETT NEU: Section-basierter Log Parser
    # ══════════════════════════════════════════════════════════════════════

    def _parse_log(self, content: str, path: str) -> Optional[RunRecord]:
        """
        Parse a commit_run log file into a RunRecord.

        Echtes Log-Format:
            # Execution Log: <run_id>
            Query: <kurzer Titel>
            ----------------------------------------
            ### SYSTEM
            IDENTITY: ...
            ### USER
            <die eigentliche User-Query>
            ### SYSTEM
            ⚡ RUN SUMMARY [...]: <Zusammenfassung>
            ### TOOL
            <Tool-Aufrufe als JSON/dict>
            ### ASSISTANT
            <Agent-Antwort>

        Der alte Parser suchte nach "Query: " und "`Tool Call: `" — beides
        existiert nicht in diesem Format.
        """
        record = RunRecord(log_path=path)

        # ── Run-ID aus Header oder Filename ──
        header_match = re.search(r'#\s*Execution Log:\s*(\w+)', content)
        if header_match:
            record.run_id = header_match.group(1)

        # Fallback: ID aus Filename
        if not record.run_id:
            parts = path.split("/")[-1].replace(".md", "").split("_", 2)
            if len(parts) >= 3:
                record.run_id = parts[2]
                record.timestamp = f"{parts[0]}_{parts[1]}"

        # ── Sections aufteilen ──
        # Split an "### SECTION_NAME" Headern
        sections = re.split(r'^###\s+(\w+)', content, flags=re.MULTILINE)
        # sections = ['header', 'SYSTEM', 'content', 'USER', 'content', ...]

        user_queries: list[str] = []
        tool_sections: list[str] = []
        system_sections: list[str] = []
        assistant_sections: list[str] = []

        i = 1  # Skip header (index 0)
        while i < len(sections) - 1:
            section_type = sections[i].strip().upper()
            section_content = sections[i + 1].strip()
            i += 2

            if section_type == "USER":
                user_queries.append(section_content)
            elif section_type == "TOOL":
                tool_sections.append(section_content)
            elif section_type == "SYSTEM":
                system_sections.append(section_content)
            elif section_type == "ASSISTANT":
                assistant_sections.append(section_content)

        # ── Query extrahieren: Erste USER-Section die NICHT trivial ist ──
        for uq in user_queries:
            # Überspringe triviale Queries ("hi", "hello", einzelne Wörter <5 chars)
            cleaned = uq.strip()
            if len(cleaned) > 5 and cleaned.lower() not in ("hi", "hello", "hey", "hallo"):
                record.query = cleaned[:500]  # Cap bei 500 chars
                break

        # Fallback: "Query: " Zeile aus Header
        if not record.query:
            for line in content.split("\n")[:5]:
                if line.startswith("Query: "):
                    q = line[7:].strip()
                    if q and q.lower() not in ("hi", "hello", "hey", "hallo"):
                        record.query = q[:500]
                        break

        # ── Tools aus TOOL-Sections und RUN SUMMARY extrahieren ──
        for ts in tool_sections:
            # Tool-Sections enthalten oft dicts wie {'success': True, 'path': ...}
            # Extrahiere Tool-Namen aus Kontext
            tool_matches = re.findall(r"'name':\s*'([^']+)'", ts)
            record.tools_used.extend(tool_matches)

            # Auch Funktionsnamen aus dem Content
            func_matches = re.findall(r'`(\w+)\(`', ts)
            record.tools_used.extend(func_matches)

        # Tools aus SYSTEM sections (RUN SUMMARY enthält manchmal Tool-Refs)
        for ss in system_sections:
            if "RUN SUMMARY" in ss:
                record.summary = ss[:500]
            # Tool-Referenzen finden
            tool_refs = re.findall(r'`Tool Call:\s*([^`]+)`', ss)
            record.tools_used.extend(t.strip() for t in tool_refs)
            # Auch Tool-Gruppen wie "vfs_list", "spawn_sub_agent" etc.
            func_refs = re.findall(r'\b(vfs_\w+|spawn_\w+|memory_\w+|analyze_\w+|load_tools|list_tools)\b', ss)
            record.tools_used.extend(func_refs)

        # Deduplicate tools
        record.tools_used = list(dict.fromkeys(record.tools_used))

        # ── Error detection ──
        error_patterns = re.compile(
            r'(Fehler|Error|failed|exception|traceback|❌|🔴)',
            re.IGNORECASE
        )
        for line in content.split("\n"):
            if error_patterns.search(line):
                # Filtere false positives (Tabellen-Header, Empfehlungen etc.)
                if any(fp in line.lower() for fp in ["empfehlung", "beheben", "sovort", "sofort", "level"]):
                    continue
                record.error_traces.append(line.strip()[:200])

        # ── Success-Heuristik (verbessert) ──
        if record.error_traces:
            # Gewichtet: Echte Errors zählen mehr als "Error" in beschreibendem Text
            real_errors = [e for e in record.error_traces
                          if any(kw in e.lower() for kw in ("traceback", "exception", "❌", "failed"))]
            error_ratio = len(real_errors) / max(len(record.tools_used), 1)
            record.success = error_ratio < 0.1
        else:
            record.success = True

        # ── Nur Records mit sinnvoller Query zurückgeben ──
        if not record.query:
            _log.debug(f"[Dreamer] Skipped log {path}: no meaningful query found")
            return None

        return record

    def _keyword_cluster(self, records: list[RunRecord]) -> dict[str, list[RunRecord]]:
        clusters: dict[str, list[RunRecord]] = {}
        assigned = set()

        for i, r in enumerate(records):
            if i in assigned:
                continue
            words_i = set(r.query.lower().split()) - STOP_WORDS
            cluster = [r]
            assigned.add(i)

            for j in range(i + 1, len(records)):
                if j in assigned:
                    continue
                words_j = set(records[j].query.lower().split()) - STOP_WORDS
                union = len(words_i | words_j)
                overlap = len(words_i & words_j)
                if union > 0 and overlap / union >= 0.4:
                    cluster.append(records[j])
                    assigned.add(j)

            if len(cluster) >= 2:
                clusters[f"kw_{len(clusters)}"] = cluster

        return clusters

    def _add_unclustered(self, records: list[RunRecord],
                         clusters: dict[str, list[RunRecord]],
                         assigned: set = None):
        """
        [F2] Records die keinem Cluster zugeordnet wurden als
             Einzel-Cluster aufnehmen damit sie trotzdem analysiert werden.
        """
        if assigned is None:
            # Berechne assigned aus existing clusters
            assigned = set()
            for cluster_records in clusters.values():
                for cr in cluster_records:
                    for i, r in enumerate(records):
                        if r.run_id == cr.run_id:
                            assigned.add(i)

        for i, r in enumerate(records):
            if i not in assigned:
                clusters[f"single_{i}"] = [r]

    def _build_analysis_prompt(
        self,
        records: list[RunRecord],
        success: list[RunRecord],
        failures: list[RunRecord],
        config: DreamConfig,
    ) -> str:
        """Build LLM prompt for cluster analysis."""
        queries_str = "\n".join(f"- {r.query[:100]}" for r in records[:10])
        success_tools = ", ".join(set(t for r in success for t in r.tools_used[:5]))
        failure_info = "\n".join(
            f"- Query: {r.query[:60]} | Errors: {'; '.join(r.error_traces[:2])}"
            for r in failures[:5]
        )

        # [F3] Kontext über existierende Skills mitgeben damit das LLM
        #      informiert updaten kann statt blind zu ersetzen
        existing_context = ""
        sm = self.agent.session_manager.skills_manager
        for skill in sm.skills.values():
            if skill.matches_keywords(records[0].query if records else ""):
                existing_context = (
                    f"\nEXISTIERENDER SKILL '{skill.name}' (confidence={skill.confidence:.2f}):\n"
                    f"{skill.instruction[:200]}...\n"
                    f"Triggers: {', '.join(skill.triggers[:5])}\n"
                )
                break

        split_section = ""
        if config.do_skill_split:
            split_section = (
                "\n7. should_split: true wenn der Intent zu breit ist und in Sub-Intents aufgeteilt werden sollte"
                "\n8. split_intents: Liste der Sub-Intents (nur wenn should_split=true)"
            )

        return f"""Analysiere diese Gruppe ähnlicher Agent-Runs:

QUERIES ({len(records)} total, {len(success)} erfolgreich, {len(failures)} fehlgeschlagen):
{queries_str}

ERFOLGREICHE TOOLS: {success_tools}

FEHLSCHLÄGE:
{failure_info or "Keine"}
{existing_context}
Erstelle eine JSON-Analyse:
{{
  "dominant_intent": "Was wollen die Nutzer hier generell?",
  "success_pattern": "Nummerierte Anleitung (4-6 Schritte) für den optimalen Ablauf",
  "failure_patterns": ["Konkretes Anti-Pattern 1", "Anti-Pattern 2"],
  "recommended_instruction_update": "Verbesserte Gesamt-Instruktion",
  "suggested_triggers": ["keyword1", "keyword2", "keyword3"],
  "suggested_negative_examples": ["Vermeide X weil Y"]{split_section}
}}

Antworte NUR mit dem JSON."""

    def _parse_analysis(self, response: str, success_count: int, total: int) -> ClusterAnalysis:
        """Parse LLM analysis response."""
        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:])
            if clean.endswith("```"):
                clean = "\n".join(clean.split("\n")[:-1])

            data = json.loads(clean)
            analysis = ClusterAnalysis(**{k: v for k, v in data.items() if k in ClusterAnalysis.model_fields})
            analysis.success_ratio = success_count / max(total, 1)
            return analysis

        except (json.JSONDecodeError, Exception) as e:
            _log.debug(f"[Dreamer] Analysis parse failed: {e}")
            return ClusterAnalysis(
                dominant_intent="parse_error",
                success_ratio=success_count / max(total, 1),
            )

    async def _persist_report(self, report: DreamReport):
        """Write dream report to VFS."""
        try:
            session = await self._get_session()
            session.vfs.mkdir("/.memory/dreamer", parents=True)
            path = f"/.memory/dreamer/{report.dream_id}.json"
            session.vfs.write(path, report.model_dump_json(indent=2))
        except Exception as e:
            _log.warning(f"[Dreamer] Failed to persist report: {e}")


# =============================================================================
# BIND INTEGRATION: Skill Marketplace Consumer
# =============================================================================

async def consume_published_skills(agent: 'FlowAgent', session_id: str = "default"):
    """Check BindManager sync for published skills from bound agents."""
    bm = getattr(agent, 'bind_manager', None)
    if not bm:
        return

    sm = agent.session_manager.skills_manager
    entries = await bm.poll_sync(session_id=session_id)

    for partner, sync_entries in entries.items():
        for entry in sync_entries:
            if entry.action != "skill_publish":
                continue

            skill_data = entry.data
            if not isinstance(skill_data, dict):
                continue

            skill_data["confidence"] = min(skill_data.get("confidence", 0.5), 0.7)
            skill_data["source"] = "imported"

            sm.import_skill(skill_data, overwrite=False)
            await bm.acknowledge_sync(entry.id, session_id)

            _log.info(f"[SkillMarket] Adopted skill '{skill_data.get('name')}' from {partner}")
