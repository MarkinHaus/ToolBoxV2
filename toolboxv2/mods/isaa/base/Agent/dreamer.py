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
"""

import asyncio
import json
import logging
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
    personas_evolved: list[str] = []
    memory_entries_added: int = 0
    errors: list[str] = []
    budget_used: int = 0


# =============================================================================
# DREAMER — Core Pipeline
# =============================================================================

class Dreamer:
    """
    Async meta-learning engine.

    Runs as Chain pipeline:
        harvest → cluster → analyze → reconcile → publish → cleanup

    Each phase receives + returns a DreamState dict, enabling
    pause/resume and partial execution.
    """

    def __init__(self, agent: 'FlowAgent'):
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
            # Hard: any error aborts entire pipeline
            return harvest >> cluster >> analyze >> reconcile >> publish >> cleanup
        else:
            # Soft: each phase wrapped in error handler that logs & continues
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
        """Get JobScheduler if available.""" # TODO VLIDATE
        try:
            from toolboxv2 import get_app
            return get_app().get_mod("isaa").job_scheduler
        except Exception:
            return None

    # ─── Main Entry Point ───────────────────────────────────────────

    async def dream(self, config: DreamConfig) -> DreamReport:
        """
        Execute a full dream cycle.

        Args:
            config: DreamConfig with budget, flags, thresholds.

        Returns:
            DreamReport audit trail.
        """
        # ── Emit dream_start ──
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
        state = await pipeline.a_run(state)

        self._report.finished_at = datetime.now().isoformat()
        self._report.budget_used = self._budget_used

        # Persist report to agent memory
        await self._persist_report(self._report)
        # ── Emit dream_end ──
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

        # Determine time window
        cutoff = self._get_cutoff(config)

        # List logs
        ls_result = session.vfs_ls(log_dir, recursive=False)
        if not ls_result.get("success"):
            session.vfs.mkdir(log_dir, parents=True)
            return state

        records: list[RunRecord] = []
        for entry in ls_result.get("entries", []):
            name = entry.get("name", "")
            if not name.endswith(".md"):
                continue

            # Parse timestamp from filename: YYYYMMDD_HHMMSS_runid.md
            try:
                ts_str = name[:15]  # "20250223_031500"
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
        _log.info(f"[Dreamer] Harvested {len(records)} logs")
        return state

    # ─── Phase 2: Embedding Clustering ──────────────────────────────

    async def _phase_cluster(self, state: dict) -> dict:
        """Group RunRecords by query similarity using embeddings."""
        records: list[RunRecord] = state["records"]
        if len(records) < 3:
            return state

        memory = self._get_memory()
        if not memory:
            # Fallback: keyword-based grouping
            state["clusters"] = self._keyword_cluster(records)
            return state

        # Get embeddings for all queries
        queries = [r.query for r in records]
        try:
            embeddings = await memory.get_embeddings(queries)
        except Exception as e:
            _log.warning(f"[Dreamer] Embedding failed, using keyword fallback: {e}")
            state["clusters"] = self._keyword_cluster(records)
            return state

        # Simple greedy clustering via cosine similarity
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

        # Convert to record lists
        state["clusters"] = {
            cid: [records[idx] for idx in indices]
            for cid, indices in clusters.items()
        }

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
                self._budget_used += 500  # estimate

                analysis = self._parse_analysis(response, len(success_records), len(records))
                analyses[cid] = analysis

            except Exception as e:
                _log.warning(f"[Dreamer] Analysis failed for {cid}: {e}")
                state["report"].errors.append(f"analyze:{cid}:{e}")

        state["analyses"] = analyses
        return state

    # ─── Phase 4: Skill Reconciliation ──────────────────────────────

    async def _phase_reconcile(self, state: dict) -> dict:
        """Evolve, create, split, or merge skills based on analyses."""
        analyses: dict[str, ClusterAnalysis] = state["analyses"]
        clusters: dict[str, list[RunRecord]] = state["clusters"]
        config: DreamConfig = state["config"]
        report: DreamReport = state["report"]
        sm: SkillsManager = self.agent.session_manager.skills_manager

        for cid, analysis in analyses.items():
            records = clusters.get(cid, [])
            if not analysis.dominant_intent:
                continue

            # ── Try match existing skill ──
            try:
                existing = await sm.match_skills_async(
                    analysis.dominant_intent, max_results=1
                )
            except Exception:
                existing = sm.match_skills(analysis.dominant_intent, max_results=1)

            if existing and config.do_skill_evolve:
                skill = existing[0]
                self._evolve_skill(skill, analysis, records)
                report.skills_evolved.append(skill.id)

                # ── Skill splitting ──
                if config.do_skill_split and analysis.should_split and analysis.split_intents:
                    new_ids = await self._split_skill(skill, analysis)
                    report.skills_split.extend(new_ids)
                    scheduler = self._get_scheduler()
                    if scheduler:
                        scheduler.event_bus.emit("dream_skill_evolved", {
                            "agent": self.agent.amd.name,
                            "skill_id": skill.id,  # oder new_skill.id
                            "action": "split",  # oder "created" / "split"
                        })

            elif not existing and config.do_create_new:
                # ── Skill genesis from cluster ──
                new_skill = self._create_skill_from_analysis(analysis, records)
                if new_skill:
                    sm.skills[new_skill.id] = new_skill
                    sm._skill_embeddings_dirty = True
                    report.skills_created.append(new_skill.id)
                    scheduler = self._get_scheduler()
                    if scheduler:
                        scheduler.event_bus.emit("dream_skill_evolved", {
                            "agent": self.agent.amd.name,
                            "skill_id": skill.id,  # oder new_skill.id
                            "action": "created",  # oder "created" / "split"
                        })

            # ── Persona evolution ──
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

        # Collect publishable skills
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

            # Publish via BindManager sync (public channel)
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
        """
        Inject consolidated knowledge into agent memory so runtime can access it.
        Clean stale entries, add fresh insights.
        """
        report: DreamReport = state["report"]
        analyses: dict[str, ClusterAnalysis] = state["analyses"]
        memory = self._get_memory()

        if not memory:
            return state

        added = 0
        for cid, analysis in analyses.items():
            if not analysis.dominant_intent:
                continue

            # Build knowledge entry from analysis
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

        # Store last dream timestamp
        session = await self._get_session()
        try:
            session.vfs.mkdir("/.memory/dreamer", parents=True)
            session.vfs.write("/.memory/dreamer/last_run", datetime.now().isoformat())
        except Exception:
            pass

        return state

    # ─── Skip handler for soft-stop ─────────────────────────────────

    async def _phase_skip(self, state: dict) -> dict:
        """Error fallback: log and pass state through."""
        if isinstance(state, Exception):
            _log.warning(f"[Dreamer] Phase error (skipped): {state}")
            if self._report:
                self._report.errors.append(str(state))
            return {}
        return state if isinstance(state, dict) else {}

    # =========================================================================
    # SKILL OPERATIONS
    # =========================================================================

    def _evolve_skill(self, skill: Skill, analysis: ClusterAnalysis, records: list[RunRecord]):
        """Refine an existing skill's instruction with cluster insights."""
        # Bump version
        version = getattr(skill, '_version', 1)
        skill._version = version + 1

        # Enrich instruction with failure patterns
        if analysis.failure_patterns:
            negatives = "\n".join(f"⚠️ {p}" for p in analysis.failure_patterns[:3])
            skill.instruction += f"\n\nBEKANNTE FALLSTRICKE (v{skill._version}):\n{negatives}"

        # Update instruction if LLM provided a better one
        if analysis.recommended_instruction_update:
            skill.instruction = analysis.recommended_instruction_update

        # Merge triggers
        existing = set(t.lower() for t in skill.triggers)
        for t in analysis.suggested_triggers:
            if t.lower() not in existing:
                skill.triggers.append(t)
                existing.add(t.lower())

        # Merge tools from successful records
        existing_tools = set(skill.tools_used)
        for r in records:
            if r.success:
                for t in r.tools_used:
                    if t not in existing_tools and t not in ("think", "final_answer"):
                        skill.tools_used.append(t)
                        existing_tools.add(t)

        # Adjust confidence from cluster success ratio
        skill.confidence = min(1.0, skill.confidence * 0.7 + analysis.success_ratio * 0.3)
        skill.last_used = datetime.now()

        # Store negative examples
        if not hasattr(skill, '_negative_examples'):
            skill._negative_examples = []
        skill._negative_examples.extend(analysis.suggested_negative_examples[:5])

        # Evolution history
        if not hasattr(skill, '_evolution_history'):
            skill._evolution_history = []
        skill._evolution_history.append({
            "version": skill._version,
            "date": datetime.now().isoformat(),
            "action": "evolved",
            "cluster_size": len(records),
            "success_ratio": analysis.success_ratio,
        })

    async def _split_skill(self, parent: Skill, analysis: ClusterAnalysis) -> list[str]:
        """Split a bloated skill into focused sub-skills."""
        sm: SkillsManager = self.agent.session_manager.skills_manager
        new_ids = []

        for intent in analysis.split_intents[:3]:  # max 3 sub-skills
            sub_id = sm._generate_skill_id(intent)
            sub_skill = Skill(
                id=sub_id,
                name=intent,
                triggers=[intent.lower()] + [w for w in intent.lower().split() if len(w) > 3],
                instruction=f"Spezialisierung von '{parent.name}':\n{intent}\n\n"
                            f"Basis-Anleitung:\n{parent.instruction[:300]}",
                tools_used=parent.tools_used.copy(),
                tool_groups=parent.tool_groups.copy(),
                source="learned",
                confidence=parent.confidence * 0.8,
                activation_threshold=0.6,
            )
            sub_skill._version = 1
            sub_skill._parent_skill = parent.id
            sm.skills[sub_id] = sub_skill
            new_ids.append(sub_id)

        sm._skill_embeddings_dirty = True
        return new_ids

    def _create_skill_from_analysis(
        self, analysis: ClusterAnalysis, records: list[RunRecord]
    ) -> Optional[Skill]:
        """Genesis: create a new skill from cluster analysis."""
        sm: SkillsManager = self.agent.session_manager.skills_manager

        if not analysis.dominant_intent or not analysis.success_pattern:
            return None

        # Collect tools from successful runs
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
            activation_threshold=0.4,  # lower threshold for new skills
        )
        skill._version = 1
        skill._origin = "dreamer_genesis"

        if analysis.failure_patterns:
            skill.instruction += "\n\nBEKANNTE FALLSTRICKE:\n" + "\n".join(
                f"⚠️ {p}" for p in analysis.failure_patterns[:3]
            )

        return skill

    # =========================================================================
    # PERSONA EVOLUTION
    # =========================================================================

    def _evolve_persona_from_analysis(self, analysis: ClusterAnalysis, records: list[RunRecord]):
        """
        Adjust agent persona config based on observed patterns.
        Stores persona insights in agent's PersonaConfig metadata.
        """
        persona = getattr(self.agent.amd, 'persona', None)
        if not persona:
            return

        # Track task-type → outcome correlations in persona metadata
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

        # Auto: read last dream timestamp
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

        # Fallback: last 24h
        return datetime.now() - timedelta(hours=24)

    def _parse_log(self, content: str, path: str) -> Optional[RunRecord]:
        """Parse a commit_run log file into a RunRecord."""
        lines = content.split("\n")
        record = RunRecord(log_path=path)

        for line in lines:
            if line.startswith("Query: "):
                record.query = line[7:].strip()
            elif line.startswith("`Tool Call: "):
                tool = line.replace("`Tool Call: ", "").replace("`", "").strip()
                if tool:
                    record.tools_used.append(tool)
            elif "Fehler" in line or "Error" in line or "failed" in line.lower():
                record.error_traces.append(line.strip()[:200])

        # Extract run_id from filename
        parts = path.split("/")[-1].replace(".md", "").split("_", 2)
        if len(parts) >= 3:
            record.run_id = parts[2]
            record.timestamp = f"{parts[0]}_{parts[1]}"

        # Determine success: if we have error traces and no tool calls, likely failure
        record.success = len(record.error_traces) == 0 or len(record.tools_used) > 2

        return record if record.query else None

    def _keyword_cluster(self, records: list[RunRecord]) -> dict[str, list[RunRecord]]:
        """Fallback clustering by keyword overlap."""
        clusters: dict[str, list[RunRecord]] = {}
        assigned = set()

        for i, r in enumerate(records):
            if i in assigned:
                continue
            words_i = set(r.query.lower().split())
            cluster = [r]
            assigned.add(i)

            for j in range(i + 1, len(records)):
                if j in assigned:
                    continue
                words_j = set(records[j].query.lower().split())
                overlap = len(words_i & words_j)
                if overlap >= 2:
                    cluster.append(records[j])
                    assigned.add(j)

            if len(cluster) >= 2:
                clusters[f"kw_{len(clusters)}"] = cluster

        return clusters

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
            # Strip markdown fences
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
    """
    Check BindManager sync for published skills from bound agents.
    Called at runtime (e.g. session init or periodic).
    """
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

            # Import with discount (shared skills start lower)
            skill_data["confidence"] = min(skill_data.get("confidence", 0.5), 0.7)
            skill_data["source"] = "imported"

            sm.import_skill(skill_data, overwrite=False)
            await bm.acknowledge_sync(entry.id, session_id)

            _log.info(f"[SkillMarket] Adopted skill '{skill_data.get('name')}' from {partner}")



